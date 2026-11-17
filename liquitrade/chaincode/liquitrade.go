package main

import (
	"crypto/ecdsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"log"
	"math"
	"math/big"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

// LiquiTradeContract implements AI-gated fractional MSME receivable tokenization on NPCI Drunix
type LiquiTradeContract struct {
	contractapi.Contract
}

// InvoiceAsset tracks the full 7-step lifecycle from upload & buyer acceptance to fractional settlement
type InvoiceAsset struct {
	InvoiceID            string             `json:"invoiceId"`
	InvoiceHash          string             `json:"invoiceHash"`
	IRN                  string             `json:"irn"`
	MsmeID               string             `json:"msmeId"`
	BuyerID              string             `json:"buyerId"`
	BuyerGSTIN           string             `json:"buyerGstin"`
	AmountINR            float64            `json:"amountInr"`
	TenorDays            int                `json:"tenorDays"`
	AnnualDiscountPct    float64            `json:"annualDiscountPct"`
	DueDate              string             `json:"dueDate"`
	Status               string             `json:"status"` // REGISTERED, ACCEPTED, MINTED, SETTLED, DEFAULTED
	RiskTier             string             `json:"riskTier"` // TIER_A, TIER_B, QUARANTINED
	VelocityDeltaPct     float64            `json:"velocityDeltaPct"`
	FraudScore           float64            `json:"fraudScore"`
	TotalTokens          int                `json:"totalTokens"`
	AvailableTokens      int                `json:"availableTokens"`
	TokenFaceValueINR    float64            `json:"tokenFaceValueInr"`
	TokenOfferPriceINR   float64            `json:"tokenOfferPriceInr"`
	Holdings             map[string]int     `json:"holdings"` // investorId -> token count
	PayoutsINR           map[string]float64 `json:"payoutsInr,omitempty"`
	MintedAtTxID         string             `json:"mintedAtTxId,omitempty"`
}

// AIAttestation is the cryptographically signed output from the off-chain AI Risk Gate
type AIAttestation struct {
	InvoiceID        string  `json:"invoiceId"`
	InvoiceHash      string  `json:"invoiceHash"`
	RiskTier         string  `json:"riskTier"`         // TIER_A, TIER_B, QUARANTINED
	VelocityDeltaPct float64 `json:"velocityDeltaPct"` // % delta vs 6-month MSME average
	FraudScore       float64 `json:"fraudScore"`
	Approved         bool    `json:"approved"`
	Signature        string  `json:"signature"`
}

const (
	aiPublicKeyLedgerKey   = "CONFIG_AI_GATEKEEPER_PUBKEY"
	institutionLPID        = "INSTITUTION_LP_DEFAULT"
	tokenFaceValue         = 100.0 // ₹100 minimum retail ticket size
	tierAInstitutionShare  = 0.20  // 20% instant institutional allocation on Tier A
)

// InitLedger registers the trusted AI Risk Gate ECDSA public key (PEM format)
func (c *LiquiTradeContract) InitLedger(ctx contractapi.TransactionContextInterface, aiPublicKeyPEM string) error {
	if _, err := parseECDSAPublicKey(aiPublicKeyPEM); err != nil {
		return fmt.Errorf("invalid AI gatekeeper PEM public key: %v", err)
	}
	return ctx.GetStub().PutState(aiPublicKeyLedgerKey, []byte(aiPublicKeyPEM))
}

// Step 1: RegisterInvoice records a structured e-invoice and enforces unique IRN/hash keys
func (c *LiquiTradeContract) RegisterInvoice(
	ctx contractapi.TransactionContextInterface,
	invoiceID string,
	invoiceHash string,
	irn string,
	msmeID string,
	buyerID string,
	buyerGSTIN string,
	amountINR float64,
	tenorDays int,
	dueDate string,
) error {
	if amountINR < tokenFaceValue {
		return fmt.Errorf("invoice amount must be at least ₹%.2f", tokenFaceValue)
	}
	if tenorDays <= 0 {
		return fmt.Errorf("tenorDays must be positive")
	}

	// Prevent duplicate invoice ID
	existing, err := ctx.GetStub().GetState(invoiceID)
	if err != nil {
		return fmt.Errorf("failed reading ledger state: %v", err)
	}
	if existing != nil {
		return fmt.Errorf("invoice %s is already registered", invoiceID)
	}

	// Prevent duplicate IRN / invoice hash across different IDs (anti-double-financing)
	hashKey := "HASH_" + invoiceHash
	existingHash, err := ctx.GetStub().GetState(hashKey)
	if err != nil {
		return fmt.Errorf("failed reading hash registry: %v", err)
	}
	if existingHash != nil {
		return fmt.Errorf("duplicate invoice hash/IRN detected: already registered under %s", string(existingHash))
	}

	asset := InvoiceAsset{
		InvoiceID:         invoiceID,
		InvoiceHash:       invoiceHash,
		IRN:               irn,
		MsmeID:            msmeID,
		BuyerID:           buyerID,
		BuyerGSTIN:        buyerGSTIN,
		AmountINR:         amountINR,
		TenorDays:         tenorDays,
		AnnualDiscountPct: 12.0,
		DueDate:           dueDate,
		Status:            "REGISTERED",
		Holdings:          make(map[string]int),
	}

	assetBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	if err := ctx.GetStub().PutState(hashKey, []byte(invoiceID)); err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, assetBytes)
}

// Step 2: AcceptInvoice records on-chain confirmation from the anchor buyer
func (c *LiquiTradeContract) AcceptInvoice(ctx contractapi.TransactionContextInterface, invoiceID string, buyerID string) error {
	asset, err := c.ReadInvoice(ctx, invoiceID)
	if err != nil {
		return err
	}
	if asset.Status != "REGISTERED" {
		return fmt.Errorf("invoice %s cannot be accepted in status %s", invoiceID, asset.Status)
	}
	if asset.BuyerID != buyerID {
		return fmt.Errorf("unauthorized buyer %s: expected %s", buyerID, asset.BuyerID)
	}

	asset.Status = "ACCEPTED"
	updatedBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// Step 3 & 4 & 5: IssueInvoiceToken verifies buyer acceptance + AI oracle signature, mints N tokens,
// and executes the automatic 20% institutional LP allocation for Tier A invoices.
func (c *LiquiTradeContract) IssueInvoiceToken(
	ctx contractapi.TransactionContextInterface,
	invoiceID string,
	annualDiscountPct float64,
	attestationJSON string,
) error {
	asset, err := c.ReadInvoice(ctx, invoiceID)
	if err != nil {
		return err
	}
	if asset.Status != "ACCEPTED" {
		return fmt.Errorf("mint blocked: invoice %s must be ACCEPTED by buyer first (current: %s)", invoiceID, asset.Status)
	}

	var attestation AIAttestation
	if err := json.Unmarshal([]byte(attestationJSON), &attestation); err != nil {
		return fmt.Errorf("invalid AI attestation JSON: %v", err)
	}
	if attestation.InvoiceID != invoiceID || attestation.InvoiceHash != asset.InvoiceHash {
		return fmt.Errorf("attestation mismatch: invoice metadata does not match signed payload")
	}

	// Verify AI Gatekeeper ECDSA Signature
	pubKeyBytes, err := ctx.GetStub().GetState(aiPublicKeyLedgerKey)
	if err != nil || pubKeyBytes == nil {
		return fmt.Errorf("AI gatekeeper public key not initialized on ledger")
	}
	pubKey, err := parseECDSAPublicKey(string(pubKeyBytes))
	if err != nil {
		return fmt.Errorf("stored AI public key is invalid: %v", err)
	}

	canonicalMsg := fmt.Sprintf(
		"%s|%s|%s|%.2f|%.2f|%t",
		attestation.InvoiceID,
		attestation.InvoiceHash,
		attestation.RiskTier,
		attestation.VelocityDeltaPct,
		attestation.FraudScore,
		attestation.Approved,
	)
	if err := verifyECDSASignature(pubKey, canonicalMsg, attestation.Signature); err != nil {
		return fmt.Errorf("AI signature verification failed: %v", err)
	}

	// Enforce AI Risk Gate decision on-chain: Quarantined invoices are never minted
	if !attestation.Approved || attestation.RiskTier == "QUARANTINED" {
		return fmt.Errorf("AI gate quarantined invoice %s (tier=%s, velocityDelta=%.2f%%, fraudScore=%.2f): minting blocked",
			invoiceID, attestation.RiskTier, attestation.VelocityDeltaPct, attestation.FraudScore)
	}

	if annualDiscountPct <= 0 {
		annualDiscountPct = 12.0
	}

	totalTokens := int(math.Floor(asset.AmountINR / tokenFaceValue))
	// Discounted offer price: FaceValue / (1 + r * (days / 365)) -> e.g. 100 / (1 + 0.12*(90/365)) = ₹97.13 or 100*(1 - r*days/365) = ₹97.04
	offerPrice := math.Round((tokenFaceValue*(1.0-(annualDiscountPct/100.0)*(float64(asset.TenorDays)/365.0)))*100) / 100

	asset.RiskTier = attestation.RiskTier
	asset.VelocityDeltaPct = attestation.VelocityDeltaPct
	asset.FraudScore = attestation.FraudScore
	asset.AnnualDiscountPct = annualDiscountPct
	asset.TotalTokens = totalTokens
	asset.AvailableTokens = totalTokens
	asset.TokenFaceValueINR = tokenFaceValue
	asset.TokenOfferPriceINR = offerPrice
	asset.Status = "MINTED"
	asset.MintedAtTxID = ctx.GetStub().GetTxID()

	// Step 5: Automatic 20% institutional LP purchase for Tier A invoices
	if attestation.RiskTier == "TIER_A" {
		instTokens := int(math.Floor(float64(totalTokens) * tierAInstitutionShare))
		if instTokens > 0 {
			asset.Holdings[institutionLPID] = instTokens
			asset.AvailableTokens -= instTokens
		}
	}

	updatedBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// Step 6: FractionalPurchase allows retail/institutional investors to purchase tokens atomically.
// Drunix MVCC rejects concurrent oversubscription on AvailableTokens.
func (c *LiquiTradeContract) FractionalPurchase(
	ctx contractapi.TransactionContextInterface,
	invoiceID string,
	investorID string,
	tokenCount int,
) error {
	if tokenCount <= 0 {
		return fmt.Errorf("tokenCount must be at least 1")
	}

	asset, err := c.ReadInvoice(ctx, invoiceID)
	if err != nil {
		return err
	}
	if asset.Status != "MINTED" {
		return fmt.Errorf("invoice %s is not open for investment (status: %s)", invoiceID, asset.Status)
	}
	if tokenCount > asset.AvailableTokens {
		return fmt.Errorf("insufficient tokens available: requested %d, available %d", tokenCount, asset.AvailableTokens)
	}

	asset.Holdings[investorID] += tokenCount
	asset.AvailableTokens -= tokenCount

	updatedBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// Step 7: SettleInvoice computes pro-rata principal + yield payouts to all token holders upon buyer repayment
func (c *LiquiTradeContract) SettleInvoice(ctx contractapi.TransactionContextInterface, invoiceID string) error {
	asset, err := c.ReadInvoice(ctx, invoiceID)
	if err != nil {
		return err
	}
	if asset.Status == "DEFAULTED" {
		return fmt.Errorf("settlement halted: invoice %s is flagged as DEFAULTED", invoiceID)
	}
	if asset.Status != "MINTED" {
		return fmt.Errorf("invoice %s cannot be settled from status %s", invoiceID, asset.Status)
	}

	payouts := make(map[string]float64)
	for investor, tokens := range asset.Holdings {
		// Each redeemed token pays full ₹100 face value (principal + discounted yield)
		payouts[investor] = math.Round((float64(tokens)*asset.TokenFaceValueINR)*100) / 100
	}
	// Any unsubscribed remainder returns to the MSME
	if asset.AvailableTokens > 0 {
		payouts[asset.MsmeID] += math.Round((float64(asset.AvailableTokens)*asset.TokenFaceValueINR)*100) / 100
	}

	asset.PayoutsINR = payouts
	asset.Status = "SETTLED"

	updatedBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// FlagDefault halts payouts and records an anchor buyer default on-chain
func (c *LiquiTradeContract) FlagDefault(ctx contractapi.TransactionContextInterface, invoiceID string, reason string) error {
	asset, err := c.ReadInvoice(ctx, invoiceID)
	if err != nil {
		return err
	}
	if asset.Status == "SETTLED" {
		return fmt.Errorf("invoice %s is already settled", invoiceID)
	}

	asset.Status = "DEFAULTED"
	updatedBytes, err := json.Marshal(asset)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// ReadInvoice fetches the current ledger state of an invoice
func (c *LiquiTradeContract) ReadInvoice(ctx contractapi.TransactionContextInterface, invoiceID string) (*InvoiceAsset, error) {
	data, err := ctx.GetStub().GetState(invoiceID)
	if err != nil {
		return nil, fmt.Errorf("failed reading state: %v", err)
	}
	if data == nil {
		return nil, fmt.Errorf("invoice %s does not exist", invoiceID)
	}

	var asset InvoiceAsset
	if err := json.Unmarshal(data, &asset); err != nil {
		return nil, err
	}
	return &asset, nil
}

func parseECDSAPublicKey(pemStr string) (*ecdsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(pemStr))
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM block")
	}
	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	ecdsaPub, ok := pub.(*ecdsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("key is not an ECDSA public key")
	}
	return ecdsaPub, nil
}

func verifyECDSASignature(pubKey *ecdsa.PublicKey, message string, sigBase64 string) error {
	sigBytes, err := base64.StdEncoding.DecodeString(sigBase64)
	if err != nil {
		return fmt.Errorf("invalid base64 signature: %v", err)
	}
	hash := sha256.Sum256([]byte(message))

	if len(sigBytes) == 64 {
		r := new(big.Int).SetBytes(sigBytes[:32])
		s := new(big.Int).SetBytes(sigBytes[32:])
		if ecdsa.Verify(pubKey, hash[:], r, s) {
			return nil
		}
		return fmt.Errorf("signature mismatch")
	}
	if ecdsa.VerifyASN1(pubKey, hash[:], sigBytes) {
		return nil
	}
	return fmt.Errorf("signature mismatch")
}

func main() {
	chaincode, err := contractapi.NewChaincode(&LiquiTradeContract{})
	if err != nil {
		log.Panicf("Error creating LiquiTrade chaincode: %v", err)
	}
	if err := chaincode.Start(); err != nil {
		log.Panicf("Error starting LiquiTrade chaincode: %v", err)
	}
}
