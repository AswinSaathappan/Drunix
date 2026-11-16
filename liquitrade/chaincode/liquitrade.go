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
	"math/big"

	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

// LiquiTradeContract manages AI-gated MSME receivable tokenization on NPCI Drunix
type LiquiTradeContract struct {
	contractapi.Contract
}

// ReceivableToken represents a tokenized MSME invoice on the ledger
type ReceivableToken struct {
	InvoiceID       string  `json:"invoiceId"`
	InvoiceHash     string  `json:"invoiceHash"`
	MsmeID          string  `json:"msmeId"`
	BuyerGSTIN      string  `json:"buyerGstin"`
	AmountINR       float64 `json:"amountInr"`
	DueDate         string  `json:"dueDate"`
	FraudScore      float64 `json:"fraudScore"`
	AIApproved      bool    `json:"aiApproved"`
	Owner           string  `json:"owner"`
	Status          string  `json:"status"` // TOKENIZED, FINANCED, SETTLED
	TokenizedAtTxID string  `json:"tokenizedAtTxId"`
}

// AIAttestation is the payload signed by the off-chain AI Fraud Gatekeeper
type AIAttestation struct {
	InvoiceID   string  `json:"invoiceId"`
	InvoiceHash string  `json:"invoiceHash"`
	FraudScore  float64 `json:"fraudScore"`
	Approved    bool    `json:"approved"`
	Signature   string  `json:"signature"` // Base64-encoded ECDSA signature
}

const (
	aiPublicKeyLedgerKey = "CONFIG_AI_GATEKEEPER_PUBKEY"
	maxFraudThreshold    = 35.0 // Reject invoices with risk score > 35/100
)

// InitLedger registers the trusted AI Fraud Gatekeeper ECDSA public key (PEM format)
func (c *LiquiTradeContract) InitLedger(ctx contractapi.TransactionContextInterface, aiPublicKeyPEM string) error {
	if _, err := parseECDSAPublicKey(aiPublicKeyPEM); err != nil {
		return fmt.Errorf("invalid AI gatekeeper PEM public key: %v", err)
	}
	return ctx.GetStub().PutState(aiPublicKeyLedgerKey, []byte(aiPublicKeyPEM))
}

// TokenizeReceivable mints a receivable token only if the AI Gatekeeper cryptographically approved it
func (c *LiquiTradeContract) TokenizeReceivable(
	ctx contractapi.TransactionContextInterface,
	invoiceID string,
	invoiceHash string,
	msmeID string,
	buyerGSTIN string,
	amountINR float64,
	dueDate string,
	attestationJSON string,
) error {
	if amountINR <= 0 {
		return fmt.Errorf("invoice amount must be positive")
	}

	// 1. Prevent double-financing / duplicate tokenization
	existing, err := ctx.GetStub().GetState(invoiceID)
	if err != nil {
		return fmt.Errorf("failed reading ledger state: %v", err)
	}
	if existing != nil {
		return fmt.Errorf("receivable %s is already tokenized (duplicate financing blocked)", invoiceID)
	}

	// 2. Parse AI Attestation
	var attestation AIAttestation
	if err := json.Unmarshal([]byte(attestationJSON), &attestation); err != nil {
		return fmt.Errorf("invalid AI attestation JSON: %v", err)
	}

	if attestation.InvoiceID != invoiceID || attestation.InvoiceHash != invoiceHash {
		return fmt.Errorf("attestation mismatch: invoice metadata does not match signed payload")
	}

	// 3. Enforce AI Gate thresholds
	if !attestation.Approved || attestation.FraudScore > maxFraudThreshold {
		return fmt.Errorf("AI gate rejected tokenization: approved=%v, fraudScore=%.2f (max %.2f)",
			attestation.Approved, attestation.FraudScore, maxFraudThreshold)
	}

	// 4. Verify ECDSA cryptographic signature from the registered AI Gatekeeper
	pubKeyBytes, err := ctx.GetStub().GetState(aiPublicKeyLedgerKey)
	if err != nil || pubKeyBytes == nil {
		return fmt.Errorf("AI gatekeeper public key not initialized on ledger")
	}

	pubKey, err := parseECDSAPublicKey(string(pubKeyBytes))
	if err != nil {
		return fmt.Errorf("stored AI public key is invalid: %v", err)
	}

	canonicalMsg := fmt.Sprintf("%s|%s|%.2f|%t", attestation.InvoiceID, attestation.InvoiceHash, attestation.FraudScore, attestation.Approved)
	if err := verifyECDSASignature(pubKey, canonicalMsg, attestation.Signature); err != nil {
		return fmt.Errorf("AI signature verification failed: %v", err)
	}

	// 5. Mint tokenized receivable onto Drunix state database (YugabyteDB / LevelDB)
	token := ReceivableToken{
		InvoiceID:       invoiceID,
		InvoiceHash:     invoiceHash,
		MsmeID:          msmeID,
		BuyerGSTIN:      buyerGSTIN,
		AmountINR:       amountINR,
		DueDate:         dueDate,
		FraudScore:      attestation.FraudScore,
		AIApproved:      true,
		Owner:           msmeID,
		Status:          "TOKENIZED",
		TokenizedAtTxID: ctx.GetStub().GetTxID(),
	}

	tokenBytes, err := json.Marshal(token)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(invoiceID, tokenBytes)
}

// TransferToFinancier transfers ownership of a tokenized receivable to a liquidity provider/bank
func (c *LiquiTradeContract) TransferToFinancier(ctx contractapi.TransactionContextInterface, invoiceID string, financierID string) error {
	token, err := c.ReadReceivable(ctx, invoiceID)
	if err != nil {
		return err
	}
	if token.Status != "TOKENIZED" {
		return fmt.Errorf("receivable %s cannot be financed in status %s", invoiceID, token.Status)
	}

	token.Owner = financierID
	token.Status = "FINANCED"

	updatedBytes, err := json.Marshal(token)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// SettleReceivable marks the receivable paid and closed upon buyer settlement
func (c *LiquiTradeContract) SettleReceivable(ctx contractapi.TransactionContextInterface, invoiceID string) error {
	token, err := c.ReadReceivable(ctx, invoiceID)
	if err != nil {
		return err
	}
	if token.Status == "SETTLED" {
		return fmt.Errorf("receivable %s is already settled", invoiceID)
	}

	token.Status = "SETTLED"
	updatedBytes, err := json.Marshal(token)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(invoiceID, updatedBytes)
}

// ReadReceivable fetches a tokenized receivable by ID
func (c *LiquiTradeContract) ReadReceivable(ctx contractapi.TransactionContextInterface, invoiceID string) (*ReceivableToken, error) {
	data, err := ctx.GetStub().GetState(invoiceID)
	if err != nil {
		return nil, fmt.Errorf("failed to read state: %v", err)
	}
	if data == nil {
		return nil, fmt.Errorf("receivable %s does not exist", invoiceID)
	}

	var token ReceivableToken
	if err := json.Unmarshal(data, &token); err != nil {
		return nil, err
	}
	return &token, nil
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

	// Support raw 64-byte (R || S) P-256 signatures
	if len(sigBytes) == 64 {
		r := new(big.Int).SetBytes(sigBytes[:32])
		s := new(big.Int).SetBytes(sigBytes[32:])
		if ecdsa.Verify(pubKey, hash[:], r, s) {
			return nil
		}
		return fmt.Errorf("signature mismatch")
	}

	// Support ASN.1 DER format (default output of Python cryptography ECDSA signing)
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
