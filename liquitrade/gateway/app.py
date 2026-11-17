import asyncio
import hashlib
import math
import time
import uuid
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="LiquiTrade Gateway & Mock UPI Settlement Adapter", version="1.0.0")

AI_GATEKEEPER_URL = "http://127.0.0.1:8000"
TOKEN_FACE_VALUE = 100.0
INSTITUTION_LP_ID = "INSTITUTION_LP_DEFAULT"

# In-memory Drunix Ledger & YugabyteDB Portfolio Projection (mirrors liquitrade.go state)
LEDGER_INVOICES: Dict[str, dict] = {}
REGISTERED_HASHES: Dict[str, str] = {}
UPI_TRANSACTIONS: List[dict] = []
INVOICE_LOCKS: Dict[str, asyncio.Lock] = {}


def get_lock(invoice_id: str) -> asyncio.Lock:
    if invoice_id not in INVOICE_LOCKS:
        INVOICE_LOCKS[invoice_id] = asyncio.Lock()
    return INVOICE_LOCKS[invoice_id]


class UploadInvoiceReq(BaseModel):
    invoiceId: str = Field(..., example="INV-50L-DEMO")
    msmeId: str = Field(default="MSME-TN-01")
    buyerId: str = Field(default="BUYER-CORP-01")
    sellerGstin: str = Field(default="33AABCU9603R1ZM")
    buyerGstin: str = Field(default="29AAACR5055K1Z7")
    amountInr: float = Field(default=5000000.0, gt=0)
    tenorDays: int = Field(default=90, gt=0)
    dueDate: str = Field(default="2026-12-30")
    irn: str = Field(default="IRN50LDEMO01")
    lineItemDescription: str = Field(
        default="Precision CNC automotive shafts - 2,000 units (HSN 8708)"
    )


class BuyerAcceptReq(BaseModel):
    buyerId: str = Field(default="BUYER-CORP-01")


class RetailBidReq(BaseModel):
    investorId: str = Field(..., example="RETAIL-USER-001")
    upiVpa: str = Field(default="investor@okicici")
    tokenCount: int = Field(..., ge=1, example=10)


class SimulateUpiPaymentReq(BaseModel):
    payerVpa: str
    payeeVpa: str
    amountInr: float
    referenceId: str


@app.post("/upi/debit")
async def mock_upi_debit(req: SimulateUpiPaymentReq):
    """Simulated NPCI UPI rail adapter for retail ticket debits & buyer settlement."""
    tx_ref = f"UPI{uuid.uuid4().hex[:12].upper()}"
    record = {
        "upiRef": tx_ref,
        "payerVpa": req.payerVpa,
        "payeeVpa": req.payeeVpa,
        "amountInr": round(req.amountInr, 2),
        "referenceId": req.referenceId,
        "status": "SUCCESS",
        "timestamp": time.time(),
    }
    UPI_TRANSACTIONS.append(record)
    return record


@app.post("/invoices/upload")
async def step1_register_invoice(req: UploadInvoiceReq):
    """Step 1: Register structured e-invoice with unique IRN/hash check."""
    canonical = f"{req.invoiceId}|{req.sellerGstin}|{req.buyerGstin}|{req.amountInr:.2f}|{req.dueDate}|{req.irn}"
    inv_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if req.invoiceId in LEDGER_INVOICES:
        raise HTTPException(status_code=409, detail=f"Invoice {req.invoiceId} already exists")
    if inv_hash in REGISTERED_HASHES:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate IRN/hash blocked (already registered as {REGISTERED_HASHES[inv_hash]})",
        )

    asset = {
        "invoiceId": req.invoiceId,
        "invoiceHash": inv_hash,
        "irn": req.irn,
        "msmeId": req.msmeId,
        "buyerId": req.buyerId,
        "sellerGstin": req.sellerGstin,
        "buyerGstin": req.buyerGstin,
        "amountInr": req.amountInr,
        "tenorDays": req.tenorDays,
        "annualDiscountPct": 12.0,
        "dueDate": req.dueDate,
        "lineItemDescription": req.lineItemDescription,
        "status": "REGISTERED",
        "riskTier": None,
        "velocityDeltaPct": None,
        "fraudScore": None,
        "qwenExplanation": None,
        "totalTokens": 0,
        "availableTokens": 0,
        "tokenFaceValueInr": TOKEN_FACE_VALUE,
        "tokenOfferPriceInr": 0.0,
        "holdings": {},
        "payoutsInr": {},
        "version": 1,
    }
    LEDGER_INVOICES[req.invoiceId] = asset
    REGISTERED_HASHES[inv_hash] = req.invoiceId
    return {"step": "1_REGISTERED", "invoice": asset}


@app.post("/invoices/{invoice_id}/accept")
async def step2_buyer_accept(invoice_id: str, req: BuyerAcceptReq):
    """Step 2: Anchor buyer on-chain acceptance."""
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if asset["status"] != "REGISTERED":
        raise HTTPException(status_code=400, detail=f"Invalid state: {asset['status']}")
    if asset["buyerId"] != req.buyerId:
        raise HTTPException(status_code=403, detail="Buyer identity mismatch")

    asset["status"] = "ACCEPTED"
    asset["version"] += 1
    return {"step": "2_ACCEPTED", "invoice": asset}


@app.post("/invoices/{invoice_id}/screen-and-mint")
async def step3_4_5_ai_gate_and_mint(invoice_id: str, annual_discount_pct: float = 12.0):
    """Steps 3, 4 & 5: Calls FastAPI AI Risk Gate, verifies attestation, mints N tokens, & executes 20% LP auto-buy."""
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if asset["status"] != "ACCEPTED":
        raise HTTPException(status_code=400, detail="Buyer must accept invoice before minting")

    async with httpx.AsyncClient(timeout=35.0) as client:
        ai_resp = await client.post(
            f"{AI_GATEKEEPER_URL}/evaluate",
            json={
                "invoiceId": asset["invoiceId"],
                "msmeId": asset["msmeId"],
                "buyerId": asset["buyerId"],
                "sellerGstin": asset["sellerGstin"],
                "buyerGstin": asset["buyerGstin"],
                "amountInr": asset["amountInr"],
                "tenorDays": asset["tenorDays"],
                "annualDiscountPct": annual_discount_pct,
                "dueDate": asset["dueDate"],
                "irn": asset["irn"],
                "lineItemDescription": asset["lineItemDescription"],
            },
        )
        ai_resp.raise_for_status()
        ai_data = ai_resp.json()

    att = ai_data["attestation"]
    asset["riskTier"] = att["riskTier"]
    asset["velocityDeltaPct"] = att["velocityDeltaPct"]
    asset["fraudScore"] = att["fraudScore"]
    asset["qwenExplanation"] = ai_data["qwenExplanation"]

    if not att["approved"] or att["riskTier"] == "QUARANTINED":
        asset["status"] = "QUARANTINED"
        asset["version"] += 1
        raise HTTPException(
            status_code=422,
            detail={
                "status": "QUARANTINED",
                "reasons": ai_data["reasons"],
                "qwenExplanation": ai_data["qwenExplanation"],
            },
        )

    total_tokens = int(math.floor(asset["amountInr"] / TOKEN_FACE_VALUE))
    offer_price = round(
        TOKEN_FACE_VALUE * (1.0 - (annual_discount_pct / 100.0) * (asset["tenorDays"] / 365.0)),
        2,
    )

    asset["annualDiscountPct"] = annual_discount_pct
    asset["totalTokens"] = total_tokens
    asset["availableTokens"] = total_tokens
    asset["tokenOfferPriceInr"] = offer_price
    asset["status"] = "MINTED"
    asset["version"] += 1

    # Step 5: Automatic 20% Institutional LP Allocation on Tier A
    auto_buy_upi = None
    if att["riskTier"] == "TIER_A":
        inst_tokens = int(math.floor(total_tokens * 0.20))
        asset["holdings"][INSTITUTION_LP_ID] = inst_tokens
        asset["availableTokens"] -= inst_tokens
        auto_buy_upi = await mock_upi_debit(
            SimulateUpiPaymentReq(
                payerVpa="treasury.lp@yesbank",
                payeeVpa="escrow@liquitrade.npci",
                amountInr=inst_tokens * offer_price,
                referenceId=f"{invoice_id}-INST-20PCT",
            )
        )

    return {
        "step": "3_4_5_MINTED",
        "attestation": att,
        "qwenExplanation": ai_data["qwenExplanation"],
        "institutionalAutoBuyUpi": auto_buy_upi,
        "invoice": asset,
    }


@app.post("/invoices/{invoice_id}/purchase")
async def step6_fractional_purchase(invoice_id: str, req: RetailBidReq):
    """Step 6: Atomic retail fractional purchase with MVCC oversubscription protection."""
    lock = get_lock(invoice_id)
    async with lock:
        asset = LEDGER_INVOICES.get(invoice_id)
        if not asset:
            raise HTTPException(status_code=404, detail="Invoice not found")
        if asset["status"] != "MINTED":
            raise HTTPException(status_code=400, detail=f"Invoice not open (status: {asset['status']})")
        if req.tokenCount > asset["availableTokens"]:
            raise HTTPException(
                status_code=409,
                detail=f"MVCC Oversubscription Blocked: requested {req.tokenCount}, available {asset['availableTokens']}",
            )

        debit_amount = round(req.tokenCount * asset["tokenOfferPriceInr"], 2)
        upi_receipt = await mock_upi_debit(
            SimulateUpiPaymentReq(
                payerVpa=req.upiVpa,
                payeeVpa="escrow@liquitrade.npci",
                amountInr=debit_amount,
                referenceId=f"{invoice_id}-{req.investorId}",
            )
        )

        asset["holdings"][req.investorId] = asset["holdings"].get(req.investorId, 0) + req.tokenCount
        asset["availableTokens"] -= req.tokenCount
        asset["version"] += 1

        return {
            "step": "6_PURCHASED",
            "investorId": req.investorId,
            "tokensBought": req.tokenCount,
            "paidInr": debit_amount,
            "remainingTokens": asset["availableTokens"],
            "upiReceipt": upi_receipt,
        }


@app.post("/invoices/{invoice_id}/settle")
async def step7_settle_invoice(invoice_id: str):
    """Step 7: Anchor buyer repays full face value; pro-rata principal + yield distributed to all holders."""
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if asset["status"] != "MINTED":
        raise HTTPException(status_code=400, detail=f"Cannot settle in status {asset['status']}")

    payouts: Dict[str, float] = {}
    for holder, count in asset["holdings"].items():
        payouts[holder] = round(count * TOKEN_FACE_VALUE, 2)
    if asset["availableTokens"] > 0:
        payouts[asset["msmeId"]] = round(
            payouts.get(asset["msmeId"], 0.0) + asset["availableTokens"] * TOKEN_FACE_VALUE, 2
        )

    asset["payoutsInr"] = payouts
    asset["status"] = "SETTLED"
    asset["version"] += 1

    return {
        "step": "7_SETTLED",
        "totalHolders": len(asset["holdings"]),
        "payoutsInr": payouts,
        "invoice": asset,
    }


@app.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return asset
