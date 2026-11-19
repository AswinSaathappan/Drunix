import asyncio
import hashlib
import math
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LiquiTrade Gateway, OCR Bridge & Mock UPI Adapter", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_GATEKEEPER_URL = "http://127.0.0.1:8000"
TOKEN_FACE_VALUE = 100.0
INSTITUTION_LP_ID = "INSTITUTION_LP_DEFAULT"
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"

LEDGER_INVOICES: Dict[str, dict] = {}
REGISTERED_HASHES: Dict[str, str] = {}
UPI_TRANSACTIONS: List[dict] = []
AUDIT_LOGS: List[dict] = []
INVOICE_LOCKS: Dict[str, asyncio.Lock] = {}


def log_event(event_type: str, detail: str):
    AUDIT_LOGS.insert(0, {"ts": time.strftime("%H:%M:%S"), "type": event_type, "detail": detail})
    del AUDIT_LOGS[100:]


def get_lock(invoice_id: str) -> asyncio.Lock:
    if invoice_id not in INVOICE_LOCKS:
        INVOICE_LOCKS[invoice_id] = asyncio.Lock()
    return INVOICE_LOCKS[invoice_id]


class CloudLLMConfig(BaseModel):
    apiKey: str = ""
    baseUrl: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    preferCloud: bool = False


class UploadInvoiceReq(BaseModel):
    invoiceId: str
    msmeId: str = "MSME-TN-01"
    buyerId: str = "BUYER-CORP-01"
    sellerGstin: str = "33AABCU9603R1ZM"
    buyerGstin: str = "29AAACR5055K1Z7"
    amountInr: float = Field(..., gt=0)
    tenorDays: int = Field(default=90, gt=0)
    dueDate: str = "2026-12-30"
    irn: str
    lineItemDescription: str = "Precision CNC automotive shafts - 2,000 units (HSN 8708)"


class BuyerAcceptReq(BaseModel):
    buyerId: str = "BUYER-CORP-01"


class ScreenMintReq(BaseModel):
    annualDiscountPct: float = 12.0
    llmConfig: Optional[CloudLLMConfig] = None


class RetailBidReq(BaseModel):
    investorId: str
    upiVpa: str = "investor@okicici"
    tokenCount: int = Field(..., ge=1)


class SimulateUpiPaymentReq(BaseModel):
    payerVpa: str
    payeeVpa: str
    amountInr: float
    referenceId: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return STATIC_INDEX.read_text(encoding="utf-8")


@app.get("/state")
async def get_full_state():
    return {
        "invoices": list(LEDGER_INVOICES.values()),
        "upiTransactions": UPI_TRANSACTIONS[:30],
        "auditLogs": AUDIT_LOGS[:40],
    }


@app.post("/ocr")
async def proxy_ocr_upload(
    file: UploadFile = File(...),
    apiKey: str = Form(""),
    baseUrl: str = Form("https://api.groq.com/openai/v1"),
    model: str = Form("llama-3.3-70b-versatile"),
    preferCloud: bool = Form(False),
):
    file_bytes = await file.read()
    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            f"{AI_GATEKEEPER_URL}/ocr-extract",
            files={"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")},
            data={
                "apiKey": apiKey,
                "baseUrl": baseUrl,
                "model": model,
                "preferCloud": str(preferCloud).lower(),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        log_event("OCR_EXTRACT", f"Extracted '{file.filename}' via {data.get('llmProviderUsed')}")
        return data


@app.post("/upi/debit")
async def mock_upi_debit(req: SimulateUpiPaymentReq):
    tx_ref = f"UPI{uuid.uuid4().hex[:12].upper()}"
    record = {
        "upiRef": tx_ref,
        "payerVpa": req.payerVpa,
        "payeeVpa": req.payeeVpa,
        "amountInr": round(req.amountInr, 2),
        "referenceId": req.referenceId,
        "status": "SUCCESS",
        "timestamp": time.strftime("%H:%M:%S"),
    }
    UPI_TRANSACTIONS.insert(0, record)
    return record


@app.post("/invoices/upload")
async def step1_register_invoice(req: UploadInvoiceReq):
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
        "llmProviderUsed": None,
        "qwenExplanation": None,
        "reasons": [],
        "totalTokens": 0,
        "availableTokens": 0,
        "tokenFaceValueInr": TOKEN_FACE_VALUE,
        "tokenOfferPriceInr": 0.0,
        "holdings": {},
        "payoutsInr": {},
    }
    LEDGER_INVOICES[req.invoiceId] = asset
    REGISTERED_HASHES[inv_hash] = req.invoiceId
    log_event("1_REGISTERED", f"{req.invoiceId} (₹{req.amountInr:,.2f}) registered on Drunix")
    return {"step": "1_REGISTERED", "invoice": asset}


@app.post("/invoices/{invoice_id}/accept")
async def step2_buyer_accept(invoice_id: str, req: BuyerAcceptReq):
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if asset["status"] != "REGISTERED":
        raise HTTPException(status_code=400, detail=f"Invalid state: {asset['status']}")
    asset["status"] = "ACCEPTED"
    log_event("2_ACCEPTED", f"Buyer {req.buyerId} confirmed {invoice_id} on-chain")
    return {"step": "2_ACCEPTED", "invoice": asset}


@app.post("/invoices/{invoice_id}/screen-and-mint")
async def step3_4_5_ai_gate_and_mint(invoice_id: str, body: Optional[ScreenMintReq] = None):
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    if asset["status"] != "ACCEPTED":
        raise HTTPException(status_code=400, detail="Buyer must accept invoice before minting")

    discount = body.annualDiscountPct if body else 12.0
    llm_cfg = body.llmConfig.model_dump() if (body and body.llmConfig) else None

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
                "annualDiscountPct": discount,
                "dueDate": asset["dueDate"],
                "irn": asset["irn"],
                "lineItemDescription": asset["lineItemDescription"],
                "llmConfig": llm_cfg,
            },
        )
        ai_resp.raise_for_status()
        ai_data = ai_resp.json()

    att = ai_data["attestation"]
    asset["riskTier"] = att["riskTier"]
    asset["velocityDeltaPct"] = att["velocityDeltaPct"]
    asset["fraudScore"] = att["fraudScore"]
    asset["llmProviderUsed"] = ai_data["llmProviderUsed"]
    asset["qwenExplanation"] = ai_data["qwenExplanation"]
    asset["reasons"] = ai_data["reasons"]

    if not att["approved"] or att["riskTier"] == "QUARANTINED":
        asset["status"] = "QUARANTINED"
        log_event("3_QUARANTINED", f"{invoice_id} blocked by AI Gate ({att['riskTier']})")
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
        TOKEN_FACE_VALUE * (1.0 - (discount / 100.0) * (asset["tenorDays"] / 365.0)),
        2,
    )

    asset["annualDiscountPct"] = discount
    asset["totalTokens"] = total_tokens
    asset["availableTokens"] = total_tokens
    asset["tokenOfferPriceInr"] = offer_price
    asset["status"] = "MINTED"

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

    log_event(
        "4_5_MINTED",
        f"{invoice_id} minted {total_tokens:,} tokens ({att['riskTier']}) via {ai_data['llmProviderUsed']}",
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
        log_event("6_PURCHASE", f"{req.investorId} bought {req.tokenCount} tokens of {invoice_id} (₹{debit_amount:,.2f})")

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
    log_event("7_SETTLED", f"{invoice_id} settled pro-rata across {len(payouts)} holders")
    return {"step": "7_SETTLED", "totalHolders": len(asset["holdings"]), "payoutsInr": payouts, "invoice": asset}


@app.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    asset = LEDGER_INVOICES.get(invoice_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return asset
