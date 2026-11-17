import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
import networkx as nx
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

app = FastAPI(title="LiquiTrade Deterministic Velocity & Qwen3 8B Risk Gate", version="3.0.0")

KEY_DIR = Path("keys")
PRIV_KEY_PATH = KEY_DIR / "ai_private.pem"
PUB_KEY_PATH = KEY_DIR / "ai_public.pem"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3:8b")


def load_or_generate_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if PRIV_KEY_PATH.exists() and PUB_KEY_PATH.exists():
        private_key = serialization.load_pem_private_key(
            PRIV_KEY_PATH.read_bytes(), password=None
        )
        public_pem = PUB_KEY_PATH.read_text()
        return private_key, public_pem

    private_key = ec.generate_private_key(ec.SECP256R1())
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    PRIV_KEY_PATH.write_bytes(priv_pem)
    PUB_KEY_PATH.write_text(pub_pem.decode("utf-8"))
    return private_key, pub_pem.decode("utf-8")


PRIVATE_KEY, PUBLIC_KEY_PEM = load_or_generate_keys()

# Synthetic 6-month historical monthly average baselines (INR) per MSME GSTIN
MSME_6M_AVG_BASELINE: Dict[str, float] = {
    "33AABCU9603R1ZM": 4500000.0,  # ₹45 Lakh avg -> ₹50L invoice is +11.1% (TIER_A)
    "29AAACR5055K1Z7": 1200000.0,  # ₹12 Lakh avg
    "27AAACA8567B1Z2": 400000.0,   # ₹4 Lakh avg -> ₹9.5L invoice is +137.5% (QUARANTINED)
}

# Supplementary Isolation Forest anomaly detector
rng = np.random.default_rng(42)
normal_invoices = np.column_stack([
    rng.lognormal(mean=14.5, sigma=0.7, size=500),
    rng.integers(30, 121, size=500),
    rng.choice([5.0, 12.0, 18.0], size=500),
    np.zeros(500),
])
anomaly_model = IsolationForest(contamination=0.05, random_state=42)
anomaly_model.fit(normal_invoices)

SEEN_HASHES: Set[str] = set()
TRADE_GRAPH = nx.DiGraph()


class RawInvoiceTextRequest(BaseModel):
    rawText: str = Field(
        ...,
        example="GST Tax Invoice INV-2026-501 | Seller GSTIN: 33AABCU9603R1ZM | Buyer GSTIN: 29AAACR5055K1Z7 | IRN: IRN50L9981 | Total: INR 5000000.00 | Due: 2026-12-30 | Items: Precision CNC automotive shafts (HSN 8708)",
    )


class InvoiceRequest(BaseModel):
    invoiceId: str = Field(..., example="INV-2026-501")
    msmeId: str = Field(..., example="MSME-TN-01")
    buyerId: str = Field(default="BUYER-CORP-01", example="BUYER-CORP-01")
    sellerGstin: str = Field(..., example="33AABCU9603R1ZM")
    buyerGstin: str = Field(..., example="29AAACR5055K1Z7")
    amountInr: float = Field(..., gt=0, example=5000000.0)
    sixMonthAvgInr: Optional[float] = Field(
        default=None,
        description="Optional override for MSME 6-month rolling average invoice size (INR)",
    )
    gstRatePct: float = Field(default=18.0, example=18.0)
    tenorDays: int = Field(default=90, example=90)
    annualDiscountPct: float = Field(default=12.0, example=12.0)
    dueDate: str = Field(..., example="2026-12-30")
    irn: str = Field(..., example="IRN50L9981")
    lineItemDescription: str = Field(
        default="Precision CNC automotive shafts - 2,000 units (HSN 8708)",
    )


class AttestationPayload(BaseModel):
    invoiceId: str
    invoiceHash: str
    riskTier: str
    velocityDeltaPct: float
    fraudScore: float
    approved: bool
    signature: str


class EvaluationResponse(BaseModel):
    attestation: AttestationPayload
    sixMonthAvgInr: float
    qwenExplanation: str
    reasons: List[str]
    registerChaincodeArgs: List[str]
    issueTokenChaincodeArgs: List[str]


def compute_invoice_hash(req: InvoiceRequest) -> str:
    canonical = f"{req.invoiceId}|{req.sellerGstin}|{req.buyerGstin}|{req.amountInr:.2f}|{req.dueDate}|{req.irn}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sign_attestation(
    invoice_id: str,
    invoice_hash: str,
    risk_tier: str,
    velocity_delta_pct: float,
    fraud_score: float,
    approved: bool,
) -> str:
    # Matches Go chaincode: fmt.Sprintf("%s|%s|%s|%.2f|%.2f|%t", ...)
    approved_str = "true" if approved else "false"
    canonical_msg = (
        f"{invoice_id}|{invoice_hash}|{risk_tier}|{velocity_delta_pct:.2f}|{fraud_score:.2f}|{approved_str}"
    ).encode("utf-8")
    der_sig = PRIVATE_KEY.sign(canonical_msg, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(der_sig).decode("utf-8")


def call_qwen_json(prompt: str) -> dict:
    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": QWEN_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1},
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return json.loads(resp.json().get("response", "{}"))


def explain_with_qwen(req: InvoiceRequest, tier: str, velocity_delta: float, flags: List[str]) -> str:
    """Slide 6 rule: Deterministic velocity gate decides the tier; local Qwen3 8B explains the result."""
    prompt = f"""You are an MSME trade-finance risk explainer for NPCI Drunix LiquiTrade.
A deterministic velocity engine has already classified this invoice:
- Invoice ID: {req.invoiceId}
- Amount: INR {req.amountInr:,.2f} (Tenor: {req.tenorDays} days)
- 6-Month Velocity Delta: {velocity_delta:+.1f}%
- Deterministic Risk Tier: {tier}
- Line Items: "{req.lineItemDescription}"
-Triggered Rules: {flags}

Provide a clear, concise 1-to-2 sentence audit explanation for the MSME and institutional LP dashboards, also noting whether the line-item description matches standard B2B trade.
Respond ONLY in JSON: {{"explanation": "..."}}"""
    try:
        parsed = call_qwen_json(prompt)
        return str(parsed.get("explanation", f"Classified as {tier} ({velocity_delta:+.1f}% vs 6-month average)."))
    except Exception as exc:
        return f"Classified as {tier} ({velocity_delta:+.1f}% vs 6-month average). [Qwen3 offline: {type(exc).__name__}]"


@app.get("/public-key")
def get_public_key():
    return {"publicKeyPem": PUBLIC_KEY_PEM}


@app.post("/extract-invoice")
def extract_invoice_fields(payload: RawInvoiceTextRequest):
    """Slide 3 Step 1 & Slide 6: Local Qwen3 8B extracts structured e-invoice JSON from raw text."""
    prompt = f"""Extract structured GST e-invoice fields from the following text into valid JSON with keys:
invoiceId, sellerGstin, buyerGstin, amountInr (number), gstRatePct (number), tenorDays (integer), dueDate (YYYY-MM-DD), irn, lineItemDescription.

Text:
\"\"\"{payload.rawText}\"\"\""""
    try:
        extracted = call_qwen_json(prompt)
        return {"status": "extracted", "model": QWEN_MODEL, "structuredInvoice": extracted}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_receivable(req: InvoiceRequest):
    reasons: List[str] = []
    inv_hash = compute_invoice_hash(req)

    # 1. Deterministic 6-Month Rolling Velocity Check (Slide 6)
    baseline_avg = req.sixMonthAvgInr or MSME_6M_AVG_BASELINE.get(req.sellerGstin.upper(), 1000000.0)
    velocity_delta = round(((req.amountInr - baseline_avg) / baseline_avg) * 100.0, 2)

    if velocity_delta <= 25.0:
        tier = "TIER_A"
        risk_score = 12.0
        reasons.append(f"Velocity +{velocity_delta:.1f}% (<= +25%): TIER_A auto-clear, eligible for 20% institutional LP buy")
    elif velocity_delta <= 100.0:
        tier = "TIER_B"
        risk_score = 32.0
        reasons.append(f"Velocity +{velocity_delta:.1f}% (+25% to +100%): TIER_B cleared without institutional auto-buy")
    else:
        tier = "QUARANTINED"
        risk_score = 88.0
        reasons.append(f"Velocity +{velocity_delta:.1f}% (> +100% spike vs 6-month avg): QUARANTINED")

    # 2. Hard Fraud & Graph Gates (Override to QUARANTINED if triggered)
    if inv_hash in SEEN_HASHES:
        tier = "QUARANTINED"
        risk_score = 98.0
        reasons.append("Duplicate invoice IRN/hash detected (double-financing blocked)")

    if req.sellerGstin.strip().upper() == req.buyerGstin.strip().upper():
        tier = "QUARANTINED"
        risk_score = 99.0
        reasons.append("Seller and Buyer GSTIN are identical (self-invoicing fraud)")

    TRADE_GRAPH.add_edge(req.sellerGstin, req.buyerGstin, amount=req.amountInr)
    if nx.has_path(TRADE_GRAPH, req.buyerGstin, req.sellerGstin):
        tier = "QUARANTINED"
        risk_score = 95.0
        reasons.append("Circular trading cycle detected in GSTIN transaction graph")

    # 3. Supplementary Isolation Forest check
    is_round = 1.0 if (req.amountInr >= 100000 and req.amountInr % 50000 == 0) else 0.0
    features = np.array([[req.amountInr, req.tenorDays, req.gstRatePct, is_round]])
    if anomaly_model.decision_function(features)[0] < 0 and tier != "QUARANTINED":
        risk_score = min(65.0, risk_score + 12.0)
        reasons.append("IsolationForest flagged minor statistical variance")

    fraud_score = round(min(100.0, max(0.0, risk_score)), 2)
    approved = tier in ("TIER_A", "TIER_B")

    # 4. Local Qwen3 8B Audit Explanation (explains deterministic result, does not decide)
    qwen_explanation = explain_with_qwen(req, tier, velocity_delta, reasons)

    if approved:
        SEEN_HASHES.add(inv_hash)

    signature = sign_attestation(req.invoiceId, inv_hash, tier, velocity_delta, fraud_score, approved)

    attestation = AttestationPayload(
        invoiceId=req.invoiceId,
        invoiceHash=inv_hash,
        riskTier=tier,
        velocityDeltaPct=velocity_delta,
        fraudScore=fraud_score,
        approved=approved,
        signature=signature,
    )

    return EvaluationResponse(
        attestation=attestation,
        sixMonthAvgInr=baseline_avg,
        qwenExplanation=qwen_explanation,
        reasons=reasons,
        registerChaincodeArgs=[
            req.invoiceId,
            inv_hash,
            req.irn,
            req.msmeId,
            req.buyerId,
            req.buyerGstin,
            f"{req.amountInr:.2f}",
            str(req.tenorDays),
            req.dueDate,
        ],
        issueTokenChaincodeArgs=[
            req.invoiceId,
            f"{req.annualDiscountPct:.2f}",
            attestation.model_dump_json(),
        ],
    )
