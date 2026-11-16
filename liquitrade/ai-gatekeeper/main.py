import base64
import hashlib
import json
import os
from pathlib import Path
from typing import List, Set

import httpx
import networkx as nx
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

app = FastAPI(title="LiquiTrade Hybrid AI Fraud Gatekeeper (ML + Qwen3 8B)", version="2.0.0")

KEY_DIR = Path("keys")
PRIV_KEY_PATH = KEY_DIR / "ai_private.pem"
PUB_KEY_PATH = KEY_DIR / "ai_public.pem"
FRAUD_THRESHOLD = 35.0

# Connects automatically to Windows Ollama via OLLAMA_URL set in ~/.bashrc
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

# Pre-trained Isolation Forest on normal MSME invoice distributions
rng = np.random.default_rng(42)
normal_invoices = np.column_stack([
    rng.lognormal(mean=11.5, sigma=0.7, size=500),
    rng.integers(15, 61, size=500),
    rng.choice([5.0, 12.0, 18.0], size=500),
    np.zeros(500),
])
anomaly_model = IsolationForest(contamination=0.05, random_state=42)
anomaly_model.fit(normal_invoices)

SEEN_HASHES: Set[str] = set()
TRADE_GRAPH = nx.DiGraph()


class InvoiceRequest(BaseModel):
    invoiceId: str = Field(..., example="INV-2026-001")
    msmeId: str = Field(..., example="MSME-TN-01")
    sellerGstin: str = Field(..., example="33AABCU9603R1ZM")
    buyerGstin: str = Field(..., example="29AAACR5055K1Z7")
    amountInr: float = Field(..., gt=0, example=145250.0)
    gstRatePct: float = Field(default=18.0, example=18.0)
    paymentTermDays: int = Field(default=30, example=30)
    dueDate: str = Field(..., example="2026-12-20")
    irn: str = Field(..., description="GST e-Invoice IRN hash", example="IRN998877")
    lineItemDescription: str = Field(
        default="CNC machined aluminum automotive brackets - 500 units (HSN 8708)",
        description="Goods/services description audited semantically by Qwen3 8B",
    )


class AttestationPayload(BaseModel):
    invoiceId: str
    invoiceHash: str
    fraudScore: float
    approved: bool
    signature: str


class EvaluationResponse(BaseModel):
    attestation: AttestationPayload
    qwenAuditSummary: str
    reasons: List[str]
    chaincodeArgs: List[str]


def compute_invoice_hash(req: InvoiceRequest) -> str:
    canonical = f"{req.invoiceId}|{req.sellerGstin}|{req.buyerGstin}|{req.amountInr:.2f}|{req.dueDate}|{req.irn}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sign_attestation(invoice_id: str, invoice_hash: str, fraud_score: float, approved: bool) -> str:
    approved_str = "true" if approved else "false"
    canonical_msg = f"{invoice_id}|{invoice_hash}|{fraud_score:.2f}|{approved_str}".encode("utf-8")
    der_sig = PRIVATE_KEY.sign(canonical_msg, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(der_sig).decode("utf-8")


def audit_with_qwen(req: InvoiceRequest) -> tuple[float, str]:
    """Calls local Qwen3 8B via Windows Ollama to audit semantic invoice legitimacy."""
    prompt = f"""You are an Indian MSME trade-finance fraud auditor. Analyze this GST B2B invoice for fraud red flags (vague shell-company descriptions, unrealistic pricing, GST rate mismatch, or fake goods).
Invoice ID: {req.invoiceId}
Amount (INR): {req.amountInr}
GST Rate (%): {req.gstRatePct}
Payment Terms (Days): {req.paymentTermDays}
Description: "{req.lineItemDescription}"

Respond ONLY in valid JSON with two keys:
- "semantic_risk": integer from 0 (completely legitimate B2B supply) to 40 (high fraud/shell-company risk)
- "reason": short 1-sentence explanation"""

    try:
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
        raw_text = resp.json().get("response", "{}")
        parsed = json.loads(raw_text)
        semantic_risk = float(np.clip(parsed.get("semantic_risk", 5.0), 0.0, 40.0))
        reason = str(parsed.get("reason", "Qwen3 8B semantic audit completed"))
        return semantic_risk, reason
    except Exception as exc:
        return 10.0, f"Qwen3 fallback ({type(exc).__name__}: {exc}) - used ML + graph gates only"


@app.get("/public-key")
def get_public_key():
    return {"publicKeyPem": PUBLIC_KEY_PEM}


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_receivable(req: InvoiceRequest):
    reasons: List[str] = []
    risk_score = 0.0
    inv_hash = compute_invoice_hash(req)

    # 1. Duplicate / Double-Financing Gate
    if inv_hash in SEEN_HASHES:
        risk_score += 80.0
        reasons.append("Duplicate invoice fingerprint detected (double-financing risk)")

    # 2. Same-Entity / Self-Invoicing Check
    if req.sellerGstin.strip().upper() == req.buyerGstin.strip().upper():
        risk_score += 90.0
        reasons.append("Seller and Buyer GSTIN are identical (self-invoicing fraud)")

    # 3. Circular Trading Graph Detection (A -> B -> C -> A)
    TRADE_GRAPH.add_edge(req.sellerGstin, req.buyerGstin, amount=req.amountInr)
    if nx.has_path(TRADE_GRAPH, req.buyerGstin, req.sellerGstin):
        risk_score += 55.0
        reasons.append("Circular trading cycle detected in GSTIN transaction graph")

    # 4. ML Isolation Forest Statistical Anomaly Score
    is_round = 1.0 if (req.amountInr >= 100000 and req.amountInr % 50000 == 0) else 0.0
    features = np.array([[req.amountInr, req.paymentTermDays, req.gstRatePct, is_round]])
    raw_anomaly = anomaly_model.decision_function(features)[0]
    if raw_anomaly < 0:
        ml_penalty = min(30.0, abs(float(raw_anomaly)) * 100.0)
        risk_score += ml_penalty
        reasons.append(f"IsolationForest flagged statistical anomaly (+{ml_penalty:.1f} risk)")

    # 5. Local Qwen3 8B Semantic Invoice Audit
    qwen_risk, qwen_reason = audit_with_qwen(req)
    risk_score += qwen_risk
    reasons.append(f"Qwen3 8B Audit (+{qwen_risk:.1f} risk): {qwen_reason}")

    fraud_score = round(min(100.0, max(0.0, risk_score)), 2)
    approved = fraud_score <= FRAUD_THRESHOLD

    if approved:
        SEEN_HASHES.add(inv_hash)

    signature = sign_attestation(req.invoiceId, inv_hash, fraud_score, approved)

    attestation = AttestationPayload(
        invoiceId=req.invoiceId,
        invoiceHash=inv_hash,
        fraudScore=fraud_score,
        approved=approved,
        signature=signature,
    )

    return EvaluationResponse(
        attestation=attestation,
        qwenAuditSummary=qwen_reason,
        reasons=reasons,
        chaincodeArgs=[
            req.invoiceId,
            inv_hash,
            req.msmeId,
            req.buyerGstin,
            f"{req.amountInr:.2f}",
            req.dueDate,
            attestation.model_dump_json(),
        ],
    )
