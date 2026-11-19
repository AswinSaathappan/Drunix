import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import httpx
import networkx as nx
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.ensemble import IsolationForest

try:
    import pytesseract
except ImportError:
    pytesseract = None

app = FastAPI(title="LiquiTrade Deterministic Velocity & Hybrid LLM Risk Gate", version="4.0.0")

KEY_DIR = Path(__file__).parent / "keys"
PRIV_KEY_PATH = KEY_DIR / "ai_private.pem"
PUB_KEY_PATH = KEY_DIR / "ai_public.pem"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3:8b")


def load_or_generate_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if PRIV_KEY_PATH.exists() and PUB_KEY_PATH.exists():
        private_key = serialization.load_pem_private_key(PRIV_KEY_PATH.read_bytes(), password=None)
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

MSME_6M_AVG_BASELINE: Dict[str, float] = {
    "33AABCU9603R1ZM": 4500000.0,  # ₹45L avg -> ₹50L is +11.1% (TIER_A)
    "29AAACR5055K1Z7": 1200000.0,  # ₹12L avg
    "27AAACA8567B1Z2": 400000.0,   # ₹4L avg -> ₹9.5L is +137.5% (QUARANTINED)
}

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


class CloudLLMConfig(BaseModel):
    apiKey: str = ""
    baseUrl: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    preferCloud: bool = False


class InvoiceRequest(BaseModel):
    invoiceId: str
    msmeId: str = "MSME-TN-01"
    buyerId: str = "BUYER-CORP-01"
    sellerGstin: str
    buyerGstin: str
    amountInr: float = Field(..., gt=0)
    sixMonthAvgInr: Optional[float] = None
    gstRatePct: float = 18.0
    tenorDays: int = 90
    annualDiscountPct: float = 12.0
    dueDate: str
    irn: str
    lineItemDescription: str = "Precision CNC automotive shafts - 2,000 units (HSN 8708)"
    llmConfig: Optional[CloudLLMConfig] = None


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
    llmProviderUsed: str
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
    approved_str = "true" if approved else "false"
    canonical_msg = (
        f"{invoice_id}|{invoice_hash}|{risk_tier}|{velocity_delta_pct:.2f}|{fraud_score:.2f}|{approved_str}"
    ).encode("utf-8")
    der_sig = PRIVATE_KEY.sign(canonical_msg, ec.ECDSA(hashes.SHA256()))
    return base64.b64encode(der_sig).decode("utf-8")


def _try_local_ollama(prompt: str) -> tuple[dict, str]:
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
        timeout=12.0,
    )
    resp.raise_for_status()
    return json.loads(resp.json().get("response", "{}")), f"local-ollama ({QWEN_MODEL})"


def _try_cloud_llm(prompt: str, cfg: CloudLLMConfig) -> tuple[dict, str]:
    base_url = cfg.baseUrl.rstrip("/")
    cloud_resp = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.apiKey.strip()}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg.model.strip(),
            "messages": [
                {"role": "system", "content": "Respond strictly with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        },
        timeout=15.0,
    )
    cloud_resp.raise_for_status()
    content = cloud_resp.json()["choices"][0]["message"]["content"]
    return json.loads(content), f"cloud-api ({cfg.model})"


def call_llm_json(prompt: str, cfg: Optional[CloudLLMConfig] = None) -> tuple[dict, str]:
    has_cloud = bool(cfg and cfg.apiKey and cfg.apiKey.strip())

    # If user explicitly toggled "Prefer Cloud API" in frontend UI, try cloud first
    if has_cloud and cfg.preferCloud:
        try:
            return _try_cloud_llm(prompt, cfg)
        except Exception:
            return _try_local_ollama(prompt)

    # Default: Local Qwen3 8B first -> Cloud API Key fallback
    try:
        return _try_local_ollama(prompt)
    except Exception as local_err:
        if has_cloud:
            return _try_cloud_llm(prompt, cfg)
        raise RuntimeError(f"Local Ollama unavailable ({local_err}) and no frontend API key provided")


def explain_with_llm(
    req: InvoiceRequest, tier: str, velocity_delta: float, flags: List[str]
) -> tuple[str, str]:
    prompt = f"""You are an MSME trade-finance risk explainer for NPCI Drunix LiquiTrade.
A deterministic velocity engine has already classified this invoice:
- Invoice ID: {req.invoiceId}
- Amount: INR {req.amountInr:,.2f} (Tenor: {req.tenorDays} days)
- 6-Month Velocity Delta: {velocity_delta:+.1f}%
- Deterministic Risk Tier: {tier}
- Line Items: "{req.lineItemDescription}"
- Triggered Rules: {flags}

Provide a clear, concise 1-to-2 sentence audit explanation for the MSME and institutional LP dashboards.
Respond ONLY in JSON: {{"explanation": "..."}}"""
    try:
        parsed, provider = call_llm_json(prompt, req.llmConfig)
        return str(parsed.get("explanation", "")), provider
    except Exception:
        summary = (
            f"Invoice {req.invoiceId} (INR {req.amountInr:,.2f}) was deterministically classified as {tier} "
            f"with a {velocity_delta:+.1f}% velocity delta against the 6-month MSME baseline. "
            f"Note: {flags[0] if flags else 'Standard B2B parameters verified'}."
        )
        return summary, "deterministic-offline-fallback"


@app.get("/public-key")
def get_public_key():
    return {"publicKeyPem": PUBLIC_KEY_PEM}


@app.post("/ocr-extract")
async def ocr_extract_document(
    file: UploadFile = File(...),
    apiKey: str = Form(""),
    baseUrl: str = Form("https://api.groq.com/openai/v1"),
    model: str = Form("llama-3.3-70b-versatile"),
    preferCloud: bool = Form(False),
):
    """Extracts raw text via PDF parser or Tesseract OCR, then structures e-invoice fields via Qwen3 / Cloud LLM."""
    content = await file.read()
    filename = (file.filename or "").lower()
    raw_text = ""

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    elif filename.endswith((".txt", ".json", ".csv")):
        raw_text = content.decode("utf-8", errors="ignore").strip()
    else:
        # Image OCR (PNG, JPG, WEBP, etc.)
        if pytesseract is not None:
            img = Image.open(io.BytesIO(content))
            raw_text = pytesseract.image_to_string(img).strip()

    if not raw_text:
        raw_text = f"Uploaded file: {file.filename} (No embedded text detected; using defaults)"

    cfg = CloudLLMConfig(apiKey=apiKey, baseUrl=baseUrl, model=model, preferCloud=preferCloud)
    prompt = f"""Extract structured Indian GST e-invoice fields from the OCR text below into valid JSON with exact keys:
invoiceId (string), sellerGstin (15-char GSTIN string), buyerGstin (15-char GSTIN string),
amountInr (number), tenorDays (integer, default 90), dueDate (YYYY-MM-DD), irn (string), lineItemDescription (string).

OCR Text:
\"\"\"{raw_text}\"\"\""""

    try:
        structured, provider = call_llm_json(prompt, cfg)
    except Exception:
        gstins = re.findall(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", raw_text)
        inv_match = re.search(r"INV-[A-Za-z0-9-]+", raw_text)
        amt_match = re.search(r"(?:INR|Rs\.?|₹)\s*([0-9,]+(?:\.[0-9]+)?)", raw_text)
        irn_match = re.search(r"IRN[A-Za-z0-9]+", raw_text)
        provider = "deterministic-regex-ocr-fallback"
        structured = {
            "invoiceId": inv_match.group(0) if inv_match else f"INV-OCR-{os.urandom(2).hex().upper()}",
            "sellerGstin": gstins[0] if len(gstins) > 0 else "33AABCU9603R1ZM",
            "buyerGstin": gstins[1] if len(gstins) > 1 else "29AAACR5055K1Z7",
            "amountInr": float(amt_match.group(1).replace(",", "")) if amt_match else 5000000.0,
            "tenorDays": 90,
            "dueDate": "2026-12-30",
            "irn": irn_match.group(0) if irn_match else f"IRN{os.urandom(3).hex().upper()}",
            "lineItemDescription": raw_text[:160].replace("\n", " "),
        }

    return {
        "filename": file.filename,
        "ocrRawText": raw_text,
        "llmProviderUsed": provider,
        "structuredInvoice": structured,
    }


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_receivable(req: InvoiceRequest):
    reasons: List[str] = []
    inv_hash = compute_invoice_hash(req)

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

    is_round = 1.0 if (req.amountInr >= 100000 and req.amountInr % 50000 == 0) else 0.0
    features = np.array([[req.amountInr, req.tenorDays, req.gstRatePct, is_round]])
    if anomaly_model.decision_function(features)[0] < 0 and tier != "QUARANTINED":
        risk_score = min(65.0, risk_score + 12.0)
        reasons.append("IsolationForest flagged minor statistical variance")

    fraud_score = round(min(100.0, max(0.0, risk_score)), 2)
    approved = tier in ("TIER_A", "TIER_B")

    qwen_explanation, llm_provider = explain_with_llm(req, tier, velocity_delta, reasons)

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
        llmProviderUsed=llm_provider,
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
