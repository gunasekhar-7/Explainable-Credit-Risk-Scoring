"""FastAPI application: /, /health, /ready, /predict, /predict/batch, /model-info."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from src.api.auth import (
    API_KEY_HEADER,
    AUTH_MODE_OPEN,
    auth_mode,
    enforce_budget,
    require_api_key,
)
from src.api.predict import InferenceEngine
from src.api.schema import (
    BatchPredictionResponse,
    CreditApplication,
    ModelInfo,
    PredictionResponse,
    ServiceInfo,
)
from src.config import (
    PROCESSED_DIR,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_ROWS_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "Explainable Credit Risk Scoring"

# Each row costs a few hundred microseconds, so an unbounded list no longer ties
# up a worker for minutes, but it still lets one caller monopolise the process.
MAX_BATCH_SIZE = 500

# Request body size limit: prevent unbounded POST bodies from tying up
# the event loop or exceeding server defaults. 100 KB is ample for any
# applicant record (19 fields x ~5KB each) and protects against abuse.
MAX_REQUEST_BODY_BYTES = 100 * 1024

_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the inference engine at startup; fail fast if no artifacts exist."""
    global _engine
    _engine = InferenceEngine().load()
    if auth_mode() == AUTH_MODE_OPEN:
        logger.warning(
            "No API_KEYS configured: scoring endpoints are unauthenticated. "
            "Set API_KEYS=<key>[,<key>...] before exposing this service beyond loopback."
        )
    yield
    _engine = None


app = FastAPI(
    title=SERVICE_NAME,
    version="1.0.0",
    description=(
        "Credit default risk scoring with per-applicant reason codes.\n\n"
        "`POST /predict` returns a default probability, a risk band, and the "
        "features that moved that applicant's score — each with the bin it fell "
        "into — plus a `data_quality` block saying how much of the record was "
        "actually supplied. `POST /predict/batch` scores up to "
        f"{MAX_BATCH_SIZE} applicants in one call. `GET /model-info` reports the "
        "champion, its challenger, and which request fields can change a score.\n\n"
        "**Authentication:** send `X-API-Key` when the service is configured with "
        "`API_KEYS`. `GET /` reports the mode it is actually running in."
    ),
    lifespan=lifespan,
)


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine().load()
    return _engine


def enforce_body_size(request: Request) -> None:
    """Validate request body size before processing.

    Raises 413 if the body exceeds MAX_REQUEST_BODY_BYTES.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        size = int(content_length)
        if size > MAX_REQUEST_BODY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request body exceeds maximum size of {MAX_REQUEST_BODY_BYTES // 1024} KB",
            )
    return None


@app.get(
    "/",
    response_model=ServiceInfo,
    summary="Service index",
    description="What this service is, which auth mode it runs in, and where to go next.",
)
def root() -> ServiceInfo:
    return ServiceInfo(
        service=SERVICE_NAME,
        version=app.version,
        docs_url="/docs",
        champion=getattr(_engine, "champion", None),
        auth_mode=auth_mode(),
        auth_header=API_KEY_HEADER,
        rate_limits={
            "requests_per_minute": RATE_LIMIT_REQUESTS_PER_MINUTE,
            "rows_per_minute": RATE_LIMIT_ROWS_PER_MINUTE,
            "window_seconds": int(RATE_LIMIT_WINDOW_SECONDS),
            "scope": "per-process",
        },
        endpoints={
            "/predict": "POST one applicant (costs 1 row of budget)",
            "/predict/batch": f"POST up to {MAX_BATCH_SIZE} applicants (costs one row each)",
            "/model-info": "GET champion, challenger, metrics, and field influence",
            "/health": "GET liveness (no model dependency, no auth)",
            "/ready": "GET readiness: 503 until the champion is loaded",
            "/docs": "interactive documentation",
        },
    )


@app.get(
    "/health",
    summary="Liveness",
    description="Returns 200 whenever the process is up. Requires no model and no API key.",
)
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/ready",
    summary="Readiness",
    description="Returns 200 once the champion is loaded, 503 before that. No API key required.",
)
def ready() -> dict:
    if _engine is None or not getattr(_engine, "_loaded", False):
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score one applicant",
    description=(
        "Returns the default probability, risk band, scorecard points and the "
        "reason codes for one applicant. Every field is optional; the "
        "`data_quality` block reports which ones were imputed, so a score can "
        "never look better-founded than the record it came from."
    ),
)
def predict(
    request: Request,
    application: CreditApplication,
    identity: str = Depends(require_api_key),
) -> PredictionResponse:
    enforce_body_size(request)
    enforce_budget(identity, rows=1)
    try:
        engine = get_engine()
        result = engine.predict_one(application.model_dump())
        return PredictionResponse(**result)
    except RuntimeError as exc:
        # Not ready (missing artifacts / failed load) -> service unavailable.
        logger.exception("Prediction unavailable")
        raise HTTPException(status_code=503, detail="model unavailable") from exc
    except Exception:  # pragma: no cover - defensive
        # Detail stays in the log: exception text can carry paths and internals.
        logger.exception("Unhandled inference failure")
        raise HTTPException(status_code=500, detail="internal prediction error") from None


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a batch of applicants",
    description=(
        f"Scores up to {MAX_BATCH_SIZE} applicants in one vectorized pass. "
        "Predictions come back in the same order as the request, and each "
        "carries its own reason codes and `data_quality` block."
    ),
)
def predict_batch(
    request: Request,
    applications: list[CreditApplication],
    identity: str = Depends(require_api_key),
) -> BatchPredictionResponse:
    enforce_body_size(request)
    if len(applications) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"batch size {len(applications)} exceeds maximum of {MAX_BATCH_SIZE}",
        )
    enforce_budget(identity, rows=max(len(applications), 1))
    try:
        engine = get_engine()
    except RuntimeError as exc:
        logger.exception("Batch prediction unavailable")
        raise HTTPException(status_code=503, detail="model unavailable") from exc
    try:
        results = engine.predict_many([app.model_dump() for app in applications])
    except RuntimeError as exc:
        logger.exception("Batch prediction unavailable")
        raise HTTPException(status_code=503, detail="model unavailable") from exc
    except Exception:  # pragma: no cover - defensive
        logger.exception("Unhandled batch inference failure")
        raise HTTPException(status_code=500, detail="internal prediction error") from None
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results]
    )


def _manifest() -> dict:
    """The champion manifest written by scripts/train.py, or {} if absent."""
    manifest_path = PROCESSED_DIR / "champion.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _metrics_for(model: str, manifest: dict | None = None) -> dict:
    """Held-out AUC/KS for one of the trained candidates."""
    manifest = _manifest() if manifest is None else manifest
    prefix = "xgb" if model == "xgboost" else "scorecard"
    metrics = {}
    if manifest.get(f"{prefix}_auc") is not None:
        metrics["auc"] = manifest[f"{prefix}_auc"]
    if manifest.get(f"{prefix}_ks") is not None:
        metrics["ks"] = manifest[f"{prefix}_ks"]
    return metrics


def _champion_metrics() -> dict:
    """Held-out metrics of the deployed champion, from data/processed/champion.json."""
    manifest = _manifest()
    if not manifest:
        return {}
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return metrics
    champion = manifest.get("champion")
    if champion in ("scorecard", "xgboost"):
        return _metrics_for(champion, manifest)
    return {}


@app.get(
    "/model-info",
    response_model=ModelInfo,
    summary="What is scoring, and what it can use",
    description=(
        "Reports the deployed champion and its held-out metrics, the challenger "
        "it beat, and the split between request fields that can move a score "
        "(`influencing_fields`) and those accepted but not modelled "
        "(`ignored_fields`)."
    ),
)
def model_info(identity: str = Depends(require_api_key)) -> ModelInfo:
    enforce_budget(identity, rows=0)
    try:
        engine = get_engine()
    except RuntimeError as exc:
        # Constant detail: the exception can name server paths.
        logger.exception("Model info unavailable")
        raise HTTPException(status_code=503, detail="model unavailable") from exc

    features = engine.model_features()
    influencing, ignored = engine.field_influence()
    manifest = _manifest()
    challenger = "xgboost" if engine.champion == "scorecard" else "scorecard"
    return ModelInfo(
        model_type=engine.champion,
        model_version=getattr(engine, "model_version", None),
        features=features,
        metrics=_champion_metrics(),
        challenger=challenger,
        challenger_metrics=_metrics_for(challenger, manifest),
        influencing_fields=influencing,
        ignored_fields=ignored,
        auth_mode=auth_mode(),
    )
