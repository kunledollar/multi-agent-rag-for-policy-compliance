from fastapi import FastAPI, Response
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .core.logging import setup_logging
from .telemetry.otel import setup_otel
from .routes.health import router as health_router
from .routes.ask import router as ask_router

# --------------------------------------------------
# IMPORTANT:
# Import custom metrics at startup so they are
# registered in the FastAPI process registry
# --------------------------------------------------
import app.telemetry.metrics  # noqa: F401


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="Sentinel API",
        version="1.0.0",
    )

    # --------------------------------------------------
    # OpenTelemetry
    # --------------------------------------------------
    setup_otel(app)

    # --------------------------------------------------
    # API Routes
    # --------------------------------------------------
    app.include_router(health_router)
    app.include_router(ask_router, prefix="/v1")

    # --------------------------------------------------
    # Prometheus request-level instrumentation
    # --------------------------------------------------
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[
            "/metrics",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ],
    ).instrument(app)

    # --------------------------------------------------
    # Prometheus Metrics Endpoint (authoritative)
    # Exposes ALL metrics from the default registry:
    #  - Instrumentator metrics
    #  - Custom agent metrics
    #  - Python runtime metrics
    # --------------------------------------------------
    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


app = create_app()
