from pydantic import BaseModel
from pathlib import Path
import os

class Settings(BaseModel):
    env: str = os.getenv("ENV", "development")
    project_name: str = os.getenv("PROJECT_NAME", "Sentinel")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    data_dir: Path = Path(os.getenv("DATA_DIR", "/app/data"))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", "/app/logs"))

    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "sentinel-api")
    otel_exporter_otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")

settings = Settings()
