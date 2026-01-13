import logging
from pathlib import Path
from .config import settings

def setup_logging() -> None:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(settings.logs_dir) / "sentinel_api.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
