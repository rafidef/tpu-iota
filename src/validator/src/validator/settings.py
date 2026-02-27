import os

from common.settings import BITTENSOR
from dotenv import load_dotenv
from loguru import logger

DOTENV_PATH = os.getenv("DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=DOTENV_PATH):
    logger.warning("No .env file found for validator")


def detect_device() -> str:
    """Detect the most capable torch device available on the host.

    Priority order: CUDA/ROCm > XLA (TPU) > Intel XPU > MPS > CPU
    """
    try:
        import torch
    except Exception as exc:
        logger.debug(f"Unable to import torch for device detection: {exc}")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        return "xla"
    except ImportError:
        pass

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    mps_backend = getattr(torch, "backends", None)
    if mps_backend is not None:
        mps = getattr(mps_backend, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"

    return "cpu"


DEVICE = os.getenv("DEVICE") or detect_device()

# WEIGHT_SUBMIT_INTERVAL: int = 3600  # submit weight every 1 hour
WEIGHT_SUBMIT_INTERVAL: int = 60 * 21 if BITTENSOR else 10  # submit weight every 21 minutes
FETCH_TASKS_INTERVAL: int = 60 if BITTENSOR else 5  # fetch tasks every minute
ORCHESTRATOR_HEALTH_CHECK_INTERVAL: int = 60  # check orchestrator health every 1 minute

# Validation Thresholds
COSINE_SIMILARITY_THRESHOLD = 0.9
ACTIVATION_MAGNITUDE_THRESHOLD = 0.8
OPTIMIZER_SIMILARITY_THRESHOLD = 0.8

# Health settings
LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"
VALIDATOR_HEALTH_HOST = os.getenv("VALIDATOR_HEALTH_HOST", "0.0.0.0")
VALIDATOR_HEALTH_PORT = int(os.getenv("VALIDATOR_HEALTH_PORT", 9000))
VALIDATOR_HEALTH_ENDPOINT = os.getenv("VALIDATOR_HEALTH_ENDPOINT", "/health")

WALLET_NAME = os.getenv("wallet_name", "test")
WALLET_HOTKEY = os.getenv("wallet_hotkey", "m1")
