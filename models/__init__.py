"""Models module for financial forecasting"""

# Make optional submodules safe to import
__all__: list[str] = []

try:
    from .tft_model import TFTModel, TFTConfig, create_tft_model  # type: ignore
    __all__.extend(["TFTModel", "TFTConfig", "create_tft_model"])
except Exception:
    # tft_model is optional; ignore if missing
    pass
