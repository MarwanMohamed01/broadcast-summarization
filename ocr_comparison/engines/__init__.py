"""Engine registry. Each engine exposes a `run(panorama) -> EngineResult`."""
from importlib import import_module


ENGINE_MODULES = {
    "tesseract": "ocr_comparison.engines.tesseract_engine",
    "easyocr": "ocr_comparison.engines.easyocr_engine",
    "paddle": "ocr_comparison.engines.paddle_engine",
    "doctr": "ocr_comparison.engines.doctr_engine",
    "gemini": "ocr_comparison.engines.gemini_engine",
    "craft_trocr": "ocr_comparison.engines.craft_trocr_engine",
}


def load_engine(name: str):
    if name not in ENGINE_MODULES:
        raise KeyError(f"Unknown engine '{name}'. Choices: {list(ENGINE_MODULES)}")
    return import_module(ENGINE_MODULES[name])
