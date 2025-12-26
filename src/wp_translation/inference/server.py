"""FastAPI server for translation inference."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..utils.logging import get_logger, setup_logging
from .translator import WordPressTranslator

logger = get_logger(__name__)

# Global translator instance
_translator: Optional[WordPressTranslator] = None


class TranslationRequest(BaseModel):
    """Request model for translation endpoint."""

    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(default="en", description="Source language code")
    target_lang: str = Field(default="nl", description="Target language code")


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation endpoint."""

    texts: list[str] = Field(..., description="List of texts to translate")
    source_lang: str = Field(default="en", description="Source language code")
    target_lang: str = Field(default="nl", description="Target language code")


class TranslationResponse(BaseModel):
    """Response model for translation endpoint."""

    translation: str = Field(..., description="Translated text")
    source_lang: str
    target_lang: str


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation endpoint."""

    translations: list[str] = Field(..., description="List of translated texts")
    count: int = Field(..., description="Number of translations")
    source_lang: str
    target_lang: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool
    model_path: Optional[str] = None


def create_app(
    model_path: Optional[str | Path] = None,
    source_lang: str = "en",
    target_lang: str = "nl",
) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_path: Path to fine-tuned model (can be set via env var)
        source_lang: Default source language
        target_lang: Default target language

    Returns:
        Configured FastAPI application
    """
    setup_logging(level="INFO")

    app = FastAPI(
        title="WordPress Translation API",
        description="API for translating WordPress content using fine-tuned LLMs",
        version="1.0.0",
    )

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        global _translator

        if model_path:
            try:
                logger.info(f"Loading model from {model_path}")
                _translator = WordPressTranslator(
                    model_path=model_path,
                    source_lang=source_lang,
                    target_lang=target_lang,
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up on shutdown."""
        global _translator

        if _translator is not None:
            _translator.cleanup()
            _translator = None

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check API health and model status."""
        return HealthResponse(
            status="healthy",
            model_loaded=_translator is not None,
            model_path=str(model_path) if model_path else None,
        )

    @app.post("/translate", response_model=TranslationResponse)
    async def translate(request: TranslationRequest):
        """Translate a single text.

        Args:
            request: Translation request with text and languages

        Returns:
            Translation response
        """
        if _translator is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please configure model_path.",
            )

        try:
            translation = _translator.translate(request.text)

            return TranslationResponse(
                translation=translation,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            )

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Translation failed: {str(e)}",
            )

    @app.post("/translate/batch", response_model=BatchTranslationResponse)
    async def translate_batch(request: BatchTranslationRequest):
        """Translate multiple texts.

        Args:
            request: Batch translation request

        Returns:
            Batch translation response
        """
        if _translator is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please configure model_path.",
            )

        if len(request.texts) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 texts per batch request.",
            )

        try:
            translations = _translator.translate_batch(
                request.texts,
                show_progress=False,
            )

            return BatchTranslationResponse(
                translations=translations,
                count=len(translations),
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            )

        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Batch translation failed: {str(e)}",
            )

    @app.get("/model/info")
    async def model_info():
        """Get information about the loaded model."""
        if _translator is None:
            return {"loaded": False}

        return {
            "loaded": True,
            **_translator.get_model_info(),
        }

    return app


def run_server(
    model_path: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    source_lang: str = "en",
    target_lang: str = "nl",
    reload: bool = False,
) -> None:
    """Run the translation server.

    Args:
        model_path: Path to fine-tuned model
        host: Host to bind to
        port: Port to listen on
        source_lang: Default source language
        target_lang: Default target language
        reload: Enable auto-reload for development
    """
    import uvicorn

    # Create app with model path
    app = create_app(
        model_path=model_path,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )
