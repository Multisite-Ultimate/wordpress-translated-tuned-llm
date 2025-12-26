"""Translation evaluation metrics."""

from dataclasses import dataclass
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """Result of translation metrics computation."""

    comet: Optional[float] = None
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    num_samples: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "comet": round(self.comet, 4) if self.comet else None,
            "bleu": round(self.bleu, 2) if self.bleu else None,
            "chrf": round(self.chrf, 4) if self.chrf else None,
            "num_samples": self.num_samples,
        }

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.comet is not None:
            parts.append(f"COMET: {self.comet:.4f}")
        if self.bleu is not None:
            parts.append(f"BLEU: {self.bleu:.2f}")
        if self.chrf is not None:
            parts.append(f"ChrF: {self.chrf:.4f}")
        return ", ".join(parts) if parts else "No metrics computed"


class TranslationMetrics:
    """Compute translation quality metrics.

    Supports COMET (neural), BLEU, and ChrF metrics.
    """

    def __init__(
        self,
        comet_model: str = "Unbabel/wmt22-comet-da",
        use_gpu: bool = True,
        comet_batch_size: int = 8,
    ):
        """Initialize metrics computer.

        Args:
            comet_model: COMET model to use
            use_gpu: Whether to use GPU for COMET
            comet_batch_size: Batch size for COMET inference
        """
        self.comet_model_name = comet_model
        self.use_gpu = use_gpu
        self.comet_batch_size = comet_batch_size

        self._comet_model = None
        self._bleu = None
        self._chrf = None

    def _load_comet(self):
        """Lazy load COMET model."""
        if self._comet_model is None:
            try:
                from comet import download_model, load_from_checkpoint

                logger.info(f"Loading COMET model: {self.comet_model_name}")
                model_path = download_model(self.comet_model_name)
                self._comet_model = load_from_checkpoint(model_path)

                if self.use_gpu:
                    import torch
                    if torch.cuda.is_available():
                        self._comet_model = self._comet_model.cuda()
                        logger.info("COMET model loaded on GPU")
                    else:
                        logger.warning("GPU requested but not available")
            except ImportError:
                logger.warning("COMET not installed, skipping COMET metric")
            except Exception as e:
                logger.error(f"Error loading COMET: {e}")

    def _load_bleu(self):
        """Lazy load BLEU metric."""
        if self._bleu is None:
            try:
                from sacrebleu.metrics import BLEU
                self._bleu = BLEU()
            except ImportError:
                logger.warning("sacrebleu not installed, skipping BLEU metric")

    def _load_chrf(self):
        """Lazy load ChrF metric."""
        if self._chrf is None:
            try:
                from sacrebleu.metrics import CHRF
                self._chrf = CHRF()
            except ImportError:
                logger.warning("sacrebleu not installed, skipping ChrF metric")

    def compute_comet(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> Optional[float]:
        """Compute COMET score.

        Args:
            sources: Source texts
            hypotheses: Model-generated translations
            references: Reference translations

        Returns:
            COMET score (0-1 scale) or None if unavailable
        """
        self._load_comet()

        if self._comet_model is None:
            return None

        try:
            data = [
                {"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)
            ]

            output = self._comet_model.predict(
                data,
                batch_size=self.comet_batch_size,
                gpus=1 if self.use_gpu else 0,
            )

            return output.system_score

        except Exception as e:
            logger.error(f"Error computing COMET: {e}")
            return None

    def compute_bleu(
        self,
        hypotheses: list[str],
        references: list[list[str]],
    ) -> Optional[float]:
        """Compute BLEU score.

        Args:
            hypotheses: Model-generated translations
            references: List of reference translations (each sample can have multiple refs)

        Returns:
            BLEU score (0-100 scale) or None if unavailable
        """
        self._load_bleu()

        if self._bleu is None:
            return None

        try:
            result = self._bleu.corpus_score(hypotheses, references)
            return result.score

        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return None

    def compute_chrf(
        self,
        hypotheses: list[str],
        references: list[list[str]],
    ) -> Optional[float]:
        """Compute ChrF score.

        Args:
            hypotheses: Model-generated translations
            references: List of reference translations

        Returns:
            ChrF score (0-1 scale) or None if unavailable
        """
        self._load_chrf()

        if self._chrf is None:
            return None

        try:
            result = self._chrf.corpus_score(hypotheses, references)
            return result.score / 100  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error computing ChrF: {e}")
            return None

    def compute_all(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
    ) -> MetricsResult:
        """Compute all available metrics.

        Args:
            sources: Source texts
            hypotheses: Model-generated translations
            references: Reference translations

        Returns:
            MetricsResult with all computed metrics
        """
        # Prepare references in list format for BLEU/ChrF
        refs_list = [[r] for r in references]

        return MetricsResult(
            comet=self.compute_comet(sources, hypotheses, references),
            bleu=self.compute_bleu(hypotheses, refs_list),
            chrf=self.compute_chrf(hypotheses, refs_list),
            num_samples=len(hypotheses),
        )

    def assess_quality(
        self,
        metrics: MetricsResult,
    ) -> dict[str, str]:
        """Assess translation quality based on thresholds.

        Args:
            metrics: Computed metrics

        Returns:
            Dictionary with quality assessments
        """
        assessments = {}

        # COMET thresholds
        if metrics.comet is not None:
            if metrics.comet >= 0.85:
                assessments["comet"] = "excellent"
            elif metrics.comet >= 0.75:
                assessments["comet"] = "good"
            elif metrics.comet >= 0.65:
                assessments["comet"] = "acceptable"
            else:
                assessments["comet"] = "needs_improvement"

        # BLEU thresholds
        if metrics.bleu is not None:
            if metrics.bleu >= 40:
                assessments["bleu"] = "excellent"
            elif metrics.bleu >= 30:
                assessments["bleu"] = "good"
            elif metrics.bleu >= 20:
                assessments["bleu"] = "acceptable"
            else:
                assessments["bleu"] = "needs_improvement"

        # ChrF thresholds
        if metrics.chrf is not None:
            if metrics.chrf >= 0.75:
                assessments["chrf"] = "excellent"
            elif metrics.chrf >= 0.65:
                assessments["chrf"] = "good"
            elif metrics.chrf >= 0.55:
                assessments["chrf"] = "acceptable"
            else:
                assessments["chrf"] = "needs_improvement"

        return assessments

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._comet_model is not None:
            del self._comet_model
            self._comet_model = None

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
