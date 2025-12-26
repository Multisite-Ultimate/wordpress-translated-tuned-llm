"""Evaluation report generation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger
from .evaluator import EvaluationResult, TranslationSample
from .metrics import MetricsResult

logger = get_logger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in various formats."""

    def __init__(
        self,
        output_dir: Path,
        model_name: str = "unknown",
        locale: str = "unknown",
    ):
        """Initialize the reporter.

        Args:
            output_dir: Directory for report output
            model_name: Name of the evaluated model
            locale: Target locale
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.locale = locale

    def generate_report(
        self,
        result: EvaluationResult,
        format: str = "markdown",
        include_samples: bool = True,
        num_samples: int = 20,
    ) -> Path:
        """Generate evaluation report.

        Args:
            result: Evaluation result
            format: Report format (markdown, json, html)
            include_samples: Whether to include sample translations
            num_samples: Number of samples to include

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "markdown":
            return self._generate_markdown(result, include_samples, num_samples, timestamp)
        elif format == "json":
            return self._generate_json(result, include_samples, num_samples, timestamp)
        elif format == "html":
            return self._generate_html(result, include_samples, num_samples, timestamp)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_markdown(
        self,
        result: EvaluationResult,
        include_samples: bool,
        num_samples: int,
        timestamp: str,
    ) -> Path:
        """Generate Markdown report."""
        output_path = self.output_dir / f"evaluation_report_{timestamp}.md"

        lines = [
            f"# Translation Evaluation Report",
            "",
            f"**Model:** {self.model_name}",
            f"**Locale:** {self.locale}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Samples Evaluated:** {result.num_samples}",
            "",
            "## Metrics Summary",
            "",
            "| Metric | Score | Assessment |",
            "|--------|-------|------------|",
        ]

        # Add metrics with assessments
        assessments = self._get_assessments(result.metrics)

        if result.metrics.comet is not None:
            lines.append(
                f"| COMET | {result.metrics.comet:.4f} | "
                f"{assessments.get('comet', 'N/A')} |"
            )
        if result.metrics.bleu is not None:
            lines.append(
                f"| BLEU | {result.metrics.bleu:.2f} | "
                f"{assessments.get('bleu', 'N/A')} |"
            )
        if result.metrics.chrf is not None:
            lines.append(
                f"| ChrF | {result.metrics.chrf:.4f} | "
                f"{assessments.get('chrf', 'N/A')} |"
            )

        lines.extend([
            "",
            "## Performance",
            "",
            f"- Total generation time: {result.generation_time_seconds:.2f}s",
            f"- Average time per sample: {result.avg_generation_time_per_sample:.4f}s",
            f"- Samples per second: {result.num_samples / result.generation_time_seconds:.2f}",
            "",
        ])

        if include_samples and result.samples:
            lines.extend([
                "## Sample Translations",
                "",
            ])

            # Add samples
            samples_to_show = result.samples[:num_samples]
            for i, sample in enumerate(samples_to_show, 1):
                lines.extend([
                    f"### Sample {i}",
                    "",
                    f"**Source:** {sample.source}",
                    "",
                    f"**Reference:** {sample.reference}",
                    "",
                    f"**Translation:** {sample.hypothesis}",
                    "",
                    "---",
                    "",
                ])

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved Markdown report to {output_path}")
        return output_path

    def _generate_json(
        self,
        result: EvaluationResult,
        include_samples: bool,
        num_samples: int,
        timestamp: str,
    ) -> Path:
        """Generate JSON report."""
        output_path = self.output_dir / f"evaluation_report_{timestamp}.json"

        report = {
            "model": self.model_name,
            "locale": self.locale,
            "timestamp": datetime.now().isoformat(),
            "num_samples": result.num_samples,
            "metrics": result.metrics.to_dict(),
            "assessments": self._get_assessments(result.metrics),
            "performance": {
                "total_generation_time_seconds": round(result.generation_time_seconds, 2),
                "avg_time_per_sample_seconds": round(
                    result.avg_generation_time_per_sample, 4
                ),
                "samples_per_second": round(
                    result.num_samples / result.generation_time_seconds, 2
                ),
            },
        }

        if include_samples:
            report["samples"] = [
                {
                    "source": s.source,
                    "reference": s.reference,
                    "hypothesis": s.hypothesis,
                    "project_type": s.project_type,
                }
                for s in result.samples[:num_samples]
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved JSON report to {output_path}")
        return output_path

    def _generate_html(
        self,
        result: EvaluationResult,
        include_samples: bool,
        num_samples: int,
        timestamp: str,
    ) -> Path:
        """Generate HTML report."""
        output_path = self.output_dir / f"evaluation_report_{timestamp}.html"

        assessments = self._get_assessments(result.metrics)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Translation Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .metrics-table {{ border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd; padding: 10px; text-align: left;
        }}
        .metrics-table th {{ background-color: #f5f5f5; }}
        .excellent {{ color: green; font-weight: bold; }}
        .good {{ color: #2e7d32; }}
        .acceptable {{ color: #ff9800; }}
        .needs_improvement {{ color: red; }}
        .sample {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .sample-source {{ color: #666; }}
        .sample-reference {{ color: #2196f3; }}
        .sample-hypothesis {{ color: #4caf50; }}
    </style>
</head>
<body>
    <h1>Translation Evaluation Report</h1>

    <p><strong>Model:</strong> {self.model_name}</p>
    <p><strong>Locale:</strong> {self.locale}</p>
    <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Samples Evaluated:</strong> {result.num_samples}</p>

    <h2>Metrics Summary</h2>
    <table class="metrics-table">
        <tr><th>Metric</th><th>Score</th><th>Assessment</th></tr>
"""

        if result.metrics.comet is not None:
            assessment = assessments.get('comet', 'N/A')
            html += f'        <tr><td>COMET</td><td>{result.metrics.comet:.4f}</td><td class="{assessment}">{assessment}</td></tr>\n'
        if result.metrics.bleu is not None:
            assessment = assessments.get('bleu', 'N/A')
            html += f'        <tr><td>BLEU</td><td>{result.metrics.bleu:.2f}</td><td class="{assessment}">{assessment}</td></tr>\n'
        if result.metrics.chrf is not None:
            assessment = assessments.get('chrf', 'N/A')
            html += f'        <tr><td>ChrF</td><td>{result.metrics.chrf:.4f}</td><td class="{assessment}">{assessment}</td></tr>\n'

        html += """    </table>

    <h2>Performance</h2>
    <ul>
"""
        html += f'        <li>Total generation time: {result.generation_time_seconds:.2f}s</li>\n'
        html += f'        <li>Average time per sample: {result.avg_generation_time_per_sample:.4f}s</li>\n'
        html += f'        <li>Samples per second: {result.num_samples / result.generation_time_seconds:.2f}</li>\n'
        html += "    </ul>\n"

        if include_samples and result.samples:
            html += "\n    <h2>Sample Translations</h2>\n"
            for i, sample in enumerate(result.samples[:num_samples], 1):
                html += f"""
    <div class="sample">
        <h3>Sample {i}</h3>
        <p class="sample-source"><strong>Source:</strong> {self._escape_html(sample.source)}</p>
        <p class="sample-reference"><strong>Reference:</strong> {self._escape_html(sample.reference)}</p>
        <p class="sample-hypothesis"><strong>Translation:</strong> {self._escape_html(sample.hypothesis)}</p>
    </div>
"""

        html += """
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"Saved HTML report to {output_path}")
        return output_path

    def _get_assessments(self, metrics: MetricsResult) -> dict[str, str]:
        """Get quality assessments for metrics."""
        assessments = {}

        if metrics.comet is not None:
            if metrics.comet >= 0.85:
                assessments["comet"] = "excellent"
            elif metrics.comet >= 0.75:
                assessments["comet"] = "good"
            elif metrics.comet >= 0.65:
                assessments["comet"] = "acceptable"
            else:
                assessments["comet"] = "needs_improvement"

        if metrics.bleu is not None:
            if metrics.bleu >= 40:
                assessments["bleu"] = "excellent"
            elif metrics.bleu >= 30:
                assessments["bleu"] = "good"
            elif metrics.bleu >= 20:
                assessments["bleu"] = "acceptable"
            else:
                assessments["bleu"] = "needs_improvement"

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

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def save_predictions(
        self,
        result: EvaluationResult,
        filename: Optional[str] = None,
    ) -> Path:
        """Save all predictions to JSONL file.

        Args:
            result: Evaluation result
            filename: Output filename (default: predictions.jsonl)

        Returns:
            Path to saved file
        """
        filename = filename or "predictions.jsonl"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in result.samples:
                record = {
                    "source": sample.source,
                    "reference": sample.reference,
                    "hypothesis": sample.hypothesis,
                    "project_type": sample.project_type,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(result.samples)} predictions to {output_path}")
        return output_path

    def compare_models(
        self,
        results: dict[str, EvaluationResult],
    ) -> Path:
        """Generate comparison report for multiple models.

        Args:
            results: Dictionary mapping model names to evaluation results

        Returns:
            Path to comparison report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"model_comparison_{timestamp}.md"

        lines = [
            "# Model Comparison Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Locale:** {self.locale}",
            "",
            "## Metrics Comparison",
            "",
            "| Model | COMET | BLEU | ChrF | Samples |",
            "|-------|-------|------|------|---------|",
        ]

        for model_name, result in results.items():
            m = result.metrics
            comet = f"{m.comet:.4f}" if m.comet else "N/A"
            bleu = f"{m.bleu:.2f}" if m.bleu else "N/A"
            chrf = f"{m.chrf:.4f}" if m.chrf else "N/A"
            lines.append(f"| {model_name} | {comet} | {bleu} | {chrf} | {result.num_samples} |")

        lines.extend([
            "",
            "## Performance Comparison",
            "",
            "| Model | Total Time (s) | Avg Time/Sample (s) | Samples/s |",
            "|-------|----------------|---------------------|-----------|",
        ])

        for model_name, result in results.items():
            total = f"{result.generation_time_seconds:.2f}"
            avg = f"{result.avg_generation_time_per_sample:.4f}"
            rate = f"{result.num_samples / result.generation_time_seconds:.2f}"
            lines.append(f"| {model_name} | {total} | {avg} | {rate} |")

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Saved comparison report to {output_path}")
        return output_path
