"""Setup quality filter for WFO trades.

Removes garbage setups before dataset ingestion. Supports:
- Basic confluence/feature checks
- CSV export for manual review
- Claude API grading (optional)
"""
import csv
import logging

from ml.config import get_config

logger = logging.getLogger(__name__)


class SetupQualityFilter:
    """Filter WFO trade setups by quality criteria."""

    def __init__(self, config: dict = None):
        self.cfg = config or get_config()
        self.min_confluence = self.cfg.get("wfo_min_confluence_score", 2)

    def filter_basic(self, trades: list[dict]) -> list[dict]:
        """Remove low-quality setups.

        Removes:
        - confluence_score below minimum
        - missing required feature columns
        - outcome == "expired" (no clear signal)
        - duplicate candle_index + direction pairs
        """
        required_features = {"ob_count", "fvg_count", "direction"}
        seen = set()
        result = []

        for trade in trades:
            # Check confluence
            if trade.get("confluence_score", 0) < self.min_confluence:
                continue

            # Check required features exist
            if not all(trade.get(f) is not None for f in required_features):
                continue

            # Skip expired (no clear outcome)
            if trade.get("outcome") == "expired":
                continue

            # Deduplicate by candle_index + direction
            key = (trade.get("candle_index"), trade.get("direction"))
            if key in seen:
                continue
            seen.add(key)

            result.append(trade)

        return result

    def export_for_review(self, trades: list[dict], path: str) -> None:
        """Export trades to CSV for manual review."""
        if not trades:
            return

        # Collect all keys across trades
        all_keys = set()
        for t in trades:
            all_keys.update(t.keys())
        fieldnames = sorted(all_keys)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for trade in trades:
                writer.writerow(trade)

    def grade_with_claude(self, trades: list[dict], api_key: str) -> list[dict]:
        """Grade setups using Claude API. Returns trades with quality tag.

        Quality levels: valid, marginal, false_positive
        false_positive trades are removed; marginal get lower weight.
        """
        try:
            import anthropic
        except ImportError:
            logger.warning("anthropic package not installed, skipping Claude grading")
            for t in trades:
                t["quality"] = "valid"
            return trades

        client = anthropic.Anthropic(api_key=api_key)
        graded = []

        # Batch in groups of 10
        batch_size = 10
        for i in range(0, len(trades), batch_size):
            batch = trades[i:i + batch_size]
            batch_desc = self._format_batch_for_grading(batch)

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Grade each ICT trade setup as 'valid', 'marginal', or 'false_positive'. "
                            "A valid setup has strong confluence (OB+FVG alignment, proper market structure). "
                            "Marginal has some confluence but weak. False_positive has no real ICT basis. "
                            "Reply with one grade per line, just the word.\n\n"
                            f"{batch_desc}"
                        ),
                    }],
                )
                grades = response.content[0].text.strip().split("\n")
                for j, trade in enumerate(batch):
                    grade = grades[j].strip().lower() if j < len(grades) else "valid"
                    if grade not in ("valid", "marginal", "false_positive"):
                        grade = "valid"
                    trade["quality"] = grade
                    if grade != "false_positive":
                        graded.append(trade)
            except Exception as e:
                logger.warning(f"Claude grading failed for batch {i}: {e}")
                for trade in batch:
                    trade["quality"] = "valid"
                    graded.append(trade)

        return graded

    def _format_batch_for_grading(self, batch: list[dict]) -> str:
        """Format a batch of trades for Claude grading prompt."""
        lines = []
        for i, t in enumerate(batch):
            lines.append(
                f"Setup {i+1}: direction={t.get('direction')}, "
                f"confluence={t.get('confluence_score',0)}, "
                f"ob_count={t.get('ob_count',0)}, "
                f"fvg_count={t.get('fvg_count',0)}, "
                f"ob_alignment={t.get('ob_alignment',0)}, "
                f"fvg_alignment={t.get('fvg_alignment',0)}, "
                f"market_structure={t.get('market_structure_score',0):.2f}, "
                f"outcome={t.get('outcome')}"
            )
        return "\n".join(lines)
