"""Metriques d'evaluation des extractions structurees."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class Evaluator:
    """Calcule les scores exacts, partiels et F1 par champ."""

    SIMPLE_FIELDS = config.STRING_FIELDS
    LIST_FIELDS = config.LIST_FIELDS
    LIST_MATCH_THRESHOLD = 0.5

    def normalize_string(self, s: str | None) -> str:
        """Normaliser une chaine pour comparaison."""
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ""
        normalized = str(s).lower().strip()
        replacements = {
            "longterm": "long term",
            "fulltime": "full time",
            "parttime": "part time",
            "andor": "and or",
            "thirdparty": "third party",
            "csssass": "css sass",
        }
        for source, target in replacements.items():
            normalized = re.sub(rf"\b{source}\b", target, normalized)
        normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def exact_match(self, pred: str | None, gt: str | None) -> float:
        """Retourner 1 si les chaines normalisees sont identiques."""
        pred_normalized = self.normalize_string(pred)
        gt_normalized = self.normalize_string(gt)
        if not pred_normalized and not gt_normalized:
            return 1.0
        if not pred_normalized or not gt_normalized:
            return 0.0
        return float(pred_normalized == gt_normalized)

    def partial_match(self, pred: str | None, gt: str | None) -> float:
        """Calculer une similarite de Jaccard sur les tokens."""
        pred_tokens = set(self.normalize_string(pred).split())
        gt_tokens = set(self.normalize_string(gt).split())
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        return len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)

    def list_f1(self, pred_list: list | None, gt_list: list | None) -> dict:
        """Calculer precision, rappel et F1 fuzzy pour deux listes."""
        pred_items = self._normalize_list(pred_list)
        gt_items = self._normalize_list(gt_list)

        if not pred_items and not gt_items:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not pred_items or not gt_items:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        matched_pred, matched_gt = self._match_list_items(pred_items, gt_items)
        precision = len(matched_pred) / len(pred_items)
        recall = len(matched_gt) / len(gt_items)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def evaluate_single(self, prediction: dict, ground_truth: dict) -> dict:
        """Evaluer une prediction contre la verite terrain."""
        metrics: dict[str, float] = {}
        field_scores = []
        informative_scores = []

        for field in self.SIMPLE_FIELDS:
            pred_value = prediction.get(field)
            gt_value = ground_truth.get(field)
            exact = self.exact_match(pred_value, gt_value)
            partial = self.partial_match(pred_value, gt_value)
            metrics[f"{field}_exact_match"] = exact
            metrics[f"{field}_partial_match"] = partial
            field_scores.append(partial)
            if self._is_informative_value(gt_value, field):
                informative_scores.append(partial)

        for field in self.LIST_FIELDS:
            scores = self.list_f1(prediction.get(field), ground_truth.get(field))
            metrics[f"{field}_precision"] = scores["precision"]
            metrics[f"{field}_recall"] = scores["recall"]
            metrics[f"{field}_f1"] = scores["f1"]
            field_scores.append(scores["f1"])
            if self._is_informative_value(ground_truth.get(field), field):
                informative_scores.append(scores["f1"])

        metrics["score_global"] = sum(field_scores) / len(field_scores) if field_scores else 0.0
        metrics["informative_score"] = (
            sum(informative_scores) / len(informative_scores)
            if informative_scores
            else 0.0
        )
        return metrics

    def evaluate_batch(self, predictions: list, ground_truths: list) -> pd.DataFrame:
        """Evaluer toutes les predictions disponibles contre la verite terrain."""
        prediction_map = {
            self._extract_id(item, fallback): self._extract_prediction(item)
            for fallback, item in enumerate(predictions)
        }

        rows = []
        for fallback, gt_item in enumerate(ground_truths):
            gt_id = self._extract_id(gt_item, fallback)
            gt_reference = self._extract_ground_truth(gt_item)
            prediction = prediction_map.get(gt_id, config.empty_extraction())
            metrics = self.evaluate_single(prediction, gt_reference)
            metrics["id"] = gt_id
            rows.append(metrics)

        df_eval = pd.DataFrame(rows)
        logger.info("Evaluation realisee sur %s exemples", len(df_eval))
        return df_eval

    def compute_global_metrics(self, df_eval: pd.DataFrame) -> dict:
        """Calculer les moyennes globales de toutes les metriques."""
        metrics: dict[str, Any] = {}
        if df_eval.empty:
            for field in self.SIMPLE_FIELDS:
                metrics[field] = {"exact_match": 0.0, "partial_match": 0.0}
            for field in self.LIST_FIELDS:
                metrics[field] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            metrics["overall_score"] = 0.0
            metrics["informative_score"] = 0.0
            return metrics

        for field in self.SIMPLE_FIELDS:
            metrics[field] = {
                "exact_match": float(df_eval.get(f"{field}_exact_match", pd.Series([0])).mean()),
                "partial_match": float(df_eval.get(f"{field}_partial_match", pd.Series([0])).mean()),
            }
        for field in self.LIST_FIELDS:
            metrics[field] = {
                "precision": float(df_eval.get(f"{field}_precision", pd.Series([0])).mean()),
                "recall": float(df_eval.get(f"{field}_recall", pd.Series([0])).mean()),
                "f1": float(df_eval.get(f"{field}_f1", pd.Series([0])).mean()),
            }
        metrics["overall_score"] = float(df_eval.get("score_global", pd.Series([0])).mean())
        metrics["informative_score"] = float(
            df_eval.get("informative_score", pd.Series([0])).mean()
        )
        return metrics

    def compare_strategies(self, all_results: dict, ground_truths: list) -> pd.DataFrame:
        """Comparer les trois strategies et sauvegarder la table CSV."""
        rows = []
        for strategy_name, predictions in all_results.items():
            df_eval = self.evaluate_batch(predictions, ground_truths)
            df_eval.to_csv(
                config.EVALUATIONS_DIR / f"{strategy_name}_detailed_metrics.csv",
                index=False,
                encoding="utf-8",
            )
            global_metrics = self.compute_global_metrics(df_eval)
            row = {
                "strategy": strategy_name,
                "overall_score": global_metrics["overall_score"],
                "informative_score": global_metrics["informative_score"],
            }
            for field in self.SIMPLE_FIELDS:
                row[field] = global_metrics[field]["partial_match"]
                row[f"{field}_exact_match"] = global_metrics[field]["exact_match"]
            for field in self.LIST_FIELDS:
                row[field] = global_metrics[field]["f1"]
                row[f"{field}_precision"] = global_metrics[field]["precision"]
                row[f"{field}_recall"] = global_metrics[field]["recall"]
            rows.append(row)

        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(config.COMPARISON_TABLE_PATH, index=False, encoding="utf-8")
        self._display_comparison(comparison_df)
        logger.info("Table comparative sauvegardee dans %s", config.COMPARISON_TABLE_PATH)
        return comparison_df

    def _normalize_list(self, values: list | None) -> list[str]:
        """Normaliser les elements d'une liste."""
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        normalized_items = []
        seen = set()
        for item in values:
            normalized = self.normalize_string(str(item))
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_items.append(normalized)
        return normalized_items

    def _is_informative_value(self, value: Any, field: str) -> bool:
        """Verifier si un champ de reference contient une information a extraire."""
        if field in self.LIST_FIELDS:
            return bool(self._normalize_list(value))
        return bool(self.normalize_string(value))

    def _match_list_items(self, pred_items: list[str], gt_items: list[str]) -> tuple[set[int], set[int]]:
        """Associer les elements de listes avec une similarite lexicale fuzzy."""
        candidates = []
        for pred_index, pred_item in enumerate(pred_items):
            for gt_index, gt_item in enumerate(gt_items):
                score = self._item_similarity(pred_item, gt_item)
                if score >= self.LIST_MATCH_THRESHOLD:
                    candidates.append((score, pred_index, gt_index))

        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        for _, pred_index, gt_index in sorted(candidates, reverse=True):
            if pred_index in matched_pred or gt_index in matched_gt:
                continue
            matched_pred.add(pred_index)
            matched_gt.add(gt_index)
        return matched_pred, matched_gt

    def _item_similarity(self, left: str, right: str) -> float:
        """Mesurer la proximite entre deux elements de liste normalises."""
        if left == right:
            return 1.0
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens and not right_tokens:
            return 1.0
        if not left_tokens or not right_tokens:
            return 0.0

        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        jaccard = intersection / union if union else 0.0
        min_size = min(len(left_tokens), len(right_tokens))
        containment = intersection / min_size if min_size >= 2 else 0.0
        return max(jaccard, containment)

    def _extract_id(self, item: Any, fallback: int) -> int:
        """Recuperer un identifiant numerique stable."""
        if isinstance(item, dict):
            try:
                return int(item.get("id", fallback))
            except (TypeError, ValueError):
                return fallback
        return fallback

    def _extract_prediction(self, item: Any) -> dict:
        """Extraire l'objet prediction depuis une entree de resultats."""
        if isinstance(item, dict) and isinstance(item.get("extraction"), dict):
            return item["extraction"]
        return item if isinstance(item, dict) else config.empty_extraction()

    def _extract_ground_truth(self, item: Any) -> dict:
        """Extraire l'objet de reference depuis une entree de verite terrain."""
        if isinstance(item, dict) and isinstance(item.get("ground_truth"), dict):
            return item["ground_truth"]
        return item if isinstance(item, dict) else config.empty_extraction()

    def _display_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Afficher un resume comparatif dans le terminal."""
        table = Table(title="Comparaison des strategies")
        table.add_column("Strategie", style="cyan")
        table.add_column("Score global", style="green", justify="right")
        table.add_column("Score informatif", style="yellow", justify="right")
        for _, row in comparison_df.iterrows():
            table.add_row(
                str(row.get("strategy")),
                f"{row.get('overall_score', 0.0):.3f}",
                f"{row.get('informative_score', row.get('overall_score', 0.0)):.3f}",
            )
        console.print(table)


if __name__ == "__main__":
    evaluator = Evaluator()
    predictions = [
        {
            "id": 0,
            "extraction": {
                **config.empty_extraction(),
                "titre_poste": "Data Analyst",
                "competences_techniques": ["Python", "SQL"],
            },
        }
    ]
    ground_truth = [
        {
            "id": 0,
            "ground_truth": {
                **config.empty_extraction(),
                "titre_poste": "Data Analyst",
                "competences_techniques": ["Python", "Excel"],
            },
        }
    ]
    comparison = evaluator.compare_strategies({"zero_shot": predictions}, ground_truth)
    console.print_json(json.dumps(comparison.to_dict(orient="records"), ensure_ascii=False))
