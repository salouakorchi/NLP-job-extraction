"""Analyse qualitative et quantitative des erreurs d'extraction."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.evaluator import Evaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class ErrorAnalyzer:
    """Categorise les erreurs et produit des rapports interpretables."""

    def __init__(self) -> None:
        """Initialiser l'evaluateur utilise pour les comparaisons."""
        self.evaluator = Evaluator()

    def categorize_error(self, pred_val: Any, gt_val: Any, field_name: str) -> str:
        """Classer l'ecart entre une prediction et la verite terrain."""
        if field_name in config.LIST_FIELDS:
            if pred_val is not None and not isinstance(pred_val, list):
                return "format_incorrect"
            pred_items = set(self.evaluator._normalize_list(pred_val))
            gt_items = set(self.evaluator._normalize_list(gt_val))
            if not pred_items and not gt_items:
                return "correct"
            if not pred_items and gt_items:
                return "champ_manquant"
            if pred_items and not gt_items:
                return "information_hallucinee"
            scores = self.evaluator.list_f1(pred_val, gt_val)
            if scores["f1"] == 1.0:
                return "correct"
            if scores["f1"] > 0:
                return "extraction_partielle"
            return "valeur_incorrecte"

        if isinstance(pred_val, (list, dict)):
            return "format_incorrect"

        pred_empty = self._is_empty(pred_val)
        gt_empty = self._is_empty(gt_val)
        if pred_empty and not gt_empty:
            return "champ_manquant"
        if not pred_empty and gt_empty:
            return "information_hallucinee"
        if self.evaluator.exact_match(pred_val, gt_val) == 1.0:
            return "correct"
        if self.evaluator.partial_match(pred_val, gt_val) > 0:
            return "extraction_partielle"
        return "valeur_incorrecte"

    def analyze_errors(self, predictions: list, ground_truths: list) -> dict:
        """Analyser les erreurs par champ, categorie et exemple."""
        prediction_map = {
            self.evaluator._extract_id(item, fallback): self.evaluator._extract_prediction(item)
            for fallback, item in enumerate(predictions)
        }
        error_by_field: dict[str, Counter] = defaultdict(Counter)
        category_counts: Counter = Counter()
        hard_examples = []

        for fallback, gt_item in enumerate(ground_truths):
            gt_id = self.evaluator._extract_id(gt_item, fallback)
            gt_reference = self.evaluator._extract_ground_truth(gt_item)
            prediction = prediction_map.get(gt_id, config.empty_extraction())
            example_errors = Counter()

            for field in config.ALL_FIELDS:
                category = self.categorize_error(
                    prediction.get(field),
                    gt_reference.get(field),
                    field,
                )
                error_by_field[field][category] += 1
                category_counts[category] += 1
                if category != "correct":
                    example_errors[category] += 1

            hard_examples.append(
                {
                    "id": gt_id,
                    "nb_erreurs": int(sum(example_errors.values())),
                    "categories": dict(example_errors),
                }
            )

        non_correct_by_field = {
            field: sum(count for category, count in counts.items() if category != "correct")
            for field, counts in error_by_field.items()
        }
        top_3_fields = [
            item
            for item in sorted(
                non_correct_by_field.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if item[1] > 0
        ][:3]
        hardest = [
            item
            for item in sorted(hard_examples, key=lambda item: item["nb_erreurs"], reverse=True)
            if item["nb_erreurs"] > 0
        ][:5]

        analysis = {
            "nombre_exemples": len(ground_truths),
            "erreurs_par_champ": {
                field: dict(counter) for field, counter in error_by_field.items()
            },
            "distribution_categories": dict(category_counts),
            "top_3_champs_erreurs": [
                {"champ": field, "nb_erreurs": int(count)} for field, count in top_3_fields
            ],
            "types_erreurs_frequents": category_counts.most_common(),
            "exemples_difficiles": hardest,
        }
        self._display_analysis(analysis)
        return analysis

    def compare_strategy_errors(self, all_results: dict, ground_truths: list) -> dict:
        """Comparer les profils d'erreurs des strategies."""
        return {
            strategy_name: self.analyze_errors(predictions, ground_truths)
            for strategy_name, predictions in all_results.items()
        }

    def generate_error_report(self, analysis: dict, strategy_name: str) -> str:
        """Generer et sauvegarder un rapport Markdown d'analyse des erreurs."""
        lines = [
            f"# Rapport d'erreurs - {strategy_name}",
            "",
            "## Tableau des erreurs par champ",
            "",
            "| Champ | Correct | Champ manquant | Valeur incorrecte | Extraction partielle | Information hallucinee | Format incorrect |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        for field in config.ALL_FIELDS:
            counts = analysis.get("erreurs_par_champ", {}).get(field, {})
            lines.append(
                "| {field} | {correct} | {missing} | {wrong} | {partial} | {hallucinated} | {format_bad} |".format(
                    field=field,
                    correct=counts.get("correct", 0),
                    missing=counts.get("champ_manquant", 0),
                    wrong=counts.get("valeur_incorrecte", 0),
                    partial=counts.get("extraction_partielle", 0),
                    hallucinated=counts.get("information_hallucinee", 0),
                    format_bad=counts.get("format_incorrect", 0),
                )
            )

        lines.extend(
            [
                "",
                "## Distribution des categories d'erreurs",
                "",
                "| Categorie | Nombre |",
                "|---|---:|",
            ]
        )
        for category, count in analysis.get("distribution_categories", {}).items():
            lines.append(f"| {category} | {count} |")

        lines.extend(["", "## Top 5 exemples les plus difficiles", ""])
        hard_examples = analysis.get("exemples_difficiles", [])
        if not hard_examples:
            lines.append("Aucun exemple difficile detecte sur le jeu d'evaluation actuel.")
        for example in hard_examples:
            lines.append(
                f"- ID {example['id']} : {example['nb_erreurs']} erreurs, categories {example['categories']}"
            )

        lines.extend(
            [
                "",
                "## Analyse des causes probables",
                "",
                "- Les champs peu explicites dans les offres generent souvent des `champ_manquant` ou des `valeur_incorrecte`.",
                "- Les listes de competences sont sensibles aux variantes lexicales, abreviations et niveaux de granularite.",
                "- Les informations absentes mais plausibles peuvent conduire a des `information_hallucinee` si le prompt n'est pas assez contraignant.",
                "",
                "## Recommandations d'amelioration",
                "",
                "- Renforcer les consignes anti-hallucination et les exemples negatifs.",
                "- Ajouter une normalisation metier plus riche pour les contrats, langues et competences.",
                "- Enrichir la verite terrain avec des synonymes acceptables pour l'evaluation.",
            ]
        )

        report = "\n".join(lines)
        output_path = config.EVALUATIONS_DIR / f"{strategy_name}_error_report.md"
        output_path.write_text(report, encoding="utf-8")
        logger.info("Rapport d'erreurs sauvegarde dans %s", output_path)
        return report

    def _is_empty(self, value: Any) -> bool:
        """Determiner si une valeur simple est absente."""
        return value is None or (isinstance(value, str) and not value.strip())

    def _display_analysis(self, analysis: dict) -> None:
        """Afficher un resume d'analyse d'erreurs."""
        table = Table(title="Analyse des erreurs")
        table.add_column("Categorie", style="cyan")
        table.add_column("Nombre", justify="right", style="green")
        for category, count in analysis.get("distribution_categories", {}).items():
            table.add_row(str(category), str(count))
        console.print(table)


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    predictions = [
        {
            "id": 0,
            "extraction": {
                **config.empty_extraction(),
                "titre_poste": "Data Analyst",
                "competences_techniques": ["Python"],
            },
        }
    ]
    ground_truth = [
        {
            "id": 0,
            "ground_truth": {
                **config.empty_extraction(),
                "titre_poste": "Data Analyst Senior",
                "competences_techniques": ["Python", "SQL"],
            },
        }
    ]
    report_analysis = analyzer.analyze_errors(predictions, ground_truth)
    analyzer.generate_error_report(report_analysis, "demo")
    console.print_json(json.dumps(report_analysis, ensure_ascii=False))
