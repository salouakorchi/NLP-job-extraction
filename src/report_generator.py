"""Generation des visualisations et du rapport final."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class ReportGenerator:
    """Produit les figures et le rapport Markdown final."""

    def plot_strategy_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Tracer une heatmap champs x strategies."""
        output_path = config.VISUALIZATIONS_DIR / "heatmap_comparison.png"
        if comparison_df.empty or "strategy" not in comparison_df.columns:
            self._save_empty_plot(output_path, "Aucune comparaison disponible")
            return

        fields = [field for field in config.ALL_FIELDS if field in comparison_df.columns]
        matrix = comparison_df.set_index("strategy")[fields].T

        plt.figure(figsize=(9, 7))
        sns.heatmap(matrix, annot=True, cmap="YlOrRd", vmin=0, vmax=1, fmt=".2f")
        plt.title("Comparaison des strategies par champ")
        plt.xlabel("Strategie")
        plt.ylabel("Champ")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info("Heatmap sauvegardee dans %s", output_path)

    def plot_global_f1(self, global_metrics: dict) -> None:
        """Tracer le score global moyen par strategie."""
        output_path = config.VISUALIZATIONS_DIR / "global_f1_barplot.png"
        if not global_metrics:
            self._save_empty_plot(output_path, "Aucun score global disponible")
            return

        strategies = list(global_metrics.keys())
        score_key = self._primary_score_key(global_metrics)
        scores = [
            float(metrics.get(score_key, metrics.get("overall_score", 0.0)))
            for metrics in global_metrics.values()
            if isinstance(metrics, dict)
        ]
        if not scores:
            self._save_empty_plot(output_path, "Aucun score global disponible")
            return

        colors = ["#2E86AB", "#F18F01", "#6A994E", "#C73E1D"]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(strategies, scores, color=colors[: len(strategies)])
        plt.ylim(0, 1)
        ylabel = "Score informatif moyen" if score_key == "informative_score" else "Score global moyen"
        plt.ylabel(ylabel)
        plt.title("Performance par strategie")
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info("Barplot global sauvegarde dans %s", output_path)

    def generate_final_report(
        self,
        all_metrics: dict,
        error_analyses: dict,
        dataset_stats: dict,
    ) -> str:
        """Generer le rapport final README_RESULTS.md en francais."""
        provider = config.get_provider_info()
        dataset_stats = self._resolve_dataset_stats(dataset_stats)
        ground_truth_count = self._ground_truth_count(error_analyses)
        lines = [
            "# Rapport final - Extraction d'informations d'offres d'emploi",
            "",
            "## Resume executif",
            "",
            "Ce projet construit un pipeline NLP complet pour convertir des offres d'emploi anglophones en objets JSON structures. Il compare trois strategies de prompting adaptees au dataset anglais `job_offers.csv` et mesure leurs performances sur une verite terrain annotee manuellement.",
            "",
            "## Dataset",
            "",
            f"- Source : `data/raw/job_offers.csv` avec fallback HuggingFace `{config.DATASET_NAME}`",
            f"- Langue des offres : anglais (`{config.DATASET_LANGUAGE}`)",
            f"- Nombre d'echantillons charges : {dataset_stats.get('nombre_echantillons', 'n/a')}",
            f"- Longueur moyenne des descriptions : {dataset_stats.get('longueur_moyenne', 0):.2f}",
            f"- Longueur min/max : {dataset_stats.get('longueur_min', 'n/a')} / {dataset_stats.get('longueur_max', 'n/a')}",
            f"- Nombre d'exemples de verite terrain evalues : {ground_truth_count}",
            f"- Fournisseur LLM actif : {provider.get('label')} / {provider.get('model')}",
            "",
            "## Schema JSON utilise",
            "",
            "```json",
            json.dumps(config.JSON_SCHEMA_FIELDS, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Strategies de prompting",
            "",
            "- `zero_shot` : consigne courte en anglais, sortie JSON demandee.",
            "- `constrained` : schema explicite, regles anti-hallucination, categories de contrat anglaises et conservation du vocabulaire source.",
            "- `few_shot` : trois exemples anglophones proches du dataset avant l'offre cible.",
            "",
            "Categories standard de `type_contrat` : `Full-time`, `Part-time`, `Contract`, `Temporary`, `Internship`, `Apprenticeship`, `Freelance`, `Other`.",
            "",
            "## Notes methodologiques sur les scores",
            "",
            "- Les champs absents des deux cotes (`null`/`null` ou `[]`/`[]`) comptent comme corrects.",
            "- Les listes de competences utilisent un matching lexical fuzzy pour accepter les variantes proches comme `excellent communication skills` et `communication skills`.",
            "- L'evaluation utilise les sorties LLM normalisees, sans enrichissement automatique depuis le texte source.",
            "- `overall_score` moyenne tous les champs, y compris les absences correctement predites.",
            "- `informative_score` moyenne seulement les champs non vides dans le ground truth ; c'est le score le plus utile pour juger l'extraction effective.",
            "- Avec un seul exemple de verite terrain, les scores indiquent seulement la performance sur ce run, pas une performance generale definitive.",
            "- Diagnostic detaille : [performance_diagnosis.md](evaluations/performance_diagnosis.md).",
            "",
            "## Tableau comparatif des performances",
            "",
            self._metrics_markdown_table(all_metrics),
            "",
            "## Visualisations",
            "",
            "- [Heatmap comparative](visualizations/heatmap_comparison.png)",
            "- [Score global par strategie](visualizations/global_f1_barplot.png)",
            "",
            "## Analyse des erreurs par strategie",
            "",
        ]

        for strategy_name, analysis in error_analyses.items():
            lines.extend(
                [
                    f"### {strategy_name}",
                    "",
                    "Champs les plus difficiles :",
                ]
            )
            top_fields = analysis.get("top_3_champs_erreurs", [])
            if top_fields:
                for item in top_fields:
                    lines.append(f"- {item['champ']} : {item['nb_erreurs']} erreurs")
            else:
                lines.append("- Aucune erreur detectee sur le jeu d'evaluation actuel.")
            lines.extend(
                [
                    "",
                    f"Rapport detaille : [rapport {strategy_name}](evaluations/{strategy_name}_error_report.md)",
                    "",
                ]
            )

        lines.extend(
            [
                "## Conclusion et recommandations",
                "",
                "Sur le jeu d'evaluation actuel, le score informatif montre mieux les differences entre les strategies que le score global, car plusieurs champs de l'offre sont volontairement absents. Il faut confirmer les resultats sur un echantillon annote plus large.",
                "",
                "Recommandations : renforcer la normalisation post-traitement, ajouter des exemples negatifs, enrichir la verite terrain avec des synonymes acceptables, et evaluer sur un corpus plus large.",
                "",
                "## Pistes futures",
                "",
                "- Ajouter une evaluation semantique pour les competences proches mais non identiques.",
                "- Tester plusieurs fournisseurs LLM et plusieurs temperatures.",
                "- Ajouter un editeur de verite terrain dedie si le corpus doit etre etendu.",
                "- Ajouter des tests unitaires dedies aux fonctions de normalisation et d'evaluation.",
            ]
        )

        report = "\n".join(lines)
        config.FINAL_REPORT_PATH.write_text(report, encoding="utf-8")
        logger.info("Rapport final sauvegarde dans %s", config.FINAL_REPORT_PATH)
        console.print(f"[green]Rapport final genere :[/green] {config.FINAL_REPORT_PATH}")
        return report

    def _field_score(self, metrics: dict, field: str) -> float:
        """Recuperer le score d'un champ selon son type."""
        field_metrics = metrics.get(field, {})
        if field in config.LIST_FIELDS:
            return float(field_metrics.get("f1", 0.0))
        return float(field_metrics.get("partial_match", 0.0))

    def _ground_truth_count(self, error_analyses: dict) -> int:
        """Recuperer le nombre d'exemples evalues depuis les analyses d'erreurs."""
        counts = [
            int(analysis.get("nombre_exemples", 0))
            for analysis in error_analyses.values()
            if isinstance(analysis, dict)
        ]
        return max(counts) if counts else 0

    def _resolve_dataset_stats(self, dataset_stats: dict) -> dict:
        """Completer les statistiques dataset depuis les CSV si elles manquent."""
        required = {"nombre_echantillons", "longueur_moyenne", "longueur_min", "longueur_max"}
        if required.issubset(dataset_stats):
            return dataset_stats

        for path in [config.PROCESSED_DATA_PATH, config.RAW_DATA_PATH]:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                logger.exception("Impossible de recalculer les stats dataset depuis %s", path)
                continue

            text_column = (
                "job_description_clean"
                if "job_description_clean" in df.columns
                else "job_description"
            )
            if text_column not in df.columns:
                continue

            lengths = df[text_column].fillna("").astype(str).str.len()
            return {
                **dataset_stats,
                "nombre_echantillons": int(len(df)),
                "longueur_min": int(lengths.min()) if len(lengths) else 0,
                "longueur_max": int(lengths.max()) if len(lengths) else 0,
                "longueur_moyenne": float(lengths.mean()) if len(lengths) else 0.0,
            }
        return dataset_stats

    def _metrics_markdown_table(self, all_metrics: dict) -> str:
        """Construire un tableau Markdown des scores globaux."""
        lines = ["| Strategie | Overall score | Informative score |", "|---|---:|---:|"]
        if not all_metrics:
            lines.append("| n/a | 0.000 | 0.000 |")
            return "\n".join(lines)
        for strategy_name, metrics in all_metrics.items():
            if not isinstance(metrics, dict):
                lines.append(f"| {strategy_name} | 0.000 | 0.000 |")
                continue
            overall = float(metrics.get("overall_score", 0.0))
            informative = float(metrics.get("informative_score", overall))
            lines.append(f"| {strategy_name} | {overall:.3f} | {informative:.3f} |")
        return "\n".join(lines)

    def _primary_score_key(self, global_metrics: dict) -> str:
        """Choisir la metrique principale pour les graphiques globaux."""
        if any(
            isinstance(metrics, dict) and "informative_score" in metrics
            for metrics in global_metrics.values()
        ):
            return "informative_score"
        return "overall_score"

    def _save_empty_plot(self, output_path: Path, message: str) -> None:
        """Sauvegarder une figure placeholder quand les donnees manquent."""
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, message, ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info("Figure placeholder sauvegardee dans %s", output_path)


if __name__ == "__main__":
    generator = ReportGenerator()
    demo_comparison = pd.DataFrame(
        [
            {"strategy": "zero_shot", "overall_score": 0.45, "informative_score": 0.35, **{field: 0.4 for field in config.ALL_FIELDS}},
            {"strategy": "constrained", "overall_score": 0.62, "informative_score": 0.55, **{field: 0.6 for field in config.ALL_FIELDS}},
            {"strategy": "few_shot", "overall_score": 0.58, "informative_score": 0.50, **{field: 0.55 for field in config.ALL_FIELDS}},
        ]
    )
    demo_metrics = {
        row["strategy"]: {
            "overall_score": row["overall_score"],
            "informative_score": row["informative_score"],
            **{
                field: {"f1": row[field]} if field in config.LIST_FIELDS else {"partial_match": row[field]}
                for field in config.ALL_FIELDS
            },
        }
        for row in demo_comparison.to_dict(orient="records")
    }
    generator.plot_strategy_comparison(demo_comparison)
    generator.plot_global_f1(demo_metrics)
    generator.generate_final_report(demo_metrics, {}, {"nombre_echantillons": 3})
