"""Chargement et exploration du dataset HuggingFace."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

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


class DatasetLoader:
    """Charge, explore et sauvegarde les offres d'emploi brutes."""

    def load_dataset(self, n_samples: int = config.TOTAL_SAMPLES) -> pd.DataFrame:
        """Charger les premiers exemples du CSV local ou du dataset HuggingFace."""
        local_df = self._load_local_job_offers(n_samples)
        if local_df is not None:
            return local_df

        from datasets import load_dataset

        logger.info("Chargement du dataset %s", config.DATASET_NAME)
        dataset = load_dataset(config.DATASET_NAME, split="train")
        n_samples = min(n_samples, len(dataset))
        dataset = dataset.select(range(n_samples))
        df = dataset.to_pandas()
        df = self._normalize_columns(df)
        logger.info("%s offres chargees depuis HuggingFace", len(df))
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliser les colonnes sources vers le format interne du projet."""
        if "titre_original" in df.columns:
            title_column = "titre_original"
        else:
            title_column = self._resolve_column(df, ["position", "position_title", "title"])

        if "entreprise_originale" in df.columns:
            company_column = "entreprise_originale"
        else:
            company_column = self._resolve_column(df, ["company_name", "company"])
        description_column = self._resolve_column(
            df,
            ["job_description", "description"],
        )

        normalized = pd.DataFrame(
            {
                "titre_original": df[title_column] if title_column else None,
                "entreprise_originale": df[company_column] if company_column else None,
                "job_description": df[description_column] if description_column else None,
            }
        )
        if "id" in df.columns:
            normalized.insert(0, "id", df["id"])
        else:
            normalized.insert(0, "id", range(len(normalized)))
        return normalized.reset_index(drop=True)

    def _load_local_job_offers(self, n_samples: int) -> pd.DataFrame | None:
        """Charger le fichier job_offers.csv local quand il est disponible."""
        candidates = [
            config.BASE_DIR / "job_offers.csv",
            config.RAW_DATA_PATH,
        ]
        for path in candidates:
            if path.exists():
                logger.info("Chargement du CSV local %s", path)
                df = pd.read_csv(path).head(n_samples)
                df = self._normalize_columns(df)
                logger.info("%s offres chargees depuis le CSV local", len(df))
                return df
        return None

    def explore_dataset(self, df: pd.DataFrame) -> dict:
        """Afficher et sauvegarder des statistiques descriptives du dataset."""
        lengths = df["job_description"].fillna("").astype(str).str.len()
        top_companies = (
            df["entreprise_originale"]
            .fillna("Inconnue")
            .astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )
        null_values = df.isna().sum().astype(int).to_dict()

        stats = {
            "nombre_echantillons": int(len(df)),
            "longueur_min": int(lengths.min()) if len(lengths) else 0,
            "longueur_max": int(lengths.max()) if len(lengths) else 0,
            "longueur_moyenne": float(lengths.mean()) if len(lengths) else 0.0,
            "top_10_entreprises": top_companies,
            "valeurs_nulles": null_values,
        }

        table = Table(title="Exploration du dataset")
        table.add_column("Mesure", style="cyan")
        table.add_column("Valeur", style="green")
        table.add_row("Nombre d'echantillons", str(stats["nombre_echantillons"]))
        table.add_row("Longueur min", str(stats["longueur_min"]))
        table.add_row("Longueur max", str(stats["longueur_max"]))
        table.add_row("Longueur moyenne", f"{stats['longueur_moyenne']:.2f}")
        table.add_row("Valeurs nulles", json.dumps(null_values, ensure_ascii=False))
        console.print(table)

        company_table = Table(title="Top 10 entreprises")
        company_table.add_column("Entreprise", style="cyan")
        company_table.add_column("Nombre", style="green", justify="right")
        for company, count in top_companies.items():
            company_table.add_row(str(company), str(count))
        console.print(company_table)

        config.DATASET_STATS_PATH.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Statistiques sauvegardees dans %s", config.DATASET_STATS_PATH)
        return stats

    def save_raw_data(self, df: pd.DataFrame) -> None:
        """Sauvegarder les offres brutes au format CSV."""
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RAW_DATA_PATH, index=False, encoding="utf-8")
        logger.info("Donnees brutes sauvegardees dans %s", config.RAW_DATA_PATH)
        console.print(f"[green]Donnees brutes sauvegardees :[/green] {config.RAW_DATA_PATH}")

    def _resolve_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Trouver la premiere colonne disponible parmi plusieurs noms possibles."""
        for column in candidates:
            if column in df.columns:
                if column != candidates[0]:
                    logger.info("Colonne %s absente, utilisation de %s", candidates[0], column)
                return column
        logger.warning("Aucune colonne trouvee parmi: %s", candidates)
        return None


if __name__ == "__main__":
    loader = DatasetLoader()
    sample_df = loader.load_dataset(n_samples=5)
    loader.explore_dataset(sample_df)
    loader.save_raw_data(sample_df)
