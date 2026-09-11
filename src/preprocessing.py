"""Pretraitement et nettoyage des descriptions d'offres d'emploi."""

from __future__ import annotations

import logging
import re
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


class Preprocessor:
    """Nettoie les textes et prepare le corpus pour l'extraction."""

    def clean_text(self, text: str) -> str:
        """Nettoyer une description d'offre d'emploi."""
        if pd.isna(text):
            return ""

        cleaned = str(text)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = "".join(char if char.isprintable() else " " for char in cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[: config.MAX_TEXT_LENGTH]

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Appliquer le nettoyage au DataFrame et supprimer les textes vides."""
        df_processed = df.copy()
        df_processed["job_description_clean"] = df_processed["job_description"].apply(
            self.clean_text
        )
        df_processed = df_processed[df_processed["job_description_clean"].str.len() > 0]
        df_processed = df_processed.reset_index(drop=True)
        df_processed["text_length"] = df_processed["job_description_clean"].str.len()
        mean_length = df_processed["text_length"].mean() if not df_processed.empty else 0.0
        max_length = int(df_processed["text_length"].max()) if not df_processed.empty else 0

        table = Table(title="Pretraitement")
        table.add_column("Mesure", style="cyan")
        table.add_column("Valeur", style="green")
        table.add_row("Lignes conservees", str(len(df_processed)))
        table.add_row("Longueur moyenne", f"{mean_length:.2f}")
        table.add_row("Longueur max", str(max_length))
        console.print(table)
        logger.info("Pretraitement termine: %s lignes", len(df_processed))
        return df_processed

    def save_processed_data(self, df: pd.DataFrame) -> None:
        """Sauvegarder les donnees nettoyees."""
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_DATA_PATH, index=False, encoding="utf-8")
        logger.info("Donnees nettoyees sauvegardees dans %s", config.PROCESSED_DATA_PATH)
        console.print(
            f"[green]Donnees traitees sauvegardees :[/green] {config.PROCESSED_DATA_PATH}"
        )


if __name__ == "__main__":
    if config.RAW_DATA_PATH.exists():
        input_df = pd.read_csv(config.RAW_DATA_PATH)
    else:
        input_df = pd.DataFrame(
            {
                "id": [0],
                "titre_original": ["Developpeur"],
                "entreprise_originale": ["Demo"],
                "job_description": ["<p>Python\nSQL\tAutonomie</p>"],
            }
        )
    preprocessor = Preprocessor()
    output_df = preprocessor.preprocess_dataset(input_df)
    preprocessor.save_processed_data(output_df)
