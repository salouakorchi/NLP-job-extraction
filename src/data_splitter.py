"""Separation reproductible entre experimentation et evaluation."""

from __future__ import annotations

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


class DataSplitter:
    """Divise les donnees en deux ensembles stables."""

    def split_dataset(
        self,
        df: pd.DataFrame,
        eval_size: int = 20,
        random_state: int = config.RANDOM_STATE,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Melanger le dataset et creer les jeux experimentation/evaluation."""
        if "id" not in df.columns:
            df = df.copy()
            df.insert(0, "id", range(len(df)))

        shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        eval_size = min(eval_size, len(shuffled))

        df_evaluation = shuffled.iloc[:eval_size].copy()
        df_experiment = shuffled.iloc[eval_size:].copy()
        df_evaluation["split"] = "evaluation"
        df_experiment["split"] = "experiment"

        config.EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.EVALUATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_experiment.to_csv(config.EXPERIMENT_SET_PATH, index=False, encoding="utf-8")
        df_evaluation.to_csv(config.EVALUATION_SET_PATH, index=False, encoding="utf-8")

        table = Table(title="Separation des donnees")
        table.add_column("Split", style="cyan")
        table.add_column("Taille", justify="right", style="green")
        table.add_column("IDs", style="magenta")
        table.add_row(
            "evaluation",
            str(len(df_evaluation)),
            ", ".join(map(str, df_evaluation["id"].head(10).tolist())),
        )
        table.add_row(
            "experiment",
            str(len(df_experiment)),
            ", ".join(map(str, df_experiment["id"].head(10).tolist())),
        )
        console.print(table)

        logger.info(
            "Split termine: %s evaluation, %s experimentation",
            len(df_evaluation),
            len(df_experiment),
        )
        return df_experiment, df_evaluation


if __name__ == "__main__":
    if config.PROCESSED_DATA_PATH.exists():
        input_df = pd.read_csv(config.PROCESSED_DATA_PATH)
    else:
        input_df = pd.DataFrame(
            {
                "id": range(10),
                "job_description": [f"Offre {i}" for i in range(10)],
                "job_description_clean": [f"Offre nettoyee {i}" for i in range(10)],
            }
        )
    splitter = DataSplitter()
    splitter.split_dataset(input_df, eval_size=2)
