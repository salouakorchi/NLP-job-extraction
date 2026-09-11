"""Pipeline CLI complet pour le projet NLP d'extraction d'offres d'emploi."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import config
from src.data_splitter import DataSplitter
from src.dataset_loader import DatasetLoader
from src.error_analyzer import ErrorAnalyzer
from src.evaluator import Evaluator
from src.llm_extractor import LLMExtractor
from src.post_processor import PostProcessor
from src.preprocessing import Preprocessor
from src.prompts import PromptingStrategy
from src.report_generator import ReportGenerator


console = Console()


def setup_logging() -> None:
    """Configurer le logging du pipeline."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(config.PIPELINE_LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


logger = logging.getLogger(__name__)


def title(message: str) -> None:
    """Afficher un titre d'etape avec rich."""
    console.rule(f"[bold cyan]=== {message} ===[/bold cyan]")


def require_file(path: Path, description: str) -> bool:
    """Verifier qu'un fichier prerequis existe."""
    if path.exists():
        return True
    console.print(f"[yellow]Prerequis manquant ({description}) :[/yellow] {path}")
    logger.warning("Prerequis manquant %s: %s", description, path)
    return False


def load_json(path: Path, default):
    """Charger un fichier JSON avec fallback."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("JSON invalide: %s", path)
        return default


def save_json(path: Path, data) -> None:
    """Sauvegarder des donnees JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ground_truth() -> list[dict]:
    """Charger le fichier de verite terrain utilise pour l'evaluation."""
    data = load_json(config.GROUND_TRUTH_PATH, [])
    return data if isinstance(data, list) else []


def step_load() -> bool:
    """Charger et explorer le dataset brut."""
    title("ETAPE LOAD : chargement du dataset")
    loader = DatasetLoader()
    df = loader.load_dataset(config.TOTAL_SAMPLES)
    loader.explore_dataset(df)
    loader.save_raw_data(df)
    return True


def step_preprocess() -> bool:
    """Nettoyer le dataset brut."""
    title("ETAPE PREPROCESS : nettoyage des descriptions")
    if not require_file(config.RAW_DATA_PATH, "donnees brutes"):
        return False
    df = pd.read_csv(config.RAW_DATA_PATH)
    preprocessor = Preprocessor()
    df_processed = preprocessor.preprocess_dataset(df)
    preprocessor.save_processed_data(df_processed)
    return True


def step_split() -> bool:
    """Separer le dataset en experimentation et evaluation."""
    title("ETAPE SPLIT : separation experiment/evaluation")
    if not require_file(config.PROCESSED_DATA_PATH, "donnees nettoyees"):
        return False
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    eval_size = int(config.TOTAL_SAMPLES * config.EVAL_RATIO)
    splitter = DataSplitter()
    splitter.split_dataset(df, eval_size=eval_size, random_state=config.RANDOM_STATE)
    return True


def step_extract() -> bool:
    """Lancer les extractions LLM sur les jeux experiment et evaluation."""
    provider = config.get_provider_info()
    title(f"ETAPE EXTRACT : extraction LLM {provider['label']}")
    if not require_file(config.EXPERIMENT_SET_PATH, "jeu d'experimentation"):
        return False
    if not require_file(config.EVALUATION_SET_PATH, "jeu d'evaluation"):
        return False

    df_experiment = pd.read_csv(config.EXPERIMENT_SET_PATH)
    df_evaluation = pd.read_csv(config.EVALUATION_SET_PATH)
    extractor = LLMExtractor()
    console.print(
        f"[cyan]Fournisseur actif :[/cyan] {provider['label']} | "
        f"[cyan]Modele :[/cyan] {provider['model'] or 'non defini'}"
    )

    for strategy in PromptingStrategy:
        experiment_path = config.extraction_results_path(
            strategy.value,
            "experiment",
            provider["name"],
        )
        evaluation_path = config.extraction_results_path(
            strategy.value,
            "evaluation",
            provider["name"],
        )

        console.print(f"[cyan]Strategie {strategy.value} - jeu d'experimentation[/cyan]")
        with console.status(
            f"Appels API {provider['label']} ({strategy.value}, experiment)...",
            spinner="dots",
        ):
            extractor.extract_batch(df_experiment, strategy, str(experiment_path))

        console.print(f"[cyan]Strategie {strategy.value} - jeu d'evaluation[/cyan]")
        with console.status(
            f"Appels API {provider['label']} ({strategy.value}, evaluation)...",
            spinner="dots",
        ):
            extractor.extract_batch(df_evaluation, strategy, str(evaluation_path))

    return True


def step_evaluate() -> bool:
    """Calculer les metriques sur la verite terrain."""
    title("ETAPE EVALUATE : calcul des metriques")
    if not require_file(config.GROUND_TRUTH_PATH, "verite terrain"):
        console.print(
            "[yellow]Evaluation ignoree : ajoutez d'abord un fichier data/ground_truth/ground_truth.json.[/yellow]"
        )
        return False

    ground_truths = load_ground_truth()
    if not ground_truths:
        console.print("[yellow]Evaluation ignoree : verite terrain vide.[/yellow]")
        return False

    all_results = load_strategy_results(post_process=True)
    if not all_results:
        console.print("[yellow]Aucun resultat d'extraction disponible pour l'evaluation.[/yellow]")
        return False

    evaluator = Evaluator()
    comparison_df = evaluator.compare_strategies(all_results, ground_truths)
    all_metrics = {}
    for strategy_name, predictions in all_results.items():
        df_strategy = evaluator.evaluate_batch(predictions, ground_truths)
        all_metrics[strategy_name] = evaluator.compute_global_metrics(df_strategy)

    save_json(config.GLOBAL_METRICS_PATH, all_metrics)
    console.print(f"[green]Metriques sauvegardees :[/green] {config.GLOBAL_METRICS_PATH}")
    console.print(f"[green]Table comparative :[/green] {config.COMPARISON_TABLE_PATH}")
    return not comparison_df.empty


def step_analyze() -> bool:
    """Analyser les erreurs par strategie."""
    title("ETAPE ANALYZE : analyse des erreurs")
    if not require_file(config.GROUND_TRUTH_PATH, "verite terrain"):
        console.print("[yellow]Analyse ignoree : verite terrain absente.[/yellow]")
        return False

    ground_truths = load_ground_truth()
    all_results = load_strategy_results(post_process=True)
    if not ground_truths or not all_results:
        console.print("[yellow]Analyse ignoree : donnees insuffisantes.[/yellow]")
        return False

    analyzer = ErrorAnalyzer()
    analyses = analyzer.compare_strategy_errors(all_results, ground_truths)
    for strategy_name, analysis in analyses.items():
        analyzer.generate_error_report(analysis, strategy_name)

    save_json(config.ERROR_ANALYSIS_PATH, analyses)
    console.print(f"[green]Analyse des erreurs sauvegardee :[/green] {config.ERROR_ANALYSIS_PATH}")
    return True


def step_report() -> bool:
    """Generer les figures et le rapport final."""
    title("ETAPE REPORT : visualisations et rapport final")
    generator = ReportGenerator()

    comparison_df = (
        pd.read_csv(config.COMPARISON_TABLE_PATH)
        if config.COMPARISON_TABLE_PATH.exists()
        else pd.DataFrame()
    )
    all_metrics = load_json(config.GLOBAL_METRICS_PATH, {})
    if not all_metrics and not comparison_df.empty:
        all_metrics = comparison_to_metrics(comparison_df)

    error_analyses = load_json(config.ERROR_ANALYSIS_PATH, {})
    dataset_stats = load_json(config.DATASET_STATS_PATH, {})

    generator.plot_strategy_comparison(comparison_df)
    generator.plot_global_f1(all_metrics)
    generator.generate_final_report(all_metrics, error_analyses, dataset_stats)
    return True


def step_demo() -> bool:
    """Executer une demonstration interactive sur un texte saisi."""
    title("ETAPE DEMO : extraction interactive")
    text = Prompt.ask("Collez une offre d'emploi en anglais en une ligne")
    extractor = LLMExtractor()
    processor = PostProcessor()

    table = Table(title="Extractions comparees")
    table.add_column("Strategie", style="cyan")
    table.add_column("Extraction JSON", style="green")
    for strategy in PromptingStrategy:
        with console.status(f"Extraction {strategy.value}...", spinner="dots"):
            extraction = extractor.extract_single(text, strategy)
        normalized = processor.normalize_extraction(extraction)
        table.add_row(strategy.value, json.dumps(normalized, ensure_ascii=False, indent=2))
    console.print(table)
    return True


def load_strategy_results(post_process: bool = False) -> dict:
    """Charger les resultats standard des trois strategies."""
    all_results = {}
    processor = PostProcessor()
    provider = config.get_active_provider()
    for strategy in PromptingStrategy:
        path = config.extraction_results_path(strategy.value, "evaluation", provider)
        if not path.exists():
            logger.warning("Resultats absents pour %s: %s", strategy.value, path)
            continue
        results = load_json(path, [])
        if post_process:
            results, _ = processor.post_process_batch(results)
        all_results[strategy.value] = results
    return all_results


def comparison_to_metrics(comparison_df: pd.DataFrame) -> dict:
    """Reconstruire un dictionnaire de metriques depuis la table CSV."""
    metrics = {}
    for _, row in comparison_df.iterrows():
        strategy_name = str(row.get("strategy"))
        overall_score = float(row.get("overall_score", 0.0))
        strategy_metrics = {
            "overall_score": overall_score,
            "informative_score": float(row.get("informative_score", overall_score)),
        }
        for field in config.ALL_FIELDS:
            score = float(row.get(field, 0.0))
            if field in config.LIST_FIELDS:
                strategy_metrics[field] = {"f1": score}
            else:
                strategy_metrics[field] = {"partial_match": score}
        metrics[strategy_name] = strategy_metrics
    return metrics


def timed_step(step_name: str, func: Callable[[], bool]) -> tuple[bool, float]:
    """Executer une etape avec mesure de temps et gestion d'erreurs."""
    start = time.time()
    try:
        success = func()
    except Exception as exc:
        logger.exception("Erreur pendant l'etape %s", step_name)
        console.print(f"[red]Erreur pendant l'etape {step_name} :[/red] {exc}")
        success = False
    duration = time.time() - start
    status = "terminee" if success else "ignoree/echec"
    console.print(f"[bold]{step_name}[/bold] {status} en {duration:.2f}s")
    logger.info("Etape %s: %s en %.2fs", step_name, status, duration)
    return success, duration


def run_all() -> bool:
    """Executer le pipeline complet avec le fichier de verite terrain existant."""
    ordered_steps: list[tuple[str, Callable[[], bool]]] = [
        ("load", step_load),
        ("preprocess", step_preprocess),
        ("split", step_split),
        ("extract", step_extract),
        ("evaluate", step_evaluate),
        ("analyze", step_analyze),
        ("report", step_report),
    ]

    summary = []
    for step_name, func in ordered_steps:
        success, duration = timed_step(step_name, func)
        summary.append((step_name, success, duration))

    table = Table(title="Resume final du pipeline")
    table.add_column("Etape", style="cyan")
    table.add_column("Statut", style="green")
    table.add_column("Temps", justify="right")
    for step_name, success, duration in summary:
        table.add_row(step_name, "OK" if success else "Ignoree/Echec", f"{duration:.2f}s")
    console.print(table)
    return True


def parse_args() -> argparse.Namespace:
    """Parser les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline NLP d'extraction d'offres d'emploi")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "load",
            "preprocess",
            "split",
            "extract",
            "evaluate",
            "analyze",
            "report",
            "demo",
            "all",
        ],
        help="Etape du pipeline a executer",
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entree du CLI."""
    setup_logging()
    args = parse_args()
    steps: dict[str, Callable[[], bool]] = {
        "load": step_load,
        "preprocess": step_preprocess,
        "split": step_split,
        "extract": step_extract,
        "evaluate": step_evaluate,
        "analyze": step_analyze,
        "report": step_report,
        "demo": step_demo,
        "all": run_all,
    }

    success, _ = timed_step(args.step, steps[args.step])
    return 0 if success or args.step == "all" else 1


if __name__ == "__main__":
    raise SystemExit(main())
