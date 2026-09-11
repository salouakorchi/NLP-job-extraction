"""Interface web locale pour piloter le pipeline NLP."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import set_key
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import config
from main import (
    run_all,
    setup_logging,
    step_analyze,
    step_evaluate,
    step_extract,
    step_load,
    step_preprocess,
    step_report,
    step_split,
)
from src.llm_extractor import LLMExtractor
from src.post_processor import PostProcessor
from src.prompts import PromptingStrategy


setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nlp-job-extraction-local-secret")


STEP_FUNCTIONS: dict[str, Callable[[], bool]] = {
    "load": step_load,
    "preprocess": step_preprocess,
    "split": step_split,
    "extract": step_extract,
    "evaluate": step_evaluate,
    "analyze": step_analyze,
    "report": step_report,
    "all": run_all,
}

TOKEN_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "tokens per minute",
    "token limit",
    "quota",
    "insufficient_quota",
    "too many requests",
    "http 429",
    "limit reached",
]

def csv_rows(path: Path) -> int | None:
    """Compter les lignes d'un CSV si le fichier existe."""
    if not path.exists():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        logger.exception("Impossible de compter les lignes CSV: %s", path)
        return None


def json_items(path: Path) -> int | None:
    """Compter les elements d'une liste JSON si le fichier existe."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else None
    except Exception:
        logger.exception("Impossible de compter les elements JSON: %s", path)
        return None


def read_json(path: Path, default):
    """Lire un fichier JSON avec fallback."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.exception("JSON invalide: %s", path)
        return default


def short_text(value: str | None, max_length: int = 260) -> str:
    """Raccourcir un texte long pour l'affichage web."""
    if not value:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    return f"{text[:max_length].rstrip()}..."


def is_token_limit_error(message: str | None) -> bool:
    """Detecter une erreur de tokens, quota ou rate limit API."""
    if not message:
        return False
    normalized = str(message).lower()
    return any(pattern in normalized for pattern in TOKEN_LIMIT_PATTERNS)


def friendly_token_message(error_message: str, provider_info: dict) -> str:
    """Construire un message clair pour l'utilisateur web."""
    retry_match = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)s",
        error_message,
        re.IGNORECASE,
    )
    retry_hint = ""
    if retry_match:
        retry_hint = f" Attendez environ {float(retry_match.group(1)):.1f}s puis relancez l'etape."
    else:
        retry_hint = " Attendez un peu, reduisez le nombre d'exemples ou changez de modele/fournisseur."

    return (
        f"Limite de tokens ou quota atteint pour {provider_info.get('label', 'le fournisseur choisi')} "
        f"({provider_info.get('model') or 'modele non defini'}).{retry_hint}"
    )


def latest_api_alert(active_provider: str, provider_info: dict) -> dict | None:
    """Retourner une alerte API lisible depuis les derniers resultats d'extraction."""
    for strategy in PromptingStrategy:
        for split_name in ["evaluation", "experiment"]:
            path = config.extraction_results_path(strategy.value, split_name, active_provider)
            data = read_json(path, [])
            if not isinstance(data, list):
                continue
            for item in reversed(data):
                if not isinstance(item, dict) or item.get("success") is not False:
                    continue
                error_message = str(item.get("error") or "")
                if is_token_limit_error(error_message):
                    return {
                        "type": "token_limit",
                        "provider": active_provider,
                        "strategy": strategy.value,
                        "split": split_name,
                        "message": friendly_token_message(error_message, provider_info),
                        "details": short_text(error_message, 600),
                    }
    return None


def app_status() -> dict:
    """Construire l'etat courant du projet pour le tableau de bord."""
    extractions = {}
    active_provider = config.get_active_provider()
    provider_info = config.get_provider_info(active_provider)
    provider_public = {
        key: value
        for key, value in provider_info.items()
        if key != "api_key"
    }
    provider_public["api_key_configured"] = bool(provider_info.get("api_key"))
    for strategy in PromptingStrategy:
        path = config.extraction_results_path(strategy.value, "evaluation", active_provider)
        data = read_json(path, [])
        if isinstance(data, list):
            extractions[strategy.value] = {
                "total": len(data),
                "success": sum(1 for item in data if item.get("success")),
                "errors": sum(1 for item in data if item.get("success") is False),
            }
        else:
            extractions[strategy.value] = {"total": 0, "success": 0, "errors": 0}

    api_alert = latest_api_alert(active_provider, provider_public)

    return {
        "active_provider": active_provider,
        "provider_info": provider_public,
        "providers": [
            {"name": name, **provider}
            for name, provider in config.LLM_PROVIDERS.items()
        ],
        "api_key_configured": bool(provider_info.get("api_key")),
        "raw_rows": csv_rows(config.RAW_DATA_PATH),
        "processed_rows": csv_rows(config.PROCESSED_DATA_PATH),
        "experiment_rows": csv_rows(config.EXPERIMENT_SET_PATH),
        "evaluation_rows": csv_rows(config.EVALUATION_SET_PATH),
        "ground_truth_items": json_items(config.GROUND_TRUTH_PATH),
        "comparison_ready": config.COMPARISON_TABLE_PATH.exists(),
        "report_ready": config.FINAL_REPORT_PATH.exists(),
        "extractions": extractions,
        "api_alert": api_alert,
    }


def tail_log(lines: int = 80) -> str:
    """Retourner les dernieres lignes du journal pipeline."""
    if not config.PIPELINE_LOG_PATH.exists():
        return ""
    content = config.PIPELINE_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


@app.route("/")
def index():
    """Afficher le tableau de bord."""
    return render_template("index.html", status=app_status(), log_tail=tail_log(30))


@app.route("/api/status")
def api_status():
    """Retourner l'etat du projet au format JSON."""
    return jsonify(app_status())


@app.route("/api/run/<step_name>", methods=["POST"])
def api_run_step(step_name: str):
    """Executer une etape du pipeline depuis l'interface web."""
    if step_name not in STEP_FUNCTIONS:
        return jsonify({"ok": False, "message": f"Etape inconnue: {step_name}"}), 404

    buffer = io.StringIO()
    start = time.time()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            success = STEP_FUNCTIONS[step_name]()
        duration = time.time() - start
        status = app_status()
        api_alert = status.get("api_alert")
        message = "Etape terminee" if success else "Etape terminee avec avertissement"
        if api_alert:
            message = api_alert.get("message", message)
        logger.info("Etape web %s terminee en %.2fs", step_name, duration)
        return jsonify(
            {
                "ok": bool(success),
                "message": message,
                "duration": duration,
                "output": buffer.getvalue(),
                "status": status,
                "api_alert": api_alert,
                "log_tail": tail_log(),
            }
        )
    except Exception as exc:
        duration = time.time() - start
        logger.exception("Erreur pendant l'etape web %s", step_name)
        status = app_status()
        api_alert = status.get("api_alert")
        if not api_alert and is_token_limit_error(str(exc)):
            provider_info = status.get("provider_info", {})
            api_alert = {
                "type": "token_limit",
                "provider": status.get("active_provider"),
                "strategy": None,
                "split": None,
                "message": friendly_token_message(str(exc), provider_info),
                "details": short_text(str(exc), 600),
            }
        return jsonify(
            {
                "ok": False,
                "message": api_alert.get("message") if api_alert else str(exc),
                "duration": duration,
                "output": buffer.getvalue(),
                "status": status,
                "api_alert": api_alert,
                "log_tail": tail_log(),
            }
        ), 500


@app.route("/settings/api-key", methods=["POST"])
def save_api_key():
    """Sauvegarder le fournisseur, la cle API et le modele dans .env."""
    provider_name = request.form.get("llm_provider", "groq").strip().lower()
    if provider_name not in config.LLM_PROVIDERS:
        provider_name = "groq"

    provider = dict(config.LLM_PROVIDERS[provider_name])
    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        api_key = os.getenv(provider["api_key_env"], "")
    model = request.form.get("model", "").strip() or provider.get("default_model", "")
    endpoint = request.form.get("endpoint", "").strip()

    env_path = config.BASE_DIR / ".env"
    if not env_path.exists():
        env_path.write_text("", encoding="utf-8")

    set_key(str(env_path), "LLM_PROVIDER", provider_name)
    set_key(str(env_path), provider["api_key_env"], api_key)
    set_key(str(env_path), provider["model_env"], model)
    os.environ["LLM_PROVIDER"] = provider_name
    os.environ[provider["api_key_env"]] = api_key
    os.environ[provider["model_env"]] = model
    config.LLM_PROVIDER = provider_name

    endpoint_env = provider.get("endpoint_env")
    if endpoint_env:
        set_key(str(env_path), endpoint_env, endpoint)
        os.environ[endpoint_env] = endpoint

    flash(f"Fournisseur {provider['label']} enregistre.", "success")
    return redirect(url_for("index"))


def load_extraction_groups() -> list[dict]:
    """Charger les resultats d'extraction disponibles pour la page Resultats."""
    groups = []
    processor = PostProcessor()
    splits = [
        ("evaluation", "Evaluation"),
        ("experiment", "Experimentation"),
    ]

    for provider_name, provider_config in config.LLM_PROVIDERS.items():
        for strategy in PromptingStrategy:
            for split_name, split_label in splits:
                path = config.extraction_results_path(
                    strategy.value,
                    split_name,
                    provider_name,
                )
                if not path.exists():
                    continue

                data = read_json(path, [])
                if not isinstance(data, list):
                    data = []

                records = []
                for result in data:
                    if not isinstance(result, dict):
                        continue
                    extraction = result.get("extraction", {})
                    records.append(
                        {
                            "id": result.get("id"),
                            "success": bool(result.get("success")),
                            "error": result.get("error"),
                            "model": result.get("model") or provider_config.get("default_model"),
                            "job_excerpt": short_text(result.get("job_description")),
                            "extraction": processor.normalize_extraction(extraction),
                        }
                    )

                groups.append(
                    {
                        "provider": provider_name,
                        "provider_label": provider_config["label"],
                        "strategy": strategy.value,
                        "split": split_name,
                        "split_label": split_label,
                        "filename": path.name,
                        "total": len(data),
                        "success": sum(1 for item in data if isinstance(item, dict) and item.get("success")),
                        "errors": sum(1 for item in data if isinstance(item, dict) and item.get("success") is False),
                        "records": records,
                    }
                )

    return groups


@app.route("/demo", methods=["GET", "POST"])
def demo():
    """Afficher une demo d'extraction sur texte libre."""
    results = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("job_description", "").strip()
        extractor = LLMExtractor()
        processor = PostProcessor()
        results = []

        for strategy in PromptingStrategy:
            extraction = extractor.extract_single(input_text, strategy)
            results.append(
                {
                    "strategy": strategy.value,
                    "provider": extractor.provider["label"],
                    "model": extractor.provider.get("model"),
                    "error": extractor.last_error,
                    "extraction": processor.normalize_extraction(extraction),
                }
            )

    return render_template(
        "demo.html",
        results=results,
        input_text=input_text,
        status=app_status(),
    )


@app.route("/results")
def results():
    """Afficher les resultats et visualisations disponibles."""
    visualization_names = [
        "heatmap_comparison.png",
        "global_f1_barplot.png",
    ]
    visualizations = [
        {
            "filename": filename,
            "exists": (config.VISUALIZATIONS_DIR / filename).exists(),
        }
        for filename in visualization_names
    ]
    comparison_html = ""
    if config.COMPARISON_TABLE_PATH.exists():
        comparison_html = pd.read_csv(config.COMPARISON_TABLE_PATH).to_html(
            classes="data-table",
            index=False,
            border=0,
        )

    return render_template(
        "results.html",
        visualizations=visualizations,
        comparison_html=comparison_html,
        extraction_groups=load_extraction_groups(),
        status=app_status(),
    )


@app.route("/visualizations/<path:filename>")
def visualization_file(filename: str):
    """Servir les images de visualisation generees."""
    return send_from_directory(config.VISUALIZATIONS_DIR, filename)


@app.template_filter("json_pretty")
def json_pretty(value) -> str:
    """Formatter un objet Python en JSON lisible."""
    return json.dumps(value, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
