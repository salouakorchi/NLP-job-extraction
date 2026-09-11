"""Configuration centrale du projet d'extraction d'offres d'emploi."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DATASET_NAME = "jacob-hugging-face/job-descriptions"
DATASET_LANGUAGE = "en"

TOTAL_SAMPLES = 5
EVAL_RATIO = 0.2
EXP_RATIO = 0.8
RANDOM_STATE = 42
MAX_TEXT_LENGTH = 2000
API_DELAY = 1.0
MAX_RETRIES = 3

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENT_DATA_DIR = DATA_DIR / "experiment"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

RESULTS_DIR = BASE_DIR / "results"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
EVALUATIONS_DIR = RESULTS_DIR / "evaluations"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

RAW_DATA_PATH = RAW_DATA_DIR / "job_offers.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "job_offers_clean.csv"
EXPERIMENT_SET_PATH = EXPERIMENT_DATA_DIR / "experiment_set.csv"
EVALUATION_SET_PATH = EVALUATION_DATA_DIR / "evaluation_set.csv"
GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "ground_truth.json"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_stats.json"
PIPELINE_LOG_PATH = RESULTS_DIR / "pipeline.log"
COMPARISON_TABLE_PATH = EVALUATIONS_DIR / "comparison_table.csv"
GLOBAL_METRICS_PATH = EVALUATIONS_DIR / "global_metrics.json"
ERROR_ANALYSIS_PATH = EVALUATIONS_DIR / "error_analysis.json"
FINAL_REPORT_PATH = RESULTS_DIR / "README_RESULTS.md"

JSON_SCHEMA_FIELDS = {
    "titre_poste": "string",
    "entreprise": "string",
    "localisation": "string",
    "type_contrat": "string",
    "experience_requise": "string",
    "formation_requise": "string",
    "competences_techniques": ["string"],
    "competences_douces": ["string"],
    "langues_requises": ["string"],
    "salaire": "string or null",
    "secteur_activite": "string",
    "avantages": ["string"],
}

STRING_FIELDS = [
    "titre_poste",
    "entreprise",
    "localisation",
    "type_contrat",
    "experience_requise",
    "formation_requise",
    "salaire",
    "secteur_activite",
]

LIST_FIELDS = [
    "competences_techniques",
    "competences_douces",
    "langues_requises",
    "avantages",
]

ALL_FIELDS = STRING_FIELDS + LIST_FIELDS

TYPE_CONTRAT_VALUES = [
    "Full-time",
    "Part-time",
    "Contract",
    "Temporary",
    "Internship",
    "Apprenticeship",
    "Freelance",
    "Other",
]

LLM_PROVIDERS = {
    "groq": {
        "label": "Groq",
        "api_type": "openai_compatible",
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "supports_json_mode": True,
        "free_note": "Compte Groq gratuit avec limites de debit.",
    },
    "openai": {
        "label": "OpenAI",
        "api_type": "openai_compatible",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "supports_json_mode": True,
        "free_note": "Necessite une cle OpenAI et des credits disponibles.",
    },
    "gemini": {
        "label": "Google Gemini",
        "api_type": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "supports_json_mode": True,
        "free_note": "Google AI Studio propose un niveau gratuit selon disponibilite.",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_type": "openai_compatible",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "supports_json_mode": True,
        "free_note": "Necessite une cle DeepSeek; conditions gratuites variables.",
    },
    "zai": {
        "label": "Z.ai",
        "api_type": "openai_compatible",
        "api_key_env": "ZAI_API_KEY",
        "model_env": "ZAI_MODEL",
        "default_model": "glm-5.1",
        "endpoint": "https://api.z.ai/api/paas/v4/chat/completions",
        "supports_json_mode": False,
        "free_note": "Z.ai peut proposer des quotas gratuits selon le compte.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "api_type": "openai_compatible",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "openrouter/auto",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "supports_json_mode": True,
        "free_note": "Permet de choisir des modeles avec suffixe :free quand disponibles.",
    },
    "together": {
        "label": "Together AI",
        "api_type": "openai_compatible",
        "api_key_env": "TOGETHER_API_KEY",
        "model_env": "TOGETHER_MODEL",
        "default_model": "openai/gpt-oss-20b",
        "endpoint": "https://api.together.ai/v1/chat/completions",
        "supports_json_mode": True,
        "free_note": "Compte gratuit possible; quotas et modeles selon disponibilite.",
    },
    "huggingface": {
        "label": "Hugging Face",
        "api_type": "openai_compatible",
        "api_key_env": "HUGGINGFACE_API_KEY",
        "model_env": "HUGGINGFACE_MODEL",
        "default_model": "openai/gpt-oss-20b",
        "endpoint": "https://router.huggingface.co/v1/chat/completions",
        "supports_json_mode": False,
        "free_note": "Inference Providers avec endpoint compatible OpenAI pour le chat.",
    },
    "mistral": {
        "label": "Mistral AI",
        "api_type": "openai_compatible",
        "api_key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "supports_json_mode": False,
        "free_note": "Necessite une cle Mistral; quotas selon le compte.",
    },
    "cerebras": {
        "label": "Cerebras",
        "api_type": "openai_compatible",
        "api_key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "gpt-oss-120b",
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "supports_json_mode": True,
        "free_note": "API compatible OpenAI; quotas gratuits selon disponibilite.",
    },
    "custom": {
        "label": "OpenAI-compatible custom",
        "api_type": "openai_compatible",
        "api_key_env": "CUSTOM_LLM_API_KEY",
        "model_env": "CUSTOM_LLM_MODEL",
        "default_model": "",
        "endpoint_env": "CUSTOM_LLM_ENDPOINT",
        "endpoint": "",
        "supports_json_mode": True,
        "free_note": "Pour tout autre fournisseur compatible /chat/completions.",
    },
}


def ensure_directories() -> None:
    """Creer tous les dossiers necessaires au pipeline."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXPERIMENT_DATA_DIR,
        EVALUATION_DATA_DIR,
        GROUND_TRUTH_DIR,
        EXTRACTIONS_DIR,
        EVALUATIONS_DIR,
        VISUALIZATIONS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def get_active_provider() -> str:
    """Retourner le fournisseur LLM actif."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    return provider if provider in LLM_PROVIDERS else "groq"


def get_provider_info(provider_name: str | None = None) -> dict:
    """Retourner la configuration resolue d'un fournisseur LLM."""
    provider_name = (provider_name or get_active_provider()).strip().lower()
    if provider_name not in LLM_PROVIDERS:
        provider_name = "groq"

    provider = dict(LLM_PROVIDERS[provider_name])
    endpoint_env = provider.get("endpoint_env")
    if endpoint_env:
        provider["endpoint"] = os.getenv(endpoint_env, provider.get("endpoint", ""))

    provider["name"] = provider_name
    provider["api_key"] = os.getenv(provider["api_key_env"], "")
    provider["model"] = os.getenv(
        provider["model_env"],
        provider.get("default_model", ""),
    )
    return provider


def extraction_results_path(
    strategy_name: str,
    split_name: str = "evaluation",
    provider_name: str | None = None,
) -> Path:
    """Construire le chemin de resultats pour un fournisseur donne."""
    provider = provider_name or get_active_provider()
    if split_name == "evaluation":
        return EXTRACTIONS_DIR / f"{provider}_{strategy_name}_results.json"
    return EXTRACTIONS_DIR / f"{provider}_{strategy_name}_{split_name}_results.json"


def empty_extraction() -> dict:
    """Retourner une extraction vide conforme au schema cible."""
    return {
        field: ([] if field in LIST_FIELDS else None)
        for field in ALL_FIELDS
    }


ensure_directories()
