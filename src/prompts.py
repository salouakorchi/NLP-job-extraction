"""Prompting strategies optimized for English job offer extraction."""

from __future__ import annotations

import json
import logging
import sys
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

PROMPT_VERSION = "english_job_extraction_v3_compact"


class PromptingStrategy(Enum):
    """Enumere les trois strategies de prompting comparees."""

    ZERO_SHOT = "zero_shot"
    CONSTRAINED = "constrained"
    FEW_SHOT = "few_shot"


def _schema() -> str:
    """Retourner le schema JSON cible formate."""
    return json.dumps(config.JSON_SCHEMA_FIELDS, ensure_ascii=False, indent=2)


def _empty_output_template() -> str:
    """Retourner un exemple vide conforme au schema."""
    return json.dumps(config.empty_extraction(), ensure_ascii=False, indent=2)


def _allowed_contracts() -> str:
    """Retourner les valeurs normalisees de type de contrat."""
    return ", ".join(config.TYPE_CONTRAT_VALUES)


def _rules(compact: bool = False) -> str:
    """Regles communes de haute precision."""
    base = [
        "Return ONLY one valid JSON object, with exactly the 12 project fields.",
        "Use CSV metadata when present: Job title -> titre_poste, Company -> entreprise.",
        "Input is English; keep extracted values in English.",
        "Do not invent. Unknown string fields = null; unknown list fields = [].",
        "List fields must always be arrays, never strings or null.",
        "If cleaned text lost a number, keep the visible phrase, e.g. 'years preferred'.",
        f"type_contrat must be explicit and one of: {_allowed_contracts()}. Do not infer it from sales/team/business wording.",
        "Separate hard skills from soft skills.",
        "Keep items short, specific, deduplicated.",
    ]
    if compact:
        return "\n".join(f"- {rule}" for rule in base)

    field_notes = [
        "titre_poste: exact title, prefer metadata.",
        "entreprise: exact company, prefer metadata.",
        "localisation: explicit city/state/country/remote/hybrid only.",
        "experience_requise: years, seniority, no-experience statement, or preferred background.",
        "formation_requise: degree, education level, diploma or certification.",
        "competences_techniques: tools, products, platforms, software, languages, methods, operational/domain skills, certifications.",
        "competences_douces: communication, leadership, teamwork, relationship building, adaptability, curiosity, motivation, problem solving.",
        "langues_requises: explicit human languages only.",
        "salaire: salary, pay, hourly rate, commission or compensation package only.",
        "secteur_activite: concise business domain.",
        "avantages: benefits/perks only, not responsibilities.",
    ]
    return "\n".join(f"- {rule}" for rule in base + field_notes)


def get_zero_shot_prompt(job_description: str) -> str:
    """Generer un prompt zero-shot compact et robuste."""
    return f"""
You extract structured data from English job postings.

Rules:
{_rules(compact=True)}

Output JSON shape:
{_empty_output_template()}

Job posting:
{job_description}
""".strip()


def get_constrained_prompt(job_description: str) -> str:
    """Generer un prompt strict avec schema et regles detaillees."""
    return f"""
You are a high-precision JSON extraction engine for English job postings.

Target schema:
{_schema()}

Rules:
{_rules(compact=False)}

Before answering, verify:
- all 12 fields exist;
- list fields are arrays;
- title/company use CSV metadata if available;
- no missing contract, salary, location, education or years were invented.

Job posting:
{job_description}
""".strip()


def get_few_shot_prompt(job_description: str) -> str:
    """Generer un prompt few-shot compact proche du dataset."""
    return f"""
Extract English job postings into the exact project JSON. Output only JSON.

Rules:
{_rules(compact=True)}

Example A
Input:
CSV metadata:
Job title: Apple Solutions Consultant
Company: Apple
Job description:
as an asc you will grow mind and market share of apple products while building longterm relationships. maintain visual merchandising. qualifications include passion to help people, excellent communication skills, years preferred working in a dynamic sales andor results driven environment, and ability to encourage a partner team.
Output:
{{
  "titre_poste": "Apple Solutions Consultant",
  "entreprise": "Apple",
  "localisation": null,
  "type_contrat": null,
  "experience_requise": "years preferred working in a dynamic sales and/or results driven environment",
  "formation_requise": null,
  "competences_techniques": ["Apple products", "visual merchandising", "sales", "customer loyalty development"],
  "competences_douces": ["communication skills", "passion to help people", "building long-term relationships", "encourage a partner team"],
  "langues_requises": [],
  "salaire": null,
  "secteur_activite": "consumer electronics retail",
  "avantages": []
}}

Example B
Input:
CSV metadata:
Job title: Web Developer
Company: TrackFive
Job description:
web developer builds layouts from psd files, dynamic web apps using php mysql frameworks, rest services and thirdparty apis. troubleshoot bugs and compatibility issues. skills include html csssass php lamp stack javascript jquery. position can be hybrid inoffice or fully remote.
Output:
{{
  "titre_poste": "Web Developer",
  "entreprise": "TrackFive",
  "localisation": "hybrid, in-office or fully remote",
  "type_contrat": null,
  "experience_requise": null,
  "formation_requise": null,
  "competences_techniques": ["web development", "PSD files", "PHP", "MySQL", "frameworks", "REST services", "third-party APIs", "bugs", "compatibility issues", "HTML", "CSS/Sass", "LAMP stack", "JavaScript", "jQuery"],
  "competences_douces": ["problem solving"],
  "langues_requises": [],
  "salaire": null,
  "secteur_activite": "recruiting technology",
  "avantages": ["hybrid option", "in-office option", "fully remote option"]
}}

Example C
Input:
CSV metadata:
Job title: Bank Teller
Company: The Bank of Grain Valley
Job description:
bank teller kansas city mo monday through friday no weekends full time and part time options. experience not necessary will train. locally owned financial institution family atmosphere no sale quotas no pressure. benefits for full time employees include k matching dental health life and vision insurance.
Output:
{{
  "titre_poste": "Bank Teller",
  "entreprise": "The Bank of Grain Valley",
  "localisation": "Kansas City, MO",
  "type_contrat": "Full-time",
  "experience_requise": "Experience not necessary; will train",
  "formation_requise": null,
  "competences_techniques": ["bank teller", "financial institution operations"],
  "competences_douces": ["customer service", "no sale quotas", "no pressure"],
  "langues_requises": [],
  "salaire": null,
  "secteur_activite": "banking",
  "avantages": ["401(k) matching", "dental insurance", "health insurance", "life insurance", "vision insurance"]
}}

Target input:
{job_description}
""".strip()


def get_prompt(strategy: PromptingStrategy | str, job_description: str) -> str:
    """Dispatcher vers la fonction de prompt associee a la strategie."""
    if isinstance(strategy, str):
        strategy = PromptingStrategy(strategy)

    if strategy == PromptingStrategy.ZERO_SHOT:
        return get_zero_shot_prompt(job_description)
    if strategy == PromptingStrategy.CONSTRAINED:
        return get_constrained_prompt(job_description)
    if strategy == PromptingStrategy.FEW_SHOT:
        return get_few_shot_prompt(job_description)
    raise ValueError(f"Strategie inconnue: {strategy}")


if __name__ == "__main__":
    sample = "CSV metadata:\nJob title: Data Engineer\nCompany: DataLab\n\nJob description:\nDataLab is hiring a full-time Data Engineer in Austin. Python, Spark, AWS and 4 years of experience required."
    for prompt_strategy in PromptingStrategy:
        console.print(
            Panel(
                get_prompt(prompt_strategy, sample),
                title=prompt_strategy.value,
                border_style="cyan",
            )
        )
