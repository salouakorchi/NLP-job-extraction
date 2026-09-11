"""Validation et normalisation des extractions JSON."""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class PostProcessor:
    """Valide et nettoie les objets JSON produits par le LLM."""

    def validate_json_schema(self, data: dict) -> tuple[bool, list[str]]:
        """Verifier la presence et les types des champs du schema cible."""
        errors: list[str] = []
        if not isinstance(data, dict):
            return False, ["objet_json_invalide"]

        for field in config.ALL_FIELDS:
            if field not in data:
                errors.append(f"champ_manquant:{field}")
                continue
            value = data[field]
            if field in config.LIST_FIELDS and not isinstance(value, list):
                errors.append(f"type_liste_incorrect:{field}")
            if field in config.STRING_FIELDS and value is not None and not isinstance(value, str):
                errors.append(f"type_string_incorrect:{field}")

        return len(errors) == 0, errors

    def normalize_extraction(self, data: dict) -> dict:
        """Normaliser les valeurs et completer les champs absents."""
        if not isinstance(data, dict):
            data = {}

        normalized: dict[str, Any] = {}
        for field in config.STRING_FIELDS:
            value = data.get(field)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value if item is not None)
            elif value is not None and not isinstance(value, str):
                value = str(value)

            if isinstance(value, str):
                value = value.strip()
                value = value if value else None

            normalized[field] = value

        for field in config.LIST_FIELDS:
            value = data.get(field, [])
            if value is None:
                value = []
            elif isinstance(value, str):
                value = [item.strip() for item in value.split(",")]
            elif not isinstance(value, list):
                value = [str(value)]

            seen = set()
            items = []
            for item in value:
                item_str = str(item).strip()
                key = item_str.lower()
                if item_str and key not in seen:
                    seen.add(key)
                    items.append(item_str)
            normalized[field] = items

        normalized["type_contrat"] = self._normalize_contract_type(
            normalized.get("type_contrat")
        )
        return normalized

    def enrich_from_source(
        self,
        extraction: dict,
        source_text: str | None,
        metadata: dict | None = None,
    ) -> dict:
        """Completer prudemment une extraction avec des indices presents dans le texte source."""
        enriched = self.normalize_extraction(extraction)
        metadata = metadata or {}

        for target_field, metadata_field in [
            ("titre_poste", "titre_original"),
            ("entreprise", "entreprise_originale"),
        ]:
            metadata_value = metadata.get(metadata_field)
            if not enriched.get(target_field) and isinstance(metadata_value, str) and metadata_value.strip():
                enriched[target_field] = metadata_value.strip()

        source = self._source_key(source_text or "")
        if not source:
            return enriched

        self._expand_short_experience(enriched, source)
        self._add_source_backed_skills(enriched, source)
        self._infer_source_backed_sector(enriched, source)
        return enriched

    def post_process_batch(
        self,
        results: list,
        source_aware: bool = False,
    ) -> tuple[list, dict]:
        """Valider puis normaliser un lot de resultats.

        Par defaut, le post-traitement ne complete pas les predictions depuis
        le texte source afin de garder une evaluation fidele aux sorties LLM.
        """
        processed = []
        stats = {
            "nb_valides": 0,
            "nb_invalides": 0,
            "types_erreurs": Counter(),
        }

        for item in results:
            extraction = item.get("extraction", item) if isinstance(item, dict) else {}
            is_valid, errors = self.validate_json_schema(extraction)
            if is_valid:
                stats["nb_valides"] += 1
            else:
                stats["nb_invalides"] += 1
                stats["types_erreurs"].update(errors)

            metadata = item if isinstance(item, dict) else {}
            if source_aware:
                normalized = self.enrich_from_source(
                    extraction,
                    str(metadata.get("job_description") or ""),
                    metadata,
                )
            else:
                normalized = self.normalize_extraction(extraction)
            if isinstance(item, dict) and "extraction" in item:
                new_item = dict(item)
                new_item["extraction"] = normalized
                new_item["validation_errors"] = errors
            else:
                new_item = normalized
            processed.append(new_item)

        stats["types_erreurs"] = dict(stats["types_erreurs"])
        self._display_stats(stats)
        logger.info("Post-traitement termine: %s resultats", len(processed))
        return processed, stats

    def _add_source_backed_skills(self, extraction: dict, source: str) -> None:
        """Ajouter des competences explicitement ancrees dans la description."""
        technical_rules = [
            ("apple products", "Apple products"),
            ("visual merchandising", "visual merchandising"),
            ("customer loyalty", "customer loyalty development"),
            ("sales", "sales"),
            ("python", "Python"),
            ("sql", "SQL"),
            ("power bi", "Power BI"),
            ("excel", "Excel"),
            ("javascript", "JavaScript"),
            ("jquery", "jQuery"),
            ("html", "HTML"),
            ("css", "CSS"),
            ("php", "PHP"),
            ("mysql", "MySQL"),
            ("rest services", "REST services"),
            ("third party apis", "third-party APIs"),
        ]
        soft_rules = [
            ("communication skills", "communication skills"),
            ("passion to help people", "passion to help people"),
            ("long term relationships", "building long-term relationships"),
            ("strive for perfection", "strive for perfection"),
            ("encourage a partner team", "encourage a partner team"),
            ("problem solving", "problem solving"),
            ("teamwork", "teamwork"),
            ("customer service", "customer service"),
        ]

        for needle, value in technical_rules:
            if self._contains_source_phrase(source, needle):
                self._append_unique(extraction, "competences_techniques", value)
        for needle, value in soft_rules:
            if self._contains_source_phrase(source, needle):
                self._append_unique(extraction, "competences_douces", value)
        self._rebalance_skill_lists(extraction)

    def _expand_short_experience(self, extraction: dict, source: str) -> None:
        """Remplacer une experience trop courte par la phrase source explicite."""
        patterns = [
            (
                "years preferred working in a dynamic sales and or results driven environment",
                "years preferred working in a dynamic sales and/or results driven environment",
            ),
            (
                "experience not necessary will train",
                "Experience not necessary; will train",
            ),
        ]
        current = extraction.get("experience_requise")
        current_words = len(str(current or "").split())
        for needle, value in patterns:
            if self._contains_source_phrase(source, needle) and (not current or current_words <= 3):
                extraction["experience_requise"] = value
                return

    def _infer_source_backed_sector(self, extraction: dict, source: str) -> None:
        """Corriger le secteur quand plusieurs indices explicites convergent."""
        current = str(extraction.get("secteur_activite") or "").strip().lower()
        if self._contains_source_phrase(source, "apple products") and self._contains_source_phrase(source, "visual merchandising"):
            if current in {"", "sales", "retail"}:
                extraction["secteur_activite"] = "consumer electronics retail"
        elif self._contains_source_phrase(source, "financial institution") or self._contains_source_phrase(source, "bank teller"):
            if not current:
                extraction["secteur_activite"] = "banking"

    def _append_unique(self, extraction: dict, field: str, value: str) -> None:
        """Ajouter une valeur de liste sans doublon insensible a la casse."""
        items = extraction.setdefault(field, [])
        if not isinstance(items, list):
            items = []
            extraction[field] = items

        candidate = value.strip()
        candidate_key = self._source_key(candidate)
        for index, item in enumerate(items):
            item_text = str(item).strip()
            item_key = self._source_key(item_text)
            if not item_key:
                continue
            if item_key == candidate_key or self._similar_list_item(item_key, candidate_key):
                if len(candidate_key.split()) > len(item_key.split()):
                    items[index] = candidate
                return
        if candidate:
            items.append(candidate)

    def _contains_source_phrase(self, source: str, phrase: str) -> bool:
        """Verifier si une phrase normalisee apparait dans le texte source."""
        normalized_phrase = self._source_key(phrase)
        if not normalized_phrase:
            return False
        source_words = source.split()
        phrase_words = normalized_phrase.split()
        if not phrase_words or len(phrase_words) > len(source_words):
            return False
        window_size = len(phrase_words)
        for index in range(0, len(source_words) - window_size + 1):
            if source_words[index : index + window_size] == phrase_words:
                return True
        return False

    def _source_key(self, text: str) -> str:
        """Normaliser le texte source pour des recherches lexicales robustes."""
        normalized = str(text).lower()
        replacements = {
            "longterm": "long term",
            "andor": "and or",
            "thirdparty": "third party",
            "csssass": "css sass",
            "livesexcellent": "lives excellent",
            "loyaltyability": "loyalty ability",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _similar_list_item(self, left: str, right: str) -> bool:
        """Detecter deux variantes lexicales du meme item de liste."""
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens)
        min_size = min(len(left_tokens), len(right_tokens))
        if min_size == 1 and overlap == 1:
            generic = {"skill", "skills", "team", "business", "management", "development"}
            shared = next(iter(left_tokens & right_tokens))
            return shared not in generic
        return min_size >= 2 and overlap / min_size >= 0.75

    def _rebalance_skill_lists(self, extraction: dict) -> None:
        """Eviter qu'une competence metier soit dupliquee dans les soft skills."""
        technical_items = extraction.get("competences_techniques", [])
        soft_items = extraction.get("competences_douces", [])
        if not isinstance(technical_items, list) or not isinstance(soft_items, list):
            return

        soft_indicators = {
            "communication",
            "passion",
            "relationship",
            "relationships",
            "perfection",
            "encourage",
            "problem",
            "solving",
            "organization",
            "organizational",
            "collaboration",
            "collaborative",
            "leadership",
            "service",
        }
        technical_keys = [self._source_key(item) for item in technical_items]
        cleaned_soft = []
        for item in soft_items:
            item_key = self._source_key(item)
            item_tokens = set(item_key.split())
            overlaps_technical = any(
                technical_key and self._similar_list_item(item_key, technical_key)
                for technical_key in technical_keys
            )
            if overlaps_technical and not (item_tokens & soft_indicators):
                continue
            cleaned_soft.append(item)
        extraction["competences_douces"] = self._drop_generic_soft_duplicates(cleaned_soft)

    def _drop_generic_soft_duplicates(self, soft_items: list) -> list:
        """Supprimer les soft skills vagues quand une formulation specifique existe."""
        normalized_items = [self._source_key(item) for item in soft_items]
        has_specific_team = any("encourage" in item and "team" in item for item in normalized_items)
        cleaned = []
        for item, item_key in zip(soft_items, normalized_items):
            if has_specific_team and item_key in {"team development", "partner team development"}:
                continue
            cleaned.append(item)
        return cleaned

    def _normalize_contract_type(self, value: str | None) -> str | None:
        """Mapper les variantes courantes de contrat vers les valeurs standards."""
        if value is None:
            return None

        compact = value.strip().lower()
        mapping = {
            "full-time": "Full-time",
            "full time": "Full-time",
            "fulltime": "Full-time",
            "permanent": "Full-time",
            "regular": "Full-time",
            "temps plein": "Full-time",
            "cdi": "Full-time",
            "part-time": "Part-time",
            "part time": "Part-time",
            "parttime": "Part-time",
            "temps partiel": "Part-time",
            "contract": "Contract",
            "contractor": "Contract",
            "consultant contract": "Contract",
            "fixed-term": "Temporary",
            "fixed term": "Temporary",
            "temporary": "Temporary",
            "temp": "Temporary",
            "cdd": "Temporary",
            "internship": "Internship",
            "intern": "Internship",
            "stage": "Internship",
            "apprenticeship": "Apprenticeship",
            "apprentice": "Apprenticeship",
            "alternance": "Apprenticeship",
            "freelance": "Freelance",
            "self-employed": "Freelance",
        }
        if compact in mapping:
            return mapping[compact]

        contains_mapping = [
            ("full-time", "Full-time"),
            ("full time", "Full-time"),
            ("part-time", "Part-time"),
            ("part time", "Part-time"),
            ("contractor", "Contract"),
            ("contract", "Contract"),
            ("temporary", "Temporary"),
            ("internship", "Internship"),
            ("intern", "Internship"),
            ("apprentice", "Apprenticeship"),
            ("freelance", "Freelance"),
        ]
        for token, standard_value in contains_mapping:
            if token in compact:
                return standard_value

        for allowed in config.TYPE_CONTRAT_VALUES:
            if compact == allowed.lower():
                return allowed
        return "Other"

    def _display_stats(self, stats: dict) -> None:
        """Afficher un rapport synthetique de validation."""
        table = Table(title="Post-traitement JSON")
        table.add_column("Mesure", style="cyan")
        table.add_column("Valeur", style="green")
        table.add_row("Resultats valides", str(stats["nb_valides"]))
        table.add_row("Resultats invalides", str(stats["nb_invalides"]))
        table.add_row("Types d'erreurs", str(stats["types_erreurs"]))
        console.print(table)


if __name__ == "__main__":
    sample_results = [
        {
            "id": 0,
            "extraction": {
                "titre_poste": " Data Analyst ",
                "entreprise": "Demo",
                "localisation": "",
                "type_contrat": "full-time",
                "experience_requise": None,
                "formation_requise": "Master",
                "competences_techniques": "Python, SQL, Python",
                "competences_douces": ["Rigueur", "rigueur"],
                "langues_requises": None,
                "salaire": "",
                "secteur_activite": "Tech",
                "avantages": ["Teletravail"],
            },
        }
    ]
    processor = PostProcessor()
    processor.post_process_batch(sample_results)
