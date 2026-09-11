"""Extraction d'informations via plusieurs APIs LLM."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.prompts import PROMPT_VERSION, PromptingStrategy, get_prompt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class PermanentProviderError(RuntimeError):
    """Erreur fournisseur qui ne doit pas etre retentee."""


class RateLimitProviderError(RuntimeError):
    """Erreur de limite temporaire fournisseur avec delai de reprise."""

    def __init__(self, message: str, retry_after: float) -> None:
        """Initialiser l'erreur avec le delai recommande."""
        super().__init__(message)
        self.retry_after = retry_after


class LLMExtractor:
    """Appelle le fournisseur LLM actif pour extraire le schema JSON cible."""

    def __init__(self, provider_name: str | None = None) -> None:
        """Initialiser l'extracteur avec le fournisseur choisi."""
        self.provider = config.get_provider_info(provider_name)
        self.provider_name = self.provider["name"]
        self.last_error: str | None = None
        self.permanent_error = False
        self.client = self._is_configured()

        if not self.client:
            logger.warning("%s non configure: les extractions retourneront un JSON vide", self.provider_name)
        else:
            logger.info(
                "Fournisseur LLM initialise: %s (%s)",
                self.provider["label"],
                self.provider["model"],
            )

    def extract_single(
        self,
        job_description: str,
        strategy: PromptingStrategy,
    ) -> dict:
        """Extraire une offre avec une strategie de prompting donnee."""
        if not self.client:
            self.last_error = self._missing_config_message()
            return config.empty_extraction()

        prompt = get_prompt(strategy, job_description)
        last_error: Exception | None = None
        self.last_error = None

        for attempt in range(config.MAX_RETRIES):
            try:
                content = self._call_provider(prompt)
                try:
                    parsed = json.loads(content)
                except JSONDecodeError:
                    parsed = self.json_repair(content)

                if isinstance(parsed, dict):
                    self.last_error = None
                    logger.info(
                        "Extraction reussie avec %s / %s",
                        self.provider_name,
                        strategy.value,
                    )
                    return parsed

                self.last_error = "Reponse JSON non objet"
                logger.warning("Reponse JSON non objet pour %s", self.provider_name)
                return config.empty_extraction()
            except RateLimitProviderError as exc:
                last_error = exc
                self.last_error = str(exc)
                wait_seconds = max(exc.retry_after, float(2**attempt))
                logger.warning(
                    "Rate limit API tentative %s/%s (%s/%s), attente %.2fs: %s",
                    attempt + 1,
                    config.MAX_RETRIES,
                    self.provider_name,
                    strategy.value,
                    wait_seconds,
                    exc,
                )
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(wait_seconds)
                    continue
                logger.error("Rate limit persistant apres retries: %s", exc)
                return config.empty_extraction()
            except PermanentProviderError as exc:
                self.last_error = str(exc)
                self.permanent_error = True
                logger.error(
                    "Erreur permanente API (%s/%s): %s",
                    self.provider_name,
                    strategy.value,
                    exc,
                )
                return config.empty_extraction()
            except Exception as exc:
                last_error = exc
                self.last_error = str(exc)
                wait_seconds = 2**attempt
                logger.warning(
                    "Erreur API tentative %s/%s (%s/%s): %s",
                    attempt + 1,
                    config.MAX_RETRIES,
                    self.provider_name,
                    strategy.value,
                    exc,
                )
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(wait_seconds)

        logger.error("Echec extraction apres retries: %s", last_error)
        self.last_error = str(last_error) if last_error else "Echec extraction"
        return config.empty_extraction()

    def json_repair(self, text: str) -> dict:
        """Tenter de reparer une reponse JSON invalide."""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return config.empty_extraction()

            candidate = text[start : end + 1]
            candidate = candidate.replace("'", '"')
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            repaired = json.loads(candidate)
            return repaired if isinstance(repaired, dict) else config.empty_extraction()
        except Exception:
            logger.exception("Echec de reparation JSON")
            return config.empty_extraction()

    def extract_batch(
        self,
        df: pd.DataFrame,
        strategy: PromptingStrategy,
        output_path: str,
    ) -> list[dict]:
        """Extraire un lot d'offres avec reprise et sauvegarde progressive."""
        path = Path(output_path)
        existing_results = self._load_existing_results(path)
        results, done_ids = self._prepare_resume(existing_results)
        success_count = sum(1 for item in results if item.get("success"))
        error_count = len(results) - success_count
        durations: list[float] = []

        path.parent.mkdir(parents=True, exist_ok=True)
        df_ids = [self._row_id(row, idx) for idx, (_, row) in enumerate(df.iterrows())]
        initial_done = min(len([row_id for row_id in df_ids if row_id in done_ids]), len(df))

        progress_disabled = not sys.stderr.isatty()
        with tqdm(
            total=len(df),
            initial=initial_done,
            desc=f"{self.provider_name}:{strategy.value}",
            disable=progress_disabled,
            dynamic_ncols=True,
            ascii=True,
        ) as progress:
            for idx, (_, row) in enumerate(df.iterrows()):
                row_id = self._row_id(row, idx)
                if row_id in done_ids:
                    continue

                text = str(row.get("job_description_clean") or row.get("job_description") or "")
                prompt_source = self._build_prompt_source(row, text)
                start_time = time.time()
                error_message = None

                if self.permanent_error:
                    extraction = config.empty_extraction()
                    success = False
                    error_message = self.last_error or "Erreur permanente fournisseur"
                else:
                    try:
                        extraction = self.extract_single(prompt_source, strategy)
                        error_message = self.last_error
                        success = error_message is None
                    except Exception as exc:
                        logger.exception("Extraction impossible pour id %s", row_id)
                        extraction = config.empty_extraction()
                        success = False
                        error_message = str(exc)

                duration = time.time() - start_time
                durations.append(duration)
                if success:
                    success_count += 1
                else:
                    error_count += 1

                results.append(
                    {
                        "id": row_id,
                        "titre_original": self._clean_metadata(row.get("titre_original")),
                        "entreprise_originale": self._clean_metadata(row.get("entreprise_originale")),
                        "job_description": text,
                        "extraction": extraction,
                        "success": success,
                        "error": error_message,
                        "provider": self.provider_name,
                        "model": self.provider.get("model"),
                        "strategy": strategy.value,
                        "split": row.get("split"),
                        "prompt_version": PROMPT_VERSION,
                    }
                )
                path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                avg_time = sum(durations) / len(durations) if durations else 0.0
                if not progress_disabled:
                    progress.set_postfix(
                        success=success_count,
                        errors=error_count,
                        avg=f"{avg_time:.2f}s",
                    )
                progress.update(1)

                if self.client and not self.permanent_error:
                    time.sleep(config.API_DELAY)

        self._display_batch_stats(strategy.value, len(results), success_count, error_count, durations)
        return results

    def save_results(self, results: list, strategy_name: str) -> None:
        """Sauvegarder les resultats d'une strategie dans le dossier standard."""
        output_path = config.extraction_results_path(strategy_name, provider_name=self.provider_name)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Resultats sauvegardes dans %s", output_path)

    def _call_provider(self, prompt: str) -> str:
        """Router l'appel HTTP vers le fournisseur configure."""
        if self.provider["api_type"] == "gemini":
            return self._call_gemini(prompt)
        return self._call_openai_compatible(prompt)

    def _build_prompt_source(self, row: pd.Series, description: str) -> str:
        """Ajouter les metadonnees du CSV au texte envoye au LLM."""
        title = self._clean_metadata(row.get("titre_original"))
        company = self._clean_metadata(row.get("entreprise_originale"))
        parts = []
        if title or company:
            parts.append("CSV metadata:")
            if title:
                parts.append(f"Job title: {title}")
            if company:
                parts.append(f"Company: {company}")
            parts.append("")
        parts.append("Job description:")
        parts.append(description)
        return "\n".join(parts)

    def _clean_metadata(self, value: Any) -> str | None:
        """Nettoyer une valeur de metadonnee CSV avant prompting."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        text = str(value).strip()
        return text or None

    def _call_openai_compatible(self, prompt: str) -> str:
        """Appeler un endpoint compatible Chat Completions."""
        payload: dict[str, Any] = {
            "model": self.provider["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000,
            "stream": False,
        }
        if self.provider.get("supports_json_mode"):
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.provider['api_key']}",
            "Content-Type": "application/json",
        }
        if self.provider_name == "openrouter":
            headers["HTTP-Referer"] = "http://127.0.0.1:5000"
            headers["X-Title"] = "NLP Job Extraction"

        response = requests.post(
            self.provider["endpoint"],
            headers=headers,
            json=payload,
            timeout=90,
        )
        self._raise_for_provider_error(response)
        data = response.json()
        return self._extract_chat_content(data)

    def _call_gemini(self, prompt: str) -> str:
        """Appeler l'API Gemini generateContent."""
        endpoint = (
            f"{self.provider['endpoint'].rstrip('/')}/models/"
            f"{self.provider['model']}:generateContent"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            endpoint,
            headers={
                "x-goog-api-key": self.provider["api_key"],
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        self._raise_for_provider_error(response)
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Format de reponse Gemini inattendu: {data}") from exc

    def _extract_chat_content(self, data: dict) -> str:
        """Extraire le texte assistant d'une reponse chat completions."""
        try:
            content = data["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Format de reponse chat inattendu: {data}") from exc

        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    chunks.append(str(item.get("text") or item.get("content") or ""))
                else:
                    chunks.append(str(item))
            return "".join(chunks)
        return str(content or "")

    def _raise_for_provider_error(self, response: requests.Response) -> None:
        """Transformer une erreur HTTP fournisseur en message lisible."""
        if response.status_code < 400:
            return
        try:
            error_data = response.json()
        except ValueError:
            error_data = response.text[:500]
        error = (
            f"{self.provider['label']} HTTP {response.status_code}: {error_data}"
        )
        if response.status_code == 429:
            raise RateLimitProviderError(
                error,
                retry_after=self._retry_after_seconds(response, error_data),
            )
        if response.status_code in {401, 403}:
            raise PermanentProviderError(error)
        raise RuntimeError(error)

    def _retry_after_seconds(self, response: requests.Response, error_data: Any) -> float:
        """Extraire le delai recommande avant retry pour une erreur 429."""
        header_value = response.headers.get("Retry-After")
        if header_value:
            try:
                return max(float(header_value), 1.0)
            except ValueError:
                logger.debug("Header Retry-After non numerique: %s", header_value)

        message = ""
        if isinstance(error_data, dict):
            error_payload = error_data.get("error", {})
            if isinstance(error_payload, dict):
                message = str(error_payload.get("message", ""))
            else:
                message = str(error_payload)
        else:
            message = str(error_data)

        match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
        if match:
            return max(float(match.group(1)) + 0.75, 1.0)
        return 5.0

    def _is_configured(self) -> bool:
        """Verifier que le fournisseur a une cle, un modele et un endpoint."""
        return bool(
            self.provider.get("api_key")
            and self.provider.get("model")
            and self.provider.get("endpoint")
        )

    def _missing_config_message(self) -> str:
        """Construire un message de configuration manquante."""
        missing = []
        if not self.provider.get("api_key"):
            missing.append(self.provider["api_key_env"])
        if not self.provider.get("model"):
            missing.append(self.provider["model_env"])
        if not self.provider.get("endpoint"):
            missing.append(self.provider.get("endpoint_env", "endpoint"))
        return f"Configuration manquante pour {self.provider['label']}: {', '.join(missing)}"

    def _load_existing_results(self, path: Path) -> list[dict]:
        """Charger un fichier de resultats partiel s'il existe."""
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except JSONDecodeError:
            logger.warning("Fichier de resultats invalide, reprise depuis zero: %s", path)
            return []

    def _prepare_resume(self, existing_results: list[dict]) -> tuple[list[dict], set[int]]:
        """Determiner les resultats deja acquis et ceux a retenter."""
        if not self.client:
            done_ids = {
                self._safe_result_id(item)
                for item in existing_results
                if self._safe_result_id(item) is not None
            }
            return existing_results, done_ids

        successful_results = [
            item
            for item in existing_results
            if isinstance(item, dict)
            and item.get("success")
            and item.get("provider", self.provider_name) == self.provider_name
            and item.get("prompt_version") == PROMPT_VERSION
        ]
        done_ids = {
            self._safe_result_id(item)
            for item in successful_results
            if self._safe_result_id(item) is not None
        }
        return successful_results, done_ids

    def _row_id(self, row: pd.Series, fallback: int) -> int:
        """Recuperer l'identifiant stable d'une ligne."""
        try:
            return int(row.get("id", fallback))
        except (TypeError, ValueError):
            return int(fallback)

    def _safe_result_id(self, item: dict) -> int | None:
        """Convertir l'identifiant d'un resultat sauvegarde."""
        if not isinstance(item, dict):
            return None
        try:
            return int(item.get("id"))
        except (TypeError, ValueError):
            return None

    def _display_batch_stats(
        self,
        strategy_name: str,
        total: int,
        success_count: int,
        error_count: int,
        durations: list[float],
    ) -> None:
        """Afficher les statistiques finales d'un lot."""
        avg_time = sum(durations) / len(durations) if durations else 0.0
        table = Table(title=f"Extraction {self.provider_name} / {strategy_name}")
        table.add_column("Mesure", style="cyan")
        table.add_column("Valeur", style="green", justify="right")
        table.add_row("Modele", str(self.provider.get("model")))
        table.add_row("Total sauvegarde", str(total))
        table.add_row("Succes", str(success_count))
        table.add_row("Erreurs", str(error_count))
        table.add_row("Temps moyen", f"{avg_time:.2f}s")
        console.print(table)


if __name__ == "__main__":
    demo_df = pd.DataFrame(
        {
            "id": [0, 1],
            "titre_original": ["Data Analyst", "Digital Marketing Manager"],
            "entreprise_originale": ["Northwind Analytics", "BrightWave Media"],
            "job_description_clean": [
                "Northwind Analytics is hiring a Data Analyst in Chicago. Requirements include Python, SQL, Power BI, Excel, and at least 3 years of experience. Full-time position.",
                "BrightWave Media is looking for a Digital Marketing Manager in Austin. Skills include SEO, paid search, Google Analytics, and campaign reporting. Contract role.",
            ],
        }
    )
    extractor = LLMExtractor()
    output = config.extraction_results_path("zero_shot", "demo", extractor.provider_name)
    extractor.extract_batch(demo_df, PromptingStrategy.ZERO_SHOT, str(output))
