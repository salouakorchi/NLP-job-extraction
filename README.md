# Projet NLP - Extraction d'informations depuis des offres d'emploi

Ce projet universitaire construit un pipeline complet pour extraire automatiquement des informations structurees, au format JSON, depuis des offres d'emploi en texte brut. Le dataset de travail est le fichier anglophone `data/raw/job_offers.csv`, issu de `jacob-hugging-face/job-descriptions`.

Les prompts sont adaptes aux offres en anglais : les noms de champs JSON restent ceux du projet, mais les valeurs extraites restent en anglais et conservent autant que possible le vocabulaire du texte source.

Le pipeline suit la methodologie demandee : chargement du dataset, pretraitement, separation experimentation/evaluation, verite terrain au format JSON, trois strategies de prompting, extraction LLM, post-traitement JSON, evaluation, analyse des erreurs et rapport final avec visualisations.

## Architecture du projet

```text
nlp_job_extraction/
|-- .env.example
|-- README.md
|-- requirements.txt
|-- config.py
|-- main.py
|-- web_app.py
|-- run_web.bat
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- experiment/
|   |-- evaluation/
|   `-- ground_truth/
|-- src/
|   |-- __init__.py
|   |-- dataset_loader.py
|   |-- preprocessing.py
|   |-- data_splitter.py
|   |-- prompts.py
|   |-- llm_extractor.py
|   |-- post_processor.py
|   |-- evaluator.py
|   |-- error_analyzer.py
|   `-- report_generator.py
|-- results/
|   |-- extractions/
|   |-- evaluations/
|   `-- visualizations/
|-- static/
|-- templates/
`-- notebooks/
    `-- analysis.ipynb
```

## Prerequis

- Python 3.10 ou plus
- Une cle API pour au moins un fournisseur LLM configure dans l'interface
- Le fichier `data/raw/job_offers.csv` ou un acces HuggingFace pour le recreer

## Installation

```bash
cd nlp_job_extraction
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Dans `.env`, renseigner au minimum le fournisseur choisi et sa cle :

```env
LLM_PROVIDER=groq
GROQ_API_KEY=votre_cle_groq
```

Fournisseurs disponibles dans l'interface :

| Fournisseur | Variable cle | Modele par defaut |
|---|---|---|
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Z.ai | `ZAI_API_KEY` | `glm-5.1` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/auto` |
| Together AI | `TOGETHER_API_KEY` | `openai/gpt-oss-20b` |
| Hugging Face | `HUGGINGFACE_API_KEY` | `openai/gpt-oss-20b` |
| Mistral AI | `MISTRAL_API_KEY` | `mistral-small-latest` |
| Cerebras | `CEREBRAS_API_KEY` | `gpt-oss-120b` |
| Custom compatible OpenAI | `CUSTOM_LLM_API_KEY` | a definir |

Le mode `custom` demande aussi :

```env
CUSTOM_LLM_ENDPOINT=https://votre-endpoint/chat/completions
CUSTOM_LLM_MODEL=votre-modele
```

## Utilisation pas a pas

```bash
python main.py --step load
python main.py --step preprocess
python main.py --step split
python main.py --step extract
python main.py --step evaluate
python main.py --step analyze
python main.py --step report
```

L'etape `load` utilise d'abord le CSV local `data/raw/job_offers.csv`. Si aucun CSV local n'est disponible, elle retombe sur HuggingFace et recree ce fichier.

## Interface web locale

L'interface web permet de choisir le fournisseur LLM, enregistrer la cle API, lancer les etapes, tester une offre en demo et consulter les resultats depuis le navigateur.

Sous Windows, double-cliquer sur :

```text
run_web.bat
```

Le script installe les dependances, cree `.env` s'il manque, lance le serveur Flask et ouvre automatiquement :

```text
http://127.0.0.1:5000/
```

Lancement manuel equivalent :

```bash
python -m pip install -r requirements.txt
python web_app.py
```

La commande suivante lance toutes les etapes automatiques :

```bash
python main.py --step all
```

Si la verite terrain n'existe pas encore, les etapes `evaluate` et `analyze` sont ignorees proprement et le rapport est genere avec des emplacements vides. Pour obtenir les metriques reelles, le fichier `data/ground_truth/ground_truth.json` doit exister.

## Choix du fournisseur LLM

Le fournisseur actif est lu depuis `LLM_PROVIDER`. Les resultats d'extraction sont sauvegardes avec le nom du fournisseur pour eviter d'ecraser les essais entre APIs :

```text
results/extractions/groq_zero_shot_results.json
results/extractions/gemini_zero_shot_results.json
results/extractions/openai_zero_shot_results.json
```

Les fournisseurs compatibles OpenAI utilisent un endpoint `chat/completions`. Gemini utilise son endpoint `generateContent`. Si une cle est invalide, le batch s'arrete rapidement et sauvegarde l'erreur sans faire des appels inutiles.

## Strategies de prompting

### zero_shot

Prompt minimal en anglais : il demande au modele d'extraire les informations importantes d'une offre anglophone et de retourner un JSON valide avec les champs du projet.

### constrained

Prompt strict en anglais : il contient le schema JSON complet, les types attendus, les valeurs autorisees pour `type_contrat`, les regles anti-hallucination et les conventions de normalisation adaptees aux offres anglaises.

### few_shot

Prompt avec exemples anglophones : il fournit trois cas proches du CSV, notamment sales/cloud, retail technology et data analytics.

## Exemple a tester dans l'interface

Collez uniquement le texte de l'offre dans la page `Demo`, pas un prompt complet :

```text
TechSolutions is a company specialized in data analytics.

We are hiring a Data Analyst for our office in Algiers.

Required skills:
- Python
- SQL
- Power BI
- Excel

Profile:
- Master's degree in Computer Science
- Minimum 3 years of experience

Contract type:
Full-time

Salary:
120000 DZD to 150000 DZD

Company:
TechSolutions
```

## Schema JSON cible

```json
{
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
  "avantages": ["string"]
}
```

Valeurs standard recommandees pour `type_contrat` :

```text
Full-time, Part-time, Contract, Temporary, Internship, Apprenticeship, Freelance, Other
```

## Verite terrain

Le projet utilise un fichier de reference deja structure pour evaluer les extractions LLM :

```text
data/ground_truth/ground_truth.json
```

Ce fichier contient les bonnes reponses pour le jeu d'evaluation. Chaque entree garde l'ID de l'offre, le texte source et l'objet JSON attendu dans la cle `ground_truth`.

## Metriques d'evaluation

- Champs simples : `exact_match` et `partial_match`.
- Champs listes : precision, rappel et F1.
- Score global : moyenne des `partial_match` pour les champs simples et des F1 pour les champs listes.

Interpretation rapide :

- `exact_match = 1` : valeur strictement correcte apres normalisation.
- `partial_match > 0` : recouvrement lexical partiel.
- `F1` : compromis entre precision et rappel pour les listes.

## Fichiers de resultats

Apres execution complete avec verite terrain :

```text
data/ground_truth/ground_truth.json
results/extractions/groq_zero_shot_results.json
results/extractions/groq_constrained_results.json
results/extractions/groq_few_shot_results.json
results/evaluations/comparison_table.csv
results/evaluations/zero_shot_error_report.md
results/evaluations/constrained_error_report.md
results/evaluations/few_shot_error_report.md
results/visualizations/heatmap_comparison.png
results/visualizations/global_f1_barplot.png
results/README_RESULTS.md
results/pipeline.log
```

## Notes de robustesse

- Les appels API sont proteges par `try/except` et retries exponentiels.
- Les extractions sont sauvegardees apres chaque offre.
- Les sorties JSON invalides sont reparees si possible.
- Le pipeline continue meme en cas d'echec ponctuel d'extraction.
- Le `random_state=42` est utilise pour garantir la reproductibilite.
