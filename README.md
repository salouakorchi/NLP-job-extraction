# NLP Job Extraction with LLMs

> **NLP · Large Language Models · Prompt Engineering · Information Extraction**

Projet universitaire de **Traitement Automatique du Langage Naturel (NLP)** visant à extraire automatiquement des informations structurées à partir d'offres d'emploi en anglais.

Le système transforme une offre en texte brut en un objet **JSON structuré** grâce à plusieurs stratégies de prompting et permet de comparer leurs performances.

## 🎯 Objectif

Extraire automatiquement les informations essentielles d'une offre d'emploi :

* **Poste**
* **Entreprise**
* **Localisation**
* **Type de contrat**
* **Expérience requise**
* **Formation**
* **Compétences techniques**
* **Compétences comportementales**
* **Langues**
* **Salaire**
* **Secteur d'activité**
* **Avantages**

## 🧠 Méthodologie

Le projet implémente un pipeline complet :

```text
Job Offers
    ↓
Data Loading
    ↓
Preprocessing
    ↓
Experiment / Evaluation Split
    ↓
Prompt Engineering
    ↓
LLM Extraction
    ↓
JSON Post-processing
    ↓
Evaluation
    ↓
Error Analysis
    ↓
Visualization & Reporting
```

### Stratégies de prompting

**Zero-Shot**
Extraction directe sans exemple.

**Constrained Prompting**
Utilisation d'un schéma JSON strict, de règles de normalisation et de contraintes anti-hallucination.

**Few-Shot Prompting**
Utilisation d'exemples afin de guider le modèle vers le format et le comportement attendus.

## 📊 Évaluation

Une **ground truth** structurée permet de comparer automatiquement les réponses produites par les LLM.

### Métriques

* **Exact Match**
* **Partial Match**
* **Precision**
* **Recall**
* **F1-score**

Les résultats détaillés sont disponibles dans :

```text
results/evaluations/
```

Les visualisations sont disponibles dans :

```text
results/visualizations/
```

## 🛠️ Technologies

```text
Python
NLP
LLMs
Prompt Engineering
Hugging Face
Pandas
JSON
REST APIs
Flask
```

## 🤖 Fournisseurs LLM

Le pipeline est conçu pour fonctionner avec plusieurs fournisseurs :

**Groq · OpenAI · Google Gemini · DeepSeek · Mistral AI · OpenRouter · Together AI · Hugging Face**

Le fournisseur utilisé peut être configuré via :

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
```

## 📁 Architecture

```text
NLP-job-extraction/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── experiment/
│   ├── evaluation/
│   └── ground_truth/
│
├── results/
│   ├── extractions/
│   ├── evaluations/
│   └── visualizations/
│
├── src/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── data_splitter.py
│   ├── prompts.py
│   ├── llm_extractor.py
│   ├── post_processor.py
│   ├── evaluator.py
│   ├── error_analyzer.py
│   └── report_generator.py
│
├── notebooks/
├── templates/
├── static/
│
├── main.py
├── web_app.py
├── config.py
└── requirements.txt
```

## 🚀 Installation

```bash
git clone https://github.com/salouakorchi/NLP-job-extraction.git
cd NLP-job-extraction

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Créer ensuite un fichier `.env` contenant la clé API du fournisseur choisi.

> **Important:** ne jamais publier le fichier `.env` contenant une clé API. Il est exclu du repository via `.gitignore`.

## ▶️ Exécution

### Pipeline complet

```bash
python main.py --step all
```

### Exécution étape par étape

```bash
python main.py --step load
python main.py --step preprocess
python main.py --step split
python main.py --step extract
python main.py --step evaluate
python main.py --step analyze
python main.py --step report
```

## 🌐 Interface web

Le projet comprend également une interface **Flask** permettant de tester une offre d'emploi et de consulter les résultats directement depuis un navigateur.

```bash
python web_app.py
```

Puis accéder à :

```text
http://127.0.0.1:5000/
```

Sous Windows, le lancement peut également être effectué avec :

```text
run_web.bat
```

## 📚 Dataset

Le projet utilise un dataset anglophone de descriptions d'emploi issu de :

**jacob-hugging-face/job-descriptions**

Les données sont prétraitées puis séparées entre les jeux d'expérimentation et d'évaluation.

## 📌 Résultats

Le repository contient :

* les extractions JSON produites par les LLM ;
* les métriques détaillées ;
* les comparaisons entre stratégies de prompting ;
* les rapports d'erreurs ;
* les visualisations ;
* les statistiques du dataset.

```text
results/
├── extractions/
├── evaluations/
└── visualizations/
```

## 🎓 Contexte académique

Projet réalisé dans le cadre de la formation d'**Ingénierie Informatique — spécialité Intelligence Artificielle**.

---

**NLP · LLMs · Prompt Engineering · Information Extraction · Python**
