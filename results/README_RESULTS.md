# Rapport final - Extraction d'informations d'offres d'emploi

## Resume executif

Ce projet construit un pipeline NLP complet pour convertir des offres d'emploi anglophones en objets JSON structures. Il compare trois strategies de prompting adaptees au dataset anglais `job_offers.csv` et mesure leurs performances sur une verite terrain annotee manuellement.

## Dataset

- Source : `data/raw/job_offers.csv` avec fallback HuggingFace `jacob-hugging-face/job-descriptions`
- Langue des offres : anglais (`en`)
- Nombre d'echantillons charges : 5
- Longueur moyenne des descriptions : 2483.20
- Longueur min/max : 828 / 3205
- Nombre d'exemples de verite terrain evalues : 1
- Fournisseur LLM actif : Groq / llama-3.3-70b-versatile

## Schema JSON utilise

```json
{
  "titre_poste": "string",
  "entreprise": "string",
  "localisation": "string",
  "type_contrat": "string",
  "experience_requise": "string",
  "formation_requise": "string",
  "competences_techniques": [
    "string"
  ],
  "competences_douces": [
    "string"
  ],
  "langues_requises": [
    "string"
  ],
  "salaire": "string or null",
  "secteur_activite": "string",
  "avantages": [
    "string"
  ]
}
```

## Strategies de prompting

- `zero_shot` : consigne courte en anglais, sortie JSON demandee.
- `constrained` : schema explicite, regles anti-hallucination, categories de contrat anglaises et conservation du vocabulaire source.
- `few_shot` : trois exemples anglophones proches du dataset avant l'offre cible.

Categories standard de `type_contrat` : `Full-time`, `Part-time`, `Contract`, `Temporary`, `Internship`, `Apprenticeship`, `Freelance`, `Other`.

## Notes methodologiques sur les scores

- Les champs absents des deux cotes (`null`/`null` ou `[]`/`[]`) comptent comme corrects.
- Les listes de competences utilisent un matching lexical fuzzy pour accepter les variantes proches comme `excellent communication skills` et `communication skills`.
- L'evaluation utilise les sorties LLM normalisees, sans enrichissement automatique depuis le texte source.
- `overall_score` moyenne tous les champs, y compris les absences correctement predites.
- `informative_score` moyenne seulement les champs non vides dans le ground truth ; c'est le score le plus utile pour juger l'extraction effective.
- Avec un seul exemple de verite terrain, les scores indiquent seulement la performance sur ce run, pas une performance generale definitive.
- Diagnostic detaille : [performance_diagnosis.md](evaluations/performance_diagnosis.md).

## Tableau comparatif des performances

| Strategie | Overall score | Informative score |
|---|---:|---:|
| zero_shot | 0.728 | 0.456 |
| constrained | 0.756 | 0.511 |
| few_shot | 0.979 | 0.958 |

## Visualisations

- [Heatmap comparative](visualizations/heatmap_comparison.png)
- [Score global par strategie](visualizations/global_f1_barplot.png)

## Analyse des erreurs par strategie

### zero_shot

Champs les plus difficiles :
- experience_requise : 1 erreurs
- secteur_activite : 1 erreurs
- competences_techniques : 1 erreurs

Rapport detaille : [rapport zero_shot](evaluations/zero_shot_error_report.md)

### constrained

Champs les plus difficiles :
- experience_requise : 1 erreurs
- secteur_activite : 1 erreurs
- competences_techniques : 1 erreurs

Rapport detaille : [rapport constrained](evaluations/constrained_error_report.md)

### few_shot

Champs les plus difficiles :
- competences_douces : 1 erreurs

Rapport detaille : [rapport few_shot](evaluations/few_shot_error_report.md)

## Conclusion et recommandations

Sur le jeu d'evaluation actuel, le score informatif montre mieux les differences entre les strategies que le score global, car plusieurs champs de l'offre sont volontairement absents. Il faut confirmer les resultats sur un echantillon annote plus large.

Recommandations : renforcer la normalisation post-traitement, ajouter des exemples negatifs, enrichir la verite terrain avec des synonymes acceptables, et evaluer sur un corpus plus large.

## Pistes futures

- Ajouter une evaluation semantique pour les competences proches mais non identiques.
- Tester plusieurs fournisseurs LLM et plusieurs temperatures.
- Ajouter un editeur de verite terrain dedie si le corpus doit etre etendu.
- Ajouter des tests unitaires dedies aux fonctions de normalisation et d'evaluation.