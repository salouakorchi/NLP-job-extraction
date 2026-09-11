# Rapport d'erreurs - zero_shot

## Tableau des erreurs par champ

| Champ | Correct | Champ manquant | Valeur incorrecte | Extraction partielle | Information hallucinee | Format incorrect |
|---|---:|---:|---:|---:|---:|---:|
| titre_poste | 1 | 0 | 0 | 0 | 0 | 0 |
| entreprise | 1 | 0 | 0 | 0 | 0 | 0 |
| localisation | 1 | 0 | 0 | 0 | 0 | 0 |
| type_contrat | 1 | 0 | 0 | 0 | 0 | 0 |
| experience_requise | 0 | 0 | 0 | 1 | 0 | 0 |
| formation_requise | 1 | 0 | 0 | 0 | 0 | 0 |
| salaire | 1 | 0 | 0 | 0 | 0 | 0 |
| secteur_activite | 0 | 0 | 1 | 0 | 0 | 0 |
| competences_techniques | 0 | 1 | 0 | 0 | 0 | 0 |
| competences_douces | 0 | 0 | 0 | 1 | 0 | 0 |
| langues_requises | 1 | 0 | 0 | 0 | 0 | 0 |
| avantages | 1 | 0 | 0 | 0 | 0 | 0 |

## Distribution des categories d'erreurs

| Categorie | Nombre |
|---|---:|
| correct | 8 |
| extraction_partielle | 2 |
| valeur_incorrecte | 1 |
| champ_manquant | 1 |

## Top 5 exemples les plus difficiles

- ID 1 : 4 erreurs, categories {'extraction_partielle': 2, 'valeur_incorrecte': 1, 'champ_manquant': 1}

## Analyse des causes probables

- Les champs peu explicites dans les offres generent souvent des `champ_manquant` ou des `valeur_incorrecte`.
- Les listes de competences sont sensibles aux variantes lexicales, abreviations et niveaux de granularite.
- Les informations absentes mais plausibles peuvent conduire a des `information_hallucinee` si le prompt n'est pas assez contraignant.

## Recommandations d'amelioration

- Renforcer les consignes anti-hallucination et les exemples negatifs.
- Ajouter une normalisation metier plus riche pour les contrats, langues et competences.
- Enrichir la verite terrain avec des synonymes acceptables pour l'evaluation.