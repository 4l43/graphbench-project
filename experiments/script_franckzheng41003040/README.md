# GraphBench — Réfutation automatique de conjectures en théorie des graphes

## Description

Ce projet implémente un système capable de réfuter automatiquement des conjectures en théorie des graphes via deux approches :

- **Partie 1** : Heuristique de recherche locale avec mutations
- **Partie 2** : Architecture FunSearch — évolution automatique de la fonction de score via LLM (Claude)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# Lancer tout le benchmark (60s par conjecture)
python main.py

# Limite de temps personnalisée
python main.py --time 30 (pour 30s par conjecture)

# Tester une conjecture spécifique
python main.py --conjecture 980

# Activer FunSearch (Partie 2, nécessite une clé API Claude)
python main.py --funsearch --funsearch-budget 300

# Valider les contre-exemples connus du benchmark
python main.py --validate

# Mode verbeux
python main.py --verbose --time 20
```

## Architecture

### Partie 1 — Heuristique simple

1. **Représentation** : graphes NetworkX (entiers comme labels)
2. **Génération initiale** : adaptée à la classe (connexe, arbre, sans griffe...)
3. **Score** : `violation(G) + bonus(G)` où violation > 0 = contre-exemple
4. **Mutations** :
   - Ajout/suppression d'arêtes
   - Ajout/suppression de sommets
   - Subdivision d'arête
   - Ajout de feuille / chemin / clique
   - Rewiring
5. **Réparation** : selon la classe (connexité, arbre couvrant, élimination de griffes)
6. **Sélection** : élitisme (top 50%) + diversification périodique

### Partie 2 — FunSearch

1. Pool de fonctions de score initialisé avec des fonctions seed
2. Évaluation de chaque fonction sur un sous-ensemble de conjectures
3. Le LLM (Claude) génère des variantes des meilleures fonctions
4. Les meilleures variantes remplacent les moins bonnes dans le pool
5. La meilleure fonction est utilisée pour la recherche principale

### Invariants supportés

| Notation | Nom | Complexité |
|----------|-----|-----------|
| n, m | ordre, taille | O(1) |
| diam, rad | diamètre, rayon | O(n²) |
| δ, Δ, avg | degrés | O(n) |
| density | densité | O(1) |
| t | triangles | O(n³) |
| ω | clique max | NP-hard (approché) |
| γ, γ_t | domination (totale) | NP-hard (greedy) |
| α | indépendance | NP-hard (approché) |
| τ | couverture | via α |
| i | domination indépendante | NP-hard (approché) |
| μ | couplage max | O(n³) |
| λ₁ | plus grande valeur propre | O(n³) |
| λ₂(L) | connectivité algébrique | O(n³) |
| ρ | plus grande val. propre distance | O(n³) |
| R, H, M₁, M₂ | indices topologiques | O(m) |
| prox, rem | proximité, éloignement | O(n²) |

## Score

Le score final est calculé selon :
- Si contre-exemple trouvé en `t` secondes : coût = `t`
- Sinon : coût = 120

**Score total = Σ coûts** (minimiser)

## Résultats

Les résultats sont sauvegardés dans `results/results.json` avec pour chaque conjecture :
- `found` : contre-exemple trouvé ?
- `graph6` : encodage graph6 du contre-exemple
- `violation` : valeur de la violation
- `time_elapsed` : temps de recherche
- `invariants` : valeurs des invariants du contre-exemple
