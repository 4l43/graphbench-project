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

Les résultats sont sauvegardés dans `results/results.json` avec pour chaque conjecture :
- `found` : contre-exemple trouvé ?
- `graph6` : encodage graph6 du contre-exemple
- `violation` : valeur de la violation
- `time_elapsed` : temps de recherche
- `invariants` : valeurs des invariants du contre-exemple
