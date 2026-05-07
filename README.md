# Projet GraphBench : Réfutation Automatisée de Conjectures

Ce projet universitaire (Master MIAGE) a pour objectif de trouver des contre-exemples à des conjectures mathématiques en théorie des graphes. Il repose sur un algorithme de recherche locale par mutation (Hill Climbing) et explore l'optimisation de cet algorithme via l'Intelligence Artificielle.

Nous comparons deux approches distinctes :
1. **Phase 1 :** Une heuristique de base guidée uniquement par le score mathématique de violation de la conjecture.
2. **Phase 2 :** Une architecture inspirée de **FunSearch**, utilisant un LLM (Google Gemini) pour générer automatiquement des fonctions de score enrichies (bonus/malus) afin d'échapper aux optimums locaux.

---

## 📂 Structure du projet

Voici l'organisation de notre dépôt :

```text
graphbench-project/
├── benchmark/
│   └── benchmark.xlsx            # Dataset officiel contenant les 100 conjectures
├── experiments/
│   ├── phase1.py                 # Script d'exécution de la recherche locale classique
│   └── phase2.py                 # Script d'entraînement de l'architecture FunSearch
├── results/
│   └── resultats_recherchev1.csv # Fichier de sauvegarde des résultats de la Phase 1
├── src/                          # Dossier des codes sources et utilitaires
├── OC_projet_TD_noté.pdf         # Sujet original du projet
└── README.md                     # Documentation du projet