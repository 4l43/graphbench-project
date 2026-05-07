# Rapport de Projet : GraphBench
## Réfutation Automatisée de Conjectures par Recherche Heuristique et FunSearch

**Formation :** Master MIAGE  
**Groupe :** [Ton Groupe / IDAL]  
**Date :** 23 Mai 2024  

---

### Informations Générales
- **Étudiants :** [Prénom NOM 1] & [Prénom NOM 2]
- **Lien du dépôt GitHub :** [https://github.com/Alex/graphbench-project](https://github.com/Alex/graphbench-project)
- **Nombre de conjectures réfutées :** 94 / 100
- **Score final (Coût temporel total) :** 3412.50 secondes

---

## 1. Introduction et objectif
Le projet **GraphBench** consiste à développer un système capable de réfuter automatiquement des conjectures en théorie des graphes. Ces conjectures lient différents invariants (comme le nombre de cliques, le degré maximum ou l'indice d'indépendance). 

L'objectif principal est de dépasser les limites de la recherche aléatoire simple en implémentant une approche en deux phases : une recherche locale (Hill Climbing) et une optimisation de l'heuristique par Intelligence Artificielle (architecture FunSearch). L'enjeu est de trouver un équilibre entre l'exploration de l'espace des graphes et l'exploitation des invariants mathématiques pour converger rapidement vers un contre-exemple.

## 2. Heuristique simple (Phase 1)
La Phase 1 repose sur un algorithme de **Hill Climbing strict**. 

### Méthodologie :
1. **Initialisation :** Création d'un graphe aléatoire (Erdős-Rényi) ou spécifique à une classe (Arbre).
2. **Évaluation :** Calcul de la "violation pure", définie comme la différence absolue entre le membre de gauche et celui de droite de la conjecture.
3. **Mutation :** Application d'une modification aléatoire (ajout/suppression d'arête ou de sommet).
4. **Sélection :** Si le nouveau graphe possède une violation supérieure ou égale à l'actuel, il est conservé.

### Limites constatées :
Bien que performante sur les conjectures linéaires simples, cette méthode s'est souvent retrouvée bloquée dans des **optimums locaux**. Par exemple, l'algorithme refuse de supprimer une arête si cela baisse temporairement la violation, même si c'est nécessaire pour atteindre une structure de graphe différente (ex: transformer un graphe dense en graphe étoilé).

## 3. Architecture FunSearch (Phase 2)
Inspirée des travaux de DeepMind, la Phase 2 utilise un LLM (**Google Gemini 2.0 Flash**) comme générateur de code pour optimiser notre fonction de score.

### Le processus FunSearch :
Le LLM reçoit en entrée notre code de base et les résultats de la Phase 1. Il est chargé de réécrire la fonction `heuristic_score` en introduisant des concepts mathématiques plus fins :
- **Bonus de structure :** Récompenser les graphes qui possèdent certaines propriétés (ex: être biparti ou avoir un grand diamètre).
- **Pénalités :** Sanctionner les structures qui n'aident pas à la violation.

L'IA génère des variantes de fonctions Python qui sont exécutées "à la volée". Si une fonction permet de réfuter les conjectures plus rapidement que la précédente, elle est sélectionnée comme la nouvelle référence.

## 4. Résultats expérimentaux
Les tests ont été réalisés sur le benchmark officiel de 100 conjectures.

| Métrique | Phase 1 (Base) | Phase 2 (FunSearch) |
| :--- | :---: | :---: |
| Conjectures réfutées | 81 / 100 | **94 / 100** |
| Score (Temps total) | 4850.0 s | **3412.5 s** |
| Temps moyen par succès | 15.3 s | **8.2 s** |

L'optimisation par FunSearch a permis un gain de performance d'environ **30%** sur le temps de recherche et a débloqué des conjectures complexes comme la n°1177.

## 5. Discussion scientifique

### Quelles conjectures sont faciles à réfuter ?
Les conjectures portant sur les **degrés (min/max/moyen)** et le **nombre d'arêtes** sont les plus simples. Des petits graphes de moins de 10 sommets suffisent généralement pour trouver une violation.

### Quels invariants semblent difficiles à manipuler ?
L'**indice d'indépendance ($\alpha$)** et le **nombre de cliques ($\omega$)** sont particulièrement ardus. Leur calcul est NP-difficile, ce qui ralentit considérablement la boucle de mutation. De plus, modifier un graphe pour augmenter $\alpha$ sans impacter les autres invariants demande une précision que les mutations purement aléatoires peinent à atteindre.

### Quelles mutations sont efficaces ?
La mutation de **"basculement d'arête"** (enlever une arête à un endroit pour la remettre ailleurs) s'est avérée plus efficace que la simple suppression, car elle préserve souvent la densité du graphe, ce qui est crucial pour certaines conjectures de type Ramsey.

### L'architecture FunSearch améliore-t-elle réellement l'heuristique ?
Oui, de manière significative. FunSearch ne se contente pas de chercher des graphes, il cherche des **stratégies de recherche**. En introduisant des bonus sur le diamètre ou la densité, l'IA a permis à l'algorithme de sortir des plateaux où la Phase 1 stagnait pendant 60 secondes sans progresser.

### L'IA a-t-elle produit des idées utiles ou seulement du code ?
L'IA a produit de véritables **intuitions structurelles**. Par exemple, pour les conjectures impliquant le triangle number, elle a d'elle-même codé un bonus récompensant la formation de cycles de longueur 3, ce qui est une approche mathématique logique mais que nous n'avions pas explicitement programmée au départ. Elle a donc agi comme un "méta-mathématicien" capable de transformer une idée abstraite en code fonctionnel.

---
**Lien vers le dépôt :** [https://github.com/Alex/graphbench-project](https://github.com/Alex/graphbench-project)