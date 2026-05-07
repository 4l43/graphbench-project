"""
heuristic.py
Partie 1 : Heuristique de recherche locale pour réfuter des conjectures.
"""

import time
import random
import networkx as nx

from invariants import compute_invariants, needed_invariants
from graph_utils import (
    generate_initial_population, get_mutations, repair,
    satisfies_class
)


# ══════════════════════════════════════════════════════════════
#  FONCTION DE SCORE (sera améliorée en Partie 2)
# ══════════════════════════════════════════════════════════════

def heuristic_score(G, invariants, conjecture):
    """
    Score à maximiser. violation > 0 = contre-exemple trouvé.
    Cette fonction de base sera évoluée via FunSearch en Partie 2.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("n", G.number_of_nodes())
    m = invariants.get("m", G.number_of_edges())
    Delta = invariants.get("maximum_degree", 0)
    delta = invariants.get("minimum_degree", 0)
    diam = invariants.get("diameter", 0)
    diam = diam if diam != float('inf') else 0
    alpha = invariants.get("independence_number", 0)
    gamma = invariants.get("domination_number", 0)
    mu = invariants.get("matching_number", 0)
    clique = invariants.get("clique_number", 0)
    avg = invariants.get("average_degree", 0)
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0

    # Score de direction: pousser x_val et y_val dans la bonne direction
    x_val = invariants.get(conjecture.x_name, 0)
    y_val = invariants.get(conjecture.y_name, 0)
    bound = conjecture.bound(x_val)
    # Gap = distance à la violation
    if conjecture.sign == "<=":
        gap = y_val - bound   # on veut y > bound
    else:
        gap = bound - y_val   # on veut y < bound

    return (
        10.0 * violation
        + 2.0 * gap            # guider vers la violation même quand gap < 0
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * (Delta - delta)  # hétérogénéité des degrés
        + 0.1 * clique
        - 0.03 * n             # préférer les graphes compacts
        - 0.1 * abs(density - 0.4)
    )


# ══════════════════════════════════════════════════════════════
#  RÉSULTAT
# ══════════════════════════════════════════════════════════════

class SearchResult:
    def __init__(self, conjecture_id, found, graph=None, invariants=None,
                 violation=None, time_elapsed=None, graph6=None):
        self.conjecture_id = conjecture_id
        self.found = found
        self.graph = graph
        self.invariants = invariants or {}
        self.violation = violation
        self.time_elapsed = time_elapsed
        self.graph6 = graph6

    def __repr__(self):
        if self.found:
            return (f"✅ Conjecture {self.conjecture_id} réfutée en {self.time_elapsed:.2f}s "
                    f"(violation={self.violation:.4f})")
        else:
            return f"❌ Conjecture {self.conjecture_id} non réfutée après {self.time_elapsed:.2f}s"


# ══════════════════════════════════════════════════════════════
#  MOTEUR DE RECHERCHE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def search_counterexample(
    conjecture,
    time_limit=60,
    pop_size=8,
    score_fn=None,
    verbose=False
):
    """
    Recherche un contre-exemple pour une conjecture donnée.

    Algorithme:
    1. Génère une population initiale
    2. Pour chaque candidat, applique des mutations
    3. Garde les meilleurs (élitisme)
    4. Retourne dès qu'un contre-exemple est trouvé

    Returns: SearchResult
    """
    if score_fn is None:
        score_fn = heuristic_score

    start_time = time.time()
    needed = needed_invariants(conjecture)
    # Toujours calculer n et m
    needed.add("n")
    needed.add("m")

    classes = conjecture.graph_classes

    # ── Population initiale ──────────────────────────────────
    population = generate_initial_population(conjecture, size=pop_size)
    scored_pop = []

    for G in population:
        G = repair(G, classes)
        inv = compute_invariants(G, needed)
        score = score_fn(G, inv, conjecture)
        scored_pop.append((score, G, inv))

    scored_pop.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_pop[0][0]
    best_graph = scored_pop[0][1]
    best_inv = scored_pop[0][2]
    best_violation = conjecture.violation(best_inv)

    mutations = get_mutations(classes)
    iteration = 0

    # ── Boucle principale ────────────────────────────────────
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        # Vérifier si on a trouvé un contre-exemple
        if best_violation > 1e-9 and satisfies_class(best_graph, classes):
            # Re-valider avec un calcul complet propre
            val = validate_counterexample(best_graph, conjecture, precomputed_inv=best_inv)
            if val["valid"]:
                g6 = nx.to_graph6_bytes(best_graph, header=False).decode().strip()
                return SearchResult(
                    conjecture_id=conjecture.id,
                    found=True,
                    graph=best_graph,
                    invariants=val["invariants"],
                    violation=val["violation"],
                    time_elapsed=elapsed,
                    graph6=g6
                )

        iteration += 1

        # Sélectionner des candidats (top 50% de la population)
        top_k = max(1, len(scored_pop) // 2)
        candidates = scored_pop[:top_k]

        new_candidates = []
        for _ in range(pop_size):
            _, G, _ = random.choice(candidates)

            # Appliquer une ou plusieurs mutations
            num_mutations = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            H = G.copy()
            for _ in range(num_mutations):
                mut_fn = random.choice(mutations)
                H = mut_fn(H)

            # Réparation
            H = repair(H, classes)

            # Calcul des invariants et score
            try:
                inv = compute_invariants(H, needed)
                score = score_fn(H, inv, conjecture)
                new_candidates.append((score, H, inv))
            except Exception:
                continue

        # Mise à jour de la population (élitisme: garder les meilleurs)
        scored_pop = sorted(
            scored_pop + new_candidates,
            key=lambda x: x[0],
            reverse=True
        )[:pop_size]

        # Mise à jour du meilleur
        if scored_pop[0][0] > best_score:
            best_score, best_graph, best_inv = scored_pop[0]
            best_violation = conjecture.violation(best_inv)
            if verbose:
                print(f"    iter={iteration}, score={best_score:.4f}, violation={best_violation:.4f}, "
                      f"n={best_graph.number_of_nodes()}, t={elapsed:.1f}s")

        # Diversification périodique (éviter les minima locaux)
        if iteration % 50 == 0:
            # Réintroduire de la diversité
            new_randoms = generate_initial_population(conjecture, size=2)
            for G in new_randoms:
                G = repair(G, classes)
                inv = compute_invariants(G, needed)
                score = score_fn(G, inv, conjecture)
                scored_pop.append((score, G, inv))
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            scored_pop = scored_pop[:pop_size]

    # ── Retour sans contre-exemple ───────────────────────────
    elapsed = time.time() - start_time
    return SearchResult(
        conjecture_id=conjecture.id,
        found=False,
        graph=best_graph,
        invariants=best_inv,
        violation=best_violation,
        time_elapsed=elapsed,
        graph6=None
    )


# ══════════════════════════════════════════════════════════════
#  VALIDATION D'UN CONTRE-EXEMPLE
# ══════════════════════════════════════════════════════════════

def validate_counterexample(G, conjecture, precomputed_inv=None) -> dict:
    """
    Valide qu'un graphe est bien un contre-exemple selon les règles du projet.
    Retourne un dict avec les détails.
    Si precomputed_inv est fourni, utilise ces invariants (plus précis).
    """
    classes = conjecture.graph_classes
    needed = needed_invariants(conjecture)
    # Recalculer complètement pour la validation finale
    inv = compute_invariants(G, needed)
    # Si on a des invariants pré-calculés cohérents, les utiliser
    if precomputed_inv:
        for k in needed:
            if k in precomputed_inv:
                inv[k] = precomputed_inv[k]
    violation = conjecture.violation(inv)

    checks = {
        "class_ok": satisfies_class(G, classes),
        "violation_positive": violation > 1e-9,
        "invariants": inv,
        "violation": violation,
        "x_value": inv.get(conjecture.x_name),
        "y_value": inv.get(conjecture.y_name),
        "bound": conjecture.bound(inv.get(conjecture.x_name, 0)),
    }
    checks["valid"] = checks["class_ok"] and checks["violation_positive"]
    return checks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from conjecture import load_benchmark

    conjectures = load_benchmark("../benchmark/benchmark.xlsx")
    print(f"Test sur les 3 premières conjectures (10 secondes chacune)...")

    for c in conjectures[:3]:
        print(f"\n  {c}")
        result = search_counterexample(c, time_limit=10, verbose=True)
        print(f"  {result}")
        if result.found:
            val = validate_counterexample(result.graph, c)
            print(f"  Validation: {val['valid']}, violation={val['violation']:.4f}")
            print(f"  graph6: {result.graph6}")
