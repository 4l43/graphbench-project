"""
heuristic.py  (version patchée)
─────────────────────────────────────────────────────────────────────────────
Changements par rapport à l'original :

1. get_search_params
   - Tailles ciblées sur les vraies distributions du benchmark :
     total_domination → [10, 24], median=18
     independent_domination → [12, 24], median=19
     remoteness/proximity → [8, 22], median=15
   - pop_size augmenté pour les conjectures lentes

2. search_counterexample
   - Seuil de restart adaptatif : plus patient pour les invariants lents
     (total_domination, independent_domination coûtent ~50× plus que degree)
   - Diversification toutes les 20 iter au lieu de 30 pour les cas durs

Tout le reste (score, population, validation) est inchangé.
"""

import time
import random
import threading
import networkx as nx
from collections import defaultdict

MUTATION_COUNTER = defaultdict(int)

def reset_mutation_counter():
    MUTATION_COUNTER.clear()

def print_mutation_stats():
    if not MUTATION_COUNTER:
        print("  Aucune mutation enregistrée.")
        return
    total = sum(MUTATION_COUNTER.values())
    print(f"  Total mutations: {total}")
    for name, count in sorted(MUTATION_COUNTER.items(), key=lambda x: -x[1]):
        print(f"    {name:40s}: {count:6d} ({100*count/total:5.1f}%)")

from invariants import compute_invariants, needed_invariants
from graph_utils import (
    generate_initial_population, get_mutations, repair,
    satisfies_class
)

SLOW_INVARIANTS = {
    'second_smallest_laplace_eigenvalue',
    'largest_eigenvalue',
    'largest_distance_eigenvalue',
}

# Invariants dont le calcul est O(2^n) ou multi-restart → budget plus généreux
EXACT_SLOW = {
    'total_domination_number',
    'independent_domination_number',
}


def compute_invariants_safe(G, needed, remaining_time):
    if remaining_time <= 0.5:
        return {}
    if not (needed & SLOW_INVARIANTS):
        try:
            return compute_invariants(G, needed)
        except Exception:
            return {}
    budget = min(8.0, remaining_time - 0.3)
    result_box = [None]
    def _run():
        try:
            result_box[0] = compute_invariants(G, needed)
        except Exception:
            result_box[0] = {}
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=budget)
    return result_box[0] if result_box[0] is not None else {}


# ══════════════════════════════════════════════════════════════
#  SCORE (inchangé)
# ══════════════════════════════════════════════════════════════

def heuristic_score(G, invariants, conjecture):
    violation = conjecture.violation(invariants)
    n  = invariants.get("n", G.number_of_nodes())
    m  = invariants.get("m", G.number_of_edges())
    Delta = invariants.get("maximum_degree", 0)
    delta = invariants.get("minimum_degree", 0)
    diam  = invariants.get("diameter", 0)
    diam  = diam if diam != float('inf') else 0
    clique = invariants.get("clique_number", 0)
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0

    x_val = invariants.get(conjecture.x_name, 0)
    y_val = invariants.get(conjecture.y_name, 0)
    bound = conjecture.bound(x_val)
    gap = (y_val - bound) if conjecture.sign == "<=" else (bound - y_val)

    return (
        10.0 * violation
        + 2.0 * gap
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * (Delta - delta)
        + 0.1 * clique
        - 0.03 * n
        - 0.1 * abs(density - 0.4)
    )


# ══════════════════════════════════════════════════════════════
#  RÉSULTAT (inchangé)
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
            return (f"✅ Conjecture {self.conjecture_id} réfutée en "
                    f"{self.time_elapsed:.2f}s (violation={self.violation:.4f})")
        return (f"❌ Conjecture {self.conjecture_id} non réfutée après "
                f"{self.time_elapsed:.2f}s")


# ══════════════════════════════════════════════════════════════
#  PARAMÈTRES ADAPTATIFS — VERSION PATCHÉE
# ══════════════════════════════════════════════════════════════

VERY_SLOW  = {'second_smallest_laplace_eigenvalue', 'largest_distance_eigenvalue',
              'proximity', 'remoteness'}
NEED_LARGE = {'triangle_number'}

def get_search_params(conjecture):
    """
    Retourne les paramètres de recherche adaptés à la conjecture.

    Tailles choisies d'après la distribution réelle des contre-exemples :
    ┌──────────────────────────────────────┬───────────┬──────────────┐
    │ Invariant                            │ n_range   │ pop_size     │
    ├──────────────────────────────────────┼───────────┼──────────────┤
    │ total_domination_number              │ [10, 24]  │ 12           │
    │ independent_domination_number        │ [12, 24]  │ 12           │
    │ remoteness / proximity               │ [8, 22]   │ 10           │
    │ spectraux (λ, λ2, D-eigenvalue)      │ libre     │ 8            │
    │ triangle_number                      │ [20, 100] │ 6            │
    │ défaut                               │ libre     │ 8            │
    └──────────────────────────────────────┴───────────┴──────────────┘
    """
    needed = {conjecture.x_name, conjecture.y_name}

    SPECTRAL = {'second_smallest_laplace_eigenvalue', 'largest_eigenvalue',
                'largest_distance_eigenvalue'}

    # Invariants exacts lents : cibler les bonnes tailles
    if 'total_domination_number' in needed:
        return dict(pop_size=12, n_override=(10, 24))

    if 'independent_domination_number' in needed:
        return dict(pop_size=12, n_override=(12, 24))

    if needed & {'remoteness', 'proximity'}:
        return dict(pop_size=10, n_override=(8, 22))

    if needed & SPECTRAL:
        return dict(pop_size=8, n_override=None)

    if needed & NEED_LARGE:
        return dict(pop_size=6, n_override=(20, 100))

    return dict(pop_size=8, n_override=None)


def _is_hard(conjecture) -> bool:
    """True si la conjecture implique des invariants coûteux à calculer."""
    needed = {conjecture.x_name, conjecture.y_name}
    return bool(needed & (EXACT_SLOW | VERY_SLOW))


# ══════════════════════════════════════════════════════════════
#  MOTEUR DE RECHERCHE — VERSION PATCHÉE
# ══════════════════════════════════════════════════════════════

def search_counterexample(conjecture, time_limit=60, pop_size=8,
                          score_fn=None, verbose=False):
    if score_fn is None:
        score_fn = heuristic_score

    start_time = time.time()
    needed = needed_invariants(conjecture)
    needed.update({"n", "m"})
    classes = conjecture.graph_classes

    params  = get_search_params(conjecture)
    pop_size = params.get("pop_size", pop_size)
    n_range  = params.get("n_override", None)

    # ── SEUIL DE RESTART adaptatif ─────────────────────────────────
    # Les invariants exacts (total_dom, indep_dom) coûtent ~10-50× plus cher
    # à calculer → on donne plus de budget entre les restarts.
    if _is_hard(conjecture):
        restart_patience = 80    # itérations avant de repartir
        diversify_every  = 15    # diversification plus fréquente
    else:
        restart_patience = 150
        diversify_every  = 30

    SPECTRAL = {'second_smallest_laplace_eigenvalue', 'largest_eigenvalue',
                'largest_distance_eigenvalue'}
    if needed & SPECTRAL:
        population = generate_initial_population(
            conjecture, size=pop_size * 2, n_range=n_range)
    else:
        population = generate_initial_population(
            conjecture, size=pop_size, n_range=n_range)

    scored_pop = []
    for G in population:
        if time.time() - start_time >= time_limit:
            break
        try:
            G = repair(G, classes)
            inv = compute_invariants_safe(
                G, needed, time_limit - (time.time() - start_time))
            if not inv or not nx.is_connected(G):
                continue
            if any(c in ['claw_free', 'tree', 'bipartite']
                   for c in [x.lower() for x in classes]):
                if not satisfies_class(G, classes):
                    continue
            score = score_fn(G, inv, conjecture)
            scored_pop.append((score, G, inv))
        except Exception:
            continue

    if not scored_pop:
        G_fb = repair(population[0] if population else nx.path_graph(5), classes)
        scored_pop = [(0.0, G_fb, {"n": G_fb.number_of_nodes(),
                                    "m": G_fb.number_of_edges()})]

    scored_pop.sort(key=lambda x: x[0], reverse=True)
    best_score, best_graph, best_inv = scored_pop[0]
    best_violation = conjecture.violation(best_inv)

    mutations = get_mutations(classes)
    iteration = 0
    restart_count = 0
    score_at_last_restart = best_score
    iter_at_last_restart  = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        # ── Vérification contre-exemple ─────────────────────────────
        class_ok = [False]
        def _chk():
            class_ok[0] = satisfies_class(best_graph, classes)
        t = threading.Thread(target=_chk, daemon=True)
        t.start()
        t.join(timeout=3.0)

        if best_violation > 1e-6 and class_ok[0]:
            val = validate_counterexample(
                best_graph, conjecture, precomputed_inv=best_inv)
            if val["valid"]:
                g6 = nx.to_graph6_bytes(best_graph, header=False).decode().strip()
                return SearchResult(
                    conjecture_id=conjecture.id, found=True,
                    graph=best_graph, invariants=val["invariants"],
                    violation=val["violation"],
                    time_elapsed=elapsed, graph6=g6)

        iteration += 1
        iters_since_restart  = iteration - iter_at_last_restart
        score_progress = scored_pop[0][0] - score_at_last_restart

        # ── RESTART adaptatif ────────────────────────────────────────
        if iters_since_restart > restart_patience and score_progress < 0.01:
            restart_count += 1
            if verbose:
                print(f"    🔄 Restart #{restart_count} "
                      f"(iter={iteration}, t={elapsed:.1f}s, "
                      f"score={scored_pop[0][0]:.4f})")
            new_pop = generate_initial_population(
                conjecture, size=pop_size, n_range=n_range)
            new_scored = []
            for G_r in new_pop:
                if time.time() - start_time >= time_limit:
                    break
                try:
                    G_r = repair(G_r, classes)
                    if not nx.is_connected(G_r):
                        continue
                    inv_r = compute_invariants_safe(
                        G_r, needed,
                        time_limit - (time.time() - start_time))
                    if not inv_r:
                        continue
                    if any(c in ['claw_free', 'tree', 'bipartite']
                           for c in [x.lower() for x in classes]):
                        if not satisfies_class(G_r, classes):
                            continue
                    new_scored.append((score_fn(G_r, inv_r, conjecture), G_r, inv_r))
                except Exception:
                    pass

            if new_scored:
                scored_pop = sorted(
                    scored_pop[:1] + new_scored,
                    key=lambda x: x[0], reverse=True)[:pop_size]

            score_at_last_restart = scored_pop[0][0]
            iter_at_last_restart  = iteration

        # ── Sélection + mutations ────────────────────────────────────
        top_k = max(1, len(scored_pop) // 2)
        candidates = scored_pop[:top_k]
        new_candidates = []

        for _ in range(pop_size):
            if time.time() - start_time >= time_limit:
                break
            _, G, _ = random.choice(candidates)
            num_mutations = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            H = G.copy()
            for _ in range(num_mutations):
                if n_range and H.number_of_nodes() >= n_range[1]:
                    safe = [f for f in mutations if f.__name__ not in {
                        "mutate_add_vertex", "mutate_add_leaf",
                        "mutate_add_clique", "mutate_add_path",
                        "mutate_tree_add_leaf", "mutate_claw_free_add_clique"}]
                    mut_fn = random.choice(safe) if safe else random.choice(mutations)
                else:
                    mut_fn = random.choice(mutations)
                H = mut_fn(H)
                MUTATION_COUNTER[mut_fn.__name__] += 1

            try:
                H = repair(H, classes)
            except Exception:
                continue

            try:
                inv = compute_invariants_safe(
                    H, needed, time_limit - (time.time() - start_time))
                if inv is None:
                    break
                if not inv or inv.get('n', 0) < 2:
                    continue
                if not nx.is_connected(H):
                    continue
                if any(c in ['claw_free', 'tree', 'bipartite']
                       for c in [x.lower() for x in classes]):
                    if not satisfies_class(H, classes):
                        continue
                new_candidates.append((score_fn(H, inv, conjecture), H, inv))
            except Exception:
                continue

        scored_pop = sorted(
            scored_pop + new_candidates,
            key=lambda x: x[0], reverse=True)[:pop_size]

        if scored_pop[0][0] > best_score:
            best_score, best_graph, best_inv = scored_pop[0]
            best_violation = conjecture.violation(best_inv)
            if verbose:
                print(f"    iter={iteration}, score={best_score:.4f}, "
                      f"violation={best_violation:.4f}, "
                      f"n={best_graph.number_of_nodes()}, t={elapsed:.1f}s")

        # ── Diversification ──────────────────────────────────────────
        if iteration % diversify_every == 0:
            new_randoms = generate_initial_population(
                conjecture, size=3, n_range=n_range)
            for G in new_randoms:
                if time.time() - start_time >= time_limit:
                    break
                try:
                    G = repair(G, classes)
                    inv = compute_invariants_safe(
                        G, needed, time_limit - (time.time() - start_time))
                    if not inv:
                        continue
                    scored_pop.append((score_fn(G, inv, conjecture), G, inv))
                except Exception:
                    pass
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            scored_pop = scored_pop[:pop_size]

    elapsed = time.time() - start_time
    return SearchResult(
        conjecture_id=conjecture.id, found=False,
        graph=best_graph, invariants=best_inv,
        violation=best_violation, time_elapsed=elapsed, graph6=None)


# ══════════════════════════════════════════════════════════════
#  VALIDATION (inchangée)
# ══════════════════════════════════════════════════════════════

def validate_counterexample(G, conjecture, precomputed_inv=None) -> dict:
    classes = conjecture.graph_classes
    needed  = needed_invariants(conjecture)
    EXACT_NEEDED = {'independence_number', 'vertex_cover_number',
                    'independent_domination_number'}
    inv = compute_invariants_safe(G, needed, remaining_time=15.0)
    if precomputed_inv:
        for k in needed:
            if k in precomputed_inv and k not in EXACT_NEEDED:
                inv[k] = precomputed_inv[k]
    for k in needed & EXACT_NEEDED:
        try:
            from invariants import INVARIANT_FUNCTIONS
            fn = INVARIANT_FUNCTIONS.get(k)
            if fn:
                if k == 'independence_number':
                    inv[k] = fn(G, exact=True)
                else:
                    inv[k] = fn(G)
        except Exception:
            pass

    violation = conjecture.violation(inv)
    class_ok_box = [False]
    def _check_class():
        class_ok_box[0] = satisfies_class(G, classes)
    t = threading.Thread(target=_check_class, daemon=True)
    t.start()
    t.join(timeout=5.0)

    checks = {
        "class_ok":          class_ok_box[0],
        "violation_positive": violation > 1e-6,
        "invariants":         inv,
        "violation":          violation,
        "x_value":            inv.get(conjecture.x_name),
        "y_value":            inv.get(conjecture.y_name),
        "bound":              conjecture.bound(inv.get(conjecture.x_name, 0)),
    }
    checks["valid"] = checks["class_ok"] and checks["violation_positive"]
    return checks