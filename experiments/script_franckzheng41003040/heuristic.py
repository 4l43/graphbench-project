"""
heuristic.py
Partie 1 : Heuristique de recherche locale pour réfuter des conjectures.
"""

import time
import random
import threading
import networkx as nx
from collections import defaultdict

# Compteur global des mutations utilisées (réinitialisé à chaque run)
MUTATION_COUNTER = defaultdict(int)

def reset_mutation_counter():
    """Réinitialise le compteur de mutations."""
    MUTATION_COUNTER.clear()

def print_mutation_stats():
    """Affiche les statistiques des mutations utilisées."""
    if not MUTATION_COUNTER:
        print("  Aucune mutation enregistrée.")
        return
    total = sum(MUTATION_COUNTER.values())
    print(f"  Total mutations: {total}")
    for name, count in sorted(MUTATION_COUNTER.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"    {name:40s}: {count:6d} ({pct:5.1f}%)")

from invariants import compute_invariants, needed_invariants
from graph_utils import (
    generate_initial_population, get_mutations, repair,
    satisfies_class
)


# Invariants qui peuvent bloquer longtemps (calcul matriciel O(n^3) ou pire)
SLOW_INVARIANTS = {
    'second_smallest_laplace_eigenvalue',
    'largest_eigenvalue',
    'largest_distance_eigenvalue',
}
# proximity/remoteness sont O(n^2) mais rapides en pratique -> pas de thread

def compute_invariants_safe(G, needed, remaining_time):
    if remaining_time <= 0.5:
        return {}
    # Si aucun invariant lent nest nécessaire, calcul direct sans thread
    if not (needed & SLOW_INVARIANTS):
        try:
            return compute_invariants(G, needed)
        except Exception:
            return {}
    # Sinon, thread avec timeout pour éviter les blocages
    # proximity/remoteness sont O(n^2) mais pas infiniment lents — budget plus généreux
    # Donner un budget généreux pour tous les invariants lents
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

    # Paramètres adaptatifs selon la conjecture
    params = get_search_params(conjecture)
    pop_size = params.get("pop_size", pop_size)
    n_range = params.get("n_override", None)

    # Multi-start: générer plusieurs populations initiales indépendantes
    # en variant la seed pour couvrir plus d'espace de recherche
    SPECTRAL = {'second_smallest_laplace_eigenvalue', 'largest_eigenvalue', 'largest_distance_eigenvalue'}
    if needed & SPECTRAL:
        # Pour les conjectures spectrales: population plus grande pour couvrir plus d'espace
        population = generate_initial_population(conjecture, size=pop_size * 2, n_range=n_range)
    else:
        population = generate_initial_population(conjecture, size=pop_size, n_range=n_range)
    scored_pop = []

    for G in population:
        if time.time() - start_time >= time_limit:
            break
        try:
            G = repair(G, classes)
            inv = compute_invariants_safe(G, needed, time_limit - (time.time() - start_time))
            if not inv:
                continue
            if not inv or not nx.is_connected(G):
                continue
            _needs_class_check2 = any(c in ['claw_free', 'tree', 'bipartite'] for c in [x.lower() for x in classes])
            if _needs_class_check2:
                from graph_utils import satisfies_class as _sc2
                if not _sc2(G, classes):
                    continue
            score = score_fn(G, inv, conjecture)
            scored_pop.append((score, G, inv))
        except Exception:
            continue

    if not scored_pop:
        # Fallback: graphe minimal
        G_fallback = repair(population[0] if population else nx.path_graph(5), classes)
        inv_fallback = {"n": G_fallback.number_of_nodes(), "m": G_fallback.number_of_edges()}
        scored_pop = [(0.0, G_fallback, inv_fallback)]

    scored_pop.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_pop[0][0]
    best_graph = scored_pop[0][1]
    best_inv = scored_pop[0][2]
    best_violation = conjecture.violation(best_inv)

    mutations = get_mutations(classes)
    iteration = 0
    restart_count = 0
    score_at_last_restart = best_score
    iter_at_last_restart = 0

    # ── Boucle principale avec multi-restart ────────────────
    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        # Vérifier si on a trouvé un contre-exemple
        _class_ok = [False]
        def _chk():
            _class_ok[0] = satisfies_class(best_graph, classes)
        _t = threading.Thread(target=_chk, daemon=True)
        _t.start()
        _t.join(timeout=3.0)

        if best_violation > 1e-6 and _class_ok[0]:
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

        # ── HARD RESTART si bloqué ───────────────────────────
        # Si le score n'a pas bougé depuis 150 itérations -> repartir de zéro
        iters_since_restart = iteration - iter_at_last_restart
        score_progress = scored_pop[0][0] - score_at_last_restart

        if iters_since_restart > 150 and score_progress < 0.01:
            restart_count += 1
            if verbose:
                print(f"    🔄 Restart #{restart_count} (iter={iteration}, t={elapsed:.1f}s, score={scored_pop[0][0]:.4f})")

            # Nouvelle population depuis une direction différente
            new_pop = generate_initial_population(conjecture, size=pop_size, n_range=n_range)
            new_scored = []
            for G_r in new_pop:
                if time.time() - start_time >= time_limit:
                    break
                try:
                    G_r = repair(G_r, classes)
                    if not nx.is_connected(G_r):
                        continue
                    inv_r = compute_invariants_safe(G_r, needed, time_limit - (time.time() - start_time))
                    if not inv_r:
                        continue
                    _needs_chk_r = any(c in ['claw_free','tree','bipartite'] for c in [x.lower() for x in classes])
                    if _needs_chk_r:
                        from graph_utils import satisfies_class as _sc_r
                        if not _sc_r(G_r, classes):
                            continue
                    score_r = score_fn(G_r, inv_r, conjecture)
                    new_scored.append((score_r, G_r, inv_r))
                except Exception:
                    pass

            if new_scored:
                # Garder le meilleur actuel + toute la nouvelle population
                scored_pop = sorted(
                    scored_pop[:1] + new_scored,
                    key=lambda x: x[0],
                    reverse=True
                )[:pop_size]

            score_at_last_restart = scored_pop[0][0]
            iter_at_last_restart = iteration

        # ── Sélection et mutations ───────────────────────────
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
                    safe_mutations = [m for m in mutations if m.__name__ not in
                                      ("mutate_add_vertex", "mutate_add_leaf",
                                       "mutate_add_clique", "mutate_add_path",
                                       "mutate_tree_add_leaf", "mutate_claw_free_add_clique")]
                    mut_fn = random.choice(safe_mutations) if safe_mutations else random.choice(mutations)
                else:
                    mut_fn = random.choice(mutations)
                H = mut_fn(H)
                MUTATION_COUNTER[mut_fn.__name__] += 1

            try:
                H = repair(H, classes)
            except Exception:
                continue

            try:
                inv = compute_invariants_safe(H, needed, time_limit - (time.time() - start_time))
                if inv is None:
                    break
                if not inv or inv.get('n', 0) < 2:
                    continue
                if not nx.is_connected(H):
                    continue
                _needs_class_check = any(c in ['claw_free', 'tree', 'bipartite'] for c in [x.lower() for x in classes])
                if _needs_class_check:
                    from graph_utils import satisfies_class as _sc
                    if not _sc(H, classes):
                        continue
                score = score_fn(H, inv, conjecture)
                new_candidates.append((score, H, inv))
            except Exception:
                continue

        scored_pop = sorted(
            scored_pop + new_candidates,
            key=lambda x: x[0],
            reverse=True
        )[:pop_size]

        if scored_pop[0][0] > best_score:
            best_score, best_graph, best_inv = scored_pop[0]
            best_violation = conjecture.violation(best_inv)
            if verbose:
                print(f"    iter={iteration}, score={best_score:.4f}, violation={best_violation:.4f}, "
                      f"n={best_graph.number_of_nodes()}, t={elapsed:.1f}s")

        # Diversification légère toutes les 30 itérations
        if iteration % 30 == 0:
            new_randoms = generate_initial_population(conjecture, size=3, n_range=n_range)
            for G in new_randoms:
                if time.time() - start_time >= time_limit:
                    break
                try:
                    G = repair(G, classes)
                    inv = compute_invariants_safe(G, needed, time_limit - (time.time() - start_time))
                    if not inv:
                        continue
                    score = score_fn(G, inv, conjecture)
                    scored_pop.append((score, G, inv))
                except Exception:
                    pass
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
    Tout est protégé par timeout via threads — aucun appel bloquant.
    """
    classes = conjecture.graph_classes
    needed = needed_invariants(conjecture)

    # Pour la validation: recalculer exactement les invariants critiques
    # (independence, vertex_cover, independent_domination peuvent être approchés pendant la recherche)
    EXACT_NEEDED = {'independence_number', 'vertex_cover_number', 'independent_domination_number'}
    inv = compute_invariants_safe(G, needed, remaining_time=15.0)
    if precomputed_inv:
        for k in needed:
            if k in precomputed_inv and k not in EXACT_NEEDED:
                inv[k] = precomputed_inv[k]
    # Recalcul exact pour les invariants approchés
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

    # satisfies_class aussi via thread pour éviter is_claw_free bloquant
    class_ok_box = [False]
    def _check_class():
        class_ok_box[0] = satisfies_class(G, classes)
    t = threading.Thread(target=_check_class, daemon=True)
    t.start()
    t.join(timeout=5.0)
    class_ok = class_ok_box[0]

    checks = {
        "class_ok": class_ok,
        "violation_positive": violation > 1e-6,
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


# ══════════════════════════════════════════════════════════════
#  PARAMÈTRES ADAPTATIFS PAR CONJECTURE
# ══════════════════════════════════════════════════════════════

# Invariants très lents → graphes petits pour avoir plus d'itérations
VERY_SLOW = {'second_smallest_laplace_eigenvalue', 'largest_distance_eigenvalue', 'proximity', 'remoteness'}
# Invariants qui nécessitent des grands graphes pour violer la borne
NEED_LARGE = {'triangle_number'}  # ex: conjecture 886 nécessite n=90

def get_search_params(conjecture):
    """
    Retourne (pop_size, n_min, n_max) adaptés à la conjecture.
    """
    needed = {conjecture.x_name, conjecture.y_name}
    classes = [c.lower() for c in conjecture.graph_classes]

    # Invariants très lents (spectraux) → les générateurs spécialisés choisissent la taille
    SPECTRAL_SLOW = {'second_smallest_laplace_eigenvalue', 'largest_eigenvalue', 'largest_distance_eigenvalue'}
    if needed & SPECTRAL_SLOW:
        return dict(pop_size=8, n_override=None)

    # independent_domination_number est lent via find_cliques → graphes modérés
    EXACT_SLOW = {'independent_domination_number'}
    if needed & EXACT_SLOW:
        return dict(pop_size=10, n_override=(5, 18))

    # Triangle number → grands graphes
    if needed & NEED_LARGE:
        return dict(pop_size=6, n_override=(20, 100))

    # Par défaut
    return dict(pop_size=8, n_override=None)