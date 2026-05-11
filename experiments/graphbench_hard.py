"""
=========================================================================================
GRAPHBENCH — PATCH HARD CONJECTURES
=========================================================================================

Ce module remplace les parties critiques de graphbench_v2.py pour les ~100 conjectures
difficiles. Problèmes identifiés :

1. total_domination_number   → calcul glouton faux (sous-estime → pas de violation)
2. independent_domination_number → mis à égal de alpha (independance number) → FAUX
3. Taille de recherche → les contre-exemples font 10-29 nœuds, on cherchait trop petit
4. claw_free + domination   → mutations qui cassent tout

FIXES :
- Calcul exact de total_domination (backtracking avec pruning, ≤22 nœuds)
- Calcul exact de independent_domination (MIS minimum par backtracking)
- Schedule de tailles ciblé sur [10, 29] selon le temps de référence
- Mutations spéciales "domination-aware" pour claw_free
=========================================================================================
"""

import networkx as nx
import random
import itertools
import math


# =====================================================================
# FIX 1 : TOTAL DOMINATION EXACTE (pour n ≤ 22, approximation sinon)
# =====================================================================

def total_domination_exact(G):
    """
    Nombre de domination totale exact par branch-and-bound.
    Un ensemble S est totalement dominant si TOUT sommet (y compris dans S)
    a AU MOINS UN voisin dans S.
    
    Complexité : O(2^n) avec pruning fort → faisable jusqu'à n=22.
    Pour n > 22 : plusieurs démarrages gloutons aléatoires.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return 0
    
    # Vérification : graphe sans sommet isolé requis
    for v in nodes:
        if G.degree(v) == 0:
            return n  # pas de domination totale possible sans arêtes
    
    neighbors = {v: frozenset(G.neighbors(v)) for v in nodes}
    node_idx = {v: i for i, v in enumerate(nodes)}
    
    # Greedy pour borne supérieure initiale
    def greedy_total_dom():
        covered = set()
        dom_set = []
        remaining = set(nodes)
        while covered != set(nodes):
            # Choisir le nœud qui couvre le plus de non-couverts
            best_v = max(remaining if remaining else nodes,
                        key=lambda v: len(neighbors[v] - covered))
            dom_set.append(best_v)
            covered |= neighbors[best_v]
            if not (neighbors[best_v] - set(nodes)):
                break
        return len(dom_set)
    
    if n > 22:
        # Pour les grands graphes : 20 redémarrages gloutons aléatoires
        best = n
        for _ in range(20):
            covered = set()
            dom_set = []
            node_order = list(nodes)
            random.shuffle(node_order)
            while covered != set(nodes):
                # Parmi les non-utilisés, prend le meilleur
                candidates = [v for v in node_order if v not in dom_set]
                if not candidates:
                    break
                v = max(candidates, key=lambda x: len(neighbors[x] - covered))
                dom_set.append(v)
                covered |= neighbors[v]
            best = min(best, len(dom_set))
        return best
    
    best = [n]  # borne supérieure (liste pour mutation dans closure)
    
    def backtrack(idx, dom_set_size, covered_mask):
        """
        idx         : index du prochain nœud à considérer
        dom_set_size: taille du set courant
        covered_mask: bitmask des nœuds couverts
        """
        if dom_set_size >= best[0]:
            return  # Élagage : déjà pire que le meilleur
        
        all_covered = (1 << n) - 1
        if covered_mask == all_covered:
            best[0] = dom_set_size
            return
        
        if idx == n:
            return
        
        # Élagage : peut-on encore couvrir tous les nœuds ?
        # Les nœuds restants peuvent couvrir au maximum tous leurs voisins
        remaining_coverage = covered_mask
        for i in range(idx, n):
            for nb in neighbors[nodes[i]]:
                remaining_coverage |= (1 << node_idx[nb])
        if remaining_coverage != all_covered:
            return  # Impossible même en ajoutant tout
        
        v = nodes[idx]
        v_mask = 1 << idx
        
        # Branche 1 : on ajoute v dans le set dominant
        new_covered = covered_mask
        for nb in neighbors[v]:
            new_covered |= (1 << node_idx[nb])
        backtrack(idx + 1, dom_set_size + 1, new_covered)
        
        # Branche 2 : on n'ajoute pas v
        backtrack(idx + 1, dom_set_size, covered_mask)
    
    backtrack(0, 0, 0)
    return best[0]


# =====================================================================
# FIX 2 : INDEPENDENT DOMINATION EXACTE
# =====================================================================

def independent_domination_exact(G):
    """
    Nombre de domination indépendante = taille du plus petit ensemble maximal indépendant.
    
    Un ensemble S est indépendant dominant si :
      - S est indépendant (pas d'arête entre deux sommets de S)
      - S est dominant (tout sommet hors de S a un voisin dans S)
      ↔ S est un ensemble indépendant MAXIMAL
    
    On cherche le minimum de ces ensembles.
    Approche : backtracking avec pruning, exact pour n ≤ 20.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return 0
    
    neighbors = {v: set(G.neighbors(v)) for v in nodes}
    
    if n > 20:
        # Pour n > 20 : heuristique (MIS aléatoires)
        best = n
        for _ in range(50):
            order = list(nodes)
            random.shuffle(order)
            indep_set = []
            dominated = set()
            for v in order:
                if v not in dominated and all(u not in indep_set for u in neighbors[v]):
                    indep_set.append(v)
                    dominated |= neighbors[v] | {v}
            # Vérifier si c'est un vrai MIS
            is_maximal = all(
                any(u in indep_set for u in neighbors[v])
                for v in nodes if v not in indep_set
            )
            if is_maximal:
                best = min(best, len(indep_set))
        return best
    
    best = [n]
    
    def backtrack(idx, current_indep, dominated):
        """
        idx           : prochain nœud à considérer
        current_indep : ensemble indépendant courant
        dominated     : ensemble de nœuds dominés (dans current_indep ou voisins)
        """
        if len(current_indep) >= best[0]:
            return  # Élagage
        
        # Vérifier si l'ensemble courant est déjà maximal
        all_dominated = dominated == set(nodes)
        if all_dominated:
            best[0] = len(current_indep)
            return
        
        if idx == n:
            return
        
        v = nodes[idx]
        
        # Peut-on encore ajouter v ? (il faut qu'il ne soit pas voisin d'un membre)
        can_add = all(u not in current_indep for u in neighbors[v])
        
        if can_add and v not in dominated:
            # Branche 1 : ajouter v
            new_dominated = dominated | {v} | neighbors[v]
            current_indep.append(v)
            backtrack(idx + 1, current_indep, new_dominated)
            current_indep.pop()
        
        # Branche 2 : ne pas ajouter v (seulement si v sera dominé autrement)
        # Si v n'est pas encore dominé ET aucun voisin n'a été ajouté ni ne peut l'être
        backtrack(idx + 1, current_indep, dominated)
    
    backtrack(0, [], set())
    return best[0]


# =====================================================================
# FIX 3 : COMPUTE_INVARIANTS CORRIGÉ
# =====================================================================

def compute_invariants_fixed(G, conjecture, use_exact=True):
    """
    Version corrigée de compute_invariants avec calculs exacts des invariants difficiles.
    À utiliser à la place de compute_invariants pour les conjectures hard.
    """
    import pandas as pd
    import numpy as np
    
    def parse_val(val):
        if pd.isna(val): return 0.0
        s = str(val).strip()
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        return float(s)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {}
    
    needed = {conjecture['X'], conjecture['Y']}
    inv = {"n": n, "m": m}
    
    degrees = [d for _, d in G.degree()]
    inv.update({
        "minimum_degree": min(degrees) if degrees else 0,
        "maximum_degree": max(degrees) if degrees else 0,
        "average_degree": sum(degrees) / n if degrees else 0,
        "density": nx.density(G),
    })
    
    is_conn = nx.is_connected(G)
    
    # Diamètre / Rayon
    if "diameter" in needed or "radius" in needed:
        if is_conn:
            inv["diameter"] = nx.diameter(G)
            inv["radius"] = nx.radius(G)
        else:
            inv["diameter"] = 0
            inv["radius"] = 0
    
    # Triangles
    if "triangle_number" in needed:
        inv["triangle_number"] = sum(nx.triangles(G).values()) / 3
    
    # Clique / Indépendance
    clique_needed = any(k in needed for k in [
        "clique_number", "independence_number", "vertex_cover_number"])
    if clique_needed:
        from graphbench_v2 import get_clique_number
        omega = get_clique_number(G)
        alpha = get_clique_number(nx.complement(G))
        inv["clique_number"] = omega
        inv["independence_number"] = alpha
        inv["vertex_cover_number"] = n - alpha
    
    # ✅ FIX : independent_domination_number (EXACT, pas = alpha)
    if "independent_domination_number" in needed:
        if use_exact and n <= 20:
            inv["independent_domination_number"] = independent_domination_exact(G)
        else:
            inv["independent_domination_number"] = independent_domination_exact(G)
    
    # Couplage (matching)
    if "matching_number" in needed:
        inv["matching_number"] = len(nx.max_weight_matching(G, maxcardinality=True))
    
    # Domination simple
    if "domination_number" in needed:
        try:
            from networkx.algorithms.approximation import min_weighted_dominating_set
            inv["domination_number"] = len(min_weighted_dominating_set(G))
        except Exception:
            inv["domination_number"] = 0
    
    # ✅ FIX : total_domination_number (EXACT, pas greedy)
    if "total_domination_number" in needed:
        inv["total_domination_number"] = total_domination_exact(G)
    
    # Proximity / Remoteness
    if "proximity" in needed or "remoteness" in needed:
        if is_conn:
            from graphbench_v2 import proximity_remoteness
            prox, rem = proximity_remoteness(G)
            inv["proximity"] = prox
            inv["remoteness"] = rem
        else:
            inv["proximity"] = 0.0
            inv["remoteness"] = 0.0
    
    # Spectral
    if "largest_eigenvalue" in needed or "second_smallest_laplace_eigenvalue" in needed:
        from graphbench_v2 import compute_spectral
        le, fiedler = compute_spectral(G)
        inv["largest_eigenvalue"] = le
        inv["second_smallest_laplace_eigenvalue"] = fiedler
    
    if "largest_distance_eigenvalue" in needed:
        from graphbench_v2 import compute_distance_eigenvalue
        inv["largest_distance_eigenvalue"] = compute_distance_eigenvalue(G)
    
    # Zagreb / Randić / Harmonique
    if "first_zagreb_index" in needed or "second_zagreb_index" in needed:
        from graphbench_v2 import zagreb_indices
        z1, z2 = zagreb_indices(G)
        inv["first_zagreb_index"] = z1
        inv["second_zagreb_index"] = z2
    
    if "randic_index" in needed:
        from graphbench_v2 import randic_index
        inv["randic_index"] = randic_index(G)
    
    if "harmonic_index" in needed:
        from graphbench_v2 import harmonic_index
        inv["harmonic_index"] = harmonic_index(G)
    
    return inv


# =====================================================================
# FIX 4 : MUTATIONS DOMINATION-AWARE
# =====================================================================

def mutate_domination_aware(G, subgroups, conjecture):
    """
    Mutations spéciales pour les conjectures impliquant total_domination
    ou independent_domination.
    
    Idée : au lieu de muter aléatoirement, on essaie de créer des structures
    qui augmentent/diminuent ces invariants spécifiquement.
    """
    H = G.copy()
    nodes = list(H.nodes())
    y_inv = conjecture['Y']
    x_inv = conjecture['X']
    sign = conjecture['Sign']
    want_large_y = (sign == '<=')
    
    action = random.choice([
        "structural", "structural", "structural",
        "rewire", "add_edge", "remove_edge"
    ])
    
    if action == "structural":
        target = y_inv if want_large_y else x_inv
        
        if target == "total_domination_number" and want_large_y:
            # Créer des sous-graphes où chaque nœud a peu de voisins
            # → domination totale difficile → γt grand
            # Stratégie : chaîne de triangles ou graphes Petersen-like
            _make_hard_to_dominate(H, subgroups)
            
        elif target == "total_domination_number" and not want_large_y:
            # Rendre la domination facile → hub central
            if nodes:
                hub = random.choice(nodes)
                for v in nodes:
                    if v != hub and not H.has_edge(hub, v):
                        if random.random() < 0.4:
                            H.add_edge(hub, v)
        
        elif target == "independent_domination_number" and want_large_y:
            # Faire un graphe où les MIS sont grands
            # → graphe Kneser-like ou grille sparse
            # Supprimer des arêtes pour augmenter l'indépendance
            if H.number_of_edges() > 0:
                e = random.choice(list(H.edges()))
                H.remove_edge(*e)
                if 'connected' in subgroups and not nx.is_connected(H):
                    H.add_edge(*e)
        
        elif target == "matching_number" and want_large_y:
            # Parfait matching → graphe régulier
            # Ajouter des arêtes entre nœuds de faible degré
            low_deg = sorted(nodes, key=lambda v: H.degree(v))
            for i in range(0, len(low_deg) - 1, 2):
                if not H.has_edge(low_deg[i], low_deg[i+1]):
                    H.add_edge(low_deg[i], low_deg[i+1])
                    break
    
    elif action == "rewire" and H.number_of_edges() > 0:
        edges = list(H.edges())
        u, v = random.choice(edges)
        candidates = [w for w in nodes if w != u and w != v and not H.has_edge(u, w)]
        if candidates:
            w = random.choice(candidates)
            H.remove_edge(u, v)
            H.add_edge(u, w)
            if 'connected' in subgroups and not nx.is_connected(H):
                H.remove_edge(u, w)
                H.add_edge(u, v)
    
    elif action == "add_edge" and len(nodes) >= 2:
        u, v = random.sample(nodes, 2)
        if not H.has_edge(u, v):
            H.add_edge(u, v)
    
    elif action == "remove_edge" and H.number_of_edges() > 0:
        e = random.choice(list(H.edges()))
        H.remove_edge(*e)
        if 'connected' in subgroups and not nx.is_connected(H):
            H.add_edge(*e)
    
    return H


def _make_hard_to_dominate(H, subgroups):
    """Transforme H vers un graphe où γt est grand."""
    nodes = list(H.nodes())
    if len(nodes) < 4:
        return
    
    # Stratégie 1 : créer une structure de cycles disjoints (si non connexe OK)
    # ou un graphe où chaque nœud a exactement 2 voisins (cycle) → γt = n/2
    # Pour graphe connexe : chaîne de 4-cycles
    
    # Supprimer quelques arêtes haut-degré pour réduire la couverture
    high_deg = sorted(nodes, key=lambda v: H.degree(v), reverse=True)
    for v in high_deg[:len(high_deg)//3]:
        nbrs = list(H.neighbors(v))
        if len(nbrs) > 2:
            u = random.choice(nbrs)
            H.remove_edge(v, u)
            if 'connected' in subgroups and not nx.is_connected(H):
                H.add_edge(v, u)


# =====================================================================
# FIX 5 : SOLVE_HARD — VERSION SPÉCIALISÉE POUR LES CAS DIFFICILES
# =====================================================================

def is_hard_conjecture(conjecture):
    """Détecte si une conjecture fait partie des ~100 difficiles."""
    hard_invariants = {
        'total_domination_number',
        'independent_domination_number',
        'remoteness',
        'largest_distance_eigenvalue',
        'second_smallest_laplace_eigenvalue',
    }
    return (conjecture['X'] in hard_invariants or
            conjecture['Y'] in hard_invariants)


def solve_hard_conjecture(conjecture):
    """
    Solver spécialisé pour les ~100 conjectures difficiles.
    
    Différences avec solve_conjecture :
    1. Utilise compute_invariants_fixed (calculs exacts)
    2. Cherche dans la plage de tailles [10, 28] ciblée
    3. Mutations domination-aware
    4. SA avec température plus haute (exploration plus large)
    5. Beam de taille 5
    """
    import ast, time, os
    import pandas as pd
    import concurrent.futures
    
    # Import depuis graphbench_v2
    from graphbench_v2 import (
        parse_valeur_math, is_valid_class, generate_smart_initial_graph,
        mutate, repair, violation_score, compute_invariants
    )
    
    c_id = conjecture['Conjecture ID']
    subgroups = str(conjecture['Subgroup'])
    
    BEAM_SIZE = 5
    T_INIT = 3.0          # Plus haute → plus d'exploration
    T_MIN = 0.005
    COOLING = 0.997
    STAG_LIMIT = 200
    TIME_LIMIT = 60.0
    
    # Plages de tailles ciblées pour les cas difficiles
    TARGET_SIZES = list(range(10, 25)) + list(range(8, 10)) + list(range(25, 30))
    
    def score_fn(G):
        """Score utilisant les calculs exacts."""
        if not is_valid_class(G, subgroups):
            return -999
        inv = compute_invariants_fixed(G, conjecture)
        return _violation(inv, conjecture)
    
    start_time = time.time()
    
    # ── Init beam avec tailles ciblées ───────────────────────────────
    beam = []
    for n0 in TARGET_SIZES[:10]:
        if time.time() - start_time > 8:
            break
        for _ in range(3):
            G0 = generate_smart_initial_graph(subgroups, conjecture, n0)
            if is_valid_class(G0, subgroups):
                s0 = score_fn(G0)
                if s0 > -999:
                    beam.append((s0, G0))
    
    if not beam:
        G0 = generate_smart_initial_graph(subgroups, conjecture, 12)
        beam = [(score_fn(G0), G0)]
    
    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:BEAM_SIZE]
    
    best_score = beam[0][0]
    best_graph = beam[0][1]
    
    T = T_INIT
    iterations = 0
    stagnation = 0
    
    while (time.time() - start_time) < TIME_LIMIT:
        iterations += 1
        
        # Sélection pondérée dans le beam
        weights = [max(0.01, s + 20) for s, _ in beam]
        tw = sum(weights)
        r = random.random() * tw
        cumul = 0
        current_G = beam[0][1]
        for (s, G), w in zip(beam, weights):
            cumul += w
            if cumul >= r:
                current_G = G
                break
        
        # 60% mutation domination-aware, 40% mutation standard
        if is_hard_conjecture(conjecture) and random.random() < 0.6:
            H = mutate_domination_aware(current_G, subgroups, conjecture)
        else:
            H = mutate(current_G, subgroups, conjecture)
        
        if not H or not is_valid_class(H, subgroups):
            stagnation += 1
            continue
        
        score_H = score_fn(H)
        
        # Victoire
        if score_H > 0:
            t = time.time() - start_time
            H_clean = nx.convert_node_labels_to_integers(H)
            g6 = nx.to_graph6_bytes(H_clean, header=False).decode('ascii').strip()
            print(f"✅ HARD [{c_id}] Réfutée en {t:.2f}s ({iterations} it.)")
            return {"ID": c_id, "Status": "SUCCESS", "Time": t,
                    "Score": score_H, "Iterations": iterations, "Graph6": g6}
        
        # Acceptation SA
        delta = score_H - best_score
        if delta > 0:
            best_score = score_H
            best_graph = H
            stagnation = 0
        elif T > T_MIN and random.random() < math.exp(max(-50, delta / T)):
            pass
        else:
            stagnation += 1
        
        # Update beam
        beam.append((score_H, H))
        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:BEAM_SIZE]
        
        T = max(T_MIN, T * COOLING)
        
        # Redémarrage
        if stagnation > STAG_LIMIT:
            stagnation = 0
            T = T_INIT
            n_new = random.choice(TARGET_SIZES)
            G_new = generate_smart_initial_graph(subgroups, conjecture, n_new)
            if is_valid_class(G_new, subgroups):
                s_new = score_fn(G_new)
                beam.append((s_new, G_new))
                beam.sort(key=lambda x: x[0], reverse=True)
                beam = beam[:BEAM_SIZE]
    
    print(f"❌ HARD [{c_id}] Échec 60s ({iterations} it.) best={best_score:.4f}")
    return {"ID": c_id, "Status": "FAILURE", "Time": 120,
            "Score": best_score, "Iterations": iterations, "Graph6": ""}


def _violation(invariants, conjecture):
    """Réplique violation_score sans import circulaire."""
    import ast
    
    def parse_val(val):
        import pandas as pd
        if pd.isna(val): return 0.0
        s = str(val).strip()
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        try:
            return float(s)
        except:
            return 0.0
    
    x_val = invariants.get(conjecture['X'], 0)
    y_val = invariants.get(conjecture['Y'], 0)
    try:
        coefficients = ast.literal_eval(str(conjecture['Coefficients']))
    except Exception:
        coefficients = []
    f_x = parse_val(conjecture['Intercept'])
    for deg, coef_str in enumerate(coefficients, start=1):
        f_x += parse_val(coef_str) * (x_val ** deg)
    signe = conjecture['Sign']
    return (y_val - f_x) if signe == '<=' else (f_x - y_val)


# =====================================================================
# POINT D'ENTRÉE — DISPATCHER AUTOMATIQUE
# =====================================================================

def solve_conjecture_auto(conjecture):
    """
    Choisit automatiquement le solver selon la difficulté de la conjecture.
    """
    if is_hard_conjecture(conjecture):
        return solve_hard_conjecture(conjecture)
    else:
        from graphbench_v2 import solve_conjecture
        return solve_conjecture(conjecture)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import pandas as pd
    import concurrent.futures
    import os
    
    print("Chargement du benchmark...")
    df = pd.read_excel("benchmark/benchmark.xlsx")
    conjectures = [row for _, row in df.iterrows()]
    resultats = []
    
    nb_coeurs = os.cpu_count() or 4
    print(f"🚀 {nb_coeurs} cœurs — dispatcher auto hard/easy")
    print("=" * 60)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_coeurs) as executor:
        futures = {executor.submit(solve_conjecture_auto, c): c
                   for c in conjectures}
        for future in concurrent.futures.as_completed(futures):
            try:
                resultats.append(future.result())
            except Exception as e:
                print(f"⚠️  Erreur : {e}")
    
    df_res = pd.DataFrame(resultats)
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/resultats_final.csv", index=False)
    
    ok = len(df_res[df_res['Status'] == 'SUCCESS'])
    total_t = df_res['Time'].sum()
    print(f"\n{'='*60}")
    print(f"🏁 Score final : {ok}/{len(df)} | Coût total : {total_t:.1f}s")
    print(f"{'='*60}")
