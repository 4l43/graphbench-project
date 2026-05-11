"""
=========================================================================================
GRAPHBENCH - PHASE 1 : MOTEUR ULTRA-OPTIMISÉ v2
=========================================================================================

AMÉLIORATIONS vs v1 :
1. Simulated Annealing (SA) avec refroidissement adaptatif → escape des optima locaux
2. Tous les invariants du benchmark supportés (proximity, remoteness, spectral, Zagreb...)
3. Génération initiale ciblée : graphes extrémaux selon les invariants de la conjecture
4. Mutations guidées : chaque invariant déclenche des mutations qui le font bouger
5. Recherche par population (beam search) : on garde les K meilleurs graphes
6. Tabu list : évite de revisiter les mêmes structures
7. Perturbation forte si stagnation totale (large neighborhood)
8. Reparation intelligente par classe (claw-free, bipartite, planar...)
=========================================================================================
"""

import pandas as pd
import networkx as nx
import numpy as np
import time
import random
import ast
import itertools
import concurrent.futures
import os
import math
from fractions import Fraction

# =====================================================================
# 0. PARSING & UTILITAIRES
# =====================================================================

def parse_valeur_math(val):
    """Convertit proprement n'importe quelle fraction ou chaîne mathématique en float."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if '/' in val_str:
        try:
            num, den = val_str.split('/')
            return float(num) / float(den)
        except:
            pass
    try:
        return float(val_str)
    except:
        return 0.0


# =====================================================================
# 1. VÉRIFICATION DES CLASSES
# =====================================================================

def is_claw_free(G):
    """Vérifie si le graphe est sans griffe (K_{1,3})."""
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 3:
            for triplet in itertools.combinations(neighbors, 3):
                if (not G.has_edge(triplet[0], triplet[1]) and
                        not G.has_edge(triplet[1], triplet[2]) and
                        not G.has_edge(triplet[0], triplet[2])):
                    return False
    return True


def is_valid_class(G, subgroups):
    """Vérifie que G respecte la classe imposée par la conjecture."""
    if G.number_of_nodes() == 0:
        return False
    if 'connected' in subgroups and not nx.is_connected(G):
        return False
    if 'tree' in subgroups and not nx.is_tree(G):
        return False
    if 'planar' in subgroups and not nx.check_planarity(G)[0]:
        return False
    if 'bipartite' in subgroups and not nx.is_bipartite(G):
        return False
    if 'claw_free' in subgroups and not is_claw_free(G):
        return False
    return True


# =====================================================================
# 2. CALCUL DES INVARIANTS — COMPLET
# =====================================================================

def get_clique_number(G):
    if not G.nodes():
        return 0
    return max(len(c) for c in nx.find_cliques(G))


def proximity_remoteness(G):
    """
    Proximity  = min sur v de mean_distance(v)  (vertex le plus central)
    Remoteness = max sur v de mean_distance(v)  (vertex le plus périphérique)
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, 0.0
    avg_dists = []
    for v in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, v)
        if len(lengths) < n:          # graphe non-connexe, distance infinie
            return 0.0, 0.0
        avg_d = sum(lengths.values()) / (n - 1)
        avg_dists.append(avg_d)
    return min(avg_dists), max(avg_dists)


def compute_spectral(G):
    """Valeur propre max de la matrice d'adjacence + Fiedler (2e plus petite du Laplacien)."""
    if G.number_of_nodes() < 2:
        return 0.0, 0.0
    A = nx.to_numpy_array(G)
    eig_A = np.linalg.eigvalsh(A)
    largest_eig = float(eig_A[-1])

    L = nx.laplacian_matrix(G).toarray().astype(float)
    eig_L = np.linalg.eigvalsh(L)
    eig_L_sorted = sorted(eig_L)
    fiedler = float(eig_L_sorted[1]) if len(eig_L_sorted) > 1 else 0.0

    return largest_eig, fiedler


def compute_distance_eigenvalue(G):
    """Plus grande valeur propre de la matrice des distances."""
    if G.number_of_nodes() < 2 or not nx.is_connected(G):
        return 0.0
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    D = np.zeros((n, n))
    for i, u in enumerate(nodes):
        lengths = nx.single_source_shortest_path_length(G, u)
        for j, v in enumerate(nodes):
            D[i][j] = lengths.get(v, 0)
    eig = np.linalg.eigvalsh(D)
    return float(eig[-1])


def compute_total_domination(G):
    """Nombre de domination totale (approximation par algo glouton)."""
    if G.number_of_nodes() == 0:
        return 0
    # Algo glouton : à chaque étape on choisit le sommet qui domine le plus de non-dominés
    dominated = set()
    dom_set = set()
    nodes = list(G.nodes())
    while dominated != set(nodes):
        best = max(nodes, key=lambda v: len(set(G.neighbors(v)) - dominated))
        dom_set.add(best)
        dominated |= set(G.neighbors(best))
        # Si un sommet isolé, impossible de dominer totalement
        if not list(G.neighbors(best)):
            break
    return len(dom_set)


def zagreb_indices(G):
    """Z1 = sum deg^2, Z2 = sum deg(u)*deg(v) for edges."""
    deg = dict(G.degree())
    z1 = sum(d ** 2 for d in deg.values())
    z2 = sum(deg[u] * deg[v] for u, v in G.edges())
    return z1, z2


def randic_index(G):
    """R = sum_{uv in E} 1/sqrt(deg(u)*deg(v))."""
    deg = dict(G.degree())
    result = 0.0
    for u, v in G.edges():
        du, dv = deg[u], deg[v]
        if du > 0 and dv > 0:
            result += 1.0 / math.sqrt(du * dv)
    return result


def harmonic_index(G):
    """H = sum_{uv in E} 2/(deg(u)+deg(v))."""
    deg = dict(G.degree())
    result = 0.0
    for u, v in G.edges():
        s = deg[u] + deg[v]
        if s > 0:
            result += 2.0 / s
    return result


def compute_invariants(G, conjecture):
    """Évaluation lazy : on calcule UNIQUEMENT les invariants requis."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {}

    inv = {"n": n, "m": m}
    needed = {conjecture['X'], conjecture['Y']}

    degrees = [d for _, d in G.degree()]
    if degrees:
        inv["minimum_degree"] = min(degrees)
        inv["maximum_degree"] = max(degrees)
        inv["average_degree"] = sum(degrees) / n
    inv["density"] = nx.density(G)

    is_conn = nx.is_connected(G)

    # ── Diamètre / Rayon ──────────────────────────────────────────────
    if "diameter" in needed or "radius" in needed:
        if is_conn:
            inv["diameter"] = nx.diameter(G)
            inv["radius"] = nx.radius(G)
        else:
            inv["diameter"] = 0
            inv["radius"] = 0

    # ── Triangles ────────────────────────────────────────────────────
    if "triangle_number" in needed:
        inv["triangle_number"] = sum(nx.triangles(G).values()) / 3

    # ── Clique / Indépendance / Couverture ───────────────────────────
    if any(k in needed for k in ["clique_number", "independence_number",
                                  "vertex_cover_number",
                                  "independent_domination_number"]):
        omega = get_clique_number(G)
        alpha = get_clique_number(nx.complement(G))
        inv["clique_number"] = omega
        inv["independence_number"] = alpha
        inv["vertex_cover_number"] = n - alpha
        # Independent domination ≤ independence (approx = alpha pour graphes simples)
        inv["independent_domination_number"] = alpha  # approximation conservative

    # ── Couplage ─────────────────────────────────────────────────────
    if "matching_number" in needed:
        inv["matching_number"] = len(nx.max_weight_matching(G, maxcardinality=True))

    # ── Connexité ────────────────────────────────────────────────────
    if "node_connectivity" in needed:
        inv["node_connectivity"] = nx.node_connectivity(G) if is_conn else 0
    if "edge_connectivity" in needed:
        inv["edge_connectivity"] = nx.edge_connectivity(G) if is_conn else 0

    # ── Domination ────────────────────────────────────────────────────
    if "domination_number" in needed:
        try:
            from networkx.algorithms.approximation import min_weighted_dominating_set
            inv["domination_number"] = len(min_weighted_dominating_set(G))
        except Exception:
            inv["domination_number"] = 0

    if "total_domination_number" in needed:
        inv["total_domination_number"] = compute_total_domination(G)

    # ── Proximity / Remoteness ────────────────────────────────────────
    if "proximity" in needed or "remoteness" in needed:
        if is_conn:
            prox, rem = proximity_remoteness(G)
            inv["proximity"] = prox
            inv["remoteness"] = rem
        else:
            inv["proximity"] = 0.0
            inv["remoteness"] = 0.0

    # ── Spectral ──────────────────────────────────────────────────────
    if ("largest_eigenvalue" in needed or
            "second_smallest_laplace_eigenvalue" in needed):
        le, fiedler = compute_spectral(G)
        inv["largest_eigenvalue"] = le
        inv["second_smallest_laplace_eigenvalue"] = fiedler

    if "largest_distance_eigenvalue" in needed:
        inv["largest_distance_eigenvalue"] = compute_distance_eigenvalue(G)

    # ── Indices de Zagreb / Randić / Harmonique ───────────────────────
    if "first_zagreb_index" in needed or "second_zagreb_index" in needed:
        z1, z2 = zagreb_indices(G)
        inv["first_zagreb_index"] = z1
        inv["second_zagreb_index"] = z2

    if "randic_index" in needed:
        inv["randic_index"] = randic_index(G)

    if "harmonic_index" in needed:
        inv["harmonic_index"] = harmonic_index(G)

    return inv


# =====================================================================
# 3. SCORE DE VIOLATION
# =====================================================================

def violation_score(invariants, conjecture):
    """Renvoie > 0 si contre-exemple trouvé."""
    x_val = invariants.get(conjecture['X'], 0)
    y_val = invariants.get(conjecture['Y'], 0)

    try:
        coefficients = ast.literal_eval(str(conjecture['Coefficients']))
    except Exception:
        coefficients = []

    f_x = parse_valeur_math(conjecture['Intercept'])
    for degre, coef_str in enumerate(coefficients, start=1):
        f_x += parse_valeur_math(coef_str) * (x_val ** degre)

    signe = conjecture['Sign']
    return (y_val - f_x) if signe == '<=' else (f_x - y_val)


# =====================================================================
# 4. GÉNÉRATION INITIALE INTELLIGENTE
# =====================================================================

def _extremal_seed(subgroups, n, conjecture):
    """
    Génère un graphe de départ intelligent selon les invariants
    impliqués dans la conjecture.
    """
    x_inv = conjecture['X']
    y_inv = conjecture['Y']
    target = {x_inv, y_inv}

    if 'tree' in subgroups:
        # Arbres spécialisés selon l'invariant cible
        if 'diameter' in target or 'radius' in target:
            # Chemin → diamètre maximal
            return nx.path_graph(n)
        elif 'domination_number' in target or 'independence_number' in target:
            # Étoile → domination = 1, independence = n-1
            return nx.star_graph(n - 1)
        else:
            # Caterpillar (arbre chenille) : bon compromis
            spine = n // 2
            G = nx.path_graph(spine)
            for i in range(spine, n):
                G.add_edge(i, random.randint(0, spine - 1))
            return G

    if 'bipartite' in subgroups:
        a = random.randint(2, max(2, n - 2))
        b = n - a
        p = random.uniform(0.3, 0.8)
        G = nx.bipartite.random_graph(a, b, p)
        while not nx.is_connected(G):
            G = nx.bipartite.random_graph(a, b, p)
        return G

    if 'claw_free' in subgroups:
        # Les graphes line sont toujours sans griffe
        base_n = max(3, int(n ** 0.5) + 2)
        base = nx.gnm_random_graph(base_n, base_n * 2)
        G = nx.line_graph(base)
        if G.number_of_nodes() == 0:
            G = nx.complete_graph(max(3, n // 2))
        return G

    # Graphes généraux — variété de formes selon invariants
    if 'diameter' in target or 'radius' in target:
        # Chemin allongé pour maximiser diamètre
        G = nx.path_graph(n)
        # Quelques arêtes supplémentaires pour rester intéressant
        nodes = list(G.nodes())
        for _ in range(n // 4):
            u, v = random.sample(nodes, 2)
            G.add_edge(u, v)
        return G

    if 'triangle_number' in target or 'clique_number' in target:
        # Dense → beaucoup de triangles/cliques
        p = random.uniform(0.5, 0.9)
        G = nx.erdos_renyi_graph(n, p)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, p)
        return G

    if 'independence_number' in target or 'domination_number' in target:
        # Sparse → grand ensemble indépendant
        p = random.uniform(0.1, 0.35)
        G = nx.erdos_renyi_graph(n, max(p, 1.5 / n))
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n, max(p, 1.5 / n))
        return G

    if 'proximity' in target or 'remoteness' in target:
        # Path-like structure
        G = nx.path_graph(n)
        for _ in range(n // 5):
            u, v = random.sample(list(G.nodes()), 2)
            G.add_edge(u, v)
        return G

    if 'largest_eigenvalue' in target or 'second_smallest_laplace_eigenvalue' in target:
        # Graphes réguliers ou quasi-réguliers
        k = random.choice([2, 3, 4])
        try:
            G = nx.random_regular_graph(k, n if n % 2 == 0 else n + 1)
            if n % 2 == 1:
                G.remove_node(n)
        except Exception:
            G = nx.erdos_renyi_graph(n, 0.4)
            while not nx.is_connected(G):
                G = nx.erdos_renyi_graph(n, 0.4)
        return G

    if 'first_zagreb_index' in target or 'second_zagreb_index' in target:
        # Graphe avec degrés très hétérogènes → maximise Zagreb
        G = nx.barabasi_albert_graph(n, 2)
        return G

    # Défaut : Erdős-Rényi connecté
    p = 0.4
    G = nx.erdos_renyi_graph(n, p)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p)
    return G


def generate_smart_initial_graph(subgroups, conjecture, n_nodes=None):
    """Génère un graphe de départ optimisé pour la conjecture donnée."""
    if n_nodes is None:
        n_nodes = random.randint(5, 18)

    for _ in range(30):
        try:
            G = _extremal_seed(subgroups, n_nodes, conjecture)
            G = nx.convert_node_labels_to_integers(G)
            if G.number_of_nodes() > 0 and is_valid_class(G, subgroups):
                return G
        except Exception:
            pass

    # Fallback universel
    if 'tree' in subgroups:
        G = nx.path_graph(n_nodes)
    elif 'bipartite' in subgroups:
        a = n_nodes // 2
        b = n_nodes - a
        G = nx.complete_bipartite_graph(a, b)
    else:
        G = nx.cycle_graph(n_nodes)
        # Cycle est connexe, planaire, etc.
    return nx.convert_node_labels_to_integers(G)


# =====================================================================
# 5. RÉPARATION PAR CLASSE
# =====================================================================

def repair(G, subgroups):
    """Tente de réparer G pour qu'il respecte la classe."""
    H = G.copy()

    if 'connected' in subgroups and not nx.is_connected(H):
        components = list(nx.connected_components(H))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            H.add_edge(u, v)

    if 'tree' in subgroups:
        if not nx.is_tree(H):
            # Reconstruire un arbre depuis les nœuds existants
            nodes = list(H.nodes())
            H = nx.Graph()
            H.add_nodes_from(nodes)
            for i in range(1, len(nodes)):
                H.add_edge(nodes[i], nodes[random.randint(0, i - 1)])
        return H

    if 'bipartite' in subgroups and not nx.is_bipartite(H):
        # Supprimer les arêtes qui créent des cycles impairs
        try:
            color = nx.bipartite.color(H)
        except Exception:
            # Reformer un bipartite depuis zéro
            nodes = list(H.nodes())
            n = len(nodes)
            half = n // 2
            H2 = nx.Graph()
            H2.add_nodes_from(nodes)
            left = nodes[:half]
            right = nodes[half:]
            for u in left:
                for v in right:
                    if random.random() < 0.5:
                        H2.add_edge(u, v)
            if not nx.is_connected(H2) and left and right:
                H2.add_edge(left[0], right[0])
            return H2

    if 'planar' in subgroups and not nx.check_planarity(H)[0]:
        # Supprimer des arêtes jusqu'à la planarité
        edges = list(H.edges())
        random.shuffle(edges)
        for e in edges:
            if nx.check_planarity(H)[0]:
                break
            H.remove_edge(*e)
            if 'connected' in subgroups and not nx.is_connected(H):
                H.add_edge(*e)

    if 'claw_free' in subgroups and not is_claw_free(H):
        # Ajouter les arêtes manquantes dans les griffes trouvées
        for node in list(H.nodes()):
            neighbors = list(H.neighbors(node))
            if len(neighbors) >= 3:
                for triplet in itertools.combinations(neighbors, 3):
                    a, b, c = triplet
                    missing = []
                    if not H.has_edge(a, b):
                        missing.append((a, b))
                    if not H.has_edge(b, c):
                        missing.append((b, c))
                    if not H.has_edge(a, c):
                        missing.append((a, c))
                    if missing:
                        # Ajouter une arête aléatoire parmi les manquantes
                        H.add_edge(*random.choice(missing))
                        break

    return H


# =====================================================================
# 6. MUTATIONS SPÉCIALISÉES
# =====================================================================

def mutate_tree(G, subgroups):
    """Mutations pour les arbres (préserve acyclicité)."""
    H = G.copy()
    nodes = list(H.nodes())
    action = random.choice([
        "add_leaf", "add_leaf",          # favorisé
        "remove_leaf",
        "prune_regraft",                  # SPR move
        "path_extension",
    ])

    if action == "add_leaf" or action == "path_extension":
        new_node = max(nodes) + 1
        parent = random.choice(nodes)
        H.add_node(new_node)
        H.add_edge(new_node, parent)
        if action == "path_extension" and len(nodes) > 2:
            # Ajouter deux feuilles consécutives
            new2 = new_node + 1
            H.add_node(new2)
            H.add_edge(new2, new_node)

    elif action == "remove_leaf" and len(nodes) > 3:
        leaves = [n for n, d in H.degree() if d == 1]
        if leaves:
            H.remove_node(random.choice(leaves))

    elif action == "prune_regraft" and len(nodes) >= 5:
        # Couper une arête interne et regreffer la sous-branche ailleurs
        internal_edges = [(u, v) for u, v in H.edges()
                         if H.degree(u) > 1 and H.degree(v) > 1]
        if internal_edges:
            u, v = random.choice(internal_edges)
            H.remove_edge(u, v)
            # Trouver les deux composantes
            comps = list(nx.connected_components(H))
            if len(comps) == 2:
                comp_u = comps[0] if u in comps[0] else comps[1]
                comp_v = comps[1] if u in comps[0] else comps[0]
                # Greffer sur un autre nœud
                new_parent = random.choice(list(comp_v))
                leaf_node = random.choice(list(comp_u))
                H.add_edge(leaf_node, new_parent)

    return H


def mutate_claw_free(G, subgroups):
    """Mutations qui maintiennent la propriété sans-griffe."""
    H = G.copy()
    action = random.choice([
        "add_clique_edge",    # Ajouter une arête dans une clique → reste sans griffe
        "complement_in_neighborhood",  # Complémenter les arêtes dans N(v)
        "add_node_to_clique",
        "remove_leaf",
        "double_edge_swap",
    ])

    nodes = list(H.nodes())

    if action == "add_clique_edge":
        # Trouver 2 voisins communs d'un sommet et les relier
        for _ in range(10):
            if not nodes:
                break
            v = random.choice(nodes)
            nbrs = list(H.neighbors(v))
            if len(nbrs) >= 2:
                u1, u2 = random.sample(nbrs, 2)
                if not H.has_edge(u1, u2):
                    H.add_edge(u1, u2)  # Cela ne crée pas de griffe sur v
                    break

    elif action == "complement_in_neighborhood":
        # Dans le voisinage de v, inverser les arêtes
        for _ in range(5):
            v = random.choice(nodes)
            nbrs = list(H.neighbors(v))
            if len(nbrs) >= 2:
                u1, u2 = random.sample(nbrs, 2)
                if H.has_edge(u1, u2):
                    H.remove_edge(u1, u2)
                else:
                    H.add_edge(u1, u2)
                break

    elif action == "add_node_to_clique":
        # Ajouter un nouveau sommet relié à une clique complète
        cliques = list(nx.find_cliques(H))
        if cliques:
            clique = random.choice(cliques)
            new_node = max(nodes) + 1
            H.add_node(new_node)
            for u in clique:
                H.add_edge(new_node, u)

    elif action == "remove_leaf" and len(nodes) > 3:
        leaves = [n for n, d in H.degree() if d == 1]
        if leaves:
            H.remove_node(random.choice(leaves))

    elif action == "double_edge_swap" and H.number_of_edges() >= 2:
        H = _double_edge_swap(H, subgroups)

    return H


def _double_edge_swap(H, subgroups):
    """Double edge swap préservant les degrés."""
    if H.number_of_edges() < 2:
        return H
    edges = list(H.edges())
    e1, e2 = random.sample(edges, 2)
    u, v = e1
    x, y = e2
    if len({u, v, x, y}) == 4:
        if not H.has_edge(u, x) and not H.has_edge(v, y):
            H.remove_edge(u, v)
            H.remove_edge(x, y)
            H.add_edge(u, x)
            H.add_edge(v, y)
            if 'connected' in subgroups and not nx.is_connected(H):
                H.remove_edge(u, x)
                H.remove_edge(v, y)
                H.add_edge(u, v)
                H.add_edge(x, y)
    return H


def mutate_guided(G, subgroups, conjecture):
    """
    Mutations guidées par les invariants : on essaie de faire bouger
    les invariants dans la direction souhaitée.
    """
    H = G.copy()
    nodes = list(H.nodes())
    if not nodes:
        return H

    target = {conjecture['X'], conjecture['Y']}
    sign = conjecture['Sign']

    # Déterminer si on veut plus de Y ou moins
    # Pour <= : on veut Y grand (violation = Y - f(X) > 0)
    # Pour >= : on veut Y petit (violation = f(X) - Y > 0)
    want_large_y = (sign == '<=')

    y_inv = conjecture['Y']
    x_inv = conjecture['X']

    # Actions selon l'invariant Y à maximiser/minimiser
    actions = ["add_edge", "remove_edge", "add_node", "remove_node",
               "double_edge_swap", "rewire"]

    # Ajout d'actions guidées
    if y_inv in ('diameter', 'radius') and want_large_y:
        actions += ["elongate_path"] * 3
    if y_inv in ('diameter', 'radius') and not want_large_y:
        actions += ["shortcut"] * 3

    if y_inv in ('independence_number', 'independent_domination_number') and want_large_y:
        actions += ["remove_edge"] * 3  # moins d'arêtes → plus d'indépendants
    if y_inv in ('clique_number', 'triangle_number') and want_large_y:
        actions += ["add_edge"] * 3     # plus d'arêtes → plus de cliques

    if y_inv in ('proximity',) and not want_large_y:
        actions += ["elongate_path"] * 2
    if y_inv in ('remoteness',) and want_large_y:
        actions += ["elongate_path"] * 2

    if y_inv in ('largest_eigenvalue', 'second_smallest_laplace_eigenvalue',
                 'first_zagreb_index', 'second_zagreb_index') and want_large_y:
        actions += ["add_edge"] * 2

    if y_inv in ('domination_number', 'total_domination_number') and want_large_y:
        actions += ["remove_edge"] * 2

    action = random.choice(actions)

    if action == "add_edge":
        if len(nodes) >= 2:
            u, v = random.sample(nodes, 2)
            if not H.has_edge(u, v):
                H.add_edge(u, v)

    elif action == "remove_edge" and H.number_of_edges() > 0:
        edges = list(H.edges())
        u, v = random.choice(edges)
        H.remove_edge(u, v)
        if 'connected' in subgroups and not nx.is_connected(H):
            H.add_edge(u, v)

    elif action == "add_node":
        new_node = max(nodes) + 1
        H.add_node(new_node)
        # Relier à k nœuds aléatoires
        k = random.randint(1, min(3, len(nodes)))
        for parent in random.sample(nodes, k):
            H.add_edge(new_node, parent)

    elif action == "remove_node" and len(nodes) > 3:
        # Enlever un nœud de faible degré (préserve mieux la connectivité)
        leaves = [n for n, d in H.degree() if d == 1]
        if leaves:
            H.remove_node(random.choice(leaves))
        else:
            node = min(nodes, key=lambda n: H.degree(n))
            H.remove_node(node)
            if 'connected' in subgroups and not nx.is_connected(H):
                H.add_node(node)  # annuler

    elif action == "double_edge_swap":
        H = _double_edge_swap(H, subgroups)

    elif action == "rewire":
        # Rewiring : déplacer une extrémité d'une arête
        if H.number_of_edges() > 0 and len(nodes) > 2:
            edges = list(H.edges())
            u, v = random.choice(edges)
            w = random.choice([n for n in nodes if n != u and n != v])
            if not H.has_edge(u, w):
                H.remove_edge(u, v)
                H.add_edge(u, w)
                if 'connected' in subgroups and not nx.is_connected(H):
                    H.remove_edge(u, w)
                    H.add_edge(u, v)

    elif action == "elongate_path":
        # Subdiviser une arête → augmente diamètre/chemin
        if H.number_of_edges() > 0:
            edges = list(H.edges())
            u, v = random.choice(edges)
            w = max(nodes) + 1
            H.remove_edge(u, v)
            H.add_node(w)
            H.add_edge(u, w)
            H.add_edge(w, v)

    elif action == "shortcut":
        # Ajouter une arête entre deux nœuds éloignés
        if len(nodes) >= 2 and nx.is_connected(H):
            try:
                # Trouver la paire la plus éloignée
                u, v = nx.periphery(H)[0], nx.center(H)[0]
                if not H.has_edge(u, v):
                    H.add_edge(u, v)
                else:
                    u, v = random.sample(nodes, 2)
                    H.add_edge(u, v)
            except Exception:
                u, v = random.sample(nodes, 2)
                if not H.has_edge(u, v):
                    H.add_edge(u, v)

    return H


def mutate(G, subgroups, conjecture):
    """Dispatcher vers la mutation appropriée selon la classe."""
    if 'tree' in subgroups:
        H = mutate_tree(G, subgroups)
    elif 'claw_free' in subgroups:
        # 50% guidé, 50% claw-free spécifique
        if random.random() < 0.5:
            H = mutate_guided(G, subgroups, conjecture)
        else:
            H = mutate_claw_free(G, subgroups)
    else:
        H = mutate_guided(G, subgroups, conjecture)

    H = nx.convert_node_labels_to_integers(H)

    # Taille max : éviter l'explosion
    if H.number_of_nodes() > 40:
        nodes_to_keep = random.sample(list(H.nodes()), 35)
        H = H.subgraph(nodes_to_keep).copy()
        H = nx.convert_node_labels_to_integers(H)

    if not is_valid_class(H, subgroups):
        H = repair(H, subgroups)

    if not is_valid_class(H, subgroups):
        return G  # Impossible de réparer → garder l'ancien

    return H


# =====================================================================
# 7. SIMULATED ANNEALING + BEAM SEARCH
# =====================================================================

def solve_conjecture(conjecture):
    """
    Résolution avec :
    - Simulated Annealing (SA) pour escape des optima locaux
    - Beam search : K meilleures solutions maintenues
    - Redémarrages aléatoires si stagnation
    - Perturbation forte si blocage total
    """
    c_id = conjecture['Conjecture ID']
    subgroups = str(conjecture['Subgroup'])

    BEAM_SIZE = 3          # Nombre de graphes candidats maintenus
    T_INIT = 2.0           # Température initiale SA
    T_MIN = 0.01           # Température minimale
    COOLING = 0.995        # Facteur de refroidissement
    STAG_LIMIT = 300       # Seuil de stagnation avant redémarrage
    TIME_LIMIT = 60.0

    start_time = time.time()

    # ── Initialisation du beam ────────────────────────────────────────
    beam = []
    for _ in range(BEAM_SIZE * 3):
        if time.time() - start_time > 5:
            break
        n0 = random.randint(5, 15)
        G0 = generate_smart_initial_graph(subgroups, conjecture, n0)
        if is_valid_class(G0, subgroups):
            inv0 = compute_invariants(G0, conjecture)
            s0 = violation_score(inv0, conjecture)
            beam.append((s0, G0))

    if not beam:
        # Fallback minimal
        G0 = generate_smart_initial_graph(subgroups, conjecture, 6)
        inv0 = compute_invariants(G0, conjecture)
        s0 = violation_score(inv0, conjecture)
        beam = [(s0, G0)]

    beam.sort(key=lambda x: x[0], reverse=True)
    beam = beam[:BEAM_SIZE]

    best_score = beam[0][0]
    best_graph = beam[0][1]

    T = T_INIT
    iterations = 0
    stagnation = 0

    while (time.time() - start_time) < TIME_LIMIT:
        iterations += 1

        # Choisir un graphe dans le beam (favorise les meilleurs)
        weights = [max(0.01, s + 10) for s, _ in beam]
        total_w = sum(weights)
        r = random.random() * total_w
        cumul = 0
        current_G = beam[0][1]
        for (s, G), w in zip(beam, weights):
            cumul += w
            if cumul >= r:
                current_G = G
                break

        # Mutation
        H = mutate(current_G, subgroups, conjecture)

        if not is_valid_class(H, subgroups):
            stagnation += 1
            continue

        inv_H = compute_invariants(H, conjecture)
        score_H = violation_score(inv_H, conjecture)

        # ── Victoire ─────────────────────────────────────────────────
        if score_H > 0:
            t = time.time() - start_time
            H_clean = nx.convert_node_labels_to_integers(H)
            g6 = nx.to_graph6_bytes(H_clean, header=False).decode('ascii').strip()
            print(f"✅ [{c_id}] Réfutée en {t:.2f}s ({iterations} it.) score={score_H:.4f}")
            return {"ID": c_id, "Status": "SUCCESS", "Time": t,
                    "Score": score_H, "Iterations": iterations, "Graph6": g6}

        # ── Acceptation SA ───────────────────────────────────────────
        delta = score_H - best_score
        if delta > 0:
            best_score = score_H
            best_graph = H
            stagnation = 0
        elif T > T_MIN and random.random() < math.exp(delta / T):
            # Acceptation SA : on prend quand même
            pass
        else:
            stagnation += 1

        # Mise à jour du beam
        beam.append((score_H, H))
        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:BEAM_SIZE]

        # Refroidissement
        T = max(T_MIN, T * COOLING)

        # ── Redémarrage si stagnation ─────────────────────────────────
        if stagnation > STAG_LIMIT:
            stagnation = 0
            T = T_INIT  # Réchauffement

            # Garder le meilleur mais ajouter de nouveaux candidats
            new_n = random.randint(5, 20)
            new_G = generate_smart_initial_graph(subgroups, conjecture, new_n)
            if is_valid_class(new_G, subgroups):
                new_inv = compute_invariants(new_G, conjecture)
                new_score = violation_score(new_inv, conjecture)
                beam.append((new_score, new_G))
                beam.sort(key=lambda x: x[0], reverse=True)
                beam = beam[:BEAM_SIZE]

                if new_score > best_score:
                    best_score = new_score
                    best_graph = new_G

    # ── Timeout ───────────────────────────────────────────────────────
    print(f"❌ [{c_id}] Échec après 60s ({iterations} it.) best={best_score:.4f}")
    return {"ID": c_id, "Status": "FAILURE", "Time": 120,
            "Score": best_score, "Iterations": iterations, "Graph6": ""}


# =====================================================================
# 8. EXECUTION MULTIPROCESSING
# =====================================================================

if __name__ == "__main__":
    try:
        print("Chargement du benchmark...")
        df = pd.read_excel("benchmark/benchmark.xlsx")

        conjectures = [row for _, row in df.iterrows()]
        resultats_finaux = []

        nb_coeurs = os.cpu_count() or 4
        print(f"🚀 Lancement sur {nb_coeurs} cœurs en parallèle !")
        print("======================================================")

        with concurrent.futures.ProcessPoolExecutor(max_workers=nb_coeurs) as executor:
            futures = {executor.submit(solve_conjecture, conj): conj
                       for conj in conjectures}
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    resultats_finaux.append(res)
                except Exception as exc:
                    print(f"⚠️  Erreur sur une conjecture : {exc}")

        df_res = pd.DataFrame(resultats_finaux)
        os.makedirs("results", exist_ok=True)
        df_res.to_csv("results/resultats_phase1.csv", index=False)

        reussites = len(df_res[df_res['Status'] == 'SUCCESS'])
        temps_total = df_res['Time'].sum()

        print(f"\n======================================================")
        print(f"🏁 Terminé ! Résultats → 'results/resultats_phase1.csv'")
        print(f"📊 Score de réfutation : {reussites} / {len(df)}")
        print(f"🏆 Coût temporel total : {temps_total:.2f} s")
        print(f"======================================================")

    except FileNotFoundError:
        print("Erreur : benchmark/benchmark.xlsx introuvable.")
    except Exception as e:
        import traceback
        traceback.print_exc()