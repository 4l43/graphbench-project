"""
invariants.py
Calcule tous les invariants de graphe nécessaires au benchmark.
"""

import networkx as nx
import numpy as np


# ─── Invariants de base ──────────────────────────────────────────────

def n(G):
    return G.number_of_nodes()

def m(G):
    return G.number_of_edges()

def diameter(G):
    if not nx.is_connected(G):
        return float('inf')
    return nx.diameter(G)

def radius(G):
    if not nx.is_connected(G):
        return float('inf')
    return nx.radius(G)

def minimum_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return min(d for _, d in G.degree())

def maximum_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return max(d for _, d in G.degree())

def average_degree(G):
    if G.number_of_nodes() == 0:
        return 0.0
    return 2 * G.number_of_edges() / G.number_of_nodes()

def density(G):
    n_ = G.number_of_nodes()
    if n_ < 2:
        return 0.0
    return 2 * G.number_of_edges() / (n_ * (n_ - 1))

def triangle_number(G):
    tris = nx.triangles(G)
    return sum(tris.values()) // 3

def clique_number(G):
    return max((len(c) for c in nx.find_cliques(G)), default=1)

# ─── Invariants de domination et couverture ──────────────────────────

def domination_number(G):
    """Nombre de domination γ(G) — approché par greedy."""
    nodes = set(G.nodes())
    dominated = set()
    domset = set()
    # Tri par degré décroissant
    for v in sorted(nodes, key=lambda x: G.degree(x), reverse=True):
        if v not in dominated:
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)

def total_domination_number(G):
    """Nombre de domination totale — chaque sommet doit avoir un voisin dans le domset."""
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0
    dominated = set()
    domset = set()
    for v in sorted(nodes, key=lambda x: G.degree(x), reverse=True):
        if v not in dominated:
            domset.add(v)
            dominated.update(G.neighbors(v))
        if dominated == set(nodes):
            break
    # Vérification et correction
    if not all(any(u in domset for u in G.neighbors(v)) for v in nodes):
        # Fallback: on ajoute des sommets manquants
        for v in nodes:
            if not any(u in domset for u in G.neighbors(v)):
                # Ajouter le voisin à plus haut degré
                nbrs = list(G.neighbors(v))
                if nbrs:
                    domset.add(max(nbrs, key=lambda x: G.degree(x)))
    return len(domset)

def independence_number(G):
    """Taille du plus grand ensemble indépendant α(G)."""
    return len(nx.maximal_independent_set(G))

def vertex_cover_number(G):
    """Taille de la couverture minimum par sommets τ(G) = n - α(G) (König pour bipartis)."""
    # Par le théorème de Gallai: τ = n - α
    alpha = independence_number(G)
    return G.number_of_nodes() - alpha

def independent_domination_number(G):
    """Nombre de domination indépendante = α dans un domset indépendant maximal."""
    # Un ensemble indépendant maximal est aussi un ensemble dominant indépendant
    ind_set = nx.maximal_independent_set(G)
    return len(ind_set)

def matching_number(G):
    """Taille du couplage maximum μ(G)."""
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)

# ─── Invariants spectraux ────────────────────────────────────────────

def largest_eigenvalue(G):
    """Plus grande valeur propre de la matrice d'adjacence."""
    if G.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(G)
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(eigvals))

def second_smallest_laplace_eigenvalue(G):
    """Connectivité algébrique λ₂(L) — valeur propre de Fiedler."""
    if G.number_of_nodes() < 2:
        return 0.0
    try:
        return float(nx.algebraic_connectivity(G))
    except Exception:
        return 0.0

def largest_distance_eigenvalue(G):
    """Plus grande valeur propre de la matrice des distances."""
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    D = nx.floyd_warshall_numpy(G)
    eigvals = np.linalg.eigvalsh(D)
    return float(np.max(eigvals))

# ─── Indices topologiques ────────────────────────────────────────────

def randic_index(G):
    """Indice de Randić."""
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du > 0 and dv > 0:
            total += 1.0 / (du * dv) ** 0.5
    return total

def harmonic_index(G):
    """Indice harmonique."""
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du + dv > 0:
            total += 2.0 / (du + dv)
    return total

def first_zagreb_index(G):
    """Premier indice de Zagreb M₁ = Σ deg(v)²."""
    return sum(d ** 2 for _, d in G.degree())

def second_zagreb_index(G):
    """Deuxième indice de Zagreb M₂ = Σ_{uv∈E} deg(u)·deg(v)."""
    return sum(G.degree(u) * G.degree(v) for u, v in G.edges())

# ─── Invariants de distance ──────────────────────────────────────────

def _distance_matrix(G):
    """Retourne la matrice des distances (dict de dicts)."""
    return dict(nx.all_pairs_shortest_path_length(G))

def proximity(G):
    """Proximité: min sur v de (1/(n-1)) * Σ_{u≠v} 1/d(u,v)."""
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    n_ = G.number_of_nodes()
    dist = _distance_matrix(G)
    min_prox = float('inf')
    for v in G.nodes():
        s = sum(1.0 / dist[v][u] for u in G.nodes() if u != v)
        prox_v = s / (n_ - 1)
        min_prox = min(min_prox, prox_v)
    return min_prox

def remoteness(G):
    """Éloignement: max sur v de (1/(n-1)) * Σ_{u≠v} d(u,v)."""
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    n_ = G.number_of_nodes()
    dist = _distance_matrix(G)
    max_rem = 0.0
    for v in G.nodes():
        avg_dist = sum(dist[v][u] for u in G.nodes() if u != v) / (n_ - 1)
        max_rem = max(max_rem, avg_dist)
    return max_rem

# ─── Fonction principale ─────────────────────────────────────────────

# Correspondance nom (benchmark) → fonction
INVARIANT_FUNCTIONS = {
    "order":                        n,
    "n":                            n,
    "size":                         m,
    "m":                            m,
    "diameter":                     diameter,
    "radius":                       radius,
    "minimum_degree":               minimum_degree,
    "maximum_degree":               maximum_degree,
    "average_degree":               average_degree,
    "density":                      density,
    "triangle_number":              triangle_number,
    "clique_number":                clique_number,
    "domination_number":            domination_number,
    "total_domination_number":      total_domination_number,
    "independence_number":          independence_number,
    "vertex_cover_number":          vertex_cover_number,
    "independent_domination_number": independent_domination_number,
    "matching_number":              matching_number,
    "largest_eigenvalue":           largest_eigenvalue,
    "second_smallest_laplace_eigenvalue": second_smallest_laplace_eigenvalue,
    "largest_distance_eigenvalue":  largest_distance_eigenvalue,
    "randic_index":                 randic_index,
    "harmonic_index":               harmonic_index,
    "first_zagreb_index":           first_zagreb_index,
    "second_zagreb_index":          second_zagreb_index,
    "proximity":                    proximity,
    "remoteness":                   remoteness,
}

# Invariants "lents" (coûteux à calculer)
SLOW_INVARIANTS = {
    "clique_number",
    "largest_distance_eigenvalue",
    "proximity",
    "remoteness",
    "second_smallest_laplace_eigenvalue",
}


def compute_invariants(G: nx.Graph, needed: set = None, fast_only: bool = False) -> dict:
    """
    Calcule les invariants demandés pour le graphe G.
    Si needed=None, calcule tout.
    Si fast_only=True, saute les invariants lents.
    """
    results = {}
    targets = needed if needed else set(INVARIANT_FUNCTIONS.keys())

    for name in targets:
        if fast_only and name in SLOW_INVARIANTS:
            continue
        func = INVARIANT_FUNCTIONS.get(name)
        if func is None:
            continue
        try:
            results[name] = func(G)
        except Exception as e:
            results[name] = 0.0

    return results


def needed_invariants(conjecture) -> set:
    """Retourne les invariants nécessaires pour une conjecture donnée."""
    return {conjecture.x_name, conjecture.y_name}


if __name__ == "__main__":
    # Test sur un graphe simple
    G = nx.petersen_graph()
    inv = compute_invariants(G)
    print("Invariants du graphe de Petersen:")
    for k, v in sorted(inv.items()):
        print(f"  {k:45s} = {v:.4f}" if isinstance(v, float) else f"  {k:45s} = {v}")
