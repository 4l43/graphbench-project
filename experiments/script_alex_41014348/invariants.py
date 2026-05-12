"""
invariants.py  (version patchée)
─────────────────────────────────────────────────────────────────────────────
Seuls total_domination_number et independent_domination_number ont été modifiés.
Tout le reste est identique à l'original.

Problèmes corrigés
──────────────────
1. total_domination_number
   L'ancien greedy ajoutait v si v n'était pas encore dominé, ce qui ne garantit
   PAS que chaque membre du domset ait lui-même un voisin dans le domset.
   → Remplacé par branch-and-bound exact (n ≤ 22) + multi-restart greedy (n > 22).

2. independent_domination_number
   L'ancien greedy produisait UN seul MIS trié par degré croissant.
   Le MIS minimum peut être très différent selon l'ordre de visite.
   → Multi-restart (30 ordres aléatoires) + exact B&B pour n ≤ 20.
"""

import random
import networkx as nx
import numpy as np


# ─── Invariants de base (inchangés) ──────────────────────────────────

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
    return sum(nx.triangles(G).values()) // 3

def clique_number(G):
    return max((len(c) for c in nx.find_cliques(G)), default=1)


# ─── Domination simple (inchangée) ───────────────────────────────────

def domination_number(G):
    nodes = set(G.nodes())
    dominated = set()
    domset = set()
    for v in sorted(nodes, key=lambda x: G.degree(x), reverse=True):
        if v not in dominated:
            domset.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(domset)


# ─── TOTAL DOMINATION — version corrigée ─────────────────────────────

def _total_dom_exact(G) -> int:
    """
    Branch-and-bound exact pour n ≤ 22.

    Un ensemble S est totalement dominant si tout sommet de G
    (y compris ceux de S) a au moins un voisin dans S.
    Autrement dit : N(S) ⊇ V(G), où N(S) = ⋃_{v∈S} N(v).

    Représentation bitmask → très rapide pour n ≤ 22.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    # Précalculer le masque de voisinage de chaque sommet
    nb_mask = [0] * n
    for v in nodes:
        for u in G.neighbors(v):
            nb_mask[idx[v]] |= (1 << idx[u])

    all_covered = (1 << n) - 1
    best = [n]  # borne supérieure

    def bt(i: int, size: int, covered: int):
        # Élagage 1 : déjà pire que le meilleur connu
        if size >= best[0]:
            return
        # Succès
        if covered == all_covered:
            best[0] = size
            return
        # Fin des nœuds
        if i == n:
            return
        # Élagage 2 : même en ajoutant tous les nœuds restants, peut-on tout couvrir ?
        reachable = covered
        for j in range(i, n):
            reachable |= nb_mask[j]
        if reachable != all_covered:
            return

        # Branche « on prend le nœud i »
        bt(i + 1, size + 1, covered | nb_mask[i])
        # Branche « on ne prend pas le nœud i »
        bt(i + 1, size, covered)

    bt(0, 0, 0)
    return best[0]


def _total_dom_multistart(G, n_starts: int = 25) -> int:
    """
    Greedy multi-restart pour n > 22.
    À chaque restart : ordre aléatoire, on ajoute v si v n'est pas encore couvert.
    Garantit un vrai ensemble totalement dominant (chaque membre a un voisin dedans).
    """
    nodes = list(G.nodes())
    neighbors = {v: list(G.neighbors(v)) for v in nodes}
    node_set = set(nodes)
    best = len(nodes)

    for trial in range(n_starts):
        order = list(nodes)
        if trial == 0:
            # Premier essai : tri par degré décroissant (greedy classique)
            order.sort(key=lambda v: len(neighbors[v]), reverse=True)
        else:
            random.shuffle(order)

        covered = set()   # N(S)
        domset = []

        for v in order:
            if v not in covered:
                domset.append(v)
                covered.update(neighbors[v])
                if covered >= node_set:
                    break

        # Vérification : tout sommet de domset doit avoir un voisin dans domset
        # (condition de domination TOTALE, pas juste domination simple)
        domset_set = set(domset)
        valid = True
        for v in domset:
            if not any(u in domset_set for u in neighbors[v]):
                valid = False
                break

        if not valid:
            # Forcer la validité : ajouter un voisin pour chaque membre non-couvert
            for v in list(domset):
                if not any(u in domset_set for u in neighbors[v]):
                    if neighbors[v]:
                        extra = max(neighbors[v], key=lambda u: len(neighbors[u]))
                        domset.append(extra)
                        domset_set.add(extra)
                        covered.update(neighbors[extra])

        best = min(best, len(domset))

    return best


def total_domination_number(G) -> int:
    """
    Nombre de domination totale γt(G).
    Exact pour n ≤ 22, multi-restart greedy au-delà.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0
    # Sommet isolé → domination totale impossible
    for v in nodes:
        if G.degree(v) == 0:
            return n

    if n <= 22:
        return _total_dom_exact(G)
    else:
        return _total_dom_multistart(G)


# ─── INDEPENDENT DOMINATION — version corrigée ───────────────────────

def _indep_dom_exact(G) -> int:
    """
    B&B exact pour n ≤ 20.
    Un ensemble indépendant dominant S est un MIS (ensemble indépendant maximal),
    car : S indépendant + S dominant ⟺ S est maximal dans l'ordre d'inclusion.
    On cherche la taille du plus petit MIS.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    nb_mask = [0] * n
    for v in nodes:
        for u in G.neighbors(v):
            nb_mask[idx[v]] |= (1 << idx[u])

    all_nodes = (1 << n) - 1
    best = [n]

    def bt(i: int, size: int, in_set: int, excluded: int):
        """
        i        : prochain sommet à considérer
        in_set   : masque des sommets déjà dans S
        excluded : masque des sommets qu'on ne peut plus ajouter (voisins des membres)
        """
        if size >= best[0]:
            return

        # Vérifier si S est déjà dominant : tout sommet hors S a un voisin dans S
        outside = all_nodes & ~in_set
        dominated = 0
        tmp = in_set
        while tmp:
            v_bit = tmp & (-tmp)
            vi = v_bit.bit_length() - 1
            dominated |= nb_mask[vi]
            tmp &= tmp - 1
        if (dominated & outside) == outside:
            # S est déjà dominant + indépendant → c'est un MIS valide
            best[0] = size
            return

        if i == n:
            return

        v_bit = 1 << i

        # Ajouter i : possible seulement si i n'est pas exclu
        if not (excluded & v_bit):
            bt(i + 1, size + 1, in_set | v_bit, excluded | nb_mask[i])

        # Ne pas ajouter i
        bt(i + 1, size, in_set, excluded)

    bt(0, 0, 0, 0)
    return best[0]


def _indep_dom_multistart(G, n_starts: int = 30) -> int:
    """
    Multi-restart greedy pour n > 20 (ou comme borne sup avant l'exact).
    Chaque restart explore un ordre différent de construction du MIS.
    """
    nodes = list(G.nodes())
    neighbors = {v: set(G.neighbors(v)) for v in nodes}
    best = len(nodes)

    for trial in range(n_starts):
        order = list(nodes)
        if trial == 0:
            order.sort(key=lambda v: len(neighbors[v]))      # degré croissant
        elif trial == 1:
            order.sort(key=lambda v: len(neighbors[v]), reverse=True)  # décroissant
        else:
            random.shuffle(order)

        ind_set = set()
        excluded = set()
        for v in order:
            if v not in excluded:
                ind_set.add(v)
                excluded |= neighbors[v]
                excluded.add(v)

        # Vérification que c'est bien dominant
        all_dominated = True
        for v in nodes:
            if v not in ind_set:
                if not neighbors[v] & ind_set:
                    all_dominated = False
                    break

        if all_dominated:
            best = min(best, len(ind_set))

    return best


def independent_domination_number(G) -> int:
    """
    Nombre de domination indépendante i(G) = taille du plus petit MIS.
    Exact pour n ≤ 20, multi-restart greedy au-delà.
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0

    # Borne supérieure rapide par multi-restart
    ub = _indep_dom_multistart(G, n_starts=15 if n <= 20 else 30)

    if n <= 20:
        # Affiner avec l'exact (initialisé avec la borne sup pour pruning fort)
        exact = _indep_dom_exact(G)
        return min(ub, exact)

    return ub


# ─── Indépendance / Couverture (inchangés) ───────────────────────────

def independence_number(G, exact=False):
    if G.number_of_nodes() == 0:
        return 0
    n = G.number_of_nodes()
    if exact or n <= 20:
        Gc = nx.complement(G)
        cliques = list(nx.find_cliques(Gc))
        return len(max(cliques, key=len)) if cliques else 1
    else:
        nodes = sorted(G.nodes(), key=lambda v: G.degree(v))
        ind_set, excluded = set(), set()
        for v in nodes:
            if v not in excluded:
                ind_set.add(v)
                excluded.update(G.neighbors(v))
        return len(ind_set)

def vertex_cover_number(G):
    return G.number_of_nodes() - independence_number(G)

def matching_number(G):
    return len(nx.max_weight_matching(G, maxcardinality=True))


# ─── Spectraux (inchangés) ────────────────────────────────────────────

def largest_eigenvalue(G):
    if G.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(G)
    return float(np.max(np.linalg.eigvalsh(A)))

def second_smallest_laplace_eigenvalue(G):
    if G.number_of_nodes() < 2:
        return 0.0
    try:
        return float(nx.algebraic_connectivity(G))
    except Exception:
        return 0.0

def largest_distance_eigenvalue(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    D = nx.floyd_warshall_numpy(G)
    return float(np.max(np.linalg.eigvalsh(D)))


# ─── Indices topologiques (inchangés) ────────────────────────────────

def randic_index(G):
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du > 0 and dv > 0:
            total += 1.0 / (du * dv) ** 0.5
    return total

def harmonic_index(G):
    total = 0.0
    for u, v in G.edges():
        s = G.degree(u) + G.degree(v)
        if s > 0:
            total += 2.0 / s
    return total

def first_zagreb_index(G):
    return sum(d ** 2 for _, d in G.degree())

def second_zagreb_index(G):
    return sum(G.degree(u) * G.degree(v) for u, v in G.edges())


# ─── Distance (inchangés) ────────────────────────────────────────────

def _distance_matrix(G):
    return dict(nx.all_pairs_shortest_path_length(G))

def proximity(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    n_ = G.number_of_nodes()
    dist = _distance_matrix(G)
    min_prox = float('inf')
    for v in G.nodes():
        s = sum(1.0 / dist[v][u] for u in G.nodes() if u != v)
        min_prox = min(min_prox, s / (n_ - 1))
    return min_prox

def remoteness(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0.0
    n_ = G.number_of_nodes()
    dist = _distance_matrix(G)
    return max(
        sum(dist[v][u] for u in G.nodes() if u != v) / (n_ - 1)
        for v in G.nodes()
    )


# ─── Registre & compute (inchangés sauf les deux fonctions patchées) ─

INVARIANT_FUNCTIONS = {
    "order": n, "n": n,
    "size": m,  "m": m,
    "diameter": diameter,
    "radius": radius,
    "minimum_degree": minimum_degree,
    "maximum_degree": maximum_degree,
    "average_degree": average_degree,
    "density": density,
    "triangle_number": triangle_number,
    "clique_number": clique_number,
    "domination_number": domination_number,
    "total_domination_number": total_domination_number,        # ← patché
    "independence_number": independence_number,
    "vertex_cover_number": vertex_cover_number,
    "independent_domination_number": independent_domination_number,  # ← patché
    "matching_number": matching_number,
    "largest_eigenvalue": largest_eigenvalue,
    "second_smallest_laplace_eigenvalue": second_smallest_laplace_eigenvalue,
    "largest_distance_eigenvalue": largest_distance_eigenvalue,
    "randic_index": randic_index,
    "harmonic_index": harmonic_index,
    "first_zagreb_index": first_zagreb_index,
    "second_zagreb_index": second_zagreb_index,
    "proximity": proximity,
    "remoteness": remoteness,
}

SLOW_INVARIANTS = {
    "clique_number",
    "largest_distance_eigenvalue",
    "proximity",
    "remoteness",
    "second_smallest_laplace_eigenvalue",
}


def compute_invariants(G: nx.Graph, needed: set = None, fast_only: bool = False) -> dict:
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
        except Exception:
            results[name] = 0.0
    return results


def needed_invariants(conjecture) -> set:
    return {conjecture.x_name, conjecture.y_name}