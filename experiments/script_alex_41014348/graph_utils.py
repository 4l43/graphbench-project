"""
graph_utils.py  (version patchée)
─────────────────────────────────────────────────────────────────────────────
Changements par rapport à l'original :

1. Deux nouveaux générateurs spécialisés :
   - gen_domination_hard(n)  : graphes où γt / i(G) est grand
     (structures sparse avec pendant paths et cycles disjoints)
   - gen_remoteness_hard(n)  : graphes path-like où remoteness est grand

2. generate_initial_population étendu :
   - Détecte automatiquement total_domination et independent_domination
     pour injecter les bons générateurs
   - Détecte remoteness/proximity

Tout le reste (vérificateurs, mutations, repair) est identique à l'original.
"""

import networkx as nx
import random


# ══════════════════════════════════════════════════════════════
#  VÉRIFICATEURS DE CLASSE (inchangés)
# ══════════════════════════════════════════════════════════════

def is_connected(G):
    return nx.is_connected(G)

def is_tree(G):
    return nx.is_tree(G)

def is_bipartite(G):
    return nx.is_bipartite(G)

def is_planar(G):
    return nx.check_planarity(G)[0]

def _find_claw(G):
    for v in G.nodes():
        nbrs = list(G.neighbors(v))
        if len(nbrs) < 3:
            continue
        sample = sorted(nbrs)
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                if G.has_edge(sample[i], sample[j]):
                    continue
                for k in range(j + 1, len(sample)):
                    a, b, c = sample[i], sample[j], sample[k]
                    if (not G.has_edge(a, b) and not G.has_edge(a, c)
                            and not G.has_edge(b, c)):
                        return (v, a, b, c)
    return None

def _rebuild_as_line_graph(G):
    n_target = G.number_of_nodes()
    source_n = max(4, n_target // 3)
    source_m = max(source_n, n_target)
    H = nx.gnm_random_graph(
        source_n, min(source_m, source_n * (source_n - 1) // 2))
    H = make_connected(H)
    L = nx.line_graph(H)
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = make_connected(L)
    return L

def has_claw(G):
    return _find_claw(G) is not None

def is_claw_free(G):
    return not has_claw(G)

def satisfies_class(G, graph_classes: list) -> bool:
    for cls in graph_classes:
        cls = cls.strip().lower()
        if cls == "connected"  and not is_connected(G):  return False
        if cls == "tree"       and not is_tree(G):        return False
        if cls == "bipartite"  and not is_bipartite(G):   return False
        if cls == "planar"     and not is_planar(G):       return False
        if cls == "claw_free"  and not is_claw_free(G):   return False
    return True


# ══════════════════════════════════════════════════════════════
#  GÉNÉRATEURS DE BASE (inchangés)
# ══════════════════════════════════════════════════════════════

def gen_connected(n=None):
    if n is None:
        n = random.randint(5, 20)
    G = nx.gnm_random_graph(
        n, max(n - 1, random.randint(n - 1, min(n * (n - 1) // 2, n * 2))))
    return make_connected(G)

def gen_tree(n=None):
    if n is None:
        n = random.randint(5, 25)
    return nx.random_labeled_tree(n)

def gen_claw_free_connected(n=None):
    if n is None:
        n = random.randint(5, 20)
    source_n = random.randint(4, max(4, n // 2 + 2))
    source_m = random.randint(n, n + source_n)
    H = nx.gnm_random_graph(
        source_n, min(source_m, source_n * (source_n - 1) // 2))
    H = make_connected(H)
    L = nx.line_graph(H)
    if L.number_of_nodes() == 0:
        return gen_claw_free_connected(n)
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = make_connected(L)
    return L

def gen_bipartite_connected(n=None):
    if n is None:
        n = random.randint(6, 20)
    n1, n2 = n // 2, n - n // 2
    p = random.uniform(0.3, 0.7)
    G = nx.bipartite.random_graph(n1, n2, p)
    G = nx.convert_node_labels_to_integers(G)
    return make_connected(G)


# ══════════════════════════════════════════════════════════════
#  NOUVEAUX GÉNÉRATEURS POUR LES CAS DIFFICILES
# ══════════════════════════════════════════════════════════════

def gen_domination_hard(n=None):
    """
    Graphes où γt (total domination) et i(G) (independent domination) sont grands.

    Stratégie : structures sparse avec plusieurs composantes fortement séparées
    reliées par des ponts uniques — chaque composante est dure à dominer
    indépendamment des autres.

    On utilise trois archétypes en rotation :
    a) Chemin de cliques K3 reliées par des ponts (pendant paths)
    b) Graphe cubique aléatoire (régulier de degré 3)
    c) Cycle avec des chemins pendants uniformément répartis
    """
    if n is None:
        n = random.randint(12, 22)

    choice = random.randint(0, 2)

    if choice == 0:
        # ── a) Chemin de K3 reliées par des ponts ────────────────────
        # Structure : K3 — pont — K3 — pont — ...
        # Chaque K3 a γt = 2, un pont force le domset à couvrir ses extrémités
        k3_count = max(2, n // 4)
        G = nx.Graph()
        offset = 0
        prev_end = None
        for _ in range(k3_count):
            # K3
            a, b, c = offset, offset + 1, offset + 2
            G.add_edges_from([(a, b), (b, c), (a, c)])
            if prev_end is not None:
                G.add_edge(prev_end, a)
            prev_end = c
            offset += 3
        # Remplir jusqu'à n avec des feuilles
        while G.number_of_nodes() < n:
            anchor = random.choice(list(G.nodes()))
            new_node = G.number_of_nodes()
            G.add_edge(new_node, anchor)
        return G

    elif choice == 1:
        # ── b) Graphe cubique aléatoire ───────────────────────────────
        # Régulier de degré 3 → γt ≥ n/3 (borne connue)
        n_reg = n if n % 2 == 0 else n + 1
        n_reg = n_reg if n_reg % 2 == 0 else n_reg + 1
        try:
            G = nx.random_regular_graph(3, n_reg)
            G = nx.convert_node_labels_to_integers(G)
            if not nx.is_connected(G):
                G = make_connected(G)
            return G
        except Exception:
            return gen_connected(n)

    else:
        # ── c) Cycle + chemins pendants ───────────────────────────────
        # Cycle de longueur n//2, avec des chemins de longueur 2 pendants
        cycle_len = max(4, n // 2)
        G = nx.cycle_graph(cycle_len)
        node_id = cycle_len
        cycle_nodes = list(range(cycle_len))
        while node_id < n:
            anchor = random.choice(cycle_nodes)
            G.add_edge(anchor, node_id)
            node_id += 1
            if node_id < n:
                G.add_edge(node_id - 1, node_id)
                node_id += 1
        return G


def gen_remoteness_hard(n=None):
    """
    Graphes où remoteness = max_v avg_distance(v) est grand.

    La remoteness est maximisée par des graphes path-like avec peu de raccourcis.
    On génère :
    a) Chemin pur (remoteness maximale pour graphes connexes)
    b) Lollipop (clique + chemin)
    c) Chemin de cycles (cycles reliés en séquence)
    """
    if n is None:
        n = random.randint(8, 20)

    choice = random.randint(0, 2)

    if choice == 0:
        # Chemin pur avec quelques arêtes aléatoires (< 10% de densité)
        G = nx.path_graph(n)
        extra = max(1, n // 10)
        nodes = list(G.nodes())
        for _ in range(extra):
            u, v = random.sample(nodes, 2)
            G.add_edge(u, v)
        return G

    elif choice == 1:
        # Lollipop : clique K_k + chemin de longueur n - k
        k = max(3, n // 4)
        tail = n - k
        G = nx.complete_graph(k)
        prev = 0
        for i in range(tail):
            new_node = k + i
            G.add_node(new_node)
            G.add_edge(prev, new_node)
            prev = new_node
        return G

    else:
        # Chaîne de petits cycles
        cycle_size = random.randint(3, 5)
        G = nx.Graph()
        offset = 0
        prev_end = None
        while G.number_of_nodes() < n:
            size = min(cycle_size, n - G.number_of_nodes())
            if size < 2:
                if prev_end is not None:
                    G.add_edge(prev_end, G.number_of_nodes())
                break
            cycle_nodes = list(range(offset, offset + size))
            for i in range(len(cycle_nodes)):
                G.add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)])
            if prev_end is not None:
                G.add_edge(prev_end, cycle_nodes[0])
            prev_end = cycle_nodes[-1]
            offset += size
        if not nx.is_connected(G):
            G = make_connected(G)
        return G


# ══════════════════════════════════════════════════════════════
#  POPULATION INITIALE — VERSION PATCHÉE
# ══════════════════════════════════════════════════════════════

def generate_initial_population(conjecture, size=10, n_range=None):
    """
    Génère une population initiale adaptée à la classe et aux invariants.

    Nouveautés par rapport à l'original :
    - Détecte total_domination_number → injecte gen_domination_hard
    - Détecte independent_domination_number → idem
    - Détecte remoteness / proximity → injecte gen_remoteness_hard
    """
    classes = [c.lower() for c in conjecture.graph_classes]
    x, y   = conjecture.x_name, conjecture.y_name
    needed = {x, y}

    deg_invariants      = {"maximum_degree", "average_degree", "minimum_degree"}
    clique_invariants   = {"clique_number"}
    spectral_invariants = {"second_smallest_laplace_eigenvalue",
                           "largest_eigenvalue", "largest_distance_eigenvalue"}
    # ── Nouveaux invariants cibles ────────────────────────────────────
    domination_hard     = {"total_domination_number",
                           "independent_domination_number"}
    remoteness_hard     = {"remoteness", "proximity"}

    population = []

    for i in range(size):
        n = None
        if n_range:
            n = random.randint(n_range[0], n_range[1])

        try:
            if "tree" in classes:
                G = gen_tree(n)

            elif "claw_free" in classes:
                if needed & spectral_invariants:
                    G = (gen_double_clique_bridge() if i % 3 == 0
                         else gen_multi_clique_path() if i % 3 == 1
                         else gen_claw_free_connected(n))
                elif needed & deg_invariants:
                    G = (gen_claw_free_star_extended(n) if i % 3 == 0
                         else gen_claw_free_path_based(n) if i % 3 == 1
                         else gen_claw_free_connected(n))
                elif needed & domination_hard:
                    # claw_free + domination → line graph de graphes sparse
                    G = (gen_domination_hard(n) if i % 2 == 0
                         else gen_claw_free_connected(n))
                    # repair garantira le claw_free
                else:
                    G = gen_claw_free_connected(n)

            elif "bipartite" in classes:
                G = gen_bipartite_connected(n)

            # ── Nouveaux cas ──────────────────────────────────────────
            elif needed & domination_hard:
                # Pour total_dom / indep_dom : alterner domination_hard
                # et graphes connexes classiques
                G = (gen_domination_hard(n) if i % 2 == 0
                     else gen_connected(n))

            elif needed & remoteness_hard:
                G = (gen_remoteness_hard(n) if i % 2 == 0
                     else gen_connected(n))

            # ── Cas originaux ─────────────────────────────────────────
            elif needed & spectral_invariants:
                G = (gen_double_clique_bridge() if i % 2 == 0
                     else gen_connected(n))
            elif needed & clique_invariants:
                G = (gen_clique_plus_leaves() if i % 2 == 0
                     else gen_connected(n))
            else:
                G = gen_connected(n)

            population.append(G)

        except Exception:
            population.append(gen_connected(n))

    return population


# ══════════════════════════════════════════════════════════════
#  RÉPARATION (inchangée)
# ══════════════════════════════════════════════════════════════

def make_connected(G):
    G = G.copy()
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1, c2 = list(components[0]), list(components[1])
        G.add_edge(random.choice(c1), random.choice(c2))
        components = list(nx.connected_components(G))
    return G

def repair_connected(G):
    return make_connected(G) if not nx.is_connected(G) else G

def repair_tree(G):
    G = G.copy()
    if len(list(G.nodes())) == 0:
        return nx.path_graph(5)
    return nx.minimum_spanning_tree(make_connected(G))

def repair_claw_free(G):
    G = G.copy()
    if G.number_of_nodes() > 30:
        return _rebuild_as_line_graph(G)
    for _ in range(20):
        claw = _find_claw(G)
        if claw is None:
            break
        _, a, b, c = claw
        G.add_edge(*random.choice([(a, b), (a, c), (b, c)]))
    return G

def repair(G, graph_classes: list):
    classes = [c.lower() for c in graph_classes]
    if G.number_of_nodes() < 2:
        G = nx.path_graph(3)
    if "tree" in classes:
        G = repair_tree(G)
    else:
        if "connected" in classes:
            G = repair_connected(G)
        if "claw_free" in classes:
            G = repair_claw_free(G)
            G = repair_connected(G)
    return G


# ══════════════════════════════════════════════════════════════
#  MUTATIONS (inchangées)
# ══════════════════════════════════════════════════════════════

def mutate_add_edge(G):
    G = G.copy()
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    for _ in range(20):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G

def mutate_remove_edge(G):
    G = G.copy()
    edges = list(G.edges())
    if not edges:
        return G
    G.remove_edge(*random.choice(edges))
    return G

def mutate_add_vertex(G):
    G = G.copy()
    new_node = max(G.nodes(), default=-1) + 1
    G.add_node(new_node)
    nodes = list(G.nodes())
    nodes.remove(new_node)
    if nodes:
        k = random.randint(1, min(3, len(nodes)))
        for v in random.sample(nodes, k):
            G.add_edge(new_node, v)
    return G

def mutate_remove_vertex(G):
    G = G.copy()
    if G.number_of_nodes() <= 3:
        return G
    v = random.choice(list(G.nodes()))
    G.remove_node(v)
    return nx.convert_node_labels_to_integers(G)

def mutate_add_leaf(G):
    G = G.copy()
    if G.number_of_nodes() == 0:
        return G
    new_node = max(G.nodes()) + 1
    G.add_edge(new_node, random.choice(list(G.nodes())))
    return G

def mutate_subdivide_edge(G):
    G = G.copy()
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    G.remove_edge(u, v)
    new_node = max(G.nodes()) + 1
    G.add_edge(u, new_node)
    G.add_edge(new_node, v)
    return G

def mutate_rewire_edge(G):
    G = G.copy()
    edges, nodes = list(G.edges()), list(G.nodes())
    if not edges or len(nodes) < 3:
        return G
    u, v = random.choice(edges)
    G.remove_edge(u, v)
    for _ in range(10):
        w = random.choice(nodes)
        if w != u and not G.has_edge(u, w):
            G.add_edge(u, w)
            return G
    G.add_edge(u, v)
    return G

def mutate_add_clique(G, size=3):
    G = G.copy()
    start = max(G.nodes(), default=-1) + 1
    new_nodes = list(range(start, start + size))
    for i in range(len(new_nodes)):
        for j in range(i + 1, len(new_nodes)):
            G.add_edge(new_nodes[i], new_nodes[j])
    existing = [n for n in G.nodes() if n not in new_nodes]
    if existing:
        G.add_edge(random.choice(new_nodes), random.choice(existing))
    return G

def mutate_add_path(G, length=3):
    G = G.copy()
    start = max(G.nodes(), default=-1) + 1
    path_nodes = list(range(start, start + length))
    for i in range(len(path_nodes) - 1):
        G.add_edge(path_nodes[i], path_nodes[i + 1])
    existing = [n for n in G.nodes() if n not in path_nodes]
    if existing:
        G.add_edge(random.choice(path_nodes), random.choice(existing))
    return G

def mutate_tree_add_leaf(G):
    return mutate_add_leaf(G)

def mutate_tree_remove_leaf(G):
    G = G.copy()
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves or G.number_of_nodes() <= 3:
        return G
    G.remove_node(random.choice(leaves))
    return nx.convert_node_labels_to_integers(G)

def mutate_tree_subdivide(G):
    return mutate_subdivide_edge(G)

def mutate_claw_free_add_edge(G):
    G = mutate_add_edge(G)
    return repair_claw_free(G)

def mutate_claw_free_add_clique(G):
    return mutate_add_clique(G, size=random.randint(2, 4))

def mutate_contract_edge(G):
    G = G.copy()
    edges = list(G.edges())
    if not edges or G.number_of_nodes() <= 3:
        return G
    u, v = random.choice(edges)
    for w in list(G.neighbors(v)):
        if w != u:
            G.add_edge(u, w)
    G.remove_node(v)
    return nx.convert_node_labels_to_integers(G)

def mutate_duplicate_vertex(G):
    G = G.copy()
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    new_node = max(G.nodes()) + 1
    for w in list(G.neighbors(v)):
        G.add_edge(new_node, w)
    if random.random() > 0.5:
        G.add_edge(new_node, v)
    return G

def mutate_add_triangle(G):
    G = G.copy()
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    n1 = max(G.nodes()) + 1
    n2 = n1 + 1
    G.add_edge(v, n1)
    G.add_edge(v, n2)
    G.add_edge(n1, n2)
    return G

def mutate_add_pendant_path(G):
    G = G.copy()
    if G.number_of_nodes() == 0:
        return G
    anchor = random.choice(list(G.nodes()))
    length = random.randint(2, 4)
    prev = anchor
    start = max(G.nodes()) + 1
    for i in range(length):
        new_node = start + i
        G.add_edge(prev, new_node)
        prev = new_node
    return G

def mutate_swap_edges(G):
    G = G.copy()
    edges = list(G.edges())
    if len(edges) < 2:
        return G
    for _ in range(10):
        (u, v), (x, y) = random.sample(edges, 2)
        if len({u, v, x, y}) == 4:
            if not G.has_edge(u, x) and not G.has_edge(v, y):
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x)
                G.add_edge(v, y)
                return G
    return G

def mutate_remove_bridge(G):
    G = G.copy()
    bridges = list(nx.bridges(G))
    if not bridges:
        return mutate_remove_edge(G)
    u, v = random.choice(bridges)
    G.remove_edge(u, v)
    return make_connected(G)

def mutate_add_star(G):
    G = G.copy()
    k = random.randint(2, 4)
    center = max(G.nodes(), default=-1) + 1
    leaves = list(range(center + 1, center + 1 + k))
    for leaf in leaves:
        G.add_edge(center, leaf)
    existing = [n for n in G.nodes() if n != center and n not in leaves]
    if existing:
        G.add_edge(center, random.choice(existing))
    return G

def mutate_change_degree(G):
    G = G.copy()
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    if random.random() > 0.5:
        non_nbrs = [u for u in nodes if u != v and not G.has_edge(u, v)]
        if non_nbrs:
            G.add_edge(v, random.choice(non_nbrs))
    else:
        nbrs = list(G.neighbors(v))
        if len(nbrs) > 1:
            G.remove_edge(v, random.choice(nbrs))
    return G


MUTATIONS = {
    "connected": [
        mutate_add_edge, mutate_remove_edge,
        mutate_add_vertex, mutate_remove_vertex,
        mutate_rewire_edge, mutate_add_clique, mutate_add_path,
        mutate_contract_edge, mutate_duplicate_vertex,
        mutate_add_triangle, mutate_add_pendant_path,
        mutate_swap_edges, mutate_remove_bridge,
        mutate_add_star, mutate_change_degree,
    ],
    "tree": [
        mutate_tree_add_leaf, mutate_tree_remove_leaf,
        mutate_tree_subdivide, mutate_add_path,
        mutate_add_pendant_path, mutate_contract_edge,
    ],
    "claw_free": [
        mutate_claw_free_add_edge, mutate_claw_free_add_clique,
        mutate_add_vertex, mutate_remove_vertex,
        mutate_duplicate_vertex, mutate_add_triangle,
        mutate_rewire_edge,
    ],
}

def get_mutations(graph_classes: list):
    classes = [c.lower() for c in graph_classes]
    if "tree" in classes:
        return MUTATIONS["tree"]
    elif "claw_free" in classes:
        return MUTATIONS["claw_free"] + MUTATIONS["connected"]
    return MUTATIONS["connected"]


# ══════════════════════════════════════════════════════════════
#  GÉNÉRATEURS SPÉCIALISÉS (inchangés)
# ══════════════════════════════════════════════════════════════

def gen_claw_free_star_extended(n=None):
    if n is None:
        hub_degree = random.randint(6, 20)
    else:
        hub_degree = max(4, n // 2)
    H = nx.star_graph(hub_degree)
    leaves = list(range(1, hub_degree + 1))
    for _ in range(random.randint(hub_degree // 2, hub_degree * 2)):
        if len(leaves) >= 2:
            H.add_edge(*random.sample(leaves, 2))
    L = nx.line_graph(H)
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = make_connected(L)
    return L

def gen_claw_free_path_based(n=None):
    if n is None:
        n = random.randint(6, 20)
    H = nx.cycle_graph(max(4, n // 2))
    nodes = list(H.nodes())
    for _ in range(random.randint(1, max(1, len(nodes) // 3))):
        H.add_edge(*random.sample(nodes, 2))
    L = nx.line_graph(H)
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = make_connected(L)
    return L

def gen_clique_plus_leaves(clique_size=None, n_leaves=None):
    if clique_size is None:
        clique_size = random.randint(3, 8)
    if n_leaves is None:
        n_leaves = random.randint(clique_size, clique_size * 5)
    G = nx.complete_graph(clique_size)
    for i in range(n_leaves):
        G.add_edge(0, clique_size + i)
    return G

def gen_double_clique_bridge(k=None):
    if k is None:
        k = random.randint(4, 15)
    G = nx.complete_graph(k)
    G2 = nx.relabel_nodes(nx.complete_graph(k), {i: i + k for i in range(k)})
    G = nx.compose(G, G2)
    G.add_edge(0, k)
    return G

def gen_multi_clique_path(n_cliques=None, clique_size=None):
    if n_cliques is None:
        n_cliques = random.randint(2, 5)
    if clique_size is None:
        clique_size = random.randint(3, 10)
    G = nx.Graph()
    offset, prev_node = 0, None
    for _ in range(n_cliques):
        nodes = list(range(offset, offset + clique_size))
        for a in nodes:
            for b in nodes:
                if a < b:
                    G.add_edge(a, b)
        if prev_node is not None:
            G.add_edge(prev_node, nodes[0])
        prev_node = nodes[-1]
        offset += clique_size
    return G