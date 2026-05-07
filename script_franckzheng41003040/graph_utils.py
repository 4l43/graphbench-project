"""
graph_utils.py
Générateurs de graphes initiaux et fonctions de réparation par classe.
"""

import networkx as nx
import random


# ══════════════════════════════════════════════════════════════
#  VÉRIFICATEURS DE CLASSE
# ══════════════════════════════════════════════════════════════

def is_connected(G):
    return nx.is_connected(G)

def is_tree(G):
    return nx.is_tree(G)

def is_bipartite(G):
    return nx.is_bipartite(G)

def is_planar(G):
    return nx.check_planarity(G)[0]

def has_claw(G):
    """Vérifie si G contient un K_{1,3} induit (une griffe)."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) >= 3:
            # Chercher 3 voisins mutuellement non-adjacents
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        a, b, c = neighbors[i], neighbors[j], neighbors[k]
                        if not G.has_edge(a, b) and not G.has_edge(a, c) and not G.has_edge(b, c):
                            return True
    return False

def is_claw_free(G):
    return not has_claw(G)

def satisfies_class(G, graph_classes: list) -> bool:
    """Vérifie que G satisfait toutes les contraintes de classe."""
    for cls in graph_classes:
        cls = cls.strip().lower()
        if cls == "connected" and not is_connected(G):
            return False
        if cls == "tree" and not is_tree(G):
            return False
        if cls == "bipartite" and not is_bipartite(G):
            return False
        if cls == "planar" and not is_planar(G):
            return False
        if cls == "claw_free" and not is_claw_free(G):
            return False
    return True


# ══════════════════════════════════════════════════════════════
#  GÉNÉRATEURS DE GRAPHES INITIAUX
# ══════════════════════════════════════════════════════════════

def gen_connected(n=None):
    """Graphe connexe aléatoire."""
    if n is None:
        n = random.randint(5, 20)
    G = nx.gnm_random_graph(n, max(n - 1, random.randint(n - 1, min(n * (n - 1) // 2, n * 2))))
    G = make_connected(G)
    return G

def gen_tree(n=None):
    """Arbre aléatoire."""
    if n is None:
        n = random.randint(5, 25)
    return nx.random_labeled_tree(n)

def gen_claw_free_connected(n=None):
    """Graphe connexe sans griffe (line graph d'un graphe connexe)."""
    if n is None:
        n = random.randint(5, 20)
    # Les line graphs sont toujours sans griffe
    # On génère un graphe source et on prend son line graph
    source_n = random.randint(4, max(4, n // 2 + 2))
    source_m = random.randint(n, n + source_n)
    H = nx.gnm_random_graph(source_n, min(source_m, source_n * (source_n - 1) // 2))
    H = make_connected(H)
    L = nx.line_graph(H)
    if L.number_of_nodes() == 0:
        return gen_claw_free_connected(n)
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = make_connected(L)
    return L

def gen_bipartite_connected(n=None):
    """Graphe biparti connexe aléatoire."""
    if n is None:
        n = random.randint(6, 20)
    n1 = n // 2
    n2 = n - n1
    p = random.uniform(0.3, 0.7)
    G = nx.bipartite.random_graph(n1, n2, p)
    G = nx.convert_node_labels_to_integers(G)
    G = make_connected(G)
    return G

def generate_initial_population(conjecture, size=10):
    """Génère une population initiale adaptée à la classe de la conjecture."""
    classes = [c.lower() for c in conjecture.graph_classes]
    population = []

    for _ in range(size):
        if "tree" in classes:
            G = gen_tree()
        elif "claw_free" in classes:
            G = gen_claw_free_connected()
        elif "bipartite" in classes:
            G = gen_bipartite_connected()
        else:
            G = gen_connected()
        population.append(G)

    return population


# ══════════════════════════════════════════════════════════════
#  RÉPARATION
# ══════════════════════════════════════════════════════════════

def make_connected(G):
    """Rend G connexe en ajoutant des arêtes entre composantes."""
    G = G.copy()
    components = list(nx.connected_components(G))
    while len(components) > 1:
        c1 = list(components[0])
        c2 = list(components[1])
        u = random.choice(c1)
        v = random.choice(c2)
        G.add_edge(u, v)
        components = list(nx.connected_components(G))
    return G

def repair_connected(G):
    """Assure la connexité."""
    if not nx.is_connected(G):
        return make_connected(G)
    return G

def repair_tree(G):
    """Répare G pour en faire un arbre."""
    G = G.copy()
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return nx.path_graph(5)

    # Construire un arbre couvrant
    T = nx.minimum_spanning_tree(make_connected(G))
    return T

def repair_claw_free(G):
    """Élimine les griffes en ajoutant des arêtes entre les voisins non-adjacents."""
    G = G.copy()
    max_iter = 50
    for _ in range(max_iter):
        claw_found = False
        for v in list(G.nodes()):
            neighbors = list(G.neighbors(v))
            if len(neighbors) >= 3:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            a, b, c = neighbors[i], neighbors[j], neighbors[k]
                            if not G.has_edge(a, b) and not G.has_edge(a, c) and not G.has_edge(b, c):
                                # Ajouter une arête pour éliminer la griffe
                                pair = random.choice([(a, b), (a, c), (b, c)])
                                G.add_edge(*pair)
                                claw_found = True
                                break
                        if claw_found:
                            break
                    if claw_found:
                        break
            if claw_found:
                break
        if not claw_found:
            break
    return G

def repair(G, graph_classes: list):
    """Répare G selon les contraintes de classe, dans l'ordre."""
    classes = [c.lower() for c in graph_classes]

    # Toujours assurer un minimum de 2 sommets
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
#  MUTATIONS
# ══════════════════════════════════════════════════════════════

def mutate_add_edge(G):
    """Ajoute une arête aléatoire."""
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
    """Supprime une arête aléatoire."""
    G = G.copy()
    edges = list(G.edges())
    if not edges:
        return G
    G.remove_edge(*random.choice(edges))
    return G

def mutate_add_vertex(G):
    """Ajoute un sommet avec 1-3 arêtes aléatoires."""
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
    """Supprime un sommet aléatoire (pas le dernier)."""
    G = G.copy()
    if G.number_of_nodes() <= 3:
        return G
    v = random.choice(list(G.nodes()))
    G.remove_node(v)
    G = nx.convert_node_labels_to_integers(G)
    return G

def mutate_add_leaf(G):
    """Ajoute une feuille à un sommet existant."""
    G = G.copy()
    if G.number_of_nodes() == 0:
        return G
    new_node = max(G.nodes()) + 1
    v = random.choice(list(G.nodes()))
    G.add_edge(new_node, v)
    return G

def mutate_subdivide_edge(G):
    """Subdivise une arête (utile pour les arbres)."""
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
    """Déplace une extrémité d'une arête."""
    G = G.copy()
    edges = list(G.edges())
    nodes = list(G.nodes())
    if not edges or len(nodes) < 3:
        return G
    u, v = random.choice(edges)
    G.remove_edge(u, v)
    for _ in range(10):
        w = random.choice(nodes)
        if w != u and not G.has_edge(u, w):
            G.add_edge(u, w)
            return G
    G.add_edge(u, v)  # Annuler
    return G

def mutate_add_clique(G, size=3):
    """Ajoute une clique de taille donnée reliée au graphe."""
    G = G.copy()
    start = max(G.nodes(), default=-1) + 1
    new_nodes = list(range(start, start + size))
    for i in range(len(new_nodes)):
        for j in range(i + 1, len(new_nodes)):
            G.add_edge(new_nodes[i], new_nodes[j])
    # Relier la clique au graphe existant
    if G.number_of_nodes() > size:
        existing = [n for n in G.nodes() if n not in new_nodes]
        G.add_edge(random.choice(new_nodes), random.choice(existing))
    return G

def mutate_add_path(G, length=3):
    """Ajoute un chemin relié au graphe."""
    G = G.copy()
    start = max(G.nodes(), default=-1) + 1
    path_nodes = list(range(start, start + length))
    for i in range(len(path_nodes) - 1):
        G.add_edge(path_nodes[i], path_nodes[i + 1])
    if G.number_of_nodes() > length:
        existing = [n for n in G.nodes() if n not in path_nodes]
        G.add_edge(random.choice(path_nodes), random.choice(existing))
    return G

# Mutations spécialisées pour les arbres
def mutate_tree_add_leaf(G):
    return mutate_add_leaf(G)

def mutate_tree_remove_leaf(G):
    """Supprime une feuille de l'arbre."""
    G = G.copy()
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves or G.number_of_nodes() <= 3:
        return G
    G.remove_node(random.choice(leaves))
    G = nx.convert_node_labels_to_integers(G)
    return G

def mutate_tree_subdivide(G):
    return mutate_subdivide_edge(G)

# Mutations spécialisées pour les graphes sans griffe
def mutate_claw_free_add_edge(G):
    """Ajoute une arête, puis répare les griffes."""
    G = mutate_add_edge(G)
    return repair_claw_free(G)

def mutate_claw_free_add_clique(G):
    """Ajoute une clique (les cliques sont sans griffe)."""
    return mutate_add_clique(G, size=random.randint(2, 4))


# Registre de mutations par classe
MUTATIONS = {
    "connected": [
        mutate_add_edge,
        mutate_remove_edge,
        mutate_add_vertex,
        mutate_remove_vertex,
        mutate_rewire_edge,
        mutate_add_clique,
        mutate_add_path,
    ],
    "tree": [
        mutate_tree_add_leaf,
        mutate_tree_remove_leaf,
        mutate_tree_subdivide,
        mutate_add_path,
    ],
    "claw_free": [
        mutate_claw_free_add_edge,
        mutate_claw_free_add_clique,
        mutate_add_vertex,
        mutate_remove_vertex,
    ],
}

def get_mutations(graph_classes: list):
    """Retourne la liste des mutations adaptées aux classes."""
    classes = [c.lower() for c in graph_classes]
    if "tree" in classes:
        return MUTATIONS["tree"]
    elif "claw_free" in classes:
        return MUTATIONS["claw_free"] + MUTATIONS["connected"]
    else:
        return MUTATIONS["connected"]


if __name__ == "__main__":
    print("Test générateurs:")
    G = gen_connected(10)
    print(f"  Connexe: n={G.number_of_nodes()}, m={G.number_of_edges()}, connexe={is_connected(G)}")
    T = gen_tree(10)
    print(f"  Arbre: n={T.number_of_nodes()}, m={T.number_of_edges()}, arbre={is_tree(T)}")
    CF = gen_claw_free_connected(10)
    print(f"  Sans griffe: n={CF.number_of_nodes()}, connexe={is_connected(CF)}, claw_free={is_claw_free(CF)}")
