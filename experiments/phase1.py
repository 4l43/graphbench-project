import pandas as pd
import networkx as nx
import time
import random
import ast
import itertools

# ==========================================
# 0. Utilitaires mathématiques et de Classes
# ==========================================
def parse_valeur_math(val):
    """Convertit de manière sécurisée une chaîne (ex: '15/8', '-1/2') en float."""
    if pd.isna(val): return 0.0
    val_str = str(val).strip()
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num) / float(den)
    return float(val_str)

def is_claw_free(G):
    """Vérifie si le graphe est sans griffe (claw-free)."""
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) >= 3:
            # S'il y a 3 voisins sans aucune arête entre eux = griffe (K1,3)
            for triplet in itertools.combinations(neighbors, 3):
                if not G.has_edge(triplet[0], triplet[1]) and \
                   not G.has_edge(triplet[1], triplet[2]) and \
                   not G.has_edge(triplet[0], triplet[2]):
                    return False
    return True

def is_valid_class(G, subgroups):
    """Vérifie le respect strict de la classe imposée."""
    if 'connected' in subgroups and not nx.is_connected(G): return False
    if 'tree' in subgroups and not nx.is_tree(G): return False
    if 'planar' in subgroups and not nx.check_planarity(G)[0]: return False
    if 'bipartite' in subgroups and not nx.is_bipartite(G): return False
    if 'claw_free' in subgroups and not is_claw_free(G): return False
    return True

def generate_random_tree(n):
    """Génère un arbre aléatoire robuste (compatible toutes versions NetworkX)."""
    G = nx.Graph()
    if n == 0: return G
    G.add_node(0)
    for i in range(1, n):
        parent = random.choice(list(G.nodes()))
        G.add_edge(i, parent)
    return G

def generate_smart_initial_graph(subgroups, n_nodes=8):
    """Génère un graphe initial adapté à la classe demandée."""
    if 'tree' in subgroups:
        return generate_random_tree(n_nodes)
    else:
        G = nx.erdos_renyi_graph(n_nodes, 0.5)
        # On force jusqu'à ce qu'il soit valide
        while not is_valid_class(G, subgroups):
            G = nx.erdos_renyi_graph(n_nodes, 0.5)
        return G

# ==========================================
# 1. Calcul complet des Invariants
# ==========================================
def get_clique_number(graph):
    """Calcule la taille de la clique max de manière robuste."""
    if not graph.nodes():
        return 0
    return max(len(c) for c in nx.find_cliques(graph))

def compute_invariants(G):
    """Calcule le catalogue d'invariants demandé par le TD."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0: return {}

    degrees = [d for node, d in G.degree()]
    min_deg, max_deg = min(degrees), max(degrees)
    avg_deg = sum(degrees) / n

    is_conn = nx.is_connected(G)
    
    # Invariants complexes basiques
    triangles = sum(nx.triangles(G).values()) / 3
    clique_num = get_clique_number(G)
    
    # Alpha (indépendance) via le graphe complémentaire
    alpha = get_clique_number(nx.complement(G))
    # Tau (couverture) via le théorème de Gallai
    tau = n - alpha 

    invariants = {
        "n": n, "m": m,
        "minimum_degree": min_deg,
        "maximum_degree": max_deg,
        "average_degree": avg_deg,
        "triangle_number": triangles,
        "density": nx.density(G),
        "diameter": nx.diameter(G) if is_conn else 0,
        "radius": nx.radius(G) if is_conn else 0,
        "clique_number": clique_num,
        "independence_number": alpha,
        "vertex_cover_number": tau,
        "matching_number": len(nx.max_weight_matching(G, maxcardinality=True)),
        "node_connectivity": nx.node_connectivity(G) if is_conn else 0,
        "edge_connectivity": nx.edge_connectivity(G) if is_conn else 0
    }
    
    # Approximation pour domination_number
    try:
        from networkx.algorithms.approximation import min_weighted_dominating_set
        invariants["domination_number"] = len(min_weighted_dominating_set(G))
    except:
        invariants["domination_number"] = 0

    return invariants

# ==========================================
# 2. Fonction de Score (Maximiser la Violation)
# ==========================================
def violation_score(invariants, conjecture):
    """Calcule la violation : Gauche - Droite."""
    x_val = invariants.get(conjecture['X'], 0)
    y_val = invariants.get(conjecture['Y'], 0)
    
    try:
        coefficients = ast.literal_eval(str(conjecture['Coefficients']))
    except:
        coefficients = []
        
    f_x = parse_valeur_math(conjecture['Intercept'])
    for degre, coef_str in enumerate(coefficients, start=1):
        f_x += parse_valeur_math(coef_str) * (x_val ** degre)
        
    signe = conjecture['Sign']
    return (y_val - f_x) if signe == '<=' else (f_x - y_val)

# ==========================================
# 3. Mutations Optimisées par Classe
# ==========================================
def mutate(G, subgroups):
    """Mutations intelligentes pour éviter de créer des graphes invalides."""
    H = G.copy()
    nodes = list(H.nodes())
    
    # STRATÉGIE SPÉCIALE ARBRES
    if 'tree' in subgroups:
        action = random.choice(["add_leaf", "remove_leaf"])
        if action == "add_leaf":
            new_node = max(nodes) + 1 if nodes else 1
            parent = random.choice(nodes) if nodes else None
            H.add_node(new_node)
            if parent: H.add_edge(new_node, parent)
        elif action == "remove_leaf" and len(nodes) > 2:
            leaves = [n for n, d in H.degree() if d == 1]
            if leaves: H.remove_node(random.choice(leaves))
        return H

    # STRATÉGIE GÉNÉRALE
    action = random.choice(["add_edge", "remove_edge", "add_node", "remove_node"])
    
    if action == "add_edge" and len(nodes) >= 2:
        u, v = random.sample(nodes, 2)
        if not H.has_edge(u, v): H.add_edge(u, v)
            
    elif action == "remove_edge" and H.number_of_edges() > 0:
        edges = list(H.edges())
        u, v = random.choice(edges)
        H.remove_edge(u, v)
        # Ne pas casser la connexité si elle est exigée
        if 'connected' in subgroups and not nx.is_connected(H):
            H.add_edge(u, v)
            
    elif action == "add_node":
        new_node = max(nodes) + 1 if nodes else 1
        H.add_node(new_node)
        if nodes: H.add_edge(new_node, random.choice(nodes))
            
    elif action == "remove_node" and len(nodes) > 2:
        H.remove_node(random.choice(nodes))
        
    return H

# ==========================================
# 4. Boucle Principale de Recherche
# ==========================================
def solve_conjecture(conjecture):
    c_id = conjecture['Conjecture ID']
    print(f"\n🚀 Conjecture {c_id}...")
    
    subgroups = str(conjecture['Subgroup'])
    
    # Génération ciblée
    best_graph = generate_smart_initial_graph(subgroups)
    best_invariants = compute_invariants(best_graph)
    best_score = violation_score(best_invariants, conjecture)
    
    start_time = time.time()
    iterations = 0
    
    while (time.time() - start_time) < 60:
        iterations += 1
        
        # Mutation
        H = mutate(best_graph, subgroups)
        
        # Filtrage strict final
        if not is_valid_class(H, subgroups): continue
            
        invariants = compute_invariants(H)
        score = violation_score(invariants, conjecture)
        
        # Hill Climbing : on accepte un score supérieur ou égal
        if score >= best_score:
            best_score, best_graph = score, H
            
            # CONDITION DE VICTOIRE
            if score > 0:
                t = time.time() - start_time
                g6 = nx.to_graph6_bytes(H, header=False).decode('ascii').strip()
                print(f"✅ Réfutée en {t:.2f}s (Itérations: {iterations}) | Score: {score:.4f}")
                return {"ID": c_id, "Status": "SUCCESS", "Time": t, "Score": score, "Iterations": iterations, "Graph6": g6}
                
    print(f"❌ Échec (Score max atteint: {best_score:.4f} en {iterations} itérations)")
    return {"ID": c_id, "Status": "FAILURE", "Time": 120, "Score": best_score, "Iterations": iterations, "Graph6": ""}

# ==========================================
# 5. Point d'entrée et Exécution
# ==========================================
if __name__ == "__main__":
    try:
        print("Chargement du benchmark...")
        df = pd.read_excel("benchmark.xlsx") 
        resultats_finaux = []
        
        print(f"Traitement des {len(df)} conjectures en cours... Laissez tourner le programme.")
        
        for _, row in df.iterrows():
            res = solve_conjecture(row)
            resultats_finaux.append(res)
            
        df_res = pd.DataFrame(resultats_finaux)
        df_res.to_csv("resultats_recherche.csv", index=False)
        
        print(f"\n🏁 Terminé ! Tous vos résultats sont sauvegardés dans 'resultats_recherche.csv'.")
        print(f"🏆 Coût Total (Objectif : minimiser le temps total) : {df_res['Time'].sum():.2f}")

    except FileNotFoundError:
        print("Erreur : Le fichier 'benchmark.xlsx' est introuvable. Placez-le dans le même dossier que ce script.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue au lancement : {e}")