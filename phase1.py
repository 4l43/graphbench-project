import pandas as pd
import networkx as nx
import time
import random
import ast

# ==========================================
# 0. Utilitaire de calcul
# ==========================================
def parse_valeur_math(val):
    """Convertit une chaîne (ex: '15/8', '-1/2') en nombre décimal."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num) / float(den)
    return float(val_str)

# ==========================================
# 1. Calcul des Invariants
# ==========================================
def compute_invariants(G):
    """Calcule les propriétés du graphe G demandées par le benchmark."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n > 0:
        degrees = [d for node, d in G.degree()]
        min_deg, max_deg = min(degrees), max(degrees)
        avg_deg = sum(degrees) / n
    else:
        min_deg, max_deg, avg_deg = 0, 0, 0

    triangles = sum(nx.triangles(G).values()) / 3
    is_connected = nx.is_connected(G) if n > 0 else False
    
    # Invariants de base demandés
    return {
        "n": n, 
        "m": m,
        "minimum_degree": min_deg,
        "maximum_degree": max_deg,
        "average_degree": avg_deg,
        "triangle_number": triangles,
        "density": nx.density(G),
        "diameter": nx.diameter(G) if is_connected else 0,
        "radius": nx.radius(G) if is_connected else 0,
        "clique_number": 0 # À étoffer avec nx.graph_clique_number(G) si besoin
    }

# ==========================================
# 2. Fonction de Score (Violation)
# ==========================================
def violation_score(invariants, conjecture):
    """Calcule la violation (Gauche - Droite). L'objectif est > 0."""
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
    # Gestion automatique minimisation/maximisation
    return (y_val - f_x) if signe == '<=' else (f_x - y_val)

# ==========================================
# 3. Mutations et Réparations
# ==========================================
def mutate(G):
    """Applique une mutation locale aléatoire."""
    H = G.copy()
    nodes = list(H.nodes())
    action = random.choice(["add_edge", "remove_edge", "add_node", "remove_node"])
    
    if action == "add_edge" and len(nodes) >= 2:
        u, v = random.sample(nodes, 2)
        if not H.has_edge(u, v): H.add_edge(u, v)
    elif action == "remove_edge" and H.number_of_edges() > 0:
        u, v = random.choice(list(H.edges()))
        H.remove_edge(u, v)
    elif action == "add_node":
        new_node = max(nodes) + 1 if nodes else 1
        H.add_node(new_node)
        if nodes: H.add_edge(new_node, random.choice(nodes))
    elif action == "remove_node" and len(nodes) > 1:
        H.remove_node(random.choice(nodes))
    return H

def is_valid_class(G, subgroups):
    """Vérifie si le graphe respecte la classe imposée."""
    if 'connected' in subgroups and not nx.is_connected(G): return False
    if 'tree' in subgroups and not nx.is_tree(G): return False
    return True

# ==========================================
# 4. Boucle Principale de Recherche
# ==========================================
def solve_conjecture(conjecture):
    """Tente de réfuter une conjecture en moins de 60 secondes."""
    c_id = conjecture['Conjecture ID']
    print(f"\n🚀 Conjecture {c_id}...")
    
    subgroups = str(conjecture['Subgroup'])
    
    # Génération d'un graphe initial valide
    best_graph = nx.erdos_renyi_graph(8, 0.5)
    while not is_valid_class(best_graph, subgroups):
        best_graph = nx.erdos_renyi_graph(8, 0.5)
        
    best_score = violation_score(compute_invariants(best_graph), conjecture)
    start_time = time.time()
    iterations = 0
    
    # Boucle avec chronomètre
    while (time.time() - start_time) < 60:
        iterations += 1
        
        H = mutate(best_graph)
        if not is_valid_class(H, subgroups): continue
            
        invariants = compute_invariants(H)
        score = violation_score(invariants, conjecture)
        
        # Hill Climbing : on garde si c'est meilleur ou égal
        if score >= best_score:
            best_score, best_graph = score, H
            
            # Condition de victoire !
            if score > 0:
                t = time.time() - start_time
                g6 = nx.to_graph6_bytes(H, header=False).decode('ascii').strip()
                print(f"✅ Trouvé en {t:.2f}s (au bout de {iterations} itérations)")
                return {"ID": c_id, "Status": "SUCCESS", "Time": t, "Score": score, "Iterations": iterations, "Graph6": g6}
                
    # Échec après 60 secondes
    print(f"❌ Échec (Score max: {best_score:.4f} en {iterations} itérations)")
    return {"ID": c_id, "Status": "FAILURE", "Time": 120, "Score": best_score, "Iterations": iterations, "Graph6": ""}

# ==========================================
# 5. Point d'entrée & Sauvegarde CSV
# ==========================================
if __name__ == "__main__":
    try:
        print("Chargement du benchmark...")
        # Assure-toi que le fichier est bien dans le même dossier
        df = pd.read_excel("benchmark.xlsx") 
        
        resultats_finaux = []
        print(f"Début du traitement de {len(df)} conjectures...")
        
        # On parcourt TOUT le fichier Excel
        for _, row in df.iterrows():
            res = solve_conjecture(row)
            resultats_finaux.append(res)
            
        # Sauvegarde dans un fichier CSV à la fin
        df_res = pd.DataFrame(resultats_finaux)
        df_res.to_csv("resultats_recherche.csv", index=False)
        
        score_total = df_res['Time'].sum()
        print(f"\n🏁 Terminé ! Résultats enregistrés dans 'resultats_recherche.csv'")
        print(f"🏆 Score Total (Coût cumulé) : {score_total:.2f}")

    except FileNotFoundError:
        print("Erreur : Le fichier 'benchmark.xlsx' est introuvable. Place-le dans le même dossier que ce script.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")