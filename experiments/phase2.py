import pandas as pd
import networkx as nx
import time
import random
import ast
import itertools
from google import genai
import re

# ==========================================
# 0. CONFIGURATION GOOGLE GEMINI
# ==========================================
# Remplace par ta vraie clé API
client = genai.Client(api_key="/////////////////////////") 

# ==========================================
# 1. OUTILS DE BASE (Calculs et Mutations)
# ==========================================
def parse_valeur_math(val):
    if pd.isna(val): return 0.0
    val_str = str(val).strip()
    if '/' in val_str:
        num, den = val_str.split('/')
        return float(num) / float(den)
    return float(val_str)

def get_clique_number(graph):
    if not graph.nodes(): return 0
    return max(len(c) for c in nx.find_cliques(graph))

def compute_invariants(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0: return {}
    degrees = [d for node, d in G.degree()]
    triangles = sum(nx.triangles(G).values()) / 3
    alpha = get_clique_number(nx.complement(G))
    return {
        "n": n, "m": m, "minimum_degree": min(degrees) if degrees else 0, 
        "maximum_degree": max(degrees) if degrees else 0,
        "triangle_number": triangles, "density": nx.density(G), 
        "independence_number": alpha, "clique_number": get_clique_number(G)
    }

def mutate(G):
    H = G.copy()
    nodes = list(H.nodes())
    action = random.choice(["add_edge", "remove_edge", "add_node", "remove_node"])
    if action == "add_edge" and len(nodes) >= 2:
        u, v = random.sample(nodes, 2)
        H.add_edge(u, v)
    elif action == "remove_edge" and H.number_of_edges() > 0:
        H.remove_edge(*random.choice(list(H.edges())))
    elif action == "add_node":
        new_node = max(nodes) + 1 if nodes else 1
        H.add_node(new_node)
        if nodes: H.add_edge(new_node, random.choice(nodes))
    elif action == "remove_node" and len(nodes) > 2:
        H.remove_node(random.choice(nodes))
    return H

def calcul_violation_pure(invariants, conjecture):
    """Calcule la violation mathématique stricte de la conjecture (Gauche - Droite)."""
    x_val = invariants.get(conjecture['X'], 0)
    y_val = invariants.get(conjecture['Y'], 0)
    try: coeffs = ast.literal_eval(str(conjecture['Coefficients']))
    except: coeffs = []
    f_x = parse_valeur_math(conjecture['Intercept'])
    for degre, coef_str in enumerate(coeffs, start=1):
        f_x += parse_valeur_math(coef_str) * (x_val ** degre)
    return (y_val - f_x) if conjecture['Sign'] == '<=' else (f_x - y_val)

# ==========================================
# 2. LE MOTEUR FUNSEARCH (Magie de l'IA)
# ==========================================
def executer_code_llm(code_python_string):
    """Prend le code texte généré par l'IA et le transforme en vraie fonction Python."""
    namespace = {}
    try:
        namespace['calcul_violation_pure'] = calcul_violation_pure
        exec(code_python_string, globals(), namespace)
        return namespace.get('heuristic_score', None)
    except Exception as e:
        print(f"  [!] L'IA a fait une erreur de syntaxe : {e}")
        return None

def evaluer_heuristique(fonction_score, df_train):
    """Teste une fonction de score sur notre petit groupe d'entraînement."""
    cout_total = 0
    
    for _, conjecture in df_train.iterrows():
        best_graph = nx.erdos_renyi_graph(8, 0.5)
        best_invariants = compute_invariants(best_graph)
        
        best_score = fonction_score(best_graph, best_invariants, conjecture)
        
        start_time = time.time()
        trouve = False
        
        # ⏱️ On limite à 15 SECONDES pour l'entraînement (pour aller vite !)
        while (time.time() - start_time) < 15: 
            H = mutate(best_graph)
            invariants = compute_invariants(H)
            
            score = fonction_score(H, invariants, conjecture)
            
            if score >= best_score:
                best_score, best_graph = score, H
                
                # VRAI contre-exemple mathématique ?
                vraie_violation = calcul_violation_pure(invariants, conjecture)
                if vraie_violation > 0:
                    cout_total += (time.time() - start_time)
                    trouve = True
                    break
                    
        if not trouve:
            cout_total += 30 # Pénalité d'échec pour ce test
            
    return cout_total

def demander_variantes_ia(prompt, n_variants=2):
    """Envoie les instructions à Gemini et récupère son code."""
    nouveaux_codes = []
    print(f"  🤖 L'IA réfléchit et propose {n_variants} nouvelles heuristiques...")
    for _ in range(n_variants):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            match = re.search(r'```python\n(.*?)\n```', response.text, re.DOTALL)
            code_propre = match.group(1) if match else response.text
            nouveaux_codes.append(code_propre)
            time.sleep(4) 
        except Exception as e:
            print(f"  [!] Erreur de connexion avec Google Gemini : {e}")
    return nouveaux_codes

# ==========================================
# 3. LA BOUCLE D'APPRENTISSAGE
# ==========================================
def lancer_funsearch(df_train, iterations=3):
    print("=== 🚀 DÉBUT DE L'ENTRAÎNEMENT FUNSEARCH ===")
    
    meilleur_code_actuel = "def heuristic_score(G, invariants, conjecture):\n    return calcul_violation_pure(invariants, conjecture)\n"
    
    fonction_depart = executer_code_llm(meilleur_code_actuel)
    meilleur_cout = evaluer_heuristique(fonction_depart, df_train)
    print(f"Coût de base (sans IA) sur cet échantillon : {meilleur_cout:.2f}s (Objectif: minimiser !)")

    for i in range(iterations):
        print(f"\n--- Itération {i+1}/{iterations} ---")
        
        prompt = (
            "Tu es un expert en recherche locale et théorie des graphes. Je cherche des contre-exemples à des conjectures.\n"
            f"Voici mon heuristique actuelle (elle obtient un score de coût de {meilleur_cout}).\n\n"
            "```python\n"
            f"{meilleur_code_actuel}\n"
            "```\n\n"
            "Améliore-la en créant une fonction 'plus informative' sous la forme : F(G) = violation(G) + bonus(G) - penalty(G).\n"
            "Utilise les clés du dictionnaire 'invariants' (ex: 'n', 'm', 'maximum_degree', 'triangle_number', 'density').\n"
            "Tu PEUX utiliser la fonction calcul_violation_pure(invariants, conjecture) pour obtenir la violation de base.\n\n"
            "Génère UNIQUEMENT le code Python complet de la fonction.\n"
            "Respecte EXACTEMENT cette signature : def heuristic_score(G, invariants, conjecture):\n"
        )
        
        codes_proposes = demander_variantes_ia(prompt, n_variants=2)
        
        for idx, code in enumerate(codes_proposes):
            fonction_test = executer_code_llm(code)
            if not fonction_test: continue
                
            cout = evaluer_heuristique(fonction_test, df_train)
            print(f"  > Test de la variante {idx+1} : Coût = {cout:.2f}s")
            
            if cout < meilleur_cout:
                print(f"  🌟 INCROYABLE ! Nouvelle meilleure heuristique trouvée !")
                meilleur_cout = cout
                meilleur_code_actuel = code
                
    print("\n=== 🏆 RÉSULTAT FINAL FUNSEARCH ===")
    print(f"Meilleur coût atteint : {meilleur_cout:.2f}s")
    print("\nVoici la fonction magique à copier dans ton rapport et dans ton script de la Partie 1 :\n")
    print(meilleur_code_actuel)

# ==========================================
# LANCEMENT
# ==========================================
if __name__ == "__main__":
    try:
        print("Chargement de ton fichier 'benchmark.xlsx'...")
        df_complet = pd.read_excel("benchmark.xlsx")
        
        # ASTUCE : On prend 5 conjectures au hasard pour entraîner l'IA rapidement !
        df_train = df_complet.sample(n=5, random_state=42)
        print(f"Sélection de 5 conjectures d'entraînement : {list(df_train['Conjecture ID'])}")
        
        # On lance 3 tours d'entraînement
        lancer_funsearch(df_train, iterations=3)
        
    except FileNotFoundError:
        print("Erreur : Le fichier 'benchmark.xlsx' n'est pas dans le dossier !")
    except Exception as e:
        print(f"Erreur inattendue au lancement : {e}")