"""
funsearch.py
Partie 2 : Architecture FunSearch — évolution automatique de la fonction de score via LLM.

Principe:
1. Un pool de fonctions de score est maintenu, trié par performance
2. Le LLM génère de nouvelles variantes à partir des meilleures
3. Les nouvelles fonctions sont testées sur un sous-ensemble de conjectures
4. Les meilleures remplacent les moins bonnes dans le pool
"""

import time
import json
import random
import traceback
import requests
from typing import Callable

# ══════════════════════════════════════════════════════════════
#  TEMPLATE ET EXEMPLES DE FONCTIONS
# ══════════════════════════════════════════════════════════════

FUNCTION_TEMPLATE = '''def heuristic_score(G, invariants, conjecture):
    """
    G : graphe NetworkX
    invariants : dictionnaire des invariants calculés
    conjecture : objet avec .violation(invariants), .x_name, .y_name, .graph_classes
    Retourne un score numérique à maximiser.
    Un score > 0 signifie potentiellement un contre-exemple.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("n", 0)
    m = invariants.get("m", 0)
    delta = invariants.get("minimum_degree", 0)
    Delta = invariants.get("maximum_degree", 0)
    diam = invariants.get("diameter", 0)
    rad = invariants.get("radius", 0)
    gamma = invariants.get("domination_number", 0)
    alpha = invariants.get("independence_number", 0)
    tau = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)
    mu = invariants.get("matching_number", 0)
    avg = invariants.get("average_degree", 0)
    clique = invariants.get("clique_number", 0)
    density = 0.0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    {body}
'''

SEED_FUNCTIONS = [
    # Fonction 0 : basique
    "return violation",

    # Fonction 1 : exemple du sujet
    """density = 0
if n > 1:
    density = 2 * m / (n * (n - 1))
return (
    10.0 * violation
    + 0.3 * diam
    + 0.2 * Delta
    + 0.1 * triangles
    - 0.05 * n
    - 0.2 * abs(density - 0.5)
)""",

    # Fonction 2 : axée sur les dominations
    """x_val = invariants.get(conjecture.x_name, 0)
y_val = invariants.get(conjecture.y_name, 0)
bound = conjecture.bound(x_val)
gap = y_val - bound if conjecture.sign == "<=" else bound - y_val
return (
    10.0 * violation
    + 0.5 * gap
    + 0.1 * (alpha - tau)
    - 0.03 * n
    + 0.2 * (Delta - delta) if Delta > 0 else 0
)""",

    # Fonction 3 : favorise les grands graphes denses
    """return (
    10.0 * violation
    + 0.4 * diam
    + 0.3 * (Delta - delta)
    + 0.2 * mu
    - 0.02 * n
    + 0.1 * clique
)""",
]

# ══════════════════════════════════════════════════════════════
#  APPEL AU LLM (Claude API)
# ══════════════════════════════════════════════════════════════

def call_llm_for_new_function(best_functions: list, scores: list, conjecture_descriptions: list) -> str:
    """
    Appelle Claude pour générer une nouvelle fonction de score améliorée.
    Retourne le corps de la fonction (le code après les déclarations de variables).
    """
    examples_text = ""
    for i, (fn_body, score) in enumerate(zip(best_functions[:3], scores[:3])):
        examples_text += f"\n--- Fonction {i+1} (score moyen: {score:.4f}) ---\n{fn_body}\n"

    conj_text = "\n".join(f"- {d}" for d in conjecture_descriptions[:5])

    prompt = f"""Tu es un expert en théorie des graphes et optimisation.
Tu dois améliorer une fonction heuristique Python pour maximiser la violation de conjectures sur des graphes.

CONTEXTE: La fonction reçoit un graphe G, ses invariants, et une conjecture de la forme:
  y(G) <= f(x(G)) ou y(G) >= f(x(G))
La violation = y - f(x) (pour <=) ou f(x) - y (pour >=).
Un contre-exemple est trouvé quand violation > 0.

VARIABLES DISPONIBLES (déjà définies):
- violation: la violation de la conjecture (objectif principal)
- n, m: ordre et taille du graphe
- delta, Delta: degrés min et max
- diam, rad: diamètre et rayon  
- gamma, alpha, tau: domination, indépendance, couverture
- triangles, mu, avg, clique, density: autres invariants
- conjecture.x_name, conjecture.y_name: noms des invariants X et Y

CONJECTURES CIBLES (exemples):
{conj_text}

MEILLEURES FONCTIONS ACTUELLES (avec leur score):
{examples_text}

OBJECTIF: Propose UNE SEULE nouvelle fonction de score MEILLEURE.
- Elle doit guider la recherche AVANT que violation > 0
- Utilise des bonus/pénalités intelligents selon les invariants X et Y de la conjecture
- Considère la taille du graphe, la structure, etc.
- Sois créatif mais reste Python valide

Réponds UNIQUEMENT avec le corps de la fonction (les instructions Python après les variables pré-définies).
PAS de markdown, PAS de def, PAS d'explication. Juste le code Python pur du return.
Exemple de réponse valide:
return 10.0 * violation + 0.5 * diam - 0.1 * n

Ta nouvelle fonction:"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        content = data.get("content", [{}])[0].get("text", "").strip()
        # Nettoyer le code
        content = content.replace("```python", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  COMPILATION ET TEST DE FONCTIONS
# ══════════════════════════════════════════════════════════════

def compile_score_function(body: str) -> Callable | None:
    """Compile le corps d'une fonction de score et retourne la fonction."""
    code = FUNCTION_TEMPLATE.format(body=body)
    try:
        namespace = {}
        exec(code, namespace)
        return namespace["heuristic_score"]
    except Exception as e:
        print(f"  [COMPILE ERROR] {e}\n  Code: {body[:100]}")
        return None


def evaluate_function(score_fn, conjectures, time_per_conj=5) -> float:
    """
    Évalue une fonction de score sur un sous-ensemble de conjectures.
    Retourne le score moyen (meilleures violations obtenues).
    """
    from heuristic import search_counterexample

    total_score = 0.0
    n_tested = 0

    for c in conjectures:
        try:
            result = search_counterexample(c, time_limit=time_per_conj, score_fn=score_fn)
            if result.found:
                total_score += 100.0 + max(0, result.violation)
            else:
                # Score partiel basé sur la meilleure violation approchée
                total_score += max(0.0, result.violation if result.violation else 0.0)
            n_tested += 1
        except Exception as e:
            pass

    return total_score / max(1, n_tested)


# ══════════════════════════════════════════════════════════════
#  BOUCLE FUNSEARCH PRINCIPALE
# ══════════════════════════════════════════════════════════════

class FunSearch:
    def __init__(self, conjectures, pool_size=6, eval_sample_size=5):
        self.conjectures = conjectures
        self.pool_size = pool_size
        self.eval_sample_size = eval_sample_size

        # Pool: liste de (score, body_str, fn)
        self.pool = []
        self._init_pool()

    def _init_pool(self):
        """Initialise le pool avec les fonctions seed."""
        print("  [FunSearch] Initialisation du pool...")
        eval_conjectures = random.sample(
            self.conjectures,
            min(self.eval_sample_size, len(self.conjectures))
        )
        for body in SEED_FUNCTIONS:
            fn = compile_score_function(body)
            if fn is None:
                continue
            score = evaluate_function(fn, eval_conjectures, time_per_conj=3)
            self.pool.append({"score": score, "body": body, "fn": fn})
            print(f"    Seed score: {score:.4f}")

        self.pool.sort(key=lambda x: x["score"], reverse=True)
        print(f"  [FunSearch] Pool initialisé: {len(self.pool)} fonctions")

    def evolve(self, n_iterations=5, time_budget=300):
        """
        Boucle d'évolution FunSearch.
        Génère de nouvelles fonctions via LLM et met à jour le pool.
        """
        start = time.time()
        print(f"\n[FunSearch] Début de l'évolution ({n_iterations} itérations)...")

        eval_conjectures = random.sample(
            self.conjectures,
            min(self.eval_sample_size, len(self.conjectures))
        )
        conjecture_descriptions = [c.text[:100] for c in eval_conjectures]

        for iteration in range(n_iterations):
            if time.time() - start > time_budget:
                print("  Budget temps épuisé.")
                break

            print(f"\n  [Iter {iteration+1}/{n_iterations}] Meilleur score actuel: {self.pool[0]['score']:.4f}")

            # Préparer les meilleures fonctions pour le LLM
            best_bodies = [p["body"] for p in self.pool[:3]]
            best_scores = [p["score"] for p in self.pool[:3]]

            # Demander au LLM une nouvelle fonction
            print("  Appel au LLM...")
            new_body = call_llm_for_new_function(best_bodies, best_scores, conjecture_descriptions)

            if new_body is None:
                print("  LLM n'a pas répondu, on continue.")
                continue

            print(f"  Nouvelle fonction reçue ({len(new_body)} chars)")

            # Compiler et évaluer
            fn = compile_score_function(new_body)
            if fn is None:
                print("  Compilation échouée, on continue.")
                continue

            score = evaluate_function(fn, eval_conjectures, time_per_conj=5)
            print(f"  Score: {score:.4f} (vs meilleur: {self.pool[0]['score']:.4f})")

            # Mettre à jour le pool
            self.pool.append({"score": score, "body": new_body, "fn": fn})
            self.pool.sort(key=lambda x: x["score"], reverse=True)
            self.pool = self.pool[:self.pool_size]  # Garder les meilleurs

            print(f"  Pool mis à jour. Top 3 scores: {[f'{p[\"score\"]:.4f}' for p in self.pool[:3]]}")

        print(f"\n[FunSearch] Évolution terminée. Meilleure fonction (score={self.pool[0]['score']:.4f}):")
        print(self.pool[0]["body"])
        return self.pool[0]["fn"], self.pool[0]["body"]

    def best_function(self):
        """Retourne la meilleure fonction de score actuelle."""
        if self.pool:
            return self.pool[0]["fn"]
        from heuristic import heuristic_score
        return heuristic_score

    def save_pool(self, path):
        """Sauvegarde le pool (sans les fonctions compilées)."""
        data = [{"score": p["score"], "body": p["body"]} for p in self.pool]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Pool sauvegardé: {path}")

    def load_pool(self, path):
        """Charge un pool sauvegardé."""
        try:
            with open(path) as f:
                data = json.load(f)
            for item in data:
                fn = compile_score_function(item["body"])
                if fn:
                    self.pool.append({"score": item["score"], "body": item["body"], "fn": fn})
            self.pool.sort(key=lambda x: x["score"], reverse=True)
            print(f"  Pool chargé: {len(self.pool)} fonctions")
        except Exception as e:
            print(f"  [WARN] Impossible de charger le pool: {e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from conjecture import load_benchmark

    conjectures = load_benchmark("../benchmark/benchmark.xlsx")
    print(f"Chargé {len(conjectures)} conjectures")

    # Test avec un petit sous-ensemble
    fs = FunSearch(conjectures[:10], pool_size=4, eval_sample_size=3)
    best_fn, best_body = fs.evolve(n_iterations=2, time_budget=120)
    fs.save_pool("../results/funsearch_pool.json")
    print("\nMeilleure fonction de score trouvée:")
    print(best_body)
