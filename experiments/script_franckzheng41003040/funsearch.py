"""
funsearch.py
Architecture FunSearch — évolution automatique de la fonction de score via LLM.
Compatible OpenAI (gpt-4o-mini).
"""

import os
import time
import json
import random
import traceback
import requests
from typing import Callable

random.seed(42)

# ============================================================
# CONFIG — mets ta clé ici OU: export OPENAI_API_KEY="sk-..."
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")

# ============================================================
# TEMPLATE DE FONCTION
# ============================================================

FUNCTION_TEMPLATE = '''def heuristic_score(G, invariants, conjecture):
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

# ============================================================
# FONCTIONS SEED
# ============================================================

SEED_FUNCTIONS = [
    "return violation",
    """return (
    10.0 * violation
    + 0.3 * diam
    + 0.2 * Delta
    + 0.1 * triangles
    - 0.05 * n
    - 0.2 * abs(density - 0.5)
)""",
    """x_val = invariants.get(conjecture.x_name, 0); y_val = invariants.get(conjecture.y_name, 0); bound = conjecture.bound(x_val); gap = (y_val - bound) if conjecture.sign == "<=" else (bound - y_val); return (10.0 * violation + 2.0 * gap + 0.1 * (alpha - tau) - 0.03 * n + (0.2 * (Delta - delta) if Delta > 0 else 0))""",
    """return (
    10.0 * violation
    + 0.4 * diam
    + 0.3 * (Delta - delta)
    + 0.2 * mu
    - 0.02 * n
    + 0.1 * clique
)""",
]

# ============================================================
# APPEL AU LLM — OpenAI chat/completions (API standard)
# ============================================================

def call_llm_for_new_function(best_functions, scores, conjecture_descriptions):

    examples_text = ""
    for i, (fn_body, score) in enumerate(zip(best_functions[:3], scores[:3])):
        examples_text += f"\n--- Fonction {i+1} (score={score:.4f}) ---\n{fn_body}\n"

    conj_text = "\n".join(f"- {d}" for d in conjecture_descriptions[:5])

    prompt = f"""Tu es un expert en théorie des graphes.
Tu dois améliorer une fonction heuristique Python pour maximiser la violation de conjectures.

VARIABLES DISPONIBLES:
- violation, n, m, delta, Delta, diam, rad
- gamma, alpha, tau, triangles, mu, avg, clique, density

CONJECTURES CIBLES:
{conj_text}

MEILLEURES FONCTIONS ACTUELLES:
{examples_text}

Génère UNE SEULE nouvelle fonction Python.
Réponds UNIQUEMENT avec le code Python du return.
PAS de markdown, PAS de def. Juste le return.

Exemple:
return 10.0 * violation + 0.5 * diam - 0.1 * n"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code != 200:
            print(f"  [LLM ERROR] Status {response.status_code}: {response.text[:200]}")
            return None

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        content = content.replace("```python", "").replace("```", "").strip()
        return content

    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None

# ============================================================
# COMPILATION
# ============================================================

SAFE_BUILTINS = {
    "abs": abs, "max": max, "min": min, "sum": sum,
    "float": float, "int": int, "pow": pow, "round": round,
}

def compile_score_function(body):
    code = FUNCTION_TEMPLATE.format(body=body)
    try:
        namespace = {}
        exec(code, {"__builtins__": SAFE_BUILTINS}, namespace)
        return namespace["heuristic_score"]
    except Exception as e:
        print(f"  [COMPILE ERROR] {e}")
        return None

# ============================================================
# EVALUATION
# ============================================================

def evaluate_function(score_fn, conjectures, time_per_conj=5):
    from heuristic import search_counterexample
    total_score = 0.0
    n_tested = 0
    for c in conjectures:
        try:
            result = search_counterexample(c, time_limit=time_per_conj, score_fn=score_fn)
            if result.found:
                total_score += 100.0 + max(0.0, result.violation or 0.0)
            else:
                total_score += max(0.0, result.violation or 0.0)
            n_tested += 1
        except Exception:
            pass
    return total_score / max(1, n_tested)

# ============================================================
# CLASSE FUNSEARCH
# ============================================================

class FunSearch:

    def __init__(self, conjectures, pool_size=6, eval_sample_size=5):
        self.conjectures = conjectures
        self.pool_size = pool_size
        self.eval_sample_size = eval_sample_size
        self.pool = []
        self._init_pool()

    def _init_pool(self):
        print("\n[FunSearch] Initialisation du pool...")
        eval_conjectures = random.sample(
            self.conjectures, min(self.eval_sample_size, len(self.conjectures))
        )
        for i, body in enumerate(SEED_FUNCTIONS):
            fn = compile_score_function(body)
            if fn is None:
                continue
            score = evaluate_function(fn, eval_conjectures, time_per_conj=3)
            self.pool.append({"score": score, "body": body, "fn": fn})
            print(f"  Seed {i}: score={score:.4f}")
        self.pool.sort(key=lambda x: x["score"], reverse=True)
        print(f"[FunSearch] Pool initialisé: {len(self.pool)} fonctions")

    def evolve(self, n_iterations=5, time_budget=300):
        start = time.time()
        eval_conjectures = random.sample(
            self.conjectures, min(self.eval_sample_size, len(self.conjectures))
        )
        conjecture_descriptions = [c.text[:100] for c in eval_conjectures]

        print(f"\n[FunSearch] Évolution ({n_iterations} itérations, budget={time_budget}s)")

        for iteration in range(n_iterations):
            if time.time() - start > time_budget:
                print("  Budget temps épuisé.")
                break

            best_score = self.pool[0]["score"]
            print(f"\n  [Iter {iteration+1}/{n_iterations}] Meilleur score: {best_score:.4f}")
            print("  Appel au LLM...")

            new_body = call_llm_for_new_function(
                [p["body"] for p in self.pool[:3]],
                [p["score"] for p in self.pool[:3]],
                conjecture_descriptions
            )

            if new_body is None:
                continue

            print(f"  Reçu: {new_body[:80]}...")

            fn = compile_score_function(new_body)
            if fn is None:
                continue

            score = evaluate_function(fn, eval_conjectures, time_per_conj=5)
            print(f"  Score: {score:.4f} {'✅ Amélioration!' if score > best_score else ''}")

            self.pool.append({"score": score, "body": new_body, "fn": fn})
            self.pool.sort(key=lambda x: x["score"], reverse=True)
            self.pool = self.pool[:self.pool_size]

        print(f"\n[FunSearch] Terminé. Meilleure fonction (score={self.pool[0]['score']:.4f}):")
        print(self.pool[0]["body"])
        return self.pool[0]["fn"], self.pool[0]["body"]

    def best_function(self):
        if self.pool:
            return self.pool[0]["fn"]
        from heuristic import heuristic_score
        return heuristic_score

    def save_pool(self, path):
        data = [{"score": p["score"], "body": p["body"]} for p in self.pool]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Pool sauvegardé: {path}")

    def load_pool(self, path):
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
            print(f"  [WARN] {e}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from conjecture import load_benchmark
    conjectures = load_benchmark("../benchmark/benchmark.xlsx")
    print(f"Chargé {len(conjectures)} conjectures")
    fs = FunSearch(conjectures[:10], pool_size=4, eval_sample_size=3)
    best_fn, best_body = fs.evolve(n_iterations=3, time_budget=120)
    fs.save_pool("../results/funsearch_pool.json")
    print("\n=== MEILLEURE FONCTION ===")
    print(best_body)