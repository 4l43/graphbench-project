"""
main.py
Point d'entrée principal — lance la recherche sur tout le benchmark.

Usage:
    python main.py                          # Lance tout avec les paramètres par défaut
    python main.py --time 60               # 60 secondes par conjecture
    python main.py --funsearch             # Active FunSearch (Partie 2)
    python main.py --conjecture 886        # Teste une conjecture spécifique
    python main.py --validate              # Valide les contre-exemples connus
"""

import sys
import os
import time
import json
import argparse
import random
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conjecture import load_benchmark
from heuristic import search_counterexample, validate_counterexample, heuristic_score
from graph_utils import satisfies_class

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmark", "benchmark.xlsx")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")


# ══════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════

def print_header():
    print("=" * 65)
    print("  GRAPHBENCH — Réfutation automatique de conjectures")
    print("=" * 65)

def print_result(result, conjecture):
    status = "✅ RÉFUTÉE" if result.found else "❌ Non réfutée"
    print(f"  [{status}] Conjecture {result.conjecture_id} — {conjecture.y_name} {conjecture.sign} f({conjecture.x_name})")
    print(f"    Classes: {conjecture.graph_classes}")
    print(f"    Temps: {result.time_elapsed:.2f}s")
    if result.found:
        print(f"    Violation: {result.violation:.6f}")
        print(f"    Graphe: n={result.graph.number_of_nodes()}, m={result.graph.number_of_edges()}")
        print(f"    graph6: {result.graph6}")
        def _fmt(v):
            try: return f"{float(v):.4f}"
            except: return str(v)
        x_val = result.invariants.get(conjecture.x_name, 0)
        y_val = result.invariants.get(conjecture.y_name, 0)
        try: borne = f"{conjecture.bound(float(x_val)):.4f}"
        except: borne = '?'
        print(f"    {conjecture.x_name}={_fmt(x_val)}, {conjecture.y_name}={_fmt(y_val)}, borne={borne}")
    else:
        print(f"    Meilleure violation approchée: {result.violation:.6f}")


# ══════════════════════════════════════════════════════════════
#  VALIDATION DES CONTRE-EXEMPLES CONNUS
# ══════════════════════════════════════════════════════════════

def validate_known(conjectures):
    """Vérifie que les contre-exemples connus sont bien détectés."""
    print("\n[Validation des contre-exemples connus du benchmark]")
    ok, fail, no_ce = 0, 0, 0
    for c in conjectures:
        if c.known_counterexample:
            try:
                G = nx.from_graph6_bytes(c.known_counterexample.encode())
                from invariants import compute_invariants, needed_invariants
                needed = needed_invariants(c)
                inv = compute_invariants(G, needed)
                violation = c.violation(inv)
                if violation > 1e-9:
                    ok += 1
                    print(f"  ✅ Conjecture {c.id}: violation={violation:.4f}")
                else:
                    fail += 1
                    print(f"  ❌ Conjecture {c.id}: violation={violation:.4f} (attendu > 0)")
            except Exception as e:
                fail += 1
                print(f"  ❌ Conjecture {c.id}: erreur {e}")
        else:
            no_ce += 1
    print(f"\nRésumé: {ok} validés, {fail} échoués, {no_ce} sans contre-exemple connu")


# ══════════════════════════════════════════════════════════════
#  BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════

def run_benchmark(conjectures, time_limit=60, use_funsearch=False,
                  funsearch_budget=300, verbose=False):
    """Lance la recherche sur toutes les conjectures."""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    results_file = os.path.join(RESULTS_PATH, "results.json")

    score_fn = heuristic_score  # Par défaut

    # ── FunSearch (Partie 2) ─────────────────────────────────
    if use_funsearch:
        print("\n[FunSearch] Phase d'évolution de la fonction de score...")
        from funsearch import FunSearch
        fs = FunSearch(conjectures, pool_size=6, eval_sample_size=5)
        score_fn, best_body = fs.evolve(
            n_iterations=5,
            time_budget=funsearch_budget
        )
        fs.save_pool(os.path.join(RESULTS_PATH, "funsearch_pool.json"))
        print(f"\n[FunSearch] Meilleure fonction:\n{best_body}\n")

    # ── Recherche principale ─────────────────────────────────
    print(f"\n[Benchmark] Lancement sur {len(conjectures)} conjectures "
          f"({time_limit}s par conjecture)\n")

    all_results = []
    n_found = 0
    total_cost = 0.0
    start_total = time.time()

    for i, c in enumerate(conjectures):
        print(f"[{i+1}/{len(conjectures)}] Conjecture {c.id}: "
              f"{c.y_name} {c.sign} f({c.x_name}) | classes={c.graph_classes}")

        result = search_counterexample(
            c,
            time_limit=time_limit,
            score_fn=score_fn,
            verbose=verbose
        )

        print_result(result, c)

        # Coût selon la règle du projet
        if result.found:
            cost = result.time_elapsed
            n_found += 1
        else:
            cost = 120.0
        total_cost += cost

        # Sauvegarder le résultat
        r = {
            "conjecture_id": c.id,
            "conjecture_text": c.text,
            "found": result.found,
            "violation": result.violation,
            "time_elapsed": result.time_elapsed,
            "cost": cost,
            "graph6": result.graph6,
            "invariants": {k: float(v) if isinstance(v, (int, float)) else str(v)
                          for k, v in result.invariants.items()},
        }
        all_results.append(r)

        # Sauvegarde intermédiaire
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  Coût: {cost:.2f}s | Coût total: {total_cost:.2f}s | Réfutées: {n_found}/{i+1}\n")

    # ── Résumé final ─────────────────────────────────────────
    elapsed_total = time.time() - start_total
    print("=" * 65)
    print(f"  RÉSULTATS FINAUX")
    print("=" * 65)
    print(f"  Conjectures réfutées : {n_found} / {len(conjectures)}")
    print(f"  Score total          : {total_cost:.2f}")
    print(f"  Temps total réel     : {elapsed_total:.1f}s")
    print(f"  Résultats sauvegardés: {results_file}")
    print("=" * 65)

    return all_results, n_found, total_cost


# ══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GraphBench — Réfutation de conjectures")
    parser.add_argument("--time", type=float, default=60,
                        help="Limite de temps par conjecture (secondes)")
    parser.add_argument("--funsearch", action="store_true",
                        help="Activer FunSearch (Partie 2)")
    parser.add_argument("--funsearch-budget", type=float, default=300,
                        help="Budget temps pour FunSearch (secondes)")
    parser.add_argument("--conjecture", type=int, default=None,
                        help="ID d'une conjecture spécifique à tester")
    parser.add_argument("--validate", action="store_true",
                        help="Valider les contre-exemples connus")
    parser.add_argument("--verbose", action="store_true",
                        help="Affichage détaillé")
    parser.add_argument("--seed", type=int, default=42,
                        help="Graine aléatoire")
    args = parser.parse_args()

    random.seed(args.seed)
    print_header()

    # Charger le benchmark
    print(f"\nChargement du benchmark: {BENCHMARK_PATH}")
    conjectures = load_benchmark(BENCHMARK_PATH)
    print(f"✅ {len(conjectures)} conjectures chargées\n")

    # Validation seule
    if args.validate:
        validate_known(conjectures)
        return

    # Conjecture spécifique
    if args.conjecture is not None:
        targets = [c for c in conjectures if c.id == args.conjecture]
        if not targets:
            print(f"❌ Conjecture {args.conjecture} non trouvée")
            return
        conjectures = targets

    # Lancement
    run_benchmark(
        conjectures,
        time_limit=args.time,
        use_funsearch=args.funsearch,
        funsearch_budget=args.funsearch_budget,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()