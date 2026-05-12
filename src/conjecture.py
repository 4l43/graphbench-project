"""
conjecture.py
Charge et représente les conjectures depuis le benchmark.xlsx
"""

import pandas as pd
from fractions import Fraction
import ast
import re


class Conjecture:
    def __init__(self, row):
        self.id = int(row["Conjecture ID"])
        self.text = row["Conjecture"]
        self.graph_classes = self._parse_classes(row["Subgroup"])
        self.x_name = row["X"]
        self.y_name = row["Y"]
        self.sign = row["Sign"]  # '<=' ou '>='
        self.coefficients = self._parse_coefficients(row["Coefficients"])
        self.intercept = self._parse_fraction(str(row["Intercept"]))
        self.degree = int(row["Degree"])
        self.known_counterexample = row.get("Counter example (g6)", None)
        if pd.isna(self.known_counterexample):
            self.known_counterexample = None

    def _parse_classes(self, val):
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return ["connected"]

    def _parse_fraction(self, s):
        s = str(s).strip()
        if not s or s == "nan" or s == "0":
            return 0.0
        try:
            return float(Fraction(s))
        except Exception:
            try:
                return float(s)
            except Exception:
                return 0.0

    def _parse_coefficients(self, val):
        try:
            lst = ast.literal_eval(str(val))
            return [self._parse_fraction(c) for c in lst]
        except Exception:
            return [0.0]

    def bound(self, x_val):
        """Calcule la borne f(x)."""
        result = self.intercept
        for i, c in enumerate(self.coefficients):
            result += c * (x_val ** (i + 1))
        return result

    def violation(self, invariants: dict) -> float:
        """
        Retourne violation > 0 si la conjecture est réfutée.
        Pour <=: violation = y - f(x)
        Pour >=: violation = f(x) - y
        """
        x_val = invariants.get(self.x_name, 0)
        y_val = invariants.get(self.y_name, 0)
        b = self.bound(x_val)
        if self.sign == "<=":
            return y_val - b
        else:
            return b - y_val

    def is_violated(self, invariants: dict) -> bool:
        return self.violation(invariants) > 1e-9

    def __repr__(self):
        return f"Conjecture(id={self.id}, classes={self.graph_classes}, {self.y_name} {self.sign} f({self.x_name}))"


def load_benchmark(path: str) -> list:
    df = pd.read_excel(path)
    conjectures = []
    for _, row in df.iterrows():
        try:
            c = Conjecture(row)
            conjectures.append(c)
        except Exception as e:
            print(f"  [WARN] Impossible de charger la ligne {row.get('Conjecture ID', '?')}: {e}")
    return conjectures


if __name__ == "__main__":
    conjectures = load_benchmark("../benchmark.xlsx")
    print(f"✅ {len(conjectures)} conjectures chargées")
    for c in conjectures[:3]:
        print(f"  {c}")
        print(f"    Classes: {c.graph_classes}")
        print(f"    Coeffs: {c.coefficients}, Intercept: {c.intercept}")
