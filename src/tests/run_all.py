#!/usr/bin/env python
"""Roda as tres suites em sequencia, sem depender de pytest.

    cd src && ./venv/bin/python tests/run_all.py
"""
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
SUITES = [
    "test_normalization_fixes.py",
    "test_auditoria_fixes.py",
    "test_pipeline_integracao.py",
]

if __name__ == "__main__":
    falhas = []
    for suite in SUITES:
        print(f"\n{'=' * 70}\n>>> {suite}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(AQUI, suite)],
                           cwd=os.path.dirname(AQUI))
        if r.returncode != 0:
            falhas.append(suite)
    print(f"\n{'=' * 70}")
    if falhas:
        print("SUITES COM FALHA:", ", ".join(falhas))
        sys.exit(1)
    print("Todas as suites passaram.")
