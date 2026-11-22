# -*- coding: utf-8 -*-
"""
metricas.py - metricas de classificacao compartilhadas pelas etapas 11 (classica) e 12 (CNN).

Classe positiva = melanoma (alvo_melanoma = 1). Em triagem, o erro mais grave e o
FALSO NEGATIVO (melanoma tratado como benigno); por isso o limiar de decisao e
escolhido para atingir uma SENSIBILIDADE-ALVO, e nao o 0,5 padrao.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def matriz_confusao(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    vp = int(((y == 1) & (pred == 1)).sum())
    vn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return vp, vn, fp, fn


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def calcular(y, prob, limiar: float) -> Dict[str, float]:
    """Sensibilidade, especificidade, precisao, F1, acuracia balanceada e AUC."""
    from sklearn.metrics import roc_auc_score
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    pred = (prob >= limiar).astype(int)
    vp, vn, fp, fn = matriz_confusao(y, pred)
    sens, esp, prec = _div(vp, vp + fn), _div(vn, vn + fp), _div(vp, vp + fp)
    f1 = _div(2 * prec * sens, prec + sens) if np.isfinite(prec) and np.isfinite(sens) else float("nan")
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    return {
        "n": int(len(y)), "positivos": int(y.sum()), "limiar": float(limiar),
        "sensibilidade": sens, "especificidade": esp, "precisao": prec, "f1": f1,
        "acuracia_balanceada": (sens + esp) / 2 if np.isfinite(sens) and np.isfinite(esp) else float("nan"),
        "auc": auc, "vp": vp, "vn": vn, "fp": fp, "fn": fn,
    }


def escolher_limiar(y, prob, sens_alvo: float) -> float:
    """Maior limiar cuja sensibilidade e >= sens_alvo (logo, a maior especificidade possivel)."""
    from sklearn.metrics import roc_curve
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, prob)
    idx = np.where(tpr >= sens_alvo)[0]
    if len(idx) == 0:
        return float(np.min(prob))
    t = float(thr[idx[0]])
    return min(t, 1.0) if np.isfinite(t) else 1.0


def arredondar(d: Dict, casas: int = 4) -> Dict:
    return {k: (round(v, casas) if isinstance(v, float) and np.isfinite(v) else v) for k, v in d.items()}


def salvar_roc(curvas: Dict[str, Tuple[np.ndarray, np.ndarray]], arq: Path, titulo: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nome, (y, prob) in curvas.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc_score(y, prob):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("1 - especificidade (taxa de falsos positivos)")
    ax.set_ylabel("Sensibilidade")
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(arq, dpi=200)
    plt.close(fig)


def salvar_matriz(m: Dict, arq: Path, titulo: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    mat = np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["benigno", "melanoma"])
    ax.set_yticks([0, 1], ["benigno", "melanoma"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(arq, dpi=200)
    plt.close(fig)
