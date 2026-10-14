# -*- coding: utf-8 -*-
"""Figura de la memoria (capítulo 2, taxonomía de ataques): muestras por categoría en NSL-KDD.

Lee el crudo (KDDTrain+.txt y KDDTest+.txt, en la carpeta dataset/) y dibuja, por categoría
(normal, DoS, Probe, R2L, U2R), tres barras: entrenamiento, test de tipos que también están en el
entrenamiento y test de tipos ausentes del entrenamiento (0-day). Escala logarítmica para que U2R se vea.

No entrena nada, no lee splits ni escribe tablas: solo la figura, en Resultados/figuras/.

    python app/figura_tipos_ataque.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from program import NSLKDDPreprocessor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(RAIZ, "dataset", "KDDTrain+.txt")
TEST = os.path.join(RAIZ, "dataset", "KDDTest+.txt")
SALIDA = os.path.join(RAIZ, "Resultados", "figuras", "nslkdd_muestras_por_categoria.png")

CATEGORIAS = ["normal", "dos", "probe", "r2l", "u2r"]
ROTULOS = ["Normal", "DoS", "Probe", "R2L", "U2R"]
AZUL, NARANJA = "#2a78d6", "#eb6834"      # paleta de referencia (validada: CVD ΔE 24, normal ΔE 30)
TINTA, TINTA_2 = "#0b0b0b", "#52514e"


def contar():
    pre = NSLKDDPreprocessor()
    tr = pd.read_csv(TRAIN, names=pre.column_names, header=None)
    te = pd.read_csv(TEST, names=pre.column_names, header=None)
    mapa = dict(pre.attack_mapping)
    mapa["normal"] = "normal"
    tr["cat"] = tr["attack"].map(mapa)
    te["cat"] = te["attack"].map(mapa)
    if tr["cat"].isna().any() or te["cat"].isna().any():
        raise SystemExit("hay tipos sin categoría en el mapeo")
    vistos = set(tr["attack"])
    te_conocido = te[te["attack"].isin(vistos)]
    te_nuevo = te[~te["attack"].isin(vistos)]
    f = lambda df: [int((df["cat"] == c).sum()) for c in CATEGORIAS]
    return f(tr), f(te_conocido), f(te_nuevo), len(tr), len(te), te_nuevo["attack"].nunique()


def dibujar(n_tr, n_te_c, n_te_n):
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 12, "legend.fontsize": 12.5})
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(CATEGORIAS))
    w = 0.27
    series = [
        (x - w, n_tr, dict(color=AZUL), "Entrenamiento (KDDTrain+)"),
        (x, n_te_c, dict(color=NARANJA), "Test: tipos presentes en el entrenamiento"),
        (x + w, n_te_n, dict(color="white", edgecolor=NARANJA, hatch="///", linewidth=1.5),
         "Test: tipos nuevos (0-day)"),
    ]
    for xs, vals, estilo, rotulo in series:
        v = np.array(vals, dtype=float)
        barras = ax.bar(xs[v > 0], v[v > 0], w * 0.92, label=rotulo, **estilo)
        for b, val in zip(barras, v[v > 0]):
            ax.text(b.get_x() + b.get_width() / 2, val * 1.12, "{:,}".format(int(val)).replace(",", "."),
                    ha="center", va="bottom", fontsize=10.5, color=TINTA_2, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(1, 10 ** 6.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ROTULOS, color=TINTA)
    ax.set_ylabel("Número de muestras (escala logarítmica)", color=TINTA)
    ax.yaxis.grid(True, color="#e3e2de", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig


def main():
    n_tr, n_te_c, n_te_n, tot_tr, tot_te, k_nuevos = contar()
    print("entrenamiento:", dict(zip(CATEGORIAS, n_tr)), "total", tot_tr)
    print("test conocidos:", dict(zip(CATEGORIAS, n_te_c)))
    print("test nuevos:", dict(zip(CATEGORIAS, n_te_n)), "tipos nuevos", k_nuevos, "total test", tot_te)
    fig = dibujar(n_tr, n_te_c, n_te_n)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    fig.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor="white")
    print("figura:", SALIDA)


if __name__ == "__main__":
    main()
