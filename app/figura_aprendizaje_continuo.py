# -*- coding: utf-8 -*-
"""
figura_aprendizaje_continuo.py — Figura del experimento de aprendizaje continuo.

Solo LEE `Resultados/metricas_aprendizaje_continuo.csv`; no mide nada. Un panel por
tipo retirado: % de categoría correcta sobre el lote de evaluación frente a las
filas etiquetadas acumuladas, para R1 y los dos brazos de I1 (media ± desviación
típica sobre las 11 semillas). Todas las líneas parten de R0 (0 filas). La línea
discontinua es el techo que impone la etapa 1: 100 − % que pasa sin alarma, que
ninguna actualización de la etapa 2 puede superar.

Salida: Resultados/figuras/aprendizaje_continuo_curvas.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

CSV = os.path.join(config.RESULTADOS_DIR, "metricas_aprendizaje_continuo.csv")
SALIDA = os.path.join(config.FIGURAS_DIR, "aprendizaje_continuo_curvas.png")

TIPOS = [("smurf", "DoS"), ("satan", "Probe"), ("warezclient", "R2L")]
TAMANOS = [0, 10, 25, 50, 100, 200]
# Tres primeros huecos de la paleta categórica validada (todas las parejas pasan
# CVD); el aqua queda por debajo de 3:1 de contraste, así que cada serie lleva
# además marcador propio y leyenda.
METODOS = [
    # Etiquetas en palabras, sin los códigos R1/I1: la figura va al cuerpo de la
    # memoria, donde los códigos no se usan.
    ("R1", "Reentrenamiento desde cero", "#2a78d6", "o"),
    ("I1_completo", "Incremental, bosque completo", "#1baf7a", "s"),
    ("I1_tercio", "Incremental, un tercio del bosque", "#eb6834", "^"),
]
TINTA, TINTA_2, REJILLA = "#1a1a1a", "#52514e", "#e4e3df"


def main():
    d = pd.read_csv(CSV)
    ev = d[d["conjunto"] == "retirado_eval"]
    x = np.arange(len(TAMANOS))

    fig, ejes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (tipo, cat) in zip(ejes, TIPOS):
        sub = ev[ev["tipo_retirado"] == tipo]
        r0 = sub[sub["metodo"] == "R0"]["pct_correcta"]
        techo = 100.0 - sub["pct_sin_alarma"].mean()

        for metodo, etiqueta, color, marca in METODOS:
            m = sub[sub["metodo"] == metodo].groupby("n_llegada")["pct_correcta"]
            media = np.concatenate([[r0.mean()], m.mean().reindex(TAMANOS[1:]).values])
            sd = np.concatenate([[r0.std()], m.std().reindex(TAMANOS[1:]).values])
            ax.fill_between(x, media - sd, np.minimum(media + sd, 100), color=color,
                            alpha=0.12, linewidth=0)
            ax.plot(x, media, color=color, linewidth=2, marker=marca, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.2, label=etiqueta, zorder=3)

        ax.axhline(techo, color=TINTA_2, linestyle="--", linewidth=1.2, zorder=2,
                   label="Máximo que permite la etapa 1")
        # El valor va a la izquierda y por ENCIMA de la línea: ahí no hay curvas
        # (R0 queda siempre por debajo del techo).
        ax.text(x[0] - 0.25, techo + 1.2, "{:.1f} %".format(techo),
                ha="left", va="bottom", fontsize=9, color=TINTA_2)
        ax.set_title("Sin {} ({})".format(tipo, cat), fontsize=12, color=TINTA)
        ax.set_xticks(x)
        ax.set_xticklabels(["sin\nactualizar"] + [str(t) for t in TAMANOS[1:]])
        ax.set_xlabel("flujos etiquetados acumulados", color=TINTA_2)
        ax.set_ylim(0, 106)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(TINTA_2)
        ax.tick_params(colors=TINTA_2)

    ejes[0].set_ylabel("% del lote de evaluación con\nla categoría correcta", color=TINTA_2)
    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(config.FIGURAS_DIR, exist_ok=True)
    fig.savefig(SALIDA, dpi=300, bbox_inches="tight", facecolor="white")
    print("   → " + SALIDA)


if __name__ == "__main__":
    main()
