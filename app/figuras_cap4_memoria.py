# -*- coding: utf-8 -*-
"""Versiones de memoria (letra grande, formato de página) de las Figuras 4.1, 4.3 y 4.4.

No toca los scripts que generan las originales ni ninguna de sus salidas: escribe tres ficheros NUEVOS con
sufijo `_memoria` en Resultados/figuras/.
- Figura 4.1 (`eda_distribuciones_divisiones_memoria.png`): los mismos nueve gráficos que
  `program.py::_plot_specialized_distributions`, con los mismos datos y el mismo cálculo, redibujados a
  página completa y con títulos en español.
- Figura 4.3 (`validacion_discriminantes_d1_vs_d3_memoria.png`) y Figura 4.4
  (`validacion_drift_ks_memoria.png`): se calculan con los métodos de `validacion.NSLKDDValidator`
  (`analyze_feature_distributions` y `detect_data_drift`), sin cambiarlos; solo se sustituyen el dibujo y el
  nombre del fichero (subclase). La 4.4 pasa a dos paneles uno encima del otro.

    python app/figuras_cap4_memoria.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import program
import validacion

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(RAIZ, "Resultados", "figuras")
VERDE, ROJO, AZUL_CLARO = "lightgreen", "lightcoral", "lightblue"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})


def guardar(fig, nombre):
    ruta = os.path.join(FIGS, nombre)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura:", ruta)


# ----------------------------------------------------------------------------- Figura 4.1
def figura_41():
    pre = program.NSLKDDPreprocessor()
    train_df, test_df = pre.load_dataset()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["attack_category"] = train_df["attack"].map(pre.attack_mapping).fillna("unknown")
    test_df["attack_category"] = test_df["attack"].map(pre.attack_mapping).fillna("unknown")
    attack_data = train_df[train_df["attack"] != "normal"]
    normal_data = train_df[train_df["attack"] == "normal"]
    nombres = {"normal": "normal", "dos": "DoS", "probe": "Probe", "r2l": "R2L", "u2r": "U2R"}

    fig, ax = plt.subplots(3, 3, figsize=(7.6, 9.6))
    ax = ax.flatten()

    ax[0].bar(["normal"], [len(normal_data)], color=VERDE, alpha=0.85)
    ax[0].set_title("D1: entrenamiento de anomalías\n(solo normal)")
    ax[0].set_ylabel("Frecuencia")

    c2 = test_df["attack_category"].value_counts()
    ax[1].bar([nombres[c] for c in c2.index], c2.values,
              color=[VERDE if c == "normal" else ROJO for c in c2.index], alpha=0.85)
    ax[1].set_title("D2: test completo\n(normal y ataques)")
    ax[1].tick_params(axis="x", rotation=45)

    c3 = attack_data["attack_category"].value_counts()
    ax[2].bar([nombres[c] for c in c3.index], c3.values, color=ROJO, alpha=0.85)
    ax[2].set_title("D3: entrenamiento de firmas\n(solo ataques conocidos)")
    ax[2].tick_params(axis="x", rotation=45)

    ax[3].bar(["D1", "D2", "D3"],
              [len(normal_data), len(test_df), len(attack_data)], color=[VERDE, AZUL_CLARO, ROJO], alpha=0.85)
    ax[3].set_title("Tamaño de los\ntres subconjuntos")
    ax[3].set_ylabel("Instancias (escala log.)")
    ax[3].set_yscale("log")

    top = attack_data["attack"].value_counts().head(8)
    ax[4].barh(range(len(top)), top.values)
    ax[4].set_yticks(range(len(top)))
    ax[4].set_yticklabels(top.index)
    ax[4].invert_yaxis()
    ax[4].set_title("Los 8 tipos de ataque\nmás frecuentes en D3")
    ax[4].set_xlabel("Frecuencia")
    ax[4].tick_params(axis="x", rotation=45)

    cats = ["normal", "dos", "probe", "r2l", "u2r"]
    d2p = [test_df["attack_category"].value_counts().get(c, 0) / len(test_df) * 100 for c in cats]
    d3p = [0] + [c3.get(c, 0) / len(attack_data) * 100 for c in cats if c != "normal"]
    x = np.arange(len(cats))
    ax[5].bar(x - 0.2, d2p, 0.4, label="D2 (test)")
    ax[5].bar(x + 0.2, d3p, 0.4, label="D3 (ataques)")
    ax[5].set_xticks(x)
    ax[5].set_xticklabels([nombres[c] for c in cats], rotation=45)
    ax[5].set_ylabel("Porcentaje (%)")
    ax[5].set_title("Reparto porcentual\nD2 frente a D3")
    ax[5].legend(frameon=False)

    feats = [f for f in ["duration", "src_bytes", "dst_bytes", "count"] if f in normal_data.columns]
    ax[6].bar(feats, [normal_data[f].mean() for f in feats], color=VERDE, alpha=0.85)
    ax[6].set_yscale("log")
    ax[6].set_title("Medias de algunas\ncaracterísticas en D1")
    ax[6].set_ylabel("Valor medio (escala log.)")
    ax[6].tick_params(axis="x", rotation=45)

    num = [f for f in ["duration", "src_bytes", "dst_bytes", "count", "srv_count"] if f in attack_data.columns]
    im = ax[7].imshow(attack_data[num].corr(), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax[7], shrink=0.8)
    ax[7].set_xticks(range(len(num)))
    ax[7].set_xticklabels(num, rotation=45, ha="right")
    ax[7].set_yticks(range(len(num)))
    ax[7].set_yticklabels(num)
    ax[7].set_title("Correlación entre\ncaracterísticas en D3")

    a, b = set(train_df["attack"].unique()), set(test_df["attack"].unique())
    ax[8].pie([len(a - b), len(b - a), len(a & b)], labels=["Solo\nentrenamiento", "Solo\ntest", "En los\ndos"],
              colors=[AZUL_CLARO, ROJO, VERDE], autopct="%1.1f%%", startangle=90,
              textprops={"fontsize": 9.5})
    ax[8].set_title("Cobertura de las\n40 etiquetas")

    fig.tight_layout(h_pad=1.6)
    guardar(fig, "eda_distribuciones_divisiones_memoria.png")


# ----------------------------------------------------------------------------- Figuras 4.3 y 4.4
class ValidadorMemoria(validacion.NSLKDDValidator):
    """Mismo cálculo que el validador; dibujo de memoria y fichero propio."""

    def _save_figure(self, filename):
        raise RuntimeError("no se escribe ninguna figura del validador original")

    def _plot_feature_distributions(self, mean_diff):
        top = mean_diff.head(12).index.tolist()
        fig, axes = plt.subplots(4, 3, figsize=(7.6, 9.4))
        for a, feat in zip(axes.flatten(), top):
            a.hist(self.D1_X[feat], bins=40, alpha=0.6, density=True, color="green", label="D1 (normal)")
            a.hist(self.D3_X[feat], bins=40, alpha=0.6, density=True, color="red", label="D3 (ataques)")
            a.set_title(feat, fontsize=10.5)
            a.tick_params(labelsize=9)
        h, l = axes.flatten()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        guardar(fig, "validacion_discriminantes_d1_vs_d3_memoria.png")

    def _plot_drift(self, drift_df):
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.4, 9.0), gridspec_kw={"height_ratios": [2.1, 1]})
        top20 = drift_df.nlargest(20, "ks_statistic")
        a0.barh(range(len(top20)), top20["ks_statistic"],
                color=["red" if x else "steelblue" for x in top20["has_drift"]])
        a0.set_yticks(range(len(top20)))
        a0.set_yticklabels(top20["feature"], fontsize=10)
        a0.invert_yaxis()
        a0.set_xlabel("Estadístico KS")
        a0.set_title("Las 20 características con mayor KS entre D1 y D2 completo\n"
                     "(en rojo, desplazamiento significativo, p < 0,01)")
        a1.hist(drift_df["ks_p_value"], bins=30, alpha=0.75, edgecolor="black")
        a1.axvline(x=0.01, color="red", linestyle="--", label="p = 0,01")
        a1.set_title("Distribución de los p-valores de las %d características" % len(drift_df))
        a1.set_xlabel("p-valor")
        a1.set_ylabel("Frecuencia")
        a1.legend(frameon=False)
        fig.tight_layout(h_pad=2.0)
        guardar(fig, "validacion_drift_ks_memoria.png")


def figuras_43_44():
    v = ValidadorMemoria()
    if not v.load_all_data():
        raise SystemExit("faltan los splits procesados: ejecuta program.py")
    v.analyze_feature_distributions()
    v.detect_data_drift()


if __name__ == "__main__":
    figura_41()
    figuras_43_44()
