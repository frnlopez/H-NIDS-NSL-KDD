# -*- coding: utf-8 -*-
"""Versiones de memoria (letra grande, formato de página, rótulos en español) de las figuras del capítulo 5.

Cubre las Figuras 5.1, 5.4, 5.5 y 5.7, con caracteres más grandes, y funde las curvas ROC y
precisión-recall en una sola figura de dos paneles.

Misma idea que `figuras_cap4_memoria.py`: mismos datos, solo cambia el dibujo. No toca los scripts que generan
las originales ni ninguna de sus salidas; escribe ficheros NUEVOS con sufijo `_memoria` en Resultados/figuras/.
Variante de 54 características, semilla 42.

DE DÓNDE SALEN LAS CIFRAS (cero `fit`, solo inferencia con los modelos persistidos)
- Etapa 1: `anomalia_<det>_54.joblib` (modelo + umbral p95) y el score unificado de
  `NSLKDDAnomalyTrainer._score`, igual que `anomalias.py::_procesar_algoritmo`.
- Etapa 2: `firma_<clf>_54.joblib` sobre los ataques de D2 de tipo conocido, con el mismo filtro que
  `firmas.py::cargar_datos`.
- Híbrido: los métodos de `NSLKDDHybridEvaluator` (`cargar_datos`, `_preparar_cascada`,
  `_ensamblar_prediccion`) con el `UMBRAL_CONF` leído de `hibrido_54.joblib`. No se llama a la calibración.
- Baseline: `baseline_rf_54.joblib` sobre D2 completo, como `baseline.py::evaluar`.
- Aprendizaje continuo: el mismo cálculo que `figura_aprendizaje_continuo.py` sobre
  `metricas_aprendizaje_continuo.csv` (constantes importadas de ese módulo).

CONTROL (aborta antes de dibujar si algo no cuadra)
- Cada matriz se compara celda a celda con REFERENCIA, transcrita de las figuras actuales
  (anomalias_cm_*_54.png / fig-cm-*-paneles.png, hibrido_cm_54.png y baseline_cm_54.png).
- Además, contra las tablas publicadas: tn/fp/fn/tp y AUC de metricas_anomalias.csv; accuracy, precisión,
  recall y soporte por clase de metricas_firmas.csv; bin_tn/fp/fn/tp y accuracy_5 de metricas_hibrido.csv;
  bin_tn/fp/fn/tp, accuracy y recall por clase de metricas_baseline.csv.

    python app/figuras_cap5_memoria.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

import config
import figura_aprendizaje_continuo as fac
from anomalias import NSLKDDAnomalyTrainer
from hibrido import NSLKDDHybridEvaluator
from program import load_specialized_splits

FIGS = config.FIGURAS_DIR
SET = "54"

# Ancho de caja de la página: ~16 cm. Las figuras se dibujan a ese ancho (6,3 in), así que el tamaño en
# puntos de matplotlib ES el tamaño impreso si se insertan a 16 cm.
ANCHO_IN = 6.3
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10})

DETECTORES = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "Autoencoder"]
CLASIFICADORES = ["DecisionTree", "RandomForest", "KNN", "HistGradientBoosting"]
COLOR_DET = {"IsolationForest": "#2a78d6", "OneClassSVM": "#eb6834",
             "LocalOutlierFactor": "#1baf7a", "Autoencoder": "#8e44ad"}
NOMBRE = {"normal": "normal", "ataque": "ataque", "dos": "DoS", "probe": "Probe", "r2l": "R2L",
          "u2r": "U2R", "unknown": "desconocido"}

# Transcritas de las figuras actuales (filas = real, columnas = predicción, en el orden de config).
REFERENCIA = {
    "anomalias": {
        "IsolationForest": [[8738, 973], [2957, 9876]],
        "OneClassSVM": [[8887, 824], [4021, 8812]],
        "LocalOutlierFactor": [[8153, 1558], [7071, 5762]],
        "Autoencoder": [[8723, 988], [2158, 10675]],
    },
    "firmas": {
        "DecisionTree": [[5326, 415, 0, 0], [2, 1104, 0, 0], [10, 320, 1856, 13], [7, 3, 16, 11]],
        "RandomForest": [[5714, 27, 0, 0], [0, 1106, 0, 0], [1, 214, 1956, 28], [0, 0, 18, 19]],
        "KNN": [[5708, 33, 0, 0], [1, 1105, 0, 0], [48, 141, 1846, 164], [0, 0, 9, 28]],
        "HistGradientBoosting": [[5684, 57, 0, 0], [1, 1105, 0, 0], [2, 274, 1891, 32], [0, 1, 17, 19]],
    },
    "hibrido": [[8723, 99, 326, 76, 8, 479], [600, 6137, 389, 0, 0, 332], [60, 171, 2158, 0, 0, 32],
                [1479, 2, 20, 864, 14, 375], [19, 0, 109, 21, 26, 25]],
    "baseline": [[9451, 67, 193, 0, 0], [1669, 5773, 16, 0, 0], [850, 138, 1433, 0, 0],
                 [2740, 0, 1, 12, 1], [194, 0, 1, 3, 2]],
}


def guardar(fig, nombre):
    ruta = os.path.join(FIGS, nombre)
    fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figura:", ruta)


def fila_publicada(tabla, algoritmo=None):
    d = pd.read_csv(os.path.join(config.RESULTADOS_DIR, tabla))
    d = d[(d["set_features"].astype(str) == SET) & (d["semilla"] == config.SEMILLA_POR_DEFECTO)]
    if algoritmo is not None:
        d = d[d["algoritmo"] == algoritmo]
    if len(d) != 1:
        raise SystemExit("%s: se esperaba una fila para %s y hay %d" % (tabla, algoritmo, len(d)))
    return d.iloc[0]


def comprobar(nombre, valor, esperado, tol=0.0):
    if isinstance(esperado, (list, np.ndarray)):
        ok = np.array_equal(np.asarray(valor), np.asarray(esperado))
    else:
        ok = abs(float(valor) - float(esperado)) <= tol
    if not ok:
        raise SystemExit("CONTROL FALLIDO %s: calculado %s, esperado %s" % (nombre, valor, esperado))
    return 1


def coma(x, dec=3):
    return ("{:.%df}" % dec).format(x).replace(".", ",")


def heatmap(ax, cm, filas, cols, tam_num=12):
    """Matriz con números grandes; texto blanco sobre las celdas oscuras."""
    ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            claro = cm[i, j] > 0.55 * cm.max()
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=tam_num,
                    color="white" if claro else "#1a1a1a")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NOMBRE[c] for c in cols])
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([NOMBRE[c] for c in filas])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)


# ----------------------------------------------------------------------------- datos
def cargar(prefijo, algo):
    d = joblib.load(os.path.join(config.MODELOS_DIR, "%s_%s_%s.joblib" % (prefijo, algo, SET)))
    if str(d.get("set_features")) != SET or int(d.get("semilla")) != config.SEMILLA_POR_DEFECTO:
        raise SystemExit("joblib %s_%s no es de la variante 54 / semilla 42" % (prefijo, algo))
    return d


def datos_etapa1(splits, n_controles):
    X_D2 = splits["D2"]["X"]
    cat = splits["D2"]["y_category"]["category_original"].astype(str).values
    y_bin = (cat != config.ETIQUETA_NORMAL).astype(int)
    anom = NSLKDDAnomalyTrainer(sin_seleccion=False)
    res = {}
    for det in DETECTORES:
        jd = cargar("anomalia", det)
        score = anom._score(det, jd["modelo"], X_D2)
        y_pred = (score > jd["umbral"]).astype(int)
        cm = confusion_matrix(y_bin, y_pred, labels=[0, 1])
        pub = fila_publicada("metricas_anomalias.csv", det)
        n_controles[0] += comprobar(det + " matriz vs figura actual", cm, REFERENCIA["anomalias"][det])
        n_controles[0] += comprobar(det + " matriz vs tabla", cm,
                                    [[pub["tn"], pub["fp"]], [pub["fn"], pub["tp"]]])
        auc, ap = roc_auc_score(y_bin, score), average_precision_score(y_bin, score)
        n_controles[0] += comprobar(det + " AUC-ROC", auc, pub["roc_auc"], 5e-7)
        n_controles[0] += comprobar(det + " AUC-PR", ap, pub["pr_auc"], 5e-7)
        res[det] = {"cm": cm, "score": score, "auc": auc, "ap": ap}
    return res, y_bin


def datos_etapa2(splits, n_controles):
    tipos_conocidos = set(splits["D3"]["y_attack"]["attack_original"].astype(str).unique())
    cat = splits["D2"]["y_category"]["category_original"].astype(str)
    tipo = splits["D2"]["y_attack"]["attack_original"].astype(str)
    mask = ((cat != config.ETIQUETA_NORMAL) & tipo.isin(tipos_conocidos)).values
    X = splits["D2"]["X"].loc[mask].reset_index(drop=True)
    y = cat.loc[mask].reset_index(drop=True).values
    labs = config.CATEGORIAS_ATAQUE
    res = {}
    for clf in CLASIFICADORES:
        y_pred = cargar("firma", clf)["modelo"].predict(X)
        cm = confusion_matrix(y, y_pred, labels=labs)
        pub = fila_publicada("metricas_firmas.csv", clf)
        n_controles[0] += comprobar(clf + " matriz vs figura actual", cm, REFERENCIA["firmas"][clf])
        n_controles[0] += comprobar(clf + " n_test", len(y), pub["n_test"])
        n_controles[0] += comprobar(clf + " accuracy", np.trace(cm) / cm.sum(), pub["accuracy_D2"], 5e-7)
        p, r, _, s = precision_recall_fscore_support(y, y_pred, labels=labs, zero_division=0)
        for k, c in enumerate(labs):
            n_controles[0] += comprobar("%s precision_%s" % (clf, c), p[k], pub["precision_" + c], 5e-7)
            n_controles[0] += comprobar("%s recall_%s" % (clf, c), r[k], pub["recall_" + c], 5e-7)
            n_controles[0] += comprobar("%s soporte_%s" % (clf, c), s[k], pub["soporte_" + c])
        res[clf] = cm
    return res


def datos_hibrido(n_controles):
    ev = NSLKDDHybridEvaluator(sin_seleccion=False)
    ev.cargar_datos()
    jd, _ = ev._cargar_joblib("anomalia", ev.detector)
    jf, _ = ev._cargar_joblib("firma", ev.firma)
    umbral_conf = joblib.load(os.path.join(config.MODELOS_DIR, "hibrido_%s.joblib" % SET))["umbral_conf_elegido"]
    ev._preparar_cascada(jd, jf)
    pred = ev._ensamblar_prediccion(umbral_conf)
    cm6 = confusion_matrix(ev.cat_D2, pred, labels=config.CATEGORIAS_HIBRIDO)
    cm = np.delete(cm6, config.CATEGORIAS_HIBRIDO.index(config.ETIQUETA_UNKNOWN), axis=0)
    pub = fila_publicada("metricas_hibrido.csv")
    n_controles[0] += comprobar("híbrido UMBRAL_CONF", umbral_conf, pub["umbral_conf_elegido"])
    n_controles[0] += comprobar("híbrido matriz vs figura actual", cm, REFERENCIA["hibrido"])
    tn = int(cm[0, 0]); fp = int(cm[0, 1:].sum()); fn = int(cm[1:, 0].sum()); tp = int(cm[1:, 1:].sum())
    n_controles[0] += comprobar("híbrido binaria vs tabla", [tn, fp, fn, tp],
                                [pub["bin_tn"], pub["bin_fp"], pub["bin_fn"], pub["bin_tp"]])
    n_controles[0] += comprobar("híbrido accuracy_5", np.trace(cm[:, :5]) / cm.sum(), pub["accuracy_5"], 5e-7)
    return cm


def datos_baseline(splits, n_controles):
    X = splits["D2"]["X"][splits["D1"]["X"].columns]
    y = splits["D2"]["y_category"]["category_original"].astype(str).values
    d = joblib.load(os.path.join(config.MODELOS_DIR, "baseline_rf_%s.joblib" % SET))
    if str(d.get("set_features")) != SET or int(d.get("semilla")) != config.SEMILLA_POR_DEFECTO:
        raise SystemExit("baseline_rf_54.joblib no es de la variante 54 / semilla 42")
    y_pred = d["modelo"].predict(X)
    labs = config.CATEGORIAS_MULTICLASE
    cm = confusion_matrix(y, y_pred, labels=labs)
    pub = fila_publicada("metricas_baseline.csv")
    n_controles[0] += comprobar("baseline matriz vs figura actual", cm, REFERENCIA["baseline"])
    tn = int(cm[0, 0]); fp = int(cm[0, 1:].sum()); fn = int(cm[1:, 0].sum()); tp = int(cm[1:, 1:].sum())
    n_controles[0] += comprobar("baseline binaria vs tabla", [tn, fp, fn, tp],
                                [pub["bin_tn"], pub["bin_fp"], pub["bin_fn"], pub["bin_tp"]])
    n_controles[0] += comprobar("baseline accuracy", np.trace(cm) / cm.sum(), pub["accuracy_D2"], 5e-7)
    r = np.diag(cm) / cm.sum(axis=1)
    for k, c in enumerate(labs):
        n_controles[0] += comprobar("baseline recall_" + c, r[k], pub["recall_" + c], 5e-7)
    return cm


# ----------------------------------------------------------------------------- dibujo
def fig_cm_anomalias(res):
    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO_IN, 5.6))
    for k, (ax, det) in enumerate(zip(ejes.flatten(), DETECTORES)):
        heatmap(ax, res[det]["cm"], ["normal", "ataque"], ["normal", "ataque"], tam_num=13)
        ax.set_title("(%s) %s" % ("abcd"[k], det), fontsize=11.5)
    fig.tight_layout(h_pad=1.8, w_pad=2.0)
    guardar(fig, "anomalias_cm_paneles_54_memoria.png")


def fig_roc_pr(res, y_bin):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(ANCHO_IN, 4.0))
    for det in DETECTORES:
        s, c = res[det]["score"], COLOR_DET[det]
        fpr, tpr, _ = roc_curve(y_bin, s, pos_label=config.CLASE_ATAQUE)
        a0.plot(fpr, tpr, color=c, linewidth=1.8)
        prec, rec, _ = precision_recall_curve(y_bin, s, pos_label=config.CLASE_ATAQUE)
        a1.plot(rec, prec, color=c, linewidth=1.8,
                label="%s (AUC-ROC = %s; AUC-PR = %s)" % (det, coma(res[det]["auc"]), coma(res[det]["ap"])))
    a0.plot([0, 1], [0, 1], "--", color="#8a8985", linewidth=1)
    a0.set_xlabel("Tasa de falsos positivos")
    a0.set_ylabel("Tasa de verdaderos positivos")
    a0.set_title("(a) Curvas ROC")
    a1.set_xlabel("Recall")
    a1.set_ylabel("Precisión")
    a1.set_title("(b) Curvas precisión-recall")
    for a in (a0, a1):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1.02)
        a.set_xticks([0, 0.25, 0.5, 0.75, 1])
        a.set_yticks([0, 0.25, 0.5, 0.75, 1])
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2)))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: coma(v, 2)))
        a.grid(color="#e4e3df", linewidth=0.8)
        a.set_axisbelow(True)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.22, 1, 1), w_pad=2.0)
    guardar(fig, "anomalias_roc_pr_54_memoria.png")


def fig_cm_firmas(res):
    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO_IN, 6.2))
    labs = config.CATEGORIAS_ATAQUE
    for k, (ax, clf) in enumerate(zip(ejes.flatten(), CLASIFICADORES)):
        heatmap(ax, res[clf], labs, labs, tam_num=11)
        ax.set_title("(%s) %s" % ("abcd"[k], "k-NN" if clf == "KNN" else clf), fontsize=11.5)
    fig.tight_layout(h_pad=1.8, w_pad=1.6)
    guardar(fig, "firmas_cm_paneles_54_memoria.png")


def fig_cm_unica(cm, filas, cols, nombre, alto):
    fig, ax = plt.subplots(figsize=(ANCHO_IN, alto))
    heatmap(ax, cm, filas, cols, tam_num=12.5)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    fig.tight_layout()
    guardar(fig, nombre)


def fig_aprendizaje_continuo():
    d = pd.read_csv(fac.CSV)
    ev = d[d["conjunto"] == "retirado_eval"]
    x = np.arange(len(fac.TAMANOS))
    fig, ejes = plt.subplots(2, 2, figsize=(ANCHO_IN, 6.4))
    techos = {}
    for k, (ax, (tipo, cat)) in enumerate(zip(ejes.flatten(), fac.TIPOS)):
        sub = ev[ev["tipo_retirado"] == tipo]
        r0 = sub[sub["metodo"] == "R0"]["pct_correcta"]
        techo = 100.0 - sub["pct_sin_alarma"].mean()
        techos[tipo] = techo
        for metodo, etiqueta, color, marca in fac.METODOS:
            m = sub[sub["metodo"] == metodo].groupby("n_llegada")["pct_correcta"]
            media = np.concatenate([[r0.mean()], m.mean().reindex(fac.TAMANOS[1:]).values])
            sd = np.concatenate([[r0.std()], m.std().reindex(fac.TAMANOS[1:]).values])
            ax.fill_between(x, media - sd, np.minimum(media + sd, 100), color=color, alpha=0.12, linewidth=0)
            ax.plot(x, media, color=color, linewidth=2, marker=marca, markersize=6.5,
                    markeredgecolor="white", markeredgewidth=1.0, label=etiqueta, zorder=3)
        ax.axhline(techo, color=fac.TINTA_2, linestyle="--", linewidth=1.2, zorder=2,
                   label="Máximo que permite la etapa 1")
        ax.text(x[0] - 0.25, techo + 2.5, coma(techo, 1) + " %", ha="left", va="bottom",
                fontsize=10, color=fac.TINTA_2)
        ax.set_title("(%s) Sin %s (%s)" % ("abc"[k], tipo, cat), color=fac.TINTA)
        ax.set_xticks(x)
        ax.set_xticklabels(["sin\nactualizar"] + [str(t) for t in fac.TAMANOS[1:]], fontsize=10)
        ax.set_xlabel("Flujos etiquetados acumulados", color=fac.TINTA_2)
        ax.set_ylim(0, 114)
        ax.set_yticks(range(0, 101, 20))
        if k != 1:
            ax.set_ylabel("% con la categoría correcta", color=fac.TINTA_2)
        ax.grid(axis="y", color=fac.REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=fac.TINTA_2)
    ley = ejes[1, 1]
    ley.axis("off")
    h, l = ejes[0, 0].get_legend_handles_labels()
    ley.legend(h, l, loc="center", frameon=False, fontsize=10.5, handlelength=2.6, labelspacing=1.1)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    guardar(fig, "aprendizaje_continuo_curvas_memoria.png")
    return techos


def main():
    config.setup_utf8()
    n = [0]
    splits = load_specialized_splits(config.base_path(sin_seleccion=False))
    if splits is None:
        raise SystemExit("faltan los splits procesados de 54: ejecuta program.py")
    e1, y_bin = datos_etapa1(splits, n)
    e2 = datos_etapa2(splits, n)
    hib = datos_hibrido(n)
    base = datos_baseline(splits, n)
    print("CONTROL: %d comprobaciones superadas (matrices celda a celda y tablas publicadas)" % n[0])
    for det in DETECTORES:
        print("   %-20s AUC-ROC=%.6f AUC-PR=%.6f" % (det, e1[det]["auc"], e1[det]["ap"]))

    fig_cm_anomalias(e1)
    fig_roc_pr(e1, y_bin)
    fig_cm_firmas(e2)
    fig_cm_unica(hib, config.CATEGORIAS_MULTICLASE, config.CATEGORIAS_HIBRIDO,
                 "hibrido_cm_54_memoria.png", 4.6)
    fig_cm_unica(base, config.CATEGORIAS_MULTICLASE, config.CATEGORIAS_MULTICLASE,
                 "baseline_cm_54_memoria.png", 4.6)
    techos = fig_aprendizaje_continuo()
    print("techos de la etapa 1 (Figura 5.7):", {k: round(v, 1) for k, v in techos.items()})


if __name__ == "__main__":
    main()
