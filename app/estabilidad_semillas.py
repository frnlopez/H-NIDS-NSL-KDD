# -*- coding: utf-8 -*-
"""
estabilidad_semillas.py — Resumen estadístico de la estabilidad frente a la semilla.

Qué pregunta responde
---------------------
¿Los resultados presentan mucha dispersión con la semilla,
o pueden considerarse ESTACIONARIOS, de modo que una ejecución sea representativa
de las demás porque conserva sus estadísticos? No importa qué semilla sea la 42;
importa si los datos son estables y cuál es el resultado general.

Tres bloques, todos en `Resultados/estabilidad_semillas.csv` (columna `bloque`) y
legibles en `Resultados/estabilidad_semillas.md`:

  1. `dispersion`: por métrica clave de cada etapa y del sistema, y por variante
     (54 y 122 características), n, media, desviación típica muestral (ddof=1),
     coeficiente de variación (sd/|media|), mín y máx de las diez semillas de
     `config.SEMILLAS_BARRIDO`, con una etiqueta de estabilidad por CV.
  2. En las mismas filas, la REPRESENTATIVIDAD de la semilla 42 (la publicada):
     su valor, si cae dentro de [mín, máx] de las diez, su z respecto a ellas, su
     percentil y una Wilcoxon de una muestra (las diez frente al valor de la 42:
     ¿está la 42 en el centro de la distribución?).
  3. `wilcoxon`: pruebas de rangos con signo PAREADAS POR SEMILLA para las
     comparaciones que sostienen decisiones del trabajo, con victorias, mediana de
     la diferencia, estadístico, p exacto bilateral, p corregido por Holm sobre
     toda la familia y tamaño del efecto (correlación rango-biserial emparejada).

Reglas de protocolo
-------------------
  * CERO `fit`. Solo lee CSV y calcula estadísticos.
  * NO ESCRIBE EN NINGUNA TABLA PUBLICADA. Lee en modo lectura las
    `metricas_*_semillas.csv` (semillas 1..10) y las `metricas_*.csv` publicadas
    (solo para el valor de la semilla 42). Sus dos salidas no empiezan por
    `metricas_` y se comprueba por ruta que no coinciden con ninguna tabla leída.
  * ABORTA si a una celda le falta alguna semilla del barrido, o si una semilla
    trae más de una fila: una media de 9 presentada como de 10 es peor que nada.
  * La semilla 42 NO entra en la media ni en las pruebas pareadas principales
    (mismo criterio que `agregar_semillas.py`: es el titular publicado y el punto
    cuya representatividad se mide). Sí entra en una columna de SENSIBILIDAD
    (`p_con_42`), que repite cada prueba pareada con once pares.
  * Los valores se comparan tal como se persistieron (6 decimales): las
    diferencias pareadas se redondean a 6 decimales antes de buscar empates, para
    que un 0,000000 no se convierta en ±1e-17 por aritmética de coma flotante.

Wilcoxon: cómo y con qué límite
-------------------------------
`scipy.stats.wilcoxon(..., alternative='two-sided', method='exact')`. Con n=10
pares el p bilateral exacto más pequeño posible es 2/2^10 ≈ 0,00195 (todas las
diferencias del mismo signo): ninguna prueba de esta tabla puede dar menos. Si hay
diferencias nulas (se descartan, `zero_method='wilcox'`) o empates en |d|, la
distribución exacta no aplica y se usa la aproximación normal con corrección de
empates; la columna `metodo` dice cuál se usó en cada fila.

Holm se aplica sobre TODA la familia de pruebas pareadas a la vez (no por bloque),
que es lo conservador. Con 18 pruebas, una victoria 10 de 10 queda en p_holm ≈
0,035: sigue por debajo de 0,05, pero ya no por mucho.

Tamaño del efecto: correlación rango-biserial emparejada, r = (W+ − W−)/(W+ + W−),
entre −1 y 1. Vale ±1 cuando todas las diferencias van en el mismo sentido.

Límites que el `.md` declara
----------------------------
Las once corridas comparten el MISMO dataset y los MISMOS splits D1/D2/D3 y la
misma selección de 54 características (`program.py` no depende de la semilla). Lo
que varía es el split 80/20 de D1 (y con él el umbral p95), las particiones de la
validación cruzada, las submuestras y la inicialización de los modelos. Por tanto
estas pruebas miden la VARIABILIDAD POR SEMILLA, no la variabilidad muestral del
dataset: «significativo» quiere decir «no es un accidente de la semilla», no «se
generaliza a otro tráfico». Con n=10 la potencia es baja: un «no significativo»
no demuestra igualdad. Y la semilla 42 se produjo con un commit distinto del de
las diez (la columna `commit_42` lo dice), así que su distancia a la banda puede
incluir deriva de código, no solo de semilla.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import config


# ----------------------------------------------------------------------------
# Especificación
# ----------------------------------------------------------------------------
SETS = ["54", "122_sin_seleccion"]
ETIQUETA_SET = {"54": "54", "122_sin_seleccion": "122"}

# Umbrales de la etiqueta de estabilidad por coeficiente de variación.
CV_ESTABLE = 0.03
CV_MODERADA = 0.10

NOMBRE_CSV = "estabilidad_semillas.csv"
NOMBRE_MD = "estabilidad_semillas.md"

# (etapa, tabla publicada, filtro de fila, columna, etiqueta legible).
# El filtro + set_features + semilla debe dar exactamente una fila.
METRICAS = [
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "Autoencoder"},
     "recall", "Autoencoder · recall (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "Autoencoder"},
     "fpr", "Autoencoder · FPR (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "Autoencoder"},
     "f1", "Autoencoder · F1 (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "Autoencoder"},
     "roc_auc", "Autoencoder · AUC-ROC (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "IsolationForest"},
     "recall", "IsolationForest · recall (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "IsolationForest"},
     "fpr", "IsolationForest · FPR (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "IsolationForest"},
     "f1", "IsolationForest · F1 (D2)"),
    ("Etapa 1", "metricas_anomalias.csv", {"algoritmo": "IsolationForest"},
     "roc_auc", "IsolationForest · AUC-ROC (D2)"),
    ("Etapa 2", "metricas_firmas.csv", {"algoritmo": "RandomForest"},
     "f1_macro", "RandomForest · F1 macro (ataques conocidos de D2)"),
    ("Etapa 2", "metricas_firmas.csv", {"algoritmo": "HistGradientBoosting"},
     "f1_macro", "HistGradientBoosting · F1 macro (ataques conocidos de D2)"),
    ("Híbrido", "metricas_hibrido.csv", {}, "recall_0day_global",
     "Híbrido · recall 0-day"),
    ("Híbrido", "metricas_hibrido.csv", {}, "fpr_cascada",
     "Híbrido · FPR"),
    ("Híbrido", "metricas_hibrido.csv", {}, "conocida_f1_macro",
     "Híbrido · F1 macro conocidas (extremo a extremo)"),
    ("Baseline", "metricas_baseline.csv", {}, "recall_0day_global",
     "Baseline · recall 0-day"),
]

# Filas de la tabla compacta para la memoria (columna, etiqueta corta).
COMPACTA = [
    ("Etapa 1", "Autoencoder · recall (D2)", "Autoencoder: recall"),
    ("Etapa 1", "Autoencoder · F1 (D2)", "Autoencoder: F1"),
    ("Etapa 1", "Autoencoder · AUC-ROC (D2)", "Autoencoder: AUC-ROC"),
    ("Etapa 2", "RandomForest · F1 macro (ataques conocidos de D2)",
     "RandomForest: F1 macro"),
    ("Híbrido", "Híbrido · recall 0-day", "Híbrido: recall 0-day"),
    ("Híbrido", "Híbrido · FPR", "Híbrido: FPR"),
    ("Híbrido", "Híbrido · F1 macro conocidas (extremo a extremo)",
     "Híbrido: F1 macro conocidas"),
    ("Baseline", "Baseline · recall 0-day", "Baseline: recall 0-day"),
]

# Comparaciones pareadas por semilla. Cada lado: (tabla, set_features, filtro,
# columna). A − B > 0 significa que gana A (todas son «mayor es mejor»).
def _lado(tabla, set_features, filtro, columna):
    return {"tabla": tabla, "set": set_features, "filtro": filtro,
            "columna": columna}


def _comparaciones():
    comps = []
    for s in SETS:
        e = ETIQUETA_SET[s]
        comps.append({
            "grupo": "Algoritmo de la etapa 2",
            "nombre": "RandomForest vs HistGradientBoosting · F1 macro D2 · {}".format(e),
            "a": "RandomForest", "b": "HistGradientBoosting",
            "lado_a": _lado("metricas_firmas.csv", s, {"algoritmo": "RandomForest"}, "f1_macro"),
            "lado_b": _lado("metricas_firmas.csv", s, {"algoritmo": "HistGradientBoosting"}, "f1_macro"),
        })
    comps.append({
        "grupo": "Variante de características",
        "nombre": "54 vs 122 · recall 0-day del híbrido",
        "a": "54", "b": "122",
        "lado_a": _lado("metricas_hibrido.csv", "54", {}, "recall_0day_global"),
        "lado_b": _lado("metricas_hibrido.csv", "122_sin_seleccion", {}, "recall_0day_global"),
    })
    comps.append({
        "grupo": "Variante de características",
        "nombre": "54 vs 122 · F1 macro D2 de RandomForest (firmas)",
        "a": "54", "b": "122",
        "lado_a": _lado("metricas_firmas.csv", "54", {"algoritmo": "RandomForest"}, "f1_macro"),
        "lado_b": _lado("metricas_firmas.csv", "122_sin_seleccion", {"algoritmo": "RandomForest"}, "f1_macro"),
    })
    for s in SETS:
        e = ETIQUETA_SET[s]
        comps.append({
            "grupo": "Híbrido frente al baseline",
            "nombre": "Híbrido vs baseline · recall 0-day · {}".format(e),
            "a": "Híbrido", "b": "Baseline",
            "lado_a": _lado("metricas_hibrido.csv", s, {}, "recall_0day_global"),
            "lado_b": _lado("metricas_baseline.csv", s, {}, "recall_0day_global"),
        })
    for col, etq in (("f1", "F1"), ("recall", "recall")):
        for s in SETS:
            e = ETIQUETA_SET[s]
            comps.append({
                "grupo": "Detector de la etapa 1",
                "nombre": "Autoencoder vs IsolationForest · {} D2 · {}".format(etq, e),
                "a": "Autoencoder", "b": "IsolationForest",
                "lado_a": _lado("metricas_anomalias.csv", s, {"algoritmo": "Autoencoder"}, col),
                "lado_b": _lado("metricas_anomalias.csv", s, {"algoritmo": "IsolationForest"}, col),
            })
    for opcion_b, algos in (("class_weight", ("DecisionTree", "RandomForest")),
                            ("nada", ("HistGradientBoosting", "KNN"))):
        for algo in algos:
            for s in SETS:
                e = ETIQUETA_SET[s]
                comps.append({
                    "grupo": "Balanceo (criterio de selección, CV sobre D3)",
                    "nombre": "SMOTE vs {} · F1 macro CV · {} · {}".format(opcion_b, algo, e),
                    "a": "SMOTE", "b": opcion_b,
                    "lado_a": _lado("metricas_balanceo.csv", s,
                                    {"algoritmo": algo, "balanceo": "SMOTE"}, "f1_macro_cv"),
                    "lado_b": _lado("metricas_balanceo.csv", s,
                                    {"algoritmo": algo, "balanceo": opcion_b}, "f1_macro_cv"),
                })
    return comps


# ----------------------------------------------------------------------------
# Utilidades numéricas
# ----------------------------------------------------------------------------
def holm(pvalores):
    """p ajustados de Holm-Bonferroni (monótonos, recortados a 1)."""
    p = np.asarray(pvalores, dtype=float)
    m = len(p)
    orden = np.argsort(p, kind="mergesort")
    ajust = np.empty(m)
    acumulado = 0.0
    for i, idx in enumerate(orden):
        valor = min(1.0, (m - i) * p[idx])
        acumulado = max(acumulado, valor)
        ajust[idx] = acumulado
    return ajust


def wilcoxon_pareado(d):
    """
    Wilcoxon de rangos con signo sobre las diferencias `d` (ya redondeadas).
    Devuelve dict con n_efectivo, W+, W-, estadístico, p, método y r rango-biserial.
    """
    d = np.asarray(d, dtype=float)
    nz = d[d != 0]
    n_ef = len(nz)
    res = {"n_efectivo": n_ef, "w_mas": np.nan, "w_menos": np.nan,
           "estadistico": np.nan, "p": np.nan, "metodo": "",
           "r_rango_biserial": np.nan,
           "p_minimo_posible": (2.0 / 2 ** n_ef) if n_ef > 0 else np.nan}
    if n_ef == 0:
        res["metodo"] = "sin diferencias no nulas"
        return res
    rangos = stats.rankdata(np.abs(nz))
    w_mas = float(rangos[nz > 0].sum())
    w_menos = float(rangos[nz < 0].sum())
    hay_empates = len(np.unique(np.abs(nz))) < n_ef
    hay_ceros = n_ef < len(d)
    if hay_empates or hay_ceros:
        metodo = "approx"
        etiqueta = "aproximación normal (hay {}{})".format(
            "ceros" if hay_ceros else "",
            (" y " if hay_ceros and hay_empates else "") +
            ("empates en |d|" if hay_empates else ""))
    else:
        metodo = "exact"
        etiqueta = "exacta"
    r = stats.wilcoxon(nz, alternative="two-sided", method=metodo,
                       zero_method="wilcox")
    res.update({
        "w_mas": w_mas, "w_menos": w_menos,
        "estadistico": float(r.statistic), "p": float(r.pvalue),
        "metodo": etiqueta,
        "r_rango_biserial": (w_mas - w_menos) / (w_mas + w_menos),
    })
    return res


def etiqueta_cv(cv):
    if cv <= CV_ESTABLE:
        return "estable"
    if cv <= CV_MODERADA:
        return "moderada"
    return "alta"


# ----------------------------------------------------------------------------
# Lectura
# ----------------------------------------------------------------------------
class LectorTablas:
    def __init__(self):
        self.cache = {}
        self.rutas_leidas = set()

    def _leer(self, nombre):
        ruta = os.path.join(config.RESULTADOS_DIR, nombre)
        if ruta not in self.cache:
            if not os.path.exists(ruta):
                raise FileNotFoundError("No existe la tabla '{}'.".format(ruta))
            self.cache[ruta] = pd.read_csv(ruta)
            self.rutas_leidas.add(os.path.normcase(os.path.abspath(ruta)))
        return self.cache[ruta]

    def valor(self, tabla_publicada, set_features, filtro, columna, semilla):
        """Valor de una celda para UNA semilla; aborta si no hay exactamente 1 fila."""
        if semilla == config.SEMILLA_POR_DEFECTO:
            nombre = tabla_publicada
        else:
            nombre = config.nombre_tabla_semillas(tabla_publicada)
        df = self._leer(nombre)
        m = (df["set_features"].astype(str) == set_features) & \
            (df["semilla"].astype(int) == int(semilla))
        for col, val in filtro.items():
            m &= df[col].astype(str) == val
        sub = df[m]
        if len(sub) != 1:
            raise RuntimeError(
                "'{}' · set={} · {} · semilla {}: se esperaba 1 fila y hay {}.".format(
                    nombre, set_features, filtro, semilla, len(sub)))
        fila = sub.iloc[0]
        return float(fila[columna]), str(fila.get("commit", ""))

    def serie(self, tabla, set_features, filtro, columna, semillas):
        valores, commits = [], set()
        for s in semillas:
            v, c = self.valor(tabla, set_features, filtro, columna, s)
            valores.append(v)
            commits.add(c)
        return np.array(valores), commits


# ----------------------------------------------------------------------------
# Cálculo
# ----------------------------------------------------------------------------
class EstabilidadSemillas:
    def __init__(self):
        self.semillas = sorted(config.SEMILLAS_BARRIDO)
        self.s42 = config.SEMILLA_POR_DEFECTO
        self.lector = LectorTablas()
        self.filas_disp = []
        self.filas_wil = []

    def _dispersion(self):
        for etapa, tabla, filtro, col, etq in METRICAS:
            for s in SETS:
                x, commits = self.lector.serie(tabla, s, filtro, col, self.semillas)
                v42, c42 = self.lector.valor(tabla, s, filtro, col, self.s42)
                media = float(np.mean(x))
                sd = float(np.std(x, ddof=1))
                cv = sd / abs(media) if media != 0 else np.nan
                # Una muestra: las diez frente al valor de la 42 (diferencias a 6 dec.)
                d1 = np.round(x - v42, 6)
                w1 = wilcoxon_pareado(d1)
                self.filas_disp.append({
                    "bloque": "dispersion",
                    "etapa": etapa,
                    "metrica": etq,
                    "tabla_origen": config.nombre_tabla_semillas(tabla),
                    "columna": col,
                    "set_features": s,
                    "n": len(x),
                    "media": media,
                    "sd": sd,
                    "cv": cv,
                    "estabilidad_cv": etiqueta_cv(cv),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "rango": float(np.max(x) - np.min(x)),
                    "valor_42": v42,
                    "dentro_rango_42": bool(np.min(x) <= v42 <= np.max(x)),
                    "z_42": (v42 - media) / sd if sd > 0 else np.nan,
                    "percentil_42": 100.0 * (np.sum(x < v42) + 0.5 * np.sum(x == v42)) / len(x),
                    "semillas_por_debajo_42": int(np.sum(x < v42)),
                    "p_wilcoxon_1muestra_42": w1["p"],
                    "metodo_1muestra": w1["metodo"],
                    "commits_barrido": " ".join(sorted(commits)),
                    "commit_42": c42,
                })

    def _pareadas(self):
        comps = _comparaciones()
        for c in comps:
            la, lb = c["lado_a"], c["lado_b"]
            xa, ca = self.lector.serie(la["tabla"], la["set"], la["filtro"], la["columna"], self.semillas)
            xb, cb = self.lector.serie(lb["tabla"], lb["set"], lb["filtro"], lb["columna"], self.semillas)
            d = np.round(xa - xb, 6)
            w = wilcoxon_pareado(d)
            # Sensibilidad: con la 42 como undécimo par.
            a42, _ = self.lector.valor(la["tabla"], la["set"], la["filtro"], la["columna"], self.s42)
            b42, _ = self.lector.valor(lb["tabla"], lb["set"], lb["filtro"], lb["columna"], self.s42)
            w11 = wilcoxon_pareado(np.append(d, round(a42 - b42, 6)))
            self.filas_wil.append({
                "bloque": "wilcoxon",
                "etapa": c["grupo"],
                "metrica": c["nombre"],
                "opcion_a": c["a"],
                "opcion_b": c["b"],
                "tabla_origen": config.nombre_tabla_semillas(la["tabla"]) + (
                    "" if la["tabla"] == lb["tabla"]
                    else " + " + config.nombre_tabla_semillas(lb["tabla"])),
                "columna": la["columna"],
                "n": len(d),
                "n_efectivo": w["n_efectivo"],
                "gana_a": int(np.sum(d > 0)),
                "gana_b": int(np.sum(d < 0)),
                "empates": int(np.sum(d == 0)),
                "media_a": float(np.mean(xa)),
                "media_b": float(np.mean(xb)),
                "mediana_diferencia": float(np.median(d)),
                "w_mas": w["w_mas"],
                "w_menos": w["w_menos"],
                "estadistico": w["estadistico"],
                "p": w["p"],
                "p_minimo_posible": w["p_minimo_posible"],
                "metodo": w["metodo"],
                "r_rango_biserial": w["r_rango_biserial"],
                "p_con_42": w11["p"],
                "commits_barrido": " ".join(sorted(ca | cb)),
            })
        ajust = holm([f["p"] for f in self.filas_wil])
        for f, ph in zip(self.filas_wil, ajust):
            f["p_holm"] = float(ph)
            f["significativa_holm_5pct"] = bool(ph < 0.05)

    # ------------------------------------------------------------------
    # Salidas
    # ------------------------------------------------------------------
    def _comprobar_salidas(self):
        for nombre in (NOMBRE_CSV, NOMBRE_MD):
            if nombre.startswith("metricas_"):
                raise RuntimeError("Salida con nombre de tabla de métricas: " + nombre)
            ruta = os.path.normcase(os.path.abspath(
                os.path.join(config.RESULTADOS_DIR, nombre)))
            if ruta in self.lector.rutas_leidas:
                raise RuntimeError("La salida coincide con una tabla leída: " + ruta)

    def _escribir_csv(self):
        df = pd.concat([pd.DataFrame(self.filas_disp), pd.DataFrame(self.filas_wil)],
                       ignore_index=True, sort=False)
        for col in ("n", "n_efectivo", "gana_a", "gana_b", "empates",
                    "semillas_por_debajo_42"):
            df[col] = df[col].astype("Int64")
        df["commit_script"] = self.commit
        df["fecha"] = self.fecha
        ruta = os.path.join(config.RESULTADOS_DIR, NOMBRE_CSV)
        df.to_csv(ruta, index=False, float_format="%.6f")
        print("   CSV: {} ({} filas)".format(ruta, len(df)))

    @staticmethod
    def _n(x, dec=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return "{:.{}f}".format(x, dec).replace(".", ",")

    @staticmethod
    def _p(p):
        if p is None or np.isnan(p):
            return "—"
        if p < 0.001:
            return "<0,001"
        return "{:.4f}".format(p).replace(".", ",")

    def _pct(self, x, dec=1):
        return self._n(100 * x, dec) + " %"

    def _md(self):
        n, d = self._n, self.filas_disp
        disp = pd.DataFrame(d)
        wil = pd.DataFrame(self.filas_wil)
        nsem = len(self.semillas)
        p_min = 2.0 / 2 ** nsem
        L = []
        L += [
            "# Estabilidad de los resultados frente a la semilla",
            "",
            "> Generado por `app/estabilidad_semillas.py`. **No se edita a "
            "mano**: se regenera corriendo el script. Cero `fit`: solo lee las tablas "
            "del barrido (`metricas_*_semillas.csv`, semillas {}) y, en lectura, las "
            "publicadas para el valor de la semilla {}.".format(
                ", ".join(str(s) for s in self.semillas), self.s42),
            "> Commit del script: `{}` · Fecha: {} · Commit(s) de las diez semillas: {} · "
            "Commit(s) de la semilla {}: {}.".format(
                self.commit, self.fecha,
                ", ".join("`{}`".format(c) for c in sorted(set(disp["commits_barrido"]))),
                self.s42,
                ", ".join("`{}`".format(c) for c in sorted(set(disp["commit_42"])))),
            "",
            "**Qué se mide.** Si los resultados cambian mucho al cambiar la semilla o si se "
            "pueden considerar estables, de modo que una sola ejecución —la publicada, "
            "semilla {}— sea representativa del conjunto. Para cada métrica: media, "
            "desviación típica (sd, muestral) y coeficiente de variación (CV = sd/media) "
            "de las {} semillas del barrido; dónde cae la semilla {} respecto a ellas; y "
            "pruebas de Wilcoxon pareadas por semilla para las comparaciones que "
            "sostienen decisiones del trabajo.".format(self.s42, nsem, self.s42),
            "",
            "Criterio de estabilidad por CV: **estable** si CV ≤ {:.0f} %, **moderada** "
            "entre {:.0f} % y {:.0f} %, **alta** por encima. Para tasas pequeñas como el "
            "FPR el CV exagera (una sd de un punto sobre una media de 0,09 ya es un 11 %): "
            "ahí conviene leer también la sd en puntos.".format(
                100 * CV_ESTABLE, 100 * CV_ESTABLE, 100 * CV_MODERADA),
            "",
        ]

        # --- Tabla compacta
        L += ["## Tabla resumen (para la memoria)", "",
              "Media ± sd de las {} semillas, CV, y valor de la semilla {} con su z "
              "(distancia a la media en desviaciones típicas). Un asterisco marca que la "
              "semilla {} cae **fuera** del intervalo [mín, máx] de las {}.".format(
                  nsem, self.s42, self.s42, nsem),
              "El FPR del híbrido es por construcción el del Autoencoder de la etapa 1 "
              "(la etapa 2 nunca devuelve un flujo a «normal»), por eso no se repite "
              "la fila del Autoencoder.", "",
              "| Métrica | 54: media ± sd | 54: CV | 54: semilla {0} (z) | 122: media ± sd | "
              "122: CV | 122: semilla {0} (z) |".format(self.s42),
              "|---|---|---|---|---|---|---|"]
        for etapa, metrica, corta in COMPACTA:
            celdas = [corta]
            for s in SETS:
                r = disp[(disp.metrica == metrica) & (disp.set_features == s)].iloc[0]
                celdas += ["{} ± {}".format(n(r.media), n(r.sd)),
                           self._pct(r.cv),
                           "{}{} ({})".format(n(r.valor_42), "" if r.dentro_rango_42 else "*",
                                              n(r.z_42, 2))]
            L.append("| " + " | ".join(celdas) + " |")
        L.append("")

        # --- Conclusión cuantitativa
        tot = len(disp)
        estables = disp[disp.estabilidad_cv == "estable"]
        moder = disp[disp.estabilidad_cv == "moderada"]
        altas = disp[disp.estabilidad_cv == "alta"]
        dentro = int(disp.dentro_rango_42.sum())
        z2 = int((disp.z_42.abs() <= 2).sum())

        def lista(sub):
            return "; ".join("{} ({}, CV {})".format(
                r.metrica, ETIQUETA_SET[r.set_features], self._pct(r.cv))
                for r in sub.itertuples()) or "ninguna"

        L += ["## Conclusión", "",
              "- **Estables (CV ≤ {:.0f} %)**: {} de {} métricas. {}.".format(
                  100 * CV_ESTABLE, len(estables), tot, lista(estables)),
              "- **Dispersión moderada ({:.0f}-{:.0f} %)**: {} de {}. {}.".format(
                  100 * CV_ESTABLE, 100 * CV_MODERADA, len(moder), tot, lista(moder)),
              "- **Dispersión alta (CV > {:.0f} %)**: {} de {}. {}.".format(
                  100 * CV_MODERADA, len(altas), tot, lista(altas)),
              "- **Semilla {} representativa**: cae dentro de [mín, máx] de las {} en {} "
              "de {} métricas, y a no más de 2 desviaciones típicas de la media en {} de "
              "{}.".format(self.s42, nsem, dentro, tot, z2, tot),
              ]
        sig = wil[wil.p_holm < 0.05]
        L.append("- **Pruebas pareadas**: {} de {} comparaciones resisten la corrección "
                 "de Holm, todas con victoria {} de {}: {}.".format(
                     len(sig), len(wil), nsem, nsem,
                     "; ".join(sig.metrica)))
        fuera = disp[~disp.dentro_rango_42]
        if len(fuera):
            L.append("- **Fuera de rango**: " + "; ".join(
                "{} ({}): {} frente a [{}, {}], z = {}".format(
                    r.metrica, ETIQUETA_SET[r.set_features], n(r.valor_42),
                    n(r.min), n(r.max), n(r.z_42, 2)) for r in fuera.itertuples()) + ".")
        L.append("")

        # --- Pruebas
        L += ["## Pruebas de Wilcoxon pareadas por semilla", "",
              "Prueba de rangos con signo, **bilateral y exacta**, emparejando las dos "
              "opciones dentro de la misma semilla ({} pares). Con {} pares el p más "
              "pequeño que puede salir es 2/2^{} ≈ {}: es lo que da una victoria de {} "
              "a 0, y ninguna prueba puede bajar de ahí. Como se hacen {} pruebas, el p se "
              "corrige por **Holm** sobre todas a la vez; se muestran los dos. El tamaño "
              "del efecto es la correlación rango-biserial emparejada *r* (de −1 a 1; ±1 "
              "si todas las diferencias van en el mismo sentido). La columna «con la "
              "{}» repite la prueba añadiendo la semilla publicada como undécimo par "
              "(sensibilidad, sin Holm).".format(
                  nsem, nsem, nsem, self._n(p_min, 5), nsem,
                  len(wil), self.s42), "",
              "| Comparación (A vs B) | Gana A | Mediana A−B | W | p | p Holm | r | "
              "p con la {} | Lectura |".format(self.s42),
              "|---|---|---|---|---|---|---|---|---|"]
        for r in wil.itertuples():
            if r.p_holm < 0.05:
                lect = "{} supera a {} de forma consistente entre semillas.".format(
                    r.opcion_a if r.mediana_diferencia > 0 else r.opcion_b,
                    r.opcion_b if r.mediana_diferencia > 0 else r.opcion_a)
            elif r.p < 0.05:
                lect = ("Diferencia a favor de {} sin corregir, pero no resiste la "
                        "corrección por pruebas múltiples.").format(
                    r.opcion_a if r.mediana_diferencia > 0 else r.opcion_b)
            else:
                lect = ("No se distingue de la variación por semilla (con {} pares "
                        "esto no prueba que sean iguales).").format(r.n)
            metodo = "" if r.metodo == "exacta" else " ({})".format(r.metodo)
            L.append("| {} | {} de {} | {} | {} | {}{} | {} | {} | {} | {} |".format(
                r.metrica, r.gana_a, r.n, self._n(r.mediana_diferencia, 4),
                self._n(r.estadistico, 1), self._p(r.p), metodo, self._p(r.p_holm),
                self._n(r.r_rango_biserial, 2), self._p(r.p_con_42), lect))
        L += ["",
              "Las filas de **balanceo** comparan el F1 macro de la validación cruzada "
              "sobre D3: son el criterio con el que se eligió el balanceo (4.3.4), no un "
              "resultado sobre D2.", ""]

        # --- Representatividad completa
        L += ["## Dispersión completa y posición de la semilla {}".format(self.s42), "",
              "Percentil = porcentaje de las {} semillas por debajo de la {} (empates "
              "cuentan la mitad). «p 1 muestra» es una Wilcoxon de las {} semillas "
              "frente al valor de la {}: un p bajo dice que la {} no está en el centro "
              "de la distribución, aunque caiga dentro del rango; un p alto no "
              "demuestra nada por sí solo. Sin corrección por pruebas múltiples: es "
              "descriptivo.".format(nsem, self.s42, nsem, self.s42, self.s42), "",
              "| Etapa | Métrica | Var. | Media | sd | CV | Mín | Máx | Semilla {0} | "
              "¿Dentro? | z | Percentil | p 1 muestra | Estabilidad |".format(self.s42),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in disp.itertuples():
            L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                r.etapa, r.metrica, ETIQUETA_SET[r.set_features], n(r.media, 4),
                n(r.sd, 4), self._pct(r.cv), n(r.min, 4), n(r.max, 4),
                n(r.valor_42, 4), "sí" if r.dentro_rango_42 else "**no**",
                n(r.z_42, 2), self._n(r.percentil_42, 0) + " %",
                self._p(r.p_wilcoxon_1muestra_42), r.estabilidad_cv))
        L.append("")

        # --- Límites
        L += ["## Límites", "",
              "- **Mismo dataset en todas las corridas.** Las {} ejecuciones usan los "
              "mismos D1, D2 y D3 y la misma selección de 54 características (no "
              "dependen de la semilla). Lo que cambia con la semilla es la partición "
              "80/20 de D1 (y con ella el umbral p95), las particiones de la validación "
              "cruzada, las submuestras y la inicialización de los modelos. Las pruebas "
              "miden por tanto la **variabilidad debida a la semilla**, no la "
              "variabilidad de muestreo del dataset: «significativo» quiere decir «no es "
              "un accidente de la semilla», no «se generaliza a otro tráfico».".format(nsem + 1),
              "- **Potencia baja.** Con {} pares solo se detectan diferencias muy "
              "consistentes; un resultado no significativo no demuestra que dos opciones "
              "sean iguales.".format(nsem),
              "- **Pares 54 vs 122.** Se emparejan por número de semilla: la misma "
              "semilla produce la misma partición de D1 en las dos variantes, pero los "
              "modelos ven columnas distintas, así que el emparejamiento es más débil "
              "que en las demás comparaciones.",
              "- **La semilla {} viene de otro commit** que las diez del barrido (ver "
              "cabecera). Su distancia a la banda puede incluir cambios de código, no "
              "solo de semilla.".format(self.s42),
              "- **El balanceo ganador no es el mismo en todas las semillas** en "
              "RandomForest (SMOTE o class_weight según los folds): su media de F1 "
              "macro promedia modelos con balanceos distintos. Está declarado en "
              "`dispersion_semillas.csv` (columna `decisiones_no_constantes`).",
              ""]
        ruta = os.path.join(config.RESULTADOS_DIR, NOMBRE_MD)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print("   MD:  {}".format(ruta))

    def ejecutar(self):
        config.setup_utf8()
        self.commit = config.commit_actual()
        self.fecha = datetime.now().isoformat(timespec="seconds")
        print("=" * 70)
        print("ESTABILIDAD FRENTE A LA SEMILLA (cero fit; solo lectura de CSV)")
        print("=" * 70)
        self._dispersion()
        self._pareadas()
        self._comprobar_salidas()
        self._escribir_csv()
        self._md()
        for f in self.filas_wil:
            print("   {:<70} {}/{}  p={:.4f}  holm={:.4f}  r={:+.2f}".format(
                f["metrica"], f["gana_a"], f["n"], f["p"], f["p_holm"],
                f["r_rango_biserial"]))
        print("=" * 70)


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Resumen de estabilidad frente a la semilla: dispersión (n, media, "
                    "sd, CV, mín, máx), representatividad de la semilla 42 y pruebas "
                    "de Wilcoxon pareadas por semilla con corrección de Holm. No "
                    "entrena nada y no escribe en ninguna tabla publicada.").parse_args()
    EstabilidadSemillas().ejecutar()
