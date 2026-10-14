# -*- coding: utf-8 -*-
"""
representacion.py — Representación de los datos y clasificadores de distancia en
la etapa 2.

Responde a una pregunta: si los árboles funcionan mejor porque las variables son
de distinta naturaleza, ¿se podría llevar los datos a una representación en la
que otro tipo de clasificador mejorara los resultados? Y a una segunda:
¿una señal de distancia («lejos de todo lo conocido») enruta mejor los 0-day a
'unknown' que la confianza del RandomForest?

El protocolo se fijó ANTES de ejecutar este script. Si el documento que lo
describe está en `docs/protocolo_representacion.md`, se estampan su SHA-256 y su
mtime en cada fila (igual que en aprendizaje_continuo.py); si no está, la huella
vale «no disponible».

Diseño
------
  * Etapa 1 FIJA: el Autoencoder publicado (54 características, semilla 42), leído
    de su .joblib con su umbral. No se reentrena. El recall 0-day tiene que salir
    idéntico al publicado en todas las variantes (control).
  * Etapa 2: representación × clasificador.
      R0  la actual (MinMax publicado)            R1  log1p en contadores de cola
      R2  QuantileTransformer (normal)                pesada + MinMax
      R3  R1 + `service` por frecuencia
    Clasificadores: KNN, MLP, RandomForest, HistGradientBoosting. Las rejillas,
    el balanceo intra-fold y los constructores salen de firmas.py (subclase); el
    MLP se añade con su rejilla.
  * Todo lo que aprende la representación va DENTRO del pipeline (antes de SMOTE):
    se ajusta en el pliegue de entrenamiento y, al final, sobre D3 completo.
  * Rechazo a 'unknown': conf (UMBRAL_CONF con la regla τ de hibrido.py,
    reutilizada), compuerta de distancia al vecino real más cercano de D3 con
    umbral por la misma regla τ sobre distancias OOF, su combinación y la
    sensibilidad p95.
  * D2 SOLO se lee para reportar. Nada se elige con él.

Salidas (ninguna pisa nada publicado)
-------------------------------------
  Resultados/representacion_trabajo/parciales/   un CSV por (semilla, rep, clf)
  Resultados/representacion_trabajo/logs/        logs de las corridas
  Resultados/metricas_representacion.csv         (--agregar) una fila por
      (semilla, representación, clasificador, regla de rechazo)
  Resultados/representacion_informe.txt          (--agregar) resumen media ± std

Uso
---
  python app\\representacion.py --control               # R0+RF semilla 42 y compara
  python app\\representacion.py --semilla 42            # 4 rep × 4 clf
  python app\\representacion.py --semillas 1 2 3 ...    # barrido
  python app\\representacion.py --agregar               # tabla + informe
"""

import argparse
import glob
import hashlib
import os
import sys
import time
import warnings
from datetime import datetime

# Los workers de joblib heredan el entorno: así el MLP no llena los logs de
# ConvergenceWarning. Debe ir antes de importar sklearn.
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from imblearn.pipeline import Pipeline as ImbPipeline

import config
import evaluacion
from firmas import NSLKDDSignatureTrainer
from hibrido import NSLKDDHybridEvaluator


REPRESENTACIONES = ["R0", "R1", "R2", "R3"]
CLASIFICADORES = ["KNN", "MLP", "RandomForest", "HistGradientBoosting"]
REGLAS = ["conf", "conf+dist_tau", "dist_tau", "conf+dist_p95"]
SEMILLAS = [config.SEMILLA_POR_DEFECTO] + list(config.SEMILLAS_BARRIDO)

# Criterio de «cola pesada», evaluado en el fit sobre el entrenamiento.
UMBRAL_ASIMETRIA = 1.0
N_CUANTILES = 1000
PREFIJO_SERVICE = "service_"
COL_SERVICE = "service_code"

# Compuerta de distancia: candidatos y regla τ (la misma de hibrido.py).
PERCENTILES_DIST = [90.0, 95.0, 99.0, 99.5, 99.9]
PERCENTIL_SENSIBILIDAD = 95.0
TAU = NSLKDDHybridEvaluator.TAU

DETECTOR = "Autoencoder"
SET_FEATURES = "54"

DIR_TRABAJO = os.path.join(config.RESULTADOS_DIR, "representacion_trabajo")
DIR_PARCIALES = os.path.join(DIR_TRABAJO, "parciales")
CSV_METRICAS = os.path.join(config.RESULTADOS_DIR, "metricas_representacion.csv")
INFORME = os.path.join(config.RESULTADOS_DIR, "representacion_informe.txt")
PREREGISTRO = os.path.join(config.REPO_DIR, "docs", "protocolo_representacion.md")
CLAVE = ("semilla", "representacion", "algoritmo", "rechazo")

# Cifras publicadas que el control tiene que reproducir con R0 + RF + conf
# y la semilla 42. Se leen de las tablas publicadas, no se escriben a mano.
TOL_CONTROL = 1e-6

ALCANCE = (
    "Experimento de representación: etapa 2 con representación y clasificador variables, etapa 1 = "
    "Autoencoder publicado (54, semilla 42). Prefijo e2_ = etapa 2 SOLA sobre los "
    "ataques de tipo conocido de D2 (argmax, sin 'unknown'; no depende de la regla); "
    "conocida_ = cascada extremo a extremo sobre esos mismos ataques ('unknown' y "
    "'normal' cuentan como error); d2_ = cascada sobre D2 con la regla de rechazo "
    "de la fila; recall_0day = etapa 1 (control, constante); _cv y oof_ = D3 "
    "(selección y calibración). D2 solo se lee para reportar")


def huella_preregistro():
    """SHA-256 y mtime del documento de protocolo, o «no disponible» si no está."""
    if not os.path.exists(PREREGISTRO):
        return "no disponible", "no disponible"
    with open(PREREGISTRO, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    mtime = datetime.fromtimestamp(os.path.getmtime(PREREGISTRO))
    return sha, mtime.isoformat(timespec="seconds")


# ======================================================================
# Representaciones R1-R3 (transformador sklearn: se ajusta en cada pliegue)
# ======================================================================
class TransformadorRepresentacion(BaseEstimator, TransformerMixin):
    """
    Entrada: las 54 columnas en magnitudes ORIGINALES (MinMax publicado invertido)
    más `service_code` (entero; -1 si el servicio no aparece en D3). Salida: la
    representación R1, R2 o R3. Todo lo que aprende (columnas de cola
    pesada, MinMax, cuantiles, frecuencias) lo aprende en fit().
    """

    def __init__(self, tipo="R1", columnas=None, random_state=None):
        self.tipo = tipo
        self.columnas = columnas
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        cols = list(self.columnas)
        self.i_srv_ = cols.index(COL_SERVICE)
        self.idx_num_ = [i for i, c in enumerate(cols) if c != COL_SERVICE]
        nombres = [cols[i] for i in self.idx_num_]
        Xn = X[:, self.idx_num_]
        self.continuas_ = [j for j in range(Xn.shape[1]) if len(np.unique(Xn[:, j])) > 2]
        self.log_ = []
        if self.tipo in ("R1", "R3"):
            for j in self.continuas_:
                v = Xn[:, j]
                if v.min() >= 0 and v.max() > 1 and skew(v) > UMBRAL_ASIMETRIA:
                    self.log_.append(j)
        self.columnas_log_ = [nombres[j] for j in self.log_]
        if self.tipo == "R3":
            self.service_onehot_ = [j for j, c in enumerate(nombres)
                                    if c.startswith(PREFIJO_SERVICE)]
            codigos = X[:, self.i_srv_].astype(int)
            vals, cnt = np.unique(codigos, return_counts=True)
            self.freq_ = dict(zip(vals.tolist(), (cnt / float(len(codigos))).tolist()))
        if self.tipo in ("R1", "R3"):
            self.scaler_ = MinMaxScaler().fit(self._pre(X))
        elif self.tipo == "R2":
            self.qt_ = QuantileTransformer(
                n_quantiles=min(N_CUANTILES, len(X)), output_distribution="normal",
                random_state=self.random_state).fit(Xn[:, self.continuas_])
        else:
            raise ValueError("Representación desconocida: " + str(self.tipo))
        return self

    def _pre(self, X):
        Xn = X[:, self.idx_num_].copy()
        if self.log_:
            Xn[:, self.log_] = np.log1p(np.clip(Xn[:, self.log_], 0, None))
        if self.tipo == "R3":
            mantener = [j for j in range(Xn.shape[1]) if j not in self.service_onehot_]
            codigos = X[:, self.i_srv_].astype(int)
            f = np.array([self.freq_.get(c, 0.0) for c in codigos.tolist()])
            Xn = np.column_stack([Xn[:, mantener], f])
        return Xn

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.tipo in ("R1", "R3"):
            return self.scaler_.transform(self._pre(X))
        Xn = X[:, self.idx_num_].copy()
        Xn[:, self.continuas_] = self.qt_.transform(Xn[:, self.continuas_])
        return Xn


class _FirmasConRepresentacion(NSLKDDSignatureTrainer):
    """firmas.py con el MLP añadido y la representación delante del pipeline.
    Reutiliza rejillas, balanceo, SMOTE intra-fold y constructores; con R0 el
    estimador es EXACTAMENTE el de firmas.py (de ahí sale el control)."""

    GRIDS = dict(NSLKDDSignatureTrainer.GRIDS, MLP={
        "hidden_layer_sizes": [(64,), (128, 64)],
        "alpha": [1e-4, 1e-3],
    })
    BALANCEO_OPCIONES = dict(NSLKDDSignatureTrainer.BALANCEO_OPCIONES,
                             MLP=["SMOTE", "nada"])

    def __init__(self, representacion, columnas, algoritmo):
        super().__init__(sin_seleccion=False)
        self.representacion = representacion
        self.columnas = list(columnas)
        self.ALGORITMOS = [algoritmo]

    def _estimador_base(self, algo, class_weight=None):
        if algo == "MLP":
            # Sin early_stopping: en sklearn 1.7.1 falla con etiquetas de texto.
            # Para por tol sobre la pérdida de entrenamiento.
            return MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                                 random_state=config.RANDOM_STATE)
        return super()._estimador_base(algo, class_weight=class_weight)

    def nueva_representacion(self):
        return TransformadorRepresentacion(tipo=self.representacion,
                                           columnas=self.columnas,
                                           random_state=config.RANDOM_STATE)

    def _estimador_para(self, algo, balanceo):
        est = super()._estimador_para(algo, balanceo)
        if self.representacion == "R0":
            return est
        if isinstance(est, ImbPipeline):
            return ImbPipeline([("rep", self.nueva_representacion())] + list(est.steps))
        return ImbPipeline([("rep", self.nueva_representacion()), ("clf", est)])

    def _param_grid(self, algo, balanceo):
        grid = super()._param_grid(algo, balanceo)
        if self.representacion == "R0" or balanceo == "SMOTE":
            return grid
        return {"clf__" + k: v for k, v in grid.items()}


# ======================================================================
# Evaluador
# ======================================================================
class NSLKDDRepresentationEvaluator:
    """Una semilla: carga D2/D3 y la etapa 1 publicada, y evalúa rep × clf."""

    def __init__(self):
        self.semilla = config.RANDOM_STATE
        # Instancia de hibrido.py SOLO para reutilizar carga, máscaras de D2,
        # _metricas_d2 y la regla de calibración. Se crea tras fijar la semilla.
        self.hd = NSLKDDHybridEvaluator()
        self._compuertas = {}

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def cargar(self):
        hd = self.hd
        hd.cargar_datos()
        assert hd.n_features == 54, "se esperaba el conjunto de 54 características"
        self.y_D3 = hd.y_D3
        self.cols = list(hd.X_D3.columns)

        # Magnitudes originales: el MinMax publicado es afín y se invierte exacto.
        trans = joblib.load(config.base_path() + "_transformers.joblib")
        sc, pre = trans["scaler"], list(trans["feature_columns_pre_seleccion"])
        idx = [pre.index(c) for c in self.cols]
        rango, minimo = sc.data_range_[idx], sc.data_min_[idx]

        # service crudo de los CSV 'original_*', alineados fila a fila (se comprueba).
        orig_D3 = pd.read_csv(config.base_path() +
                              "_original_D3_known_attacks_for_signatures.csv")
        orig_D2 = pd.read_csv(config.base_path() + "_original_D2_complete_test.csv")
        servicios = sorted(orig_D3["service"].astype(str).unique())
        codigo = {s: i for i, s in enumerate(servicios)}
        for X, orig in ((hd.X_D3, orig_D3), (hd.X_D2, orig_D2)):
            assert len(X) == len(orig)
            for c in self.cols:
                if c.startswith(PREFIJO_SERVICE):
                    s = c[len(PREFIJO_SERVICE):]
                    assert ((X[c].values > 0.5) == (orig["service"].values == s)).all(), \
                        "CSV original desalineado en " + c

        def crudo(X, orig):
            R = pd.DataFrame(X.values * rango + minimo, columns=self.cols, index=X.index)
            R[COL_SERVICE] = orig["service"].astype(str).map(codigo).fillna(-1).astype(int).values
            return R

        self.cols_crudo = self.cols + [COL_SERVICE]
        self.Xc_D3 = crudo(hd.X_D3, orig_D3)
        self.Xc_D2 = crudo(hd.X_D2, orig_D2)
        # Comprobación de la inversión: src_bytes original frente a la invertida.
        assert np.allclose(self.Xc_D3["src_bytes"].values, orig_D3["src_bytes"].values,
                           rtol=1e-6, atol=1e-3)

        # Etapa 1 publicada (solo lectura). No se usa hibrido._cargar_joblib porque
        # aborta si la semilla de la corrida no es la del joblib, y aquí la etapa 1
        # es la de la 42 en TODAS las semillas.
        ruta = os.path.join(config.MODELOS_DIR, "anomalia_{}_54.joblib".format(DETECTOR))
        det = joblib.load(ruta)
        assert str(det["set_features"]) == SET_FEATURES and int(det["semilla"]) == 42
        hd.score_cascada = hd._anom._score(DETECTOR, det["modelo"], hd.X_D2)
        hd.es_sospechoso = hd.score_cascada > det["umbral"]
        self.umbral_det = float(det["umbral"])

        self.mask_conocida = hd.mask_ataque & hd.mask_tipo_conocido
        self.mask_0day = hd.mask_ataque & ~hd.mask_tipo_conocido
        print("   Etapa 1 publicada: {} alarmas en D2 · {} conocidas con alarma · "
              "{} 0-day detectados".format(int(hd.es_sospechoso.sum()),
                                           int((self.mask_conocida & hd.es_sospechoso).sum()),
                                           int((self.mask_0day & hd.es_sospechoso).sum())))

    def _entradas(self, rep):
        if rep == "R0":
            return self.hd.X_D3, self.hd.X_D2, self.cols
        return self.Xc_D3, self.Xc_D2, self.cols_crudo

    # ------------------------------------------------------------------
    # Compuerta de distancia (depende solo de la representación)
    # ------------------------------------------------------------------
    def compuerta(self, rep, firm):
        if rep in self._compuertas:
            return self._compuertas[rep]
        t0 = time.perf_counter()
        X3, X2, _cols = self._entradas(rep)
        X3v, y = np.asarray(X3.values, dtype=float), self.y_D3

        def ajustar(Xtr):
            if rep == "R0":
                return None, Xtr
            r = firm.nueva_representacion().fit(Xtr)
            return r, r.transform(Xtr)

        d_oof = np.empty(len(y))
        for tr, va in firm.cv.split(X3v, y):
            r, Ztr = ajustar(X3v[tr])
            Zva = X3v[va] if r is None else r.transform(X3v[va])
            nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Ztr)
            d_oof[va] = nn.kneighbors(Zva)[0][:, 0]

        tabla = []
        elegido = None
        for p in PERCENTILES_DIST:
            u = float(np.percentile(d_oof, p))
            rech = d_oof > u
            macro = float(np.mean([rech[y == c].mean() for c in config.CATEGORIAS_ATAQUE]))
            tabla.append((p, u, macro, float(rech.mean())))
            if elegido is None and macro <= TAU:
                elegido = (p, u, macro, float(rech.mean()))
        reserva = elegido is None
        if reserva:
            elegido = tabla[-1]
        p95 = [t for t in tabla if t[0] == PERCENTIL_SENSIBILIDAD][0]

        # Final: representación ajustada sobre D3 completo y distancias de D2.
        r, Z3 = ajustar(X3v)
        X2v = np.asarray(X2.values, dtype=float)
        Z2 = X2v if r is None else r.transform(X2v)
        nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(Z3)
        d2 = np.full(len(X2v), np.nan)
        s = self.hd.es_sospechoso
        d2[s] = nn.kneighbors(Z2[s])[0][:, 0]
        res = {
            "d2": d2, "tabla": tabla, "reserva": reserva,
            "percentil_tau": elegido[0], "umbral_tau": elegido[1],
            "oof_rechazo_macro_tau": elegido[2], "oof_rechazo_global_tau": elegido[3],
            "umbral_p95": p95[1], "oof_rechazo_macro_p95": p95[2],
            "oof_rechazo_global_p95": p95[3],
            "columnas_log": "|".join(getattr(r, "columnas_log_", []) or []),
            "n_features_rep": int(Z3.shape[1]),
            "tiempo_s": time.perf_counter() - t0,
        }
        print("   Compuerta {}: percentil τ = {}{} (umbral {:.4g}, rechazo OOF macro "
              "{:.4f}) · p95 umbral {:.4g} · {:.1f}s".format(
                  rep, elegido[0], " [RESERVA: ninguno cabe en τ]" if reserva else "",
                  elegido[1], elegido[2], p95[1], res["tiempo_s"]))
        self._compuertas[rep] = res
        return res

    # ------------------------------------------------------------------
    # Una combinación representación × clasificador
    # ------------------------------------------------------------------
    def evaluar(self, rep, algo, sha, mtime):
        hd = self.hd
        print("\n" + "-" * 70)
        print("SEMILLA {} · {} · {}".format(self.semilla, rep, algo))
        print("-" * 70)
        X3, X2, cols = self._entradas(rep)
        firm = _FirmasConRepresentacion(rep, cols, algo)
        firm.X_D3, firm.y_D3 = X3, self.y_D3

        # 1. Balanceo + rejilla (firmas.py), solo D3.
        t0 = time.perf_counter()
        firm._experimento_balanceo()
        balanceo = firm.balanceo_ganador[algo]
        gs = GridSearchCV(firm._estimador_para(algo, balanceo),
                          firm._param_grid(algo, balanceo),
                          scoring="f1_macro", cv=firm.cv, n_jobs=-1, refit=True)
        gs.fit(X3, self.y_D3)
        t_ent = time.perf_counter() - t0
        modelo = gs.best_estimator_
        print("   Balanceo {} · config {} · f1_macro(CV)={:.4f} · {:.1f}s".format(
            balanceo, gs.best_params_, gs.best_score_, t_ent))

        # 2. Etapa 2 sola sobre los ataques de tipo conocido de D2 (como firmas.py).
        Xk = X2.loc[self.mask_conocida]
        t1 = time.perf_counter()
        pred_k = modelo.predict(Xk)
        t_inf = time.perf_counter() - t1
        e2 = evaluacion.evaluar_multiclase(hd.cat_D2[self.mask_conocida], pred_k,
                                           labels=config.CATEGORIAS_ATAQUE)

        # 3. UMBRAL_CONF con la regla τ de hibrido.py (reutilizada; solo D3).
        t2 = time.perf_counter()
        hd._sig = firm
        hd.X_D3, hd.y_D3 = X3, self.y_D3
        umbral_conf, oof = hd._calibrar_umbral_conf(
            {"algoritmo": algo, "balanceo": balanceo, "config_ganadora": gs.best_params_})
        t_cal = time.perf_counter() - t2

        # 4. Compuerta de distancia de la representación.
        cmp_ = self.compuerta(rep, firm)

        # 5. Cascada: etapa 2 sobre los flujos con alarma.
        s = hd.es_sospechoso
        t3 = time.perf_counter()
        proba = modelo.predict_proba(X2.loc[s])
        t_casc = time.perf_counter() - t3
        clases = np.asarray(modelo.classes_)
        conf = proba.max(axis=1)
        cat = clases[proba.argmax(axis=1)].astype(object)
        dist = cmp_["d2"][s]

        # Calidad de las señales sin umbral (descriptiva): 0-day detectados (1)
        # frente a conocidas con alarma (0).
        sub0 = self.mask_0day[s]
        subk = self.mask_conocida[s]
        sel = sub0 | subk
        auc_conf = float(roc_auc_score(sub0[sel], (1 - conf)[sel]))
        auc_dist = float(roc_auc_score(sub0[sel], dist[sel]))

        reglas = {
            "conf": conf < umbral_conf,
            "conf+dist_tau": (conf < umbral_conf) | (dist > cmp_["umbral_tau"]),
            "dist_tau": dist > cmp_["umbral_tau"],
            "conf+dist_p95": (conf < umbral_conf) | (dist > cmp_["umbral_p95"]),
        }
        filas = []
        for regla in REGLAS:
            cat_r = cat.copy()
            cat_r[reglas[regla]] = config.ETIQUETA_UNKNOWN
            pred = np.full(len(hd.X_D2), config.ETIQUETA_NORMAL, dtype=object)
            pred[s] = cat_r
            m = hd._metricas_d2(pred)
            filas.append(self._fila(rep, algo, regla, pred, m, e2, gs, balanceo,
                                    umbral_conf, oof, cmp_, auc_conf, auc_dist,
                                    t_ent, t_inf, t_cal, t_casc, len(Xk), sha, mtime))
        self._controles(rep, algo, filas, modelo, Xk)
        return filas

    def _fila(self, rep, algo, regla, pred, m, e2, gs, balanceo, umbral_conf, oof, cmp_,
              auc_conf, auc_dist, t_ent, t_inf, t_cal, t_casc, n_k, sha, mtime):
        hd = self.hd
        s = hd.es_sospechoso
        unk = pred == config.ETIQUETA_UNKNOWN
        k_al = self.mask_conocida & s
        n_al = (hd.cat_D2 == config.ETIQUETA_NORMAL) & s
        c, r0 = m["conocida"], m["recall_0day"]["__global__"]
        oof_u = oof[umbral_conf]
        fila = {
            "algoritmo": algo, "representacion": rep, "rechazo": regla,
            "alcance": ALCANCE, "set_features": SET_FEATURES, "sin_seleccion": False,
            "n_features": cmp_["n_features_rep"],
            "columnas_log": cmp_["columnas_log"],
            # Selección y calibración (D3)
            "balanceo": balanceo, "config_ganadora": str(gs.best_params_),
            "f1_macro_cv": round(float(gs.best_score_), 6),
            "umbral_conf": umbral_conf,
            "oof_recall_macro_baseline": round(oof_u["recall_macro_baseline_oof"], 6),
            "oof_recall_macro_conocidas": round(oof_u["recall_macro_oof"], 6),
            "oof_tasa_falsos_unknown_conf": round(oof_u["tasa_falsos_unknown_oof"], 6),
            "percentil_dist_tau": cmp_["percentil_tau"],
            "reserva_dist_tau": bool(cmp_["reserva"]),
            "umbral_dist_tau": cmp_["umbral_tau"],
            "oof_rechazo_dist_macro_tau": round(cmp_["oof_rechazo_macro_tau"], 6),
            "oof_rechazo_dist_global_tau": round(cmp_["oof_rechazo_global_tau"], 6),
            "umbral_dist_p95": cmp_["umbral_p95"],
            "oof_rechazo_dist_macro_p95": round(cmp_["oof_rechazo_macro_p95"], 6),
            # Etapa 2 sola sobre los ataques de tipo conocido de D2
            "e2_f1_macro": round(e2["f1_macro"], 6),
            "e2_recall_macro": round(e2["recall_macro"], 6),
            "e2_accuracy": round(e2["accuracy"], 6),
        }
        for cl in config.CATEGORIAS_ATAQUE:
            pc = e2["por_clase"].get(cl, {})
            fila["e2_recall_" + cl] = round(pc.get("recall", np.nan), 6)
            fila["e2_f1_" + cl] = round(pc.get("f1", np.nan), 6)
        fila.update({
            "conocida_f1_macro": round(c["f1_macro"], 6),
            "conocida_recall_macro": round(c["recall_macro"], 6),
        })
        for cl in config.CATEGORIAS_ATAQUE:
            pc = c["por_clase"].get(cl, {})
            fila["conocida_recall_" + cl] = round(pc.get("recall", np.nan), 6)
            fila["conocida_f1_" + cl] = round(pc.get("f1", np.nan), 6)
        fila.update({
            "d2_pct_unknown_0day_det": round(100 * m["pct_unknown_entre_0day_detectados"], 4),
            "d2_n_0day_det": int((self.mask_0day & s).sum()),
            "d2_pct_unknown_conocidas_alarma": round(100 * float(unk[k_al].mean()), 4),
            "d2_n_conocidas_alarma": int(k_al.sum()),
        })
        for cl in config.CATEGORIAS_ATAQUE:
            mk = k_al & (hd.cat_D2 == cl)
            fila["d2_pct_unknown_conocidas_alarma_" + cl] = (
                round(100 * float(unk[mk].mean()), 4) if mk.any() else np.nan)
        fila.update({
            "d2_pct_unknown_normales_alarma": round(100 * float(unk[n_al].mean()), 4),
            "d2_f1_macro_5_conservador": round(m["conservador"]["f1_macro"], 6),
            "d2_auc_rechazo_conf": round(auc_conf, 6),
            "d2_auc_rechazo_dist": round(auc_dist, 6),
            "recall_0day_global": round(r0["recall"], 6),
            "n_0day": r0["n"],
            "tiempo_calibracion_s": round(t_cal, 3),
            "tiempo_compuerta_s": round(cmp_["tiempo_s"], 3),
            "tiempo_etapa2_cascada_s": round(t_casc, 4),
            "alcance_tiempo_s": (
                "tiempo_entrenamiento_s = mini-experimento de balanceo + GridSearchCV + "
                "reajuste sobre D3; tiempo_inferencia_s = predict de la etapa 2 sobre "
                "los ataques de tipo conocido de D2; tiempo de pared en máquina "
                "compartida, solo orden de magnitud"),
        })
        fila.update(evaluacion.metricas_tiempo(t_ent, t_inf, n_k))
        fila.update({"semilla": self.semilla, "commit": config.commit_actual(),
                     "fecha": datetime.now().isoformat(timespec="seconds"),
                     "preregistro_sha256": sha, "preregistro_mtime": mtime})
        evaluacion.validar_esquema_minimo(fila, CSV_METRICAS, auxiliar=True,
                                          columnas=evaluacion.COLUMNAS_MINIMAS_AUXILIARES)
        return fila

    # ------------------------------------------------------------------
    # Controles
    # ------------------------------------------------------------------
    def _controles(self, rep, algo, filas, modelo, Xk):
        pub_h = pd.read_csv(os.path.join(config.RESULTADOS_DIR, "metricas_hibrido.csv"))
        pub_h = pub_h[pub_h["set_features"].astype(str) == SET_FEATURES].iloc[0]
        for f in filas:
            assert abs(f["recall_0day_global"] - pub_h["recall_0day_global"]) < TOL_CONTROL, \
                "el recall 0-day cambió: la etapa 1 no es la publicada"
        if self.semilla != 42 or rep != "R0":
            return
        pub_f = pd.read_csv(os.path.join(config.RESULTADOS_DIR, "metricas_firmas.csv"))
        pub_f = pub_f[pub_f["set_features"].astype(str) == SET_FEATURES].set_index("algoritmo")
        f = [x for x in filas if x["rechazo"] == "conf"][0]
        if algo in pub_f.index:
            ok = abs(f["e2_f1_macro"] - pub_f.loc[algo, "f1_macro"]) < TOL_CONTROL
            print("   CONTROL R0·{}: e2_f1_macro {} frente a publicado {} → {}".format(
                algo, f["e2_f1_macro"], pub_f.loc[algo, "f1_macro"], "OK" if ok else "DIFIERE"))
            if algo != "RandomForest" and not ok:
                print("   [aviso] control secundario no reproducido")
        if algo != "RandomForest":
            return
        cal = pd.read_csv(os.path.join(config.RESULTADOS_DIR, "metricas_hibrido_calibracion.csv"))
        cal = cal[(cal["set_features"].astype(str) == SET_FEATURES) & cal["elegido"]].iloc[0]
        esperado = {
            "e2_f1_macro": pub_f.loc["RandomForest", "f1_macro"],
            "umbral_conf": pub_h["umbral_conf_elegido"],
            "conocida_f1_macro": pub_h["conocida_f1_macro"],
            "d2_pct_unknown_0day_det": 100 * cal["d2_pct_unknown_entre_0day_det"],
            "recall_0day_global": pub_h["recall_0day_global"],
        }
        fallos = {k: (f[k], v) for k, v in esperado.items()
                  if abs(float(f[k]) - float(v)) > (1e-4 if k.startswith("d2_pct") else TOL_CONTROL)}
        # Además: predicciones idénticas a las del joblib publicado (solo lectura).
        pub = joblib.load(os.path.join(config.MODELOS_DIR, "firma_RandomForest_54.joblib"))
        iguales = bool((pub["modelo"].predict(Xk) == modelo.predict(Xk)).all())
        print("   CONTROL PRINCIPAL R0·RF·conf: {} · predicciones idénticas al joblib "
              "publicado: {}".format("REPRODUCIDO" if not fallos else "FALLA " + str(fallos),
                                     iguales))
        if fallos or not iguales:
            raise RuntimeError("El control R0+RandomForest no reproduce lo publicado: "
                               "{} · predicciones iguales={}. Se para.".format(
                                   fallos, iguales))


def ruta_parcial(semilla, rep, algo):
    return os.path.join(DIR_PARCIALES, "semilla{}_{}_{}.csv".format(semilla, rep, algo))


def correr(semilla, reps, algos, huella, rehacer=False):
    config.fijar_semilla(semilla)
    config.setup_utf8()
    print("=" * 70)
    print("REPRESENTACIÓN · semilla {} · {} × {}".format(semilla, reps, algos))
    print("=" * 70)
    ev = NSLKDDRepresentationEvaluator()
    ev.cargar()
    os.makedirs(DIR_PARCIALES, exist_ok=True)
    for rep in reps:
        for algo in algos:
            ruta = ruta_parcial(semilla, rep, algo)
            if os.path.exists(ruta) and not rehacer:
                print("   (ya hecho) " + ruta)
                continue
            t0 = time.perf_counter()
            filas = ev.evaluar(rep, algo, *huella)
            pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")
            f = filas[0]
            print("   → {} · e2_f1={:.4f} · conocida_f1(conf)={:.4f} · 0day→unk(conf)={:.1f}% "
                  "· {:.0f}s".format(os.path.basename(ruta), f["e2_f1_macro"],
                                    f["conocida_f1_macro"], f["d2_pct_unknown_0day_det"],
                                    time.perf_counter() - t0))


# ======================================================================
# Agregación: tabla única + informe
# ======================================================================
def _ms(serie, fmt="{:.4f}"):
    v = serie.dropna().astype(float)
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return fmt.format(v.iloc[0])
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1))


def agregar():
    partes = sorted(glob.glob(os.path.join(DIR_PARCIALES, "*.csv")))
    if not partes:
        raise RuntimeError("No hay parciales en " + DIR_PARCIALES)
    df = pd.concat([pd.read_csv(p) for p in partes], ignore_index=True)
    df.to_csv(CSV_METRICAS, index=False, encoding="utf-8")
    evaluacion.comprobar_unicidad(CSV_METRICAS, clave=CLAVE)
    shas = df["preregistro_sha256"].unique()
    assert len(shas) == 1, "El preregistro cambió entre corridas: huellas {}".format(list(shas))
    cuenta = df.groupby("semilla").size()
    for s, n in cuenta.items():
        assert n % len(REGLAS) == 0
    print("   → {} ({} filas; semillas {})".format(CSV_METRICAS, len(df),
                                                   dict(cuenta)))
    escribir_informe(df, shas[0])


def escribir_informe(df, sha):
    L = []
    w = L.append
    semillas = sorted(df["semilla"].unique())
    w("INFORME DEL EXPERIMENTO DE REPRESENTACIÓN — Representación y clasificadores de distancia (etapa 2)")
    w("=" * 78)
    w("Generado: {} · commit {} · preregistro sha256 {}".format(
        datetime.now().isoformat(timespec="seconds"), config.commit_actual(), sha))
    w("Semillas: {} (n={}). Cifras: media ± desviación típica entre semillas "
      "(una sola cifra si n=1).".format(semillas, len(semillas)))
    w("Etapa 1 fija: Autoencoder publicado (54, semilla 42). D2 solo se reporta.")
    w("")
    conf = df[df["rechazo"] == "conf"]
    completas = conf.groupby(["representacion", "algoritmo"])["semilla"].nunique()
    w("Semillas por combinación: " + ", ".join(
        "{}·{}={}".format(r, a, n) for (r, a), n in completas.items()))
    w("")

    w("1. CONTROL")
    w("-" * 78)
    c = conf[(conf["semilla"] == 42) & (conf["representacion"] == "R0")]
    for _, f in c.iterrows():
        w("   R0·{:<22} e2_f1_macro={:.6f} conocida_f1={:.6f} 0day→unk={:.4f}% "
          "recall_0day={:.6f} umbral_conf={}".format(
              f["algoritmo"], f["e2_f1_macro"], f["conocida_f1_macro"],
              f["d2_pct_unknown_0day_det"], f["recall_0day_global"], f["umbral_conf"]))
    w("   recall 0-day en TODAS las filas: {}".format(sorted(df["recall_0day_global"].unique())))
    w("")

    def tabla(titulo, sub, cols, fmts):
        w(titulo)
        w("-" * 78)
        w("   {:<4} {:<22} ".format("rep", "clasificador") +
          " ".join("{:>22}".format(c[:22]) for c in cols))
        for (r, a), g in sub.groupby(["representacion", "algoritmo"]):
            w("   {:<4} {:<22} ".format(r, a) +
              " ".join("{:>22}".format(_ms(g[c], f)) for c, f in zip(cols, fmts)))
        w("")

    tabla("2. ETAPA 2 SOLA (ataques de tipo conocido de D2) y CV sobre D3", conf,
          ["e2_f1_macro", "e2_recall_r2l", "e2_recall_u2r", "f1_macro_cv"],
          ["{:.4f}"] * 4)
    tabla("3. CASCADA con la regla conf (la publicada)", conf,
          ["conocida_f1_macro", "d2_pct_unknown_0day_det",
           "d2_pct_unknown_conocidas_alarma", "umbral_conf"],
          ["{:.4f}", "{:.2f}", "{:.2f}", "{:.2f}"])
    for regla in REGLAS[1:]:
        tabla("4. CASCADA con la regla " + regla, df[df["rechazo"] == regla],
              ["conocida_f1_macro", "d2_pct_unknown_0day_det",
               "d2_pct_unknown_conocidas_alarma", "d2_pct_unknown_conocidas_alarma_r2l"],
              ["{:.4f}", "{:.2f}", "{:.2f}", "{:.2f}"])
    tabla("5. SEÑAL DE RECHAZO SIN UMBRAL (AUC 0-day detectados vs conocidas con alarma) "
          "y compuerta", conf,
          ["d2_auc_rechazo_conf", "d2_auc_rechazo_dist", "percentil_dist_tau",
           "oof_rechazo_dist_macro_tau"], ["{:.4f}", "{:.4f}", "{:.1f}", "{:.4f}"])
    tabla("6. COSTE (s; latencia en ms por flujo, solo predict)", conf,
          ["tiempo_entrenamiento_s", "latencia_ms_por_flujo", "tiempo_calibracion_s",
           "tiempo_compuerta_s"], ["{:.1f}", "{:.4f}", "{:.1f}", "{:.1f}"])

    w("7. DIFERENCIAS EMPAREJADAS POR SEMILLA (e2_f1_macro; media, std y nº de "
      "semillas con signo positivo)")
    w("-" * 78)
    piv = conf.pivot_table(index="semilla", columns=["representacion", "algoritmo"],
                           values="e2_f1_macro")

    def dif(a, b, etiqueta):
        if a not in piv.columns or b not in piv.columns:
            return
        d = (piv[a] - piv[b]).dropna()
        if len(d) == 0:
            return
        w("   {:<44} Δ={:+.4f} ± {:.4f} · positivas {}/{}".format(
            etiqueta, d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0,
            int((d > 0).sum()), len(d)))

    for a in CLASIFICADORES:
        for r in ["R1", "R2", "R3"]:
            dif((r, a), ("R0", a), "{}·{} − R0·{}".format(r, a, a))
    for a in ["KNN", "MLP"]:
        for r in REPRESENTACIONES:
            dif((r, a), ("R0", "RandomForest"), "{}·{} − R0·RF".format(r, a))
    w("")
    w("8. COMPUERTA DE DISTANCIA: Δ (conf+dist_tau − conf) en pp, emparejado por semilla")
    w("-" * 78)
    for (r, a), g in df.groupby(["representacion", "algoritmo"]):
        p = g.pivot_table(index="semilla", columns="rechazo",
                          values=["d2_pct_unknown_0day_det", "d2_pct_unknown_conocidas_alarma"])
        d0 = p["d2_pct_unknown_0day_det"]["conf+dist_tau"] - p["d2_pct_unknown_0day_det"]["conf"]
        dk = (p["d2_pct_unknown_conocidas_alarma"]["conf+dist_tau"] -
              p["d2_pct_unknown_conocidas_alarma"]["conf"])
        w("   {}·{:<22} 0-day→unk {:+.2f} pp · conocidas→unk {:+.2f} pp · cumple "
          "(≥10 y ≤5): {}".format(r, a, d0.mean(), dk.mean(),
                                  bool(d0.mean() >= 10 and dk.mean() <= 5)))
    w("")
    w("9. COMBINACIÓN QUE ELEGIRÍA LA CV SOBRE D3 (mayor f1_macro_cv medio)")
    w("-" * 78)
    g = conf.groupby(["representacion", "algoritmo"])
    cv = g["f1_macro_cv"].mean().sort_values(ascending=False)
    for (r, a), v in cv.head(5).items():
        w("   {}·{:<22} f1_macro_cv={:.4f} → e2_f1_macro D2 = {}".format(
            r, a, v, _ms(g.get_group((r, a))["e2_f1_macro"])))
    w("")
    w("10. COLUMNAS CON log1p (R1/R3, decididas en el fit sobre D3; semilla 42)")
    w("-" * 78)
    for r in ["R1", "R3"]:
        v = df[(df["representacion"] == r)]["columnas_log"].dropna().unique()
        w("   {}: {}".format(r, list(v)))
    with open(INFORME, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("   → " + INFORME)
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description="Representación y clasificadores de "
                                "distancia en la etapa 2.")
    p.add_argument("--semilla", type=int, default=None)
    p.add_argument("--semillas", type=int, nargs="+", default=None)
    p.add_argument("--representaciones", nargs="+", default=REPRESENTACIONES,
                   choices=REPRESENTACIONES)
    p.add_argument("--clasificadores", nargs="+", default=CLASIFICADORES,
                   choices=CLASIFICADORES)
    p.add_argument("--control", action="store_true",
                   help="Solo R0 + RandomForest con la semilla 42 (control).")
    p.add_argument("--rehacer", action="store_true", help="Recalcula parciales existentes.")
    p.add_argument("--agregar", action="store_true", help="Tabla única + informe.")
    a = p.parse_args()
    if a.agregar:
        config.setup_utf8()
        agregar()
        return 0
    huella = huella_preregistro()
    print("Protocolo · sha256 {} · modificado {}".format(*huella))
    if a.control:
        correr(42, ["R0"], ["RandomForest"], huella, rehacer=a.rehacer)
        return 0
    semillas = a.semillas or ([a.semilla] if a.semilla is not None else [42])
    for s in semillas:
        correr(s, a.representaciones, a.clasificadores, huella, rehacer=a.rehacer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
