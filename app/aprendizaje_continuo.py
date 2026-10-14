# -*- coding: utf-8 -*-
"""
aprendizaje_continuo.py — Experimento de aprendizaje continuo de la etapa 2.

Responde a UNA pregunta: **cuando aparece un tipo de ataque que el sistema no
conocía y un analista etiqueta unos pocos flujos, ¿cuánto mejora el sistema al
actualizarse, cuánto olvida de lo que ya sabía y cuánto cuesta?**

El protocolo se fijó ANTES de ejecutar este script. Si el documento que lo
describe está en `docs/protocolo_aprendizaje_continuo.md`, cada corrida estampa
en sus filas su SHA-256 y su fecha de modificación al arrancar; si no está, la
huella vale «no disponible».

Diseño
------
  * Tres VERSIONES del sistema, cada una construida desde cero SIN UN tipo:
    smurf (dos), satan (probe) y warezclient (r2l). u2r queda fuera: su tipo
    más grande tiene 30 filas.
  * Cada versión rehace TODO lo que aprende sin el tipo retirado: one-hot,
    escalado, selección de características, búsqueda del Autoencoder y su
    umbral p95, mini-experimento de balanceo y búsqueda del RandomForest, y
    calibración de UMBRAL_CONF por la regla τ. Detector y clasificador se
    mantienen (Autoencoder y RandomForest): no se vuelven a elegir.
  * Las filas del tipo retirado se parten 50/50: lote de EVALUACIÓN (el sistema
    no lo ve nunca) y reserva de LLEGADA. De la reserva llegan lotes acumulados
    de 10 → 25 → 50 → 100 → 200 filas, en secuencia: cada actualización se hace
    ENCIMA de la anterior. La cifra principal es el estado tras 100.
  * Métodos: R0 (no actualizar), R1 (reentrenar desde cero con la configuración
    de la versión sobre su D3 + lo llegado), I1 (warm_start: se añaden árboles
    entrenados con lo llegado + un replay fijo de 2.000 filas de D3). I1 en dos
    brazos fijados de antemano y SIEMPRE ejecutados: un tercio del bosque de partida
    por actualización (principal) y tantos como el bosque de partida
    (sensibilidad).
  * Solo se actualiza la etapa 2. La etapa 1 solo ve tráfico normal y la
    representación (columnas, escalado, selección) queda congelada al construir
    la versión: una actualización no cambia el espacio de características.
  * UMBRAL_CONF queda FIJO tras las actualizaciones. Comprobación a 100 filas:
    se recalibra el de R1 con la misma regla y se anota si elige otro valor.
  * Se mide sobre: el lote de evaluación del tipo retirado; sus filas de D2
    (otra captura; smurf y satan, warezclient no tiene); los ataques de tipo
    conocido de D2 por categoría (olvido); las normales de D2 (FPR). D2 se LEE
    para medir y no decide nada: todo lo que se elige se elige sobre KDDTrain+.
  * Cada semilla reconstruye la versión entera: 42 (referencia) + 1..10.

Salidas (ninguna pisa nada publicado)
-------------------------------------
  Resultados/aprendizaje_continuo_trabajo/parciales/   una pareja de CSV por
                                                       (tipo, semilla)
  Resultados/metricas_aprendizaje_continuo.csv          (--agregar) una fila por
      (tipo, semilla, método, filas llegadas, conjunto, categoría)
  Resultados/metricas_aprendizaje_continuo_versiones.csv (--agregar) una fila por
      (tipo, semilla): cómo quedó construida cada versión y sus tiempos

Uso
---
  python app\\aprendizaje_continuo.py --solo-plan
  python app\\aprendizaje_continuo.py --piloto --tipo smurf      # solo cronómetro
  python app\\aprendizaje_continuo.py --tipo smurf               # 11 semillas
  python app\\aprendizaje_continuo.py --todos                    # 3 tipos x 11
  python app\\aprendizaje_continuo.py --agregar                  # funde parciales
  python app\\aprendizaje_continuo.py --reconstruccion --todos   # reconstrucción completa
  python app\\aprendizaje_continuo.py --agregar-reconstruccion

Reconstrucción completa (R2)
----------------------------
  --reconstruccion mide R2: el sistema reconstruido ENTERO con las 100 primeras
  filas de la reserva de llegada dentro del entrenamiento, para comparar coste y
  prestaciones de reentrenar todo frente a actualizar solo la etapa 2. Cada
  (tipo, semilla) repite en el mismo proceso la versión y sus actualizaciones
  (tiempos en las mismas condiciones y control de que R0-I1 reproducen la tabla
  publicada) y después hace R2. Escribe en parciales_reconstruccion/ y
  --agregar-reconstruccion funde en dos tablas propias; no toca las anteriores.

Los tres tipos pueden lanzarse a la vez en tres terminales: cada (tipo, semilla)
escribe sus propios parciales y solo --agregar toca las tablas finales.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime

# Backend no interactivo antes de cualquier import que arrastre matplotlib.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

import config
import evaluacion
from program import NSLKDDPreprocessor
from anomalias import NSLKDDAnomalyTrainer
from firmas import NSLKDDSignatureTrainer
from hibrido import NSLKDDHybridEvaluator


D1_KEY = "D1_normal_for_anomaly"
D2_KEY = "D2_complete_test"
D3_KEY = "D3_known_attacks_for_signatures"

# Tipo retirado -> su categoría. Fijados antes de ejecutar (el mayor tipo
# viable de cada categoría, con filas de sobra para 200 de llegada y cientos de
# evaluación); no se eligieron mirando resultados.
TIPOS_RETIRADOS = {"smurf": "dos", "satan": "probe", "warezclient": "r2l"}

FRAC_EVALUACION = 0.5
TAMANOS_LLEGADA = [10, 25, 50, 100, 200]
PASO_PRINCIPAL = 100

N_REPLAY = 2000
MIN_REPLAY_POR_TIPO = 10
CATEGORIAS_REPLAY_ENTERAS = ("u2r",)   # 52 filas en todo el train: entran todas

# Árboles añadidos por actualización de I1, como fracción del bosque de partida.
BRAZOS_I1 = {"I1_tercio": 1.0 / 3.0, "I1_completo": 1.0}

DETECTOR = "Autoencoder"
FIRMA = "RandomForest"
SEMILLAS = [config.SEMILLA_POR_DEFECTO] + list(config.SEMILLAS_BARRIDO)

SET_FEATURES = "seleccion_rehecha"
SALIDAS = ("sin_alarma", "desconocido", "correcta", "equivocada")

DIR_TRABAJO = os.path.join(config.RESULTADOS_DIR, "aprendizaje_continuo_trabajo")
DIR_PARCIALES = os.path.join(DIR_TRABAJO, "parciales")
CSV_METRICAS = os.path.join(config.RESULTADOS_DIR, "metricas_aprendizaje_continuo.csv")
CSV_VERSIONES = os.path.join(config.RESULTADOS_DIR,
                             "metricas_aprendizaje_continuo_versiones.csv")
PREREGISTRO = os.path.join(config.REPO_DIR, "docs", "protocolo_aprendizaje_continuo.md")

ALCANCE = ("aprendizaje continuo de la etapa 2: reparto de las cuatro salidas "
           "de la cascada por conjunto y categoría real, para una versión del "
           "sistema construida sin el tipo retirado y cada estado de actualización; "
           "D2 solo se lee para medir")
ALCANCE_VERSIONES = ("construcción de cada versión sin el tipo retirado: "
                     "configuración elegida sobre KDDTrain+ menos el tipo y tiempos")

# --- reconstrucción completa (R2)
N_RECONSTRUCCION = PASO_PRINCIPAL
DIR_PARCIALES_R2 = os.path.join(DIR_TRABAJO, "parciales_reconstruccion")
CSV_RECONSTRUCCION = os.path.join(config.RESULTADOS_DIR,
                                  "metricas_aprendizaje_continuo_reconstruccion.csv")
CSV_RECONSTRUCCION_TIEMPOS = os.path.join(
    config.RESULTADOS_DIR, "metricas_aprendizaje_continuo_reconstruccion_tiempos.csv")
ALCANCE_R2 = ("sistema reconstruido entero con las {} primeras filas de "
              "llegada dentro del entrenamiento (R2); mismas filas de evaluación y mismos "
              "conjuntos de D2 que R0-I1".format(N_RECONSTRUCCION))
ALCANCE_R2_TIEMPOS = ("tiempos de construir la versión, de R2 y de cada "
                      "actualización a {} filas, medidos en serie en el mismo proceso, y "
                      "control de reproducción de R0-I1".format(N_RECONSTRUCCION))
COLS_CONTROL = ("n", "n_sin_alarma", "n_desconocido", "n_correcta", "n_equivocada",
                "n_arboles", "umbral_conf")

CLAVE_METRICAS = ("tipo_retirado", "semilla", "metodo", "n_llegada", "conjunto",
                  "categoria")


def huella_preregistro():
    """SHA-256 y mtime del documento de protocolo, o «no disponible» si no está."""
    if not os.path.exists(PREREGISTRO):
        return "no disponible", "no disponible"
    with open(PREREGISTRO, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    mtime = datetime.fromtimestamp(os.path.getmtime(PREREGISTRO))
    return sha, mtime.isoformat(timespec="seconds")


class NSLKDDContinualLearner:
    """Una versión del sistema sin un tipo, sus actualizaciones y su medida."""

    def __init__(self, tipo, piloto=False, reconstruccion=False):
        if tipo not in TIPOS_RETIRADOS:
            raise ValueError("Tipo no previsto en el preregistro: " + str(tipo))
        self.tipo = tipo
        self.categoria = TIPOS_RETIRADOS[tipo]
        self.piloto = piloto
        # Reconstrucción completa: la versión se construye CON las filas llegadas (R2).
        self.reconstruccion = reconstruccion
        self.semilla = config.RANDOM_STATE
        # Instancias SOLO para reutilizar rejillas, constructores y la regla de
        # calibración (como seleccion_ciega.py). Se crean DESPUÉS de fijar la
        # semilla: sus StratifiedKFold la congelan en __init__.
        self._anom = NSLKDDAnomalyTrainer()
        self._firm = NSLKDDSignatureTrainer()
        self._firm.ALGORITMOS = [FIRMA]   # el balanceo solo se busca para el RF
        self._hib = NSLKDDHybridEvaluator()
        self.filas = []
        self.version = {"tipo_retirado": tipo, "categoria_retirada": self.categoria,
                        "semilla": self.semilla}
        self.tiempos = {}

    # ------------------------------------------------------------------
    # 1. Particiones sobre el crudo
    # ------------------------------------------------------------------
    def particionar(self, train_df, test_df):
        """
        Aparta TODAS las filas del tipo retirado de KDDTrain+ y las parte en
        evaluación (50 %) y reserva de llegada (50 %), con la semilla de la
        corrida. D2 se extiende con esas filas para que las transforme el mismo
        preprocesado de la versión; se reconocen por su índice desplazado.
        """
        rng = np.random.RandomState(self.semilla)
        train = train_df.reset_index(drop=True)
        test = test_df.reset_index(drop=True)

        es_retirado = train["attack"] == self.tipo
        retiradas = train[es_retirado]
        orden = rng.permutation(retiradas.index.values)
        n_eval = int(len(orden) * FRAC_EVALUACION)
        self.idx_eval = orden[:n_eval]
        self.idx_llegada = orden[n_eval:]   # su ORDEN es el orden de llegada
        if len(self.idx_llegada) < max(TAMANOS_LLEGADA):
            raise RuntimeError("{}: la reserva de llegada ({}) no cubre {} filas".format(
                self.tipo, len(self.idx_llegada), max(TAMANOS_LLEGADA)))

        train_version = train[~es_retirado]
        # Guarda dura: ni una fila del tipo retirado en el entrenamiento.
        assert not (train_version["attack"] == self.tipo).any()

        self.offset = len(test) + 10 ** 6
        extra = retiradas.copy()
        extra.index = extra.index + self.offset
        test_ext = pd.concat([test, extra], axis=0)

        self.version.update({
            "n_filas_tipo": int(len(retiradas)),
            "n_evaluacion": int(len(self.idx_eval)),
            "n_reserva_llegada": int(len(self.idx_llegada)),
            "n_train_version": int(len(train_version)),
        })
        return train_version, test_ext

    # ------------------------------------------------------------------
    # 2. Construcción de la versión (se rehace todo sin el tipo)
    # ------------------------------------------------------------------
    def construir(self, train_version, test_ext):
        t0 = time.perf_counter()
        pre = NSLKDDPreprocessor()
        splits = pre.create_specialized_data_splits(train_version, test_ext)
        proc = pre.preprocess_specialized_splits(splits)
        destino = os.path.join(DIR_TRABAJO, "seleccion",
                               "{}_semilla{}{}".format(self.tipo, self.semilla,
                                                      "_r2" if self.reconstruccion else ""))
        os.makedirs(destino, exist_ok=True)
        # output_dir desviado: el valor por defecto pisaría selected_features.txt.
        pre.select_features(proc, output_dir=destino)
        self.tiempos["preprocesado_s"] = time.perf_counter() - t0

        X_D2_ext = proc[D2_KEY]["X"]
        cat_D2_ext = pd.Series(proc[D2_KEY]["y_category_original"], index=X_D2_ext.index)
        tipo_D2_ext = pd.Series(proc[D2_KEY]["y_attack_original"], index=X_D2_ext.index)
        es_extra = X_D2_ext.index >= self.offset

        self.X_D2 = X_D2_ext.loc[~es_extra]
        self.cat_D2 = cat_D2_ext.loc[~es_extra].astype(str).values
        self.tipo_D2 = tipo_D2_ext.loc[~es_extra].astype(str).values
        X_ret = X_D2_ext.loc[es_extra]
        self.X_eval = X_ret.loc[self.idx_eval + self.offset]
        self.X_llegada = X_ret.loc[self.idx_llegada + self.offset]

        self.X_D1 = proc[D1_KEY]["X"]
        self.X_D3 = proc[D3_KEY]["X"]
        self.y_D3 = np.asarray(proc[D3_KEY]["y_category_original"]).astype(str)
        self.tipo_D3 = np.asarray(proc[D3_KEY]["y_attack_original"]).astype(str)
        tipos_d3 = set(self.tipo_D3)
        # Guarda dura: sin el tipo en la versión; con él, y solo con él, en R2.
        assert (self.tipo in tipos_d3) == self.reconstruccion
        self.tipos_conocidos = tipos_d3 - {self.tipo}
        self.version["n_features"] = int(self.X_D3.shape[1])

        self._etapa1()
        self._etapa2()
        self._calibrar()
        self.tiempos["construccion_total_s"] = time.perf_counter() - t0

    def _etapa1(self):
        """Protocolo de anomalias.py: 80/20 de D1, AUC sobre D1_val + 5.000 de
        D3, umbral p95 sobre D1_val. Solo el Autoencoder."""
        t0 = time.perf_counter()
        a = self._anom
        a.X_D1_train, a.X_D1_val = train_test_split(
            self.X_D1, test_size=0.2, random_state=config.RANDOM_STATE)
        n_d3 = min(a.N_MUESTRA_D3, len(self.X_D3))
        muestra = self.X_D3.sample(n=n_d3, random_state=config.RANDOM_STATE)
        a.X_val_lab = pd.concat([a.X_D1_val, muestra], axis=0, ignore_index=True)
        a.y_val_lab = np.concatenate([np.full(len(a.X_D1_val), config.CLASE_NORMAL),
                                      np.full(n_d3, config.CLASE_ATAQUE)])
        modelo, cfg, auc_val, _t_fit, _t_score, _n_iter = a._seleccionar_config(DETECTOR)
        umbral = float(np.percentile(a._score(DETECTOR, modelo, a.X_D1_val),
                                     a.PERCENTIL_UMBRAL))
        self.detector, self.umbral_det = modelo, umbral
        self.version.update({"detector_config": str(cfg), "detector_auc_val": round(auc_val, 6),
                             "detector_umbral": umbral})
        self.tiempos["etapa1_s"] = time.perf_counter() - t0

    def _etapa2(self):
        """Protocolo de firmas.py: balanceo por f1_macro en CV y GridSearchCV
        sobre el D3 de la versión. Solo el RandomForest."""
        t0 = time.perf_counter()
        s = self._firm
        s.X_D3, s.y_D3 = self.X_D3, self.y_D3
        s._experimento_balanceo()
        balanceo = s.balanceo_ganador[FIRMA]
        gs = GridSearchCV(s._estimador_para(FIRMA, balanceo), s._param_grid(FIRMA, balanceo),
                          scoring="f1_macro", cv=s.cv, n_jobs=-1, refit=True)
        gs.fit(self.X_D3, self.y_D3)
        self.balanceo, self.config_rf = balanceo, dict(gs.best_params_)
        self.base = gs.best_estimator_
        self.n_arboles_base = int(self._clf(self.base).n_estimators)
        self.version.update({"balanceo": balanceo, "rf_config": str(self.config_rf),
                             "rf_f1_macro_cv": round(float(gs.best_score_), 6),
                             "n_arboles_base": self.n_arboles_base})
        self.tiempos["etapa2_s"] = time.perf_counter() - t0

    def _descriptor_firma(self):
        return {"algoritmo": FIRMA, "balanceo": self.balanceo,
                "config_ganadora": self.config_rf}

    def _calibrar(self, X=None, y=None):
        """Regla τ de hibrido.py sobre OOF (nunca D2). Devuelve el umbral."""
        t0 = time.perf_counter()
        h = self._hib
        h.X_D3 = self.X_D3 if X is None else X
        h.y_D3 = self.y_D3 if y is None else y
        umbral, _tabla = h._calibrar_umbral_conf(self._descriptor_firma())
        if X is None:
            self.umbral_conf = float(umbral)
            self.version["umbral_conf"] = self.umbral_conf
            self.tiempos["calibracion_s"] = time.perf_counter() - t0
        return float(umbral)

    @staticmethod
    def _clf(modelo):
        """El RandomForest dentro del pipeline de SMOTE, o el propio modelo."""
        return modelo.named_steps["clf"] if hasattr(modelo, "named_steps") else modelo

    # ------------------------------------------------------------------
    # 3. Replay
    # ------------------------------------------------------------------
    def _replay(self):
        """
        ~2.000 filas del D3 de la versión, estratificadas por tipo: u2r entero,
        el resto proporcional con un mínimo de MIN_REPLAY_POR_TIPO (o el tipo
        entero si tiene menos). Misma muestra en las cinco actualizaciones.
        """
        rng = np.random.RandomState(self.semilla)
        tipos = pd.Series(self.tipo_D3)
        cats = pd.Series(self.y_D3)
        enteros = tipos[cats.isin(CATEGORIAS_REPLAY_ENTERAS)].index.values
        resto = tipos[~cats.isin(CATEGORIAS_REPLAY_ENTERAS)]
        presupuesto = N_REPLAY - len(enteros)
        elegidos = list(enteros)
        for _t, sub in sorted(resto.groupby(resto), key=lambda kv: kv[0]):
            n = int(round(presupuesto * len(sub) / len(resto)))
            n = min(len(sub), max(MIN_REPLAY_POR_TIPO, n))
            elegidos.extend(rng.choice(sub.index.values, size=n, replace=False))
        pos = np.array(sorted(elegidos))
        assert set(self.y_D3[pos]) == set(config.CATEGORIAS_ATAQUE), \
            "el replay debe traer las cuatro clases (warm_start lo exige)"
        self.version["n_replay"] = int(len(pos))
        return self.X_D3.iloc[pos], self.y_D3[pos]

    # ------------------------------------------------------------------
    # 4. Medida de un estado del sistema
    # ------------------------------------------------------------------
    def _preparar_medida(self):
        """La etapa 1 no cambia: sus alarmas se calculan una sola vez."""
        a = self._anom
        self.sosp_eval = a._score(DETECTOR, self.detector, self.X_eval) > self.umbral_det
        self.sosp_D2 = a._score(DETECTOR, self.detector, self.X_D2) > self.umbral_det
        es_ataque = self.cat_D2 != config.ETIQUETA_NORMAL
        self.conjuntos = [("retirado_eval", self.categoria, None)]
        mask_ret_d2 = self.tipo_D2 == self.tipo
        if mask_ret_d2.any():
            self.conjuntos.append(("retirado_d2", self.categoria, mask_ret_d2))
        conocidos = es_ataque & np.isin(self.tipo_D2, list(self.tipos_conocidos))
        for c in config.CATEGORIAS_ATAQUE:
            self.conjuntos.append(("conocidos_d2", c, conocidos & (self.cat_D2 == c)))
        self.conjuntos.append(("conocidos_d2", "__total__", conocidos))
        self.conjuntos.append(("normal_d2", config.ETIQUETA_NORMAL, ~es_ataque))
        self.version["fpr_d2"] = round(float(self.sosp_D2[~es_ataque].mean()), 6)

    def _prediccion(self, modelo, X, sosp):
        """Cascada de hibrido.py: sin alarma -> 'normal'; con alarma -> argmax
        de la etapa 2, o 'unknown' si la confianza no llega a UMBRAL_CONF."""
        pred = np.full(len(X), config.ETIQUETA_NORMAL, dtype=object)
        t0 = time.perf_counter()
        if sosp.any():
            proba = modelo.predict_proba(X.loc[sosp])
            clases = np.asarray(modelo.classes_)
            cat = clases[proba.argmax(axis=1)].astype(object)
            cat[proba.max(axis=1) < self.umbral_conf] = config.ETIQUETA_UNKNOWN
            pred[sosp] = cat
        return pred, time.perf_counter() - t0, int(sosp.sum())

    def medir(self, modelo, metodo, n_llegada, t_actualizacion):
        pred_eval, _t, _n = self._prediccion(modelo, self.X_eval, self.sosp_eval)
        pred_D2, t_inf, n_inf = self._prediccion(modelo, self.X_D2, self.sosp_D2)
        latencia_ms = 1000.0 * t_inf / n_inf if n_inf else np.nan
        n_arboles = int(len(self._clf(modelo).estimators_))

        for conjunto, categoria, mask in self.conjuntos:
            if conjunto == "retirado_eval":
                pred = pred_eval
                real = np.full(len(pred), categoria, dtype=object)
            else:
                pred = pred_D2[mask]
                real = self.cat_D2[mask]
            n = int(len(pred))
            sin_alarma = int((pred == config.ETIQUETA_NORMAL).sum())
            desconocido = int((pred == config.ETIQUETA_UNKNOWN).sum())
            con_categoria = (pred != config.ETIQUETA_NORMAL) & (pred != config.ETIQUETA_UNKNOWN)
            correcta = int((con_categoria & (pred == real)).sum())
            equivocada = int((con_categoria & (pred != real)).sum())
            assert sin_alarma + desconocido + correcta + equivocada == n
            fila = {
                "tipo_retirado": self.tipo, "categoria_retirada": self.categoria,
                "metodo": metodo, "n_llegada": int(n_llegada),
                "conjunto": conjunto, "categoria": categoria, "n": n,
                "n_sin_alarma": sin_alarma, "n_desconocido": desconocido,
                "n_correcta": correcta, "n_equivocada": equivocada,
            }
            for s in SALIDAS:
                fila["pct_" + s] = round(100.0 * fila["n_" + s] / n, 4) if n else np.nan
            fila.update({
                "n_arboles": n_arboles, "umbral_conf": self.umbral_conf,
                "tiempo_actualizacion_s": round(t_actualizacion, 4)
                if t_actualizacion is not None else np.nan,
                "latencia_ms_por_flujo": round(latencia_ms, 6),
            })
            self.filas.append(fila)

    # ------------------------------------------------------------------
    # 5. Actualizaciones
    # ------------------------------------------------------------------
    def actualizar(self):
        self._preparar_medida()
        self.medir(self.base, "R0", 0, None)

        X_rep, y_rep = self._replay()
        brazos = {nombre: copy.deepcopy(self.base) for nombre in BRAZOS_I1}
        for n in TAMANOS_LLEGADA:
            X_lleg = self.X_llegada.iloc[:n]
            y_lleg = np.full(n, self.categoria, dtype=object).astype(str)

            # R1: la configuración de la versión, desde cero, sobre D3 + lo llegado.
            X_r1 = pd.concat([self.X_D3, X_lleg], axis=0)
            y_r1 = np.concatenate([self.y_D3, y_lleg])
            r1 = self._firm._estimador_para(FIRMA, self.balanceo)
            r1.set_params(**self.config_rf)
            t0 = time.perf_counter()
            r1.fit(X_r1, y_r1)
            self.medir(r1, "R1", n, time.perf_counter() - t0)
            if n == PASO_PRINCIPAL:
                # Comprobación: ¿elegiría la regla τ otro umbral para R1?
                self.version["umbral_conf_recalibrado_r1_{}".format(n)] = \
                    self._calibrar(X_r1, y_r1)

            # I1: se añaden árboles entrenados con lo llegado + el replay fijo.
            X_i1 = pd.concat([X_lleg, X_rep], axis=0)
            y_i1 = np.concatenate([y_lleg, y_rep])
            for nombre, frac in BRAZOS_I1.items():
                modelo = brazos[nombre]
                clf = self._clf(modelo)
                k = int(round(frac * self.n_arboles_base))
                prefijo = "clf__" if hasattr(modelo, "named_steps") else ""
                modelo.set_params(**{prefijo + "warm_start": True,
                                     prefijo + "n_estimators": clf.n_estimators + k})
                t0 = time.perf_counter()
                modelo.fit(X_i1, y_i1)
                t_act = time.perf_counter() - t0
                assert list(self._clf(modelo).classes_) == list(config.CATEGORIAS_ATAQUE)
                self.medir(modelo, nombre, n, t_act)

        self._comprobar_etapa1_constante()

    def _comprobar_etapa1_constante(self):
        """La etapa 1 no cambia, así que 'sin alarma' tiene que ser idéntica
        en todos los estados de cada conjunto. Si se mueve, hay un error."""
        df = pd.DataFrame(self.filas)
        variacion = df.groupby(["conjunto", "categoria"])["n_sin_alarma"].nunique()
        assert (variacion == 1).all(), "la columna 'sin alarma' cambió entre estados"

    # ------------------------------------------------------------------
    # 6. Escritura del parcial de (tipo, semilla)
    # ------------------------------------------------------------------
    def escribir_parcial(self, sha, mtime, dir_=DIR_PARCIALES, csv_met=CSV_METRICAS,
                         csv_ver=CSV_VERSIONES, alcance=ALCANCE,
                         alcance_ver=ALCANCE_VERSIONES, version_extra=None):
        procedencia = {"semilla": self.semilla, "commit": config.commit_actual(),
                       "fecha": datetime.now().isoformat(timespec="seconds"),
                       "set_features": SET_FEATURES, "sin_seleccion": False,
                       "n_features": self.version["n_features"],
                       "preregistro_sha256": sha, "preregistro_mtime": mtime}
        filas = []
        for f in self.filas:
            fila = dict(f, alcance=alcance, **procedencia)
            evaluacion.validar_esquema_minimo(fila, csv_met, auxiliar=True,
                                              columnas=evaluacion.COLUMNAS_MINIMAS_AUXILIARES)
            filas.append(fila)
        version = dict(self.version, alcance=alcance_ver, **procedencia)
        version.update({k: round(v, 2) for k, v in self.tiempos.items()})
        if version_extra:
            version.update(version_extra)
        evaluacion.validar_esquema_minimo(version, csv_ver, auxiliar=True,
                                          columnas=evaluacion.COLUMNAS_MINIMAS_AUXILIARES)

        os.makedirs(dir_, exist_ok=True)
        base = os.path.join(dir_, "{}_semilla{}".format(self.tipo, self.semilla))
        pd.DataFrame(filas).to_csv(base + "_metricas.csv", index=False, encoding="utf-8")
        pd.DataFrame([version]).to_csv(base + "_version.csv", index=False, encoding="utf-8")
        print("   → {}_metricas.csv ({} filas) y _version.csv".format(base, len(filas)))


def filas_esperadas(tipo):
    """R0 una vez + (R1 + brazos de I1) por tamaño, por cada conjunto medido."""
    n_estados = 1 + (1 + len(BRAZOS_I1)) * len(TAMANOS_LLEGADA)
    n_conjuntos = 1 + (1 if tipo in ("smurf", "satan") else 0) + \
        len(config.CATEGORIAS_ATAQUE) + 1 + 1
    return n_estados * n_conjuntos


def correr(tipo, semilla, train_df, test_df, piloto=False, huella=None):
    config.fijar_semilla(semilla)
    t0 = time.perf_counter()
    print("\n" + "=" * 70)
    print("VERSIÓN SIN '{}' · semilla {}{}".format(tipo, semilla,
                                                  " · PILOTO (solo cronómetro)" if piloto else ""))
    print("=" * 70)
    ap = NSLKDDContinualLearner(tipo, piloto=piloto)
    train_v, test_ext = ap.particionar(train_df, test_df)
    ap.construir(train_v, test_ext)
    t_constr = time.perf_counter() - t0
    ap.actualizar()
    ap.tiempos["actualizaciones_s"] = time.perf_counter() - t0 - t_constr
    ap.tiempos["total_s"] = time.perf_counter() - t0

    if piloto:
        # El piloto NO imprime ni guarda métricas de detección, solo tiempos.
        os.makedirs(DIR_TRABAJO, exist_ok=True)
        ruta = os.path.join(DIR_TRABAJO, "piloto_tiempos_{}_semilla{}.json".format(tipo, semilla))
        with open(ruta, "w", encoding="utf-8") as fh:
            json.dump({k: round(v, 1) for k, v in ap.tiempos.items()}, fh, indent=2)
        print("\nPILOTO — tiempos (s): {}".format(
            {k: round(v, 1) for k, v in ap.tiempos.items()}))
        print("   → {}".format(ruta))
        return

    # La huella es la leída AL ARRANCAR (main), no la del momento de escribir: una
    # edición del protocolo a mitad de ejecución no puede colarse en las tablas.
    sha, mtime = huella
    esperadas = filas_esperadas(tipo)
    assert len(ap.filas) == esperadas, "{} filas, se esperaban {}".format(len(ap.filas), esperadas)
    ap.escribir_parcial(sha, mtime)
    print("   Tiempos (s): {}".format({k: round(v, 1) for k, v in ap.tiempos.items()}))


def agregar():
    """Funde los parciales en las dos tablas y comprueba unicidad y recuento."""
    import glob
    met = sorted(glob.glob(os.path.join(DIR_PARCIALES, "*_metricas.csv")))
    ver = sorted(glob.glob(os.path.join(DIR_PARCIALES, "*_version.csv")))
    if not met:
        raise RuntimeError("No hay parciales en " + DIR_PARCIALES)
    df = pd.concat([pd.read_csv(p) for p in met], ignore_index=True)
    dv = pd.concat([pd.read_csv(p) for p in ver], ignore_index=True)
    df.to_csv(CSV_METRICAS, index=False, encoding="utf-8")
    dv.to_csv(CSV_VERSIONES, index=False, encoding="utf-8")
    evaluacion.comprobar_unicidad(CSV_METRICAS, clave=CLAVE_METRICAS)
    evaluacion.comprobar_unicidad(CSV_VERSIONES, clave=("tipo_retirado", "semilla"))
    for (tipo, semilla), sub in df.groupby(["tipo_retirado", "semilla"]):
        assert len(sub) == filas_esperadas(tipo), \
            "{} semilla {}: {} filas".format(tipo, semilla, len(sub))
    # Que no falte ninguna pareja (tipo, semilla): un parcial ausente no puede
    # pasar en silencio.
    esperadas = {(t, s) for t in TIPOS_RETIRADOS for s in SEMILLAS}
    presentes = set(zip(dv["tipo_retirado"], dv["semilla"]))
    assert presentes == esperadas, "faltan versiones: {}".format(sorted(esperadas - presentes))
    shas = df["preregistro_sha256"].unique()
    assert len(shas) == 1, "El preregistro cambió entre corridas: huellas {}".format(list(shas))
    print("   → {} ({} filas) · {} ({} versiones)".format(
        CSV_METRICAS, len(df), CSV_VERSIONES, len(dv)))
    print("   Versiones por tipo: {}".format(dv.groupby("tipo_retirado").size().to_dict()))
    print("   Huella única del preregistro: {}".format(shas[0]))


def _control_reproduccion(ap):
    """Compara R0-I1 de la repetición con la tabla publicada. Devuelve el número
    de filas que difieren en alguna columna de COLS_CONTROL."""
    pub = pd.read_csv(CSV_METRICAS)
    pub = pub[(pub["tipo_retirado"] == ap.tipo) & (pub["semilla"] == ap.semilla)]
    rep_ = pd.DataFrame(ap.filas)
    clave = ["metodo", "n_llegada", "conjunto", "categoria"]
    m = rep_.merge(pub, on=clave, how="outer", suffixes=("_rep", "_pub"), indicator=True)
    if (m["_merge"] != "both").any():
        return int((m["_merge"] != "both").sum())
    difiere = np.zeros(len(m), dtype=bool)
    for c in COLS_CONTROL:
        a, b = m[c + "_rep"].astype(float), m[c + "_pub"].astype(float)
        difiere |= ~np.isclose(a, b, rtol=0, atol=1e-6)
    return int(difiere.sum())


def correr_reconstruccion(tipo, semilla, train_df, test_df, huella):
    """Reconstrucción completa: la versión y sus actualizaciones (como correr) y, justo después,
    R2: el sistema reconstruido entero con las N_RECONSTRUCCION primeras filas
    de llegada dentro del entrenamiento."""
    config.fijar_semilla(semilla)
    print("\n" + "=" * 70)
    print("RECONSTRUCCIÓN COMPLETA · sin '{}' · semilla {}".format(tipo, semilla))
    print("=" * 70)
    ap = NSLKDDContinualLearner(tipo)
    train_v, test_ext = ap.particionar(train_df, test_df)
    ap.construir(train_v, test_ext)
    ap.actualizar()
    n_dif = _control_reproduccion(ap)
    print("   Control: {} filas de R0-I1 difieren de la tabla publicada".format(n_dif))

    # R2: misma semilla, misma partición (RandomState(semilla)), las filas
    # llegadas dentro. Las instancias se crean tras fijar la semilla, como en correr.
    config.fijar_semilla(semilla)
    r2 = NSLKDDContinualLearner(tipo, reconstruccion=True)
    train_v2, test_ext2 = r2.particionar(train_df, test_df)
    assert (r2.idx_eval == ap.idx_eval).all() and (r2.idx_llegada == ap.idx_llegada).all()
    llegadas = train_df.reset_index(drop=True).loc[r2.idx_llegada[:N_RECONSTRUCCION]]
    assert (llegadas["attack"] == tipo).all()
    train_r2 = pd.concat([train_v2, llegadas], axis=0)
    r2.construir(train_r2, test_ext2)
    r2._preparar_medida()
    r2.medir(r2.base, "R2_reconstruccion", N_RECONSTRUCCION,
             r2.tiempos["construccion_total_s"])

    # Tiempos de la versión y de cada método a N_RECONSTRUCCION filas.
    fa = pd.DataFrame(ap.filas)
    a100 = fa[fa["n_llegada"] == N_RECONSTRUCCION].groupby("metodo").first()
    r0 = fa[fa["metodo"] == "R0"].iloc[0]
    extra = {"version_construccion_total_s": round(ap.tiempos["construccion_total_s"], 2),
             "version_preprocesado_s": round(ap.tiempos["preprocesado_s"], 2),
             "version_etapa1_s": round(ap.tiempos["etapa1_s"], 2),
             "version_etapa2_s": round(ap.tiempos["etapa2_s"], 2),
             "version_calibracion_s": round(ap.tiempos["calibracion_s"], 2),
             "latencia_ms_R0": r0["latencia_ms_por_flujo"],
             "n_arboles_R0": int(r0["n_arboles"]),
             "n_filas_control_distintas": n_dif}
    for metodo in ("R1",) + tuple(BRAZOS_I1):
        extra["tiempo_{}_s".format(metodo)] = a100.loc[metodo, "tiempo_actualizacion_s"]
        extra["latencia_ms_{}".format(metodo)] = a100.loc[metodo, "latencia_ms_por_flujo"]
        extra["n_arboles_{}".format(metodo)] = int(a100.loc[metodo, "n_arboles"])
    extra["latencia_ms_R2"] = r2.filas[0]["latencia_ms_por_flujo"]
    extra["n_arboles_R2"] = r2.filas[0]["n_arboles"]
    extra["n_llegada"] = N_RECONSTRUCCION

    sha, mtime = huella
    esperadas = len(r2.conjuntos)
    assert len(r2.filas) == esperadas, "{} filas R2, se esperaban {}".format(len(r2.filas), esperadas)
    r2.escribir_parcial(sha, mtime, dir_=DIR_PARCIALES_R2, csv_met=CSV_RECONSTRUCCION,
                        csv_ver=CSV_RECONSTRUCCION_TIEMPOS, alcance=ALCANCE_R2,
                        alcance_ver=ALCANCE_R2_TIEMPOS, version_extra=extra)
    print("   R2 construcción {:.1f} s · versión {:.1f} s · R1 {:.2f} s".format(
        r2.tiempos["construccion_total_s"], ap.tiempos["construccion_total_s"],
        extra["tiempo_R1_s"]))


def agregar_reconstruccion():
    import glob
    met = sorted(glob.glob(os.path.join(DIR_PARCIALES_R2, "*_metricas.csv")))
    ver = sorted(glob.glob(os.path.join(DIR_PARCIALES_R2, "*_version.csv")))
    if not met:
        raise RuntimeError("No hay parciales en " + DIR_PARCIALES_R2)
    df = pd.concat([pd.read_csv(p) for p in met], ignore_index=True)
    dv = pd.concat([pd.read_csv(p) for p in ver], ignore_index=True)
    df.to_csv(CSV_RECONSTRUCCION, index=False, encoding="utf-8")
    dv.to_csv(CSV_RECONSTRUCCION_TIEMPOS, index=False, encoding="utf-8")
    evaluacion.comprobar_unicidad(CSV_RECONSTRUCCION, clave=CLAVE_METRICAS)
    evaluacion.comprobar_unicidad(CSV_RECONSTRUCCION_TIEMPOS, clave=("tipo_retirado", "semilla"))
    esperadas = {(t, s) for t in TIPOS_RETIRADOS for s in SEMILLAS}
    presentes = set(zip(dv["tipo_retirado"], dv["semilla"]))
    assert presentes == esperadas, "faltan versiones: {}".format(sorted(esperadas - presentes))
    shas = df["preregistro_sha256"].unique()
    assert len(shas) == 1, "El preregistro cambió entre corridas: huellas {}".format(list(shas))
    print("   → {} ({} filas) · {} ({} filas)".format(
        CSV_RECONSTRUCCION, len(df), CSV_RECONSTRUCCION_TIEMPOS, len(dv)))
    print("   Filas de control distintas de la tabla publicada: {}".format(
        int(dv["n_filas_control_distintas"].sum())))


def imprimir_plan(train_df):
    print("\nPLAN (protocolo del experimento)")
    for tipo, cat in TIPOS_RETIRADOS.items():
        n = int((train_df["attack"] == tipo).sum())
        n_eval = int(n * FRAC_EVALUACION)
        print("   {:<12} ({:<5}) {:>5} filas → evaluación {:>5} · reserva {:>5} · "
              "llegada {} · {} filas por semilla".format(
                  tipo, cat, n, n_eval, n - n_eval, TAMANOS_LLEGADA, filas_esperadas(tipo)))
    print("   Semillas: {} (cada una reconstruye la versión)".format(SEMILLAS))
    print("   I1: {} · replay ~{} filas · UMBRAL_CONF fijo".format(BRAZOS_I1, N_REPLAY))
    print("   Documento de protocolo: {} ({})".format(
        PREREGISTRO, "existe" if os.path.exists(PREREGISTRO) else "NO EXISTE"))


def main():
    parser = argparse.ArgumentParser(
        description="Experimento de aprendizaje continuo de la etapa 2.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--tipo", choices=sorted(TIPOS_RETIRADOS))
    grupo.add_argument("--todos", action="store_true", help="Los tres tipos, en serie.")
    parser.add_argument("--semillas", type=int, nargs="+", default=None,
                        help="Por defecto 42 + 1..10.")
    parser.add_argument("--piloto", action="store_true",
                        help="Una semilla, solo tiempos: no imprime ni guarda métricas.")
    parser.add_argument("--solo-plan", action="store_true")
    parser.add_argument("--agregar", action="store_true",
                        help="Funde los parciales en las tablas finales.")
    parser.add_argument("--reconstruccion", action="store_true",
                        help="Mide R2 (reconstrucción completa).")
    parser.add_argument("--agregar-reconstruccion", action="store_true",
                        help="Funde los parciales de R2 en sus dos tablas.")
    args = parser.parse_args()

    if args.agregar:
        agregar()
        return 0
    if args.agregar_reconstruccion:
        agregar_reconstruccion()
        return 0

    pre = NSLKDDPreprocessor()
    train_df, test_df = pre.load_dataset()
    if train_df is None:
        raise RuntimeError("No se pudo cargar el dataset.")

    if args.solo_plan:
        imprimir_plan(train_df)
        return 0

    tipos = sorted(TIPOS_RETIRADOS) if args.todos else ([args.tipo] if args.tipo else [])
    if not tipos:
        parser.error("indica --tipo, --todos, --solo-plan o --agregar")
    semillas = args.semillas or (SEMILLAS[:1] if args.piloto else SEMILLAS)

    huella = None
    if not args.piloto:
        huella = huella_preregistro()
        print("Protocolo · sha256 {} · modificado {}".format(*huella))

    for tipo in tipos:
        for semilla in semillas:
            if args.reconstruccion:
                correr_reconstruccion(tipo, semilla, train_df, test_df, huella)
            else:
                correr(tipo, semilla, train_df, test_df, piloto=args.piloto, huella=huella)
    return 0


if __name__ == "__main__":
    sys.exit(main())
