# -*- coding: utf-8 -*-
"""
anomalias.py — Etapa 1 del H-NIDS: detección de anomalías.

Compara CUATRO algoritmos no supervisados entrenados SOLO con tráfico normal (D1)
y evaluados sobre el test completo (D2), bajo protocolo idéntico (mismo split, la
misma semilla —`config.RANDOM_STATE`: 42 por defecto, la del barrido de semillas
con `--semilla`—, mismo umbral percentil 95, mismas métricas):

  - IsolationForest
  - OneClassSVM (kernel RBF)
  - LocalOutlierFactor (novelty=True)
  - Autoencoder implementado como MLPRegressor (fit(X, X)); PROHIBIDO TF/Keras.

Todos exponen un anomaly score UNIFICADO donde MAYOR = MÁS ANÓMALO, para que la
comparación (AUC, umbral, matrices) sea homogénea.

Reglas de protocolo (invalidan el TFG si se rompen):
  * D2 es INTOCABLE para ajustar hiperparámetros o el umbral.
  * Selección de hiperparámetros: mejor AUC-ROC sobre un set etiquetado formado por
    D1_val (normal=0) + una muestra de D3 (ataques CONOCIDOS del train=1). Jamás D2.
  * Umbral: percentil 95 del anomaly score sobre D1_val (solo normal). Igual para los 4.

El set de features es parametrizable (--sin-seleccion → 122 features
en lugar de las 54 por defecto) para poder correr el experimento 54-vs-122 sin
reescribir el script. Nada de rutas hardcodeadas: todo sale de config.base_path().

Semilla (--semilla N): la corrida acepta una semilla distinta de la 42.
POR DEFECTO ES LA 42 y se usan los nombres publicados (mismos nombres de
artefacto, misma tabla publicada, mismas figuras). Con otra semilla los
.joblib y las figuras se sufijan '_semilla<N>' y las métricas van a
'metricas_anomalias_semillas.csv': ni las tablas ni los artefactos de la 42 se
tocan. El mecanismo y su razón están en el encabezado de config.py; las diez
semillas del barrido, en config.SEMILLAS_BARRIDO (la 42 NO está entre ellas).
"""
import argparse
import time

# Backend no interactivo fijado de forma explícita ANTES de cualquier import que
# arrastre matplotlib (evaluacion/program). Así no dependemos del orden de import
# para que nada bloquee ni intente abrir ventanas.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

import config
import evaluacion  # importado antes que program: fija el backend Agg de matplotlib
from program import load_specialized_splits


class NSLKDDAnomalyTrainer:
    """
    Entrena y compara los cuatro detectores de anomalías de la etapa 1 del H-NIDS.

    Sigue el estilo de NSLKDDPreprocessor / NSLKDDValidator: una sola clase que
    orquesta carga, selección de hiperparámetros, umbral y evaluación.
    """

    # --- Grids pequeños. Se recorren enteros, no hay GridSearchCV
    # porque el problema es no supervisado: la selección se hace por AUC-ROC sobre
    # el set etiquetado D1_val + muestra de D3. ---
    GRIDS = {
        "IsolationForest": [
            {"n_estimators": n, "max_samples": m}
            for n in (100, 200, 300)
            for m in (0.5, 1.0)
        ],
        "OneClassSVM": [
            {"nu": nu, "gamma": g}
            for nu in (0.01, 0.05, 0.1)
            for g in ("scale", 0.01, 0.1)
        ],
        "LocalOutlierFactor": [
            {"n_neighbors": k} for k in (10, 20, 35, 50)
        ],
        "Autoencoder": [
            {"hidden_layer_sizes": h} for h in ((64, 32, 64), (32, 16, 32))
        ],
    }

    ALGORITMOS = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "Autoencoder"]

    # OneClassSVM es O(n^2): entrenarlo con las ~67k filas de D1 es inviable.
    # Se entrena con una submuestra reproducible.
    N_SUBMUESTRA_OCSVM = 20000
    # Tamaño de la muestra de ataques CONOCIDOS (D3) para la selección de config.
    N_MUESTRA_D3 = 5000
    PERCENTIL_UMBRAL = 95  # ~5% de FPR aceptado sobre lo normal.

    def __init__(self, sin_seleccion=False):
        """
        Parameters
        ----------
        sin_seleccion : bool
            False (por defecto) → set de 54 features (selección 4.3.5 aplicada).
            True → set de 122 features (variante sin selección).
        """
        self.sin_seleccion = sin_seleccion
        self.base_path = config.base_path(sin_seleccion=sin_seleccion)
        # Etiqueta legible del set de features para la tabla de métricas.
        self.set_features = "122_sin_seleccion" if sin_seleccion else "54"
        # Token para los NOMBRES de artefacto (joblibs y figuras). Con la semilla
        # 42 es igual a set_features, así que los nombres publicados no cambian;
        # con otra semilla añade '_semilla<N>' y el barrido de semillas no pisa nada.
        # Se congela aquí: config.fijar_semilla() se llama antes de instanciar.
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)

        # Rellenados por cargar_datos()
        self.X_D1_train = None
        self.X_D1_val = None
        self.X_D2 = None
        self.y_bin = None          # etiqueta binaria de D2 (1=ataque, 0=normal)
        self.X_val_lab = None      # D1_val + muestra D3 (para selección de config)
        self.y_val_lab = None
        self.n_features = None

        # Rellenados por entrenar_todos()
        self.resultados = {}       # {algo: dict con modelo, umbral, config, métricas}

    # ------------------------------------------------------------------
    # 1-2. Carga de datos y construcción de conjuntos
    # ------------------------------------------------------------------
    def cargar_datos(self):
        """Carga D1/D2/D3, define la etiqueta binaria de D2 y prepara los splits."""
        print("=" * 70)
        print("CARGA DE DATOS (set de features: {})".format(self.set_features))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError(
                "No se pudieron cargar las divisiones desde: " + self.base_path
            )

        X_D1 = splits["D1"]["X"]
        self.X_D2 = splits["D2"]["X"]
        self.n_features = X_D1.shape[1]

        # Etiqueta binaria de D2: positivo = ataque = 1 (config.CLASE_ATAQUE).
        cat_D2 = splits["D2"]["y_category"]["category_original"].values
        self.y_bin = (cat_D2 != config.ETIQUETA_NORMAL).astype(int)

        # Split de D1 (solo normal): 80% entrenamiento, 20% validación, con la
        # semilla de la corrida (42 por defecto; otra en el barrido de semillas).
        self.X_D1_train, self.X_D1_val = train_test_split(
            X_D1, test_size=0.2, random_state=config.RANDOM_STATE
        )

        # Set de validación ETIQUETADO para elegir hiperparámetros:
        # D1_val (normal=0) + muestra de D3 (ataques conocidos del train=1).
        # NUNCA se usa D2 para esto.
        X_D3 = splits["D3"]["X"]
        n_d3 = min(self.N_MUESTRA_D3, len(X_D3))
        X_D3_muestra = X_D3.sample(n=n_d3, random_state=config.RANDOM_STATE)

        self.X_val_lab = pd.concat([self.X_D1_val, X_D3_muestra], axis=0, ignore_index=True)
        self.y_val_lab = np.concatenate([
            np.full(len(self.X_D1_val), config.CLASE_NORMAL),
            np.full(len(X_D3_muestra), config.CLASE_ATAQUE),
        ])

        print("   D1: {} filas ({} train / {} val) · {} features".format(
            len(X_D1), len(self.X_D1_train), len(self.X_D1_val), self.n_features))
        print("   D2: {} filas ({} ataques / {} normales)".format(
            len(self.X_D2), int(self.y_bin.sum()), int((self.y_bin == 0).sum())))
        print("   Set validación etiquetado: {} normales (D1_val) + {} ataques (D3)".format(
            len(self.X_D1_val), n_d3))

    # ------------------------------------------------------------------
    # Construcción de modelos, datos de entrenamiento y anomaly score unificado
    # ------------------------------------------------------------------
    def _construir(self, algo, cfg):
        """Instancia el modelo del algoritmo con la configuración dada.

        La semilla es la de la corrida (config.RANDOM_STATE: 42 por defecto, otra
        con --semilla en el barrido de semillas), no un literal.
        """
        if algo == "IsolationForest":
            return IsolationForest(random_state=config.RANDOM_STATE, n_jobs=-1, **cfg)
        if algo == "OneClassSVM":
            # OneClassSVM no admite random_state (algoritmo determinista).
            return OneClassSVM(kernel="rbf", **cfg)
        if algo == "LocalOutlierFactor":
            # novelty=True OBLIGATORIO para poder puntuar datos nuevos (D2).
            return LocalOutlierFactor(novelty=True, n_jobs=-1, **cfg)
        if algo == "Autoencoder":
            # Autoencoder = MLPRegressor reconstruyendo su entrada (fit(X, X)).
            # early_stopping reserva internamente un 10% para parar sin tocar D2.
            return MLPRegressor(
                random_state=config.RANDOM_STATE,
                early_stopping=True,
                max_iter=300,
                **cfg,
            )
        raise ValueError("Algoritmo desconocido: " + algo)

    @staticmethod
    def _iteraciones_ajuste(algo, model):
        """
        Épocas que consumió el ajuste de UN modelo ya entrenado — SOLO para el
        Autoencoder; None para los otros tres.

        POR QUÉ EXISTE: el Autoencoder es MLPRegressor con
        early_stopping=True y max_iter=300, así que su 'tiempo_entrenamiento_s'
        depende de DOS cosas que el tiempo solo no distingue —cuántas épocas necesitó y
        cómo de cargada estaba la máquina—. Sin este dato, dos filas del
        Autoencoder con tiempos de ajuste muy distintos no se pueden interpretar.

        CUIDADO CON EL DENOMINADOR: esta función devuelve las épocas de UNA configuración. La fila publica DOS
        columnas a partir de ella y no son intercambiables:
          - 'n_iter_ganador'    = épocas del modelo GANADOR. Dice si el ajuste que
                                  se publica se cortó por max_iter o convergió.
                                  NO es el denominador de 'tiempo_entrenamiento_s'.
          - 'n_iter_total_grid' = suma de las épocas de TODAS las configuraciones
                                  del grid, que es exactamente el conjunto de fit
                                  que 'tiempo_entrenamiento_s' cronometra. Ese sí
                                  es el denominador válido para segundos por época.
        Se acumula en _seleccionar_config(), dentro del mismo bucle que suma los
        fit, para que ambos cubran el mismo conjunto por construcción.

        POR QUÉ SOLO EL AUTOENCODER y no un getattr genérico: en sklearn moderno
        OneClassSVM también expone 'n_iter_', pero como ndarray de libsvm —una
        celda con un array dentro del CSV— y no significa lo mismo. IsolationForest
        y LocalOutlierFactor no ajustan iterativamente y no tienen el atributo.
        Las tres filas quedan con la celda VACÍA, que es la convención ya usada en
        el resto de columnas no aplicables (nunca un 0 que se pueda leer como
        medida).
        """
        if algo != "Autoencoder":
            return None
        n_iter = getattr(model, "n_iter_", None)
        return int(n_iter) if n_iter is not None else None

    def _datos_entrenamiento(self, algo, X_train):
        """
        Datos con los que se entrena cada algoritmo. OneClassSVM usa una submuestra
        de 20.000 (O(n^2)); el resto usa todo D1_train. Submuestra reproducible.
        """
        if algo == "OneClassSVM" and len(X_train) > self.N_SUBMUESTRA_OCSVM:
            return X_train.sample(n=self.N_SUBMUESTRA_OCSVM, random_state=config.RANDOM_STATE)
        return X_train

    def _ajustar(self, algo, model, X_fit):
        """Entrena el modelo. El autoencoder reconstruye su entrada: fit(X, X)."""
        if algo == "Autoencoder":
            model.fit(X_fit, X_fit)
        else:
            model.fit(X_fit)
        return model

    def _score(self, algo, model, X):
        """
        Anomaly score UNIFICADO: MAYOR = MÁS ANÓMALO (más probable ataque).

        En sklearn, score_samples/decision_function dan "mayor = más normal";
        por eso se niegan. El autoencoder usa el MSE de reconstrucción (ya positivo
        y creciente con la anomalía).
        """
        if algo in ("IsolationForest", "LocalOutlierFactor"):
            return -model.score_samples(X)
        if algo == "OneClassSVM":
            return -np.asarray(model.decision_function(X)).ravel()
        if algo == "Autoencoder":
            recon = model.predict(X)
            return ((recon - X.values) ** 2).mean(axis=1)
        raise ValueError("Algoritmo desconocido: " + algo)

    # ------------------------------------------------------------------
    # 4. Selección de hiperparámetros por AUC-ROC (no supervisada de facto)
    # ------------------------------------------------------------------
    def _seleccionar_config(self, algo):
        """
        Entrena cada configuración del grid sobre D1_train y elige la de mejor
        AUC-ROC sobre el set etiquetado (D1_val + muestra D3). Devuelve el modelo
        ganador (ya entrenado sobre D1_train), su config, su AUC de validación,
        el tiempo de ENTRENAMIENTO acumulado (solo los fit del grid, sin contar
        el scoring: se separan entrenamiento e inferencia), el tiempo de SCORING
        del set etiquetado acumulado sobre todas las configuraciones del grid y
        las ÉPOCAS acumuladas sobre esas mismas configuraciones (None salvo en el
        Autoencoder).

        El segundo cronómetro es el tramo (2) del bloque: se MIDE en lugar de
        estimarse porque es el componente grande de la parte de 'tiempo_s' que no
        es ni ajuste ni inferencia sobre D2, y cualquier reparto calculado a mano
        —por filas puntuadas o ponderando el grid— depende de un modelo de coste
        que el dato publicado no declara. Se publica como columna
        'tiempo_score_seleccion_s'.

        Las épocas acumuladas se suman DENTRO de este mismo bucle, y ese es el
        punto: 't_n_iter_total' cubre exactamente los mismos fit que
        't_fit_total', así que su cociente sí son segundos por época. Las épocas
        del ganador solo (n_iter_ del modelo devuelto) NO sirven para esa
        división — ver _iteraciones_ajuste().
        """
        X_fit = self._datos_entrenamiento(algo, self.X_D1_train)
        mejor = {"auc": -np.inf, "cfg": None, "model": None}
        t_fit_total = 0.0
        t_score_total = 0.0
        n_iter_total = None   # se queda en None salvo en el Autoencoder

        for cfg in self.GRIDS[algo]:
            model = self._construir(algo, cfg)
            t_fit = time.perf_counter()
            self._ajustar(algo, model, X_fit)
            t_fit_total += time.perf_counter() - t_fit
            n_iter_cfg = self._iteraciones_ajuste(algo, model)
            if n_iter_cfg is not None:
                n_iter_total = (n_iter_cfg if n_iter_total is None
                                else n_iter_total + n_iter_cfg)
            t_score = time.perf_counter()
            scores = self._score(algo, model, self.X_val_lab)
            t_score_total += time.perf_counter() - t_score
            auc = roc_auc_score(self.y_val_lab, scores)
            print("      cfg={} → AUC-ROC(val)={:.4f}".format(cfg, auc))
            if auc > mejor["auc"]:
                mejor = {"auc": auc, "cfg": cfg, "model": model}

        return (mejor["model"], mejor["cfg"], mejor["auc"], t_fit_total,
                t_score_total, n_iter_total)

    # ------------------------------------------------------------------
    # 3-6. Entrenamiento, umbral y evaluación de un algoritmo
    # ------------------------------------------------------------------
    def _procesar_algoritmo(self, algo):
        """Selecciona config, fija el umbral en D1_val y evalúa sobre D2."""
        print("-" * 70)
        print("ALGORITMO: {}".format(algo))
        print("-" * 70)
        # perf_counter (no time.time) en TODAS las medidas de duración: en Windows
        # time.time() tiene ~15,6 ms de resolución y cuantiza los tiempos cortos.
        # Es monótono y sin época: solo vale para diferencias (la columna 'fecha'
        # la sigue dando datetime.now() dentro de guardar_metricas).
        t0 = time.perf_counter()

        model, cfg, auc_val, t_entrenamiento, t_score_seleccion, n_iter_total = \
            self._seleccionar_config(algo)
        print("   Config ganadora: {} (AUC-ROC val={:.4f})".format(cfg, auc_val))

        # 5. Umbral = percentil 95 del score sobre D1_val (solo normal). Igual para
        # los 4 algoritmos. Nunca se mira D2 para esto.
        # Este scoring es el tramo (3) del bloque y se cronometra aparte: junto con
        # 'tiempo_score_seleccion_s' cierra POR MEDIDA el residual de 'tiempo_s'.
        t_score_umbral = time.perf_counter()
        scores_val = self._score(algo, model, self.X_D1_val)
        t_score_umbral = time.perf_counter() - t_score_umbral
        umbral = float(np.percentile(scores_val, self.PERCENTIL_UMBRAL))
        print("   Umbral (p{} sobre D1_val) = {:.6f}".format(self.PERCENTIL_UMBRAL, umbral))

        # 6. Evaluación sobre D2 (binaria). y_pred = score > umbral.
        # El scoring de D2 es la INFERENCIA del detector: se cronometra aparte
        # del entrenamiento para poder reportar latencia por flujo.
        t_inf = time.perf_counter()
        score_D2 = self._score(algo, model, self.X_D2)
        t_inferencia = time.perf_counter() - t_inf
        y_pred = (score_D2 > umbral).astype(int)
        metricas = evaluacion.evaluar_binario(self.y_bin, y_pred, score_D2)

        print("   D2 → AUC-ROC={:.4f} AUC-PR={:.4f} F1={:.4f} recall={:.4f} FPR={:.4f}".format(
            metricas.get("roc_auc", float("nan")), metricas.get("pr_auc", float("nan")),
            metricas["f1"], metricas["recall"], metricas["fpr"]))

        # Matriz de confusión 2x2 con etiquetas legibles (normal/ataque).
        etiquetas = ["normal", "ataque"]
        y_true_str = np.where(self.y_bin == config.CLASE_ATAQUE, "ataque", "normal")
        y_pred_str = np.where(y_pred == config.CLASE_ATAQUE, "ataque", "normal")
        evaluacion.plot_matriz_confusion(
            y_true_str, y_pred_str, labels=etiquetas,
            titulo="Matriz de confusión — {} (anomalías, D2)".format(algo),
            filename="anomalias_cm_{}_{}.png".format(algo, self.sufijo_artefactos),
        )

        # Épocas (solo Autoencoder; None en los otros tres). DOS cifras y no una:
        # las del ganador dicen si ESE ajuste se cortó por max_iter, y las del
        # grid entero son el denominador COHERENTE con el que
        # 'tiempo_entrenamiento_s' —que suma los fit de TODAS las
        # configuraciones— da segundos por época dentro de ESTA corrida. Ese
        # cociente NO separa causas: si el recuento de épocas es igual entre dos
        # corridas, el cociente es el numerador reescalado y absorbe la carga de
        # máquina entera.
        n_iter_ganador = self._iteraciones_ajuste(algo, model)
        if n_iter_ganador is not None:
            print("   Épocas: {} el ajuste ganador · {} todo el grid (max_iter="
                  "300 · early_stopping=True). 'tiempo_entrenamiento_s' cubre el "
                  "grid entero: dividir por la segunda cifra, no por la primera. "
                  "El s/época resultante vale DENTRO de esta corrida: no decide "
                  "si una diferencia de tiempo es de épocas o de carga de "
                  "máquina".format(n_iter_ganador, n_iter_total))

        self.resultados[algo] = {
            "modelo": model,
            "config_ganadora": cfg,
            "n_iter_ganador": n_iter_ganador,
            "n_iter_total_grid": n_iter_total,
            "auc_val": auc_val,
            "umbral": umbral,
            "score_D2": score_D2,
            "metricas": metricas,
            # OJO: 'tiempo_s' es el bloque COMPLETO del algoritmo, no el ajuste.
            # Dentro del cronómetro (t0 arriba) caen, por orden: los fit del grid
            # (= t_entrenamiento) · el scoring del set etiquetado UNA VEZ POR
            # CONFIGURACIÓN (18.469 filas × 6/9/4/2 configs = t_score_seleccion) ·
            # el scoring de D1_val del umbral (= t_score_umbral) · la inferencia
            # sobre D2 (= t_inferencia) · y la cola de evaluar_binario + UNA
            # figura. Una parte apreciable de 'tiempo_s' no es ni ajuste ni
            # inferencia, y ese resto ya NO se estima: los dos tramos grandes se
            # publican medidos y la cola sale por resta. Va declarado en el dato
            # con config.ALCANCE_TIEMPO_S_BLOQUE_ANOMALIAS.
            "tiempo_s": time.perf_counter() - t0,
            "tiempo_entrenamiento_s": t_entrenamiento,
            "tiempo_score_seleccion_s": t_score_seleccion,
            "tiempo_score_umbral_s": t_score_umbral,
            "tiempo_inferencia_s": t_inferencia,
        }
        print("   Tiempo: {:.1f}s total (entrenamiento {:.1f}s · scoring de la "
              "selección {:.2f}s · scoring del umbral {:.2f}s · inferencia D2 "
              "{:.2f}s)".format(self.resultados[algo]["tiempo_s"],
                                t_entrenamiento, t_score_seleccion,
                                t_score_umbral, t_inferencia))

    # ------------------------------------------------------------------
    # 7. Persistencia (modelos joblib, CSV de métricas, figuras ROC/PR)
    # ------------------------------------------------------------------
    def _fila_metricas(self, algo, r):
        """
        Fila de la tabla de métricas para un algoritmo. Empieza por el conjunto
        mínimo obligatorio: 'algoritmo' + 'alcance' (aquí, binario
        normal-vs-ataque sobre D2 completo) + variante + nº de features. La
        procedencia (semilla, commit, fecha) la inyecta guardar_metricas.

        OJO con las dos columnas que NO son de D2 (config.ALCANCE_SELECCION):
          - 'auc_val': AUC-ROC sobre D1_val + muestra de D3, la cifra con la que
            se ELIGIÓ la configuración. Es sistemáticamente más alta que el
            'roc_auc' de D2 (IsolationForest 54: 0,9918 frente a 0,9229), así que
            citarla como "AUC del detector" sería citar un número del train.
          - 'umbral': percentil 95 del score sobre D1_val, también del train.
        El sufijo '_val' y la columna 'umbral' están declarados en
        config.ALCANCE_SUFIJOS / ALCANCE_COLUMNAS con ese alcance propio.

        'n_iter_ganador' y 'n_iter_total_grid' tampoco son métricas de D2: son
        ÉPOCAS, y solo las rellena el Autoencoder (celda vacía en los otros tres;
        ver _iteraciones_ajuste). No son la misma cifra ni son intercambiables: la
        primera cuenta el ajuste GANADOR —dice si se cortó por max_iter— y la
        segunda suma TODO el grid, que es el conjunto de fit que cronometra
        'tiempo_entrenamiento_s' y por tanto el denominador COHERENTE con el que
        ese tiempo se lee en segundos por época DENTRO de una corrida. Ojo con lo
        que ese cociente NO hace: si el recuento de épocas coincide entre dos
        corridas, el denominador es constante, el cociente es el numerador
        reescalado y NO decide si una diferencia de tiempo entre dos filas es de
        épocas o de carga de máquina. Ver config.ALCANCE_N_ITER_TOTAL.

        Y con 'tiempo_s': aquí mide el BLOQUE COMPLETO del algoritmo —los fit del
        grid + el scoring del set etiquetado en CADA config + el scoring de D1_val
        del umbral + la inferencia sobre D2 + una figura—. No es una suma de dos columnas: una parte
        apreciable de este número no aparece ni en 'tiempo_entrenamiento_s' ni en
        'tiempo_inferencia_s'. Ese resto ya no hay que estimarlo:
        sus dos tramos grandes se publican MEDIDOS en 'tiempo_score_seleccion_s'
        (el scoring repetido de la selección) y 'tiempo_score_umbral_s' (el
        scoring de D1_val), y la cola de métricas + figura es lo que queda al
        restar las cuatro columnas de tiempo. En metricas_baseline.csv la
        columna homónima es solo el GridSearchCV y en metricas_hibrido.csv es el
        tramo de la carga de los splits al cierre de la fila. Por eso cada fila
        lleva 'alcance_tiempo_s': el dato dice qué mide.
        """
        m = r["metricas"]
        fila = {
            "algoritmo": algo,
            "alcance": config.ALCANCE_ANOMALIAS,
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "config_ganadora": str(r["config_ganadora"]),
            # Épocas del ajuste. Solo las rellena el Autoencoder (MLPRegressor
            # con early_stopping=True y max_iter=300); en IsolationForest /
            # OneClassSVM / LocalOutlierFactor las celdas van VACÍAS porque el
            # atributo no aplica —ver _iteraciones_ajuste()—, y nunca a 0: un 0
            # se leería como "cero iteraciones".
            # Las DOS columnas son necesarias: 'n_iter_ganador' es de UNA config
            # y 'n_iter_total_grid' de todas, que es lo que cronometra
            # 'tiempo_entrenamiento_s'. Dividir ese tiempo por la primera daría
            # un número sin significado.
            "n_iter_ganador": (float("nan") if r["n_iter_ganador"] is None
                               else r["n_iter_ganador"]),
            "n_iter_total_grid": (float("nan") if r["n_iter_total_grid"] is None
                                  else r["n_iter_total_grid"]),
            "auc_val": round(r["auc_val"], 6),
            "umbral": round(r["umbral"], 6),
            "roc_auc": round(m.get("roc_auc", float("nan")), 6),
            "pr_auc": round(m.get("pr_auc", float("nan")), 6),
            "precision": round(m["precision"], 6),
            "recall": round(m["recall"], 6),
            "f1": round(m["f1"], 6),
            "accuracy": round(m["accuracy"], 6),
            "fpr": round(m["fpr"], 6),
            "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"],
            "tiempo_s": round(r["tiempo_s"], 2),
            # Tramos (2) y (3) del bloque, MEDIDOS (no estimados): con ellos y con
            # 'tiempo_entrenamiento_s'/'tiempo_inferencia_s' el residual de
            # 'tiempo_s' queda cerrado por resta desde el propio CSV.
            "tiempo_score_seleccion_s": round(r["tiempo_score_seleccion_s"], 3),
            "tiempo_score_umbral_s": round(r["tiempo_score_umbral_s"], 3),
            "alcance_tiempo_s": config.ALCANCE_TIEMPO_S_BLOQUE_ANOMALIAS,
        }
        # Tiempos separados: entrenamiento = fits del grid; inferencia =
        # scoring de D2 completo, que es lo que mide latencia y caudal.
        fila.update(evaluacion.metricas_tiempo(
            r["tiempo_entrenamiento_s"], r["tiempo_inferencia_s"], len(self.X_D2)
        ))
        return fila

    def _persistir(self):
        """Guarda modelos, la tabla de métricas y las curvas ROC/PR superpuestas."""
        # Con la semilla 42, la tabla publicada de siempre; con cualquier otra,
        # config.ruta_tabla() desvía a 'metricas_anomalias_semillas.csv' y esta
        # corrida no abre la publicada.
        csv_path = config.ruta_tabla("metricas_anomalias.csv")

        filas = {algo: self._fila_metricas(algo, r)
                 for algo, r in self.resultados.items()}

        # Idempotencia por variante, con la función única de evaluacion.py: el CSV es único y acumulado, pero al re-ejecutar una
        # variante (54 o 122) se borran primero sus filas previas para no
        # acumular duplicados. La otra variante se conserva intacta.
        evaluacion.limpiar_variante_csv(
            csv_path, self.set_features,
            evaluacion.cabecera_esperada(next(iter(filas.values())))
            if filas else None,
        )

        for algo, r in self.resultados.items():
            # Un joblib por algoritmo, sufijado por variante de features para
            # que 54 y 122 coexistan sin pisarse. Todo lo necesario para desplegar.
            ruta_modelo = config.MODELOS_DIR + r"\anomalia_{}_{}.joblib".format(
                algo, self.sufijo_artefactos)
            joblib.dump(
                {
                    "algoritmo": algo,
                    "modelo": r["modelo"],
                    "umbral": r["umbral"],
                    "config_ganadora": r["config_ganadora"],
                    # Épocas del ajuste (Autoencoder; None en los otros tres):
                    # las del ganador y las del grid entero, más el score de la
                    # validación interna del early_stopping si el estimador lo
                    # expone. Ese último no viaja al CSV: el .joblib no tiene
                    # esquema fijo y no ensucia ninguna tabla.
                    "n_iter_ganador": r["n_iter_ganador"],
                    "n_iter_total_grid": r["n_iter_total_grid"],
                    "best_validation_score": getattr(
                        r["modelo"], "best_validation_score_", None),
                    "base_path_usado": self.base_path,
                    "set_features": self.set_features,
                    "n_features": self.n_features,
                    "percentil_umbral": self.PERCENTIL_UMBRAL,
                    "semilla": config.RANDOM_STATE,
                    "commit": config.commit_actual(),
                },
                ruta_modelo,
            )

            # Una fila por algoritmo en la tabla acumulada (→ tabla 5.1.2).
            evaluacion.guardar_metricas(filas[algo], csv_path)
            print("   Guardado modelo: {}".format(ruta_modelo))

        # La clave de unicidad (variante × algoritmo × alcance) y el recuento de
        # filas de la variante (4 detectores → 8 con las dos variantes) se
        # verifican, no se suponen.
        evaluacion.comprobar_unicidad(csv_path)
        evaluacion.comprobar_recuento(csv_path, self.set_features)

        # Figura estrella de 5.1: ROC/PR de los 4 algoritmos superpuestas.
        scores_por_algo = {algo: r["score_D2"] for algo, r in self.resultados.items()}
        nombre_fig = "anomalias_{}".format(self.sufijo_artefactos)
        evaluacion.plot_roc_pr(
            scores_por_algo, self.y_bin,
            nombre_fig=nombre_fig,
            titulo="Detección de anomalías (D2, {} features)".format(self.set_features),
        )
        print("   Tabla de métricas: {}".format(csv_path))
        print("   Figuras ROC/PR: {0}_roc.png / {0}_pr.png".format(nombre_fig))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def entrenar_todos(self):
        """Ejecuta el pipeline completo para los cuatro algoritmos."""
        config.setup_utf8()
        config.ensure_dirs()
        self.cargar_datos()
        for algo in self.ALGORITMOS:
            self._procesar_algoritmo(algo)
        self._persistir()
        print("=" * 70)
        print("ANOMALÍAS COMPLETADO ({} features)".format(self.set_features))
        print("=" * 70)
        return self.resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Etapa 1 del H-NIDS: detección de anomalías (4 algoritmos "
                    "comparados bajo protocolo idéntico sobre NSL-KDD)."
    )
    parser.add_argument(
        "--sin-seleccion", action="store_true",
        help="Usa el set de 122 features (variante sin selección 4.3.5) en lugar "
             "de las 54 por defecto. Para el experimento 54-vs-122.",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SEMILLA_POR_DEFECTO,
        help=config.AYUDA_CLI_SEMILLA,
    )
    args = parser.parse_args()

    # ANTES de instanciar: la clase congela en su __init__ el sufijo de los
    # artefactos (y otras clases, su StratifiedKFold). Ver config.py, encabezado.
    config.fijar_semilla(args.semilla)

    trainer = NSLKDDAnomalyTrainer(sin_seleccion=args.sin_seleccion)
    trainer.entrenar_todos()
