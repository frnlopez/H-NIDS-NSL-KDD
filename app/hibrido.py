# -*- coding: utf-8 -*-
"""
hibrido.py — Sistema híbrido H-NIDS: cascada anomalías → firmas (sección 5.3).

Combina las dos etapas ya entrenadas y persistidas por `anomalias.py` y `firmas.py`
y las evalúa de extremo a extremo sobre D2. NO re-entrena: carga los `.joblib`
persistidos (una sola fuente de verdad; las métricas cuadran con 5.1/5.2). Lo único
que reconstruye es el estimador de firmas —desde su config guardada— para obtener
probabilidades OUT-OF-FOLD sobre D3 y calibrar con ellas `UMBRAL_CONF` sin leakage.

Cascada:
  etapa 1  score = anomaly_score(detector, X_D2);  es_sospechoso = score > umbral
  etapa 2  sobre los sospechosos: proba = firma.predict_proba(X);  categoría = argmax
           si max(proba) < UMBRAL_CONF → 'unknown' (ataque desconocido / 0-day)
  no sospechoso → 'normal'
  Detector por defecto = Autoencoder (mejor en 5.1); firma por defecto = RandomForest
  (mejor en 54 y con predict_proba suave: base fina para un umbral de confianza).

Reglas de protocolo (invalidan la evaluación si se rompen):
  * LÍNEA ROJA: `UMBRAL_CONF` se SELECCIONA sobre probabilidades OOF de D3
    (nunca sobre D2). La función de calibración ni siquiera recibe D2 en su firma.
    D2 se usa SOLO para reportar (tabla de sensibilidad). Elegir el umbral por su
    resultado en D2 sería leakage.
  * D1 no se carga: el detector y su umbral llegan ya entrenados del joblib.
  * UNA SOLA semilla en toda la corrida —`config.RANDOM_STATE`, que es 42 por defecto
    y la que diga `--semilla` en el barrido de semillas—: mismo StratifiedKFold(5)
    que firmas.py para el OOF y ningún literal de semilla en el código.

Semilla (--semilla N): POR DEFECTO ES LA 42 y entonces la cascada carga
exactamente los .joblib publicados y escribe en las tres tablas publicadas. Con
otra semilla carga los '_semilla<N>' que dejaron anomalias.py y firmas.py en ese
mismo pase —y ABORTA si el joblib declara otra semilla, para no mezclar dos
corridas—, sufija su descriptor y su figura, y escribe en
'metricas_hibrido*_semillas.csv'. Ojo con lo que arrastra la semilla aquí: cambia
el StratifiedKFold del OOF, así que UMBRAL_CONF se recalibra y puede salir
distinto. El mecanismo está en el encabezado de config.py; las diez semillas del
barrido, en config.SEMILLAS_BARRIDO (la 42 NO está entre ellas).
"""
# --- Qué significa 'tiempo_entrenamiento_s' EN ESTA TABLA ------------------------
# El nombre de la columna se CONSERVA porque el conjunto mínimo de columnas es
# homogéneo entre las cuatro tablas y renombrarlo lo rompería; pero aquí no mide
# lo mismo que en las otras tres.
SIGNIFICADO_TIEMPO_ENTRENAMIENTO = (
    "En metricas_hibrido.csv 'tiempo_entrenamiento_s' NO es tiempo de ajuste de "
    "la cascada: el híbrido carga los .joblib de las dos etapas y no las "
    "re-entrena. Ese tiempo es EXCLUSIVAMENTE "
    "la calibración de UMBRAL_CONF por cross_val_predict sobre D3 "
    "(probabilidades out-of-fold): cero ajuste del detector de anomalías, cero "
    "ajuste del clasificador de firmas y cero uso de D2. Es la única excepción a "
    "«el híbrido no re-entrena», y ni siquiera altera los modelos persistidos: "
    "reconstruye una copia del estimador de firmas desde su config guardada."
)

import argparse
import time

# Backend no interactivo fijado ANTES de cualquier import que arrastre matplotlib
# (evaluacion/anomalias/firmas/program). Así nada bloquea ni abre ventanas.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
import evaluacion  # importado pronto: reafirma el backend Agg de matplotlib
from anomalias import NSLKDDAnomalyTrainer
from firmas import NSLKDDSignatureTrainer
from program import load_specialized_splits


class NSLKDDHybridEvaluator:
    """
    Evalúa el sistema híbrido en cascada sobre D2 y calibra `UMBRAL_CONF` con OOF de D3.

    Mismo estilo clase-por-script que el resto del pipeline. No re-entrena modelos:
    orquesta carga de joblibs, calibración sin leakage, cascada y reporte por alcance.
    """

    DETECTORES = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "Autoencoder"]
    FIRMAS = ["DecisionTree", "RandomForest", "KNN", "HistGradientBoosting"]

    # Rango a explorar y presupuesto de caída de recall macro OOF.
    UMBRALES_CONF = [0.4, 0.5, 0.6]
    TAU = 0.02  # 2 pp: caída máxima tolerada del recall macro OOF de clases conocidas.

    def __init__(self, sin_seleccion=False, detector="Autoencoder", firma="RandomForest"):
        if detector not in self.DETECTORES:
            raise ValueError("Detector desconocido: " + str(detector))
        if firma not in self.FIRMAS:
            raise ValueError("Firma desconocida: " + str(firma))

        self.sin_seleccion = sin_seleccion
        self.detector = detector
        self.firma = firma
        self.base_path = config.base_path(sin_seleccion=sin_seleccion)
        self.set_features = "122_sin_seleccion" if sin_seleccion else "54"
        # Token para los NOMBRES de artefacto: los .joblib que CARGA (etapas 1 y 2),
        # el descriptor que escribe y la figura 5x6. Con la semilla 42 es igual a
        # set_features, así que el híbrido publicado sigue cargando exactamente los
        # mismos ficheros; con otra semilla apunta a los '_semilla<N>' que dejaron
        # anomalias.py y firmas.py en ese mismo pase del barrido de semillas.
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)

        # Reutilizamos los builders de las etapas: scoring de anomalías y
        # reconstrucción del estimador de firmas. No se reimplementa nada.
        self._anom = NSLKDDAnomalyTrainer(sin_seleccion=sin_seleccion)
        self._sig = NSLKDDSignatureTrainer(sin_seleccion=sin_seleccion)

        # Mismo CV que firmas.py: StratifiedKFold(5, shuffle) con la semilla de
        # la corrida —42 por defecto, otra con --semilla en el barrido de semillas—.
        self.cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config.RANDOM_STATE
        )

        # Rellenados por cargar_datos()
        self.X_D2 = None
        self.cat_D2 = None          # category_original de D2 (5 clases reales)
        self.tipo_D2 = None         # attack_original de D2 (tipo específico)
        self.y_bin = None           # binaria D2: 1=ataque, 0=normal
        self.X_D3 = None
        self.y_D3 = None            # category_original de D3 (4 clases de ataque)
        self.tipos_conocidos = None
        self.tipos_0day = None
        self.mask_ataque = None
        self.mask_tipo_conocido = None
        self.n_features = None

        # Rellenados por la cascada y la calibración
        self.score_cascada = None   # anomaly score del detector de la cascada sobre D2
        self.es_sospechoso = None   # etapa 1 (bool) — ES la binaria del híbrido
        self.proba_susp = None      # predict_proba de firma sobre los sospechosos
        self.clases_firma = None    # orden de clases de la firma de la cascada
        self.umbral_conf = None     # elegido por la regla de presupuesto
        self.tabla_calibracion = []  # filas OOF+D2 de los 3 umbrales
        self.tabla_0day = []        # recall 0-day por tipo de los 4 detectores
        self.metricas_run = None    # fila resumen de la corrida
        # Tiempos separados. Aquí "entrenamiento" es SOLO la calibración OOF
        # sobre D3: ver SIGNIFICADO_TIEMPO_ENTRENAMIENTO (módulo) para el texto
        # inequívoco que va a la memoria.
        self.tiempo_calibracion_s = None
        self.tiempo_inferencia_s = None

    # ------------------------------------------------------------------
    # 1. Carga (solo D2 y D3; D1 no hace falta)
    # ------------------------------------------------------------------
    def cargar_datos(self):
        print("=" * 70)
        print("HÍBRIDO — carga (set: {} · detector: {} · firma: {})".format(
            self.set_features, self.detector, self.firma))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError("No se pudieron cargar las divisiones: " + self.base_path)

        self.X_D2 = splits["D2"]["X"]
        self.cat_D2 = splits["D2"]["y_category"]["category_original"].astype(str).values
        self.tipo_D2 = splits["D2"]["y_attack"]["attack_original"].astype(str).values
        self.y_bin = (self.cat_D2 != config.ETIQUETA_NORMAL).astype(int)
        self.n_features = self.X_D2.shape[1]

        self.X_D3 = splits["D3"]["X"]
        self.y_D3 = splits["D3"]["y_category"]["category_original"].astype(str).values
        tipos_D3 = splits["D3"]["y_attack"]["attack_original"].astype(str)
        self.tipos_conocidos = set(tipos_D3.unique())

        # Máscaras de D2 y tipos 0-day = ataques de D2 cuyo tipo NO está en el train.
        self.mask_ataque = (self.cat_D2 != config.ETIQUETA_NORMAL)
        self.mask_tipo_conocido = np.isin(self.tipo_D2, list(self.tipos_conocidos))
        self.tipos_0day = sorted(
            set(self.tipo_D2[self.mask_ataque & ~self.mask_tipo_conocido])
        )

        print("   D2: {} filas ({} ataques / {} normales) · {} features".format(
            len(self.X_D2), int(self.y_bin.sum()), int((self.y_bin == 0).sum()),
            self.n_features))
        print("   D3: {} filas · categorías {}".format(
            len(self.X_D3), sorted(set(self.y_D3))))
        print("   Tipos 0-day en D2 (ausentes del train): {} → {}".format(
            len(self.tipos_0day), self.tipos_0day))

    # ------------------------------------------------------------------
    # Carga de joblibs persistidos con verificación de variante
    # ------------------------------------------------------------------
    def _cargar_joblib(self, prefijo, algo):
        ruta = config.MODELOS_DIR + r"\{}_{}_{}.joblib".format(
            prefijo, algo, self.sufijo_artefactos)
        # Si el .joblib no está, el traceback crudo de joblib.load() no dice ni qué
        # etapa falta ni qué script la produce. Se relanza como RuntimeError con el
        # mismo tono que cascada_invertida._leer_umbral_conf: qué fichero falta, qué
        # hay que correr antes y —si el nombre lleva la marca de semilla— con qué
        # semilla. Sigue ABORTANDO: no se sustituye ni se continúa a medias.
        try:
            datos = joblib.load(ruta)
        except FileNotFoundError:
            script_previo = "anomalias.py" if prefijo == "anomalia" else "firmas.py"
            # El nombre de la variante no basta para reproducir la corrida que falta:
            # la de 122 features se pide con --sin-seleccion y la de 54 es la de por
            # defecto (sin flag). Se nombra el flag literal, igual que --semilla N.
            flag_variante = " --sin-seleccion" if self.sin_seleccion else ""
            aviso_semilla = ""
            if not config.es_semilla_por_defecto():
                aviso_semilla = (" Este nombre lleva la marca de la semilla en curso "
                                 "({}), así que hay que correrlo con --semilla {}.".format(
                                     config.RANDOM_STATE, config.RANDOM_STATE))
            # 'from None' corta el encadenado del FileNotFoundError crudo: el objetivo
            # es leer ESTE mensaje, no el traceback de joblib.load() delante de él.
            raise RuntimeError(
                "No existe {}: la cascada carga las dos etapas ya entrenadas y no "
                "entrena ninguna. Ejecuta primero {}{} para la variante '{}'.{}".format(
                    ruta, script_previo, flag_variante, self.set_features,
                    aviso_semilla)) from None
        # Salvaguarda: el joblib debe ser de la misma variante de features (54/122)
        # que estamos evaluando, o las columnas no cuadrarían con X_D2.
        if str(datos.get("set_features")) != str(self.set_features):
            raise RuntimeError(
                "El joblib {} es de la variante '{}' pero se esperaba '{}'".format(
                    ruta, datos.get("set_features"), self.set_features))
        # Salvaguarda equivalente para la SEMILLA: una cascada que mezclase la
        # etapa 1 de una semilla con la etapa 2 de otra —o con las de la 42— no
        # mediría la dispersión de nada, y el nombre del fichero no basta para
        # detectarlo si alguien renombra a mano. Los scripts de modelos persisten
        # 'semilla' en todos los .joblib; si faltase (artefacto antiguo) se avisa
        # y se continúa, no se aborta.
        semilla_joblib = datos.get("semilla")
        if semilla_joblib is None:
            print("   [aviso] {} no declara 'semilla': no se puede comprobar que "
                  "sea de esta corrida.".format(ruta))
        elif int(semilla_joblib) != int(config.RANDOM_STATE):
            raise RuntimeError(
                "El joblib {} se entrenó con la semilla {} y esta corrida usa la "
                "{}: la cascada mezclaría dos corridas distintas.".format(
                    ruta, semilla_joblib, config.RANDOM_STATE))
        return datos, ruta

    # ------------------------------------------------------------------
    # 2. Calibración de UMBRAL_CONF sobre OOF de D3.
    #    LÍNEA ROJA: esta función NO recibe D2. Solo D3.
    # ------------------------------------------------------------------
    def _calibrar_umbral_conf(self, joblib_firma):
        """
        Reconstruye el estimador de firmas desde su config, obtiene probabilidades
        OOF sobre D3 (cross_val_predict, mismo CV que firmas.py) y selecciona
        `UMBRAL_CONF` por la regla de presupuesto τ=2pp. Devuelve
        (umbral_elegido, prob_oof_por_umbral) — todo derivado SOLO de D3.
        """
        print("-" * 70)
        print("CALIBRACIÓN de UMBRAL_CONF sobre OOF de D3 (sin tocar D2)")
        print("-" * 70)

        algo = joblib_firma["algoritmo"]
        balanceo = joblib_firma["balanceo"]
        cfg = joblib_firma["config_ganadora"]
        # Reconstrucción fiel: mismo builder + set_params (claves ya con 'clf__'
        # si SMOTE, como las persiste firmas.py::_entrenar_final).
        est = self._sig._estimador_para(algo, balanceo)
        est.set_params(**cfg)

        # Probabilidades out-of-fold: cada muestra de D3 puntuada por un modelo que
        # NO la vio en su fold de entrenamiento (SMOTE se aplica dentro del fold vía
        # ImbPipeline → sin leakage). Columnas en orden alfabético de clases.
        proba_oof = cross_val_predict(
            est, self.X_D3, self.y_D3, cv=self.cv, method="predict_proba", n_jobs=-1
        )
        clases_oof = np.unique(self.y_D3)  # orden que usa cross_val_predict
        pred_oof = clases_oof[proba_oof.argmax(axis=1)]
        conf_oof = proba_oof.max(axis=1)

        # Baseline sin umbral (nada pasa a 'unknown'): recall macro de las 4 conocidas.
        recall_baseline = evaluacion.evaluar_multiclase(
            self.y_D3, pred_oof, labels=config.CATEGORIAS_ATAQUE
        )["recall_macro"]
        print("   Recall macro OOF (sin umbral, baseline) = {:.4f}".format(recall_baseline))

        oof_por_umbral = {}
        candidatos = []
        for u in self.UMBRALES_CONF:
            pred_u = pred_oof.copy().astype(object)
            baja_conf = conf_oof < u
            pred_u[baja_conf] = config.ETIQUETA_UNKNOWN
            recall_u = evaluacion.evaluar_multiclase(
                self.y_D3, pred_u, labels=config.CATEGORIAS_ATAQUE
            )["recall_macro"]
            tasa_falsos_unknown = float(baja_conf.mean())  # D3 son todos conocidos
            caida = recall_baseline - recall_u
            oof_por_umbral[u] = {
                "recall_macro_oof": recall_u,
                "tasa_falsos_unknown_oof": tasa_falsos_unknown,
                "caida_vs_baseline": caida,
            }
            dentro = caida <= self.TAU
            if dentro:
                candidatos.append(u)
            print("   umbral={:.1f} → recall_macro_oof={:.4f} (caída {:.4f}) "
                  "· falsos_unknown={:.4f} {}".format(
                      u, recall_u, caida, tasa_falsos_unknown,
                      "[dentro de τ]" if dentro else "[fuera de τ]"))

        # Regla de presupuesto: el umbral MÁS ALTO cuya caída ≤ τ. Si ninguno
        # cabe, el más conservador (el más bajo).
        elegido = max(candidatos) if candidatos else min(self.UMBRALES_CONF)
        print("   → UMBRAL_CONF elegido (τ={:.2f}) = {:.1f}".format(self.TAU, elegido))

        for u in self.UMBRALES_CONF:
            oof_por_umbral[u]["recall_macro_baseline_oof"] = recall_baseline
        return elegido, oof_por_umbral

    # ------------------------------------------------------------------
    # 3. Cascada sobre D2. Score de etapa 1 una sola vez.
    # ------------------------------------------------------------------
    def _preparar_cascada(self, joblib_det, joblib_firma):
        """Calcula el score de la etapa 1 y la proba de firma sobre los sospechosos.
        Ambos son independientes de UMBRAL_CONF, así que se computan una única vez.

        Este bloque ES la inferencia de extremo a extremo del híbrido sobre D2
        (etapa 1 + etapa 2 sobre los sospechosos), así que se cronometra aparte
        para reportar latencia por flujo y flujos/segundo."""
        # perf_counter (no time.time) en TODAS las medidas de duración: en Windows
        # time.time() tiene ~15,6 ms de resolución y cuantizaba los tiempos cortos.
        # Monótono y sin época: solo para diferencias; 'fecha' la da datetime.now().
        t_inf = time.perf_counter()

        # Etapa 1: anomaly score y sospechosos (reutiliza _score de anomalias.py).
        model_det = joblib_det["modelo"]
        umbral_det = joblib_det["umbral"]
        self.score_cascada = self._anom._score(self.detector, model_det, self.X_D2)
        self.es_sospechoso = self.score_cascada > umbral_det  # binaria del híbrido

        # Etapa 2: predict_proba de la firma SOLO sobre los sospechosos.
        model_firma = joblib_firma["modelo"]
        self.clases_firma = np.asarray(model_firma.classes_)
        Xs = self.X_D2.loc[self.es_sospechoso]
        if len(Xs) > 0:
            self.proba_susp = model_firma.predict_proba(Xs)
        else:
            self.proba_susp = np.empty((0, len(self.clases_firma)))

        self.tiempo_inferencia_s = time.perf_counter() - t_inf
        print("   Etapa 1: {} sospechosos / {} ({} normales dejados pasar)".format(
            int(self.es_sospechoso.sum()), len(self.X_D2),
            int((~self.es_sospechoso).sum())))
        print("   Inferencia cascada completa sobre D2: {:.2f}s".format(
            self.tiempo_inferencia_s))

    def _ensamblar_prediccion(self, umbral_conf):
        """Construye la predicción final sobre TODO D2 para un UMBRAL_CONF dado.
        No sospechoso → 'normal'; sospechoso → categoría argmax, o 'unknown' si la
        confianza máxima < umbral_conf. Barato: solo re-umbraliza la proba cacheada."""
        pred = np.full(len(self.X_D2), config.ETIQUETA_NORMAL, dtype=object)
        if self.proba_susp.shape[0] > 0:
            conf = self.proba_susp.max(axis=1)
            cat_susp = self.clases_firma[self.proba_susp.argmax(axis=1)].astype(object)
            cat_susp[conf < umbral_conf] = config.ETIQUETA_UNKNOWN
            pred[self.es_sospechoso] = cat_susp
        return pred

    # ------------------------------------------------------------------
    # 4. Evaluación sobre D2 por alcance para un UMBRAL_CONF dado
    # ------------------------------------------------------------------
    def _metricas_d2(self, pred_final):
        """Métricas del híbrido sobre D2, cada una con su alcance. Sin número
        único engañoso. Devuelve un dict con: binaria (=etapa 1), por-categoría-
        conocida (coste del filtro de anomalías), f1_macro 5 clases conservador
        (unknown = error) y el desglose 0-day (routing a unknown)."""
        # (a) Binaria = etapa 1 por construcción: reutiliza es_sospechoso.
        binaria = evaluacion.evaluar_binario(
            self.y_bin, self.es_sospechoso.astype(int), self.score_cascada
        )

        # (b) Por categoría conocida: end-to-end sobre ataques de tipo conocido de D2.
        # Mide el coste de anteponer el filtro de anomalías frente a firmas-solo (5.2).
        mask_conocida = self.mask_ataque & self.mask_tipo_conocido
        conocida = evaluacion.evaluar_multiclase(
            self.cat_D2[mask_conocida], pred_final[mask_conocida],
            labels=config.CATEGORIAS_ATAQUE,
        )

        # (c) f1_macro a 5 clases CONSERVADOR: los 'unknown' cuentan como error
        # (no están en labels). Infravalora al híbrido a propósito; aun así la tesis
        # se sostiene por el 0-day. Es el número comparable con baseline.py.
        conservador = evaluacion.evaluar_multiclase(
            self.cat_D2, pred_final, labels=config.CATEGORIAS_MULTICLASE,
        )

        # (d) 0-day: recall de "marcado sospechoso" por tipo (etapa 1, indep. del
        # umbral) + de los detectados, % enrutados a 'unknown' vs mal-etiquetados.
        r0 = evaluacion.evaluar_0day_por_tipo(
            self.tipo_D2, self.es_sospechoso, self.tipos_0day
        )
        mask_0day = self.mask_ataque & ~self.mask_tipo_conocido
        det_0day = mask_0day & self.es_sospechoso
        n_det_0day = int(det_0day.sum())
        pred_0day_det = pred_final[det_0day]
        pct_unknown = (float((pred_0day_det == config.ETIQUETA_UNKNOWN).mean())
                       if n_det_0day > 0 else float("nan"))

        return {
            "binaria": binaria,
            "conocida": conocida,
            "conservador": conservador,
            "recall_0day": r0,
            "pct_unknown_entre_0day_detectados": pct_unknown,
            "n_conocida": int(mask_conocida.sum()),
        }

    # ------------------------------------------------------------------
    # 5. Tabla 0-day de los 4 detectores, con su FPR
    # ------------------------------------------------------------------
    def _tabla_0day_cuatro_detectores(self):
        """Para CADA uno de los 4 detectores: carga su joblib, puntúa D2 con su
        umbral guardado y calcula el recall 0-day por tipo + su FPR binario. El
        recall 0-day depende SOLO del detector y su umbral (no de la firma), así que
        la comparación 54 frente a 122 deja de depender de elegir IF vs AE."""
        print("-" * 70)
        print("RECALL 0-DAY DE LOS 4 DETECTORES — con FPR")
        print("-" * 70)
        for det in self.DETECTORES:
            try:
                jd, _ = self._cargar_joblib("anomalia", det)
            except (FileNotFoundError, RuntimeError) as e:
                # Un joblib ausente o de otra variante (54/122) no debe abortar
                # toda la tabla; se omite ese detector con aviso y se sigue con el resto.
                # OJO: este except también absorbe el abort por SEMILLA distinta
                # (ver _cargar_joblib), que en el resto del script sí es fatal: la
                # tabla queda corta con aviso; la cascada principal sí aborta.
                print("   [aviso] detector {} no disponible ({}); se omite.".format(det, e))
                continue
            score = self._anom._score(det, jd["modelo"], self.X_D2)
            es_sosp = score > jd["umbral"]
            fpr = evaluacion.evaluar_binario(self.y_bin, es_sosp.astype(int))["fpr"]
            r0 = evaluacion.evaluar_0day_por_tipo(self.tipo_D2, es_sosp, self.tipos_0day)
            es_cascada = (det == self.detector)
            print("   {:<20} recall_0day_global={:.4f} · FPR={:.4f}{}".format(
                det, r0["__global__"]["recall"], fpr,
                "  ← detector de la cascada" if es_cascada else ""))
            for tipo, d in r0.items():
                self.tabla_0day.append({
                    "alcance": config.ALCANCE_0DAY,
                    "set_features": self.set_features,
                    # sin_seleccion y n_features permiten filtrar por variante
                    # sin parsear texto.
                    "sin_seleccion": bool(self.sin_seleccion),
                    "n_features": self.n_features,
                    "detector": det,
                    "es_cascada": bool(es_cascada),
                    "fpr_detector": round(fpr, 6),
                    "tipo": tipo,
                    "n": d["n"],
                    "detectados": d["detectados"],
                    "recall": round(d["recall"], 6) if d["recall"] == d["recall"] else d["recall"],
                })

    # ------------------------------------------------------------------
    # 6. Figura: matriz de confusión 5×6
    # ------------------------------------------------------------------
    def _plot_matriz_5x6(self, pred_final):
        """Matriz filas=5 clases reales × columnas=6 (con 'unknown'). No usa el
        helper cuadrado de evaluacion: aquí filas≠columnas por construcción."""
        from sklearn.metrics import confusion_matrix
        cm6 = confusion_matrix(self.cat_D2, pred_final, labels=config.CATEGORIAS_HIBRIDO)
        # La fila 'unknown' (índice 5) es todo ceros (no hay 'unknown' real) → se quita.
        idx_unknown = config.CATEGORIAS_HIBRIDO.index(config.ETIQUETA_UNKNOWN)
        cm = np.delete(cm6, idx_unknown, axis=0)
        filas = config.CATEGORIAS_MULTICLASE      # 5 reales
        cols = config.CATEGORIAS_HIBRIDO          # 6 predichas

        fig, ax = plt.subplots(figsize=(1.5 * len(cols) + 2, 1.4 * len(filas) + 2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                    xticklabels=cols, yticklabels=filas, ax=ax)
        ax.set_xlabel("Predicción (híbrido)")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusión — híbrido {}→{} (D2, {} features)".format(
            self.detector, self.firma, self.set_features))
        fig.tight_layout()
        ruta = config.FIGURAS_DIR + r"\hibrido_cm_{}.png".format(
            self.sufijo_artefactos)
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("   Matriz 5×6: {}".format(ruta))

    # ------------------------------------------------------------------
    # 7. Persistencia: 3 CSV idempotentes + figura + descriptor joblib
    # ------------------------------------------------------------------
    def _persistir(self, ruta_det, ruta_firma):
        # Idempotencia por variante con la función única de evaluacion.py.
        # (1) Tabla de calibración: sensibilidad de los 3 umbrales (OOF + D2).
        # Las tres rutas por config.ruta_tabla(): con la semilla 42 son las tablas
        # publicadas y con cualquier otra las '*_semillas.csv' del barrido.
        csv_cal = config.ruta_tabla("metricas_hibrido_calibracion.csv")
        evaluacion.limpiar_variante_csv(
            csv_cal, self.set_features,
            evaluacion.cabecera_esperada(self.tabla_calibracion[0])
            if self.tabla_calibracion else None,
        )
        for fila in self.tabla_calibracion:
            evaluacion.guardar_metricas(fila, csv_cal)
        print("   Tabla de calibración: {}".format(csv_cal))

        # (2) Tabla 0-day por tipo de los 4 detectores, con FPR.
        csv_0day = config.ruta_tabla("metricas_hibrido_0day.csv")
        evaluacion.limpiar_variante_csv(
            csv_0day, self.set_features,
            evaluacion.cabecera_esperada(self.tabla_0day[0])
            if self.tabla_0day else None,
        )
        for fila in self.tabla_0day:
            evaluacion.guardar_metricas(fila, csv_0day)
        print("   Tabla 0-day (4 detectores): {}".format(csv_0day))

        # (3) Descriptor reproducible: referencias + umbral + τ. NO re-serializa
        # los modelos (ya existen en sus joblibs); guarda qué usa y el umbral elegido.
        #
        # VA ANTES DE LA FILA RESUMEN, y el orden importa. Si se escribiera la fila
        # y después el descriptor, un corte del proceso entre ambos dejaría la fila
        # escrita y el descriptor sin volcar. En el barrido reanudable eso es un
        # bloqueo: barrido_semillas.ya_hecho() da el paso por hecho porque la fila
        # está, pero cascada_invertida._leer_umbral_conf() aborta porque el
        # descriptor —de donde sale UMBRAL_CONF— no existe. Volcando primero el
        # descriptor, un corte ahí lo deja huérfano, ya_hecho() devuelve False y la
        # reanudación reejecuta el paso, que lo sobrescribe. Es el mismo orden que
        # siguen anomalias.py, firmas.py y baseline.py (joblib antes de CSV).
        ruta_desc = config.MODELOS_DIR + r"\hibrido_{}.joblib".format(
            self.sufijo_artefactos)
        joblib.dump({
            "tipo": "descriptor_hibrido",
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "detector": self.detector,
            "firma": self.firma,
            "joblib_detector": ruta_det,
            "joblib_firma": ruta_firma,
            "umbral_conf_elegido": self.umbral_conf,
            "umbrales_explorados": self.UMBRALES_CONF,
            "tau": self.TAU,
            "tipos_0day": self.tipos_0day,
            "base_path_usado": self.base_path,
            "semilla": config.RANDOM_STATE,
            "commit": config.commit_actual(),
        }, ruta_desc)
        print("   Descriptor del híbrido: {}".format(ruta_desc))

        # (4) Fila resumen de la corrida (métricas por alcance con el umbral elegido).
        # ÚLTIMA a propósito: es la que barrido_semillas.ya_hecho() consulta para dar
        # el paso por hecho, así que se escribe cuando ya está en disco todo lo que
        # los pasos siguientes necesitan (ver el comentario del bloque (3)).
        csv_res = config.ruta_tabla("metricas_hibrido.csv")
        evaluacion.limpiar_variante_csv(
            csv_res, self.set_features,
            evaluacion.cabecera_esperada(self.metricas_run),
        )
        evaluacion.guardar_metricas(self.metricas_run, csv_res)
        evaluacion.comprobar_unicidad(csv_res)
        evaluacion.comprobar_recuento(csv_res, self.set_features)
        print("   Tabla resumen: {}".format(csv_res))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def evaluar_todo(self):
        config.setup_utf8()
        config.ensure_dirs()
        # t0 arranca ANTES de cargar datos y joblibs: la carga de los splits SÍ
        # entra en el 'tiempo_s' de esta tabla. El cronómetro se LEE más abajo al
        # construir metricas_run, así que lo posterior (figura 5x6, tabla 0-day de
        # los cuatro detectores y escritura de CSV) queda FUERA: es lo que declara
        # config.ALCANCE_TIEMPO_S_CARGA_A_CIERRE_FILA, y no es un tiempo de ajuste.
        # perf_counter por resolución (ver los otros scripts).
        t0 = time.perf_counter()
        self.cargar_datos()

        # Cargar los joblibs de la cascada.
        joblib_det, ruta_det = self._cargar_joblib("anomalia", self.detector)
        joblib_firma, ruta_firma = self._cargar_joblib("firma", self.firma)

        # Calibración SOLO con D3 (la función no recibe D2). Es el único
        # ajuste que hace el híbrido, así que su tiempo es el que se reporta como
        # "entrenamiento"; la cascada en sí no entrena nada.
        t_cal = time.perf_counter()
        self.umbral_conf, oof_por_umbral = self._calibrar_umbral_conf(joblib_firma)
        self.tiempo_calibracion_s = time.perf_counter() - t_cal

        # Preparar cascada sobre D2 (score etapa 1 + proba etapa 2), una sola vez.
        print("-" * 70)
        print("CASCADA SOBRE D2")
        print("-" * 70)
        self._preparar_cascada(joblib_det, joblib_firma)

        # Tabla de sensibilidad: los 3 umbrales, cada uno con OOF (ya calculado) + D2.
        for u in self.UMBRALES_CONF:
            pred_u = self._ensamblar_prediccion(u)
            m = self._metricas_d2(pred_u)
            oof = oof_por_umbral[u]
            self.tabla_calibracion.append({
                # Fila MIXTA por diseño: 'oof_*' es D3 out-of-fold (de ahí sale la
                # elección del umbral) y 'd2_*' es D2 completo (solo reporte). El
                # 'alcance' lo dice para que no se lea como una sola medida.
                "alcance": config.ALCANCE_HIBRIDO_CALIBRACION,
                "set_features": self.set_features,
                "sin_seleccion": bool(self.sin_seleccion),
                "n_features": self.n_features,
                "detector": self.detector,
                "firma": self.firma,
                "umbral_conf": u,
                "elegido": bool(u == self.umbral_conf),
                "oof_recall_macro_conocidas": round(oof["recall_macro_oof"], 6),
                "oof_recall_macro_baseline": round(oof["recall_macro_baseline_oof"], 6),
                "oof_tasa_falsos_unknown": round(oof["tasa_falsos_unknown_oof"], 6),
                "d2_f1_macro_5_conservador": round(m["conservador"]["f1_macro"], 6),
                "d2_recall_0day_global": round(m["recall_0day"]["__global__"]["recall"], 6),
                "d2_pct_unknown_entre_0day_det": (
                    round(m["pct_unknown_entre_0day_detectados"], 6)
                    if m["pct_unknown_entre_0day_detectados"] ==
                    m["pct_unknown_entre_0day_detectados"] else float("nan")),
            })

        # Predicción final con el umbral ELEGIDO → métricas resumen + figura 5×6.
        pred_final = self._ensamblar_prediccion(self.umbral_conf)
        m = self._metricas_d2(pred_final)
        b = m["binaria"]
        c = m["conocida"]
        cons = m["conservador"]
        r0g = m["recall_0day"]["__global__"]
        print("-" * 70)
        print("RESUMEN (umbral_conf={:.1f})".format(self.umbral_conf))
        print("   Binaria (=etapa 1): recall={:.4f} FPR={:.4f} f1={:.4f} AUC-ROC={:.4f}".format(
            b["recall"], b["fpr"], b["f1"], b.get("roc_auc", float("nan"))))
        print("   Por categoría conocida: recall_macro={:.4f} f1_macro={:.4f} (n={})".format(
            c["recall_macro"], c["f1_macro"], m["n_conocida"]))
        print("   f1_macro 5 clases (conservador, unknown=error): {:.4f}".format(cons["f1_macro"]))
        print("   Recall 0-day global (etapa 1): {:.4f} (n={})".format(
            r0g["recall"], r0g["n"]))

        # Fila resumen. Empieza por el conjunto mínimo obligatorio de columnas:
        # 'algoritmo' identifica la cascada concreta (detector->firma) y 'alcance'
        # dice que es la evaluación extremo a extremo sobre D2 completo y advierte
        # de que 'tiempo_entrenamiento_s' aquí es solo la calibración OOF (así el
        # aviso de SIGNIFICADO_TIEMPO_ENTRENAMIENTO viaja también en el DATO, no
        # solo en el código; la columna NO se renombra: el conjunto mínimo es
        # homogéneo entre las cuatro tablas). Los
        # bloques prefijados siguen config.ALCANCE_PREFIJOS: 'bin_' = binario
        # normal-vs-ataque sobre D2 completo (= etapa 1), 'conocida_' =
        # config.ALCANCE_HIBRIDO_CONOCIDA y 'recall_0day' = ataques de tipo
        # ausente del train.
        #
        # 'conocida_*' NO es lo mismo que las columnas homónimas de firmas: se
        # mide de extremo a extremo, así que los ataques que la etapa 1 no marcó
        # entran como 'normal' y la cifra baja (conocida_recall_macro 0,6709
        # frente al recall_macro 0,8496 de firmas-solo en 54). De ahí que tenga
        # alcance propio en lugar de reutilizar el de firmas.
        self.metricas_run = {
            "algoritmo": "{}->{}".format(self.detector, self.firma),
            "alcance": config.ALCANCE_HIBRIDO,
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "detector": self.detector,
            "firma": self.firma,
            "umbral_conf_elegido": self.umbral_conf,
            "tau": self.TAU,
            # binaria (= etapa 1)
            "bin_recall": round(b["recall"], 6),
            "bin_precision": round(b["precision"], 6),
            "bin_f1": round(b["f1"], 6),
            "bin_fpr": round(b["fpr"], 6),
            "bin_accuracy": round(b["accuracy"], 6),
            "bin_roc_auc": round(b.get("roc_auc", float("nan")), 6),
            "bin_pr_auc": round(b.get("pr_auc", float("nan")), 6),
            "bin_tn": b["tn"], "bin_fp": b["fp"], "bin_fn": b["fn"], "bin_tp": b["tp"],
            # por categoría conocida (coste del filtro de anomalías)
            "conocida_precision_macro": round(c["precision_macro"], 6),
            "conocida_recall_macro": round(c["recall_macro"], 6),
            "conocida_f1_macro": round(c["f1_macro"], 6),
            "n_conocida": m["n_conocida"],
            # f1_macro 5 clases conservador (comparable con baseline.py)
            "f1_macro_5_conservador": round(cons["f1_macro"], 6),
            "accuracy_5": round(cons["accuracy"], 6),
            # 0-day (titular) + FPR de la cascada
            "recall_0day_global": round(r0g["recall"], 6),
            "n_0day": r0g["n"],
            # MISMO número que 'bin_fpr' de esta misma fila y no por descuido: la
            # etapa 2 solo reclasifica lo que la etapa 1 marcó y nunca devuelve un
            # flujo a 'normal', así que la cascada no puede tener otro FPR que el
            # de su etapa 1. Las dos columnas se conservan porque responden a dos
            # usos distintos (el bloque binario y el FPR junto al titular de
            # 0-day) y la coincidencia queda DECLARADA en el dato, en
            # config.ALCANCE_HIBRIDO: dos nombres para una misma magnitud no se
            # admiten sin decirlo. Si algún día la etapa 2 pudiera absolver un flujo
            # (devolverlo a 'normal'), este round dejaría de ser b["fpr"] y habría
            # que recalcularlo desde las predicciones finales de la cascada.
            "fpr_cascada": round(b["fpr"], 6),
            "tiempo_s": round(time.perf_counter() - t0, 2),
            # Tercer significado de la columna homónima: aquí es el tramo de
            # hibrido.py que va de la carga de los splits al cierre de ESTA fila
            # (carga de D1/D2/D3 + joblibs + calibración OOF + cascada + tabla de
            # sensibilidad). Las figuras, la tabla 0-day de los cuatro detectores
            # y la escritura de los CSV van después y NO cuentan. Ni bloque de
            # algoritmo (anomalias/firmas) ni entrenamiento (baseline). El dato lo
            # declara.
            "alcance_tiempo_s": config.ALCANCE_TIEMPO_S_CARGA_A_CIERRE_FILA,
        }
        # Tiempos separados: "entrenamiento" = calibración OOF sobre D3 (el
        # híbrido no re-entrena la cascada; ver la constante del módulo
        # SIGNIFICADO_TIEMPO_ENTRENAMIENTO); inferencia = cascada completa sobre
        # los flujos de D2.
        self.metricas_run.update(evaluacion.metricas_tiempo(
            self.tiempo_calibracion_s, self.tiempo_inferencia_s, len(self.X_D2)
        ))

        # Figura 5×6 y tabla 0-day de los 4 detectores.
        self._plot_matriz_5x6(pred_final)
        self._tabla_0day_cuatro_detectores()

        # Persistir todo.
        self._persistir(ruta_det, ruta_firma)
        # Este total NO es el 'tiempo_s' de la fila: se lee aquí, con las figuras,
        # la tabla 0-day y los CSV ya escritos, mientras que la columna se congeló
        # antes de todo eso. Son dos medidas distintas a propósito y se etiquetan
        # como tales para que quien coteje log y CSV no crea que hay un error.
        print("=" * 70)
        print("HÍBRIDO COMPLETADO ({} features) en {:.1f}s de pared".format(
            self.set_features, time.perf_counter() - t0))
        print("   (incluye figuras y escritura de CSV; la columna 'tiempo_s' del "
              "CSV vale {:.2f}s y se cierra ANTES de eso)".format(
                  self.metricas_run["tiempo_s"]))
        print("=" * 70)
        return self.metricas_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema híbrido H-NIDS: cascada anomalías→firmas evaluada sobre D2 "
                    "(sección 5.3). Carga los modelos persistidos; no re-entrena."
    )
    parser.add_argument(
        "--sin-seleccion", action="store_true",
        help="Usa el set de 122 features (variante sin selección 4.3.5) en lugar de "
             "las 54 por defecto. Para el experimento 54-vs-122.",
    )
    parser.add_argument(
        "--detector", choices=NSLKDDHybridEvaluator.DETECTORES, default="Autoencoder",
        help="Detector de la etapa 1 (por defecto Autoencoder, el mejor en 5.1).",
    )
    parser.add_argument(
        "--firma", choices=NSLKDDHybridEvaluator.FIRMAS, default="RandomForest",
        help="Clasificador de la etapa 2 (por defecto RandomForest).",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SEMILLA_POR_DEFECTO,
        help=config.AYUDA_CLI_SEMILLA + " Requiere haber corrido antes anomalias.py "
             "y firmas.py con ESA MISMA semilla: el híbrido carga sus .joblib y "
             "aborta si la semilla que declaran no es la de esta corrida.",
    )
    args = parser.parse_args()

    # ANTES de instanciar: el __init__ congela el StratifiedKFold del OOF, el
    # sufijo de los artefactos y los dos trainers auxiliares. Ver config.py.
    config.fijar_semilla(args.semilla)

    evaluador = NSLKDDHybridEvaluator(
        sin_seleccion=args.sin_seleccion, detector=args.detector, firma=args.firma
    )
    evaluador.evaluar_todo()
