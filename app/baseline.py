# -*- coding: utf-8 -*-
"""
baseline.py — Baseline de control del H-NIDS: Random Forest MONOLÍTICO.

NO es una etapa del sistema híbrido. Es el clasificador de referencia que la
literatura NSL-KDD usa por defecto: un ÚNICO RandomForest supervisado entrenado
sobre TODO el train (D1 normal + D3 ataques = 5 clases: normal/dos/probe/r2l/u2r)
y evaluado sobre TODO D2. Su papel es responder en el capítulo de resultados a la
pregunta del tribunal: "¿para qué el híbrido si un RF da ~99%?".

La respuesta —y la tesis del TFG— es el 0-day: un RF supervisado solo puede predecir
clases que vio en el train; los 17 tipos de ataque ausentes del train no los conoce.
(Los 17 tipos están CONTADOS sobre `Resultados/metricas_hibrido_0day.csv` —17 valores
distintos de la columna `tipo`, 3.750 filas de D2—, no citados: la cifra 14 que da
Tavallaee et al. se refiere a KDD'99, no a NSL-KDD.)
Su única salida ante un 0-day es clasificarlo como una clase conocida (posiblemente
'normal'). Por eso la MÉTRICA DE DECISIÓN frente al híbrido es el **recall 0-day por
tipo** (misma tabla que la del híbrido): fracción de cada tipo nuevo que el RF marca
como ataque (pred != 'normal'). Se contrasta con el recall 0-day de la etapa de
anomalías; la hipótesis es que el del RF monolítico es sensiblemente inferior.

Reglas de protocolo (invalidan el TFG si se rompen):
  * D2 es INTOCABLE: no se usa para elegir hiperparámetros (todo por CV sobre el train).
    D2 solo para la evaluación final.
  * UNA SOLA semilla en TODO (StratifiedKFold, RandomForest, GridSearchCV), y es
    `config.RANDOM_STATE`: 42 por defecto —la de todo lo publicado— y la que diga
    `--semilla` en el barrido de semillas. Ni un literal de semilla en el código.
  * class_weight='balanced' fijo (no se cruza con el eje de balanceo); mismo
    criterio que la selección 4.3.5.

El set de features es parametrizable (--sin-seleccion → 122 features en lugar de
las 54 por defecto). Nada de rutas hardcodeadas: todo sale de config.

Semilla (--semilla N): POR DEFECTO ES LA 42 y entonces se usan los nombres
publicados (mismo nombre de .joblib y de figura, mismas dos tablas). Con otra semilla el .joblib y la figura se sufijan '_semilla<N>' y las
métricas van a 'metricas_baseline_semillas.csv' y
'metricas_baseline_0day_semillas.csv'. El mecanismo está en el encabezado de
config.py; las diez semillas del barrido, en config.SEMILLAS_BARRIDO (la 42 NO
está entre ellas).
"""
import argparse
import time

# Backend no interactivo fijado ANTES de cualquier import que arrastre matplotlib
# (evaluacion/program). Así nada bloquea ni intenta abrir ventanas.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import config
import evaluacion  # importado antes que program: fija el backend Agg de matplotlib
from program import load_specialized_splits


class NSLKDDBaselineTrainer:
    """
    Entrena el RandomForest monolítico de control.

    Mismo estilo clase-por-script que NSLKDDPreprocessor / NSLKDDAnomalyTrainer /
    NSLKDDSignatureTrainer: una sola clase que orquesta carga, GridSearchCV,
    evaluación sobre D2 (multiclase + binaria + 0-day) y persistencia.
    """

    # Grid RF pequeño, idéntico al de firmas.py (comparabilidad de protocolo).
    GRID_RF = {
        "n_estimators": [100, 300],
        "max_depth": [10, None],
    }

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
        self.set_features = "122_sin_seleccion" if sin_seleccion else "54"
        # Token para los NOMBRES de artefacto (joblib y figura). Con la semilla 42
        # es igual a set_features —los nombres publicados no cambian—; con otra
        # añade '_semilla<N>'. Se congela aquí: config.fijar_semilla() se
        # llama antes de instanciar la clase.
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)

        # Protocolo CV único (mismo split, misma semilla) que firmas.py.
        self.cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config.RANDOM_STATE
        )

        # Rellenados por cargar_datos()
        self.X_train = None
        self.y_train = None          # category (5 clases: normal + 4 ataques)
        self.X_D2 = None
        self.y_D2_cat = None         # category_original de D2 (5 clases)
        self.y_D2_tipo = None        # attack_original de D2 (tipo específico)
        self.tipos_conocidos = None  # tipos de ataque vistos en el train (D3)
        self.tipos_0day = None       # tipos de D2 ataque ausentes del train
        self.n_features = None

        # Rellenados por entrenar() / evaluar()
        self.modelo = None
        self.config_ganadora = None
        self.f1_macro_cv = None
        self.metricas_mc = None      # multiclase 5 clases
        self.metricas_bin = None     # binaria normal vs ataque
        self.metricas_0day = None    # recall 0-day por tipo
        self.tiempo_s = None             # = tiempo de entrenamiento (GridSearchCV)
        self.tiempo_inferencia_s = None  # predict sobre D2 completo

    # ------------------------------------------------------------------
    # 1. Carga: train = D1+D3 (5 clases), evaluación = D2 completo
    # ------------------------------------------------------------------
    def cargar_datos(self):
        """Construye el train monolítico (D1+D3) y prepara la evaluación sobre D2."""
        print("=" * 70)
        print("CARGA DE DATOS (set de features: {})".format(self.set_features))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError(
                "No se pudieron cargar las divisiones desde: " + self.base_path
            )

        # --- Train MONOLÍTICO: D1 (normal) + D3 (ataques). Etiqueta = categoría. ---
        X_D1 = splits["D1"]["X"]
        X_D3 = splits["D3"]["X"]
        cat_D1 = splits["D1"]["y_category"]["category_original"].astype(str)
        cat_D3 = splits["D3"]["y_category"]["category_original"].astype(str)

        # Alineación de columnas: D1 y D3 comparten el mismo vocabulario tras el
        # preprocesado (paso 0), pero se reindexa D3 al orden de D1 por seguridad.
        X_D3 = X_D3[X_D1.columns]
        self.X_train = pd.concat([X_D1, X_D3], axis=0, ignore_index=True)
        self.y_train = pd.concat([cat_D1, cat_D3], axis=0, ignore_index=True).values
        self.n_features = self.X_train.shape[1]

        # Tipos ESPECÍFICOS de ataque vistos en el train (para marcar los 0-day de D2).
        tipos_D3 = splits["D3"]["y_attack"]["attack_original"].astype(str)
        self.tipos_conocidos = set(tipos_D3.unique())

        # --- Evaluación: D2 COMPLETO (a diferencia de firmas, aquí no se filtra). ---
        self.X_D2 = splits["D2"]["X"][X_D1.columns]
        self.y_D2_cat = splits["D2"]["y_category"]["category_original"].astype(str).values
        self.y_D2_tipo = splits["D2"]["y_attack"]["attack_original"].astype(str).values

        # 0-day = tipos de ataque presentes en D2 y AUSENTES del train.
        es_ataque_D2 = (self.y_D2_cat != config.ETIQUETA_NORMAL)
        tipos_D2_ataque = set(pd.Series(self.y_D2_tipo)[es_ataque_D2].unique())
        self.tipos_0day = sorted(tipos_D2_ataque - self.tipos_conocidos)

        vals, cnts = np.unique(self.y_train, return_counts=True)
        print("   Train monolítico (D1+D3): {} filas · {} features".format(
            len(self.X_train), self.n_features))
        print("   Distribución train (5 clases): {}".format(
            dict(zip(vals.tolist(), cnts.tolist()))))
        print("   D2 evaluación (completo): {} filas ({} ataques / {} normales)".format(
            len(self.X_D2), int(es_ataque_D2.sum()), int((~es_ataque_D2).sum())))
        print("   Tipos de ataque conocidos (train): {}".format(len(self.tipos_conocidos)))
        print("   Tipos 0-day en D2 (ausentes del train): {} tipos: {}".format(
            len(self.tipos_0day), self.tipos_0day))

    # ------------------------------------------------------------------
    # 2. Entrenamiento: GridSearchCV del RF monolítico
    # ------------------------------------------------------------------
    def entrenar(self):
        """GridSearchCV(RF, f1_macro) sobre el train completo; refit en todo el train."""
        print("=" * 70)
        print("GRIDSEARCH RF MONOLÍTICO — f1_macro en CV sobre el train (D1+D3)")
        print("=" * 70)
        # perf_counter (no time.time) en TODAS las medidas de duración: en Windows
        # time.time() tiene ~15,6 ms de resolución y cuantiza los tiempos cortos.
        # Monótono y sin época: solo para diferencias; la columna 'fecha' la sigue
        # dando datetime.now() dentro de guardar_metricas.
        t0 = time.perf_counter()

        estimador = RandomForestClassifier(
            random_state=config.RANDOM_STATE, n_jobs=-1, class_weight="balanced",
        )
        busqueda = GridSearchCV(
            estimator=estimador,
            param_grid=self.GRID_RF,
            scoring="f1_macro",
            cv=self.cv,
            n_jobs=-1,
            refit=True,
        )
        busqueda.fit(self.X_train, self.y_train)

        self.modelo = busqueda.best_estimator_
        self.config_ganadora = busqueda.best_params_
        self.f1_macro_cv = float(busqueda.best_score_)
        # 'tiempo_s' de esta tabla = SOLO el entrenamiento (a diferencia del de
        # anomalias/firmas, que es el bloque completo). Declarado en el dato con
        # config.ALCANCE_TIEMPO_S_SOLO_ENTRENAMIENTO.
        self.tiempo_s = time.perf_counter() - t0
        print("   Mejor config: {} · f1_macro(CV)={:.4f}".format(
            self.config_ganadora, self.f1_macro_cv))
        print("   Tiempo entrenamiento: {:.1f}s".format(self.tiempo_s))

    # ------------------------------------------------------------------
    # 3. Evaluación sobre D2: multiclase (5), binaria y recall 0-day por tipo
    # ------------------------------------------------------------------
    def evaluar(self):
        """Evalúa el RF sobre D2 completo: multiclase, binaria y recall 0-day."""
        print("=" * 70)
        print("EVALUACIÓN SOBRE D2 COMPLETO")
        print("=" * 70)

        # El predict sobre D2 completo se cronometra aparte del entrenamiento:
        # es la inferencia del baseline y de ella salen latencia por flujo
        # y flujos/segundo, comparables con las del híbrido.
        t_inf = time.perf_counter()
        y_pred = self.modelo.predict(self.X_D2)
        self.tiempo_inferencia_s = time.perf_counter() - t_inf
        print("   Inferencia sobre D2: {:.2f}s ({} flujos)".format(
            self.tiempo_inferencia_s, len(self.X_D2)))

        # (a) Multiclase 5 clases (→ comparativa con el híbrido en 5.3). labels FIJOS.
        self.metricas_mc = evaluacion.evaluar_multiclase(
            self.y_D2_cat, y_pred, labels=config.CATEGORIAS_MULTICLASE
        )
        print("   Multiclase (5) → f1_macro={:.4f} recall_macro={:.4f} acc={:.4f}".format(
            self.metricas_mc["f1_macro"], self.metricas_mc["recall_macro"],
            self.metricas_mc["accuracy"]))

        # (b) Binaria normal(0) vs ataque(1). y_score = 1 - P(normal) para AUC.
        y_bin_true = (self.y_D2_cat != config.ETIQUETA_NORMAL).astype(int)
        y_bin_pred = (y_pred != config.ETIQUETA_NORMAL).astype(int)
        proba = self.modelo.predict_proba(self.X_D2)
        idx_normal = list(self.modelo.classes_).index(config.ETIQUETA_NORMAL)
        score_ataque = 1.0 - proba[:, idx_normal]
        self.metricas_bin = evaluacion.evaluar_binario(
            y_bin_true, y_bin_pred, y_score=score_ataque
        )
        print("   Binaria → recall={:.4f} precision={:.4f} F1={:.4f} FPR={:.4f} "
              "AUC-ROC={:.4f}".format(
                  self.metricas_bin["recall"], self.metricas_bin["precision"],
                  self.metricas_bin["f1"], self.metricas_bin["fpr"],
                  self.metricas_bin.get("roc_auc", float("nan"))))

        # (c) Recall 0-day por tipo (MÉTRICA DE DECISIÓN vs híbrido). Un RF supervisado
        # "detecta" un 0-day solo si lo predice como ALGÚN ataque (pred != 'normal').
        es_sospechoso = (y_pred != config.ETIQUETA_NORMAL)
        self.metricas_0day = evaluacion.evaluar_0day_por_tipo(
            self.y_D2_tipo, es_sospechoso, self.tipos_0day
        )
        glob = self.metricas_0day["__global__"]
        print("   Recall 0-day GLOBAL (pred!=normal): {}/{} = {:.4f}".format(
            glob["detectados"], glob["n"], glob["recall"]))

        # Matriz de confusión 5×5 (→ 5.3), sufijada por variante de features.
        evaluacion.plot_matriz_confusion(
            self.y_D2_cat, y_pred, labels=config.CATEGORIAS_MULTICLASE,
            titulo="Matriz de confusión — RF monolítico (baseline, D2 completo)",
            filename="baseline_cm_{}.png".format(self.sufijo_artefactos),
        )

    # ------------------------------------------------------------------
    # 4. Persistencia (modelo, métricas, tabla 0-day por tipo)
    # ------------------------------------------------------------------
    def _fila_metricas(self):
        """
        Fila única de la tabla del baseline. Empieza por el conjunto mínimo
        obligatorio: 'algoritmo' + 'alcance' — aquí, 5 clases sobre D2
        COMPLETO, que es lo que distingue este `accuracy_D2` del homónimo de
        metricas_firmas.csv (4 clases sobre los ataques de tipo conocido). Los
        bloques prefijados siguen la convención de config.ALCANCE_PREFIJOS:
        'bin_' = binario normal-vs-ataque sobre D2 completo, 'recall_0day' =
        ataques de D2 de tipo ausente del train.

        OJO con 'f1_macro_cv': NO es de D2. Es el f1_macro medio en CV sobre el
        train D1+D3 con el que se eligió la configuración (sufijo '_cv' →
        config.ALCANCE_SELECCION). La brecha es enorme —0,9094 en CV frente a
        0,4721 de 'f1_macro' sobre D2 en la variante de 54— y ahí está justamente
        la tesis del TFG: el RF monolítico brilla en lo que ya vio. Citarla como
        rendimiento del baseline sería citar 44 pp de más bajo una fila cuyo
        'alcance' declara "sobre D2 completo".
        """
        m = self.metricas_mc
        b = self.metricas_bin
        fila = {
            "algoritmo": "RandomForest_monolitico",
            "alcance": config.ALCANCE_BASELINE,
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "config_ganadora": str(self.config_ganadora),
            "f1_macro_cv": round(self.f1_macro_cv, 6),
            # Multiclase (5 clases) sobre D2 completo.
            "accuracy_D2": round(m["accuracy"], 6),
            "precision_macro": round(m["precision_macro"], 6),
            "recall_macro": round(m["recall_macro"], 6),
            "f1_macro": round(m["f1_macro"], 6),
            "f1_weighted": round(m["f1_weighted"], 6),
            # Binaria normal vs ataque. 'bin_accuracy' es la cifra con la que la
            # literatura NSL-KDD compara (los baselines canónicos de Tavallaee et
            # al. y el 0,8605 del híbrido): sin ella esta tabla no sería comparable.
            "bin_accuracy": round(b["accuracy"], 6),
            "bin_precision": round(b["precision"], 6),
            "bin_recall": round(b["recall"], 6),
            "bin_f1": round(b["f1"], 6),
            "bin_fpr": round(b["fpr"], 6),
            "bin_roc_auc": round(b.get("roc_auc", float("nan")), 6),
            "bin_pr_auc": round(b.get("pr_auc", float("nan")), 6),
            "bin_tn": b["tn"], "bin_fp": b["fp"], "bin_fn": b["fn"], "bin_tp": b["tp"],
            # Recall 0-day global (métrica de decisión vs híbrido).
            "recall_0day_global": round(self.metricas_0day["__global__"]["recall"], 6),
            "n_0day": int(self.metricas_0day["__global__"]["n"]),
            "n_test": int(len(self.y_D2_cat)),
            "tiempo_s": round(self.tiempo_s, 2),
            # Esta tabla es la excepción: su 'tiempo_s' NO es el bloque completo
            # como en anomalias/firmas, sino solo el GridSearchCV — de hecho
            # coincide con 'tiempo_entrenamiento_s' de la misma fila. El dato lo
            # declara para que nadie lo compare con las otras tablas.
            "alcance_tiempo_s": config.ALCANCE_TIEMPO_S_SOLO_ENTRENAMIENTO,
        }
        # Tiempos separados: entrenamiento = GridSearchCV + refit sobre
        # D1+D3; inferencia = predict sobre D2 completo.
        fila.update(evaluacion.metricas_tiempo(
            self.tiempo_s, self.tiempo_inferencia_s, len(self.X_D2)
        ))
        # Métricas por categoría (5 clases) para el detalle del capítulo 5.
        for clase in config.CATEGORIAS_MULTICLASE:
            pc = m["por_clase"].get(clase, {})
            fila["precision_" + clase] = round(pc.get("precision", float("nan")), 6)
            fila["recall_" + clase] = round(pc.get("recall", float("nan")), 6)
            fila["f1_" + clase] = round(pc.get("f1", float("nan")), 6)
            fila["soporte_" + clase] = int(pc.get("soporte", 0))
        return fila

    def _persistir(self):
        """Guarda el modelo, la fila de métricas y la tabla 0-day por tipo."""
        # --- Modelo joblib (sufijado por variante para que 54 y 122 coexistan). ---
        ruta_modelo = config.MODELOS_DIR + r"\baseline_rf_{}.joblib".format(
            self.sufijo_artefactos)
        joblib.dump(
            {
                "algoritmo": "RandomForest_monolitico",
                "modelo": self.modelo,
                "config_ganadora": self.config_ganadora,
                "f1_macro_cv": self.f1_macro_cv,
                "base_path_usado": self.base_path,
                "set_features": self.set_features,
                "sin_seleccion": bool(self.sin_seleccion),
                "n_features": self.n_features,
                "feature_names": list(self.X_train.columns),
                "clases": config.CATEGORIAS_MULTICLASE,
                "tipos_conocidos": sorted(self.tipos_conocidos),
                "tipos_0day": self.tipos_0day,
                "semilla": config.RANDOM_STATE,
                "commit": config.commit_actual(),
            },
            ruta_modelo,
        )
        print("   Guardado modelo: {}".format(ruta_modelo))

        # --- Tabla de métricas (una fila; → capítulo 5, comparativa con el híbrido). ---
        # Con la semilla 42, la tabla publicada; con otra, la del barrido de semillas.
        csv_path = config.ruta_tabla("metricas_baseline.csv")
        fila = self._fila_metricas()
        evaluacion.limpiar_variante_csv(
            csv_path, self.set_features, evaluacion.cabecera_esperada(fila)
        )
        evaluacion.guardar_metricas(fila, csv_path)
        evaluacion.comprobar_unicidad(csv_path)
        evaluacion.comprobar_recuento(csv_path, self.set_features)
        print("   Tabla de métricas (baseline): {}".format(csv_path))

        # --- Tabla 0-day por tipo (misma forma que la del híbrido; → 5.3). ---
        # Tabla AUXILIAR (fila = tipo de ataque): fuera del conjunto mínimo de las
        # principales, cumple evaluacion.COLUMNAS_MINIMAS_AUXILIARES, con
        # 'alcance' explícito porque su columna 'recall' se refiere a una
        # partición distinta de la de la tabla principal.
        csv_0day = config.ruta_tabla("metricas_baseline_0day.csv")
        filas_0day = []
        for tipo in list(self.tipos_0day) + ["__global__"]:
            d = self.metricas_0day[tipo]
            filas_0day.append({
                "algoritmo": "RandomForest_monolitico",
                "alcance": config.ALCANCE_0DAY,
                "set_features": self.set_features,
                "sin_seleccion": bool(self.sin_seleccion),
                "n_features": self.n_features,
                "tipo_0day": tipo,
                "n": d["n"],
                "detectados": d["detectados"],
                "recall": round(d["recall"], 6) if d["n"] > 0 else float("nan"),
            })
        evaluacion.limpiar_variante_csv(
            csv_0day, self.set_features,
            evaluacion.cabecera_esperada(filas_0day[0]) if filas_0day else None,
        )
        for fila_0day in filas_0day:
            evaluacion.guardar_metricas(fila_0day, csv_0day)
        print("   Tabla 0-day por tipo (baseline): {}".format(csv_0day))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def entrenar_todos(self):
        """Ejecuta el pipeline completo del baseline monolítico."""
        config.setup_utf8()
        config.ensure_dirs()
        self.cargar_datos()
        self.entrenar()
        self.evaluar()
        self._persistir()
        print("=" * 70)
        print("BASELINE RF MONOLÍTICO COMPLETADO ({} features)".format(self.set_features))
        print("=" * 70)
        return {
            "config_ganadora": self.config_ganadora,
            "f1_macro_cv": self.f1_macro_cv,
            "metricas_mc": self.metricas_mc,
            "metricas_bin": self.metricas_bin,
            "metricas_0day": self.metricas_0day,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline de control del H-NIDS: RandomForest monolítico "
                    "(un único clasificador supervisado sobre todo el train, "
                    "evaluado sobre D2 completo)."
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

    # ANTES de instanciar: el __init__ congela el StratifiedKFold del protocolo y
    # el sufijo de los artefactos. Ver config.py, encabezado.
    config.fijar_semilla(args.semilla)

    trainer = NSLKDDBaselineTrainer(sin_seleccion=args.sin_seleccion)
    trainer.entrenar_todos()
