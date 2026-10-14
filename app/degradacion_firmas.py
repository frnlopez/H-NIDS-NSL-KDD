# -*- coding: utf-8 -*-
"""
degradacion_firmas.py — Degradación FUERA DE DOMINIO de la etapa 2.

Qué mide
--------
Cuánto pierde un clasificador de firmas al salir del dominio en el que se eligió:
de la validación cruzada sobre D3 —el train, sin drift— a D2, que es el conjunto
de test del NSL-KDD y trae distribución distinta. Se mide sobre HistGradientBoosting
(el candidato alternativo) y sobre RandomForest (el clasificador PUBLICADO de la etapa 2), que
es lo que convierte la cifra en un contraste y no en un dato suelto.

Tres regímenes por algoritmo, de menos a más lejos del dominio de entrenamiento:

  1. 'oof_d3'               out-of-fold sobre D3 (StratifiedKFold 5 · shuffle · la
                            semilla de la corrida). Es el régimen DENTRO de dominio
                            y el que sirve de referencia para las dos deltas.
  2. 'd2_tipo_conocido'     los ataques de D2 cuyo TIPO específico está en el train
                            (los 0-day fuera). Es EXACTAMENTE el alcance publicado
                            en metricas_firmas.csv: toda la caída que aparezca aquí
                            es drift de distribución sobre tipos ya vistos.
  3. 'd2_todos_los_ataques' TODOS los ataques de D2, incluidos los 17 tipos ausentes
                            del train. La categoría sigue siendo una de las cuatro
                            pero el tipo puede no haberse visto nunca: es el régimen
                            fuera de dominio más duro y NO está publicado en ninguna
                            tabla. Es solo la etapa 2 en aislado — en el sistema real
                            esas filas las filtra antes la etapa 1.

De cada régimen se publica f1_macro y el RECALL POR CLASE (dos/probe/r2l/u2r), que
es donde se ve que la degradación no se reparte por igual: la macro esconde qué
clase se hunde.

Por qué este script SÍ entrena (y por qué no había alternativa)
--------------------------------------------------------------
El régimen (1) necesita predicciones OUT-OF-FOLD y en disco no existen: los .joblib
de firmas.py guardan el modelo REFIT sobre D3 entero, y pedirle predicciones sobre
D3 sería preguntarle por sus propios datos de entrenamiento —un f1 optimista que no
compara con nada—. metricas_firmas.csv publica 'f1_macro_cv', pero es un ESCALAR
(la media de folds del GridSearchCV) y no trae recall por clase, que es justo lo que
se quiere medir. Así que hay 5 `fit` por algoritmo, y se declaran:

  * NO son un re-entrenamiento del sistema: la configuración y el esquema de
    balanceo NO se vuelven a buscar, se LEEN del .joblib persistido
    ('config_ganadora' y 'balanceo') y se reconstruyen con el mismo builder de
    firmas.py (`_estimador_para`). Ni un hiperparámetro nuevo.
  * NO tocan D2: cross_val_predict solo ve D3. Es el mismo precedente que
    `hibrido.py::_calibrar_umbral_conf`, que ya reajusta por fold para obtener OOF
    sobre D3 sin leakage.
  * Los regímenes (2) y (3) NO entrenan nada: usan el modelo persistido tal cual.

D2 SOLO SE REPORTA: de esta tabla no sale ninguna elección, ningún
hiperparámetro y ningún cambio en el clasificador publicado. La comparación
HistGB vs RandomForest es DESCRIPTIVA — que HistGB degrade menos en una columna no
reabre la elección de la etapa 2, que ya está tomada y publicada.

Verificación contra lo publicado
--------------------------------
El régimen 'd2_tipo_conocido' tiene exactamente el alcance de metricas_firmas.csv,
así que sus cifras se cotejan columna a columna contra esa tabla (f1_macro,
recall_macro, accuracy y los cuatro recall por clase). Si no coinciden hasta el
redondeo publicado, el script ABORTA: significaría que el .joblib o los splits no
son los de la corrida publicada y entonces la fila OOF tampoco describiría al
modelo que cita la memoria.

Salida
------
  Resultados/metricas_degradacion_firmas.csv   (algoritmos × 3 regímenes) filas por
                                               variante.

El recuento lo verifica `_comprobar_tabla()` releyendo el CSV escrito (recuento,
unicidad del par algoritmo × régimen y presencia de los tres regímenes por
algoritmo).

Semilla (--semilla N)
---------------------
POR DEFECTO ES LA 42. Con otra semilla lee los .joblib '_semilla<N>' de ese pase del
barrido, usa su StratifiedKFold y escribe en
'metricas_degradacion_firmas_semillas.csv'. Los .joblib se comprueban por variante
Y por semilla y la corrida aborta si declaran otra.
"""
import argparse
import os

# Backend no interactivo ANTES de cualquier import que arrastre matplotlib.
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import joblib

from sklearn.model_selection import cross_val_predict

import config
import evaluacion  # importado antes que program: fija el backend Agg
from firmas import NSLKDDSignatureTrainer
from program import load_specialized_splits


class NSLKDDSignatureDriftMeasurer:
    """
    Mide la caída de la etapa 2 al pasar del train (OOF sobre D3) a D2.

    Una clase por script, como el resto del pipeline. Lee la configuración ya
    elegida de los .joblib, no busca ninguna nueva y no toca D2 para decidir nada.
    """

    # Por defecto, el candidato alternativo y el publicado. Se pueden ampliar por
    # CLI a los cuatro de firmas.ALGORITMOS.
    ALGORITMOS_POR_DEFECTO = ["RandomForest", "HistGradientBoosting"]

    # Régimen de REFERENCIA (dentro de dominio) contra el que se calculan las
    # deltas. No es un adorno: sin él las dos filas de D2 son cifras sueltas.
    REGIMEN_REFERENCIA = "oof_d3"
    REGIMEN_D2_CONOCIDO = "d2_tipo_conocido"
    REGIMEN_D2_TODOS = "d2_todos_los_ataques"

    TABLA = "metricas_degradacion_firmas.csv"
    TABLA_FIRMAS = "metricas_firmas.csv"

    # Tolerancia del cotejo contra metricas_firmas.csv: esa tabla publica sus
    # métricas REDONDEADAS a 6 decimales, así que la diferencia máxima admisible
    # frente al valor recalculado sin redondear es media unidad del último decimal.
    TOLERANCIA_COTEJO = 1e-6

    def __init__(self, sin_seleccion=False, algoritmos=None):
        self.sin_seleccion = sin_seleccion
        self.base_path = config.base_path(sin_seleccion=sin_seleccion)
        self.set_features = "122_sin_seleccion" if sin_seleccion else "54"
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)
        self.algoritmos = list(algoritmos or self.ALGORITMOS_POR_DEFECTO)

        # Constructor de estimadores de firmas.py: se reutiliza para reconstruir el
        # pipeline (balanceo + clasificador) EXACTAMENTE como lo montó la corrida
        # publicada, en vez de copiar el builder aquí. Instanciarlo no carga
        # datos ni entrena nada.
        self._sig = NSLKDDSignatureTrainer(sin_seleccion=sin_seleccion)

        # El CV NO se reconstruye aquí: se TOMA el del propio NSLKDDSignatureTrainer,
        # que es el mismo objeto de protocolo (StratifiedKFold 5 · shuffle · semilla
        # de la corrida) con el que firmas.py eligió la configuración. Rehacerlo con
        # un n_splits escrito a mano sería un literal que podría desviarse del de
        # firmas.py sin que nada lo detectase.
        self.cv = self._sig.cv

        # Rellenados por cargar_datos()
        self.X_D3 = None
        self.y_D3 = None
        self.X_D2_conocido = None
        self.y_D2_conocido = None
        self.X_D2_todos = None
        self.y_D2_todos = None
        self.n_features = None

        self.filas = []

    # ------------------------------------------------------------------
    # 1. Carga: D3 y las dos poblaciones de D2
    # ------------------------------------------------------------------
    def cargar_datos(self):
        """Carga D3 y arma las dos poblaciones de ataque de D2 de los regímenes 2 y 3."""
        print("=" * 70)
        print("DEGRADACIÓN DE LA ETAPA 2 — carga (set de features: {})".format(
            self.set_features))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError(
                "No se pudieron cargar las divisiones desde: " + self.base_path)

        self.X_D3 = splits["D3"]["X"]
        self.y_D3 = splits["D3"]["y_category"]["category_original"].astype(str).values
        self.n_features = self.X_D3.shape[1]
        tipos_conocidos = set(
            splits["D3"]["y_attack"]["attack_original"].astype(str).unique())

        X_D2 = splits["D2"]["X"]
        cat_D2 = splits["D2"]["y_category"]["category_original"].astype(str)
        tipo_D2 = splits["D2"]["y_attack"]["attack_original"].astype(str)

        # Misma construcción que firmas.py::cargar_datos, para que el régimen 2 sea
        # la MISMA población que la tabla publicada y el cotejo tenga sentido.
        es_ataque = (cat_D2 != config.ETIQUETA_NORMAL).values
        tipo_conocido = tipo_D2.isin(tipos_conocidos).values

        mask_conocido = es_ataque & tipo_conocido
        self.X_D2_conocido = X_D2.loc[mask_conocido]
        self.y_D2_conocido = cat_D2.loc[mask_conocido].values

        self.X_D2_todos = X_D2.loc[es_ataque]
        self.y_D2_todos = cat_D2.loc[es_ataque].values

        print("   D3 (train de la etapa 2): {} filas · {} features".format(
            len(self.X_D3), self.n_features))
        print("   D2 ataques de tipo conocido: {} filas (alcance publicado)".format(
            len(self.X_D2_conocido)))
        print("   D2 ataques totales: {} filas ({} de tipo ausente del train)".format(
            len(self.X_D2_todos),
            int((es_ataque & ~tipo_conocido).sum())))

    # ------------------------------------------------------------------
    # 2. Carga del .joblib de firmas
    # ------------------------------------------------------------------
    def _cargar_joblib(self, algo):
        """Carga `firma_<algo>_<set>.joblib` comprobando variante y semilla."""
        ruta = os.path.join(
            config.MODELOS_DIR,
            "firma_{}_{}.joblib".format(algo, self.sufijo_artefactos))
        try:
            datos = joblib.load(ruta)
        except FileNotFoundError:
            flag_variante = " --sin-seleccion" if self.sin_seleccion else ""
            aviso_semilla = ""
            if not config.es_semilla_por_defecto():
                aviso_semilla = (" El nombre lleva la marca de la semilla en curso "
                                 "({0}): hay que correrlo con --semilla {0}.".format(
                                     config.RANDOM_STATE))
            raise RuntimeError(
                "No existe {}: esta medición NO busca hiperparámetros y necesita la "
                "configuración ya elegida. Ejecuta antes firmas.py{} para la "
                "variante '{}'.{}".format(ruta, flag_variante, self.set_features,
                                          aviso_semilla)) from None

        if str(datos.get("set_features")) != str(self.set_features):
            raise RuntimeError(
                "El joblib {} es de la variante '{}' pero se esperaba '{}'".format(
                    ruta, datos.get("set_features"), self.set_features))
        semilla_joblib = datos.get("semilla")
        if semilla_joblib is None:
            print("   [aviso] {} no declara 'semilla': no se puede comprobar que "
                  "sea de esta corrida.".format(ruta))
        elif int(semilla_joblib) != int(config.RANDOM_STATE):
            raise RuntimeError(
                "El joblib {} se entrenó con la semilla {} y esta corrida usa la "
                "{}: se mezclarían dos corridas distintas.".format(
                    ruta, semilla_joblib, config.RANDOM_STATE))
        return datos, ruta

    # ------------------------------------------------------------------
    # 3. Los tres regímenes de un algoritmo
    # ------------------------------------------------------------------
    def _fila(self, algo, datos, regimen, alcance, y_true, y_pred):
        """Arma la fila del CSV de un régimen a partir de sus predicciones."""
        m = evaluacion.evaluar_multiclase(
            y_true, y_pred, labels=config.CATEGORIAS_ATAQUE)
        fila = {
            "alcance": alcance,
            "regimen": regimen,
            "es_referencia": bool(regimen == self.REGIMEN_REFERENCIA),
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "algoritmo": algo,
            # Procedencia de la configuración: NO se ha buscado aquí, se ha leído.
            "balanceo": datos["balanceo"],
            "config_ganadora": str(datos["config_ganadora"]),
            # El escalar que publica metricas_firmas.csv (media de folds del
            # GridSearchCV). Viaja en las tres filas porque es propiedad del
            # algoritmo, no del régimen, y permite situar la fila OOF de aquí:
            # NO son la misma cifra (media de folds frente a agrupado out-of-fold).
            "f1_macro_cv_persistido": round(float(datos["f1_macro_cv"]), 6),
            "n": int(len(y_true)),
            "accuracy": round(m["accuracy"], 6),
            "precision_macro": round(m["precision_macro"], 6),
            "recall_macro": round(m["recall_macro"], 6),
            "f1_macro": round(m["f1_macro"], 6),
            "f1_weighted": round(m["f1_weighted"], 6),
        }
        # Por clase: el reparto de la caída, que la macro esconde. Las cuatro
        # categorías salen del vocabulario de config, no de una lista local.
        for clase in config.CATEGORIAS_ATAQUE:
            d = m["por_clase"][clase]
            fila["precision_" + clase] = round(d["precision"], 6)
            fila["recall_" + clase] = round(d["recall"], 6)
            fila["f1_" + clase] = round(d["f1"], 6)
            fila["soporte_" + clase] = int(d["soporte"])
        return fila

    def _medir_algoritmo(self, algo):
        """Mide los tres regímenes de un algoritmo y devuelve sus tres filas."""
        print("-" * 70)
        print("ALGORITMO: {}".format(algo))
        datos, ruta = self._cargar_joblib(algo)
        print("   .joblib: {} · balanceo={} · config={}".format(
            os.path.basename(ruta), datos["balanceo"], datos["config_ganadora"]))

        # --- Régimen 1: out-of-fold sobre D3. ÚNICO punto con `fit` del script. ---
        est = self._sig._estimador_para(algo, datos["balanceo"])
        est.set_params(**datos["config_ganadora"])
        print("   OOF sobre D3 ({} folds, {} filas): reajuste por fold de la "
              "configuración YA elegida — no se busca ninguna nueva y D2 no "
              "interviene.".format(self.cv.get_n_splits(), len(self.X_D3)))
        pred_oof = cross_val_predict(
            est, self.X_D3, self.y_D3, cv=self.cv, n_jobs=-1)
        filas = [self._fila(algo, datos, self.REGIMEN_REFERENCIA,
                            config.ALCANCE_DEGRADACION_OOF_D3,
                            self.y_D3, pred_oof)]

        # --- Regímenes 2 y 3: el modelo persistido, sin tocar. ---
        modelo = datos["modelo"]
        filas.append(self._fila(
            algo, datos, self.REGIMEN_D2_CONOCIDO, config.ALCANCE_FIRMAS,
            self.y_D2_conocido, modelo.predict(self.X_D2_conocido)))
        filas.append(self._fila(
            algo, datos, self.REGIMEN_D2_TODOS, config.ALCANCE_DEGRADACION_D2_TODOS,
            self.y_D2_todos, modelo.predict(self.X_D2_todos)))

        # Deltas contra la referencia. Se escriben en el dato: calcularlas al citar
        # invita a restar dos filas de regímenes distintos sin darse cuenta.
        ref = filas[0]
        for f in filas:
            f["delta_f1_macro_vs_oof"] = round(
                f["f1_macro"] - ref["f1_macro"], 6)
            f["delta_recall_macro_vs_oof"] = round(
                f["recall_macro"] - ref["recall_macro"], 6)
            for clase in config.CATEGORIAS_ATAQUE:
                f["delta_recall_" + clase + "_vs_oof"] = round(
                    f["recall_" + clase] - ref["recall_" + clase], 6)

        for f in filas:
            print("   {:<22} n={:>6} · f1_macro={:.4f} (Δ {:+.4f}) · "
                  "recall_macro={:.4f} (Δ {:+.4f}) · recall por clase {}".format(
                      f["regimen"], f["n"], f["f1_macro"],
                      f["delta_f1_macro_vs_oof"], f["recall_macro"],
                      f["delta_recall_macro_vs_oof"],
                      {c: round(f["recall_" + c], 4)
                       for c in config.CATEGORIAS_ATAQUE}))
        return filas

    # ------------------------------------------------------------------
    # 4. Cotejo contra la tabla publicada
    # ------------------------------------------------------------------
    def _cotejar_con_publicado(self):
        """
        Compara la fila 'd2_tipo_conocido' de cada algoritmo contra
        metricas_firmas.csv, que tiene exactamente ese alcance. Aborta a la primera
        diferencia por encima del redondeo publicado.
        """
        ruta = config.ruta_tabla(self.TABLA_FIRMAS)
        if not os.path.exists(ruta):
            print("   [aviso] no existe {}: sin cotejo contra lo publicado.".format(
                ruta))
            return
        df = pd.read_csv(ruta)
        mask = df["set_features"].astype(str) == str(self.set_features)
        if "semilla" in df.columns:
            mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
        pub = df[mask]

        # (columna de esta tabla, columna homónima de metricas_firmas.csv)
        pares = [("f1_macro", "f1_macro"), ("recall_macro", "recall_macro"),
                 ("precision_macro", "precision_macro"),
                 ("f1_weighted", "f1_weighted"), ("accuracy", "accuracy_D2"),
                 ("n", "n_test")]
        for clase in config.CATEGORIAS_ATAQUE:
            pares.append(("recall_" + clase, "recall_" + clase))
            pares.append(("f1_" + clase, "f1_" + clase))

        comprobados = 0
        for fila in self.filas:
            if fila["regimen"] != self.REGIMEN_D2_CONOCIDO:
                continue
            sub = pub[pub["algoritmo"].astype(str) == fila["algoritmo"]]
            if sub.empty:
                print("   [aviso] {} no está en {} para la variante '{}': sin "
                      "cotejo.".format(fila["algoritmo"], os.path.basename(ruta),
                                       self.set_features))
                continue
            for aqui, alli in pares:
                esperado = float(sub[alli].iloc[0])
                obtenido = float(fila[aqui])
                if abs(esperado - obtenido) > self.TOLERANCIA_COTEJO:
                    raise RuntimeError(
                        "DISCREPANCIA con {}: {} régimen '{}' da {}={:.8f} y la "
                        "tabla publicada dice {}={:.8f}. El .joblib o los splits no "
                        "son los de la corrida publicada.".format(
                            os.path.basename(ruta), fila["algoritmo"],
                            fila["regimen"], aqui, obtenido, alli, esperado))
                comprobados += 1
        print("   Cotejo contra {}: {} valores del régimen '{}' coinciden dentro "
              "del redondeo publicado.".format(
                  os.path.basename(ruta), comprobados, self.REGIMEN_D2_CONOCIDO))

    # ------------------------------------------------------------------
    # 5. Persistencia y comprobación del artefacto
    # ------------------------------------------------------------------
    def _persistir(self):
        csv_path = config.ruta_tabla(self.TABLA)
        evaluacion.limpiar_variante_csv(
            csv_path, self.set_features,
            evaluacion.cabecera_esperada(self.filas[0]) if self.filas else None,
        )
        for fila in self.filas:
            evaluacion.guardar_metricas(fila, csv_path)
        print("   Guardada tabla: {}".format(csv_path))
        self._comprobar_tabla(csv_path)

    def _comprobar_tabla(self, csv_path):
        """
        Recuento comprobado por CÓDIGO releyendo el CSV escrito. Aborta si la unidad
        en curso no trae exactamente algoritmos × 3 regímenes, si algún par
        (algoritmo, régimen) se repite o falta, o si algún algoritmo se ha quedado
        sin su fila de referencia —sin ella las deltas de sus otras dos filas no
        tendrían contra qué leerse—.
        """
        nombre = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        mask = df["set_features"].astype(str) == str(self.set_features)
        de_semillas = config.es_tabla_de_semillas(nombre)
        if de_semillas and "semilla" in df.columns:
            mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
        df_var = df[mask]

        if de_semillas:
            unidad = "la variante '{}' con la semilla {}".format(
                self.set_features, config.RANDOM_STATE)
        else:
            unidad = "la variante '{}'".format(self.set_features)

        regimenes = [self.REGIMEN_REFERENCIA, self.REGIMEN_D2_CONOCIDO,
                     self.REGIMEN_D2_TODOS]
        esperadas = len(self.algoritmos) * len(regimenes)
        if len(df_var) != esperadas:
            raise ValueError(
                "{}: {} tiene {} filas y debería tener {} ({} algoritmos × {} "
                "regímenes). Regenera la tabla con un pase completo.".format(
                    nombre, unidad, len(df_var), esperadas,
                    len(self.algoritmos), len(regimenes)))

        observados = sorted(zip(df_var["algoritmo"].astype(str),
                                df_var["regimen"].astype(str)))
        objetivo = sorted((a, r) for a in self.algoritmos for r in regimenes)
        if observados != objetivo:
            raise ValueError(
                "{}: {} trae los pares {} y debería traer exactamente {} (una vez "
                "cada uno).".format(nombre, unidad, observados, objetivo))

        for algo in self.algoritmos:
            sub = df_var[df_var["algoritmo"].astype(str) == algo]
            refs = sub[sub["regimen"].astype(str) == self.REGIMEN_REFERENCIA]
            if len(refs) != 1:
                raise ValueError(
                    "{}: {} deja a '{}' con {} filas de régimen de referencia '{}' "
                    "y sus deltas no se podrían leer.".format(
                        nombre, unidad, algo, len(refs), self.REGIMEN_REFERENCIA))

        print("   Recuento {}: {} filas en {} · pares algoritmo × régimen únicos y "
              "completos · cada algoritmo con su fila de referencia".format(
                  nombre, len(df_var), unidad))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def ejecutar(self):
        config.setup_utf8()
        config.ensure_dirs()
        self.cargar_datos()
        self.filas = []
        for algo in self.algoritmos:
            self.filas.extend(self._medir_algoritmo(algo))
        print("-" * 70)
        self._cotejar_con_publicado()
        self._persistir()
        print("=" * 70)
        print("DEGRADACIÓN DE LA ETAPA 2 COMPLETADA ({} features)".format(
            self.set_features))
        print("=" * 70)
        return self.filas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Degradación fuera de dominio de la etapa 2: OOF sobre D3 "
                    "frente a los dos regímenes de D2. Reajusta por fold SOLO la "
                    "configuración ya elegida y solo sobre D3; D2 solo se reporta."
    )
    parser.add_argument(
        "--sin-seleccion", action="store_true",
        help="Usa el set de 122 features (variante sin selección 4.3.5) en lugar "
             "de las 54 por defecto.",
    )
    parser.add_argument(
        "--algoritmos", nargs="+", choices=NSLKDDSignatureTrainer.ALGORITMOS,
        default=NSLKDDSignatureDriftMeasurer.ALGORITMOS_POR_DEFECTO,
        help="Clasificadores de firmas a medir. Por defecto el pendiente "
             "(HistGradientBoosting) y el publicado (RandomForest).",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SEMILLA_POR_DEFECTO,
        help=config.AYUDA_CLI_SEMILLA + " Requiere los .joblib de firmas.py de ESA "
             "MISMA semilla: se aborta si declaran otra.",
    )
    args = parser.parse_args()

    # ANTES de instanciar: el __init__ congela el sufijo de los .joblib que se leen
    # y el StratifiedKFold del OOF. Ver config.py, encabezado.
    config.fijar_semilla(args.semilla)

    medidor = NSLKDDSignatureDriftMeasurer(
        sin_seleccion=args.sin_seleccion, algoritmos=args.algoritmos)
    medidor.ejecutar()
