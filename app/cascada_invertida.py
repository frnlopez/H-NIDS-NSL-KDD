# -*- coding: utf-8 -*-
"""
cascada_invertida.py — Medición de la cascada invertida.

Pone NÚMERO a la afirmación de 3.2.2 («si las firmas fueran primero, condenarían
tráfico legítimo en masa»), que no queda medida en ningún otro artefacto.

Qué mide, exactamente
---------------------
Carga el clasificador de firmas YA ENTRENADO Y PERSISTIDO (`firma_<algo>_<set>.joblib`)
y le pasa SOLO las filas NORMALES de D2. Cuenta cuántas recibirían etiqueta de
ataque con confianza >= UMBRAL_CONF. Es una medida CONTRAFACTUAL: en el sistema
real esas filas nunca llegan al clasificador de firmas porque la etapa 1 no las
marca como sospechosas (`hibrido.py::_ensamblar_prediccion` solo reescribe
`pred[es_sospechoso]`), así que este número no existe en ninguna tabla del híbrido.

Por qué CUALQUIER fila recibe etiqueta de ataque: la etapa 2 se entrena sobre D3,
que son SOLO ataques (`firmas.py`), así que no tiene clase 'normal' y su argmax es
siempre una de las cuatro categorías de ataque. Lo único que puede frenar una
condena es el umbral de confianza. Ese es justo el punto de 3.2.2.

QUÉ NO ES el número de la fila '__global__'.

NO es "el FPR que tendría un sistema de firmas-primero"
sino una COTA INFERIOR de ese FPR. Motivo, y es de convención, no de medida: en el
sistema publicado `unknown` NO equivale a `normal`, es ALARMA (en
`hibrido.py::_ensamblar_prediccion` el flujo degradado a 'unknown' sigue dentro de
`es_sospechoso`, y la binaria del híbrido ES `es_sospechoso`). Así que los flujos
normales que aquí caen por debajo del umbral NO quedan exonerados: en una cascada
con las firmas delante pasarían a la etapa siguiente y podrían acabar en alarma
igualmente. Lo que esta tabla mide sin ambigüedad son los falsos positivos
IRRECUPERABLES —'n_condenadas'—, los que ninguna etapa posterior puede deshacer
porque salen ya con etiqueta de ataque y con confianza suficiente. Enunciarlo como
igualdad contradiría la convención 'unknown'=alarma o postularía en silencio un
sistema distinto, que queda fuera del alcance del trabajo (no se puede invertir
esta cascada, solo construir otro sistema).

Reglas de protocolo (invalidan la medición si se rompen)
--------------------------------------------------------
  * NO se entrena ni se re-entrena NADA: solo `predict_proba` sobre modelos ya
    persistidos. Cero `fit`.
  * D2 solo se REPORTA: de esta medición no sale ninguna elección, ningún
    hiperparámetro y ningún umbral. No toca la línea roja de la calibración
    (UMBRAL_CONF se selecciona solo sobre D3).
  * UMBRAL_CONF NO se escribe aquí como literal: se LEE del descriptor
    `hibrido_<set>.joblib` (clave 'umbral_conf_elegido'), que es donde lo dejó la
    calibración out-of-fold sobre D3 de `hibrido.py`. Si el descriptor no está, el
    script ABORTA en lugar de inventarse un valor. La procedencia del umbral viaja
    además en el propio dato, en la columna 'origen_umbral_conf'.
  * Artefacto PROPIO (`metricas_cascada_invertida.csv`): no se escribe ni una fila
    en las cuatro tablas principales — su borrado es por variante, su
    CLAVE_UNICIDAD no incluye 'semilla' y su recuento por variante es fijo.

Semilla (--semilla N)
---------------------
POR DEFECTO ES LA 42 y entonces la medición lee exactamente los .joblib publicados
y escribe en la tabla publicada. Con otra semilla lee los '_semilla<N>' de ese pase
del barrido y escribe en 'metricas_cascada_invertida_semillas.csv'. Los DOS
artefactos que carga —el clasificador de firmas y el descriptor del híbrido de
donde sale UMBRAL_CONF— se comprueban por variante Y por semilla, y la corrida
ABORTA si declaran otra: con un umbral de otra semilla, 'n_condenadas' —la cifra
citable— dejaría de corresponder a la fila que la publica. Este script entra en el
barrido de semillas: son CINCO scripts por semilla, no cuatro. Las diez semillas están en
config.SEMILLAS_BARRIDO (la 42 NO está entre ellas).

Salida
------
  Resultados/metricas_cascada_invertida.csv   5 filas por variante (4 categorías
                                              de ataque + '__global__')
  Resultados/figuras/cascada_invertida_<set>.png

Las 5 filas por variante NO son un invariante de palabra: las verifica
`_comprobar_tabla()` releyendo el CSV al final de la corrida (recuento, unicidad
de 'categoria_asignada', partición de las normales de D2 entre los buckets del
argmax y coherencia de la fila agregada). La comprobación es
propia del script y no pasa por `evaluacion.comprobar_recuento()`: esta tabla es
AUXILIAR, no lleva columna 'algoritmo' y no está —ni debe estar— en
`evaluacion.FILAS_ESPERADAS_POR_VARIANTE`, cuyo contrato es el de las cuatro
tablas principales y no se toca.
"""
import argparse
import os

# Backend no interactivo ANTES de cualquier import que arrastre matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

import config
import evaluacion
from program import load_specialized_splits


class NSLKDDInvertedCascadeMeasurer:
    """
    Mide qué haría la etapa 2 (firmas) con el tráfico legítimo si fuese la primera.

    Una clase por script, como el resto del pipeline. No entrena: carga joblibs,
    puntúa las normales de D2 y publica el recuento con su alcance y procedencia.
    """

    # Los mismos cuatro de hibrido.FIRMAS. Se declaran aquí en lugar de importar
    # hibrido.py para no arrastrar anomalias.py (y su cadena de imports) en una
    # medición que solo necesita un predict_proba.
    FIRMAS = ["DecisionTree", "RandomForest", "KNN", "HistGradientBoosting"]

    # Etiqueta de la fila agregada del CSV (mismo convenio que
    # evaluacion.evaluar_0day_por_tipo, que ya usa '__global__').
    FILA_GLOBAL = "__global__"

    # Filas que esta tabla debe tener POR VARIANTE: las 4 categorías de ataque
    # (los buckets del argmax) + la fila agregada. Es fijo por construcción, no
    # una expectativa: el bucle de medir() escribe exactamente esas cinco.
    FILAS_ESPERADAS_POR_VARIANTE = len(config.CATEGORIAS_ATAQUE) + 1

    NOMBRE_CSV = "metricas_cascada_invertida.csv"

    def __init__(self, sin_seleccion=False, firma="RandomForest"):
        if firma not in self.FIRMAS:
            raise ValueError("Firma desconocida: " + str(firma))

        self.sin_seleccion = sin_seleccion
        self.firma = firma
        self.base_path = config.base_path(sin_seleccion=sin_seleccion)
        self.set_features = "122_sin_seleccion" if sin_seleccion else "54"
        # Token para los NOMBRES de artefacto: los dos .joblib que LEE (el de
        # firmas y el descriptor del híbrido, de donde sale UMBRAL_CONF) y la
        # figura. Con la semilla 42 es igual a set_features —lee exactamente lo
        # publicado—; con otra apunta a los '_semilla<N>' de ese pase del barrido.
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)

        # Rellenados por cargar_datos()
        self.X_normales = None      # filas de D2 con category_original == 'normal'
        self.n_normales = None
        self.n_features = None

        # Rellenados por medir()
        self.umbral_conf = None
        self.origen_umbral_conf = None
        self.clases_firma = None
        self.conf = None            # confianza máxima por fila normal
        self.cat_argmax = None      # categoría de ataque más probable por fila
        self.filas = []             # filas del CSV de salida

    # ------------------------------------------------------------------
    # 1. Carga: SOLO las filas normales de D2
    # ------------------------------------------------------------------
    def cargar_datos(self):
        print("=" * 70)
        print("CASCADA INVERTIDA — set: {} · firma: {}".format(
            self.set_features, self.firma))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError("No se pudieron cargar las divisiones: " + self.base_path)

        X_D2 = splits["D2"]["X"]
        cat_D2 = splits["D2"]["y_category"]["category_original"].astype(str).values
        mask_normal = (cat_D2 == config.ETIQUETA_NORMAL)

        self.X_normales = X_D2.loc[mask_normal]
        self.n_normales = int(mask_normal.sum())
        self.n_features = X_D2.shape[1]

        print("   D2 completo: {} filas · {} features".format(len(X_D2), self.n_features))
        print("   Filas NORMALES de D2 (denominador de esta medición): {}".format(
            self.n_normales))
        # El recuento se REPORTA tal cual salga del split; no se fuerza a ninguna
        # cifra esperada (en NSL-KDD son 9.711, pero si el split diese otro número
        # el dato manda y el aviso lo deja a la vista).
        if self.n_normales != 9711:
            print("   [aviso] el recuento de normales de D2 NO es 9.711 sino {}: "
                  "se reporta el valor real del split.".format(self.n_normales))

    # ------------------------------------------------------------------
    # 2. Joblibs: el clasificador de firmas y el UMBRAL_CONF ya calibrado
    # ------------------------------------------------------------------
    def _cargar_joblib_firma(self):
        ruta = config.MODELOS_DIR + r"\firma_{}_{}.joblib".format(
            self.firma, self.sufijo_artefactos)
        # Simetría con _leer_umbral_conf(): el traceback crudo de joblib.load() no
        # dice qué falta ni de dónde sale, así que se relanza con un mensaje
        # explicativo y se sigue ABORTANDO (sin este modelo no hay nada que medir).
        try:
            datos = joblib.load(ruta)
        except FileNotFoundError:
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
                "No existe {}: es el clasificador de firmas que esta medición hace "
                "juzgar a los normales de D2, y aquí no se entrena nada. Ejecuta "
                "primero firmas.py{} para la variante '{}'.{}".format(
                    ruta, flag_variante, self.set_features, aviso_semilla)) from None
        # Misma salvaguarda que hibrido.py: un joblib de otra variante no cuadraría
        # en columnas con X_normales.
        if str(datos.get("set_features")) != str(self.set_features):
            raise RuntimeError(
                "El joblib {} es de la variante '{}' pero se esperaba '{}'".format(
                    ruta, datos.get("set_features"), self.set_features))
        nombres = datos.get("feature_names")
        if nombres is not None and list(nombres) != list(self.X_normales.columns):
            raise RuntimeError(
                "Las columnas de D2 no coinciden con las 'feature_names' del "
                "joblib {}: la medición no sería válida.".format(ruta))
        # Misma salvaguarda de SEMILLA que hibrido._cargar_joblib: medir con el
        # modelo de otra semilla daría una fila cuya columna 'semilla' mentiría.
        semilla_joblib = datos.get("semilla")
        if semilla_joblib is None:
            print("   [aviso] {} no declara 'semilla': no se puede comprobar que "
                  "sea de esta corrida.".format(ruta))
        elif int(semilla_joblib) != int(config.RANDOM_STATE):
            raise RuntimeError(
                "El joblib {} se entrenó con la semilla {} y esta corrida usa la "
                "{}: la fila declararía una semilla que no es la del modelo.".format(
                    ruta, semilla_joblib, config.RANDOM_STATE))
        return datos, ruta

    def _leer_umbral_conf(self):
        """
        UMBRAL_CONF NO se escribe aquí: se lee de donde ya vive, el descriptor del
        híbrido, que lo guarda en 'umbral_conf_elegido' tras calibrarlo por
        cross_val_predict out-of-fold sobre D3 (hibrido.py::_calibrar_umbral_conf,
        regla de presupuesto τ). Si el descriptor falta, se ABORTA: un literal
        escrito a mano aquí sería un número inventado.

        SALVAGUARDAS SIMÉTRICAS A LAS DE _cargar_joblib_firma(): este descriptor
        es el SEGUNDO artefacto que el script lee, y el nombre del fichero no
        basta para identificarlo: basta un renombrado a mano, o
        un `hibrido_54_semilla3.joblib` copiado de otra corrida, para que la
        medición use un umbral ajeno mientras la columna 'semilla' de las cinco
        filas del CSV declara la de ESTA corrida. Como el umbral es el criterio de
        condena, un umbral de otra semilla mueve directamente 'n_condenadas', que
        es la cifra citable. Se comprueban las dos cosas con el mismo criterio que
        el joblib de firmas: la variante ABORTA si no cuadra, y la semilla ABORTA
        si no cuadra pero solo AVISA si el descriptor no la declara (artefacto
        antiguo).
        """
        ruta = config.MODELOS_DIR + r"\hibrido_{}.joblib".format(
            self.sufijo_artefactos)
        try:
            desc = joblib.load(ruta)
        except FileNotFoundError:
            raise RuntimeError(
                "No existe {}: UMBRAL_CONF vive ahí (lo deja la calibración OOF de "
                "hibrido.py) y este script no puede inventárselo. Ejecuta primero "
                "hibrido.py para esta variante.".format(ruta))
        if str(desc.get("set_features")) != str(self.set_features):
            raise RuntimeError(
                "El descriptor {} es de la variante '{}' pero se esperaba '{}': su "
                "UMBRAL_CONF se calibró sobre otro set de características".format(
                    ruta, desc.get("set_features"), self.set_features))
        semilla_desc = desc.get("semilla")
        if semilla_desc is None:
            print("   [aviso] {} no declara 'semilla': no se puede comprobar que "
                  "el UMBRAL_CONF sea el de esta corrida.".format(ruta))
        elif int(semilla_desc) != int(config.RANDOM_STATE):
            raise RuntimeError(
                "El descriptor {} se calibró con la semilla {} y esta corrida usa "
                "la {}: la medición condenaría con un umbral ajeno mientras la "
                "fila declara esta semilla.".format(
                    ruta, semilla_desc, config.RANDOM_STATE))
        umbral = desc.get("umbral_conf_elegido")
        if umbral is None:
            raise RuntimeError(
                "El descriptor {} no trae 'umbral_conf_elegido'.".format(ruta))
        if str(desc.get("firma")) != str(self.firma):
            print("   [aviso] el UMBRAL_CONF se calibró para la firma '{}' y aquí "
                  "se mide '{}'; se usa igualmente porque es EL umbral del sistema "
                  "y se declara en la columna 'origen_umbral_conf'.".format(
                      desc.get("firma"), self.firma))
        origen = "{}::umbral_conf_elegido".format(
            "hibrido_{}.joblib".format(self.sufijo_artefactos))
        return float(umbral), origen

    # ------------------------------------------------------------------
    # 3. La medición
    # ------------------------------------------------------------------
    def medir(self):
        joblib_firma, ruta_firma = self._cargar_joblib_firma()
        self.umbral_conf, self.origen_umbral_conf = self._leer_umbral_conf()
        print("-" * 70)
        print("   Modelo: {}".format(ruta_firma))
        print("   UMBRAL_CONF = {} (leído de {}; NO es un literal de este script)".format(
            self.umbral_conf, self.origen_umbral_conf))
        print("-" * 70)

        modelo = joblib_firma["modelo"]
        self.clases_firma = np.asarray(modelo.classes_)
        proba = modelo.predict_proba(self.X_normales)
        self.conf = proba.max(axis=1)
        self.cat_argmax = self.clases_firma[proba.argmax(axis=1)]

        # MISMA regla de corte que la cascada real (hibrido._ensamblar_prediccion,
        # la regla de 'unknown'; NO la regla τ, que es la de calibración): la etiqueta de
        # ataque se CONSERVA cuando conf >= UMBRAL_CONF y se degrada a 'unknown'
        # cuando conf < UMBRAL_CONF. Aquí "condena" = conservar etiqueta de ataque.
        condenadas = self.conf >= self.umbral_conf

        for cat in config.CATEGORIAS_ATAQUE:
            mask_cat = (self.cat_argmax == cat)
            self.filas.append(self._fila(cat, mask_cat, condenadas))
        # Fila agregada: n_argmax = todas las normales, porque la etapa 2 no tiene
        # clase 'normal' y por construcción asigna una categoría de ataque a cada
        # fila. Su 'tasa_condena' es una COTA INFERIOR del FPR de un sistema de
        # firmas-primero, NO ese FPR: lo que queda bajo el umbral no se exonera
        # ('unknown' es alarma), pasaría a la etapa siguiente. Ver el
        # encabezado del módulo.
        self.filas.append(self._fila(
            self.FILA_GLOBAL, np.ones(self.n_normales, dtype=bool), condenadas))

        n_cond = int(condenadas.sum())
        print("   CONDENADAS (etiqueta de ataque con conf >= {:.2f}): {} de {} "
              "normales de D2 = {:.2f} %".format(
                  self.umbral_conf, n_cond, self.n_normales,
                  100.0 * n_cond / self.n_normales))
        print("   → falsos positivos IRRECUPERABLES y COTA INFERIOR del FPR de un "
              "sistema de firmas-primero (no ese FPR: ver encabezado del módulo)")
        print("   Por debajo del umbral (irían a '{}' = alarma igualmente en la "
              "convención así que NO quedan exonerados): {}".format(
                  config.ETIQUETA_UNKNOWN, self.n_normales - n_cond))
        print("   Desglose por categoría de ataque asignada (argmax):")
        for fila in self.filas:
            print("      {:<12} argmax={:>6}  condenadas={:>6}  "
                  "tasa_condena={:.4f}".format(
                      fila["categoria_asignada"], fila["n_argmax"],
                      fila["n_condenadas"], fila["tasa_condena"]))
        return self.filas

    def _fila(self, categoria, mask_cat, condenadas):
        """Fila del CSV para un bucket de categoría asignada (o el agregado)."""
        n_argmax = int(mask_cat.sum())
        n_cond = int((mask_cat & condenadas).sum())
        conf_bucket = self.conf[mask_cat]
        return {
            "alcance": config.ALCANCE_CASCADA_INVERTIDA,
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "firma": self.firma,
            "umbral_conf": self.umbral_conf,
            "origen_umbral_conf": self.origen_umbral_conf,
            "categoria_asignada": categoria,
            "n_normales_d2": self.n_normales,
            "n_argmax": n_argmax,
            "n_condenadas": n_cond,
            "n_bajo_umbral": n_argmax - n_cond,
            # Fracción (0-1) sobre TODAS las normales de D2, no sobre el bucket:
            # así los RECUENTOS de las cuatro categorías suman el de la fila
            # '__global__'. Las 'tasa_condena' PUBLICADAS solo suman hasta el
            # redondeo a 6 decimales (en 122 dan 0.342808 frente a 0.342807), así
            # que la suma se comprueba sobre 'n_condenadas' —enteros— y no aquí.
            "tasa_condena": round(n_cond / self.n_normales, 6),
            # Celda VACÍA (NaN) si el bucket está vacío: no se publica un 0,0 que
            # se leería como "confianza cero medida".
            "conf_mediana_argmax": (round(float(np.median(conf_bucket)), 6)
                                    if n_argmax > 0 else float("nan")),
        }

    # ------------------------------------------------------------------
    # 4. Figura
    # ------------------------------------------------------------------
    def _plot(self):
        cats = list(config.CATEGORIAS_ATAQUE)
        por_cat = {f["categoria_asignada"]: f for f in self.filas}
        condenadas = [por_cat[c]["n_condenadas"] for c in cats]
        bajo = [por_cat[c]["n_bajo_umbral"] for c in cats]

        x = np.arange(len(cats))
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.bar(x, condenadas, 0.55, label="Condenadas (conf ≥ {:.2f})".format(
            self.umbral_conf), color="lightcoral", edgecolor="grey")
        ax.bar(x, bajo, 0.55, bottom=condenadas,
               label="Por debajo del umbral (→ '{}': alarma también)".format(
                   config.ETIQUETA_UNKNOWN),
               color="lightsteelblue", edgecolor="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_xlabel("Categoría de ataque asignada por el clasificador de firmas")
        ax.set_ylabel("Filas normales de D2")
        g = por_cat[self.FILA_GLOBAL]
        ax.set_title(
            "Cascada invertida: qué haría la etapa 2 con el tráfico legítimo\n"
            "{} · {} features · {} de {} normales de D2 condenadas ({:.1f} %)".format(
                self.firma, self.n_features, g["n_condenadas"], self.n_normales,
                100.0 * g["tasa_condena"]))
        ax.legend()
        for xi, (c, b) in enumerate(zip(condenadas, bajo)):
            if c + b > 0:
                ax.text(xi, c + b, str(c + b), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        ruta = config.FIGURAS_DIR + r"\cascada_invertida_{}.png".format(
            self.sufijo_artefactos)
        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("   Figura: {}".format(ruta))

    # ------------------------------------------------------------------
    # 5. Persistencia — tabla PROPIA, nunca las cuatro principales
    # ------------------------------------------------------------------
    def _persistir(self):
        # Con la semilla 42, la tabla propia publicada; con otra, su '*_semillas.csv'
        # del barrido. En ningún caso ninguna de las cuatro principales.
        csv_path = config.ruta_tabla(self.NOMBRE_CSV)
        evaluacion.limpiar_variante_csv(
            csv_path, self.set_features,
            evaluacion.cabecera_esperada(self.filas[0]) if self.filas else None,
        )
        for fila in self.filas:
            evaluacion.guardar_metricas(fila, csv_path)
        print("   Tabla de la cascada invertida: {}".format(csv_path))
        self._comprobar_tabla(csv_path)

    def _comprobar_tabla(self, csv_path):
        """
        Recuento comprobado por CÓDIGO de la tabla propia, releyendo el CSV ya
        escrito. Es el equivalente de `evaluacion.comprobar_recuento()` +
        `comprobar_unicidad()` para esta tabla, que no puede pasar por ellas: es
        AUXILIAR (no tiene columna 'algoritmo' ni la CLAVE_UNICIDAD) y meterla en
        `FILAS_ESPERADAS_POR_VARIANTE` sería tocar el contrato de las cuatro
        tablas principales, que no se toca.

        Sin esto, el invariante "5 filas por variante" que declara el encabezado
        del módulo no lo verificaría nadie y una corrida parcial —o un borrado a
        medias— dejaría la tabla corta en silencio. Aborta en cuatro casos:

          (1) la variante en curso no tiene exactamente las 5 filas;
          (2) alguna 'categoria_asignada' se repite o falta dentro de la variante;
          (3) los 'n_argmax' de las 4 categorías no suman 'n_normales_d2'. ESTA es
              la comprobación de que los buckets del argmax PARTICIONAN las
              normales de D2: una partición rota que afectase únicamente a
              filas por debajo del umbral pasaría entera bajo el radar de (4);
          (4) los 'n_condenadas' de las 4 categorías no suman el de '__global__'.
              Se comprueba sobre ENTEROS y no sobre 'tasa_condena', que solo
              cuadra hasta el redondeo a 6 decimales.
        """
        nombre = os.path.basename(csv_path)
        df = pd.read_csv(csv_path)
        mask = df["set_features"].astype(str) == str(self.set_features)
        # En la tabla del barrido de semillas la unidad es (variante, semilla):
        # sin este filtro, dos semillas de la misma variante darían 10 filas y el
        # recuento de 5 abortaría siendo ambas correctas.
        de_semillas = config.es_tabla_de_semillas(nombre)
        if de_semillas and "semilla" in df.columns:
            mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
        df_var = df[mask]

        # Cómo se nombra la unidad comprobada en los mensajes: la variante en la
        # tabla publicada, el par (variante, semilla) en la del barrido.
        if de_semillas:
            unidad = "la variante '{}' con la semilla {}".format(
                self.set_features, config.RANDOM_STATE)
        else:
            unidad = "la variante '{}'".format(self.set_features)

        if len(df_var) != self.FILAS_ESPERADAS_POR_VARIANTE:
            raise ValueError(
                "{}: {} tiene {} filas y debería tener {}. Regenera "
                "la tabla con un pase completo del script para esa unidad.".format(
                    nombre, unidad, len(df_var),
                    self.FILAS_ESPERADAS_POR_VARIANTE)
            )

        esperadas = list(config.CATEGORIAS_ATAQUE) + [self.FILA_GLOBAL]
        observadas = list(df_var["categoria_asignada"].astype(str))
        if sorted(observadas) != sorted(esperadas):
            raise ValueError(
                "{}: la variante '{}' trae las categorías {} y debería traer "
                "exactamente {} (una vez cada una).".format(
                    nombre, self.set_features, sorted(observadas),
                    sorted(esperadas))
            )

        cats = df_var[df_var["categoria_asignada"] != self.FILA_GLOBAL]
        glob = df_var[df_var["categoria_asignada"] == self.FILA_GLOBAL]

        # (3) PARTICIÓN: cada fila normal de D2 cae en uno y solo un bucket del
        # argmax, así que los 'n_argmax' de las 4 categorías suman el total de
        # normales. Se comprueba aparte de (4) porque una partición rota que solo
        # afectase a filas por debajo del umbral dejaría los 'n_condenadas'
        # cuadrando y aun así invalidaría el desglose publicado.
        suma_argmax = int(cats["n_argmax"].sum())
        n_normales_d2 = int(df_var["n_normales_d2"].iloc[0])
        if suma_argmax != n_normales_d2:
            raise ValueError(
                "{}: en la variante '{}' los 'n_argmax' de las 4 categorías suman "
                "{} y las normales de D2 son {}. Los buckets del argmax deben "
                "PARTICIONAR las normales de D2: ni una fila fuera ni una fila "
                "en dos buckets.".format(
                    nombre, self.set_features, suma_argmax, n_normales_d2)
            )

        suma_cat = int(cats["n_condenadas"].sum())
        suma_glob = int(glob["n_condenadas"].iloc[0])
        if suma_cat != suma_glob:
            raise ValueError(
                "{}: en la variante '{}' los 'n_condenadas' de las 4 categorías "
                "suman {} y la fila '{}' dice {}. Los buckets del argmax "
                "particionan las normales de D2: no pueden discrepar.".format(
                    nombre, self.set_features, suma_cat, self.FILA_GLOBAL,
                    suma_glob)
            )

        # En la tabla del barrido NO se declara un total esperado del fichero:
        # depende de cuántas semillas se hayan corrido y el barrido es incremental.
        if de_semillas:
            nota_total = "total en fichero: {} filas".format(len(df))
        else:
            nota_total = ("total en fichero: {} de {} esperadas con las dos "
                          "variantes corridas".format(
                              len(df), self.FILAS_ESPERADAS_POR_VARIANTE * 2))
        print("   Recuento {}: {} filas en {} ({}) · categorías "
              "únicas y completas · argmax particiona las {} normales de D2 · "
              "recuentos coherentes con '{}'".format(
                  nombre, len(df_var), unidad, nota_total, n_normales_d2,
                  self.FILA_GLOBAL))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def ejecutar(self):
        config.setup_utf8()
        config.ensure_dirs()
        self.cargar_datos()
        self.medir()
        self._plot()
        self._persistir()
        print("=" * 70)
        print("CASCADA INVERTIDA COMPLETADA ({} features)".format(self.set_features))
        print("=" * 70)
        return self.filas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medición de la cascada invertida: cuántas de las filas "
                    "NORMALES de D2 condenaría el clasificador de firmas si fuese "
                    "la primera etapa. No entrena nada; D2 solo se reporta."
    )
    parser.add_argument(
        "--sin-seleccion", action="store_true",
        help="Usa el set de 122 features (variante sin selección 4.3.5) en lugar "
             "de las 54 por defecto.",
    )
    parser.add_argument(
        "--firma", choices=NSLKDDInvertedCascadeMeasurer.FIRMAS,
        default="RandomForest",
        help="Clasificador de firmas a medir (por defecto RandomForest, el de la "
             "cascada del híbrido).",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SEMILLA_POR_DEFECTO,
        help=config.AYUDA_CLI_SEMILLA + " Requiere los .joblib de firmas y del "
             "híbrido de ESA MISMA semilla: se abortaría si declarasen otra.",
    )
    args = parser.parse_args()

    # ANTES de instanciar: el __init__ congela el sufijo con el que se buscan los
    # dos .joblib que este script LEE. Ver config.py, encabezado.
    config.fijar_semilla(args.semilla)

    medidor = NSLKDDInvertedCascadeMeasurer(
        sin_seleccion=args.sin_seleccion, firma=args.firma
    )
    medidor.ejecutar()
