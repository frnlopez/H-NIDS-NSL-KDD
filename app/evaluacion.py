# -*- coding: utf-8 -*-
"""
evaluacion.py — Módulo común de evaluación del H-NIDS.

Funciones reutilizadas por anomalias.py, firmas.py, baseline.py, hibrido.py y
cascada_invertida.py para no repetir en cada uno el cálculo de métricas ni el
guardado de figuras/tablas. Todas las
convenciones de clase (positivo = ataque = 1, orden de categorías) viven en
config.py; aquí solo se consumen.

Contrato:
  - evaluar_binario(y_true, y_pred, y_score=None)      -> dict
  - evaluar_multiclase(y_true, y_pred, labels=None)    -> dict
  - evaluar_0day_por_tipo(y_tipo_real, es_sospechoso, tipos_0day) -> dict
  - plot_matriz_confusion(y_true, y_pred, labels, ...) -> ruta PNG
  - plot_roc_pr(scores_por_algo, y_true, ...)          -> dict de AUCs
  - guardar_metricas(fila, csv_path)                   -> None (append)

Esquema de las tablas de métricas:
  - COLUMNAS_MINIMAS / CLAVE_UNICIDAD / TABLAS_PRINCIPALES  (esquema declarado)
  - COLUMNAS_MINIMAS_AUXILIARES / TABLAS_AUXILIARES         (idem auxiliares)
  - metricas_tiempo(t_entrenamiento_s, t_inferencia_s, n)   -> dict de 5 columnas
  - limpiar_variante_csv(csv_path, set_features, ...)       -> None (idempotencia)
  - comprobar_unicidad(csv_path, clave=None)                -> None (verifica)
  - comprobar_recuento(csv_path, set_features)              -> None (verifica)
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend no interactivo: nunca bloquea (nada de plt.show)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)

import config


# ---------------------------------------------------------------------------
# ESQUEMA DE MÉTRICAS — declarado aquí, comprobado en ejecución
# ---------------------------------------------------------------------------
# Conjunto MÍNIMO OBLIGATORIO de columnas, común a las cuatro tablas de métricas
# de los modelos. No es un comentario de buena voluntad: guardar_metricas()
# valida su presencia y aborta si falta alguna, así que una tabla sin ellas no
# puede llegar a existir.
#
#   algoritmo      qué se midió (nombre del modelo o de la cascada)
#   alcance        QUÉ CLASES y sobre QUÉ PARTICIÓN se calcula la fila
#                  (valores fijos en config.ALCANCE_*; cierra el problema de las
#                  columnas homónimas con alcances distintos entre tablas)
#   set_features   variante de características: '54' o '122_sin_seleccion'
#   sin_seleccion  bandera booleana equivalente, para filtrar sin parsear texto
#   n_features     nº real de columnas de entrada usadas
#   semilla        config.RANDOM_STATE con el que se produjo la fila (procedencia:
#                  así viaja con el dato y no vive solo en config.py)
#   commit         hash corto del repo (+ '-sucio' si había cambios sin commitear
#                  EN 'app/' —solo el código, no todo el árbol: los
#                  scripts escriben en 'Resultados/' durante la corrida y eso
#                  marcaría '-sucio' siempre; ver config._RUTA_SUCIEDAD—. Tres
#                  valores posibles: '<hash>', '<hash>-sucio' y
#                  '<hash>-suciedad_desconocida' (el `git status` falló; el hash
#                  es válido y la suciedad, indeterminada)
#   fecha          instante de la corrida (ISO-8601, segundos)
#   alcance_tiempo_s  QUÉ mide exactamente la columna 'tiempo_s' de ESTA tabla
#                  (valores fijos en config.ALCANCE_TIEMPO_S_*). Es obligatoria
#                  porque 'tiempo_s' tiene el mismo nombre de columna y tres
#                  magnitudes distintas —bloque completo del algoritmo en
#                  anomalías y firmas, SOLO el GridSearchCV en baseline, el tramo
#                  de la carga de los splits al cierre de la fila en el híbrido—.
#                  Ni se homogeneiza el cálculo ni se renombra la columna (la
#                  memoria cita ambos); se declara.
COLUMNAS_MINIMAS = (
    "algoritmo", "alcance", "set_features", "sin_seleccion", "n_features",
    "semilla", "commit", "fecha", "alcance_tiempo_s",
)

# CLAVE DE UNICIDAD de las tablas de métricas: una fila por combinación de
# variante de características × algoritmo × alcance. Es la clave que
# comprobar_unicidad() verifica al terminar.
#
# SUSTITUCIÓN DELIBERADA (documentada porque no es lo que la clave sugiere): el
# borrado de limpiar_variante_csv() NO es por clave completa, es por VARIANTE —
# borra todas las filas de la variante en curso, no solo la de la clave que se va
# a escribir. Consecuencia real: correr `hibrido.py --detector IsolationForest`
# después del pase por defecto NO añade una fila junto a 'Autoencoder->
# RandomForest'; la sustituye. Se elige así a propósito: cada tabla publicada
# refleja EL ÚLTIMO PASE COMPLETO de cada variante, con un único commit y una
# única fecha por variante. Borrar por clave completa permitiría un CSV con filas
# de corridas distintas —y de commits distintos— mezcladas dentro de la misma
# variante, que es peor para citar la tabla. Si algún día se quisieran comparar
# cascadas alternativas, van a una tabla propia, no a esta.
CLAVE_UNICIDAD = ("set_features", "algoritmo", "alcance")

# CLAVE DE UNICIDAD de las tablas del BARRIDO DE SEMILLAS. La de arriba no
# les vale y es exactamente el muro (b) que documenta comprobar_recuento(): diez
# semillas de la misma variante comparten (set_features, algoritmo, alcance), así
# que serían diez duplicados. Aquí `semilla` ES parte de la identidad de la fila.
CLAVE_UNICIDAD_SEMILLAS = CLAVE_UNICIDAD + ("semilla",)

# Las cuatro tablas de métricas de los modelos: sobre ellas se exige el conjunto
# mínimo de columnas.
TABLAS_PRINCIPALES = frozenset({
    "metricas_anomalias.csv",
    "metricas_firmas.csv",
    "metricas_baseline.csv",
    "metricas_hibrido.csv",
})

# Tablas AUXILIARES: su fila tiene otra granularidad (algoritmo × balanceo, tipo
# de ataque, umbral candidato, categoría asignada) y por eso no llevan
# 'algoritmo' ni la clave de unicidad. Pero sí llevan PROCEDENCIA Y ALCANCE: sin
# 'alcance' la columna 'recall' de metricas_baseline_0day.csv o el 'f1_macro_cv'
# de metricas_balanceo.csv se leerían como métricas sobre D2 y no lo son. Se
# exigen todas por igual.
COLUMNAS_MINIMAS_AUXILIARES = (
    "alcance", "set_features", "sin_seleccion", "n_features",
    "semilla", "commit", "fecha",
)

#
# 'metricas_cascada_invertida.csv' entra aquí y NO en las principales,
# y no por comodidad: su fila es categoría-asignada × variante (no lleva
# 'algoritmo' ni la clave de unicidad), y sobre todo su contenido es una medida
# CONTRAFACTUAL sobre las filas NORMALES de D2 —lo que haría la etapa 2 si fuese
# la primera—, que no es ningún resultado del sistema publicado. Meterla en las
# principales chocaría además con el recuento fijo de FILAS_ESPERADAS_POR_VARIANTE.
# Su recuento (5 filas por variante) NO queda por eso sin comprobar: lo verifica
# el propio script, en NSLKDDInvertedCascadeMeasurer._comprobar_tabla(), releyendo
# el CSV escrito. Es la comprobación equivalente a comprobar_recuento() +
# comprobar_unicidad() para una tabla que no puede pasar por ellas.
#
# 'metricas_cruce_detectores.csv' y 'metricas_degradacion_firmas.csv' entran por
# el mismo motivo y con la misma contrapartida: su fila
# es tipo-de-ataque × variante y algoritmo × régimen, ninguna de las dos lleva la
# CLAVE_UNICIDAD de las principales, y su recuento por variante lo verifica su
# propio script releyendo el CSV (_comprobar_tabla()). Declararlas aquí NO es
# burocracia: es lo que hace que guardar_metricas() les exija 'alcance' y la
# procedencia completa, y sin 'alcance' la columna 'f1_macro' de la tabla de
# degradación —que vale una cosa out-of-fold sobre D3 y otra muy distinta sobre
# D2— se leería como la de metricas_firmas.csv. Entrar aquí NO las mete en el
# barrido de semillas: ese recorrido tiene su propia lista explícita en
# agregar_semillas.py y no lee este frozenset.
TABLAS_AUXILIARES = frozenset({
    "metricas_balanceo.csv",
    "metricas_baseline_0day.csv",
    "metricas_hibrido_0day.csv",
    "metricas_hibrido_calibracion.csv",
    "metricas_cascada_invertida.csv",
    "metricas_cruce_detectores.csv",
    "metricas_degradacion_firmas.csv",
})

# Nº de filas que cada tabla principal debe tener POR VARIANTE de características
# (54 y 122_sin_seleccion), y por tanto el total con las dos variantes corridas:
#   metricas_anomalias.csv  4 por variante → 8   (4 detectores)
#   metricas_firmas.csv     4 por variante → 8   (4 clasificadores)
#   metricas_baseline.csv   1 por variante → 2   (un RF monolítico)
#   metricas_hibrido.csv    1 por variante → 2   (una cascada)
# comprobar_recuento() lo verifica al final de cada corrida, así que el recuento
# no depende de revisarlo a ojo.
FILAS_ESPERADAS_POR_VARIANTE = {
    "metricas_anomalias.csv": 4,
    "metricas_firmas.csv": 4,
    "metricas_baseline.csv": 1,
    "metricas_hibrido.csv": 1,
}

# ---------------------------------------------------------------------------
# Tablas del BARRIDO DE SEMILLAS
# ---------------------------------------------------------------------------
# Una corrida con `--semilla N` (N != 42) NO escribe en ninguna de las nueve
# tablas de arriba: config.ruta_tabla() la desvía a '<nombre>_semillas.csv'. Estos
# dos conjuntos son esas tablas nuevas, DERIVADAS de los anteriores y no
# enumeradas a mano, para que añadir una tabla no obligue a acordarse de la suya.
#
# Tienen las MISMAS columnas que su tabla base —incluida `semilla`, que
# inyecta guardar_metricas— así que pasan la MISMA validación de
# esquema. Lo que cambia es lo que depende de la identidad de la fila, y son tres
# cosas, resueltas en las tres funciones de abajo:
#   limpiar_variante_csv   borra por (variante, semilla) y no por variante sola:
#                          re-correr la semilla 7 no puede llevarse la 3 y la 5.
#   comprobar_unicidad     clave CLAVE_UNICIDAD_SEMILLAS (con `semilla`).
#   comprobar_recuento     cuenta 4/4/1/1 por (variante, semilla), no por variante.
#
# CÓMO LO SABEN: por el NOMBRE del fichero (config.es_tabla_de_semillas), no por
# un parámetro nuevo en cada llamada. Es deliberado: las nueve llamadas de los
# cinco scripts ya construyen la ruta con config.ruta_tabla(), así que la regla
# vive en UN sitio y es imposible olvidarla en un call site. El precio es que las
# tres funciones leen config.RANDOM_STATE del estado global; se acepta porque la
# semilla ES global en esta corrida (la fija fijar_semilla() antes de todo).
TABLAS_PRINCIPALES_SEMILLAS = frozenset(
    config.nombre_tabla_semillas(n) for n in TABLAS_PRINCIPALES
)
TABLAS_AUXILIARES_SEMILLAS = frozenset(
    config.nombre_tabla_semillas(n) for n in TABLAS_AUXILIARES
)

# Columnas de tiempo: el tiempo de ENTRENAMIENTO y el de INFERENCIA no son
# la misma magnitud ni responden a la misma pregunta. El de inferencia es el que
# habla de despliegue, y se reporta además normalizado por flujo.
#
# CÓMO SE MIDEN. Los cuatro scripts cronometran con time.perf_counter(), no con
# time.time(): en Windows time.time() tiene una resolución de ~15,6 ms, por debajo
# de la cual quedan los predict más baratos —y publicarían un
# 'latencia_ms_por_flujo' = 0,0, un imposible—. perf_counter es monótono y de alta
# resolución, pero NO tiene época: solo sirve para diferencias, nunca para la
# columna 'fecha' (esa la da datetime.now()).
#
# LO QUE perf_counter NO ARREGLA: la varianza. Es wall-clock en una máquina no
# dedicada y compartida, así que la misma corrida con la misma semilla da tiempos
# muy distintos.
#
# Aquí NO se escribe ninguna cota ni ningún factor de dispersión ni ninguna
# cifra de ninguna corrida (ver config.py): ese material se falsa con cada corrida
# nueva y en el código —o peor, dentro del CSV— no se puede corregir sin re-correr.
# Consecuencia práctica, y hay que decirla al citar: NINGUNA columna de tiempo es
# reproducible ni sirve para afirmar que un algoritmo es "un 20 % más rápido" que
# otro. Son comparación relativa de coste y de orden de magnitud, y solo dentro de
# la misma corrida. Lo demás de la tabla sí es reproducible (semilla 42); estas
# cinco columnas y 'tiempo_s', no.
#
# QUÉ HAY DENTRO DE LA INFERENCIA MEDIDA (declaración obligatoria antes de citar
# cualquiera de las dos derivadas): SOLO el predict/score sobre características ya
# calculadas y ya cargadas en memoria como DataFrame. NO incluye captura de
# tráfico, ensamblado del flujo ni extracción de las 41 características del
# registro NSL-KDD —que es donde vive el coste real de un despliegue—. Por eso el
# caudal del DecisionTree NO es capacidad operativa: presentarlo como tal sería
# caer en el sesgo de evaluación solo en laboratorio. Cada fila lo lleva escrito en
# 'alcance_tiempo_s' (config._AVISO_LATENCIA_SOLO_PREDICT).
COLUMNAS_TIEMPO = (
    "tiempo_entrenamiento_s", "tiempo_inferencia_s", "n_inferencia",
    "latencia_ms_por_flujo", "flujos_por_segundo",
)


def metricas_tiempo(t_entrenamiento_s, t_inferencia_s, n_inferencia):
    """
    Bloque de columnas de tiempo, idéntico en las cuatro tablas.

    Parameters
    ----------
    t_entrenamiento_s : float
        Segundos de AJUSTE del modelo (fit / GridSearchCV / calibración). Cada
        script documenta en su llamada qué incluye exactamente.
    t_inferencia_s : float
        Segundos de PREDICCIÓN sobre la partición evaluada, medidos aparte del
        entrenamiento. Responde solo en parte al sesgo de evaluación en
        laboratorio: mide el predict, pero solo en parte porque el predict opera
        sobre características YA calculadas y YA en memoria — la captura, el
        ensamblado del flujo y la extracción de las 41 características quedan
        fuera, así que ni esta columna ni sus derivadas son capacidad operativa.
    n_inferencia : int
        Nº de flujos puntuados en ese tiempo (el denominador de la latencia).

    Returns
    -------
    dict con COLUMNAS_TIEMPO. latencia_ms_por_flujo y flujos_por_segundo son
    DERIVADAS del mismo par (t_inf, n), así que comparten GUARDA: o se publican
    las dos, o ninguna (NaN → celda vacía en el CSV).

    Que la guarda sea única no es cosmética. Con dos guardas distintas (`n > 0`
    para la latencia y `t_inf > 0` para el caudal) un t_inf de 0,0 publicaría
    `latencia_ms_por_flujo = 0.0` junto a un `flujos_por_segundo` vacío: dos
    celdas de la misma fila contradiciéndose, y la latencia afirmando un imposible
    (0 ms por flujo) en vez de reconocer que no se midió. Con perf_counter un
    t_inf de exactamente 0,0 es prácticamente imposible, pero un tiempo no
    medible debe dar celda vacía, nunca 0,0.
    """
    t_ent = float(t_entrenamiento_s)
    t_inf = float(t_inferencia_s)
    n = int(n_inferencia)
    medible = n > 0 and t_inf > 0
    latencia = (t_inf / n) * 1000.0 if medible else float("nan")
    caudal = n / t_inf if medible else float("nan")
    return {
        "tiempo_entrenamiento_s": round(t_ent, 3),
        "tiempo_inferencia_s": round(t_inf, 3),
        "n_inferencia": n,
        "latencia_ms_por_flujo": round(latencia, 6),
        "flujos_por_segundo": round(caudal, 1),
    }


def validar_esquema_minimo(fila, csv_path="", columnas=None, auxiliar=False):
    """
    Comprueba que la fila trae con valor el conjunto mínimo obligatorio (por
    defecto COLUMNAS_MINIMAS; las auxiliares pasan COLUMNAS_MINIMAS_AUXILIARES).
    Aborta si no: una tabla de resultados sin alcance ni procedencia no debe
    poder escribirse.

    Avisa además (sin abortar) de columnas cuyo nombre empieza por un marcador de
    config.PREFIJOS_SIN_DECLARAR ('oof_', 'd2_', 'cv_', 'val_', 'train_', 'd1_',
    'd3_') y que el vocabulario aplicable no sabe resolver: es una columna de otro
    alcance colada sin declarar.

    `auxiliar` decide QUÉ vocabulario aplica, y es la razón de que el parámetro
    exista: 'oof_' y 'd2_' solo tienen sentido en la tabla de calibración del
    híbrido, así que se resuelven (vía config.ALCANCE_PREFIJOS_AUXILIARES) SOLO
    con auxiliar=True. En una tabla PRINCIPAL avisan, que es lo que se quiere: un
    'oof_recall_macro' en metricas_firmas.csv sería una columna del train dentro
    de una fila cuyo `alcance` dice "sobre los ataques de D2".
    """
    columnas = tuple(columnas) if columnas is not None else COLUMNAS_MINIMAS
    faltan = [c for c in columnas
              if c not in fila or fila[c] is None or fila[c] == ""]
    if faltan:
        raise ValueError(
            "Fila de métricas incompleta para {}: faltan las columnas mínimas "
            "{} (conjunto obligatorio: {})".format(
                os.path.basename(csv_path) or "<csv>", faltan, list(columnas))
        )

    sospechosas = [
        c for c in fila
        if c.startswith(config.PREFIJOS_SIN_DECLARAR)
        and config.alcance_de_columna(c, incluir_auxiliares=auxiliar) is None
    ]
    if sospechosas:
        print("   [aviso] {}: las columnas {} llevan un marcador de alcance no "
              "declarado para una tabla {} (config.ALCANCE_PREFIJOS{}). "
              "Decláralo o renómbralas.".format(
                  os.path.basename(csv_path) or "<csv>", sospechosas,
                  "auxiliar" if auxiliar else "principal",
                  "_AUXILIARES/SUFIJOS" if auxiliar else "/SUFIJOS"))


# ---------------------------------------------------------------------------
# Métricas binarias (normal vs ataque) — usadas por anomalias.py e hibrido.py
# ---------------------------------------------------------------------------
def evaluar_binario(y_true, y_pred, y_score=None):
    """
    Métricas del problema binario normal(0) vs ataque(1).

    Parameters
    ----------
    y_true, y_pred : array-like de {0, 1}
        Etiquetas y predicciones (positivo = ataque = 1, ver config).
    y_score : array-like, opcional
        Puntuación continua donde MAYOR = más probable ataque (p. ej. el
        anomaly_score ya negado). Si se aporta, se calculan AUC-ROC y AUC-PR
        (independientes del umbral).

    Returns
    -------
    dict con precision/recall/f1 de la clase ataque, accuracy, FPR real,
    la matriz de confusión 2x2 (tn, fp, fn, tp) y, si hay y_score, roc_auc/pr_auc.
    El FPR es NaN (no 0,0) si en y_true no hay ningún flujo normal: sin
    negativos la tasa de falsas alarmas no está definida, no vale cero.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # pos_label=ataque para que precision/recall se refieran a detectar ataques
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[config.CLASE_ATAQUE],
        pos_label=config.CLASE_ATAQUE, average="binary", zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[config.CLASE_NORMAL, config.CLASE_ATAQUE])
    tn, fp, fn, tp = cm.ravel()
    # Guarda simétrica a la de metricas_tiempo(): si no hay NINGÚN flujo normal
    # (fp + tn == 0), la tasa de falsas alarmas NO ES 0,0 — es no medible. Un
    # 0,0 aquí publicaría "cero falsas alarmas" (el mejor resultado posible) en
    # una fila donde no había nada que poder equivocar. NaN → celda vacía en el
    # CSV, igual que la latencia. Propaga a 'fpr' de anomalias, a 'bin_fpr' de
    # baseline e hibrido y a 'fpr_cascada'/'fpr_detector' del híbrido, que solo
    # redondean este valor. Con D2 real (9.711 normales) no se dispara nunca;
    # existe porque el imposible no debe ser representable.
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    resultado = {
        "precision": float(p),
        "recall": float(r),          # = TPR / tasa de detección de ataques
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(fpr),           # tasa de falsas alarmas sobre tráfico normal
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        y_score = np.asarray(y_score).ravel()
        # Requiere ambas clases presentes en y_true
        if len(np.unique(y_true)) == 2:
            resultado["roc_auc"] = float(roc_auc_score(y_true, y_score))
            resultado["pr_auc"] = float(average_precision_score(y_true, y_score))
        else:
            resultado["roc_auc"] = float("nan")
            resultado["pr_auc"] = float("nan")
    return resultado


# ---------------------------------------------------------------------------
# Métricas multiclase — usadas por firmas.py (4 clases) e hibrido.py (5-6)
# ---------------------------------------------------------------------------
def evaluar_multiclase(y_true, y_pred, labels=None):
    """
    Métricas multiclase con orden de etiquetas FIJO (config).

    Parameters
    ----------
    y_true, y_pred : array-like de etiquetas string (p. ej. 'dos', 'probe'...).
    labels : list, opcional
        Orden explícito de clases. Si None, se usa el orden observado ordenado.

    Returns
    -------
    dict con precision/recall/f1 por clase, macro y weighted, la matriz de
    confusión (con sus labels) y el classification_report como texto.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))

    p, r, f1, soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    por_clase = {
        clase: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "soporte": int(soporte[i]),
        }
        for i, clase in enumerate(labels)
    }
    return {
        "labels": list(labels),
        "por_clase": por_clase,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "matriz_confusion": cm,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0,
        ),
    }


# ---------------------------------------------------------------------------
# Recall 0-day por tipo — métrica central de la detección 0-day (apartado 5.3)
# ---------------------------------------------------------------------------
def evaluar_0day_por_tipo(y_tipo_real, es_sospechoso, tipos_0day):
    """
    Fracción de cada tipo de ataque 0-day que la etapa 1 (anomalías) marca como
    sospechoso. Es la métrica con la que se elige entre los sets de 54 y 122
    features y el análisis central del híbrido.

    Parameters
    ----------
    y_tipo_real : array-like
        Tipo de ataque específico por muestra (columna 'attack_original' de D2).
    es_sospechoso : array-like de bool
        Salida binaria de la etapa 1 (True = marcado como anomalía/ataque).
    tipos_0day : iterable
        Tipos de ataque presentes en D2 y AUSENTES del train (los "0-day").
        Se calculan en runtime: set(tipos D2) - {'normal'} - set(tipos D3).

    Returns
    -------
    dict {tipo: {'n': soporte, 'detectados': int, 'recall': float}} + clave
    '__global__' con el recall agregado sobre todos los 0-day.
    """
    y_tipo_real = np.asarray(y_tipo_real).ravel()
    es_sospechoso = np.asarray(es_sospechoso).ravel().astype(bool)

    resultado = {}
    total_n = total_det = 0
    for tipo in sorted(tipos_0day):
        mask = (y_tipo_real == tipo)
        n = int(mask.sum())
        detectados = int(es_sospechoso[mask].sum())
        resultado[tipo] = {
            "n": n,
            "detectados": detectados,
            "recall": float(detectados / n) if n > 0 else float("nan"),
        }
        total_n += n
        total_det += detectados
    resultado["__global__"] = {
        "n": total_n,
        "detectados": total_det,
        "recall": float(total_det / total_n) if total_n > 0 else float("nan"),
    }
    return resultado


# ---------------------------------------------------------------------------
# Figuras — todas a FIGURAS_DIR, 300 dpi, títulos en español (van a la memoria)
# ---------------------------------------------------------------------------
def plot_matriz_confusion(y_true, y_pred, labels, titulo, filename, normalizar=False):
    """
    Guarda la matriz de confusión como heatmap. `filename` es solo el nombre
    (se resuelve contra FIGURAS_DIR). Devuelve la ruta absoluta del PNG.
    """
    config.ensure_dirs()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalizar:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        datos, fmt = cm_norm, ".2f"
    else:
        datos, fmt = cm, "d"

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 1.4 * len(labels) + 2))
    sns.heatmap(datos, annot=True, fmt=fmt, cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    fig.tight_layout()
    ruta = os.path.join(config.FIGURAS_DIR, filename)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def plot_roc_pr(scores_por_algo, y_true, nombre_fig, titulo=""):
    """
    Superpone las curvas ROC y PR de varios algoritmos para la clase ataque.

    Parameters
    ----------
    scores_por_algo : dict {nombre_algo: y_score}
        y_score con MAYOR = más probable ataque.
    y_true : array-like {0, 1}
    nombre_fig : str
        Base del nombre; genera <base>_roc.png y <base>_pr.png en FIGURAS_DIR.

    Returns
    -------
    dict {algo: {'roc_auc': ..., 'pr_auc': ...}}
    """
    config.ensure_dirs()
    y_true = np.asarray(y_true).ravel()
    aucs = {}

    # ROC
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    for algo, score in scores_por_algo.items():
        score = np.asarray(score).ravel()
        fpr, tpr, _ = roc_curve(y_true, score, pos_label=config.CLASE_ATAQUE)
        auc = roc_auc_score(y_true, score)
        aucs.setdefault(algo, {})["roc_auc"] = float(auc)
        ax_roc.plot(fpr, tpr, label=f"{algo} (AUC={auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatorio")
    ax_roc.set_xlabel("Tasa de falsos positivos (FPR)")
    ax_roc.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax_roc.set_title(f"Curvas ROC{' — ' + titulo if titulo else ''}")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    ruta_roc = os.path.join(config.FIGURAS_DIR, f"{nombre_fig}_roc.png")
    fig_roc.savefig(ruta_roc, dpi=300, bbox_inches="tight")
    plt.close(fig_roc)

    # Precision-Recall
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for algo, score in scores_por_algo.items():
        score = np.asarray(score).ravel()
        prec, rec, _ = precision_recall_curve(y_true, score, pos_label=config.CLASE_ATAQUE)
        ap = average_precision_score(y_true, score)
        aucs.setdefault(algo, {})["pr_auc"] = float(ap)
        ax_pr.plot(rec, prec, label=f"{algo} (AP={ap:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Curvas Precision-Recall{' — ' + titulo if titulo else ''}")
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    ruta_pr = os.path.join(config.FIGURAS_DIR, f"{nombre_fig}_pr.png")
    fig_pr.savefig(ruta_pr, dpi=300, bbox_inches="tight")
    plt.close(fig_pr)

    return aucs


# ---------------------------------------------------------------------------
# Tabla única acumulada de experimentos
# ---------------------------------------------------------------------------
def cabecera_esperada(fila):
    """
    Cabecera que escribirá guardar_metricas() para una fila dada, incluidas las
    columnas de procedencia que inyecta ella misma. Sirve para pasarla a
    limpiar_variante_csv() y detectar un CSV con esquema anterior.
    """
    columnas = list(fila.keys())
    for c in ("semilla", "commit", "fecha"):
        if c not in columnas:
            columnas.append(c)
    return columnas


SUFIJO_RESPALDO = ".esquema-anterior.bak"


def _respaldar_csv(csv_path):
    """
    Aparta un CSV de esquema anterior renombrándolo a
    '<nombre>.esquema-anterior.bak' en lugar de borrarlo.

    Motivo: tras un cambio de esquema, el primer pase de cada script se llevaría
    por delante las filas de la otra variante, y ese estado intermedio sería
    silencioso. Con el respaldo queda a la vista.
    os.replace (no os.rename) porque en Windows rename falla si el destino existe.
    """
    destino = csv_path + SUFIJO_RESPALDO
    os.replace(csv_path, destino)
    return destino


def limpiar_variante_csv(csv_path, set_features, columnas_esperadas=None):
    """
    Idempotencia por variante de características, única para los cuatro
    scripts de modelos.

    El CSV es único y acumulado: contiene las filas de las dos variantes (54 y
    122). Al re-ejecutar UNA variante se borran primero SUS filas para no
    acumular duplicados; las de la otra variante se conservan intactas.

    OJO (ver la nota de CLAVE_UNICIDAD): el borrado es por VARIANTE, no por clave
    completa. Una corrida reescribe TODAS las filas de su variante, así que no
    puede duplicar ninguna clave, pero tampoco conviven dos cascadas o dos
    subconjuntos de algoritmos de la misma variante: la tabla refleja el último
    pase completo.

    EXCEPCIÓN — TABLAS DEL BARRIDO DE SEMILLAS: en una tabla '*_semillas.csv'
    el borrado es por (VARIANTE, SEMILLA). Borrar por variante sola sería el muro
    (a) de comprobar_recuento(): la semilla 7 se llevaría por delante las filas de
    la 3 y la 5 y las diez colapsarían a la última. La semilla no llega por
    parámetro: se lee de config.RANDOM_STATE, que es la de esta corrida y la que
    guardar_metricas() acaba de escribir en esas filas.

    Parameters
    ----------
    csv_path : str
    set_features : str
        Variante de esta corrida ('54' o '122_sin_seleccion').
    columnas_esperadas : iterable, opcional
        Cabecera que va a escribir la corrida. Se compara como CONJUNTO contra la
        del fichero ANTES de tocar nada: si difiere en cualquier sentido (falta
        una columna nueva o sobra una retirada) el fichero es de otro esquema, se
        aparta con _respaldar_csv() y se regenera al correr las dos variantes.
        Se comparan conjuntos, y antes de mutar, porque comprobar solo las
        columnas que faltan, y después del filtrado, no detectaría una columna
        retirada: el CSV se reescribiría ya sin las filas de la variante en curso
        y luego guardar_metricas abortaría tras minutos de entrenamiento.
    """
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # CSV ilegible (corrupto, vacío, sin cabecera): NO se puede "sobrescribir
        # con las filas nuevas" —guardar_metricas() lo encuentra existente y hace
        # pd.read_csv(nrows=0), que vuelve a lanzar EmptyDataError sin capturar y
        # revienta con traza cruda al final del entrenamiento—. Se aparta, que es
        # lo que ya se hace con un esquema incompatible.
        destino = _respaldar_csv(csv_path)
        print("   [aviso] {} no se puede leer como CSV (vacío o corrupto); se "
              "aparta como {} y se regenerará.".format(
                  os.path.basename(csv_path), os.path.basename(destino)))
        return

    # (1) Compatibilidad de esquema ANTES de mutar el fichero y ANTES del guarda
    # de 'set_features': un CSV sin esa columna es, por definición, de otro
    # esquema, y salir aquí sin apartarlo dejaba a guardar_metricas() abortando
    # por cabecera incompatible tras minutos de GridSearch. Por eso este bloque
    # va primero.
    if columnas_esperadas is not None:
        esperadas = set(columnas_esperadas)
        actuales = set(df.columns)
        if esperadas != actuales:
            destino = _respaldar_csv(csv_path)
            print("   [aviso] {} tenía otro esquema (faltaban {} · sobraban {}); "
                  "se aparta como {} y se regenerará al correr también la otra "
                  "variante.".format(
                      os.path.basename(csv_path),
                      sorted(esperadas - actuales), sorted(actuales - esperadas),
                      os.path.basename(destino)))
            return

    # (2) Sin 'set_features' no hay variante que filtrar. Solo se llega aquí con
    # columnas_esperadas=None (si venían, el bloque (1) ya habría apartado el
    # fichero: 'set_features' está en COLUMNAS_MINIMAS y en las AUXILIARES).
    if "set_features" not in df.columns:
        return

    # (3) Filtrado de las filas de ESTA variante — y, en las tablas del barrido,
    # solo las de ESTA semilla (si a la tabla le faltase la columna `semilla` sería
    # de otro esquema y el bloque (1) ya la habría apartado).
    a_borrar = df["set_features"].astype(str) == str(set_features)
    if config.es_tabla_de_semillas(csv_path) and "semilla" in df.columns:
        a_borrar &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
    df = df[~a_borrar]

    # Sin filas de otras variantes → se elimina el fichero para que
    # guardar_metricas lo recree con cabecera fresca (evita cabeceras rancias).
    # Aquí no hay nada que respaldar: todas las filas eran de la variante en
    # curso y esta misma corrida las reescribe acto seguido.
    if len(df) == 0:
        os.remove(csv_path)
        return

    df.to_csv(csv_path, index=False)


def comprobar_unicidad(csv_path, clave=None):
    """
    Verifica que el CSV no tiene filas duplicadas según la CLAVE_UNICIDAD
    (variante × algoritmo × alcance). Aborta si las hay: una tabla de resultados
    con dos filas para la misma clave es una tabla que no se puede citar.

    Si al CSV le falta alguna columna de la clave, no se comprueba nada (tablas
    auxiliares con otra granularidad de fila).

    En las tablas del BARRIDO DE SEMILLAS la clave por defecto es
    CLAVE_UNICIDAD_SEMILLAS: allí diez semillas de la misma variante comparten
    (variante, algoritmo, alcance) sin ser duplicados, y `semilla` es lo que las
    distingue. Un `clave` explícito manda sobre las dos.
    """
    if clave is None and config.es_tabla_de_semillas(csv_path):
        clave = CLAVE_UNICIDAD_SEMILLAS
    clave = tuple(clave) if clave is not None else CLAVE_UNICIDAD
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if any(c not in df.columns for c in clave):
        return
    duplicadas = df[df.duplicated(subset=list(clave), keep=False)]
    if len(duplicadas) > 0:
        raise ValueError(
            "{} tiene {} filas duplicadas para la clave de unicidad {}".format(
                os.path.basename(csv_path), len(duplicadas), list(clave))
        )


def comprobar_recuento(csv_path, set_features):
    """
    Verifica el nº de filas de una tabla PRINCIPAL para la variante que se acaba
    de correr, contra FILAS_ESPERADAS_POR_VARIANTE, e imprime el total del
    fichero. Hace comprobable por código el recuento esperado (8/8/2/2 con las
    dos variantes corridas) en lugar de dejarlo a ojo.

    Aborta si la variante en curso no tiene exactamente las filas esperadas (una
    corrida parcial, un algoritmo caído en silencio o un borrado incompleto).
    Tablas no declaradas o sin la columna: no se comprueba nada.

    LÍMITE DECLARADO (dispersión entre semillas): las cuatro tablas PRINCIPALES
    son de PASE ÚNICO con la semilla 42, y este abort lo da por supuesto. La
    dispersión de las 10 semillas va a TABLA PROPIA, con clave de unicidad que
    incluya `semilla`; NO se añaden 10 filas por algoritmo aquí. Meterlas en las
    principales chocaría con tres muros a la vez —(a) limpiar_variante_csv()
    borra por VARIANTE, así que 10 semillas colapsarían a la última; (b)
    CLAVE_UNICIDAD no incluye `semilla`, así que dos semillas de la misma
    variante son duplicado y comprobar_unicidad() aborta; (c)
    FILAS_ESPERADAS_POR_VARIANTE es fijo 4/4/1/1, así que 40 filas abortan
    aquí—. El abort se conserva a propósito: ninguna invocación legítima de la
    CLI lo dispara y es lo que hace comprobable el recuento.

    CÓMO LO RESUELVE EL BARRIDO: la corrida con `--semilla N` no llega a
    esta tabla —config.ruta_tabla() la desvía a '<nombre>_semillas.csv'— y los
    tres muros se levantan allí de otra forma: (a) limpiar_variante_csv() borra por
    (variante, semilla); (b) la clave es CLAVE_UNICIDAD_SEMILLAS, con `semilla`
    dentro; (c) el 4/4/1/1 de aquí abajo se exige por (variante, semilla) y no por
    variante, así que las diez semillas caben y cada una debe estar COMPLETA.
    El abort de las cuatro tablas principales se conserva intacto.
    """
    nombre = os.path.basename(csv_path)
    de_semillas = config.es_tabla_de_semillas(nombre)
    # El nº de filas esperado por pase es el de la tabla BASE: una corrida del
    # barrido produce los mismos 4/4/1/1, solo que por (variante, semilla).
    esperadas = FILAS_ESPERADAS_POR_VARIANTE.get(
        config.nombre_tabla_base(nombre) if de_semillas else nombre
    )
    if esperadas is None or not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if "set_features" not in df.columns:
        return

    mask = df["set_features"].astype(str) == str(set_features)
    if de_semillas:
        if "semilla" not in df.columns:
            return
        mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
    n_variante = int(mask.sum())

    if n_variante != esperadas:
        raise ValueError(
            "{}: {} tiene {} filas y debería tener {}. Regenera la tabla con un "
            "pase completo del script para esa {}.".format(
                nombre,
                "la variante '{}' con la semilla {}".format(
                    set_features, config.RANDOM_STATE)
                if de_semillas else "la variante '{}'".format(set_features),
                n_variante, esperadas,
                "combinación de variante y semilla" if de_semillas else "variante")
        )

    if de_semillas:
        # Aquí NO se declara un total esperado del fichero: depende de cuántas
        # semillas se hayan corrido ya y el barrido es incremental. Se informa de
        # lo que hay, que es lo comprobable sin inventar una expectativa.
        n_semillas = int(df["semilla"].nunique())
        print("   Recuento {}: {} filas en la variante '{}' con la semilla {} "
              "(total en fichero: {} filas de {} semillas distintas)".format(
                  nombre, n_variante, set_features, config.RANDOM_STATE,
                  len(df), n_semillas))
        return

    total_esperado = esperadas * 2  # dos variantes: 54 y 122_sin_seleccion
    print("   Recuento {}: {} filas en la variante '{}' (total en fichero: {} "
          "de {} esperadas con las dos variantes corridas)".format(
              nombre, n_variante, set_features, len(df), total_esperado))


def guardar_metricas(fila, csv_path):
    """
    Añade una fila (dict) a un CSV acumulado, una fila por experimento. Ese CSV
    es la "tabla única" de la que salen las comparativas del capítulo de
    Resultados.

    Inyecta automáticamente la PROCEDENCIA de la fila: 'semilla' (la de
    config, para que no viva solo en el código), 'commit' (hash corto del repo,
    con '-sucio' si el código tenía cambios) y 'fecha'. Si el CSV es una de las
    TABLAS_PRINCIPALES valida COLUMNAS_MINIMAS y si es una de las
    TABLAS_AUXILIARES valida COLUMNAS_MINIMAS_AUXILIARES; aborta si falta alguna.

    Todas las filas de un mismo CSV deben traer el MISMO conjunto de claves: al
    hacer append se comprueba contra la cabecera existente y se reordena la fila
    para que encaje; si los conjuntos difieren, se aborta en lugar de escribir
    columnas desalineadas en silencio.
    """
    fila = dict(fila)
    fila.setdefault("semilla", config.RANDOM_STATE)
    fila.setdefault("commit", config.commit_actual())
    fila.setdefault("fecha", datetime.now().isoformat(timespec="seconds"))

    # Las tablas del barrido de semillas validan el MISMO esquema que su tabla
    # base: son las mismas columnas con `semilla` como parte de la identidad de la
    # fila, no un esquema nuevo.
    nombre = os.path.basename(csv_path)
    if nombre in TABLAS_PRINCIPALES or nombre in TABLAS_PRINCIPALES_SEMILLAS:
        validar_esquema_minimo(fila, csv_path)
    elif nombre in TABLAS_AUXILIARES or nombre in TABLAS_AUXILIARES_SEMILLAS:
        # Las auxiliares no tienen 'algoritmo' ni clave de unicidad, pero sí
        # alcance y procedencia: las cinco por igual (coherencia de procedencia).
        validar_esquema_minimo(fila, csv_path,
                               columnas=COLUMNAS_MINIMAS_AUXILIARES,
                               auxiliar=True)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_fila = pd.DataFrame([fila])
    if os.path.exists(csv_path):
        cabecera = list(pd.read_csv(csv_path, nrows=0).columns)
        if set(cabecera) != set(fila.keys()):
            raise ValueError(
                "Esquema incompatible al añadir una fila a {}: la cabecera del "
                "fichero es {} y la fila trae {}. Regenera el CSV completo (las "
                "dos variantes) en lugar de mezclar esquemas.".format(
                    os.path.basename(csv_path), cabecera, sorted(fila.keys()))
            )
        df_fila = df_fila[cabecera]
        df_fila.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_fila.to_csv(csv_path, index=False)
