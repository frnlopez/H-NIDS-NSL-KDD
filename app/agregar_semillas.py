# -*- coding: utf-8 -*-
"""
agregar_semillas.py — Agregador de la dispersión entre semillas.

Convierte las tablas del barrido (`metricas_*_semillas.csv`) en la tabla de
dispersión que cita el anexo de robustez de la memoria: por cada combinación de
variante de características × algoritmo × alcance × métrica, el recuento `n`, la
media, la desviación típica MUESTRAL (ddof=1), el mínimo y el máximo, a 4
decimales.

Emite además, por cada celda de calidad, el valor de la SEMILLA 42 al lado de esa
banda y el veredicto dentro/fuera, con su recuento agregado («N de M celdas fuera
de banda»): ver la sección "EL TITULAR (SEMILLA 42) FRENTE A LA BANDA".

POR QUÉ EXISTE: sin él la tabla del anexo saldría de cálculo manual sobre diez
filas por celda. Emite además la misma tabla ya formateada en Markdown
(`dispersion_semillas.md`) para pegarla en el anexo sin recalcular ni un número a
mano.

Reglas de protocolo (invalidan el TFG si se rompen)
---------------------------------------------------
  * CERO `fit`. Este script solo lee CSV y calcula estadísticos descriptivos.
  * NO ESCRIBE EN NINGUNA TABLA PUBLICADA. Sus tres únicas salidas son
    `dispersion_semillas.csv`, `comparaciones_pareadas.csv` y
    `dispersion_semillas.md`; se comprueba en
    _comprobar_salidas_no_publicadas() que ni el NOMBRE ni la RUTA de esas tres
    salidas coincidan con los de ninguna tabla de métricas (publicada o del
    barrido). Escribir en las cuatro principales sería
    incompatible con su borrado por variante, con su `CLAVE_UNICIDAD` sin
    `semilla` y con su recuento fijo.
  * LAS TABLAS PUBLICADAS SÍ SE LEEN, y SOLO en modo lectura. Las abre
    `_leer_publicada()`
    —con `pd.read_csv`, nunca con `to_csv`— y únicamente para recuperar el valor
    de la SEMILLA 42 de cada celda de calidad, que es lo que permite emitir el
    titular «N de M celdas fuera de banda» desde el artefacto en vez de contarlo
    a mano. La agregación (n/media/sd/mín/máx) NO las usa: sale exclusivamente de
    las `*_semillas.csv`.
  * NO CONSUME `config.RANDOM_STATE`: las rutas se construyen a mano contra
    `config.RESULTADOS_DIR` en vez de con `config.ruta_tabla()`, que depende de la
    semilla global de la corrida. Este script no tiene semilla: agrega las de otros.
  * ABORTA si a alguna combinación le falta alguna de las semillas de
    `config.SEMILLAS_BARRIDO`. Una media de 7 puntos presentada como de 10 es peor
    que no tener tabla.
  * NO DECIDE NADA. No dice si dos intervalos se solapan ni calcula p-valores:
    emite `n`/media/sd/mín/máx y la lectura se hace en prosa. Los contrastes
    estadísticos los calcula `estabilidad_semillas.py`.

La semilla 42 NO entra en la agregación, y no porque se filtre aquí: es que no
está en `config.SEMILLAS_BARRIDO` ni escribe en estas tablas (ver el encabezado
de `config.py`). Es el TITULAR de 5.1-5.3, un punto independiente de la banda.

---------------------------------------------------------------------------
EL TITULAR (SEMILLA 42) FRENTE A LA BANDA — de dónde sale «N de M»
---------------------------------------------------------------------------
POR QUÉ: el titular «N de M» —cuántas celdas de calidad tienen el valor de la
semilla 42 FUERA del intervalo [mín, máx] de las diez semillas— es una cifra que
va a la memoria, así que la emite el script en vez de contarse a mano.

NO ES UN UNDÉCIMO PUNTO. `n`, `media`, `sd`, `min` y `max` se siguen calculando
SOLO sobre las diez semillas de `config.SEMILLAS_BARRIDO`. Lo que se añade es una
COMPARACIÓN del titular contra esa banda ya cerrada: el valor de la 42 no entra en
ningún estadístico. Son once corridas por variante —diez de la banda más la
publicada— y no diez con el titular dentro (encabezado de `config.py`).

CRITERIO DE CASADO (una celda de calidad ←→ su valor homólogo de la 42):
  1. TABLA. La publicada homóloga de `tabla_origen`, vía
     `config.nombre_tabla_base()`: `metricas_firmas_semillas.csv` casa con
     `metricas_firmas.csv` y con ninguna otra. Es el primer discriminante y el que
     impide confundir el `RandomForest` de firmas con el de la cascada invertida.
  2. FILA. La CLAVE_AGRUPACION completa —`set_features`, `algoritmo` y el TEXTO de
     `alcance`—, que es la `CLAVE_UNICIDAD` de las tablas de métricas, MÁS el
     mismo `filtro` declarado en la ESPECIFICACIÓN (en la cascada invertida,
     `categoria_asignada == '__global__'`) y `semilla == 42`. El `algoritmo` se
     normaliza igual que en la agregación (en la cascada viaja en la columna
     `firma`).
  3. COLUMNA. La misma `metrica` de la fila de dispersión.
  4. UNICIDAD. Debe salir EXACTAMENTE UNA fila. Si salen cero o más de una, o si
     el valor no es numérico, la celda queda SIN CASAR: sus columnas del titular
     van vacías, se avisa, y esa celda NO cuenta ni en el numerador ni en el
     denominador de «N de M». No se fuerza ningún casado aproximado: un casado
     laxo (por ejemplo ignorando `alcance`) confundiría medidas de alcance
     distinto.

DENTRO/FUERA SE DECIDE CON LOS EXTREMOS TAL COMO SE PERSISTIERON EN EL CSV DE
ORIGEN —es decir, a los 6 DECIMALES con los que los escriben `anomalias.py`,
`firmas.py`, `baseline.py`, `hibrido.py` y `cascada_invertida.py`—, y NO con el
`min`/`max` a 4 decimales de la tabla de dispersión. Decir "sin redondear" sería
inexacto: nadie compara aquí contra el float de 17 dígitos que salió del modelo,
porque ese número no está en ningún fichero. Y no es un detalle:
con el `min`/`max` de la tabla (4 decimales) el recuento sale 14 en vez de 13,
porque una celda cuyo mínimo real ES el valor de la 42 (`f1_u2r` de RandomForest
en 122: 0,318182) parece caer fuera al comparar contra un mínimo redondeado hacia
arriba. El redondeo es para leer, no para decidir. Un valor IGUAL a un extremo
cuenta como DENTRO. La columna `distancia_fuera_banda_42` publica lo lejos que
queda cada titular de su banda, para que las celdas que salen fuera por menos de
lo que se ve a 4 decimales se reconozcan como lo que son.

SOLO EL BLOQUE DE CALIDAD. Las filas del bloque de máquina (los tiempos) llevan
estas columnas VACÍAS a propósito: su banda mide carga de máquina y no el
algoritmo (ver RAZON_BLOQUE_MAQUINA), así que un «fuera de banda» ahí no diría
nada del sistema y solo inflaría el denominador del titular.

LA SALVEDAD DE PROCEDENCIA DEL TITULAR LLEVA EL RECUENTO POR COMMIT.
El `.md` no solo ENUMERA los commits de los que sale el valor de la semilla 42:
dice cuántas de las M celdas casadas aporta cada uno y cuántos de los N «fuera»
salen de cada versión del código (`_reparto_commits_titular_md()`). Sin ese peso,
«el titular viene de dos commits» admite cualquier reparto —mitad y mitad, o 97 y
1—, y de él depende cuánta de la distancia listada podría ser deriva de código. Los
recuentos se CUENTAN celda a celda sobre el mismo conjunto que «N de M» (bloque de
calidad, solo casadas); nunca se escriben a mano, y se comprueba que sumen M.

Procedencia: cada celda lleva DOS sellos y no uno —`commits_origen` (los commits de
las diez filas agregadas, que es el que vale para citar la banda) y `commit_agregador`
(el del agregador)—, y el script AVISA si una celda mezcla commits. También avisa, con el
reparto y sin abortar, si una celda no comparte la misma DECISIÓN en las diez
semillas (`balanceo`, `config_ganadora`): los folds pueden voltear el balanceo
ganador de 4.3.4, y una media sobre modelos con balanceos distintos hay que
declararla. Ese reparto va además en la COLUMNA `decisiones_no_constantes` de cada
celda del CSV, no solo en los avisos del `.md`: quien arme el anexo desde la tabla ve
la mezcla en la misma fila que la media que la promedia.

---------------------------------------------------------------------------
COMPARACIONES PAREADAS POR SEMILLA
---------------------------------------------------------------------------
Tercer bloque del script, con salida propia (`comparaciones_pareadas.csv`) y
sección propia en el `.md`. Enfrenta DOS opciones dentro de la MISMA semilla y
cuenta en cuántas gana cada una: es la formulación que sostiene «un algoritmo es
mejor que el otro», y la de dos titulares de la memoria (el «8 de 10» de
RandomForest frente a HistGradientBoosting en `f1_macro`, y el «40 de 40» de SMOTE
frente a no balancear), que así salen del artefacto y no de un recuento a mano.

Qué se comprueba y por qué basta: victorias de A + victorias de B + empates =
número de pares, cada bloque trae exactamente un par por semilla, y la fila
`__global__` cuadra contra la suma de sus bloques. El total se contrasta además
contra el `pares_esperados` DECLARADO en la especificación, así que un barrido
que perdiese una celda aborta en vez de publicar «8 de 9» rotulado como «de 10».
Los empates van en su propia columna y NO se reparten.

NO ES EL «8 DE 10» DE LOS TIEMPOS. Existe un homónimo —8 de 10 pares
intra-corrida de TIEMPOS de entrenamiento— que es otra cifra, de otra magnitud y
de otro bloque. Este script solo parea CALIDAD. La salvedad se IMPRIME también en
la sección del `.md` (callout ««8 de 10» es un homónimo…»), porque el aviso que
vive solo en el código no viaja con el artefacto, y ese artefacto lleva el bloque
de tiempos en el mismo documento.

La fila `__global__` («16 de 20») se rotula en el `.md` como AGREGADO de sus
bloques y no como titular: es la suma de los dos «8 de 10», no una medición nueva.

La tabla pareada lleva también `decisiones_no_constantes`, como su
hermana la de dispersión y sobre las mismas celdas: el `balanceo` ganador de
RandomForest voltea entre semillas, así que «RF gana a HGB en 8 de 10» NO es una
comparación entre dos configuraciones fijas. Se mira dentro de cada opción y se
excluye el propio eje comparado.

Salidas
-------
  Resultados/dispersion_semillas.csv   una fila por (variante, algoritmo, alcance,
                                       métrica, bloque)
  Resultados/comparaciones_pareadas.csv  una fila por (comparación, bloque) más la
                                       fila `__global__` de cada comparación
  Resultados/dispersion_semillas.md    la misma tabla formateada para el anexo, más
                                       la sección del titular con el recuento
                                       «N de M», las celdas que caen fuera y la
                                       sección de comparaciones pareadas

Uso
---
  python app\\agregar_semillas.py
"""
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config
import evaluacion


class AgregadorSemillas:
    """
    Agrega las tablas del barrido de semillas en una tabla de dispersión.

    Una clase por script, como el resto del pipeline. No entrena, no dibuja y no
    escribe en ninguna tabla publicada: lee los `*_semillas.csv`, comprueba que
    cada combinación esté completa y emite CSV + Markdown.
    """

    # --- QUÉ MÉTRICAS SE AGREGAN -----------------------------------------------
    # Lista CERRADA por tabla. No se agrega "todo lo numérico" a propósito: en
    # estas tablas conviven columnas de D2 con columnas del train (`f1_macro_cv`,
    # `auc_val`), recuentos enteros (tn/fp/fn/tp, n_test) y texto; una media de
    # todas ellas produciría una tabla que invita a citar cifras de selección de
    # hiperparámetros como si fuesen resultados. Cada entrada declara además de
    # qué columna sale el nombre del algoritmo y qué filas se filtran.
    #
    # `umbral` (anomalías) y `umbral_conf_elegido` (híbrido) NO son métricas sobre
    # D2 y se agregan igualmente porque son justo lo que el barrido pone a prueba:
    # el p95 se recalcula en cada semilla (el split 80/20 de D1 cambia) y el
    # UMBRAL_CONF se recalibra por OOF. Su alcance propio ya viaja en las tablas
    # de origen (config.ALCANCE_SELECCION / la columna `alcance`).
    ESPECIFICACION = (
        {
            "tabla": "metricas_anomalias.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "metricas": ("roc_auc", "pr_auc", "f1", "fpr", "umbral"),
            "con_tiempos": True,
        },
        {
            "tabla": "metricas_firmas.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "metricas": ("f1_macro", "accuracy_D2", "recall_macro", "f1_u2r"),
            "con_tiempos": True,
        },
        {
            "tabla": "metricas_baseline.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "metricas": ("bin_accuracy", "bin_fpr", "f1_macro",
                         "recall_0day_global"),
            "con_tiempos": True,
        },
        {
            "tabla": "metricas_hibrido.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "metricas": ("bin_recall", "bin_fpr", "bin_accuracy",
                         "conocida_f1_macro", "recall_0day_global",
                         "fpr_cascada", "umbral_conf_elegido"),
            "con_tiempos": True,
        },
        {
            # AUXILIAR y con otra granularidad de fila: no tiene columna
            # 'algoritmo' (el modelo medido viaja en 'firma') y sus cinco filas por
            # (variante, semilla) son los cuatro buckets del argmax + '__global__'.
            # Se agrega SOLO la fila agregada, que es la que se cita.
            #
            # SE AGREGAN LAS DOS COLUMNAS, y 'n_condenadas' no es redundante aunque
            # sea 'tasa_condena' × 'n_normales_d2' con denominador constante (las
            # 9.711 normales de D2): config.ALCANCE_CASCADA_INVERTIDA declara que
            # LO CITABLE es 'n_condenadas', y con solo la tasa a 4 decimales habría
            # que multiplicar por 9.711 a mano, perdiendo ~1 flujo de resolución.
            # Es un entero: agregarlo es gratis y sin pérdida.
            "tabla": "metricas_cascada_invertida.csv",
            "columna_algoritmo": "firma",
            "filtro": ("categoria_asignada", "__global__"),
            "metricas": ("n_condenadas", "tasa_condena"),
            "con_tiempos": False,
        },
    )

    # --- LAS COLUMNAS DE TIEMPO VAN EN BLOQUE APARTE ---------------------------
    # No entran en la tabla de calidad. Motivo, y hay que escribirlo al lado de la
    # cifra: son wall-clock en una máquina no dedicada. Está medido —el
    # Autoencoder de 54 fue de 37,71 s a 181,91 s entre dos corridas, 4,8×, con
    # calidad IDÉNTICA AL BIT—, así que la sd de estas columnas mide carga de
    # máquina y no algoritmo. Se emiten porque documentan esa dispersión, no como
    # resultado.
    #
    # OJO CON LAS ÉPOCAS: el recuento de épocas (162 en 54, 128 en 122) sale
    # IGUAL en las DOS únicas corridas que registran `n_iter_total_grid`, y
    # NINGUNA de ellas es una de las dos del contraejemplo de wall-clock: las del
    # 37,71 s → 181,91 s no registran épocas. Constancia OBSERVADA con n=2, no
    # determinismo demostrado — y la banda de 4,8x se declara como magnitud
    # OBSERVADA, sin causa atribuida. Que la sd de estos tiempos no sea calidad no depende de esa
    # atribución: son wall-clock en máquina no dedicada, y con eso basta.
    #
    # 'alcance_tiempo_s' queda FUERA de esta tupla aunque es una de las seis
    # columnas de tiempo: es PROSA (el texto de
    # config.ALCANCE_TIEMPO_S_*), no un número, y no admite media ni sd. Se
    # comprueba en su lugar que sea CONSTANTE dentro de cada combinación: si dos
    # semillas de la misma celda declarasen alcances distintos, la celda estaría
    # mezclando dos magnitudes y el aviso lo deja a la vista.
    COLUMNAS_TIEMPO = ("tiempo_s", "tiempo_entrenamiento_s",
                       "tiempo_inferencia_s", "latencia_ms_por_flujo",
                       "flujos_por_segundo")

    RAZON_BLOQUE_MAQUINA = (
        "wall-clock en máquina no dedicada: su sd mide CARGA DE MÁQUINA y no el "
        "algoritmo. Medido: el Autoencoder de 54 features pasó de 37,71 s a "
        "181,91 s entre dos corridas (4,8x) con calidad idéntica al bit. Esa banda "
        "es una magnitud OBSERVADA y se publica SIN causa atribuida: las dos "
        "corridas del contraejemplo no registran el recuento de épocas, así que no "
        "se puede afirmar que sea determinista ni descartar que las épocas "
        "expliquen parte de la banda (los 162 en 54 y 128 en 122 salen iguales "
        "solo en las dos corridas que sí registran esa columna, y no son estas). "
        "No es una propiedad del algoritmo y no se cita como resultado."
    )

    # --- DECISIONES QUE PUEDEN VOLTEAR ENTRE SEMILLAS -------------------------
    # Columnas que NO son métricas: son la DECISIÓN que tomó cada corrida. Se
    # comprueba que sean constantes dentro de cada celda y, si no lo son, se avisa
    # CON EL REPARTO (sin abortar). Motivo: los folds del StratifiedKFold mueven el
    # balanceo ganador de 4.3.4 y el best_params_ del GridSearchCV, y el balanceo
    # por algoritmo se fijó con una sola corrida (BALANCEO_OPCIONES en firmas.py).
    # Sin este aviso, el anexo publicaría la media de 'f1_macro' de RandomForest sobre
    # diez modelos entrenados con DOS esquemas de balanceo distintos sin decirlo.
    #
    # AVISO Y NO ABORTO, a propósito: que la decisión voltee en 3 de 10 semillas no
    # es un error del barrido — es un HALLAZGO para 5.4 (la fragilidad de una
    # decisión tomada con una sola corrida), y abortar lo escondería.
    COLUMNAS_CONSTANCIA = ("balanceo", "config_ganadora")

    BLOQUE_CALIDAD = "calidad"
    BLOQUE_MAQUINA = "dispersion_de_maquina"

    # --- EL TITULAR (SEMILLA 42) FRENTE A LA BANDA ----------------------------
    # Cuatro columnas por celda de calidad. El razonamiento y el criterio de
    # casado están en el encabezado del módulo, sección "EL TITULAR (SEMILLA 42)
    # FRENTE A LA BANDA"; aquí solo los nombres, que son los que se citan en el
    # anexo.
    #
    # Llevan todas el sufijo '_42' a propósito: en esta tabla 'min'/'max'/'media'
    # son de las DIEZ semillas del barrido y estas cuatro son de UNA que no está
    # entre ellas. Sin la marca en el nombre, 'valor' y 'commit' se leerían como
    # dos columnas más de la misma banda (la misma regla de homónimos que obligó a
    # llamar 'commit_agregador' al commit del agregador).
    COL_VALOR_42 = "valor_semilla_42"
    COL_DENTRO_42 = "dentro_banda_42"
    COL_DISTANCIA_42 = "distancia_fuera_banda_42"
    COL_COMMIT_42 = "commit_semilla_42"

    VEREDICTO_DENTRO = "dentro"
    VEREDICTO_FUERA = "fuera"
    VEREDICTO_SIN_CASAR = "sin_casar"

    # --- LAS CELDAS DE CALIDAD QUE NO SON MÉTRICAS SOBRE D2 -------------------
    # Se agregan a propósito (ver el comentario de ESPECIFICACION: son justo lo que
    # el barrido pone a prueba), pero entran en el DENOMINADOR del titular «N de M»
    # junto a métricas de rendimiento sobre D2, y eso hay que declararlo en el
    # artefacto: este `.md` se pega en el anexo, y un denominador rotulado como
    # "métricas" que incluye umbrales es una etiqueta falsa. El .md dice cuántas
    # son, cuál sería el denominador sin ellas y si alguno de los empates de borde
    # —los que deciden el recuento— es un umbral. NO se descuentan: el titular se
    # publica sobre todas las celdas de calidad casadas.
    METRICAS_NO_D2 = ("umbral", "umbral_conf_elegido")

    # Decimales de 'distancia_fuera_banda_42'. MÁS que los 4 de la tabla y no por
    # capricho: su trabajo es hacer visibles justo las celdas que se salen por
    # menos de lo que se ve a 4 decimales (la menor medida hoy es 9,7e-05, en
    # 'roc_auc' de OneClassSVM en 54). A 4 decimales esa distancia se imprimiría
    # como 0,0001 o como 0,0000 y la celda parecería un error de redondeo.
    DECIMALES_DISTANCIA = 6

    # Etiqueta del reparto por commit para las celdas casadas cuya fila publicada
    # NO trae commit. No se omiten del reparto: si se omitiesen, la suma de los
    # recuentos por commit no cuadraría con el denominador «M» y el lector no
    # sabría si falta un commit o falta una celda.
    ETIQUETA_SIN_COMMIT_42 = "_sin commit declarado_"

    # La columna 'semilla' de las tablas publicadas. Se filtra por ella además de
    # por la clave: una tabla publicada solo debería contener filas de la 42, pero
    # el filtro es gratis y convierte una tabla contaminada en un "sin casar"
    # ruidoso en lugar de en un titular silenciosamente equivocado.
    COLUMNA_SEMILLA = "semilla"

    # Columna de procedencia de las filas de ORIGEN. No es la del agregador: ver
    # _commits_de_celda().
    COLUMNA_COMMIT = "commit"

    # Clave de agrupación. Es la CLAVE_UNICIDAD de las tablas de métricas (sin
    # `semilla`, que es justo el eje sobre el que se agrega): así cada celda de la
    # tabla de dispersión corresponde exactamente a una fila de la tabla publicada
    # de la semilla 42, y el titular se puede poner al lado de su banda.
    CLAVE_AGRUPACION = ("set_features", "algoritmo", "alcance")

    DECIMALES = 4

    NOMBRE_CSV = "dispersion_semillas.csv"
    NOMBRE_MD = "dispersion_semillas.md"
    NOMBRE_CSV_PAREADAS = "comparaciones_pareadas.csv"

    # Etiqueta de la fila que agrega todos los bloques de una comparación pareada.
    # Se reutiliza el mismo literal que ya usa la cascada invertida para su fila
    # agregada: es la convención del proyecto para "esta fila no es un caso, es el
    # total".
    AMBITO_GLOBAL = "__global__"

    # --- COMPARACIONES PAREADAS POR SEMILLA -----------------------------------
    # POR QUÉ EXISTE ESTE BLOQUE: dos titulares de la memoria —«RandomForest gana a
    # HistGradientBoosting en 8 de las 10 semillas» (5.2.4.1, la formulación que se usa
    # al afirmar que un algoritmo es mejor que el otro) y «SMOTE gana en 40 de 40
    # celdas» del eje SMOTE vs nada (5.2.4.2)— son recuentos que van a la memoria y por
    # eso los emite el script en vez de contarse a mano.
    #
    # CUIDADO CON EL HOMÓNIMO: hay otro «8 de 10» que es de TIEMPOS de entrenamiento
    # (8 de 10 pares intra-corrida). NO es esta cifra, no se fusionan y no se citan la
    # una por la otra. Este bloque solo mide CALIDAD.
    #
    # Cada entrada declara: de qué tabla del barrido sale, sobre qué EJE se parea (la
    # columna cuyas dos opciones se enfrentan), qué dos opciones, con qué métrica, en qué
    # dirección es mejor, qué columnas forman el BLOQUE (dentro de cada bloque hay un par
    # por semilla) y cuántos pares tiene que haber en total. 'pares_esperados' NO es
    # decorativo: si el barrido diese otro número, el titular no se publica (aborta), que
    # es la red que impide publicar un recuento incompleto.
    COMPARACIONES_PAREADAS = (
        {
            "id": "firmas_RandomForest_vs_HistGradientBoosting_f1_macro",
            "tabla": "metricas_firmas.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "eje": "algoritmo",
            "opcion_a": "RandomForest",
            "opcion_b": "HistGradientBoosting",
            "metrica": "f1_macro",
            "mayor_es_mejor": True,
            "bloques": ("set_features",),
            "filtro_bloque": None,
            # 2 variantes x 10 semillas.
            "pares_esperados": 20,
        },
        {
            # Eje SMOTE vs NADA: solo KNN e HistGradientBoosting, los dos algoritmos
            # que no admiten class_weight en scikit-learn (BALANCEO_OPCIONES en
            # firmas.py).
            "id": "balanceo_SMOTE_vs_nada_f1_macro_cv",
            "tabla": "metricas_balanceo.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "eje": "balanceo",
            "opcion_a": "SMOTE",
            "opcion_b": "nada",
            "metrica": "f1_macro_cv",
            "mayor_es_mejor": True,
            "bloques": ("set_features", "algoritmo"),
            "filtro_bloque": {"algoritmo": ("KNN", "HistGradientBoosting")},
            # 2 variantes x 2 algoritmos x 10 semillas.
            "pares_esperados": 40,
        },
        {
            # Eje SMOTE vs class_weight: solo DecisionTree y RandomForest. Se emite
            # AUNQUE su reparto ya se pudiese leer en los avisos de
            # 'decisiones_no_constantes': el aviso solo aparece cuando la decisión
            # NO es constante, así que un eje unánime no generaría línea. Los
            # dos ejes se emiten igual y sobre denominadores idénticos.
            "id": "balanceo_SMOTE_vs_class_weight_f1_macro_cv",
            "tabla": "metricas_balanceo.csv",
            "columna_algoritmo": "algoritmo",
            "filtro": None,
            "eje": "balanceo",
            "opcion_a": "SMOTE",
            "opcion_b": "class_weight",
            "metrica": "f1_macro_cv",
            "mayor_es_mejor": True,
            "bloques": ("set_features", "algoritmo"),
            "filtro_bloque": {"algoritmo": ("DecisionTree", "RandomForest")},
            "pares_esperados": 40,
        },
    )

    def __init__(self, semillas=None):
        # Las diez semillas esperadas salen de config (lista cerrada, con sus
        # aserciones). El parámetro existe para poder auditar un barrido parcial a
        # mano, NO para publicar: el defecto es la lista completa.
        self.semillas = sorted(config.SEMILLAS_BARRIDO if semillas is None
                               else semillas)
        self.filas = []        # filas de la tabla de dispersión
        # Filas de la tabla de COMPARACIONES PAREADAS. Van a un CSV aparte y no a
        # `self.filas`: son otra unidad de fila (un par de opciones, no una celda de
        # banda) y mezclarlas en la tabla de dispersión obligaría a leer la mitad de
        # sus columnas como vacías.
        self.filas_pareadas = []
        self.avisos = []       # avisos no fatales, se imprimen y van al .md
        self.commit = None     # commit del AGREGADOR (no el de las filas de origen)
        self.fecha = None
        # Todos los commits vistos en las filas agregadas, para la cabecera del .md.
        self.commits_vistos = set()
        # Tablas publicadas ya leídas (solo lectura), cacheadas por nombre: el
        # casado del titular consulta la misma tabla una vez por métrica y por
        # celda, y releerla cada vez sería releer el mismo fichero ~40 veces.
        self._publicadas = {}
        # Celdas de calidad casadas contra el titular de la 42, para el recuento
        # «N de M» y para la sección propia del .md.
        self.celdas_fuera_banda = []
        self.celdas_dentro_banda = 0
        self.celdas_sin_casar = 0
        # Celdas de calidad CASADAS cuya métrica no es una métrica sobre D2
        # (METRICAS_NO_D2), por nombre de métrica: el desglose del denominador.
        self.celdas_no_d2 = {}
        # Celdas CASADAS cuyo titular cae EXACTAMENTE sobre un extremo de la banda.
        # Cuentan como DENTRO (criterio declarado), y por eso deciden el recuento:
        # si los empates contasen como fuera, «N de M» sería otro.
        self.celdas_borde = []
        # Commits vistos en la fila PUBLICADA de la semilla 42, para poder
        # compararlos con los de las bandas (`commits_origen`): ver
        # _salvedad_procedencia_titular().
        self.commits_titular = set()
        # Reparto de las celdas CASADAS por commit de origen del titular:
        # {commit: {"casadas": n, "fuera": n}}. Se cuenta sobre EXACTAMENTE el
        # mismo conjunto de celdas del que sale el titular «N de M» —solo bloque
        # de CALIDAD y solo casadas—, así que la suma de 'casadas' tiene que ser M.
        # Sirve para que la salvedad de procedencia no se limite a ENUMERAR los
        # commits: dice cuántas celdas aporta cada uno y cuántos de los «fuera»
        # salen de cada versión del código, que es lo que permite juzgar cuánta de
        # la distancia podría ser deriva de código.
        self.recuento_titular_por_commit = {}
        # Claves de avisos ya emitidos, para no repetir el mismo aviso una vez por
        # métrica (hasta 40 veces sobre la misma tabla).
        self._avisos_emitidos = set()

    # ------------------------------------------------------------------
    # 1. Lectura de una tabla del barrido
    # ------------------------------------------------------------------
    def _ruta_tabla_semillas(self, nombre_publicado):
        """
        Ruta de la tabla del BARRIDO correspondiente a una tabla publicada.

        No se usa config.ruta_tabla() a propósito: esa función decide por la
        semilla GLOBAL de la corrida y este script no tiene semilla. El desvío al
        `*_semillas.csv` es incondicional, así que las BANDAS no pueden salir de
        una tabla publicada.

        OJO: esta función no es la única puerta a `RESULTADOS_DIR`. `_ruta_tabla_publicada()` abre las nueve
        publicadas EN LECTURA para recuperar el valor de la semilla 42, así que la
        garantía que sigue en pie —y es la que importa— no es «no se abren», sino
        «no se escriben»: lo comprueba _comprobar_salidas_no_publicadas(),
        contrastando las rutas de escritura contra las de todas las publicadas.
        """
        return os.path.join(config.RESULTADOS_DIR,
                            config.nombre_tabla_semillas(nombre_publicado))

    def _leer(self, spec):
        """Lee y filtra una tabla del barrido. Aborta si no existe o está incompleta."""
        ruta = self._ruta_tabla_semillas(spec["tabla"])
        if not os.path.exists(ruta):
            raise RuntimeError(
                "No existe {}. El barrido de las {} semillas no se ha corrido (o "
                "no ha llegado a esta tabla): lánzalo con "
                "'python app\\barrido_semillas.py' antes de agregar.".format(
                    ruta, len(self.semillas)))
        df = pd.read_csv(ruta)
        df = self._normalizar_algoritmo(df, spec, ruta)
        df = self._aplicar_filtro(df, spec, ruta)

        faltan = [c for c in self.CLAVE_AGRUPACION + ("semilla",)
                  if c not in df.columns]
        if faltan:
            raise RuntimeError(
                "{} no trae las columnas {}: es de otro esquema y no se puede "
                "agregar.".format(ruta, faltan))
        return df, ruta

    def _normalizar_algoritmo(self, df, spec, ruta):
        """
        Columna 'algoritmo' unificada: en las cuatro principales ya se llama así;
        en la tabla de la cascada invertida el modelo medido viaja en 'firma'.

        Se aplica IGUAL a la tabla del barrido y a su homóloga publicada (las dos
        comparten esquema), para que el casado del titular use exactamente la
        misma noción de 'algoritmo' que la agregación.
        """
        col_algo = spec["columna_algoritmo"]
        if col_algo not in df.columns:
            raise RuntimeError(
                "{} no trae la columna '{}': no se puede identificar el "
                "algoritmo de cada fila.".format(ruta, col_algo))
        if col_algo != "algoritmo":
            df = df.copy()
            df["algoritmo"] = df[col_algo].astype(str)
        return df

    def _aplicar_filtro(self, df, spec, ruta):
        """Filtro de filas declarado en la ESPECIFICACIÓN ('__global__' en la cascada invertida)."""
        if spec["filtro"] is None:
            return df
        columna, valor = spec["filtro"]
        if columna not in df.columns:
            raise RuntimeError(
                "{} no trae la columna '{}' del filtro declarado.".format(
                    ruta, columna))
        return df[df[columna].astype(str) == str(valor)]

    # ------------------------------------------------------------------
    # 2. Comprobación de completitud
    # ------------------------------------------------------------------
    def _comprobar_completitud(self, df, ruta):
        """
        Cada combinación (variante, algoritmo, alcance) debe traer EXACTAMENTE una
        fila por cada semilla esperada. Aborta si falta alguna, si sobra una
        semilla ajena a la lista o si alguna aparece repetida.

        Es el requisito central del agregador: publicar una media de 7 semillas
        rotulada como de 10 sería una cifra falsa dentro del anexo, y nadie la
        detectaría releyendo la tabla.
        """
        esperadas = set(self.semillas)
        problemas = []
        for clave, grupo in df.groupby(list(self.CLAVE_AGRUPACION), dropna=False):
            presentes = [int(s) for s in grupo["semilla"].tolist()]
            conjunto = set(presentes)
            if len(presentes) != len(conjunto):
                repetidas = sorted(s for s in conjunto
                                   if presentes.count(s) > 1)
                problemas.append("{} → semillas REPETIDAS {}".format(
                    clave, repetidas))
                continue
            if conjunto != esperadas:
                problemas.append("{} → faltan {} · sobran {}".format(
                    clave, sorted(esperadas - conjunto),
                    sorted(conjunto - esperadas)))
        if problemas:
            raise RuntimeError(
                "{}: el barrido está INCOMPLETO y no se agrega. Se esperaban las "
                "semillas {} en cada combinación {}. Problemas:\n     - {}\n"
                "   Relanza 'python app\\barrido_semillas.py' (es reanudable: "
                "salta lo que ya está) y vuelve a agregar.".format(
                    os.path.basename(ruta), self.semillas,
                    list(self.CLAVE_AGRUPACION), "\n     - ".join(problemas))
            )

    # ------------------------------------------------------------------
    # 3. Estadísticos de una celda
    # ------------------------------------------------------------------
    def _estadisticos(self, valores):
        """
        n, media, sd MUESTRAL (ddof=1), mín y máx de una serie, a 4 decimales.

        Devuelve ADEMÁS 'min_crudo' y 'max_crudo' sin el redondeo a 4 decimales de
        esta tabla —o sea, con la precisión con la que se persistió el CSV de
        origen: 6 decimales—, que no van al CSV: son los extremos con los que se
        decide si el titular de la semilla 42 cae dentro o fuera de la banda.
        Decidir con los de 4 decimales daría un recuento distinto (ver el
        encabezado del módulo). 'crudo' significa aquí "sin redondear POR ESTE
        SCRIPT", no "con toda la precisión del modelo": esa ya se perdió al
        escribir el CSV.

        ddof=1 y no ddof=0: las diez semillas son una MUESTRA de las infinitas
        inicializaciones posibles, no la población entera. numpy usa ddof=0 por
        defecto (que subestima), así que se pasa explícito. Con n=10 la diferencia
        es un factor sqrt(10/9) = 1,054 sobre la sd — pequeño, pero es la
        definición correcta y la que se declara en el anexo.
        """
        serie = pd.to_numeric(pd.Series(valores), errors="coerce").dropna()
        n = int(len(serie))
        if n == 0:
            return None
        arr = serie.to_numpy(dtype=float)
        # Con n=1 la sd muestral no está definida (división por n-1 = 0): celda
        # vacía en lugar de un 0,0 que se leería como "dispersión nula medida".
        sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        return {
            "n": n,
            "media": round(float(arr.mean()), self.DECIMALES),
            "sd": (round(sd, self.DECIMALES) if sd == sd else float("nan")),
            "min": round(float(arr.min()), self.DECIMALES),
            "max": round(float(arr.max()), self.DECIMALES),
            "min_crudo": float(arr.min()),
            "max_crudo": float(arr.max()),
        }

    # ------------------------------------------------------------------
    # 3-bis. El titular de la semilla 42 frente a la banda
    # ------------------------------------------------------------------
    def _ruta_tabla_publicada(self, spec):
        """
        Ruta de la tabla PUBLICADA (semilla 42) homóloga de la del barrido.

        No se usa config.ruta_tabla() por el mismo motivo que en
        _ruta_tabla_semillas(): esa función decide por la semilla global de la
        corrida y este script no tiene semilla. config.nombre_tabla_base() es
        idempotente sobre un nombre ya publicado y deja explícito de qué tabla del
        barrido es homóloga esta.
        """
        return os.path.join(config.RESULTADOS_DIR,
                            config.nombre_tabla_base(spec["tabla"]))

    def _leer_publicada(self, spec):
        """
        Tabla publicada de la semilla 42, EN MODO LECTURA y cacheada.

        Es la única función de este script que abre una tabla publicada, y no
        escribe en ella jamás: solo se consulta el valor del titular para
        compararlo con la banda. Si no existe, se avisa UNA vez y el casado de esa
        familia de filas queda sin hacer (columnas vacías), sin abortar: la tabla
        de dispersión —que es el entregable principal— no depende de ella.
        """
        ruta = self._ruta_tabla_publicada(spec)
        if ruta in self._publicadas:
            return self._publicadas[ruta], ruta
        if not os.path.exists(ruta):
            self._publicadas[ruta] = None
            self.avisos.append(
                "No existe {}: las celdas de esa tabla quedan SIN el valor de la "
                "semilla 42 y no cuentan en el recuento de fuera de banda.".format(
                    os.path.basename(ruta)))
            return None, ruta
        df = pd.read_csv(ruta)
        try:
            df = self._normalizar_algoritmo(df, spec, ruta)
            df = self._aplicar_filtro(df, spec, ruta)
        except RuntimeError as error:
            # La publicada es de otro esquema: se degrada el casado, no se aborta.
            self._publicadas[ruta] = None
            self.avisos.append(
                "{}: no se puede casar con el barrido ({}). Sus celdas quedan SIN "
                "el valor de la semilla 42.".format(os.path.basename(ruta), error))
            return None, ruta
        self._publicadas[ruta] = df
        return df, ruta

    def _avisar_una_vez(self, clave_aviso, texto):
        """
        Añade un aviso SOLO la primera vez que se pide con la misma `clave_aviso`.

        Existe porque _valor_semilla_42() se llama una vez POR MÉTRICA (hasta 7 por
        celda, y hasta 40 sobre la misma tabla) y sus avisos son propiedades de la
        TABLA o de la CELDA, no de la métrica: sin esta deduplicación, una tabla
        publicada que faltase llenaría la sección «Avisos» del `.md` con el mismo
        párrafo decenas de veces y el resto de avisos —que sí son distintos entre
        sí— quedaría enterrado. No cambia ningún recuento: `celdas_sin_casar` se
        sigue incrementando por celda y por métrica, que es lo que se cuenta.
        """
        if clave_aviso in self._avisos_emitidos:
            return
        self._avisos_emitidos.add(clave_aviso)
        self.avisos.append(texto)

    def _valor_semilla_42(self, spec, clave, metrica):
        """
        Valor publicado de la semilla 42 para UNA celda, y el commit que lo produjo.

        Criterio de casado completo en el encabezado del módulo. En resumen: misma
        tabla homóloga, misma CLAVE_AGRUPACION (incluido el TEXTO de `alcance`),
        mismo filtro de la ESPECIFICACIÓN, `semilla` = 42, y EXACTAMENTE una fila.

        Devuelve (valor, commit) o (None, "") si no casa. Nunca fuerza un casado
        aproximado: si la clave estricta no da una fila única, se avisa y la celda
        queda fuera del recuento. Cuando el fallo es SOLO por el texto de
        `alcance` el aviso lo dice con ese nombre, porque esa es la avería
        probable —una constante ALCANCE_* editada después de la corrida
        publicada— y confundirla con "no hay fila" mandaría a buscar donde no es.
        """
        df, ruta = self._leer_publicada(spec)
        if df is None:
            return None, ""
        set_features, algoritmo, alcance = clave
        base = df[(df["set_features"].astype(str) == str(set_features))
                  & (df["algoritmo"].astype(str) == str(algoritmo))]
        sel = base[base["alcance"].astype(str) == str(alcance)]
        if self.COLUMNA_SEMILLA in df.columns:
            sel = sel[sel[self.COLUMNA_SEMILLA].astype(str)
                      == str(config.SEMILLA_POR_DEFECTO)]
        else:
            # Propiedad de la TABLA, no de la métrica: una vez por tabla.
            self._avisar_una_vez(
                ("sin_columna_semilla", ruta),
                "{}: sin columna '{}'; el casado del titular no puede comprobar "
                "que la fila sea la de la semilla {}.".format(
                    os.path.basename(ruta), self.COLUMNA_SEMILLA,
                    config.SEMILLA_POR_DEFECTO))
        if len(sel) != 1:
            if len(sel) == 0 and len(base) > 0:
                motivo = ("hay {} fila(s) con esa variante y ese algoritmo pero "
                          "ninguna con el mismo texto de 'alcance': la constante "
                          "ALCANCE_* ha cambiado desde la corrida publicada".format(
                              len(base)))
            else:
                motivo = "la clave casa con {} filas y debe casar con 1".format(
                    len(sel))
            # Propiedad de la CELDA (la fila no casa para ninguna métrica): una vez
            # por tabla y clave, no una por cada métrica de la celda.
            self._avisar_una_vez(
                ("sin_casar", ruta, clave),
                "{} · {}: SIN valor de la semilla {} ({}). Las celdas de calidad de "
                "esta combinación no entran en el recuento de fuera de banda.".format(
                    os.path.basename(ruta), clave, config.SEMILLA_POR_DEFECTO,
                    motivo))
            return None, ""
        fila = sel.iloc[0]
        if metrica not in sel.columns:
            self.avisos.append(
                "{} · {}: la tabla publicada no trae la columna '{}'; celda sin "
                "titular.".format(os.path.basename(ruta), clave, metrica))
            return None, ""
        valor = pd.to_numeric(pd.Series([fila[metrica]]), errors="coerce").iloc[0]
        if valor != valor:  # NaN: celda vacía o no numérica en la publicada
            self.avisos.append(
                "{} · {} · '{}': el valor de la semilla {} no es numérico; celda "
                "sin titular.".format(os.path.basename(ruta), clave, metrica,
                                      config.SEMILLA_POR_DEFECTO))
            return None, ""
        commit = ""
        if self.COLUMNA_COMMIT in sel.columns:
            commit = str(fila[self.COLUMNA_COMMIT])
        return float(valor), commit

    def _comparar_con_titular(self, spec, clave, metrica, est):
        """
        Compara el titular de la semilla 42 con la banda [mín, máx] de las diez.

        La comparación usa los extremos SIN redondear ('min_crudo'/'max_crudo') y
        un valor IGUAL a un extremo cuenta como DENTRO (ver el encabezado). No
        modifica ningún estadístico: la 42 no es un undécimo punto de la banda.

        Devuelve el dict de las cuatro columnas del titular y actualiza los
        contadores del recuento «N de M».
        """
        valor, commit = self._valor_semilla_42(spec, clave, metrica)
        if valor is None:
            self.celdas_sin_casar += 1
            return {
                self.COL_VALOR_42: "",
                self.COL_DENTRO_42: self.VEREDICTO_SIN_CASAR,
                self.COL_DISTANCIA_42: "",
                self.COL_COMMIT_42: "",
            }
        if commit:
            self.commits_titular.add(commit)
        if metrica in self.METRICAS_NO_D2:
            self.celdas_no_d2[metrica] = self.celdas_no_d2.get(metrica, 0) + 1
        minimo, maximo = est["min_crudo"], est["max_crudo"]
        if valor in (minimo, maximo):
            self.celdas_borde.append({
                "set_features": clave[0],
                "algoritmo": clave[1],
                "metrica": metrica,
                "extremo": "mín" if valor == minimo else "máx",
                "es_umbral": metrica in self.METRICAS_NO_D2,
            })
        if valor < minimo:
            dentro, distancia = False, minimo - valor
        elif valor > maximo:
            dentro, distancia = False, valor - maximo
        else:
            dentro, distancia = True, 0.0
        # Reparto por commit de origen del titular. Se cuenta AQUÍ y no releyendo
        # el CSV: esta rama es exactamente la de las celdas casadas del bloque de
        # calidad, las mismas que forman «N de M».
        etiqueta = commit if commit else self.ETIQUETA_SIN_COMMIT_42
        entrada = self.recuento_titular_por_commit.setdefault(
            etiqueta, {"casadas": 0, "fuera": 0})
        entrada["casadas"] += 1
        if not dentro:
            entrada["fuera"] += 1
        if dentro:
            self.celdas_dentro_banda += 1
        else:
            self.celdas_fuera_banda.append({
                "tabla_origen": config.nombre_tabla_semillas(spec["tabla"]),
                "set_features": clave[0],
                "algoritmo": clave[1],
                "alcance": clave[2],
                "metrica": metrica,
                "valor_42": valor,
                "min": minimo,
                "max": maximo,
                "distancia": distancia,
            })
        return {
            self.COL_VALOR_42: round(valor, self.DECIMALES),
            self.COL_DENTRO_42: (self.VEREDICTO_DENTRO if dentro
                                 else self.VEREDICTO_FUERA),
            self.COL_DISTANCIA_42: round(distancia, self.DECIMALES_DISTANCIA),
            self.COL_COMMIT_42: commit,
        }

    def _columnas_titular_vacias(self):
        """
        Las cuatro columnas del titular, VACÍAS: es lo que llevan las filas del
        bloque de máquina. No es un dato que falte, es una comparación que no se
        hace a propósito (RAZON_BLOQUE_MAQUINA): la banda de un wall-clock mide
        carga de máquina, así que un 'fuera de banda' ahí no diría nada del
        sistema y solo inflaría el denominador del titular.
        """
        return {
            self.COL_VALOR_42: "",
            self.COL_DENTRO_42: "",
            self.COL_DISTANCIA_42: "",
            self.COL_COMMIT_42: "",
        }

    def _agregar_metrica(self, spec, grupo, clave, metrica, bloque,
                         commits_origen="", decisiones_no_constantes=""):
        """Añade una fila de dispersión para una métrica de una combinación."""
        if metrica not in grupo.columns:
            if bloque == self.BLOQUE_CALIDAD:
                raise RuntimeError(
                    "{}: la columna '{}' no existe y es una de las métricas de "
                    "calidad declaradas para esta tabla.".format(
                        spec["tabla"], metrica))
            self.avisos.append("{}: sin columna '{}' (bloque de máquina); se "
                               "omite.".format(spec["tabla"], metrica))
            return

        est = self._estadisticos(grupo[metrica])
        if est is None or est["n"] != len(self.semillas):
            observados = 0 if est is None else est["n"]
            mensaje = ("{} · {} · métrica '{}': {} valores numéricos de {} "
                       "semillas (celdas vacías o no numéricas)".format(
                           spec["tabla"], clave, metrica, observados,
                           len(self.semillas)))
            if bloque == self.BLOQUE_CALIDAD:
                # Mismo criterio que la completitud de filas: una media de menos
                # puntos de los que declara no se publica.
                raise RuntimeError(
                    mensaje + ". No se publica una media incompleta como si "
                    "fuese de {} semillas.".format(len(self.semillas)))
            self.avisos.append(mensaje + "; se emite con la n real.")
            if est is None:
                return

        set_features, algoritmo, alcance = clave
        # El titular de la 42 solo se compara en el bloque de CALIDAD (ver
        # _columnas_titular_vacias()).
        columnas_42 = (self._comparar_con_titular(spec, clave, metrica, est)
                       if bloque == self.BLOQUE_CALIDAD
                       else self._columnas_titular_vacias())
        self.filas.append({
            "tabla_origen": config.nombre_tabla_semillas(spec["tabla"]),
            "bloque": bloque,
            "set_features": set_features,
            "algoritmo": algoritmo,
            "metrica": metrica,
            "n": est["n"],
            "media": est["media"],
            "sd": est["sd"],
            "min": est["min"],
            "max": est["max"],
            # Las cuatro columnas del TITULAR van pegadas a la banda que
            # comparan, no al final de la fila: quien lea el CSV ve el valor de la
            # 42 y su veredicto junto al mín/máx del que salen. La 42 NO entra en
            # 'n' ni en 'media' ni en 'sd' — es una comparación, no un sumando.
            self.COL_VALOR_42: columnas_42[self.COL_VALOR_42],
            self.COL_DENTRO_42: columnas_42[self.COL_DENTRO_42],
            self.COL_DISTANCIA_42: columnas_42[self.COL_DISTANCIA_42],
            # Procedencia del titular, que NO es la de la banda: la fila publicada
            # se produjo con otro commit que las diez del barrido (mismo motivo por
            # el que 'commits_origen' y 'commit_agregador' son dos columnas).
            self.COL_COMMIT_42: columnas_42[self.COL_COMMIT_42],
            "semillas": " ".join(str(s) for s in self.semillas),
            # El alcance de la fila de origen viaja con la cifra agregada: sin él,
            # un 'f1_macro' de firmas y uno de baseline se leerían como la misma
            # medida.
            "alcance": alcance,
            "nota_bloque": ("" if bloque == self.BLOQUE_CALIDAD
                            else self.RAZON_BLOQUE_MAQUINA),
            # DOS columnas de procedencia y no una, porque son dos cosas distintas
            # y confundirlas es publicar una banda sellada con un commit que no
            # produjo ninguno de sus puntos (regla de procedencia de
            # evaluacion.py:100-105):
            #   'commits_origen' = los commits de las 10 filas que se agregan. Es
            #                      el que vale para citar la banda. Con un barrido
            #                      reanudable a lo largo de días una celda puede
            #                      mezclar commits —basta que cambie el sufijo
            #                      '-sucio'—, y entonces esta celda lo dice.
            #   'commit_agregador' = el del AGREGADOR, es decir con qué versión de
            #                      ESTE script se calcularon media/sd. No dice nada
            #                      sobre con qué código se produjeron los puntos.
            #
            # SE LLAMA 'commit_agregador' Y NO 'commit':
            # en las nueve 'metricas_*.csv' la columna 'commit' significa «el código
            # que produjo esta fila», y aquí significaría «el del agregador, que no
            # produjo ninguno de los diez puntos». Es el homónimo con dos
            # significados que prohíbe la regla de config.py:326-332, y esta tabla no
            # pasa por guardar_metricas()/validar_esquema_minimo(), así que no hay
            # red que lo detecte. El nombre distinto ES la red.
            # EL REPARTO DE UNA DECISIÓN NO CONSTANTE ES COLUMNA, no solo aviso: si
            # un reparto como «SMOTE en 7 · class_weight en 3» fuese solo a stdout y
            # a la sección «Avisos» del .md, quien armase el anexo desde el CSV
            # —que es lo natural: es la tabla— vería la media sin enterarse de que
            # promedia modelos con dos balanceos distintos. Vacío = la decisión fue
            # la misma en las diez semillas.
            "decisiones_no_constantes": decisiones_no_constantes,
            "commits_origen": commits_origen,
            "commit_agregador": self.commit,
            "fecha": self.fecha,
        })

    # ------------------------------------------------------------------
    # 4. Agregación de una tabla
    # ------------------------------------------------------------------
    def _agregar_tabla(self, spec):
        df, ruta = self._leer(spec)
        self._comprobar_completitud(df, ruta)
        n_celdas = 0
        for clave, grupo in df.groupby(list(self.CLAVE_AGRUPACION), dropna=False):
            self._comprobar_alcance_tiempo_constante(spec, grupo, clave)
            decisiones = self._comprobar_decisiones_constantes(spec, grupo, clave)
            commits = self._commits_de_celda(spec, grupo, clave)
            for metrica in spec["metricas"]:
                self._agregar_metrica(spec, grupo, clave, metrica,
                                      self.BLOQUE_CALIDAD, commits, decisiones)
            if spec["con_tiempos"]:
                for metrica in self.COLUMNAS_TIEMPO:
                    self._agregar_metrica(spec, grupo, clave, metrica,
                                          self.BLOQUE_MAQUINA, commits, decisiones)
            n_celdas += 1
        print("   {}: {} combinaciones × {} semillas".format(
            os.path.basename(ruta), n_celdas, len(self.semillas)))

    def _comprobar_alcance_tiempo_constante(self, spec, grupo, clave):
        """
        'alcance_tiempo_s' es texto: no se agrega, se verifica que sea el MISMO en
        las diez filas de la celda. Si dos semillas declarasen alcances distintos,
        el bloque de máquina de esa celda estaría promediando dos magnitudes
        diferentes y el aviso lo saca a la
        luz sin abortar: la tabla de calidad no depende de esta columna.
        """
        if not spec["con_tiempos"] or "alcance_tiempo_s" not in grupo.columns:
            return
        distintos = grupo["alcance_tiempo_s"].astype(str).nunique()
        if distintos > 1:
            self.avisos.append(
                "{} · {}: la columna 'alcance_tiempo_s' toma {} valores distintos "
                "entre semillas; las columnas de tiempo de esa celda no miden lo "
                "mismo.".format(spec["tabla"], clave, distintos))

    def _comprobar_decisiones_constantes(self, spec, grupo, clave):
        """
        Comprueba que las columnas de DECISIÓN (COLUMNAS_CONSTANCIA) sean las mismas
        en las diez semillas de la celda, y AVISA con el reparto si no lo son.

        Mismo patrón que _comprobar_alcance_tiempo_constante() y por el mismo
        motivo —una celda que promedia dos cosas distintas sin decirlo—, pero aquí
        lo que puede voltear no es un texto de alcance: es el balanceo ganador de
        4.3.4 o el best_params_ del GridSearchCV, las dos decisiones que dependen de
        los folds. No aborta (ver el comentario de COLUMNAS_CONSTANCIA): el reparto
        es material de 5.4, no un fallo del barrido.

        Devuelve el reparto ya formateado para la columna
        'decisiones_no_constantes' del CSV ('' si todas fueron constantes), para que
        el aviso no viva solo en el .md: ver el comentario de esa columna en
        _agregar_metrica().
        """
        repartos = []
        for columna in self.COLUMNAS_CONSTANCIA:
            if columna not in grupo.columns:
                continue
            valores = grupo[columna].astype(str)
            reparto = valores.value_counts()
            if len(reparto) <= 1:
                continue
            detalle = " · ".join(
                "{} en {} semilla(s)".format(valor, int(veces))
                for valor, veces in reparto.items())
            # Sin comas: este texto viaja como VALOR dentro del CSV (misma regla que
            # los textos de alcance de config.py).
            repartos.append("{}: {}".format(columna, detalle))
            self.avisos.append(
                "{} · {}: la columna de DECISIÓN '{}' NO es constante entre las {} "
                "semillas ({}). La media de esta celda promedia modelos con "
                "decisiones distintas: hay que declararlo al citarla (material de "
                "5.4, no un error).".format(
                    spec["tabla"], clave, columna, len(self.semillas), detalle))
        return " | ".join(repartos)

    def _commits_de_celda(self, spec, grupo, clave):
        """
        Conjunto DISTINTO de commits de las filas que se agregan en esta celda, ya
        formateado para la columna 'commits_origen'. Avisa si hay más de uno.

        POR QUÉ NO BASTA EL COMMIT DEL AGREGADOR: el barrido es reanudable y puede correrse a lo
        largo de días, así que una celda puede mezclar commits —basta que cambie el
        sufijo '-sucio'—. Sellar la banda con el commit del agregador publicaría en
        el anexo diez puntos rotulados con un commit que no produjo ninguno de ellos, y
        el lector no tendría forma de detectarlo. La regla de procedencia del
        proyecto (evaluacion.py:100-105) pide lo contrario: que el dato viaje con el
        código que lo produjo.
        """
        if self.COLUMNA_COMMIT not in grupo.columns:
            self.avisos.append(
                "{} · {}: sin columna '{}'; la celda queda SIN procedencia de "
                "origen.".format(spec["tabla"], clave, self.COLUMNA_COMMIT))
            return ""
        commits = sorted(set(grupo[self.COLUMNA_COMMIT].astype(str)))
        self.commits_vistos.update(commits)
        if len(commits) > 1:
            self.avisos.append(
                "{} · {}: las {} semillas de esta celda se produjeron con {} "
                "commits distintos ({}). La banda NO sale de una sola versión del "
                "código y hay que decirlo al citarla.".format(
                    spec["tabla"], clave, len(self.semillas), len(commits),
                    ", ".join(commits)))
        return " ".join(commits)

    # ------------------------------------------------------------------
    # 4-bis. Comparaciones PAREADAS por semilla
    # ------------------------------------------------------------------
    def _leer_para_pareadas(self, spec):
        """
        Tabla del barrido para una comparación pareada, sin la comprobación de
        completitud de la agregación.

        NO se reutiliza _leer()/_comprobar_completitud() a propósito: esa
        comprobación exige UNA fila por semilla dentro de la CLAVE_AGRUPACION, y
        aquí las tablas traen VARIAS por semilla dentro de la misma clave (dos
        opciones del eje comparado: dos balanceos en `metricas_balanceo_semillas.csv`).
        La completitud de estas comparaciones se comprueba con su propio criterio,
        que es más estricto: cada bloque debe traer las DOS opciones en CADA una de
        las semillas esperadas, y si no, se aborta.
        """
        ruta = self._ruta_tabla_semillas(spec["tabla"])
        if not os.path.exists(ruta):
            raise RuntimeError(
                "No existe {}: la comparación pareada '{}' no se puede calcular. "
                "Lanza 'python app\\barrido_semillas.py' antes de agregar.".format(
                    ruta, spec["id"]))
        df = pd.read_csv(ruta)
        df = self._normalizar_algoritmo(df, spec, ruta)
        df = self._aplicar_filtro(df, spec, ruta)
        return df, ruta

    def _valores_por_semilla(self, spec, grupo, ambito, ruta):
        """
        {semilla: (valor_a, valor_b)} de un bloque, o aborta si el pareo no cierra.

        Aborta —no avisa— si a alguna semilla le falta una de las dos opciones, si
        alguna aparece repetida, si sobra una semilla ajena a la lista o si un valor
        no es numérico: un «gana en 8 de 10» calculado sobre 9 pares es exactamente
        el tipo de cifra que este script existe para no publicar.
        """
        eje, metrica = spec["eje"], spec["metrica"]
        opciones = (spec["opcion_a"], spec["opcion_b"])
        pares, problemas = {}, []
        for semilla in self.semillas:
            filas = grupo[grupo["semilla"].astype(int) == int(semilla)]
            valores = []
            for opcion in opciones:
                sel = filas[filas[eje].astype(str) == str(opcion)]
                if len(sel) != 1:
                    problemas.append(
                        "semilla {} · '{}' = '{}': {} fila(s) y debe haber 1".format(
                            semilla, eje, opcion, len(sel)))
                    valores = None
                    break
                valor = pd.to_numeric(pd.Series([sel.iloc[0][metrica]]),
                                      errors="coerce").iloc[0]
                if valor != valor:
                    problemas.append(
                        "semilla {} · '{}' = '{}': '{}' no es numérico".format(
                            semilla, eje, opcion, metrica))
                    valores = None
                    break
                valores.append(float(valor))
            if valores is not None:
                pares[int(semilla)] = (valores[0], valores[1])
        ajenas = sorted(set(int(s) for s in grupo["semilla"]) - set(self.semillas))
        if ajenas:
            problemas.append("semillas ajenas a la lista esperada: {}".format(ajenas))
        if problemas:
            raise RuntimeError(
                "{} · comparación '{}' · {}: el pareo por semilla NO cierra y no se "
                "publica un recuento incompleto. Problemas:\n     - {}".format(
                    os.path.basename(ruta), spec["id"], ambito,
                    "\n     - ".join(problemas)))
        return pares

    def _decisiones_no_constantes_pareada(self, spec, grupo, ambito):
        """
        Reparto de las COLUMNAS_CONSTANCIA que VOLTEAN entre semillas dentro de cada
        opción del par, ya formateado para la columna 'decisiones_no_constantes'.

        POR QUÉ LA TABLA PAREADA TAMBIÉN LA LLEVA: su tabla hermana —la de dispersión— declara este reparto celda a celda, y son
        LAS MISMAS celdas. Sin él, «RandomForest gana a HistGradientBoosting en 8 de
        10» se lee como una comparación entre dos configuraciones FIJAS, y no lo es:
        el `balanceo` ganador de RandomForest voltea entre semillas (es uno de los
        avisos que el .md ya lista). Es material de 5.4, igual que allí, y por eso se
        avisa sin abortar.

        SE EXCLUYE EL PROPIO EJE de la comprobación: en el eje `balanceo` la columna
        `balanceo` es no constante POR CONSTRUCCIÓN (son las dos opciones que se
        enfrentan), y contarla como decisión que voltea sería un aviso vacío que
        enseñaría a ignorar los demás. Se mira DENTRO de cada opción, no en la mezcla.
        """
        repartos = []
        for opcion in (spec["opcion_a"], spec["opcion_b"]):
            sub = grupo[grupo[spec["eje"]].astype(str) == str(opcion)]
            for columna in self.COLUMNAS_CONSTANCIA:
                if columna == spec["eje"] or columna not in sub.columns:
                    continue
                reparto = sub[columna].astype(str).value_counts()
                if len(reparto) <= 1:
                    continue
                detalle = " · ".join(
                    "{} en {} semilla(s)".format(valor, int(veces))
                    for valor, veces in reparto.items())
                # Sin comas: viaja como VALOR dentro del CSV (misma regla que los
                # textos de alcance de config.py).
                repartos.append("{} / {}: {}".format(opcion, columna, detalle))
                self.avisos.append(
                    "Comparación '{}' · {}: la opción '{}' NO mantiene constante la "
                    "columna de DECISIÓN '{}' entre las {} semillas ({}). El "
                    "recuento pareado enfrenta configuraciones que cambian de "
                    "semilla en semilla: hay que declararlo al citar el titular "
                    "(material de 5.4, no un error).".format(
                        spec["id"], ambito, opcion, columna, len(self.semillas),
                        detalle))
        return " | ".join(repartos)

    def _fila_pareada(self, spec, ambito, set_features, algoritmo, pares,
                      alcance, commits, decisiones=""):
        """
        Una fila de la tabla de comparaciones pareadas a partir de {semilla: (a, b)}.

        El recuento lo hace el script y se COMPRUEBA que cierre: victorias de A,
        victorias de B y empates tienen que sumar el número de pares. Un empate
        NO cuenta como victoria de nadie y va en su propia columna: si se repartiese
        a favor de A, «gana en N de M» dependería de una convención invisible.
        """
        gana_a = gana_b = empates = 0
        diferencias = []
        semillas_a, semillas_b, semillas_empate = [], [], []
        for semilla in sorted(pares):
            valor_a, valor_b = pares[semilla]
            # 'mayor es mejor' se declara en la ESPECIFICACIÓN y no se asume: en
            # estas tablas conviven métricas a maximizar (f1_macro) con columnas
            # donde menos es mejor (un fpr), y dar por hecho el sentido convertiría
            # el titular en su contrario sin que nada lo delatase.
            diferencia = (valor_a - valor_b if spec["mayor_es_mejor"]
                          else valor_b - valor_a)
            diferencias.append(diferencia)
            if valor_a == valor_b:
                empates += 1
                semillas_empate.append(semilla)
            elif diferencia > 0:
                gana_a += 1
                semillas_a.append(semilla)
            else:
                gana_b += 1
                semillas_b.append(semilla)
        n_pares = len(pares)
        if gana_a + gana_b + empates != n_pares:
            raise RuntimeError(
                "Comparación '{}' · {}: {} + {} + {} no suma los {} pares.".format(
                    spec["id"], ambito, gana_a, gana_b, empates, n_pares))
        if ambito != self.AMBITO_GLOBAL and n_pares != len(self.semillas):
            raise RuntimeError(
                "Comparación '{}' · {}: {} pares y se esperaban {} (uno por "
                "semilla).".format(spec["id"], ambito, n_pares,
                                   len(self.semillas)))
        arr = np.array(diferencias, dtype=float)
        return {
            "comparacion": spec["id"],
            "tabla_origen": config.nombre_tabla_semillas(spec["tabla"]),
            "eje": spec["eje"],
            "opcion_a": spec["opcion_a"],
            "opcion_b": spec["opcion_b"],
            "metrica": spec["metrica"],
            "mayor_es_mejor": bool(spec["mayor_es_mejor"]),
            "ambito": ambito,
            "set_features": set_features,
            "algoritmo": algoritmo,
            "n_pares": n_pares,
            "gana_a": gana_a,
            "gana_b": gana_b,
            "empates": empates,
            # El titular ya redactado, para que quien cite «8 de 10» copie la
            # cadena del artefacto en vez de recomponerla de dos columnas.
            "titular": "{} gana en {} de {}".format(
                spec["opcion_a"], gana_a, n_pares),
            "semillas_gana_a": " ".join(str(s) for s in semillas_a),
            "semillas_gana_b": " ".join(str(s) for s in semillas_b),
            "semillas_empate": " ".join(str(s) for s in semillas_empate),
            # Diferencia PAREADA (a favor de A con el signo ya orientado por
            # 'mayor_es_mejor'). Es lo que aporta el pareo sobre comparar dos
            # bandas: media y extremos de la diferencia DENTRO de cada semilla.
            "media_diferencia": round(float(arr.mean()),
                                      self.DECIMALES_DISTANCIA),
            "min_diferencia": round(float(arr.min()), self.DECIMALES_DISTANCIA),
            "max_diferencia": round(float(arr.max()), self.DECIMALES_DISTANCIA),
            "semillas": " ".join(str(s) for s in self.semillas),
            # Reparto de las decisiones que VOLTEAN dentro de cada opción del par
            # (vacío = fueron las mismas en las diez semillas). Ver
            # _decisiones_no_constantes_pareada(): el titular «gana en N de M» no es
            # una comparación entre dos configuraciones fijas si esta columna trae
            # texto.
            "decisiones_no_constantes": decisiones,
            "alcance": alcance,
            "commits_origen": commits,
            "commit_agregador": self.commit,
            "fecha": self.fecha,
        }

    def _alcance_de_grupo(self, spec, grupo, ambito):
        """Texto de 'alcance' del bloque; avisa si no es el mismo en todas sus filas."""
        if "alcance" not in grupo.columns:
            return ""
        distintos = sorted(set(grupo["alcance"].astype(str)))
        if len(distintos) > 1:
            self.avisos.append(
                "Comparación '{}' · {}: la columna 'alcance' toma {} valores "
                "distintos entre las filas pareadas; las dos opciones del eje no "
                "declaran la misma medida.".format(
                    spec["id"], ambito, len(distintos)))
        return distintos[0] if distintos else ""

    def _comparar_pareada(self, spec):
        """
        Calcula una comparación pareada: por bloque y agregada ('__global__').

        POR QUÉ PAREADA Y NO POR BANDAS: las dos opciones comparadas comparten la
        semilla, y con ella el mismo split, la misma muestra y los mismos folds del
        StratifiedKFold. Comparar sus intervalos [mín, máx] tira esa información y
        puede dar «no establecido» donde el pareo da una ventaja consistente: es
        justo el par de lecturas que conviven en 5.2.4.1. La formulación pareada es
        la única que sostiene «un algoritmo es mejor que el otro», y por eso tiene
        que salir de un artefacto y no de un recuento a mano.
        """
        df, ruta = self._leer_para_pareadas(spec)
        columnas = (spec["eje"], spec["metrica"], "semilla") + spec["bloques"]
        faltan = [c for c in columnas if c not in df.columns]
        if faltan:
            raise RuntimeError(
                "{}: no trae las columnas {} que necesita la comparación pareada "
                "'{}'.".format(os.path.basename(ruta), faltan, spec["id"]))
        for columna, permitidos in (spec["filtro_bloque"] or {}).items():
            if columna not in df.columns:
                raise RuntimeError(
                    "{}: sin la columna '{}' del filtro de bloque de '{}'.".format(
                        os.path.basename(ruta), columna, spec["id"]))
            df = df[df[columna].astype(str).isin([str(v) for v in permitidos])]
        df = df[df[spec["eje"]].astype(str).isin(
            [str(spec["opcion_a"]), str(spec["opcion_b"])])]
        if df.empty:
            raise RuntimeError(
                "{}: ninguna fila casa con la comparación '{}' (eje '{}', opciones "
                "'{}' y '{}'): la comparación está declarada contra una tabla que "
                "no la contiene.".format(
                    os.path.basename(ruta), spec["id"], spec["eje"],
                    spec["opcion_a"], spec["opcion_b"]))

        filas_bloque, pares_globales, commits_todos = [], {}, set()
        decisiones_todas = []
        for clave, grupo in df.groupby(list(spec["bloques"]), dropna=False):
            clave = clave if isinstance(clave, tuple) else (clave,)
            etiquetas = dict(zip(spec["bloques"], (str(v) for v in clave)))
            ambito = " · ".join(etiquetas[c] for c in spec["bloques"])
            pares = self._valores_por_semilla(spec, grupo, ambito, ruta)
            commits = ""
            if self.COLUMNA_COMMIT in grupo.columns:
                vistos = sorted(set(grupo[self.COLUMNA_COMMIT].astype(str)))
                commits_todos.update(vistos)
                self.commits_vistos.update(vistos)
                commits = " ".join(vistos)
            decisiones = self._decisiones_no_constantes_pareada(
                spec, grupo, ambito)
            if decisiones:
                decisiones_todas.append("{} → {}".format(ambito, decisiones))
            filas_bloque.append(self._fila_pareada(
                spec, ambito,
                etiquetas.get("set_features", self.AMBITO_GLOBAL),
                etiquetas.get("algoritmo", self.AMBITO_GLOBAL),
                pares, self._alcance_de_grupo(spec, grupo, ambito), commits,
                decisiones))
            # Los pares de todos los bloques, reindexados para que el recuento
            # global sea el de TODAS las celdas (los «40 de 40» son 2 variantes x
            # 2 algoritmos x 10 semillas, no 10).
            for semilla, valores in pares.items():
                pares_globales[(ambito, semilla)] = valores

        esperados = len(filas_bloque) * len(self.semillas)
        if len(pares_globales) != esperados:
            raise RuntimeError(
                "Comparación '{}': {} pares globales y se esperaban {} ({} bloques "
                "x {} semillas).".format(
                    spec["id"], len(pares_globales), esperados,
                    len(filas_bloque), len(self.semillas)))
        if spec["pares_esperados"] is not None \
                and esperados != spec["pares_esperados"]:
            raise RuntimeError(
                "Comparación '{}': el barrido da {} pares y la ESPECIFICACIÓN "
                "declara {}. Uno de los dos está mal y el titular no se "
                "publica.".format(spec["id"], esperados,
                                  spec["pares_esperados"]))
        # La fila global arrastra el reparto de TODOS sus bloques, con el bloque
        # delante: es la fila cuyo titular se cita ('16 de 20'), así que es donde
        # menos puede faltar la salvedad.
        fila_global = self._fila_pareada(
            spec, self.AMBITO_GLOBAL, self.AMBITO_GLOBAL, self.AMBITO_GLOBAL,
            pares_globales, "", " ".join(sorted(commits_todos)),
            " | ".join(decisiones_todas))
        # La fila global tiene que cuadrar con la suma de sus bloques: si no, el
        # titular que se cita no sería el de las filas que lo acompañan.
        for columna in ("n_pares", "gana_a", "gana_b", "empates"):
            suma = sum(f[columna] for f in filas_bloque)
            if suma != fila_global[columna]:
                raise RuntimeError(
                    "Comparación '{}': la fila '{}' da {} = {} y sus bloques suman "
                    "{}.".format(spec["id"], self.AMBITO_GLOBAL, columna,
                                 fila_global[columna], suma))
        self.filas_pareadas.extend(filas_bloque + [fila_global])
        print("   {}: comparación pareada '{}' · {} bloques x {} semillas = {} "
              "pares · {}".format(
                  os.path.basename(ruta), spec["id"], len(filas_bloque),
                  len(self.semillas), fila_global["n_pares"],
                  fila_global["titular"]))

    def _escribir_csv_pareadas(self):
        ruta = os.path.join(config.RESULTADOS_DIR, self.NOMBRE_CSV_PAREADAS)
        df = pd.DataFrame(self.filas_pareadas)
        # Orden estable: la fila '__global__' de cada comparación al final de su
        # bloque, que es donde se lee el titular.
        df["_orden"] = (df["ambito"] == self.AMBITO_GLOBAL).astype(int)
        df = df.sort_values(
            ["comparacion", "_orden", "set_features", "algoritmo"]
        ).drop(columns=["_orden"])
        df.to_csv(ruta, index=False)
        print("   Comparaciones pareadas (CSV): {}".format(ruta))
        return ruta

    def _seccion_pareadas_md(self):
        """Sección del .md con el titular de cada comparación y su desglose."""
        if not self.filas_pareadas:
            return []
        partes = [
            "## Comparaciones pareadas por semilla",
            "",
            "Cada fila enfrenta **dos opciones dentro de la MISMA semilla** —y por "
            "tanto sobre el mismo split, la misma muestra y los mismos folds—, y "
            "cuenta en cuántas semillas gana cada una. Los recuentos los calcula y "
            "los CUADRA el agregador (victorias + empates = pares, y la fila "
            "`__global__` contra la suma de sus bloques): **no se cuentan a mano**. "
            "Tabla completa en `comparaciones_pareadas.csv`.",
            "",
            # CADA LÍNEA DEL CALLOUT, UN ELEMENTO DE `partes`: escritas como
            # literales adyacentes se concatenarían en UN solo elemento, y los '> '
            # de continuación acabarían incrustados a mitad de frase en el .md.
            "> [!warning] No confundir con la comparación de bandas",
            "> Un solapamiento de los intervalos [mín, máx] de las tablas de abajo",
            "> **no niega** una ventaja pareada consistente: comparar bandas tira la",
            "> información de que las dos opciones comparten la semilla. La",
            "> formulación pareada es la que sostiene «un algoritmo es mejor que el",
            "> otro»; la de bandas solo dice que la ventaja no es una distancia fija.",
            "",
            # SALVEDAD DEL HOMÓNIMO EN EL ARTEFACTO, no solo en el código: el .md
            # imprime «gana en 8 de 10» y lleva el bloque de TIEMPOS —donde vive el
            # otro «8 de 10»— en el MISMO documento.
            "> [!warning] «8 de 10» es un homónimo en este proyecto",
            "> Los recuentos de esta sección son de **CALIDAD** (`f1_macro` sobre D2",
            "> y `f1_macro_cv` del CV de balanceo). Hay otro «8 de 10» en la memoria",
            "> que es de **TIEMPOS de entrenamiento** (8 de 10 pares intra-corrida)",
            "> y que además vive en el bloque",
            "> «Dispersión de máquina» de ESTE mismo documento, más abajo. No son la",
            "> misma cifra, no se fusionan y no se citan la una por la otra.",
            "",
            "| Comparación | Ámbito | Métrica | Pares | Gana A | Gana B | Empates | "
            "Media dif. (A−B) | Titular |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for f in self.filas_pareadas:
            es_global = f["ambito"] == self.AMBITO_GLOBAL
            # El titular de la fila `__global__` se rotula como lo que es —la SUMA
            # de los bloques de encima—, no como el titular portante: «16 de 20» no
            # es una medición nueva, son los dos «8 de 10» sumados, y sin el rótulo
            # invita a citarlo en lugar de ellos.
            titular = ("**{}** _(agregado de sus bloques)_".format(f["titular"])
                       if es_global else f["titular"])
            partes.append(
                "| `{}` vs `{}` | {} | `{}` | {} | {} | {} | {} | {:+.6f} | {} "
                "|".format(
                    f["opcion_a"], f["opcion_b"],
                    "**{}**".format(f["ambito"]) if es_global else f["ambito"],
                    f["metrica"], f["n_pares"], f["gana_a"], f["gana_b"],
                    f["empates"], f["media_diferencia"], titular))
        partes += [
            "",
            "«Gana A» es la primera opción del par y «Gana B» la segunda. Los "
            "empates van en su propia columna y **no se reparten**: no cuentan como "
            "victoria de nadie. La diferencia es (A − B) ya orientada a «mayor es "
            "mejor»; el CSV trae además su mínimo y su máximo, la lista de "
            "semillas en que gana cada opción y la columna "
            "`decisiones_no_constantes`.",
            "",
            "**La fila `__global__` no es un titular aparte:** es la suma de los "
            "bloques de su comparación (por eso el agregador la CUADRA contra ellos). "
            "Lo que se cita es el recuento del bloque —«8 de 10» en cada variante—; "
            "el agregado sirve para ver si los bloques apuntan al mismo lado, no para "
            "sustituirlos.",
            "",
        ]
        partes += self._decisiones_pareadas_md()
        partes += self._ejes_indistinguibles_md()
        return partes

    def _decisiones_pareadas_md(self):
        """
        Las comparaciones cuyo par NO enfrenta configuraciones fijas, listadas.

        La tabla hermana (dispersión) ya declara este reparto celda a celda y son las
        mismas celdas: si el `balanceo` ganador de una de las dos opciones voltea
        entre semillas, «gana en N de M» no es una comparación entre dos
        configuraciones fijas y hay que decirlo donde se lee el recuento.
        """
        con_reparto = [f for f in self.filas_pareadas
                       if f["ambito"] != self.AMBITO_GLOBAL
                       and f["decisiones_no_constantes"]]
        if not con_reparto:
            return []
        partes = [
            "> [!warning] Estos pares no enfrentan configuraciones FIJAS",
            "> En los bloques de abajo, alguna de las dos opciones cambió de "
            "decisión",
            "> entre semillas (el `balanceo` ganador de 4.3.4 o el `best_params_` "
            "del",
            "> GridSearchCV dependen de los folds). El recuento sigue siendo válido "
            "—cada",
            "> par comparte semilla, split y folds—, pero **no** se puede leer como "
            "«esta",
            "> configuración gana a esta otra». Es material de 5.4, igual que en la "
            "tabla",
            "> de dispersión, que lo declara sobre estas mismas celdas.",
        ]
        for f in con_reparto:
            partes.append("> - `{}` vs `{}` · {}: {}".format(
                f["opcion_a"], f["opcion_b"], f["ambito"],
                f["decisiones_no_constantes"]))
        partes.append("")
        return partes

    # Umbral por debajo del cual la media de la diferencia pareada se imprime como
    # indistinguible de cero: es la resolución con la que se publica esa columna
    # (DECIMALES_DISTANCIA = 6), así que por debajo de media unidad del último
    # decimal el signo del titular no lo sostiene el dato.
    UMBRAL_DIFERENCIA_NULA = 0.5 * 10 ** -6

    def _ejes_indistinguibles_md(self):
        """
        Avisa de los ejes cuyo agregado gana «por mayoría» con media de diferencia
        indistinguible de cero.

        POR QUÉ: `SMOTE vs class_weight` sale «SMOTE gana en 23 de 40» con media
        +0.000000. Leído como titular diría «SMOTE gana», y no dice eso: dice que el
        eje NO DECIDE NADA. El recuento se publica igual —silenciarlo sería elegir
        qué resultados se ven—, pero acompañado de su magnitud.
        """
        sospechosos = [
            f for f in self.filas_pareadas
            if f["ambito"] == self.AMBITO_GLOBAL
            and abs(float(f["media_diferencia"])) < self.UMBRAL_DIFERENCIA_NULA]
        if not sospechosos:
            return []
        partes = [
            "> [!caution] Mayoría sin magnitud: estos ejes NO DECIDEN NADA",
            "> Ganar «en más de la mitad» de las semillas con una diferencia media "
            "de",
            "> +0.000000 no es ganar: es un eje indistinguible del azar a la "
            "resolución",
            "> con la que se publica la métrica. No se cita como «gana X».",
        ]
        for f in sospechosos:
            partes.append(
                "> - `{}` vs `{}` (`{}`): {} de {} pares, media de la diferencia "
                "{:+.6f} · **no decide nada**".format(
                    f["opcion_a"], f["opcion_b"], f["metrica"], f["gana_a"],
                    f["n_pares"], f["media_diferencia"]))
        partes.append("")
        return partes

    # ------------------------------------------------------------------
    # 5. Salidas
    # ------------------------------------------------------------------
    def _escribir_csv(self):
        ruta = os.path.join(config.RESULTADOS_DIR, self.NOMBRE_CSV)
        df = pd.DataFrame(self.filas)
        # Orden estable y legible: primero la calidad, después la máquina, y
        # dentro por tabla de origen / variante / algoritmo / métrica. Sin esto el
        # orden lo decidiría el groupby y el fichero diffearía sin motivo.
        df["_orden_bloque"] = (df["bloque"] == self.BLOQUE_MAQUINA).astype(int)
        df = df.sort_values(
            ["_orden_bloque", "tabla_origen", "set_features", "algoritmo",
             "metrica"]
        ).drop(columns=["_orden_bloque"])
        df.to_csv(ruta, index=False)
        print("   Tabla de dispersión (CSV): {}".format(ruta))
        return ruta

    # Recorte del 'alcance' en el .md: el texto íntegro llega a 1.258 caracteres
    # (el de la cascada invertida) y dentro de una celda Markdown haría la tabla
    # ilegible. Se recorta a los primeros
    # LARGO_ALCANCE_MD caracteres, que BASTAN para distinguir los cinco alcances de
    # esta tabla —empiezan por «binario…», «cascada extremo a extremo…», «medida
    # CONTRAFACTUAL…», «multiclase 4 categorías…» y «multiclase 5 clases…»—, y el
    # texto completo queda en la columna 'alcance' del CSV, que es la fuente.
    LARGO_ALCANCE_MD = 70

    def _celda_md(self, texto):
        """Texto plano seguro dentro de una celda Markdown, recortado si es largo."""
        limpio = " ".join(str(texto).split()).replace("|", "\\|")
        if len(limpio) > self.LARGO_ALCANCE_MD:
            limpio = limpio[:self.LARGO_ALCANCE_MD].rstrip() + "…"
        return limpio

    def _tabla_md(self, df_bloque, con_titular=False):
        """
        Una tabla Markdown por bloque, ya formateada a 4 decimales.

        `con_titular` añade las dos columnas del titular de la semilla 42 (su
        valor y el veredicto dentro/fuera de la banda). Solo la tabla de CALIDAD
        las lleva: en el bloque de máquina esas columnas van vacías a propósito
        (ver _columnas_titular_vacias()) y una columna entera de guiones en el
        anexo solo invitaría a preguntar por un dato que no debe existir.

        LLEVA 'alcance' Y 'tabla_origen': sin ellas el .md —que es el artefacto
        que se pega en el anexo— rotularía «54 | RandomForest» tanto para el
        CLASIFICADOR DE FIRMAS como para la MEDICIÓN CONTRAFACTUAL de la cascada
        invertida, dos alcances incompatibles bajo la misma etiqueta (ver el
        comentario de 'alcance' en _agregar_metrica): el alcance de la fila de
        origen tiene que viajar con la cifra agregada.
        """
        cabecera = ("| Tabla de origen | Variante | Algoritmo | Alcance | Métrica "
                    "| n | Media | sd | Mín | Máx |")
        separador = "|---|---|---|---|---|---:|---:|---:|---:|---:|"
        if con_titular:
            cabecera += " Semilla 42 | ¿En banda? |"
            separador += "---:|---|"
        lineas = [cabecera, separador]
        for _, f in df_bloque.iterrows():
            def fmt(valor):
                return ("—" if valor != valor
                        else "{:.4f}".format(float(valor)))
            fila = "| `{}` | {} | {} | {} | `{}` | {} | {} | {} | {} | {} |".format(
                f["tabla_origen"], f["set_features"], f["algoritmo"],
                self._celda_md(f["alcance"]), f["metrica"], int(f["n"]),
                fmt(f["media"]), fmt(f["sd"]), fmt(f["min"]), fmt(f["max"]))
            if con_titular:
                valor_42 = f[self.COL_VALOR_42]
                veredicto = str(f[self.COL_DENTRO_42])
                texto_valor = ("—" if valor_42 == "" or valor_42 != valor_42
                               else "{:.4f}".format(float(valor_42)))
                # En NEGRITA solo el 'fuera': es lo que se cuenta en el titular y
                # lo que hay que poder localizar de un vistazo.
                texto_veredicto = {
                    self.VEREDICTO_DENTRO: "dentro",
                    self.VEREDICTO_FUERA: "**FUERA**",
                    self.VEREDICTO_SIN_CASAR: "_sin casar_",
                }.get(veredicto, "—")
                fila += " {} | {} |".format(texto_valor, texto_veredicto)
            lineas.append(fila)
        return "\n".join(lineas)

    def _recuento_titular(self):
        """
        (fuera, casadas) — el numerador y el DENOMINADOR del titular «N de M».

        `casadas` NO es "todas las celdas de calidad": es cuántas se pudieron casar
        con su fila publicada. Las que no casan quedan fuera de las dos cifras, y el
        aviso dice cuáles: un denominador que las incluyese haría pasar por
        "dentro de banda" a una celda que nunca se comparó.
        """
        return len(self.celdas_fuera_banda), (len(self.celdas_fuera_banda)
                                              + self.celdas_dentro_banda)

    def _nota_no_metricas_md(self):
        """
        Qué parte del denominador NO son métricas sobre D2, y qué pasaría al quitarlas.

        POR QUÉ ESTÁ AQUÍ: este `.md` se pega en el anexo de la memoria, y «N de M
        celdas de calidad» se lee como «N de M métricas». No lo es:
        entre las M hay celdas de `umbral` y `umbral_conf_elegido`, que son
        DECISIONES DEL PIPELINE recalculadas en cada semilla —el p95 sobre el 20 %
        de D1 y el UMBRAL_CONF por OOF—, no rendimiento sobre D2. Se agregan a
        propósito (son justo lo que el barrido pone a prueba) y NO se descuentan del
        titular, pero el lector tiene que poder verlas: sin esta nota, el anexo
        publicaría un denominador rotulado como métricas que no lo es del todo.
        Y el descuento no sería inocuo: uno de los empates de borde puede ser un
        umbral, y los empates —que cuentan como DENTRO— son justo lo que decide el
        recuento.
        """
        total = sum(self.celdas_no_d2.values())
        if not total:
            return []
        reparto = " · ".join(
            "{} de `{}`".format(n, m)
            for m, n in sorted(self.celdas_no_d2.items()))
        _, casadas = self._recuento_titular()
        partes = [
            "**No todas esas celdas son métricas sobre D2.** De las {} del "
            "denominador, **{} son umbrales** ({}): no miden rendimiento sobre D2 "
            "sino una decisión del pipeline que se recalcula en cada semilla (el "
            "p95 sobre el 20 % de D1 y el `UMBRAL_CONF` calibrado por OOF), y se "
            "agregan a propósito porque son justo lo que el barrido pone a prueba. "
            "**No se descuentan** del titular; si se descontasen, el denominador "
            "sería **{}** y la cifra habría que recontarla.".format(
                casadas, total, reparto, casadas - total),
            "",
        ]
        if self.celdas_borde:
            umbrales_borde = [c for c in self.celdas_borde if c["es_umbral"]]
            detalle = " · ".join(
                "{} {} `{}` (= {})".format(c["set_features"], c["algoritmo"],
                                           c["metrica"], c["extremo"])
                for c in self.celdas_borde)
            n_borde = len(self.celdas_borde)
            partes += [
                "Y el descuento no sería inocuo: **{} {}** en el borde exacto de su "
                "banda ({}), {} como **dentro** por el criterio declarado arriba y "
                "por eso {} el recuento — y **{}**.".format(
                    n_borde,
                    "celda cae" if n_borde == 1 else "celdas caen",
                    detalle,
                    "cuenta" if n_borde == 1 else "cuentan",
                    "decide" if n_borde == 1 else "deciden",
                    ("ninguna de ellas es un umbral" if not umbrales_borde
                     else ("una de ellas es un umbral" if len(umbrales_borde) == 1
                           else "{} de ellas son umbrales".format(
                               len(umbrales_borde))))),
                "",
            ]
        return partes

    def _reparto_commits_titular_md(self):
        """
        Cuántas celdas casadas aporta CADA commit de origen del titular, y cuántos
        de los «fuera de banda» salen de cada uno. Texto ya formateado ('' si no
        hay reparto).

        POR QUÉ: enumerar los commits del titular no dice con qué peso entra cada
        uno. Con la enumeración sola, «el titular viene de dos commits» se lee como si el
        reparto fuese cualquiera —mitad y mitad, o 97 y 1—, y de ese peso depende
        cuánta de la distancia listada abajo podría ser deriva de código. El
        recuento se CALCULA celda a celda (misma fuente que la columna
        `commit_semilla_42` del CSV), nunca se escribe a mano.

        SE COMPRUEBA QUE LA SUMA SEA EL DENOMINADOR «M»: si no lo fuese, el reparto
        estaría contando sobre otro conjunto de celdas que el titular y publicarlo
        al lado de «N de M» sería engañoso, así que se avisa.
        """
        if not self.recuento_titular_por_commit:
            return ""
        _, casadas = self._recuento_titular()
        suma = sum(d["casadas"]
                   for d in self.recuento_titular_por_commit.values())
        if suma != casadas:
            self.avisos.append(
                "El reparto de celdas por commit del titular suma {} y el "
                "denominador del titular es {}: los dos recuentos NO salen del "
                "mismo conjunto de celdas y el reparto no se puede leer al lado de "
                "«N de M».".format(suma, casadas))
        # Orden: primero el commit que más celdas aporta; a igualdad, por nombre.
        orden = sorted(self.recuento_titular_por_commit.items(),
                       key=lambda par: (-par[1]["casadas"], par[0]))
        trozos = []
        for etiqueta, datos in orden:
            nombre = (etiqueta if etiqueta == self.ETIQUETA_SIN_COMMIT_42
                      else "`{}`".format(etiqueta))
            n_fuera = datos["fuera"]
            if n_fuera == 0:
                cola = "ninguna fuera de banda"
            elif n_fuera == 1:
                cola = "1 fuera de banda"
            else:
                cola = "{} fuera de banda".format(n_fuera)
            trozos.append("{} aporta **{}** ({})".format(
                nombre, datos["casadas"], cola))
        return ("Reparto de las {} celdas casadas por commit de origen del "
                "titular (columna `commit_semilla_42`, contado por el "
                "agregador): {}.".format(casadas, " · ".join(trozos)))

    def _nota_procedencia_titular_md(self):
        """
        Salvedad de procedencia: el titular y la banda pueden venir de otro commit.

        POR QUÉ: _commits_de_celda avisa si una BANDA mezcla commits, pero no
        compara el commit de la fila publicada de la semilla 42 contra el de las
        diez del barrido. Si no coinciden, parte de la distancia que mide
        `distancia_fuera_banda_42` puede ser DERIVA DE CÓDIGO entre dos versiones y
        no dispersión por semilla.

        NO INVALIDA NADA y así se redacta: cada cifra es el resultado real de su
        corrida, y el titular publicado se produjo antes que el barrido por pura
        cronología del proyecto. Es una salvedad que hay que declarar al citar
        «N de M», no un motivo para no citarlo.
        """
        if not self.commits_titular:
            return []
        reparto = self._reparto_commits_titular_md()
        if self.commits_titular == self.commits_vistos:
            partes = [
                "Procedencia: el titular y las bandas salen del **mismo** commit "
                "({}), así que la comparación es entre corridas del mismo "
                "código.".format(", ".join("`{}`".format(c) for c in
                                           sorted(self.commits_titular))),
                "",
            ]
            if reparto:
                partes += [reparto, ""]
            return partes
        partes = [
            "> [!warning] **Salvedad de procedencia: el titular y la banda no salen "
            "del mismo commit.** Los valores de la semilla 42 vienen de {} "
            "(columna `commit_semilla_42` del CSV, celda a celda) y los diez puntos "
            "de cada banda de {} (columna `commits_origen`). Parte de la distancia "
            "que se lista abajo **podría ser deriva de código** entre esas "
            "versiones y no dispersión por semilla. No invalida ninguna cifra —cada "
            "una es el resultado real de su corrida—, pero hay que declararlo al "
            "citar «N de M».".format(
                ", ".join("`{}`".format(c) for c in sorted(self.commits_titular)),
                ", ".join("`{}`".format(c) for c in sorted(self.commits_vistos))
                or "_sin commit declarado_"),
            "",
        ]
        # El reparto va DENTRO de la salvedad, como continuación del callout: es el
        # dato que le da peso ("dos commits" con qué proporción) y separarlo lo
        # dejaría leyéndose como una nota independiente.
        if reparto:
            partes[0] += " {}".format(reparto)
        return partes

    def _seccion_titular_md(self):
        """
        Sección propia del titular: el recuento «N de M» y las celdas que caen fuera.

        ES LA RAZÓN DE SER de esta parte del script: que esta cifra salga del
        artefacto y no de un recuento a mano. El listado va ordenado por distancia DECRECIENTE
        —las desviaciones grandes primero— porque es el orden en el que se lee al
        redactar, y con la distancia a la vista para que las celdas que se salen por
        menos de lo que se ve a 4 decimales no se citen como si fuesen iguales a
        las que se salen por dos centésimas.
        """
        fuera, casadas = self._recuento_titular()
        partes = [
            "## El titular (semilla 42) frente a la banda",
            "",
            "**{} de {}** celdas de calidad tienen el valor de la semilla 42 "
            "**fuera** del intervalo [mín, máx] de las diez semillas del barrido. "
            "Esta cifra la calcula el agregador: no se cuenta a mano.".format(
                fuera, casadas),
            "",
            "Cómo se decide: el valor de la 42 se toma de su tabla PUBLICADA "
            "homóloga (misma variante, mismo algoritmo, mismo texto de `alcance`, "
            "`semilla` = 42; criterio completo en el encabezado del script), y se "
            "compara con los extremos **con la precisión con la que se persistió "
            "el CSV de origen** (6 decimales), no con el `mín`/`máx` a 4 decimales "
            "que publican las tablas de abajo: esos están redondeados para leerlos "
            "y decidir con ellos daría otro recuento. Un valor **igual** a un "
            "extremo cuenta como dentro.",
            "",
            "Solo el bloque de **calidad** entra: en el de máquina la banda mide "
            "carga de máquina y no el algoritmo, así que un «fuera» ahí no diría "
            "nada del sistema.",
            "",
        ]
        partes += self._nota_no_metricas_md()
        partes += self._nota_procedencia_titular_md()
        if self.celdas_sin_casar:
            partes += [
                "> [!warning] {} celda(s) de calidad NO se pudieron casar con su "
                "fila publicada y quedan fuera de las dos cifras de arriba (ni "
                "numerador ni denominador). El bloque de avisos dice cuáles y por "
                "qué.".format(self.celdas_sin_casar),
                "",
            ]
        if not self.celdas_fuera_banda:
            partes += ["Ninguna celda cae fuera de su banda.", ""]
            return partes
        partes += [
            "| Tabla de origen | Variante | Algoritmo | Alcance | Métrica | "
            "Semilla 42 | Mín (10) | Máx (10) | Distancia |",
            "|---|---|---|---|---|---:|---:|---:|---:|",
        ]
        for c in sorted(self.celdas_fuera_banda,
                        key=lambda x: x["distancia"], reverse=True):
            partes.append(
                "| `{}` | {} | {} | {} | `{}` | {:.4f} | {:.4f} | {:.4f} | "
                "{:.6f} |".format(
                    c["tabla_origen"], c["set_features"], c["algoritmo"],
                    self._celda_md(c["alcance"]), c["metrica"], c["valor_42"],
                    c["min"], c["max"], c["distancia"]))
        partes.append("")
        return partes

    def _comprobar_salidas_no_publicadas(self):
        """
        Ninguna ruta de escritura de este script puede ser la de una tabla publicada.

        Barato y protege del único accidente que arruinaría lo publicado: que una
        edición futura reutilizase estas constantes para escribir en una
        `metricas_*.csv`. Las cuatro principales borran por variante y su
        `CLAVE_UNICIDAD` no incluye `semilla`, así que una escritura ahí no se
        podría deshacer sin re-correr el pipeline entero.

        DOS COMPROBACIONES: (1) el NOMBRE no empieza por `metricas_`, y (2) la
        RUTA absoluta de cada salida no coincide con la de ninguna tabla publicada
        de la ESPECIFICACION —las mismas que se abren para el titular— ni con la de
        su homóloga del barrido. La primera sola es un chequeo de cadena: si
        alguien renombrase las salidas o cambiase la convención de nombres, pasaría
        limpia mientras escribe encima de una tabla. Comparar rutas normalizadas
        cierra ese hueco.
        """
        for nombre in (self.NOMBRE_CSV, self.NOMBRE_MD,
                       self.NOMBRE_CSV_PAREADAS):
            base = os.path.basename(nombre)
            if base.startswith("metricas_"):
                raise RuntimeError(
                    "La salida '{}' tiene nombre de tabla de métricas: este script "
                    "NO escribe en tablas publicadas.".format(base))

        def normalizar(ruta):
            return os.path.normcase(os.path.abspath(ruta))

        prohibidas = {}
        # Las de la ESPECIFICACION y las de las COMPARACIONES PAREADAS: la de
        # balanceo solo entra por la segunda, y dejarla fuera abriría un hueco
        # justo en esa tabla.
        for spec in self.ESPECIFICACION + self.COMPARACIONES_PAREADAS:
            for ruta in (self._ruta_tabla_publicada(spec),
                         self._ruta_tabla_semillas(spec["tabla"])):
                prohibidas[normalizar(ruta)] = ruta
        for nombre in (self.NOMBRE_CSV, self.NOMBRE_MD,
                       self.NOMBRE_CSV_PAREADAS):
            salida = os.path.join(config.RESULTADOS_DIR, nombre)
            choque = prohibidas.get(normalizar(salida))
            if choque is not None:
                raise RuntimeError(
                    "La salida '{}' apunta a la MISMA RUTA que la tabla '{}': este "
                    "script no escribe en tablas de métricas, ni publicadas ni del "
                    "barrido.".format(salida, choque))

    def _escribir_md(self):
        ruta = os.path.join(config.RESULTADOS_DIR, self.NOMBRE_MD)
        df = pd.DataFrame(self.filas)
        partes = [
            "# Dispersión entre semillas",
            "",
            "> Generado por `app/agregar_semillas.py`. **No se "
            "edita a mano**: se regenera corriendo el agregador.",
            # DOS sellos y no uno: el del agregador NO es el de las filas. Ver el
            # comentario de 'commits_origen' en _agregar_metrica().
            "> Commit del **agregador** (columna `commit_agregador` del CSV: con qué "
            "versión de `agregar_semillas.py` se calcularon media y sd): `{}` · "
            "Fecha: {}".format(self.commit, self.fecha),
            "> Commit(s) de las **filas agregadas** (con qué versión del código se "
            "produjeron los puntos de las bandas): {}. La columna "
            "`commits_origen` del CSV lo da por celda; si aquí hay más de uno, el "
            "bloque de avisos dice en qué celdas.".format(
                ", ".join("`{}`".format(c)
                          for c in sorted(self.commits_vistos))
                or "_ninguno: las tablas de origen no traen columna `commit`_"),
            "> Semillas agregadas ({}): {}.".format(
                len(self.semillas),
                ", ".join(str(s) for s in self.semillas)),
            "",
            "La **semilla 42 no entra en ninguna banda**, y es deliberado: es el "
            "titular de 5.1-5.3 y un punto **independiente**, no uno de los "
            "sumandos de su propia media (razón completa en el encabezado de "
            "`config.py`). `n`, `media`, `sd`, `mín` y `máx` son de las **diez** "
            "semillas del barrido. Lo que sí aparece es su "
            "valor **al lado** de la banda, para poder decir cuántas veces cae "
            "fuera sin contarlo a mano: ver «El titular frente a la banda».",
            "",
            "Lo que **sí varía** entre estas corridas y lo que **no**: varían el split 80/20 de D1 (y con él el umbral "
            "p95), la muestra de 5.000 de D3, la submuestra de 20.000 de "
            "OneClassSVM, los folds del CV y la inicialización de los modelos; "
            "**no** varían los splits D1/D2/D3 ni la selección de las 54 "
            "características, porque `program.py` no está parametrizado por "
            "semilla.",
            "",
            "`sd` es la desviación típica **muestral** (`ddof=1`). **Sin "
            "p-valor**: 10 puntos sobre un único dataset no sostienen un "
            "contraste, y la renuncia se declara.",
            "",
            "Las tablas de abajo son la vista legible, y llevan `tabla_origen` y "
            "`alcance` porque **sin el alcance la etiqueta es ambigua**: «54 · "
            "RandomForest» nombra por igual el CLASIFICADOR DE FIRMAS y la MEDICIÓN "
            "CONTRAFACTUAL de la cascada invertida, que son dos medidas distintas. "
            "El `alcance` va **recortado a {} caracteres** para que la tabla quepa; "
            "su texto íntegro está en la columna `alcance` del CSV, que es la "
            "fuente.".format(self.LARGO_ALCANCE_MD),
            "",
            "El CSV (`dispersion_semillas.csv`) trae por celda cinco columnas más "
            "que aquí no caben y que hay que mirar antes de citar una banda: "
            "`commits_origen`, `commit_agregador`, `decisiones_no_constantes` (el "
            "reparto del `balanceo`/`config_ganadora` cuando no fue el mismo en las "
            "diez semillas; vacío = lo fue), `distancia_fuera_banda_42` y "
            "`commit_semilla_42`.",
            "",
        ]
        partes += self._seccion_titular_md()
        partes += self._seccion_pareadas_md()
        partes += [
            "",
            "## Calidad",
            "",
        ]
        calidad = df[df["bloque"] == self.BLOQUE_CALIDAD].sort_values(
            ["tabla_origen", "set_features", "algoritmo", "metrica"])
        partes.append(self._tabla_md(calidad, con_titular=True))

        maquina = df[df["bloque"] == self.BLOQUE_MAQUINA].sort_values(
            ["tabla_origen", "set_features", "algoritmo", "metrica"])
        if len(maquina) > 0:
            partes += [
                "",
                "## Dispersión de máquina (NO es calidad y NO se cita como "
                "resultado)",
                "",
                "Bloque aparte a propósito: {}".format(
                    self.RAZON_BLOQUE_MAQUINA),
                "",
                self._tabla_md(maquina),
            ]

        if self.avisos:
            partes += ["", "## Avisos de la agregación", ""]
            partes += ["- {}".format(a) for a in self.avisos]

        partes.append("")
        with open(ruta, "w", encoding="utf-8") as f:
            f.write("\n".join(partes))
        print("   Tabla de dispersión (Markdown para el anexo de robustez): {}".format(ruta))
        return ruta

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def ejecutar(self):
        config.setup_utf8()
        config.ensure_dirs()
        self.commit = config.commit_actual()
        self.fecha = datetime.now().isoformat(timespec="seconds")

        print("=" * 70)
        print("AGREGACIÓN DE LA DISPERSIÓN ENTRE SEMILLAS")
        print("=" * 70)
        print("   Semillas esperadas ({}): {}".format(
            len(self.semillas), self.semillas))
        print("   La semilla {} NO entra en ninguna banda (es el titular "
              "publicado, punto independiente): solo se COMPARA contra "
              "ella".format(config.SEMILLA_POR_DEFECTO))
        print("   Bandas: SOLO desde las '*{}.csv'. Las publicadas se abren en "
              "LECTURA y solo para el valor del titular; ninguna se "
              "escribe".format(config.SUFIJO_TABLA_SEMILLAS))
        self._comprobar_salidas_no_publicadas()
        print("-" * 70)

        for spec in self.ESPECIFICACION:
            self._agregar_tabla(spec)

        if not self.filas:
            raise RuntimeError(
                "No se agregó ninguna celda: revisa las tablas del barrido.")

        for spec in self.COMPARACIONES_PAREADAS:
            self._comparar_pareada(spec)

        print("-" * 70)
        self._escribir_csv()
        self._escribir_csv_pareadas()
        self._escribir_md()
        for aviso in self.avisos:
            print("   [aviso] {}".format(aviso))
        print("=" * 70)
        print("AGREGACIÓN COMPLETADA: {} filas ({} de calidad · {} de "
              "máquina)".format(
                  len(self.filas),
                  sum(1 for f in self.filas
                      if f["bloque"] == self.BLOQUE_CALIDAD),
                  sum(1 for f in self.filas
                      if f["bloque"] == self.BLOQUE_MAQUINA)))
        # EL TITULAR, emitido por el script (no se cuenta a mano).
        fuera, casadas = self._recuento_titular()
        print("TITULAR (semilla {}) FUERA DE BANDA: {} de {} celdas de calidad "
              "casadas".format(config.SEMILLA_POR_DEFECTO, fuera, casadas))
        # Reparto por commit de origen del titular, contado (no escrito a mano).
        # Se imprime desde el dict y no llamando a _reparto_commits_titular_md():
        # ese método puede añadir un aviso y llamarlo dos veces lo duplicaría.
        for etiqueta, datos in sorted(
                self.recuento_titular_por_commit.items(),
                key=lambda par: (-par[1]["casadas"], par[0])):
            print("   commit del titular '{}': {} celda(s) casada(s) · {} fuera de "
                  "banda".format(etiqueta, datos["casadas"], datos["fuera"]))
        if self.celdas_sin_casar:
            print("   ({} celda(s) de calidad SIN casar: no cuentan en ninguna de "
                  "las dos cifras — ver avisos)".format(self.celdas_sin_casar))
        # TITULARES PAREADOS, emitidos por el script (el «8 de 10» de firmas y el
        # «40 de 40» del balanceo).
        print("TITULARES PAREADOS (calidad; NO son los '8 de 10' de TIEMPOS):")
        for fila in self.filas_pareadas:
            if fila["ambito"] != self.AMBITO_GLOBAL:
                continue
            print("   {} · {} (vs {}): {} · empates {}".format(
                fila["comparacion"], fila["titular"], fila["opcion_b"],
                "{} pares".format(fila["n_pares"]), fila["empates"]))
        print("=" * 70)
        return self.filas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agrega las tablas del barrido de semillas "
                    "('metricas_*_semillas.csv') en la tabla de dispersión de "
                    "el anexo de robustez: n, media, sd muestral, mín y máx, más la comparación del "
                    "titular de la semilla 42 contra cada banda ('N de M celdas "
                    "fuera'). No entrena nada y no escribe en ninguna tabla "
                    "publicada: las abre en LECTURA solo para el valor del titular."
    )
    parser.add_argument(
        "--semillas", type=int, nargs="+", default=None,
        help="Semillas que se esperan en cada combinación (por defecto las {} de "
             "config.SEMILLAS_BARRIDO: {}). Solo para auditar un barrido parcial "
             "a mano: lo que se publica en el anexo de robustez se agrega con la lista "
             "completa.".format(len(config.SEMILLAS_BARRIDO),
                                config.SEMILLAS_BARRIDO),
    )
    args = parser.parse_args()

    # Referencia explícita para que quede claro que este script NO redefine el
    # esquema: la clave de agrupación es la CLAVE_UNICIDAD de evaluacion.py sin
    # `semilla`, que es el eje sobre el que se agrega.
    assert AgregadorSemillas.CLAVE_AGRUPACION == evaluacion.CLAVE_UNICIDAD, (
        "La clave de agrupación debe seguir siendo evaluacion.CLAVE_UNICIDAD")

    AgregadorSemillas(semillas=args.semillas).ejecutar()
