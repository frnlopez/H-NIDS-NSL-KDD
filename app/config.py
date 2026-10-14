# -*- coding: utf-8 -*-
"""
config.py — Configuración central de los modelos del H-NIDS.

Consumido por evaluacion.py, anomalias.py, firmas.py, baseline.py, hibrido.py y
cascada_invertida.py. Centraliza rutas, semilla y convenciones de clase para que
sean IDÉNTICAS en todos ellos y no queden repetidas en cada script.

program.py NO depende de este módulo: define sus propias rutas relativas a la
raíz del repositorio.

validacion.py SÍ lo importa, por DOS motivos distintos:

  (1) PROCEDENCIA: llama a commit_actual() para estampar `commit` y `fecha`
      dentro de SEIS artefactos —que son TRES TIPOS × las DOS variantes, 54 y 122
      features—: la cabecera de los dos `*_validation_report.txt`, dos columnas de
      los dos `*_vocabulario_onehot.csv` y dos columnas de los dos
      `*_composicion_d3.csv`. Así se evita duplicar el mecanismo por copia.
  (2) VOCABULARIO DE CATEGORÍAS: lee CATEGORIAS_ATAQUE en medir_composicion_d3()
      SOLO para fijar el ORDEN de las filas del reparto de D3 y marcar la columna
      `declarada_en_config`. Nunca para inventar recuentos: esos salen de
      value_counts()/len(), y lo que no esté en el vocabulario se publica igual.

NOTA para quien edite este fichero: al importarse NO debe tener efectos. Nada de
llamar a ensure_dirs(), setup_utf8() ni tocar matplotlib/stdout a nivel de módulo,
y nada de importar módulos del proyecto (habría riesgo de ciclo). validacion.py
depende de ambas cosas.

---------------------------------------------------------------------------
CÓMO SE INYECTA LA SEMILLA — y por qué así
---------------------------------------------------------------------------
VÍA ELEGIDA: un flag de CLI `--semilla N` (argparse) en los cinco scripts
ejecutables —anomalias.py, firmas.py, baseline.py, hibrido.py y
cascada_invertida.py— que llama a `config.fijar_semilla(N)` en el bloque
`__main__`, ANTES de construir la clase del script. Una sola vía, la misma en
los cinco.

POR QUÉ EL FLAG Y NO UNA VARIABLE DE ENTORNO NI UN ARGUMENTO DE FUNCIÓN:
  * Los cinco consumidores leen `config.RANDOM_STATE` por ATRIBUTO DE MÓDULO en
    el momento de usarlo (nunca `from config import RANDOM_STATE`), así que
    mutar la global desde fijar_semilla() propaga a los ~15 sitios que la usan
    sin tocar ni una firma de función ni una llamada. Un argumento de función
    habría obligado a enhebrar `semilla=` por toda la cadena, y es justo la
    clase de cambio invasivo que puede alterar el comportamiento con 42.
  * El flag QUEDA EN EL COMANDO, así que el log del barrido dice con qué semilla
    se produjo cada pase. Una variable de entorno se pierde y deja el dato sin
    forma de comprobarse desde el log.
  * `fijar_semilla` es una FUNCIÓN: importar config.py sigue sin tener efectos, y
    validacion.py —que lo importa solo por commit_actual()— no se ve afectado.

EL DEFECTO ES 42. Correr cualquier script SIN `--semilla` produce los resultados
publicados: mismos nombres de artefactos, mismas cuatro tablas principales,
mismas figuras. random_state=42 se usa en TODO; `--semilla` es solo un override
explícito.

ORDEN DE LLAMADA — REQUISITO, no recomendación: fijar_semilla() debe llamarse
antes de instanciar la clase del script. Tres de las clases construyen un
`StratifiedKFold(random_state=config.RANDOM_STATE)` en su `__init__`
(firmas.py, baseline.py, hibrido.py), así que una llamada posterior dejaría ese
CV con la semilla anterior mientras el resto de la corrida usa la nueva. En los
cinco `__main__` la llamada va inmediatamente después de parse_args().

QUÉ PASA CON LOS ARTEFACTOS Y LAS TABLAS CUANDO LA SEMILLA NO ES 42 (las dos
reglas que impiden que el barrido pise lo publicado):
  1. NOMBRES DE ARTEFACTO. Los .joblib, las figuras y firmas_reglas_*.txt se
     sufijan con `sufijo_semilla()` = '_semilla<N>' vía `sufijo_artefactos()`.
     Con 42 el sufijo es CADENA VACÍA, así que los ficheros de la semilla 42 que
     sostienen las cifras publicadas conservan su nombre exacto y NINGUNA corrida
     del barrido los sobrescribe.
  2. TABLAS DE MÉTRICAS. `ruta_tabla()` desvía la escritura a una tabla NUEVA
     '<nombre>_semillas.csv'. Las nueve tablas publicadas no se abren siquiera.
     Es obligatorio y no una precaución: las cuatro principales borran por
     VARIANTE (limpiar_variante_csv), su CLAVE_UNICIDAD no incluye `semilla` y su
     recuento es fijo 4/4/1/1, así que 10 semillas en ellas colapsarían a la
     última o abortarían. Está escrito también en el docstring de
     evaluacion.comprobar_recuento().

QUÉ SEMILLAS SE BARREN Y POR QUÉ LA 42 NO ESTÁ ENTRE ELLAS (SEMILLAS_BARRIDO):
el barrido son las diez semillas 1..10 y la 42 queda FUERA a propósito. Dos
razones:
  1. `--semilla 42` es INDISTINGUIBLE de una corrida por defecto: sufijo_semilla()
     devuelve cadena vacía y ruta_tabla() no desvía nada, así que ese pase
     sobrescribiría las cuatro tablas publicadas, los 20 .joblib y las 39 figuras
     de la 42 — exactamente lo que todo este mecanismo existe para impedir.
  2. La 42 es el resultado principal de los apartados 5.1-5.3 de la memoria y por
     tanto un PUNTO INDEPENDIENTE de la banda: se cita como el resultado publicado
     y la banda de dispersión se cita al lado. Meterla dentro la convertiría en
     uno de los diez sumandos de su propia media, y la comparación «el resultado
     publicado cae dentro de la banda» dejaría de significar nada.
Consecuencia: son ONCE corridas por variante —diez del barrido más la
publicada—, no diez de las cuales una es la publicada.
"""
import os
import subprocess
import sys

# --- Reproducibilidad (random_state=42 en TODO) ---
# 42 es el DEFECTO y es la semilla de todo lo publicado. RANDOM_STATE es
# global mutable a propósito: fijar_semilla() la reescribe y los consumidores la
# leen por atributo (ver el bloque "CÓMO SE INYECTA LA SEMILLA" del encabezado).
SEMILLA_POR_DEFECTO = 42
RANDOM_STATE = SEMILLA_POR_DEFECTO

# --- Rutas base del proyecto ---
# Raíz del repositorio: este fichero vive en <raíz>/app/config.py → dos niveles
# arriba. El dataset va en <raíz>/dataset y las salidas en <raíz>/Resultados.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS_DIR = os.path.join(REPO_DIR, "Resultados")
DATASET_DIR = os.path.join(REPO_DIR, "dataset")
FIGURAS_DIR = os.path.join(RESULTADOS_DIR, "figuras")
MODELOS_DIR = os.path.join(RESULTADOS_DIR, "modelos")

# --- Prefijos de los CSV procesados (set de features parametrizable) ---
# 54 features = selección del apartado 4.3.5 aplicada (por defecto).
# 122 features = variante sin selección (generada con `program.py --sin-seleccion`).
# Los scripts de modelos eligen el set vía flag, no reescribiendo código.
BASE_PATH_54 = os.path.join(RESULTADOS_DIR, "specialized_nsl_kdd")
BASE_PATH_122 = os.path.join(RESULTADOS_DIR, "specialized_nsl_kdd_sin_seleccion")


def base_path(sin_seleccion=False):
    """Prefijo de los CSV procesados según la variante de features."""
    return BASE_PATH_122 if sin_seleccion else BASE_PATH_54


# ---------------------------------------------------------------------------
# Semilla parametrizable: inyección, nombres y desvío de tablas
# ---------------------------------------------------------------------------
# El razonamiento completo de la vía elegida está en el encabezado del módulo.
# Aquí solo el mecanismo. Tres piezas y ninguna más:
#   fijar_semilla()      la escribe (única forma legítima de cambiarla)
#   sufijo_artefactos()  la mete en el NOMBRE de joblibs, figuras y .txt
#   ruta_tabla()         desvía las métricas a la tabla nueva '*_semillas.csv'

# Marca que llevan los artefactos de una corrida con semilla distinta de 42.
_PLANTILLA_SUFIJO_SEMILLA = "_semilla{}"

# La misma marca SIN el número: '_semilla'. Es el discriminante entre "artefacto
# publicado" (no la lleva) y "artefacto del barrido" (sí), y lo usan el borrado por
# semilla y la verificación de los .joblib publicados de barrido_semillas.py. Se
# deriva de la plantilla en lugar de escribirse aparte para que no puedan
# divergir: si la plantilla cambiara, el filtro del borrado cambia con ella.
MARCA_SEMILLA = _PLANTILLA_SUFIJO_SEMILLA.format("")


def sufijo_de_semilla(valor):
    """
    Sufijo de artefacto de UNA semilla dada, sin mirar la global de la corrida.

    sufijo_semilla() devuelve el de config.RANDOM_STATE; esta compone el de una
    semilla arbitraria, que es lo que necesita el borrado del barrido
    (barrido_semillas.py) para construir '_semilla7.joblib' mientras la corrida en
    curso no tiene ninguna semilla fijada. Se mantiene aquí, junto a la plantilla,
    para que el filtro del borrado NO se escriba como un glob a mano: es la
    operación que podría alcanzar los .joblib publicados si divergiera.
    """
    return _PLANTILLA_SUFIJO_SEMILLA.format(int(valor))

# Sufijo del nombre de las tablas del barrido. La tabla '<x>.csv' publicada con la
# semilla 42 y la tabla '<x>_semillas.csv' del barrido son DOS FICHEROS: la primera
# no se abre en una corrida con semilla != 42.
SUFIJO_TABLA_SEMILLAS = "_semillas"


def fijar_semilla(valor):
    """
    Fija la semilla global de la corrida. ÚNICA forma legítima de cambiarla.

    Debe llamarse desde el `__main__` del script, justo después de parse_args() y
    ANTES de instanciar su clase (ver el encabezado: hay `StratifiedKFold` que se
    construyen en `__init__` y congelarían la semilla anterior).

    Parameters
    ----------
    valor : int
        Semilla entera y no negativa. `None` se acepta y se interpreta como "no
        se pidió semilla": deja RANDOM_STATE como está (42), de modo que el
        `__main__` puede llamar sin ramificar.

    Returns
    -------
    int : la semilla vigente tras la llamada.
    """
    global RANDOM_STATE
    if valor is None:
        return RANDOM_STATE
    try:
        semilla = int(valor)
    except (TypeError, ValueError):
        raise ValueError("La semilla debe ser un entero: " + repr(valor))
    if semilla < 0:
        # Un negativo reventaría dentro de sklearn con un mensaje peor, y además
        # produciría nombres de artefacto con guion ('_semilla-3').
        raise ValueError("La semilla debe ser >= 0: " + repr(valor))
    RANDOM_STATE = semilla
    return RANDOM_STATE


def es_semilla_por_defecto():
    """True si la corrida usa la semilla 42 (la de todo lo publicado)."""
    return RANDOM_STATE == SEMILLA_POR_DEFECTO


def sufijo_semilla():
    """
    Sufijo de la semilla para nombres de fichero: '' con 42, '_semilla<N>' si no.

    Que con 42 sea CADENA VACÍA es el requisito central del barrido: los artefactos
    de la semilla 42 —los que sostienen las cifras de 5.1-5.3— conservan su nombre
    exacto y ninguna corrida del barrido puede sobrescribirlos.
    """
    if es_semilla_por_defecto():
        return ""
    return _PLANTILLA_SUFIJO_SEMILLA.format(RANDOM_STATE)


def sufijo_artefactos(set_features):
    """
    Token que identifica una corrida en el NOMBRE de sus artefactos:
    '<set_features>' con la semilla 42 y '<set_features>_semilla<N>' con las demás.

    Lo usan los cinco scripts para los .joblib, las figuras y
    firmas_reglas_*.txt. NO sustituye a `set_features` en la COLUMNA homónima de
    los CSV: esa columna sigue valiendo '54' o '122_sin_seleccion' a secas (la
    semilla viaja en su propia columna, inyectada por
    evaluacion.guardar_metricas). Mezclar las dos cosas rompería el filtrado por
    variante y el vocabulario declarado del esquema de métricas.
    """
    return "{}{}".format(set_features, sufijo_semilla())


# Texto de ayuda del flag `--semilla`, único para los cinco scripts: la vía de
# inyección es la misma en todos y su documentación no debe divergir por copia.
AYUDA_CLI_SEMILLA = (
    "Semilla de la corrida (por defecto {}, la de todo lo publicado). Con "
    "cualquier otro valor la corrida NO toca las tablas publicadas —escribe en "
    "'metricas_*{}.csv'— ni los artefactos de la 42: sus .joblib y figuras se "
    "sufijan con '_semilla<N>'. Es la vía del barrido de dispersión; los "
    "titulares del capítulo 5 siguen siendo los de la semilla {}."
).format(SEMILLA_POR_DEFECTO, SUFIJO_TABLA_SEMILLAS, SEMILLA_POR_DEFECTO)


# ---------------------------------------------------------------------------
# Las diez semillas del barrido de dispersión
# ---------------------------------------------------------------------------
# Lista CERRADA. El razonamiento completo —por qué la 42 queda fuera— está en el
# encabezado del módulo, sección "QUÉ SEMILLAS SE BARREN"; en resumen: con 42 no
# habría desvío ni de artefactos ni de tablas (el pase pisaría lo publicado) y
# además la 42 es el resultado publicado, un punto independiente de la banda, no
# uno de sus diez sumandos.
#
# La consume barrido_semillas.py (el lanzador) y la comprueba agregar_semillas.py
# (el agregador aborta si a una combinación le faltan semillas de esta lista). Que
# viva aquí y no en cada script es lo que impide que las dos piezas divergan.
SEMILLAS_BARRIDO = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Invariantes de la lista, comprobados al importar: son barato y protegen de la
# única forma en que esta constante puede corromperse (una edición a mano). Un
# duplicado daría una "media de 10" con 9 puntos, y colar la 42 haría que una
# corrida del barrido sobrescribiese las cuatro tablas publicadas.
assert len(SEMILLAS_BARRIDO) == 10, (
    "SEMILLAS_BARRIDO debe tener 10 valores: el barrido es de 10 semillas")
assert len(set(SEMILLAS_BARRIDO)) == len(SEMILLAS_BARRIDO), (
    "SEMILLAS_BARRIDO tiene valores repetidos: la media saldría con menos "
    "puntos de los que declara")
assert SEMILLA_POR_DEFECTO not in SEMILLAS_BARRIDO, (
    "La semilla {} NO puede estar en SEMILLAS_BARRIDO: con ella no hay desvío "
    "de artefactos ni de tablas y el pase pisaría todo lo publicado".format(
        SEMILLA_POR_DEFECTO))
assert all(isinstance(s, int) and s >= 0 for s in SEMILLAS_BARRIDO), (
    "Las semillas de SEMILLAS_BARRIDO deben ser enteros no negativos")


def nombre_tabla_semillas(nombre_csv):
    """Nombre de la tabla del barrido correspondiente a una tabla publicada."""
    raiz, ext = os.path.splitext(nombre_csv)
    return raiz + SUFIJO_TABLA_SEMILLAS + ext


def es_tabla_de_semillas(nombre_csv):
    """True si el nombre (o la ruta) es una tabla del barrido de semillas."""
    raiz, _ = os.path.splitext(os.path.basename(nombre_csv))
    return raiz.endswith(SUFIJO_TABLA_SEMILLAS)


def nombre_tabla_base(nombre_csv):
    """Inversa de nombre_tabla_semillas(): '<x>_semillas.csv' -> '<x>.csv'."""
    nombre = os.path.basename(nombre_csv)
    raiz, ext = os.path.splitext(nombre)
    if raiz.endswith(SUFIJO_TABLA_SEMILLAS):
        raiz = raiz[:-len(SUFIJO_TABLA_SEMILLAS)]
    return raiz + ext


def ruta_tabla(nombre_csv):
    """
    Ruta de una tabla de métricas, DESVIADA a '*_semillas.csv' si la semilla no
    es 42. Los scripts construyen así las nueve rutas: ninguno concatena ya el
    nombre a RESULTADOS_DIR por su cuenta.

    Con la semilla 42 devuelve la ruta publicada de siempre. Con cualquier otra
    devuelve la tabla nueva, y por eso el barrido no puede tocar las cuatro
    principales ni las cinco auxiliares: no llega a abrirlas.
    """
    nombre = os.path.basename(nombre_csv)
    if not es_semilla_por_defecto():
        nombre = nombre_tabla_semillas(nombre)
    return os.path.join(RESULTADOS_DIR, nombre)


# --- Convenciones de clase — únicas para los tres modelos ---
# Binario: positivo = ataque = 1, normal = 0.
CLASE_NORMAL = 0
CLASE_ATAQUE = 1
ETIQUETA_NORMAL = "normal"
ETIQUETA_UNKNOWN = "unknown"  # solo la produce el híbrido (baja confianza / 0-day)

# Orden FIJO de categorías para matrices de confusión y reportes.
CATEGORIAS_ATAQUE = ["dos", "probe", "r2l", "u2r"]                  # 4 clases de firmas (D3)
CATEGORIAS_MULTICLASE = [ETIQUETA_NORMAL] + CATEGORIAS_ATAQUE       # 5 clases
CATEGORIAS_HIBRIDO = CATEGORIAS_MULTICLASE + [ETIQUETA_UNKNOWN]     # 6 (con 0-day)


# ---------------------------------------------------------------------------
# ALCANCE de cada fila de métricas
# ---------------------------------------------------------------------------
# Problema que resuelve: hay columnas HOMÓNIMAS con alcances distintos entre
# tablas. `accuracy_D2` vale 0,9683 en metricas_firmas.csv (4 clases, solo los
# ataques de D2 de tipo conocido) y 0,7395 en metricas_baseline.csv (5 clases,
# D2 entero). Mismo nombre, dos cosas. Por eso cada fila declara su alcance en
# una columna `alcance` con estos valores fijos (uno por tabla), de modo que
# ninguna cifra se pueda leer fuera de su contexto ni comparar con otra que no
# le corresponde.
#
# Sin comas: estos textos viajan como valor dentro de un CSV.
ALCANCE_ANOMALIAS = "binario normal-vs-ataque (2 clases) sobre D2 completo"
ALCANCE_FIRMAS = ("multiclase 4 categorías de ataque sobre los ataques de D2 "
                  "de tipo conocido")
ALCANCE_BASELINE = "multiclase 5 clases (normal + 4 ataques) sobre D2 completo"
# El añadido sobre 'tiempo_entrenamiento_s' NO es adorno: es el único sitio del
# DATO publicado donde se avisa de que esa columna homónima no mide lo mismo aquí
# que en las otras tres tablas (el híbrido no re-entrena la cascada; ver
# hibrido.SIGNIFICADO_TIEMPO_ENTRENAMIENTO). La columna NO se renombra: el
# conjunto mínimo homogéneo entre las cuatro tablas es requisito del esquema. Así, quien
# lea la fila suelta sin abrir el código ya sabe qué tiene delante.
#
# Segundo aviso que viaja en el dato: 'fpr_cascada' y 'bin_fpr' son literalmente
# el mismo número en la misma fila (hibrido.py los redondea del mismo b["fpr"]).
# No es un fallo: la etapa 2 solo reclasifica lo que la etapa 1 marcó y NUNCA
# devuelve un flujo a 'normal' así que la cascada no puede tener otro FPR que el
# de su etapa 1. Se conservan las DOS columnas —'bin_fpr' cierra el bloque binario
# completo y 'fpr_cascada' es el FPR que debe acompañar al recall de 0-day— pero
# el esquema no admite dos nombres para una magnitud sin declararlo. Se declara aquí.
ALCANCE_HIBRIDO = ("cascada extremo a extremo (5 clases + unknown) sobre D2 "
                   "completo; ojo: 'tiempo_entrenamiento_s' aquí es SOLO la "
                   "calibración de UMBRAL_CONF por OOF sobre D3 — el híbrido no "
                   "re-entrena las dos etapas. Y ojo también: 'fpr_cascada' y "
                   "'bin_fpr' son EL MISMO NÚMERO por construcción y no dos "
                   "medidas — la etapa 2 solo reclasifica los flujos que la etapa "
                   "1 marcó como sospechosos y nunca devuelve uno a 'normal' así "
                   "que el FPR de la cascada es por fuerza el de la etapa 1; se "
                   "publican las dos porque el FPR acompaña al titular de "
                   "0-day y el bloque binario se publica completo")
ALCANCE_0DAY = ("recall de marcado como ataque sobre los ataques de D2 de tipo "
                "ausente del train")

# Alcance de la tabla de la CASCADA INVERTIDA. No es ALCANCE_FIRMAS ni
# ALCANCE_D2_REPORTE: aquellos describen medidas sobre ATAQUES y esta se calcula
# justo sobre lo contrario —las filas NORMALES de D2—, que en el sistema real no
# llegan nunca a la etapa 2. Es una medida CONTRAFACTUAL y así debe leerse: no
# describe el híbrido publicado sino lo que pasaría con el orden invertido.
#
# Aquí va SOLO lo estable —el contrafactual, la población, el denominador, la
# procedencia del umbral y que D2 solo se reporta—, porque este texto se repite
# en las 10 filas del CSV y un CSV no se edita sin re-correr.
#
# Dos matices de lectura:
#   1. La fila '__global__' NO es el FPR que tendría un sistema de firmas-primero:
#      es una COTA INFERIOR, porque 'unknown' cuenta como alarma y no como
#      'normal'.
#   2. Las cuatro categorías suman la fila '__global__' en los RECUENTOS; las
#      'tasa_condena' publicadas no exactamente, por el redondeo a 6 decimales
#      (en la variante de 122 suman 0.342808 frente al 0.342807 de '__global__').
# Sin comas: estos textos viajan como valor dentro de un CSV.
ALCANCE_CASCADA_INVERTIDA = (
    "medida CONTRAFACTUAL de la cascada invertida: predict_proba del "
    "clasificador de firmas YA ENTRENADO sobre la POBLACIÓN de las filas normales "
    "de D2 —que en el sistema real NUNCA llegan a la etapa 2— para contar cuántas "
    "conservarían etiqueta de ataque con confianza >= UMBRAL_CONF. NO describe el "
    "híbrido publicado y NO es comparable con ninguna columna de "
    "metricas_firmas.csv ni de metricas_hibrido.csv. La etapa 2 carece de clase "
    "'normal' (D3 son solo ataques) así que etiqueta como ataque TODA fila: por "
    "eso 'n_argmax' de '__global__' es el total de normales. DENOMINADOR de "
    "'tasa_condena': ese total y no el bucket de la fila —así los RECUENTOS de las "
    "cuatro categorías suman el de '__global__'; las tasas solo hasta el "
    "redondeo—. LÍMITE: 'tasa_condena' de '__global__' es COTA INFERIOR del FPR de "
    "un sistema de firmas-primero y lo citable es 'n_condenadas' — la cota es inferior "
    "porque 'unknown' cuenta como alarma y no como 'normal'. UMBRAL_CONF se lee del descriptor del "
    "híbrido (ver 'origen_umbral_conf') donde lo dejó la calibración out-of-fold "
    "sobre D3. Cero fit: solo inferencia sobre modelos persistidos y D2 SOLO se "
    "reporta — ninguna decisión sale de esta tabla"
)

# Alcance de la tabla auxiliar del CRUCE FILA A FILA entre detectores de la etapa 1
# (`metricas_cruce_detectores.csv`, script `cruce_detectores.py`).
#
# Por qué no le vale ALCANCE_0DAY: aquella tabla publica RECUENTOS POR DETECTOR
# —cuántas filas ve cada uno— y de ahí NO se deduce si dos detectores ven LAS
# MISMAS filas o filas distintas. Esta tabla mide justamente eso: la intersección
# y la diferencia de los CONJUNTOS de filas marcadas sobre una MISMA población.
# Las dos cifras no son intercambiables y leer una como la otra es el error que
# esta medición viene a cerrar.
#
# LÍMITE que la fila no puede perder de vista: una unión medida NO es un detector.
# 'union_ab' cuenta las filas que vería un sistema que corriese los dos detectores
# en paralelo y disparase con cualquiera de los dos; ese sistema tendría también la
# unión de sus FALSOS POSITIVOS y ningún FPR conjunto está medido aquí. La columna
# es un argumento MECANÍSTICO sobre complementariedad —qué tipos caza cada uno— y
# no el rendimiento de una fusión. Sin comas: este texto viaja dentro de un CSV.
ALCANCE_CRUCE_DETECTORES = (
    "cruce FILA A FILA de los conjuntos de filas marcadas por cada detector de la "
    "etapa 1 sobre las filas de D2 del RÉGIMEN CIEGO (los tipos 0-day con recall "
    "prácticamente nulo en la cascada publicada) con el umbral p95 ya persistido "
    "de cada detector. Cero fit: solo scoring sobre modelos persistidos y D2 SOLO "
    "se reporta. NO es una tabla de recuentos por detector —eso es "
    "metricas_hibrido_0day.csv— sino de intersección/diferencia entre detectores. "
    "'union_ab' y 'union_todos' NO son el rendimiento de una fusión en paralelo: "
    "esa fusión uniría también los falsos positivos y aquí NO hay ningún FPR "
    "conjunto medido"
)

# Alcance de los tres regímenes de la tabla de DEGRADACIÓN FUERA DE DOMINIO de la
# etapa 2 (`metricas_degradacion_firmas.csv`, script `degradacion_firmas.py`).
#
# El primero NO reutiliza ALCANCE_OOF_D3: aquel texto afirma que «de aquí SALE la
# decisión de UMBRAL_CONF» y de esta tabla no sale ninguna decisión. El segundo SÍ
# es literalmente ALCANCE_FIRMAS (misma población y mismas 4 clases) y por eso se
# reutiliza tal cual en lugar de duplicarlo con otras palabras: la fila de este
# régimen tiene que poder cotejarse contra metricas_firmas.csv columna a columna.
# El tercero es población NUEVA —ataques de D2 de CUALQUIER tipo— y no existe en
# ninguna tabla publicada. Sin comas: estos textos viajan dentro de un CSV.
ALCANCE_DEGRADACION_OOF_D3 = (
    "out-of-fold sobre D3 (StratifiedKFold 5 · shuffle · la semilla de la corrida "
    "está en la columna 'semilla') con la configuración YA ELEGIDA y persistida: "
    "es el régimen DENTRO de dominio contra el que se mide la degradación. NO "
    "selecciona nada y de esta tabla no sale ninguna decisión"
)
ALCANCE_DEGRADACION_D2_TODOS = (
    "multiclase 4 categorías de ataque sobre TODOS los ataques de D2 —incluidos "
    "los 0-day de tipo ausente del train— así que la etiqueta de categoría existe "
    "pero el TIPO puede no haberse visto nunca: es el régimen fuera de dominio más "
    "duro y NO es comparable con metricas_firmas.csv que excluye los 0-day. Solo "
    "reporte"
)

# Alcance del bloque `conocida_` del HÍBRIDO. NO es ALCANCE_FIRMAS aunque se
# calcule sobre las mismas filas de D2: en el híbrido la medida es de extremo a
# extremo y los ataques que la etapa 1 no marcó entran como 'normal', así que
# baja respecto a firmas-solo. Cuánto baja depende de QUÉ métrica se mire: en la
# variante de 54, el RECALL MACRO pasa de 0,850 (RandomForest en
# metricas_firmas.csv) a 0,671 ('conocida_recall_macro' en metricas_hibrido.csv),
# 17,9 pp, mientras que con f1_macro la caída son 7,4 pp (0,822 -> 0,748).
# Reutilizar el texto de firmas haría parecer la misma medida a dos cosas
# distintas.
ALCANCE_HIBRIDO_CONOCIDA = (
    "cascada extremo a extremo restringida a los ataques de D2 de tipo conocido "
    "(4 categorías); los ataques que la etapa 1 no marcó como sospechosos "
    "cuentan como 'normal' → NO comparable con la columna homónima de "
    "metricas_firmas.csv"
)

# Alcance de las columnas que salen de la SELECCIÓN DE HIPERPARÁMETROS / del
# ajuste del umbral. No son métricas sobre D2 y no deben leerse como tales:
#   auc_val        (anomalias) AUC-ROC sobre D1_val + muestra de D3
#   umbral         (anomalias) percentil 95 del score sobre D1_val
#   f1_macro_cv    (firmas, baseline, balanceo) f1_macro medio en CV sobre train
# Son sistemáticamente MÁS ALTAS que la métrica homóloga sobre D2 (IsolationForest
# 54: auc_val 0.9918 vs roc_auc 0.9229; baseline 54: f1_macro_cv 0.9094 vs
# f1_macro 0.4721), así que sin alcance propio invitan a citarlas como resultado.
ALCANCE_SELECCION = (
    "selección de hiperparámetros / ajuste del umbral: CV sobre el train "
    "(D3 o D1+D3); auc_val sobre D1_val + muestra de D3; umbral sobre D1_val "
    "solo — NUNCA D2"
)

# Alcance de la tabla auxiliar de BALANCEO (4.3.4). ALCANCE_SELECCION no le vale:
# su texto habla de 'auc_val' y de 'umbral', dos columnas que metricas_balanceo.csv
# NO tiene (su fila es algoritmo × balanceo y solo trae f1_macro_cv/_cv_std), así
# que describía media tabla ajena y ninguna de las suyas. Este sí dice lo que hay:
# de dónde sale el f1_macro_cv y por qué el SMOTE de esas filas no es leakage.
#
# SIN NÚMERO DE SEMILLA EN EL TEXTO: firmas.py escribe esta constante en la
# columna 'alcance' de metricas_balanceo.csv, así que un literal «semilla 42»
# produciría, en una corrida '--semilla 7', filas cuya columna 'semilla' dice 7 y
# cuyo 'alcance' afirma 42 — una afirmación falsa DENTRO del dato, y un CSV no se
# corrige sin re-correr. Tampoco se interpola RANDOM_STATE aquí: esta constante se
# evalúa AL IMPORTAR el módulo y fijar_semilla() se llama después, así que la
# interpolación congelaría el 42 igual.
# La semilla de la corrida viaja en su propia columna ('semilla', inyectada por
# evaluacion.guardar_metricas) y ahí no puede quedar desfasada.
ALCANCE_BALANCEO = (
    "selección del esquema de balanceo (4.3.4): f1_macro medio en CV sobre D3 "
    "(StratifiedKFold 5 · shuffle · la semilla de la corrida está en la columna "
    "'semilla') con SMOTE aplicado SOLO dentro de cada fold vía imblearn.Pipeline "
    "— nunca D2 y nunca citable como resultado"
)

ALCANCE_HIBRIDO_CALIBRACION = (
    "sensibilidad de UMBRAL_CONF: las columnas 'oof_' son out-of-fold sobre D3 "
    "y de ellas SALE la decisión; las columnas 'd2_' son sobre D2 completo y "
    "solo se REPORTAN (nunca deciden)"
)

# Los dos bloques de esa fila mixta, por separado.
# Sin número de semilla, por el mismo motivo que ALCANCE_BALANCEO: el CV lo
# construye hibrido.py con config.RANDOM_STATE, que en el barrido no es 42. Este
# texto es DOCUMENTACIÓN (ALCANCE_PREFIJOS_AUXILIARES, no se escribe en ningún CSV),
# pero un literal desfasado aquí acabaría copiado a un sitio donde sí importa.
ALCANCE_OOF_D3 = ("out-of-fold sobre D3 (StratifiedKFold 5 · shuffle · la semilla "
                  "de la corrida está en la columna 'semilla'): de aquí SALE la "
                  "decisión de UMBRAL_CONF — nunca D2")
ALCANCE_D2_REPORTE = ("sobre D2 completo y SOLO como reporte: ninguna decisión se "
                      "toma con estas columnas")

# ---------------------------------------------------------------------------
# La columna 'tiempo_s': tres cálculos y cuatro declaraciones — en el dato
# ---------------------------------------------------------------------------
# 'tiempo_s' tiene el mismo nombre en tres tablas y mide tres magnitudes
# distintas, así que el dato tiene que decirlo. Los CÁLCULOS son tres (bloque completo del algoritmo · solo entrenamiento ·
# tramo carga→cierre de fila) y los TEXTOS cuatro: anomalias.py y firmas.py
# comparten cálculo y NO comparten composición, así que un texto único no dejaba
# reconstruir ninguno de los dos. Cada fila de las cuatro tablas principales lleva
# una columna hermana 'alcance_tiempo_s' con uno de estos textos.
#
# NO se renombra la columna (el apartado 4.4 de la memoria la cita por su nombre)
# y NO se homogeneiza el cálculo (cada uno es el que describe su capítulo).
# Por eso 'tiempo_s' NO está en ALCANCE_COLUMNAS: ese diccionario da UN alcance
# por nombre de columna y aquí el alcance depende de la tabla. Va en el dato.
#
# ---------------------------------------------------------------------------
# QUÉ PUEDE ESCRIBIRSE EN ESTOS CUATRO TEXTOS
# ---------------------------------------------------------------------------
# Solo lo ESTABLE, es decir lo que se deduce del código y no de ninguna corrida:
#   - qué tramos entran en 'tiempo_s' y cuáles quedan fuera;
#   - qué columna mide cada tramo y cuál sale por resta;
#   - el aviso sobre qué mide (y qué NO) la latencia.
# PROHIBIDO en estos textos: porcentajes, bandas, factores de dispersión, cifras
# de ninguna corrida y cualquier atribución causal. Motivo: publicar
# interpretación DENTRO del dato crea un bucle — el dato no se puede editar sin
# re-correr y cada corrida nueva falsa la cifra que dejó la anterior.
# ---------------------------------------------------------------------------
#
# POR QUÉ perf_counter. Los cuatro scripts miden con time.perf_counter() y no con
# time.time(), que en Windows tiene una resolución de ~15,6 ms — suficiente para
# publicar un 'latencia_ms_por_flujo' = 0,0 en el predict del DecisionTree.
# perf_counter es monótono y NO tiene época: solo sirve para diferencias (la
# columna 'fecha' la sigue dando datetime.now()).
#
# LO QUE perf_counter NO ARREGLA: la varianza. Es wall-clock en una máquina no
# dedicada. Aquí NO se declara ninguna cota ni ningún factor.
# Sin comas: estos textos viajan como valor dentro de un CSV.
_AVISO_TIEMPO_VARIANZA = (
    " · wall-clock de un pase único en máquina no dedicada: NO es reproducible ni "
    "comparable entre corridas y NO es una propiedad del algoritmo — vale como "
    "orden de magnitud y como comparación relativa DENTRO de esta misma corrida. "
    "La dispersión medida entre corridas y el umbral por debajo del cual una "
    "diferencia entre dos filas no significa nada se documentan fuera del "
    "dato — anclados al commit de cada corrida y NO aquí a "
    "propósito: una cifra dentro del dato no se puede corregir sin re-correr"
)

# Lo que NO entra en la inferencia medida. Va pegado a los cuatro alcances de
# tiempo porque acompaña al par (latencia, caudal) que publica cada fila: sin esta
# frase, citar el caudal del DecisionTree como capacidad operativa sería
# exactamente el sesgo de evaluación solo en laboratorio (Lab-Only Evaluation).
_AVISO_LATENCIA_SOLO_PREDICT = (
    " · 'latencia_ms_por_flujo' y 'flujos_por_segundo' miden SOLO el predict/score "
    "sobre características ya calculadas y ya en memoria: NO incluyen captura de "
    "tráfico · ensamblado del flujo · extracción de las 41 características "
    "—donde vive el coste real de un despliegue— así que no son capacidad "
    "operativa del sistema"
)

# anomalias.py: bloque completo del algoritmo. El tramo que no es ajuste ni
# inferencia sobre D2 es grande; sus dos componentes principales se MIDEN y se publican como columnas propias
# ('tiempo_score_seleccion_s' y 'tiempo_score_umbral_s'), de modo que el reparto
# se lee de la propia fila y no hace falta estimarlo ni publicarlo aquí.
ALCANCE_TIEMPO_S_BLOQUE_ANOMALIAS = (
    "'tiempo_s' = el bloque completo del algoritmo. Sus cinco tramos principales "
    "en orden de ejecución: (1) los fit del grid — y solo ellos son "
    "'tiempo_entrenamiento_s' — (2) puntuar el set de validación etiquetado "
    "(D1_val + la muestra de D3) UNA VEZ POR CONFIGURACIÓN del grid = "
    "'tiempo_score_seleccion_s' (3) puntuar D1_val para fijar el umbral p95 = "
    "'tiempo_score_umbral_s' (4) la inferencia sobre D2 = 'tiempo_inferencia_s' y "
    "(5) la cola de evaluar_binario + UNA figura —la matriz de confusión; la de "
    "curvas ROC/PR se dibuja al final del script y queda fuera—. 'Principales' es "
    "literal: dentro de la misma ventana caen también el submuestreo de "
    "OneClassSVM · la construcción del estimador en cada iteración del grid y un "
    "roc_auc_score por configuración — despreciables pero no nulos — así que el "
    "tramo (5) no tiene columna propia: es lo que queda al restar de 'tiempo_s' "
    "las cuatro columnas de tiempo anteriores. El reparto entre tramos se LEE de "
    "las columnas de esta misma fila y no se estima. AVISO al reconstruirlo: los "
    "tramos (2) (3) y (4) miden la MISMA operación sobre conjuntos de tamaño "
    "distinto así que un cociente CRUDO entre dos de ellos NO es un cociente de "
    "coste por flujo: hay que dividirlo por la razón de tamaños de sus conjuntos "
    "antes de leerlo — y aun normalizado sigue siendo wall-clock de un pase único "
    "en una corrida que no controla ni el estado de caché ni la carga de la "
    "máquina ni n_jobs"
    + _AVISO_TIEMPO_VARIANZA + _AVISO_LATENCIA_SOLO_PREDICT
)

# firmas.py: mismo cálculo que el de anomalías y otra composición. Aquí el
# residual es UN SOLO tramo con coste apreciable —la cola de métricas + figura—
# así que sale EXACTO por resta de las tres columnas publicadas y no necesita
# columna propia. La selección de balanceo (4.3.4) NO está dentro.
ALCANCE_TIEMPO_S_BLOQUE_FIRMAS = (
    "'tiempo_s' = el bloque completo del algoritmo. Sus tres tramos en orden de "
    "ejecución: (1) el GridSearchCV con el balanceo ganador + el refit sobre todo "
    "D3 = 'tiempo_entrenamiento_s' (2) el predict sobre los flujos de D2 de tipo "
    "conocido = 'tiempo_inferencia_s' y (3) la cola de evaluar_multiclase + UNA "
    "figura —la matriz de confusión 4x4—. El (3) no lleva columna propia porque "
    "sale EXACTO "
    "restando de 'tiempo_s' las otras dos columnas de tiempo —salvo dos "
    "menudencias que caen en la misma ventana: leer busqueda.best_estimator_ y "
    "dos print—. NO incluye el mini-experimento de balanceo de 4.3.4 que corre "
    "antes y se publica en metricas_balanceo.csv"
    + _AVISO_TIEMPO_VARIANZA + _AVISO_LATENCIA_SOLO_PREDICT
)

# baseline.py: SOLO el GridSearchCV. Es el que rompe la comparación entre tablas.
# Dentro de la ventana del cronómetro solo hay la construcción del estimador y el
# GridSearchCV.fit, así que el residual 'tiempo_s' − 'tiempo_entrenamiento_s' es
# cero por construcción (salvo redondeo).
ALCANCE_TIEMPO_S_SOLO_ENTRENAMIENTO = (
    "'tiempo_s' = SOLO el entrenamiento (GridSearchCV + refit sobre D1+D3): "
    "coincide con 'tiempo_entrenamiento_s' de esta misma fila —el residual entre "
    "las dos columnas es cero salvo redondeo— y NO incluye inferencia ni figuras "
    "— no comparable con el 'tiempo_s' de metricas_anomalias.csv ni de "
    "metricas_firmas.csv" + _AVISO_TIEMPO_VARIANZA + _AVISO_LATENCIA_SOLO_PREDICT
)

# hibrido.py: el tramo de la corrida que va de la carga de datos al cierre de la
# fila resumen. NO es "el script entero": el cronómetro se lee al construir el
# dict metricas_run y después aún quedan la figura 5x6 la tabla 0-day de los
# cuatro detectores y la escritura de los CSV. Por eso el log final del script
# imprime una cifra MAYOR que esta columna: son dos medidas distintas y el texto
# lo dice para que nadie lea la discrepancia como un error. El cronómetro NO se
# mueve: reordenar la persistencia para que la fila se escriba con su propio
# tiempo dentro es un cambio de código que no aporta nada al dato.
ALCANCE_TIEMPO_S_CARGA_A_CIERRE_FILA = (
    "'tiempo_s' = el tramo de la corrida que va desde la carga de los splits "
    "hasta el cierre de esta fila (carga de D1/D2/D3 + carga de los .joblib de "
    "las dos etapas + calibración OOF sobre D3 = 'tiempo_entrenamiento_s' + "
    "cascada sobre D2 = 'tiempo_inferencia_s' + tabla de sensibilidad de los 3 "
    "umbrales); lo que queda al restar esas dos columnas son la carga de los "
    "splits los dos .joblib y las pasadas de métricas de la tabla de sensibilidad "
    "— ninguna columna los mide por separado. NO incluye lo que se hace DESPUÉS "
    "de tomar la medida: la figura 5x6 · la tabla 0-day de los cuatro detectores "
    "· la escritura de los CSV — de ahí que el log del script imprima una cifra "
    "mayor que esta; no es tiempo de ajuste —el híbrido no re-entrena— ni "
    "comparable con el 'tiempo_s' de las otras tres tablas"
    + _AVISO_TIEMPO_VARIANZA + _AVISO_LATENCIA_SOLO_PREDICT
)

# Convención de PREFIJOS y SUFIJOS de columna, complementaria a `alcance`. Una
# fila puede llevar más de un bloque de métricas; `alcance` describe el bloque
# PRINCIPAL (las columnas sin prefijo) y estos marcadores identifican los
# bloques secundarios, con el mismo significado en las cuatro tablas.
#
# ALCANCE (de este bloque): es DOCUMENTACIÓN, y está bien que lo sea. Ninguna de
# estas tres tablas (ALCANCE_PREFIJOS / ALCANCE_SUFIJOS / ALCANCE_COLUMNAS) la
# comprueba el código: validar_esquema_minimo() solo llega a alcance_de_columna()
# para columnas que casan con PREFIJOS_SIN_DECLARAR, y ninguna clave de aquí
# ('bin_', 'conocida_', 'recall_0day', '_cv', '_cv_std', '_val', 'umbral') puede
# casar con ese filtro. El único diccionario que el aviso resuelve de verdad es
# ALCANCE_PREFIJOS_AUXILIARES. Se declaran igual porque son el vocabulario que
# cita la memoria y el que siguen los scripts al nombrar sus columnas.
ALCANCE_PREFIJOS = {
    "bin_": ALCANCE_ANOMALIAS,
    "conocida_": ALCANCE_HIBRIDO_CONOCIDA,
    "recall_0day": ALCANCE_0DAY,
}

# Prefijos que solo aparecen en las tablas AUXILIARES (hoy, la de calibración del
# híbrido, cuya fila cruza los dos alcances a propósito). "Solo" es literal: en
# una tabla PRINCIPAL no se resuelven (alcance_de_columna(..., incluir_auxiliares
# =False)) y validar_esquema_minimo() avisa, porque un 'oof_' —columna del train—
# dentro de metricas_firmas.csv, cuyo alcance dice "sobre los ataques de D2",
# sería exactamente la mezcla de alcances que este esquema evita.
ALCANCE_PREFIJOS_AUXILIARES = {
    "oof_": ALCANCE_OOF_D3,
    "d2_": ALCANCE_D2_REPORTE,
}

# Sufijos: todo lo que termina en _cv / _cv_std / _val viene del train (nunca D2).
ALCANCE_SUFIJOS = {
    "_cv": ALCANCE_SELECCION,
    "_cv_std": ALCANCE_SELECCION,
    "_val": ALCANCE_SELECCION,
}

# Columnas concretas con alcance propio que no encajan en prefijo ni sufijo.
# 'tiempo_s' NO entra aquí a propósito: su alcance depende de la TABLA (tres
# significados; ver el bloque ALCANCE_TIEMPO_S_*) y este diccionario solo sabe dar
# uno por nombre de columna. Se declara en el dato, con 'alcance_tiempo_s'.
#
# 'n_iter_ganador' y 'n_iter_total_grid' (metricas_anomalias.csv) tampoco son
# métricas sobre D2 y sin alcance propio se leerían como tales: son ÉPOCAS.
#
# POR QUÉ SON DOS COLUMNAS Y NO UNA: 'n_iter_ganador' son las
# épocas de UNA configuración —la ganadora— mientras que 'tiempo_entrenamiento_s'
# suma los fit de TODAS las del grid. Dividir la una por la otra no da segundos
# por época: da un número sin significado. Se publica por eso 'n_iter_total_grid' —la suma de las épocas de todas
# las configuraciones del grid, exactamente el mismo conjunto de fit que
# 'tiempo_entrenamiento_s' cronometra—, que es el denominador COHERENTE con el que
# esa división tiene sentido (media ponderada entre las dos arquitecturas del
# grid, no una constante del modelo: ver ALCANCE_N_ITER_TOTAL).
# 'n_iter_ganador' se conserva porque responde a otra
# pregunta: si el ajuste del modelo que se publica se cortó por max_iter o por
# convergencia.
ALCANCE_N_ITER = (
    "épocas consumidas por el ajuste del modelo GANADOR sobre D1_train "
    "(Autoencoder = MLPRegressor con early_stopping=True y max_iter=300) — NO es "
    "una métrica sobre D2 y NO es el denominador de 'tiempo_entrenamiento_s': "
    "esta columna cuenta UNA configuración y ese tiempo suma los fit de TODAS las "
    "del grid (para eso está 'n_iter_total_grid'). Sirve para saber si el ajuste "
    "que se publica se cortó por el tope —un valor igual a max_iter=300— o por "
    "convergencia. Celda VACÍA en IsolationForest OneClassSVM y "
    "LocalOutlierFactor: no ajustan por iteraciones comparables y un 0 se leería "
    "como medida"
)

ALCANCE_N_ITER_TOTAL = (
    "suma de las épocas de TODAS las configuraciones del grid sobre D1_train "
    "(Autoencoder = MLPRegressor con early_stopping=True y max_iter=300) — NO es "
    "una métrica sobre D2. Cubre exactamente el mismo conjunto de fit que "
    "cronometra 'tiempo_entrenamiento_s' así que es el denominador COHERENTE "
    "para leer ese tiempo en segundos por época DENTRO de una corrida. NO decide "
    "si una diferencia de tiempo entre dos filas es de épocas o de carga de "
    "máquina: el recuento de épocas salió IGUAL en lo medido —los mismos 162 (54) "
    "y 128 (122) en las DOS únicas corridas que registran esta columna— así que "
    "en ellas el denominador es constante y el cociente es el numerador "
    "reescalado y absorbe la carga de máquina entera sin separar las dos causas. "
    "Constancia OBSERVADA y no determinismo demostrado: son n=2 corridas y "
    "NINGUNA de ellas es una de las dos del contraejemplo de wall-clock del "
    "Autoencoder de 54 features (37.71 s frente a 181.91 s — punto decimal para "
    "no meter comas en un valor de CSV) que no registran "
    "épocas — así que esa banda queda como magnitud observada SIN causa "
    "atribuida. Coherente y "
    "no 'el único válido': el grid del autoencoder tiene DOS arquitecturas "
    "(64 32 64) y (32 16 32) cuyo coste por época no es el mismo así que ese "
    "cociente es una MEDIA PONDERADA entre dos costes por época distintos — "
    "interpretable pero no una constante del modelo. Celda VACÍA en "
    "IsolationForest OneClassSVM y LocalOutlierFactor por el mismo motivo que "
    "'n_iter_ganador'"
)

ALCANCE_COLUMNAS = {
    "umbral": ALCANCE_SELECCION,
    "n_iter_ganador": ALCANCE_N_ITER,
    "n_iter_total_grid": ALCANCE_N_ITER_TOTAL,
}

# Marcadores que PARECEN de alcance y se vigilan: si una columna empieza por uno
# de estos y el vocabulario declarado no sabe resolverla, es una columna de otro
# alcance colada sin declarar y validar_esquema_minimo() avisa.
PREFIJOS_SIN_DECLARAR = ("oof_", "d2_", "cv_", "val_", "train_", "d1_", "d3_")


def alcance_de_columna(columna, incluir_auxiliares=True):
    """
    Alcance de UNA columna según el vocabulario de prefijos/sufijos declarado.

    Devuelve None si la columna no lleva marcador: entonces su alcance es el
    PRINCIPAL de la fila (el de la columna `alcance`).

    incluir_auxiliares : bool
        Si False, ALCANCE_PREFIJOS_AUXILIARES NO se consulta, así que 'oof_' y
        'd2_' quedan sin resolver. Lo usa validar_esquema_minimo() con las tablas
        PRINCIPALES: ahí esos prefijos no tienen sitio (son de la tabla de
        calibración del híbrido) y deben disparar el aviso, no resolverse.
    """
    for prefijo, alcance in ALCANCE_PREFIJOS.items():
        if columna.startswith(prefijo):
            return alcance
    if incluir_auxiliares:
        for prefijo, alcance in ALCANCE_PREFIJOS_AUXILIARES.items():
            if columna.startswith(prefijo):
                return alcance
    for sufijo, alcance in ALCANCE_SUFIJOS.items():
        if columna.endswith(sufijo):
            return alcance
    return ALCANCE_COLUMNAS.get(columna)


# ---------------------------------------------------------------------------
# Procedencia por fila: semilla y commit viajan CON el dato
# ---------------------------------------------------------------------------
_COMMIT_CACHE = None


# Alcance del `git status` que decide el sufijo '-sucio': SOLO el código
# (`app/`). Motivo: los propios scripts escriben en `Resultados/`
# (figuras, CSV de métricas) durante la corrida y esas rutas están versionadas,
# así que un `git status` de todo el árbol daría '-sucio' SIEMPRE —incluso
# partiendo de un checkout limpio— en cuanto se hubiese guardado la primera
# figura, y la columna `commit` perdería su capacidad de discriminar. Limitarlo
# al código es además lo que la columna quiere decir: con qué versión del código
# se produjo la fila. Así da igual el orden en que se llame a commit_actual().
_RUTA_SUCIEDAD = "app"


def commit_actual():
    """
    Hash corto del commit del repositorio en el momento de la corrida,
    con sufijo '-sucio' si hay cambios sin commitear EN EL CÓDIGO
    (`app/`; ver _RUTA_SUCIEDAD para el motivo del acotado).

    Tres valores posibles en la columna `commit` de las tablas:
      '<hash>'                        código sin cambios sin commitear
      '<hash>-sucio'                  había cambios en `app/`
      '<hash>-suciedad_desconocida'   el hash es válido pero el `git status` falló
                                      o agotó el timeout: suciedad indeterminada

    Se cachea (una sola invocación de git por proceso). Si git no está
    disponible o el directorio no es un repositorio, devuelve 'desconocido':
    la procedencia se degrada, nunca rompe la corrida. Solo lectura.
    """
    global _COMMIT_CACHE
    if _COMMIT_CACHE is not None:
        return _COMMIT_CACHE

    def _git(*args):
        return subprocess.run(
            ("git",) + args, cwd=REPO_DIR, capture_output=True, text=True,
            timeout=15, check=True,
        ).stdout.strip()

    # Dos try SEPARADOS a propósito: un `status` que agote el timeout o falle no
    # debe tirar el hash que ya se había obtenido (el árbol arrastra ~1,1 GB de
    # ficheros ignorados y `status` es mucho más caro que `rev-parse`). En ese
    # caso se conserva el hash y se marca la suciedad como indeterminada.
    try:
        commit = _git("rev-parse", "--short", "HEAD")
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError):
        _COMMIT_CACHE = "desconocido"
        return _COMMIT_CACHE

    try:
        if _git("status", "--porcelain", "--", _RUTA_SUCIEDAD):
            commit += "-sucio"
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError):
        commit += "-suciedad_desconocida"

    _COMMIT_CACHE = commit
    return commit


def ensure_dirs():
    """Crea los directorios de salida si no existen."""
    for d in (RESULTADOS_DIR, FIGURAS_DIR, MODELOS_DIR):
        os.makedirs(d, exist_ok=True)


def setup_utf8():
    """Fuerza salida UTF-8 en consolas Windows (mismo patrón que program.py):
    sin esto los prints con emojis cascan con cp1252."""
    if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            # stdout no reconfigurable (p. ej. redirigido): se continúa sin forzar
            pass
