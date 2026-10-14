# -*- coding: utf-8 -*-
"""
cruce_detectores.py — Cruce FILA A FILA entre los detectores de la etapa 1 sobre el
RÉGIMEN CIEGO de D2.

Qué mide que no estuviera medido
--------------------------------
`metricas_hibrido_0day.csv` publica, por detector y por tipo de ataque, CUÁNTAS
filas de D2 marca cada uno. En el set de 54 y sobre las 806 filas del régimen
ciego eso da IsolationForest 277 y LocalOutlierFactor 265. De esos dos recuentos
NO se deduce nada sobre cobertura conjunta: pueden ser las MISMAS filas (unión
277) o filas casi disjuntas (unión ~542) y la tabla publicada es idéntica en los
dos casos. Todo lo que se dijese sobre «combinar los dos detectores» a partir de
esos recuentos sería una inferencia no medida, y no debe hacerse sin esta
medición.

Este script mide los CONJUNTOS, no los recuentos: sobre las mismas filas y con los
umbrales ya persistidos calcula |A ∩ B|, |A \\ B|, |B \\ A|, |A ∪ B| y las filas que
no ve ninguno, globalmente y POR TIPO de ataque. Extiende la unión a los cuatro
detectores y cuenta las filas INVISIBLES para todos.

Qué permite concluir (y qué NO)
-------------------------------
Si los dos detectores se reparten los tipos —uno caza `snmpguess` y el otro
`mailbomb`— hay argumento MECANÍSTICO para una fusión en paralelo, porque la
complementariedad tiene una causa identificable y no es ruido. Si solapan, no lo
hay y esa línea de trabajo futuro pierde su fundamento.

LÍMITE, y no es una salvedad de adorno: una unión medida aquí NO es un detector.
Un sistema que corriese los dos en paralelo y disparase con cualquiera de los dos
heredaría también la UNIÓN DE SUS FALSOS POSITIVOS, y este script no mide ningún
FPR conjunto — sus dos detectores tienen FPR individuales distintos (0,100 y 0,160
en la corrida publicada del set de 54), así que ni siquiera estarían comparándose a
igual coste. 'union_ab' es un argumento sobre el MECANISMO, no un resultado de
rendimiento, y citarla como «recall de la fusión» sería exactamente el error que
esta tabla viene a impedir.

Cómo se define el RÉGIMEN CIEGO (no es una lista escrita a mano)
---------------------------------------------------------------
Se DERIVA de `metricas_hibrido_0day.csv`, de las filas de la cascada PUBLICADA
(es_cascada=True, variante por defecto, semilla de la corrida): se ordenan los
tipos 0-day por su recall publicado y se corta en el SALTO MÁS GRANDE de esa
distribución. En la corrida publicada el corte es nítido —cuatro tipos en cero
exacto más uno en 0,018 y el siguiente ya en 0,333— así que el conjunto sale solo
y no hay que fijar ningún umbral a ojo. El script IMPRIME el salto y los dos
recalls que lo bordean para que el corte sea verificable, y ABORTA si el conjunto
resultante no contiene ningún tipo con recall exactamente 0 (entonces «ciego» sería
un nombre falso) o si el reparto por tipo no cuadra con las filas de D2 en disco.

Reglas de protocolo
-------------------
  * CERO `fit`. Se cargan los `anomalia_<algo>_<set>.joblib` ya entrenados y se usa
    su propio 'umbral' (percentil 95 sobre D1_val) tal y como lo dejó anomalias.py.
    El scoring reutiliza `NSLKDDAnomalyTrainer._score` (el mismo código que
    produjo las cifras publicadas, no una copia).
  * D2 SOLO SE REPORTA. De esta tabla no sale ningún hiperparámetro, ningún
    umbral y ninguna elección de detector: el par que se cruza se elige por
    COBERTURA del régimen ciego, que es una descripción de lo ya publicado, y el
    sistema publicado no cambia por nada de lo que salga aquí.
  * VERIFICACIÓN CONTRA LO PUBLICADO: antes de cruzar nada, los recuentos por
    detector y por tipo recalculados aquí se cotejan contra
    `metricas_hibrido_0day.csv`. Si no coinciden EXACTAMENTE, el script aborta: o
    los .joblib no son los de la corrida publicada o los splits han cambiado, y en
    cualquiera de los dos casos el cruce no describiría la tabla que la memoria
    cita.
  * Tabla PROPIA (`metricas_cruce_detectores.csv`): no se escribe ni una fila en
    las cuatro principales ni en las auxiliares existentes.

Salida
------
  Resultados/metricas_cruce_detectores.csv   (nº de tipos ciegos + 1) filas por
                                             variante: una por tipo y la agregada
                                             '__global__'.

El recuento no es una promesa de la docstring: lo verifica `_comprobar_tabla()`
releyendo el CSV escrito (recuento, unicidad de 'tipo' y aditividad de los
recuentos de los tipos sobre la fila '__global__').

Semilla (--semilla N)
---------------------
POR DEFECTO ES LA 42: lee los .joblib publicados y escribe en la tabla publicada.
Con otra semilla lee los '_semilla<N>' de ese pase del barrido y escribe en
'metricas_cruce_detectores_semillas.csv'. Los .joblib se comprueban por variante Y
por semilla y la corrida aborta si declaran otra.
"""
import argparse
import os

# Backend no interactivo ANTES de cualquier import que arrastre matplotlib.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import joblib

import config
import evaluacion  # importado antes que program: fija el backend Agg
from anomalias import NSLKDDAnomalyTrainer
from program import load_specialized_splits


class NSLKDDDetectorOverlapMeasurer:
    """
    Cruza fila a fila los conjuntos de filas marcadas por los detectores de la
    etapa 1 sobre el régimen ciego de D2.

    Una clase por script, como el resto del pipeline. No entrena: carga joblibs,
    puntúa 806 filas y publica intersecciones con su alcance y su procedencia.
    """

    # Los mismos cuatro de anomalias.ALGORITMOS / hibrido.DETECTORES. Se declaran
    # aquí —en vez de importarlos— por el mismo motivo que cascada_invertida.py
    # declara los suyos: el orden de esta lista fija el orden de las columnas
    # 'detectados_*' del CSV y no debe cambiar por un refactor ajeno.
    DETECTORES = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "Autoencoder"]

    # Etiqueta de la fila agregada (mismo convenio que evaluacion.evaluar_0day_por_tipo).
    FILA_GLOBAL = "__global__"

    # Tabla publicada de la que se DERIVA el régimen ciego y contra la que se
    # verifican los recuentos recalculados.
    TABLA_0DAY = "metricas_hibrido_0day.csv"
    TABLA = "metricas_cruce_detectores.csv"

    # La cascada PUBLICADA es la de la variante por defecto (selección 4.3.5
    # aplicada, 54 features): es la que corre `hibrido.py` sin --sin-seleccion y la
    # que cita el capítulo de resultados. El régimen ciego se define UNA vez ahí y
    # se aplica igual a las dos variantes, para que 54 y 122 se crucen sobre la
    # MISMA población de filas y sus cifras sean comparables entre sí.
    SIN_SELECCION_PUBLICADA = False

    def __init__(self, sin_seleccion=False, detector_a=None, detector_b=None):
        """
        Parameters
        ----------
        sin_seleccion : bool
            False (por defecto) → set de 54 features. True → 122.
        detector_a, detector_b : str or None
            Par de detectores a cruzar en las columnas '*_ab'. Si son None se
            eligen SOLOS: los dos de mayor cobertura del régimen ciego en esta
            variante. No es una elección del sistema tomada mirando D2 (no
            ajusta nada ni cambia el sistema): es qué par merece la pena
            describir, y queda escrito en las columnas 'detector_a'/'detector_b'.
        """
        self.sin_seleccion = sin_seleccion
        self.base_path = config.base_path(sin_seleccion=sin_seleccion)
        self.set_features = self._etiqueta_set(sin_seleccion)
        self.sufijo_artefactos = config.sufijo_artefactos(self.set_features)
        self.detector_a = detector_a
        self.detector_b = detector_b

        # Rellenados por cargar_datos()
        self.X_D2 = None
        self.tipo_D2 = None
        self.n_features = None
        self.tipos_ciegos = None       # lista de tipos del régimen ciego
        self.recall_publicado = None   # {tipo: recall de la cascada publicada}
        self.detector_publicado = None

        # Rellenados por medir()
        self.marcadas = {}   # {detector: máscara booleana sobre las filas ciegas}
        self.filas = []

    @staticmethod
    def _etiqueta_set(sin_seleccion):
        """Etiqueta de la variante tal y como la escriben las tablas publicadas."""
        return "122_sin_seleccion" if sin_seleccion else "54"

    # ------------------------------------------------------------------
    # 1. Carga de D2 y derivación del régimen ciego
    # ------------------------------------------------------------------
    def cargar_datos(self):
        """Carga D2 y deriva el régimen ciego desde la tabla 0-day publicada."""
        print("=" * 70)
        print("CRUCE DE DETECTORES — carga (set de features: {})".format(
            self.set_features))
        print("=" * 70)
        splits = load_specialized_splits(self.base_path)
        if splits is None:
            raise RuntimeError(
                "No se pudieron cargar las divisiones desde: " + self.base_path
            )

        self.X_D2 = splits["D2"]["X"]
        self.tipo_D2 = splits["D2"]["y_attack"]["attack_original"].astype(str).values
        self.n_features = self.X_D2.shape[1]
        print("   D2: {} filas · {} features".format(len(self.X_D2), self.n_features))

        self._derivar_regimen_ciego()

    def _derivar_regimen_ciego(self):
        """
        Deriva los tipos del régimen ciego de la tabla 0-day PUBLICADA, cortando la
        distribución de recalls por su salto más grande. Deja en `self.tipos_ciegos`
        la lista ordenada por n descendente y en `self.recall_publicado` el recall
        de cada uno.
        """
        # config.ruta_tabla() y no una concatenación a mano: con la semilla 42 da
        # la tabla publicada y con cualquier otra la del barrido, que es la que
        # corresponde a los .joblib de esa semilla.
        ruta = config.ruta_tabla(self.TABLA_0DAY)
        if not os.path.exists(ruta):
            raise RuntimeError(
                "No existe {}: el régimen ciego se DERIVA de esa tabla y no se "
                "escribe a mano. Ejecuta antes hibrido.py para la variante "
                "publicada.".format(ruta))

        df = pd.read_csv(ruta)
        es_cascada = df["es_cascada"].astype(str).str.lower() == "true"
        mask = (
            (df["set_features"].astype(str) == self._etiqueta_set(self.SIN_SELECCION_PUBLICADA))
            & es_cascada
            & (df["tipo"].astype(str) != self.FILA_GLOBAL)
        )
        if "semilla" in df.columns:
            mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
        pub = df[mask].copy()

        if pub.empty:
            raise RuntimeError(
                "{} no trae filas de la cascada publicada para la variante '{}' y "
                "la semilla {}: sin ellas el régimen ciego no se puede derivar.".format(
                    os.path.basename(ruta),
                    self._etiqueta_set(self.SIN_SELECCION_PUBLICADA),
                    config.RANDOM_STATE))
        detectores_pub = sorted(set(pub["detector"].astype(str)))
        if len(detectores_pub) != 1:
            raise RuntimeError(
                "{} marca como cascada publicada más de un detector ({}): el "
                "régimen ciego dejaría de estar definido.".format(
                    os.path.basename(ruta), detectores_pub))
        self.detector_publicado = detectores_pub[0]

        # Corte por el salto más grande de la distribución de recalls.
        pub = pub.sort_values(["recall", "tipo"]).reset_index(drop=True)
        recalls = pub["recall"].astype(float).values
        if len(recalls) < 2:
            raise RuntimeError(
                "Solo hay {} tipo(s) 0-day publicados: no hay distribución que "
                "cortar.".format(len(recalls)))
        saltos = np.diff(recalls)
        corte = int(np.argmax(saltos))          # último índice del grupo bajo
        ciegos = pub.iloc[:corte + 1]

        if not (ciegos["recall"].astype(float) == 0.0).any():
            raise RuntimeError(
                "El corte por el salto máximo deja un grupo sin ningún tipo de "
                "recall 0 (mínimo {:.4f}): llamarlo 'régimen ciego' sería falso. "
                "Revisa {} antes de seguir.".format(
                    float(ciegos["recall"].min()), os.path.basename(ruta)))

        self.tipos_ciegos = list(
            ciegos.sort_values(["n", "tipo"], ascending=[False, True])["tipo"].astype(str)
        )
        self.recall_publicado = dict(
            zip(ciegos["tipo"].astype(str), ciegos["recall"].astype(float))
        )
        n_publicado = dict(zip(ciegos["tipo"].astype(str), ciegos["n"].astype(int)))

        print("-" * 70)
        print("RÉGIMEN CIEGO derivado de {} (cascada publicada: detector {})".format(
            os.path.basename(ruta), self.detector_publicado))
        print("   Corte por el salto máximo de la distribución de recalls: "
              "{:.6f} → {:.6f} (salto {:.6f}); los {} tipos por debajo forman el "
              "régimen".format(recalls[corte], recalls[corte + 1],
                               float(saltos[corte]), len(self.tipos_ciegos)))
        for t in self.tipos_ciegos:
            print("      {:<16} n={:>4}  recall publicado={:.6f}".format(
                t, n_publicado[t], self.recall_publicado[t]))

        # Coherencia con los datos en disco: el reparto por tipo de la tabla tiene
        # que ser el de D2. Si no lo es, la tabla publicada y los splits no son de
        # la misma corrida y ninguna cifra de aquí describiría a la otra.
        for t in self.tipos_ciegos:
            n_disco = int((self.tipo_D2 == t).sum())
            if n_disco != n_publicado[t]:
                raise RuntimeError(
                    "El tipo '{}' tiene {} filas en D2 y {} en {}: los splits en "
                    "disco y la tabla publicada no son de la misma corrida.".format(
                        t, n_disco, n_publicado[t], os.path.basename(ruta)))

        self.mask_ciego = np.isin(self.tipo_D2, self.tipos_ciegos)
        self.n_ciego = int(self.mask_ciego.sum())
        print("   Población del régimen ciego: {} filas de D2 ({} tipos)".format(
            self.n_ciego, len(self.tipos_ciegos)))

    # ------------------------------------------------------------------
    # 2. Carga de los .joblib de la etapa 1 y marcado de las filas ciegas
    # ------------------------------------------------------------------
    def _cargar_joblib(self, algo):
        """Carga `anomalia_<algo>_<set>.joblib` comprobando variante y semilla."""
        ruta = os.path.join(
            config.MODELOS_DIR,
            "anomalia_{}_{}.joblib".format(algo, self.sufijo_artefactos))
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
                "No existe {}: este cruce NO entrena nada y necesita el detector "
                "ya persistido. Ejecuta antes anomalias.py{} para la variante "
                "'{}'.{}".format(ruta, flag_variante, self.set_features,
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
                "{}: se cruzarían dos corridas distintas.".format(
                    ruta, semilla_joblib, config.RANDOM_STATE))
        return datos, ruta

    def _marcar(self):
        """
        Para cada detector: puntúa SOLO las filas del régimen ciego con su modelo
        persistido y las marca con su propio umbral. Deja en `self.marcadas` una
        máscara booleana por detector, alineada con las filas ciegas de D2.

        Puntuar el subconjunto y no D2 entero es legítimo y además se COMPRUEBA:
        los cuatro scores son fila a fila independientes (score_samples /
        decision_function / MSE de reconstrucción no miran a las demás filas de la
        matriz) y la verificación contra la tabla publicada —calculada sobre D2
        completo— es justamente la prueba de que el recuento sale idéntico.
        """
        X_ciego = self.X_D2.loc[self.mask_ciego]
        # El scoring se delega en anomalias.py. La instancia no carga datos
        # ni entrena: _score solo necesita (algo, modelo, X).
        anom = NSLKDDAnomalyTrainer(sin_seleccion=self.sin_seleccion)

        print("-" * 70)
        print("MARCADO DE LAS {} FILAS CIEGAS (umbral p95 persistido de cada "
              "detector)".format(self.n_ciego))
        for algo in self.DETECTORES:
            datos, ruta = self._cargar_joblib(algo)
            score = anom._score(algo, datos["modelo"], X_ciego)
            marcada = np.asarray(score > datos["umbral"])
            self.marcadas[algo] = marcada
            print("   {:<20} ve {:>4}/{} · umbral={:.6f} · {}".format(
                algo, int(marcada.sum()), self.n_ciego, float(datos["umbral"]),
                os.path.basename(ruta)))

    def _verificar_contra_publicado(self):
        """
        Cotejo EXACTO de los recuentos recalculados contra
        `metricas_hibrido_0day.csv` (filas es_cascada=False de esta variante), tipo
        a tipo y detector a detector. Aborta a la primera discrepancia.

        Sin esto, un .joblib regenerado con otro código —o unos splits
        re-preprocesados— darían un cruce internamente coherente que NO describe la
        tabla que cita la memoria, y nada lo delataría.
        """
        ruta = config.ruta_tabla(self.TABLA_0DAY)
        df = pd.read_csv(ruta)
        mask = df["set_features"].astype(str) == str(self.set_features)
        if "semilla" in df.columns:
            mask &= df["semilla"].astype(str) == str(config.RANDOM_STATE)
        pub = df[mask]

        tipo_ciego = self.tipo_D2[self.mask_ciego]
        comprobados = 0
        for algo in self.DETECTORES:
            sub = pub[pub["detector"].astype(str) == algo]
            if sub.empty:
                print("   [aviso] {} no aparece en {} para la variante '{}': ese "
                      "detector no se puede cotejar.".format(
                          algo, os.path.basename(ruta), self.set_features))
                continue
            # Una misma fila (variante × detector × tipo) puede estar dos veces si
            # el detector es además el de la cascada: las filas es_cascada=True y
            # es_cascada=False son la MISMA medida de etapa 1 (el recall 0-day
            # depende solo del detector y su umbral), así que se toma la primera.
            for t in self.tipos_ciegos:
                fila = sub[sub["tipo"].astype(str) == t]
                if fila.empty:
                    print("   [aviso] {} / {} no está en la tabla publicada: sin "
                          "cotejo para ese par.".format(algo, t))
                    continue
                esperado = int(fila["detectados"].iloc[0])
                obtenido = int(self.marcadas[algo][tipo_ciego == t].sum())
                if esperado != obtenido:
                    raise RuntimeError(
                        "DISCREPANCIA con {}: {} sobre '{}' marca aquí {} filas y "
                        "la tabla publicada dice {}. Los .joblib o los splits no "
                        "son los de la corrida publicada; el cruce no describiría "
                        "la tabla que cita la memoria.".format(
                            os.path.basename(ruta), algo, t, obtenido, esperado))
                comprobados += 1
        print("   Verificación contra {}: {} recuentos (detector × tipo) "
              "coinciden EXACTAMENTE.".format(os.path.basename(ruta), comprobados))

    # ------------------------------------------------------------------
    # 3. El cruce propiamente dicho
    # ------------------------------------------------------------------
    def _elegir_par(self):
        """
        Fija el par de detectores de las columnas '*_ab'. Si no vino por CLI, los
        dos de MAYOR cobertura del régimen ciego en esta variante (desempate por
        orden de DETECTORES, que es el de la tabla publicada). Cruzar los dos que
        no ven nada no respondería a ninguna pregunta.
        """
        if self.detector_a and self.detector_b:
            if self.detector_a == self.detector_b:
                raise ValueError("El par a cruzar son DOS detectores distintos.")
            print("   Par fijado por CLI: {} × {}".format(
                self.detector_a, self.detector_b))
            return

        orden = sorted(
            self.DETECTORES,
            key=lambda a: (-int(self.marcadas[a].sum()), self.DETECTORES.index(a)),
        )
        # Qué DOS se cruzan lo decide la cobertura; cuál es 'a' y cuál es 'b' lo
        # decide el orden de DETECTORES, no la cobertura. Si el par se ordenase por
        # cobertura, 'solo_a' significaría IsolationForest en la variante de 54 y
        # LocalOutlierFactor en la de 122 —que es justo donde se invierten— y las
        # dos filas '__global__' dejarían de ser comparables columna a columna.
        par = sorted(orden[:2], key=self.DETECTORES.index)
        self.detector_a, self.detector_b = par[0], par[1]
        print("   Par elegido por cobertura del régimen ciego: {} ({}) × {} ({}) "
              "· orden a/b fijado por la lista DETECTORES"
              .format(self.detector_a, int(self.marcadas[self.detector_a].sum()),
                      self.detector_b, int(self.marcadas[self.detector_b].sum())))

    @staticmethod
    def _tasa(numerador, denominador):
        """Tasa con denominador 0 → NaN (tipos de n pequeño sin ninguna marcada)."""
        if denominador == 0:
            return float("nan")
        return float(numerador) / float(denominador)

    def _fila(self, tipo, sel):
        """Construye la fila del CSV para el subconjunto booleano `sel`."""
        n = int(sel.sum())
        a = self.marcadas[self.detector_a][sel]
        b = self.marcadas[self.detector_b][sel]

        inter = int((a & b).sum())
        solo_a = int((a & ~b).sum())
        solo_b = int((b & ~a).sum())
        union = int((a | b).sum())
        ninguno = n - union

        union_todos = np.zeros(n, dtype=bool)
        for algo in self.DETECTORES:
            union_todos |= self.marcadas[algo][sel]
        n_union_todos = int(union_todos.sum())

        fila = {
            "alcance": config.ALCANCE_CRUCE_DETECTORES,
            "set_features": self.set_features,
            "sin_seleccion": bool(self.sin_seleccion),
            "n_features": self.n_features,
            "tipo": tipo,
            "n": n,
            # Recall del tipo en la CASCADA PUBLICADA: es la cifra que define el
            # régimen y sin ella la fila no se puede situar en la tabla de origen.
            # NaN en la fila agregada: la publicada solo lo da por tipo.
            "recall_publicado_cascada": (
                round(self.recall_publicado[tipo], 6)
                if tipo in self.recall_publicado else float("nan")),
        }
        # Recuento por detector (el dato que YA estaba publicado, recalculado aquí
        # y verificado contra él) — el orden de columnas lo fija DETECTORES.
        for algo in self.DETECTORES:
            fila["detectados_" + algo.lower()] = int(self.marcadas[algo][sel].sum())
        # El cruce, que es lo que NO estaba medido.
        fila.update({
            "detector_a": self.detector_a,
            "detector_b": self.detector_b,
            "inter_ab": inter,
            "solo_a": solo_a,
            "solo_b": solo_b,
            "union_ab": union,
            "ninguno_ab": ninguno,
            # Jaccard = |A ∩ B| / |A ∪ B|: 1 = los mismos flujos, 0 = disjuntos.
            # NaN si ninguno de los dos ve nada (0/0 no es 0).
            "jaccard_ab": round(self._tasa(inter, union), 6),
            "recall_union_ab": round(self._tasa(union, n), 6),
            "recall_inter_ab": round(self._tasa(inter, n), 6),
            "union_todos": n_union_todos,
            "invisibles_todos": n - n_union_todos,
            "recall_union_todos": round(self._tasa(n_union_todos, n), 6),
            "n_detectores": len(self.DETECTORES),
        })
        return fila

    def medir(self):
        """Marca, verifica contra lo publicado y cruza los conjuntos."""
        self._marcar()
        self._verificar_contra_publicado()
        print("-" * 70)
        print("CRUCE FILA A FILA")
        self._elegir_par()

        tipo_ciego = self.tipo_D2[self.mask_ciego]
        self.filas = []
        for t in self.tipos_ciegos:
            self.filas.append(self._fila(t, tipo_ciego == t))
        self.filas.append(
            self._fila(self.FILA_GLOBAL, np.ones(self.n_ciego, dtype=bool)))

        for f in self.filas:
            print("   {:<16} n={:>4} · {}={:>4} {}={:>4} · ∩={:>4} sólo_a={:>4} "
                  "sólo_b={:>4} ∪={:>4} ninguno={:>4} · J={} · ∪4={:>4} "
                  "invisibles4={:>4}".format(
                      f["tipo"], f["n"],
                      self.detector_a[:3], f["detectados_" + self.detector_a.lower()],
                      self.detector_b[:3], f["detectados_" + self.detector_b.lower()],
                      f["inter_ab"], f["solo_a"], f["solo_b"], f["union_ab"],
                      f["ninguno_ab"],
                      "n/a" if f["jaccard_ab"] != f["jaccard_ab"]
                      else "{:.3f}".format(f["jaccard_ab"]),
                      f["union_todos"], f["invisibles_todos"]))

    # ------------------------------------------------------------------
    # 4. Persistencia y comprobación del artefacto
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
        Recuento comprobado por CÓDIGO releyendo el CSV ya escrito. Equivalente de
        comprobar_recuento() + comprobar_unicidad() para una tabla auxiliar que no
        lleva 'algoritmo' ni la CLAVE_UNICIDAD y que por eso no puede pasar por
        ellas. Aborta en cuatro casos:

          (1) la unidad en curso no trae una fila por tipo ciego más la agregada;
          (2) algún 'tipo' se repite o falta;
          (3) los 'n' de los tipos no suman el 'n' de '__global__' (los tipos
              PARTICIONAN el régimen: es lo que permite leer la fila agregada);
          (4) los recuentos del cruce (intersección, sólo_a, sólo_b, unión de los
              cuatro e invisibles) no suman los de '__global__'. Se comprueba sobre
              ENTEROS y no sobre las tasas, que solo cuadran hasta el redondeo.
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

        esperadas = len(self.tipos_ciegos) + 1
        if len(df_var) != esperadas:
            raise ValueError(
                "{}: {} tiene {} filas y debería tener {} ({} tipos ciegos + la "
                "fila '{}'). Regenera la tabla con un pase completo.".format(
                    nombre, unidad, len(df_var), esperadas,
                    len(self.tipos_ciegos), self.FILA_GLOBAL))

        observados = list(df_var["tipo"].astype(str))
        objetivo = list(self.tipos_ciegos) + [self.FILA_GLOBAL]
        if sorted(observados) != sorted(objetivo):
            raise ValueError(
                "{}: {} trae los tipos {} y debería traer exactamente {} (una vez "
                "cada uno).".format(nombre, unidad, sorted(observados),
                                    sorted(objetivo)))

        por_tipo = df_var[df_var["tipo"].astype(str) != self.FILA_GLOBAL]
        glob = df_var[df_var["tipo"].astype(str) == self.FILA_GLOBAL].iloc[0]
        columnas_aditivas = ["n", "inter_ab", "solo_a", "solo_b", "union_ab",
                             "ninguno_ab", "union_todos", "invisibles_todos"]
        for algo in self.DETECTORES:
            columnas_aditivas.append("detectados_" + algo.lower())
        for col in columnas_aditivas:
            suma = int(por_tipo[col].sum())
            if suma != int(glob[col]):
                raise ValueError(
                    "{}: la columna '{}' suma {} entre los tipos y la fila '{}' "
                    "dice {}: los tipos no particionan el régimen o la fila "
                    "agregada no es de esta corrida.".format(
                        nombre, col, suma, self.FILA_GLOBAL, int(glob[col])))

        print("   Recuento {}: {} filas en {} · tipos únicos y completos · los "
              "{} recuentos aditivos cuadran con '{}'".format(
                  nombre, len(df_var), unidad, len(columnas_aditivas),
                  self.FILA_GLOBAL))

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def ejecutar(self):
        config.setup_utf8()
        config.ensure_dirs()
        self.cargar_datos()
        self.medir()
        self._persistir()
        print("=" * 70)
        print("CRUCE DE DETECTORES COMPLETADO ({} features)".format(self.set_features))
        print("=" * 70)
        return self.filas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cruce fila a fila de los detectores de la etapa 1 sobre el "
                    "régimen ciego de D2: intersección, diferencias y unión por "
                    "tipo de ataque. No entrena nada; D2 solo se reporta."
    )
    parser.add_argument(
        "--sin-seleccion", action="store_true",
        help="Usa el set de 122 features (variante sin selección 4.3.5) en lugar "
             "de las 54 por defecto.",
    )
    parser.add_argument(
        "--detector-a", choices=NSLKDDDetectorOverlapMeasurer.DETECTORES,
        default=None,
        help="Primer detector del par cruzado. Por defecto el de mayor cobertura "
             "del régimen ciego en esa variante.",
    )
    parser.add_argument(
        "--detector-b", choices=NSLKDDDetectorOverlapMeasurer.DETECTORES,
        default=None,
        help="Segundo detector del par cruzado. Por defecto el segundo de mayor "
             "cobertura.",
    )
    parser.add_argument(
        "--semilla", type=int, default=config.SEMILLA_POR_DEFECTO,
        help=config.AYUDA_CLI_SEMILLA + " Requiere los .joblib de anomalias.py de "
             "ESA MISMA semilla: se aborta si declaran otra.",
    )
    args = parser.parse_args()

    # ANTES de instanciar: el __init__ congela el sufijo con el que se buscan los
    # .joblib que este script LEE. Ver config.py, encabezado.
    config.fijar_semilla(args.semilla)

    medidor = NSLKDDDetectorOverlapMeasurer(
        sin_seleccion=args.sin_seleccion,
        detector_a=args.detector_a,
        detector_b=args.detector_b,
    )
    medidor.ejecutar()
