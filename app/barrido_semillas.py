# -*- coding: utf-8 -*-
"""
barrido_semillas.py — Lanzador desatendido del barrido de dispersión entre semillas.

Recorre las 10 semillas de `config.SEMILLAS_BARRIDO` × 2 variantes de
características × 5 scripts ejecutables, en el orden de dependencias, borrando los
`.joblib` de cada semilla al terminarla y saltando lo que ya esté hecho.

POR QUÉ UN DRIVER EN PYTHON Y NO UN `.ps1`
------------------------------------------
El entorno de desarrollo es Windows 11, así que un `.ps1` era la opción
obvia. Se elige Python por tres cosas que el barrido necesita y que en
PowerShell serían código frágil escrito dos veces:
  1. REANUDACIÓN. Decidir si una `(semilla, set, script)` ya está hecha exige leer
     los `metricas_*_semillas.csv` y contar filas por (variante, semilla). Aquí es
     `pandas` y tres líneas; en PowerShell sería un `Import-Csv` con comparaciones
     de texto sobre columnas cuyo tipo importa (`semilla` es entera).
  2. UNA SOLA FUENTE DE VERDAD. La lista de semillas, el sufijo de artefactos, el
     nombre de las tablas del barrido y el recuento esperado por pase ya viven en
     `config.py` y `evaluacion.py`. Desde Python se IMPORTAN; desde PowerShell
     habría que copiarlos, y una copia que se desincronice del código es
     exactamente cómo el barrido acabaría borrando lo que no debe.
  3. EL BORRADO. Filtrar por el sufijo `_semilla<N>` sin tocar los 20 `.joblib`
     publicados de la semilla 42 es la operación más peligrosa de este script:
     se hace con la misma función que compone los nombres (`config`), y con
     guardas explícitas, no con un patrón de glob escrito a mano.
Se invoca igual desde PowerShell, con el entorno virtual activado:
`python app\\barrido_semillas.py`. Usa `sys.executable`, así que los hijos heredan
el intérprete del venv activo sin resolverlo por ruta.

Reglas de protocolo (invalidan el TFG si se rompen)
---------------------------------------------------
  * NINGUNA corrida de este barrido usa la semilla 42. `config.SEMILLAS_BARRIDO`
    lo garantiza con una aserción y aquí se vuelve a comprobar antes de lanzar
    nada: con 42 no hay desvío de artefactos ni de tablas y el pase pisaría las
    cuatro tablas publicadas, los 20 `.joblib` y las 39 figuras.
  * EL BORRADO SOLO ALCANZA FICHEROS CON `_semilla<N>` EN EL NOMBRE, y solo
    después de que `hibrido.py` y `cascada_invertida.py` de esa semilla —que
    consumen esos `.joblib`— hayan terminado. Los 20 `.joblib` publicados no se
    tocan jamás.
  * D2 sigue intocable: este script no cambia ni un hiperparámetro ni un umbral,
    solo invoca los cinco scripts del pipeline con `--semilla`.

Salidas
-------
  Resultados/logs_barrido/<script>_<set>_semilla<N>.log   un log por corrida
  Resultados/verificacion_semilla_joblib.txt              traza de la verificación
                                                          de `semilla = 42` en los
                                                          .joblib publicados. La
                                                          reescribe `--solo-verificar`
                                                          (y el preflight, SOLO si
                                                          encuentra problemas): un
                                                          lanzamiento limpio no la
                                                          re-sella, porque su
                                                          cabecera se cita verbatim
                                                          en la documentación

Uso
---
  python app\\barrido_semillas.py                  # el barrido completo
  python app\\barrido_semillas.py --dry-run        # imprime el plan, no ejecuta
                                                  # (pero SÍ corre el preflight:
                                                  #  no es una pasada solo lectura)
  python app\\barrido_semillas.py --solo-verificar # solo la traza de la 42
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

import joblib
import pandas as pd

import config
import evaluacion


class BarridoSemillas:
    """
    Orquesta el barrido de semillas: plan, reanudación, ejecución y borrado.

    Una clase por script, como el resto del pipeline. No entrena ni evalúa nada
    por sí misma: lanza los cinco scripts del pipeline como procesos hijo y
    administra sus artefactos.
    """

    # Los cinco scripts, EN ORDEN DE DEPENDENCIAS dentro de cada (semilla, set):
    #   anomalias  → deja los 4 'anomalia_*.joblib' que consume el híbrido
    #   firmas     → deja los 4 'firma_*.joblib' que consumen híbrido y cascada
    #   hibrido    → carga los dos anteriores y deja 'hibrido_*.joblib' (umbral)
    #   cascada_invertida → carga 'firma_*' y LEE el umbral de 'hibrido_*'
    #   baseline   → independiente del híbrido (depende solo del preprocesado);
    #                va al final para no retrasar la cadena que sí tiene
    #                dependencias, y su joblib entra igual en el borrado.
    # Cada entrada declara la tabla por la que se comprueba si ya está hecha y
    # cuántas filas debe tener por (variante, semilla). Los recuentos NO se
    # escriben a mano: salen de evaluacion.FILAS_ESPERADAS_POR_VARIANTE y de
    # cascada_invertida (5 = 4 categorías + '__global__'), para que no puedan
    # divergir del contrato de los scripts.
    PASOS = (
        {"script": "anomalias.py", "tabla": "metricas_anomalias.csv"},
        {"script": "firmas.py", "tabla": "metricas_firmas.csv"},
        {"script": "hibrido.py", "tabla": "metricas_hibrido.csv"},
        {"script": "cascada_invertida.py",
         "tabla": "metricas_cascada_invertida.csv",
         "filas": len(config.CATEGORIAS_ATAQUE) + 1},
        {"script": "baseline.py", "tabla": "metricas_baseline.csv"},
    )

    # Las dos variantes de características, en el orden en que se corren.
    VARIANTES = (
        {"set_features": "54", "sin_seleccion": False},
        {"set_features": "122_sin_seleccion", "sin_seleccion": True},
    )

    DIR_LOGS = os.path.join(config.RESULTADOS_DIR, "logs_barrido")
    TRAZA_VERIFICACION = os.path.join(
        config.RESULTADOS_DIR, "verificacion_semilla_joblib.txt")

    def __init__(self, semillas=None, dry_run=False):
        self.semillas = list(config.SEMILLAS_BARRIDO if semillas is None
                             else semillas)
        self.dry_run = bool(dry_run)
        self.dir_app = os.path.dirname(os.path.abspath(__file__))
        # cwd de los hijos = raíz del repositorio (el padre de app/), que es como
        # los documenta el README ('python app\\anomalias.py'). Las rutas de datos
        # y salidas son absolutas (config.py), así que el cwd no cambia ningún
        # artefacto; se fija por coherencia con la documentación.
        self.dir_impl = os.path.dirname(self.dir_app)
        self.saltados = 0
        self.ejecutados = 0

        # Guarda de seguridad, independiente de la aserción de config: ninguna
        # corrida de este barrido puede llevar la semilla publicada.
        if config.SEMILLA_POR_DEFECTO in self.semillas:
            raise ValueError(
                "La semilla {} no puede entrar en el barrido: con ella no hay "
                "desvío de artefactos ni de tablas y el pase sobrescribiría todo "
                "lo publicado.".format(config.SEMILLA_POR_DEFECTO))
        if any(int(s) < 0 for s in self.semillas):
            raise ValueError("Las semillas deben ser enteros no negativos")

    # ------------------------------------------------------------------
    # 1. Verificación de los .joblib PUBLICADOS (semilla 42) — traza reproducible
    # ------------------------------------------------------------------
    def verificar_joblibs_publicados(self, escribir_si_ok=True,
                                     modo="--solo-verificar"):
        """
        Comprueba que TODOS los `.joblib` publicados (los que no llevan
        `_semilla<N>` en el nombre) declaran `semilla = 42`, y deja la traza en
        `Resultados/verificacion_semilla_joblib.txt`.

        modo : str
            Cómo se invocó la verificación, tal cual se estampa en la línea
            'Generado por:' de la traza. NO es un literal fijo a propósito: el
            preflight también escribe el fichero cuando encuentra problemas, y con
            el literal '--solo-verificar' clavado esa traza afirmaría haber salido
            de un modo que no se usó — justo en el único caso en que el fichero es
            el diagnóstico de un abort y alguien lo va a leer.

        escribir_si_ok : bool
            Si False, la traza NO se reescribe cuando la verificación sale limpia;
            solo se imprime. Si encuentra problemas se escribe SIEMPRE, porque
            entonces el fichero es el diagnóstico del abort.

            POR QUÉ ESTE PARÁMETRO: esta función se
            invoca en dos papeles distintos. Como PREFLIGHT de cada lanzamiento del
            barrido —incluido `--dry-run`— es una comprobación, y reescribir la
            traza en cada relanzamiento re-sella su cabecera (`Commit del código:`
            y la fecha) con el estado del árbol de ese momento; la documentación
            cita ese sello VERBATIM, así que el primer lanzamiento real dejaría esa
            frase falsa sin que nadie tocase una cifra. Como `--solo-verificar` el
            papel es el contrario: producir el artefacto. El preflight pasa
            False; `--solo-verificar` deja el defecto True.

        POR QUÉ ES UN PASO DEL LANZADOR y no una comprobación a mano: la
        afirmación «los 20 .joblib publicados declaran semilla = 42» sostiene dos
        cosas —que una corrida por defecto pasa la salvaguarda de mezcla de
        `hibrido.py`/`cascada_invertida.py`, y que el filtro del borrado por
        sufijo no puede alcanzarlos—. Una afirmación así necesita traza
        reproducible, no memoria de quien la escribió.

        Aborta si algún publicado declara otra semilla o no la declara: en ese
        caso el barrido no debe arrancar, porque el estado de lo publicado no es
        el que la documentación dice.
        """
        rutas = sorted(
            r for r in glob.glob(os.path.join(config.MODELOS_DIR, "*.joblib"))
            if config.MARCA_SEMILLA not in
            os.path.basename(r)
        )
        # El filtro de arriba usa la MARCA '_semilla' (la plantilla sin número),
        # que es exactamente la que compone sufijo_semilla(): así "publicado"
        # significa aquí lo mismo que en el resto del andamiaje.
        lineas = [
            "Verificación de la semilla declarada en los .joblib PUBLICADOS",
            "=" * 70,
            "Generado por: app/barrido_semillas.py ({})".format(
                modo),
            "Commit del código: {}".format(config.commit_actual()),
            "Fecha de la verificación: {}".format(
                datetime.now().isoformat(timespec="seconds")),
            "Directorio: {}".format(config.MODELOS_DIR),
            "Criterio de 'publicado': el nombre NO contiene la marca '{}'".format(
                config.MARCA_SEMILLA),
            "Semilla esperada: {} (config.SEMILLA_POR_DEFECTO)".format(
                config.SEMILLA_POR_DEFECTO),
            "",
        ]
        problemas = []
        for ruta in rutas:
            nombre = os.path.basename(ruta)
            problema = None
            semilla = commit = None
            try:
                datos = joblib.load(ruta)
                semilla = datos.get("semilla")
                commit = datos.get("commit")
                if semilla != config.SEMILLA_POR_DEFECTO:
                    problema = "{}: declara semilla={!r} y debería declarar {}".format(
                        nombre, semilla, config.SEMILLA_POR_DEFECTO)
            except Exception as e:                       # noqa: BLE001
                problema = "{}: no se pudo leer ({})".format(nombre, e)
            if problema is not None:
                problemas.append(problema)
            lineas.append("{:<8} {:<48} semilla={!s:<6} commit={}".format(
                "REVISAR" if problema else "OK", nombre, semilla, commit))

        lineas += [
            "",
            "Ficheros publicados examinados: {}".format(len(rutas)),
            "Con semilla = {}: {}".format(
                config.SEMILLA_POR_DEFECTO, len(rutas) - len(problemas)),
            "Con problema: {}".format(len(problemas)),
        ]
        if problemas:
            lineas += ["", "PROBLEMAS:"] + ["  - " + p for p in problemas]
        else:
            lineas += [
                "",
                "CONSECUENCIAS COMPROBADAS (las dos afirmaciones que esta traza "
                "sostiene):",
                "  1. Una corrida por defecto pasa la salvaguarda de mezcla de "
                "semillas de hibrido._cargar_joblib() y de "
                "cascada_invertida._cargar_joblib_firma()/_leer_umbral_conf().",
                "  2. El borrado por sufijo del barrido "
                "(BarridoSemillas.borrar_joblibs_de_semilla) no puede alcanzar "
                "ninguno de estos ficheros: ninguno lleva la marca '{}'.".format(
                    config.MARCA_SEMILLA),
            ]

        config.ensure_dirs()
        escribir = bool(problemas) or escribir_si_ok
        if escribir:
            with open(self.TRAZA_VERIFICACION, "w", encoding="utf-8") as f:
                f.write("\n".join(lineas) + "\n")
        print("\n".join(lineas))
        print("-" * 70)
        if escribir:
            print("   Traza: {}".format(self.TRAZA_VERIFICACION))
        else:
            print("   Verificación OK; la traza {} NO se reescribe (para "
                  "re-sellarla: "
                  "--solo-verificar)".format(
                      os.path.basename(self.TRAZA_VERIFICACION)))
        if problemas:
            raise RuntimeError(
                "Hay {} .joblib publicados que no declaran semilla = {}: el "
                "barrido no arranca. Revisa la traza {}.".format(
                    len(problemas), config.SEMILLA_POR_DEFECTO,
                    self.TRAZA_VERIFICACION))
        return len(rutas)

    # ------------------------------------------------------------------
    # 2. Reanudación: ¿esta (semilla, set, script) ya está hecha?
    # ------------------------------------------------------------------
    def _filas_esperadas(self, paso):
        """Filas que el paso deja por (variante, semilla) en su tabla."""
        if "filas" in paso:
            return paso["filas"]
        esperadas = evaluacion.FILAS_ESPERADAS_POR_VARIANTE.get(paso["tabla"])
        if esperadas is None:
            raise RuntimeError(
                "El paso {} no declara filas esperadas y su tabla {} no está en "
                "evaluacion.FILAS_ESPERADAS_POR_VARIANTE.".format(
                    paso["script"], paso["tabla"]))
        return esperadas

    def ya_hecho(self, paso, variante, semilla):
        """
        True si la tabla del barrido ya trae las filas COMPLETAS de esta
        (variante, semilla) para este paso.

        Es el requisito de reanudación: el barrido corre desatendido mientras la
        máquina se usa para otras cosas, así que tiene que poder interrumpirse y
        continuar sin repetir horas de cómputo ni duplicar filas.

        Se exige el recuento EXACTO, no «al menos una fila»: una corrida cortada a
        medias puede haber dejado 2 de las 4 filas de anomalías, y darla por hecha
        dejaría la celda incompleta —que es justo lo que el agregador aborta al
        final, después del barrido entero—.
        """
        ruta = os.path.join(config.RESULTADOS_DIR,
                            config.nombre_tabla_semillas(paso["tabla"]))
        if not os.path.exists(ruta):
            return False
        try:
            df = pd.read_csv(ruta)
        except Exception:                                # noqa: BLE001
            # Tabla ilegible: no se da nada por hecho (el propio script la
            # apartará con limpiar_variante_csv al escribir).
            return False
        if "set_features" not in df.columns or "semilla" not in df.columns:
            return False
        mask = ((df["set_features"].astype(str) == variante["set_features"])
                & (df["semilla"].astype(str) == str(semilla)))
        return int(mask.sum()) == self._filas_esperadas(paso)

    # ------------------------------------------------------------------
    # 3. Ejecución de una corrida
    # ------------------------------------------------------------------
    def _comando(self, paso, variante, semilla):
        cmd = [sys.executable, os.path.join(self.dir_app, paso["script"]),
               "--semilla", str(semilla)]
        if variante["sin_seleccion"]:
            cmd.append("--sin-seleccion")
        return cmd

    def _ruta_log(self, paso, variante, semilla):
        return os.path.join(self.DIR_LOGS, "{}_{}_semilla{}.log".format(
            os.path.splitext(paso["script"])[0], variante["set_features"],
            semilla))

    def ejecutar_paso(self, paso, variante, semilla):
        """
        Lanza un script como proceso hijo, con su log propio. Aborta el barrido
        si el hijo devuelve código distinto de 0.

        POR QUÉ FALLA RÁPIDO en lugar de seguir con la semilla siguiente: los
        cinco scripts corren el mismo código en las diez semillas, así que un
        fallo casi nunca es de una semilla concreta —es del código, del entorno o
        del disco— y continuar produciría diez veces el mismo error con horas de
        cómputo en medio. Además la cadena tiene dependencias: sin `firmas.py` no
        hay `hibrido.py` ni cascada. Al abortar NO se borran los `.joblib` de la
        semilla en curso: quedan para diagnosticar, y la reanudación los reutiliza.
        """
        cmd = self._comando(paso, variante, semilla)
        log = self._ruta_log(paso, variante, semilla)
        etiqueta = "{} · set {} · semilla {}".format(
            paso["script"], variante["set_features"], semilla)
        if self.dry_run:
            print("   [dry-run] {}".format(" ".join(cmd)))
            return
        os.makedirs(self.DIR_LOGS, exist_ok=True)
        print("   > {} ...".format(etiqueta), flush=True)
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            f.write("$ {}\n".format(" ".join(cmd)))
            f.write("inicio: {}\n\n".format(
                datetime.now().isoformat(timespec="seconds")))
            f.flush()
            completado = subprocess.run(
                cmd, cwd=self.dir_impl, stdout=f, stderr=subprocess.STDOUT,
            )
        dur = time.perf_counter() - t0
        if completado.returncode != 0:
            raise RuntimeError(
                "FALLÓ {} (código {}) tras {:.1f}s. El barrido se detiene aquí y "
                "los .joblib de la semilla {} NO se borran. Log completo: "
                "{}".format(etiqueta, completado.returncode, dur, semilla, log))
        self.ejecutados += 1
        print("     hecho en {:.1f}s · log: {}".format(dur, os.path.basename(log)))

    # ------------------------------------------------------------------
    # 4. Borrado de los .joblib de una semilla
    # ------------------------------------------------------------------
    def borrar_joblibs_de_semilla(self, semilla):
        """
        Borra los `.joblib` de UNA semilla, en las dos variantes, DESPUÉS de que
        `hibrido.py` y `cascada_invertida.py` de esa semilla los hayan consumido.

        POR QUÉ EXISTE: son 20 .joblib por semilla —los 20 cubren YA las dos
        variantes: 10 de '54' y 10 de '122_sin_seleccion'— × 10 semillas = 200
        ficheros ≈ 4,8 GB si no se borra nada. La cuenta sale de medir el pase
        publicado: `Resultados/modelos/` ocupa 481,33 MB con esos mismos 20 ficheros
        (`firma_KNN_122` pesa 173 MB por sí solo), así que 481,33 MB × 10
        semillas. Con borrado por semilla el pico es el de una semilla (481 MB),
        no el de diez.

        TRES GUARDAS, y ninguna es decorativa:
          1. La semilla no puede ser la 42.
          2. Solo se borra si el nombre TERMINA en '_semilla<N>.joblib'. El
             sufijo se compone con la misma plantilla que usa config, no con un
             glob a mano, así que '_semilla1' no puede llevarse '_semilla10'.
          3. Se rechaza cualquier ruta que no esté dentro de MODELOS_DIR.
        Los `.joblib` publicados no llevan la marca '_semilla', así que quedan
        fuera del filtro por construcción (traza en
        Resultados/verificacion_semilla_joblib.txt).
        """
        if int(semilla) == config.SEMILLA_POR_DEFECTO:
            raise ValueError(
                "Nunca se borran los .joblib de la semilla {}: son los "
                "publicados.".format(config.SEMILLA_POR_DEFECTO))
        sufijo = config.sufijo_de_semilla(semilla) + ".joblib"
        borrados, liberado = 0, 0
        for ruta in sorted(glob.glob(os.path.join(config.MODELOS_DIR,
                                                  "*.joblib"))):
            nombre = os.path.basename(ruta)
            if not nombre.endswith(sufijo):
                continue
            if os.path.dirname(os.path.abspath(ruta)) != os.path.abspath(
                    config.MODELOS_DIR):
                continue
            liberado += os.path.getsize(ruta)
            if self.dry_run:
                print("   [dry-run] borraría {}".format(nombre))
            else:
                os.remove(ruta)
            borrados += 1
        print("   Borrados {} .joblib de la semilla {} ({:.1f} MB liberados){}"
              .format(borrados, semilla, liberado / (1024.0 * 1024.0),
                      " [dry-run: no se borró nada]" if self.dry_run else ""))
        return borrados

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------
    def ejecutar(self):
        config.setup_utf8()
        config.ensure_dirs()
        print("=" * 70)
        print("BARRIDO DE SEMILLAS{}".format(
            " — DRY RUN, no se ejecuta ni se borra nada" if self.dry_run else ""))
        print("=" * 70)
        print("   Semillas ({}): {}".format(len(self.semillas), self.semillas))
        print("   Variantes: {}".format(
            [v["set_features"] for v in self.VARIANTES]))
        print("   Scripts por (semilla, variante), en orden: {}".format(
            [p["script"] for p in self.PASOS]))
        print("   Corridas totales: {}".format(
            len(self.semillas) * len(self.VARIANTES) * len(self.PASOS)))
        print("   La semilla {} NO entra: es la publicada.".format(
            config.SEMILLA_POR_DEFECTO))
        print("-" * 70)

        # Preflight: el estado de lo publicado es el que la documentación dice.
        # escribir_si_ok=False → si sale limpio NO re-sella la traza (su cabecera se
        # cita verbatim en la documentación); si encuentra problemas la escribe y
        # aborta.
        self.verificar_joblibs_publicados(
            escribir_si_ok=False,
            modo="preflight del barrido{}".format(
                " --dry-run" if self.dry_run else ""))

        t0 = time.perf_counter()
        for semilla in self.semillas:
            print("=" * 70)
            print("SEMILLA {}".format(semilla))
            print("=" * 70)
            for variante in self.VARIANTES:
                print("-- set {} ".format(variante["set_features"]) + "-" * 40)
                for paso in self.PASOS:
                    if self.ya_hecho(paso, variante, semilla):
                        self.saltados += 1
                        print("   = {} · set {} · semilla {}: ya está en {} "
                              "(reanudación)".format(
                                  paso["script"], variante["set_features"],
                                  semilla,
                                  config.nombre_tabla_semillas(paso["tabla"])))
                        continue
                    self.ejecutar_paso(paso, variante, semilla)
            # Solo aquí, con hibrido.py y cascada_invertida.py de esta semilla ya
            # corridos en las DOS variantes (sus .joblib comparten el sufijo de
            # semilla, así que el borrado alcanza a las dos a la vez).
            self.borrar_joblibs_de_semilla(semilla)

        print("=" * 70)
        print("BARRIDO {}: {} corridas ejecutadas · {} saltadas por reanudación "
              "· {:.1f} min".format(
                  "PLANIFICADO (dry-run)" if self.dry_run else "COMPLETADO",
                  self.ejecutados, self.saltados,
                  (time.perf_counter() - t0) / 60.0))
        print("   Siguiente paso: python app\\agregar_semillas.py")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lanzador desatendido del barrido de dispersión entre "
                    "semillas: 10 semillas × 2 sets de características × 5 "
                    "scripts, con borrado de .joblib por semilla y reanudación."
    )
    parser.add_argument(
        "--semillas", type=int, nargs="+", default=None,
        help="Semillas a correr (por defecto las {} de config.SEMILLAS_BARRIDO: "
             "{}). La {} está PROHIBIDA aquí.".format(
                 len(config.SEMILLAS_BARRIDO), config.SEMILLAS_BARRIDO,
                 config.SEMILLA_POR_DEFECTO),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Imprime el plan (comandos y borrados) sin ejecutar ningún script ni "
             "borrar ningún .joblib. NO es 'solo lectura': el preflight de "
             "verificación de los .joblib publicados se hace igual (leyendo sus "
             "descriptores, cero fit) y ABORTA si alguno no declara semilla {}. La "
             "traza no se reescribe si sale limpia; si sale con problemas SÍ, "
             "porque es el diagnóstico del abort. Y se crean los directorios de "
             "salida si faltan (config.ensure_dirs()).".format(
                 config.SEMILLA_POR_DEFECTO),
    )
    parser.add_argument(
        "--solo-verificar", action="store_true",
        help="Solo verifica que los .joblib publicados declaran semilla {} y "
             "escribe la traza en Resultados/verificacion_semilla_joblib.txt; no "
             "lanza el barrido.".format(config.SEMILLA_POR_DEFECTO),
    )
    args = parser.parse_args()

    # Este script NO fija la semilla global: no entrena nada. La semilla viaja a
    # cada hijo por su propio flag --semilla, que es la única vía de inyección
    # (config.py, encabezado). Llamar aquí a config.fijar_semilla() no tendría
    # efecto sobre los procesos hijo y solo confundiría.
    barrido = BarridoSemillas(semillas=args.semillas, dry_run=args.dry_run)
    if args.solo_verificar:
        config.setup_utf8()
        barrido.verificar_joblibs_publicados()
    else:
        barrido.ejecutar()
