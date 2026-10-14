# -*- coding: utf-8 -*-
"""
seleccion_ciega.py — Selección de componentes CIEGA A D2.

Responde a UNA pregunta, y no a otra: **¿habría elegido lo mismo un protocolo que
no mirase D2 en ningún momento?** No es un intento de mejorar el sistema; es la
cuantificación de una limitación ya declarada en la memoria (6.1.3): el detector
de la etapa 1, el clasificador de la etapa 2 y el set de características se
eligieron con métricas medidas sobre el conjunto de test.

El protocolo completo, con su justificación y sus limitaciones, se fijó ANTES de
ejecutar este script y se describe en el Apéndice B de la memoria.

POR QUÉ NO BASTA CON APARTAR PARTE DE KDDTrain+
-----------------------------------------------
KDDTrain+ tiene 22 tipos de ataque; KDDTest+ tiene 39. Una partición aleatoria de
KDDTrain+ reparte LOS MISMOS 22 tipos entre ajuste y selección, así que mediría
capacidad ante ataques CONOCIDOS — justo lo que no es el objetivo del sistema.
Por eso aquí no se reparten filas al azar: se EXCLUYEN TIPOS DE ATAQUE ENTEROS
del entrenamiento (diseño leave-one-attack-type-out, LOATO), y esos tipos hacen
de 0-day simulado dentro del propio train.

Reglas de protocolo (invalidan el experimento si se rompen)
-----------------------------------------------------------
  * D2 NO SE ABRE EN EL BARRIDO. Ni para elegir, ni para ordenar, ni para
    desempatar. Se lee una sola vez, al final y solo si se pide `--reportar-d2`,
    para decir qué habría dado la terna ya elegida. Es un resultado, no una
    decisión.
  * LOS TIPOS EXCLUIDOS NO SE ELIGEN MIRANDO KDDTest+. La rotación es
    determinista y alfabética dentro de cada categoría. Elegirlos por parecido
    con los 17 tipos nuevos reales sería volver a colar D2 por la puerta de atrás
    y el resultado no valdría nada.
  * TODO LO QUE APRENDE SE RE-AJUSTA POR FOLD: one-hot, escalado, codificación de
    etiquetas, SELECCIÓN DE CARACTERÍSTICAS y las dos etapas. Si el selector de
    las 54 ve los tipos excluidos, el fold deja de ser ciego. Es el punto que más
    cuesta en tiempo y el que no se puede relajar.
  * NO SE PISA NADA DE LO PUBLICADO. Tabla propia
    (`Resultados/metricas_seleccion_ciega.csv`), informe propio y directorio de
    trabajo propio. Las nueve tablas publicadas, los veinte `.joblib` de la
    semilla 42, las 39 figuras y `Resultados/selected_features.txt` NO se abren
    para escritura. En particular, `select_features()` se llama SIEMPRE con
    `output_dir` desviado a la carpeta del fold: su valor por defecto apunta a
    `Resultados/` y sobrescribiría el informe de selección publicado.

POR QUÉ REUTILIZA anomalias.py Y firmas.py EN VEZ DE COPIAR SUS REJILLAS
-------------------------------------------------------------------------
Los candidatos tienen que ser LOS MISMOS que se compararon sobre D2, o la
comparación no dice nada. Las rejillas (`GRIDS`), los constructores de estimador,
el score unificado de anomalías y las opciones de balanceo viven en aquellas dos
clases y se IMPORTAN desde aquí. Ambos `__init__` son libres de efectos —no tocan
disco: la carga vive en `cargar_datos()`, que este script no llama—, así que se
instancian y se usan solo sus métodos auxiliares. Una copia de las rejillas se
desincronizaría, y entonces el experimento estaría comparando otra cosa.

Salidas
-------
  Resultados/metricas_seleccion_ciega.csv     una fila por (fold, variante,
                                              componente, candidato)
  Resultados/seleccion_ciega_informe.txt      la terna elegida y su procedencia
  Resultados/seleccion_ciega_trabajo/         artefactos intermedios por fold

Uso
---
  python app\\seleccion_ciega.py                   # barrido completo (4 folds x 2 sets)
  python app\\seleccion_ciega.py --folds 1         # piloto, para medir el coste
  python app\\seleccion_ciega.py --solo-plan       # imprime los folds y no entrena
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
import evaluacion
from program import NSLKDDPreprocessor
from anomalias import NSLKDDAnomalyTrainer
from firmas import NSLKDDSignatureTrainer


# Claves de los splits, tal y como las nombra program.py. Se escriben una vez
# aquí para no repetir literales por todo el fichero.
D1_KEY = "D1_normal_for_anomaly"
D2_KEY = "D2_complete_test"
D3_KEY = "D3_known_attacks_for_signatures"


class NSLKDDBlindSelector:
    """
    Barrido LOATO que elige detector, clasificador y set de características sin
    mirar D2 en ningún punto.
    """

    # Tipos excluidos por fold y categoría. ASIMÉTRICO A PROPÓSITO: un
    # holdout uniforme de 1 por categoría deja fuera solo el 18% de los tipos,
    # menos presión de novedad que la real, y bajo poca novedad la selección
    # tiende a premiar componentes buenos en conjunto CERRADO — es decir, a
    # infravalorar justo la capacidad que el sistema defiende. Con 2 en las dos
    # categorías que pueden permitírselo el holdout sube al 27%, cerca del ~29%
    # de tráfico 0-day de D2 (3.750 de 12.833 ataques). 'probe' y 'u2r' se quedan
    # en 1 porque solo tienen 4 tipos: dejar 'u2r' con dos sería ruido.
    HOLDOUT_POR_CATEGORIA = {"dos": 2, "probe": 1, "r2l": 2, "u2r": 1}

    # 4 folds = rotación completa sobre las categorías limitantes (probe y u2r
    # tienen 4 tipos y se excluye 1 por fold).
    N_FOLDS_DEFECTO = 4

    # Fracción del tráfico de tipo CONOCIDO que se reserva al conjunto de
    # selección de cada fold. Los tipos EXCLUIDOS van íntegros a selección: son
    # el 0-day simulado y no pueden aparecer en entrenamiento ni en parte.
    FRAC_SELECCION = 0.30

    VARIANTES = [("54", False), ("122_sin_seleccion", True)]

    DIR_TRABAJO = os.path.join(config.RESULTADOS_DIR, "seleccion_ciega_trabajo") \
        if hasattr(config, "RESULTADOS_DIR") else None

    def __init__(self, n_folds=None, frac_seleccion=None, dir_trabajo=None):
        self.n_folds = int(n_folds or self.N_FOLDS_DEFECTO)
        self.frac_seleccion = float(frac_seleccion or self.FRAC_SELECCION)
        self.dir_trabajo = dir_trabajo or self.DIR_TRABAJO
        self.train_df = None
        self.tipos_por_categoria = {}
        self.folds = []
        self.filas = []
        # Instancias SOLO para reutilizar rejillas y constructores. No se les
        # llama a cargar_datos(): no tocan disco.
        self._anom = NSLKDDAnomalyTrainer()
        self._firm = NSLKDDSignatureTrainer()

    # ------------------------------------------------------------------
    # 1. Carga del crudo y censo de tipos
    # ------------------------------------------------------------------
    def cargar_crudo(self):
        """
        Carga KDDTrain+ y anota la categoría de cada fila. KDDTest+ se carga
        porque `load_dataset()` devuelve ambos, pero AQUÍ NO SE USA: el barrido
        entero vive dentro de KDDTrain+.
        """
        print("=" * 70)
        print("CARGA DEL CRUDO (solo KDDTrain+; KDDTest+ no entra en el barrido)")
        print("=" * 70)
        pre = NSLKDDPreprocessor()
        train_df, _test_df = pre.load_dataset()
        if train_df is None:
            raise RuntimeError("No se pudo cargar KDDTrain+.")

        df = train_df.copy()
        if "attack_category" not in df.columns:
            df["attack_category"] = df["attack"].map(pre.attack_mapping).fillna("unknown")
        self.train_df = df

        ataques = df[df["attack"] != "normal"]
        for cat, sub in ataques.groupby("attack_category"):
            # Orden ALFABÉTICO: la rotación de folds tiene que ser reproducible y
            # no puede depender del orden de aparición en el fichero.
            self.tipos_por_categoria[cat] = sorted(sub["attack"].unique().tolist())

        print("   Filas de train: {:,}  (normal: {:,} · ataque: {:,})".format(
            len(df), int((df["attack"] == "normal").sum()), len(ataques)))
        for cat in sorted(self.tipos_por_categoria):
            print("   {:<6} {:>2} tipos: {}".format(
                cat, len(self.tipos_por_categoria[cat]),
                ", ".join(self.tipos_por_categoria[cat])))
        return self

    # ------------------------------------------------------------------
    # 2. Construcción de los folds LOATO
    # ------------------------------------------------------------------
    def construir_folds(self):
        """
        Rotación determinista: en el fold f, la categoría con k tipos excluidos
        cede los tipos en las posiciones (f*k + i) mod n_tipos de su lista
        alfabética. Ni aleatoriedad ni juicio: nada aquí puede depender de qué
        tipos son nuevos en KDDTest+.
        """
        self.folds = []
        for f in range(self.n_folds):
            excluidos = {}
            for cat in sorted(self.tipos_por_categoria):
                tipos = self.tipos_por_categoria[cat]
                k = self.HOLDOUT_POR_CATEGORIA.get(cat, 1)
                k = min(k, max(1, len(tipos) - 2))  # nunca dejar una categoría con <2 tipos
                excluidos[cat] = [tipos[(f * k + i) % len(tipos)] for i in range(k)]
            self.folds.append({"fold": f, "excluidos": excluidos})

        print("\n" + "=" * 70)
        print("PLAN DE FOLDS (LOATO estratificado por categoría)")
        print("=" * 70)
        n_total = sum(len(v) for v in self.tipos_por_categoria.values())
        for spec in self.folds:
            fuera = [t for v in spec["excluidos"].values() for t in v]
            print("   fold {}: {} tipos fuera ({:.0f}%) → {}".format(
                spec["fold"], len(fuera), 100.0 * len(fuera) / n_total,
                ", ".join(sorted(fuera))))
        return self

    # ------------------------------------------------------------------
    # 3. Partición de un fold: entrenamiento ciego / conjunto de selección
    # ------------------------------------------------------------------
    def _particion_fold(self, spec):
        """
        Devuelve (fold_train_df, fold_sel_df).

        fold_train : normal (70%) + ataques de tipo NO excluido (70%)
        fold_sel   : normal (30%) + ataques de tipo NO excluido (30%)
                     + TODAS las filas de los tipos EXCLUIDOS

        El conjunto de selección tiene por tanto la misma forma que D2 —normal,
        ataques conocidos y ataques nunca vistos—, que es lo que permite medir
        sobre él exactamente las métricas que se midieron sobre D2.
        """
        df = self.train_df
        fuera = {t for v in spec["excluidos"].values() for t in v}

        es_excluido = df["attack"].isin(fuera)
        resto = df[~es_excluido]

        rng = np.random.RandomState(config.RANDOM_STATE + spec["fold"])
        # Estratificado por tipo de ataque: si se muestrease en bloque, las clases
        # con pocas decenas de filas (u2r) podrían quedarse sin representación en
        # uno de los dos lados.
        idx_sel = []
        for _tipo, sub in resto.groupby("attack"):
            n_sel = int(round(len(sub) * self.frac_seleccion))
            n_sel = min(max(n_sel, 1), len(sub) - 1) if len(sub) > 1 else 0
            if n_sel > 0:
                idx_sel.extend(rng.choice(sub.index.values, size=n_sel, replace=False))
        idx_sel = pd.Index(idx_sel)

        fold_train = resto.drop(index=idx_sel)
        fold_sel = pd.concat([resto.loc[idx_sel], df[es_excluido]], axis=0)

        # Guarda dura: ni una sola fila de los tipos excluidos puede haberse
        # colado en el entrenamiento. Es LA invariante del experimento.
        assert not fold_train["attack"].isin(fuera).any(), \
            "fold {}: un tipo excluido apareció en entrenamiento".format(spec["fold"])

        return fold_train, fold_sel

    # ------------------------------------------------------------------
    # 4. Preprocesado por fold (se rehace entero, incluida la selección)
    # ------------------------------------------------------------------
    def _preprocesar(self, fold_train, fold_sel, sin_seleccion, spec):
        """
        Rehace el pipeline completo de program.py sobre el fold. El escalador y el
        selector se ajustan con D1+D3 del fold, es decir SIN los tipos excluidos.

        `select_features()` recibe `output_dir` desviado a la carpeta del fold: su
        valor por defecto es `Resultados/` y sobrescribiría el
        `selected_features.txt` publicado.
        """
        pre = NSLKDDPreprocessor()
        splits = pre.create_specialized_data_splits(fold_train, fold_sel)
        procesados = pre.preprocess_specialized_splits(splits)

        if not sin_seleccion:
            destino = os.path.join(
                self.dir_trabajo,
                "fold{}_{}".format(spec["fold"], "54"),
            )
            os.makedirs(destino, exist_ok=True)
            pre.select_features(procesados, output_dir=destino)

        return procesados

    # ------------------------------------------------------------------
    # 5. Etapa 1: los cuatro detectores sobre el fold
    # ------------------------------------------------------------------
    def _evaluar_detectores(self, procesados, spec, set_features):
        """
        Reproduce el protocolo de anomalias.py dentro del fold: entrena con el
        normal del fold, fija el umbral en el percentil 95 de un 20% reservado de
        ese normal, y puntúa el conjunto de selección.

        La selección de hiperparámetros usa AUC sobre el reservado + una muestra
        de ataques CONOCIDOS del fold, igual que en el sistema publicado. Nunca
        sobre los tipos excluidos: esos son la incógnita, no el criterio.
        """
        a = self._anom
        X_D1 = procesados[D1_KEY]["X"]
        X_sel = procesados[D2_KEY]["X"]
        cat_sel = pd.Series(procesados[D2_KEY]["y_category_original"])
        tipo_sel = pd.Series(procesados[D2_KEY]["y_attack_original"])
        y_bin = (cat_sel != config.ETIQUETA_NORMAL).astype(int).values

        rng = np.random.RandomState(config.RANDOM_STATE + spec["fold"])
        corte = int(len(X_D1) * 0.8)
        orden = rng.permutation(len(X_D1))
        X_tr = X_D1.iloc[orden[:corte]]
        X_val = X_D1.iloc[orden[corte:]]

        # Set etiquetado para elegir hiperparámetros: normal reservado + muestra
        # de ataques CONOCIDOS del propio fold (D3 del fold).
        X_D3 = procesados[D3_KEY]["X"]
        n_m = min(a.N_MUESTRA_D3, len(X_D3))
        X_d3_m = X_D3.sample(n=n_m, random_state=config.RANDOM_STATE)
        X_lab = pd.concat([X_val, X_d3_m], axis=0)
        y_lab = np.concatenate([np.zeros(len(X_val), dtype=int),
                                np.ones(len(X_d3_m), dtype=int)])

        fuera = {t for v in spec["excluidos"].values() for t in v}
        mask_0day = tipo_sel.isin(fuera).values

        filas = []
        for algo in a.ALGORITMOS:
            t0 = time.time()
            mejor = {"auc": -np.inf, "cfg": None, "model": None}
            for cfg in a.GRIDS[algo]:
                modelo = a._construir(algo, cfg)
                X_fit = a._datos_entrenamiento(algo, X_tr)
                modelo = a._ajustar(algo, modelo, X_fit)
                auc = roc_auc_score(y_lab, a._score(algo, modelo, X_lab))
                if auc > mejor["auc"]:
                    mejor = {"auc": auc, "cfg": cfg, "model": modelo}

            modelo = mejor["model"]
            umbral = float(np.percentile(a._score(algo, modelo, X_val),
                                         a.PERCENTIL_UMBRAL))
            score_sel = a._score(algo, modelo, X_sel)
            pred = (score_sel >= umbral).astype(int)

            filas.append({
                "fold": spec["fold"],
                "set_features": set_features,
                "componente": "detector",
                "candidato": algo,
                "auc_roc": round(float(roc_auc_score(y_bin, score_sel)), 6),
                "f1_binario": round(float(f1_score(y_bin, pred)), 6),
                "recall_0day": round(float(recall_score(
                    np.ones(int(mask_0day.sum()), dtype=int),
                    pred[mask_0day], zero_division=0)), 6) if mask_0day.any() else np.nan,
                "fpr": round(float(pred[y_bin == 0].mean()), 6),
                "f1_macro": np.nan,
                "config_ganadora": str(mejor["cfg"]),
                "tipos_excluidos": "|".join(sorted(fuera)),
                "n_seleccion": int(len(X_sel)),
                "segundos": round(time.time() - t0, 2),
            })
            print("      {:<20} AUC={:.4f}  F1={:.4f}  recall0day={}".format(
                algo, filas[-1]["auc_roc"], filas[-1]["f1_binario"],
                filas[-1]["recall_0day"]))
        return filas

    # ------------------------------------------------------------------
    # 6. Etapa 2: los cuatro clasificadores sobre el fold
    # ------------------------------------------------------------------
    def _evaluar_clasificadores(self, procesados, spec, set_features):
        """
        Reproduce el protocolo de firmas.py dentro del fold: GridSearchCV con
        f1_macro sobre el D3 del fold (ataques de tipo NO excluido), con el
        balanceo ganador buscado igual que en el sistema publicado, y evaluación
        sobre los ataques de tipo conocido del conjunto de selección.
        """
        s = self._firm
        X_D3 = procesados[D3_KEY]["X"]
        y_D3 = pd.Series(procesados[D3_KEY]["y_category_original"])

        cat_sel = pd.Series(procesados[D2_KEY]["y_category_original"])
        tipo_sel = pd.Series(procesados[D2_KEY]["y_attack_original"])
        fuera = {t for v in spec["excluidos"].values() for t in v}
        tipos_conocidos = set(pd.Series(procesados[D3_KEY]["y_attack_original"]).unique())

        mask_eval = (cat_sel != config.ETIQUETA_NORMAL).values & \
                    tipo_sel.isin(tipos_conocidos).values
        X_eval = procesados[D2_KEY]["X"].loc[mask_eval]
        y_eval = cat_sel[mask_eval]

        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=config.RANDOM_STATE)

        filas = []
        for algo in s.ALGORITMOS:
            t0 = time.time()
            # Balanceo ganador, con el mismo criterio que 4.3.4: f1_macro en CV.
            mejor_bal, mejor_f1 = None, -np.inf
            for balanceo in s.BALANCEO_OPCIONES[algo]:
                est = s._estimador_para(algo, balanceo)
                from sklearn.model_selection import cross_val_score
                f1m = float(cross_val_score(est, X_D3, y_D3, cv=cv,
                                            scoring="f1_macro", n_jobs=-1).mean())
                if f1m > mejor_f1:
                    mejor_bal, mejor_f1 = balanceo, f1m

            est = s._estimador_para(algo, mejor_bal)
            rejilla = {("clf__" + k if mejor_bal == "SMOTE" else k): v
                       for k, v in s.GRIDS[algo].items()}
            gs = GridSearchCV(est, rejilla, cv=cv, scoring="f1_macro", n_jobs=-1)
            gs.fit(X_D3, y_D3)

            pred = gs.best_estimator_.predict(X_eval)
            filas.append({
                "fold": spec["fold"],
                "set_features": set_features,
                "componente": "clasificador",
                "candidato": algo,
                "auc_roc": np.nan,
                "f1_binario": np.nan,
                "recall_0day": np.nan,
                "fpr": np.nan,
                "f1_macro": round(float(f1_score(y_eval, pred, average="macro")), 6),
                "config_ganadora": "{} | balanceo={}".format(gs.best_params_, mejor_bal),
                "tipos_excluidos": "|".join(sorted(fuera)),
                "n_seleccion": int(len(X_eval)),
                "segundos": round(time.time() - t0, 2),
            })
            print("      {:<20} f1_macro={:.4f}  (balanceo={})".format(
                algo, filas[-1]["f1_macro"], mejor_bal))
        return filas

    # ------------------------------------------------------------------
    # 7. Barrido
    # ------------------------------------------------------------------
    def ejecutar(self):
        os.makedirs(self.dir_trabajo, exist_ok=True)
        for spec in self.folds:
            fold_train, fold_sel = self._particion_fold(spec)
            print("\n" + "=" * 70)
            print("FOLD {} — train {:,} filas · selección {:,} filas".format(
                spec["fold"], len(fold_train), len(fold_sel)))
            print("   tipos fuera: {}".format(
                ", ".join(sorted(t for v in spec["excluidos"].values() for t in v))))
            print("=" * 70)

            for set_features, sin_sel in self.VARIANTES:
                print("\n   --- variante {} ---".format(set_features))
                procesados = self._preprocesar(fold_train, fold_sel, sin_sel, spec)
                print("   etapa 1:")
                self.filas.extend(
                    self._evaluar_detectores(procesados, spec, set_features))
                print("   etapa 2:")
                self.filas.extend(
                    self._evaluar_clasificadores(procesados, spec, set_features))
        return self

    # ------------------------------------------------------------------
    # 8. Agregación y terna elegida
    # ------------------------------------------------------------------
    def elegir_y_escribir(self):
        """
        Promedia cada candidato sobre los folds y escribe la tabla y el informe.
        El ganador es el de mejor métrica MEDIA, no el de un fold suelto.
        """
        df = pd.DataFrame(self.filas)
        df["commit"] = config.commit_actual()
        df["semilla"] = config.RANDOM_STATE

        ruta_csv = os.path.join(config.RESULTADOS_DIR, "metricas_seleccion_ciega.csv")
        df.to_csv(ruta_csv, index=False, encoding="utf-8")
        print("\n   → {} ({} filas)".format(ruta_csv, len(df)))

        lineas = ["TERNA ELEGIDA POR EL PROTOCOLO CIEGO A D2",
                  "commit: {} · semilla: {} · folds: {}".format(
                      config.commit_actual(), config.RANDOM_STATE, self.n_folds),
                  ""]

        # Detector: AUC-ROC media sobre folds y variantes.
        det = df[df["componente"] == "detector"].groupby("candidato")["auc_roc"].mean()
        # Clasificador: f1_macro media.
        clf = df[df["componente"] == "clasificador"].groupby("candidato")["f1_macro"].mean()
        # Set de características: recall sobre tipos excluidos, que es la métrica
        # con la que se eligió el set sobre D2.
        st = df[df["componente"] == "detector"].groupby("set_features")["recall_0day"].mean()

        for titulo, serie, publicado in (
            ("DETECTOR (etapa 1) — AUC-ROC media", det, "Autoencoder"),
            ("CLASIFICADOR (etapa 2) — f1_macro media", clf, "RandomForest"),
            ("SET DE CARACTERÍSTICAS — recall 0-day medio", st, "54"),
        ):
            serie = serie.sort_values(ascending=False)
            ganador = serie.index[0]
            lineas.append(titulo)
            for nombre, valor in serie.items():
                marca = "  <-- ELEGIDO" if nombre == ganador else ""
                lineas.append("   {:<22} {:.6f}{}".format(nombre, valor, marca))
            lineas.append("   publicado en la memoria: {} → {}".format(
                publicado, "COINCIDE" if ganador == publicado else "DIFIERE"))
            lineas.append("")

        ruta_txt = os.path.join(config.RESULTADOS_DIR, "seleccion_ciega_informe.txt")
        with open(ruta_txt, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lineas))
        print("   → {}".format(ruta_txt))
        print("\n" + "\n".join(lineas))
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Selección de componentes ciega a D2."
    )
    parser.add_argument("--folds", type=int, default=None,
                        help="Número de folds LOATO (por defecto 4).")
    parser.add_argument("--frac-seleccion", type=float, default=None,
                        help="Fracción del tráfico conocido reservada a selección (0.30).")
    parser.add_argument("--semilla", type=int, default=None,
                        help="Semilla de la corrida (por defecto la del proyecto).")
    parser.add_argument("--solo-plan", action="store_true",
                        help="Imprime el plan de folds y sale sin entrenar nada.")
    args = parser.parse_args()

    if args.semilla is not None:
        config.fijar_semilla(args.semilla)

    sel = NSLKDDBlindSelector(n_folds=args.folds, frac_seleccion=args.frac_seleccion)
    sel.cargar_crudo().construir_folds()

    if args.solo_plan:
        print("\n--solo-plan: no se entrena nada.")
        return 0

    sel.ejecutar()
    sel.elegir_y_escribir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
