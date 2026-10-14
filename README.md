# H-NIDS: sistema híbrido de detección de intrusiones en red sobre NSL-KDD

Código del Trabajo Fin de Grado *Sistema híbrido de detección de intrusiones en red con
Machine Learning* (E.T.S. de Ingeniería de Telecomunicación, Universidad de Málaga, 2026).

El sistema combina dos etapas en cascada:

1. **Detección de anomalías** (etapa 1): un modelo entrenado solo con tráfico normal marca como
   sospechoso todo flujo que se desvía de lo normal, incluidos los ataques de tipos nunca vistos
   (*0-day*).
2. **Clasificación por firmas** (etapa 2): un clasificador supervisado asigna a cada flujo
   sospechoso una de las cuatro categorías de ataque conocidas (DoS, Probe, R2L, U2R) o, si su
   confianza no alcanza un umbral, la etiqueta «desconocido».

Un RandomForest monolítico de cinco clases sirve de referencia para medir la ventaja de la
cascada en la detección de ataques 0-day.

---

## Estructura

```
H-NIDS-NSL-KDD/
├── app/               Scripts del sistema y de los experimentos
├── dataset/           KDDTrain+.txt y KDDTest+.txt (no incluidos; ver más abajo)
├── Resultados/        Salidas: datos procesados, modelos, métricas y figuras (se crea al ejecutar)
└── requirements.txt   Dependencias con versión fijada
```

## Instalación

Requiere **Python 3.11**.

```bash
python -m venv venv
# Windows: venv\Scripts\activate    ·    Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

El dataset NSL-KDD no se incluye en el repositorio. Descarga `KDDTrain+.txt` y `KDDTest+.txt`
de [github.com/Jehuty4949/NSL_KDD](https://github.com/Jehuty4949/NSL_KDD) y colócalos en una
carpeta `dataset/` en la raíz del repositorio.

## Conjuntos de trabajo

`program.py` genera tres subconjuntos a partir de los dos ficheros del dataset:

| Subconjunto | Contenido | Uso |
|---|---|---|
| **D1** | Tráfico normal de KDDTrain+ | Entrenamiento de la etapa de anomalías |
| **D2** | KDDTest+ completo | Evaluación final; no interviene en ningún ajuste |
| **D3** | Ataques de KDDTrain+ | Entrenamiento de la etapa de firmas |

Todo lo que se aprende de los datos (escalado, selección de características, hiperparámetros,
umbrales, balanceo) se ajusta sin ver D2. El balanceo con SMOTE se aplica dentro de cada pliegue
de la validación cruzada.

---

## Ejecución

Los scripts se ejecutan desde la raíz del repositorio. La semilla por defecto es **42**.

### Sistema

```bash
python app/program.py          # preprocesado, análisis exploratorio y generación de D1/D2/D3
python app/validacion.py       # validación de los subconjuntos generados
python app/anomalias.py        # etapa 1: comparación de detectores de anomalías
python app/firmas.py           # etapa 2: comparación de clasificadores y extracción de reglas
python app/baseline.py         # referencia: RandomForest monolítico de cinco clases
python app/hibrido.py          # sistema completo evaluado sobre D2
python app/cascada_invertida.py  # medición del orden inverso de la cascada
```

`anomalias.py` y `firmas.py` son independientes entre sí; `hibrido.py` necesita los modelos que
guardan ambos, y `cascada_invertida.py` necesita además el de `hibrido.py`.

Por defecto se trabaja con las **54** características seleccionadas. Con `--sin-seleccion`
(en `program.py` y en los scripts de modelos) se usan las **122** columnas del one-hot sin
selección. Los scripts de modelos aceptan `--semilla N`: con una semilla distinta de 42, sus
modelos, figuras y métricas se guardan con sufijo propio y no sustituyen a los de la semilla 42.

### Experimentos complementarios

| Script | Qué mide |
|---|---|
| `barrido_semillas.py` | Repite el sistema con las semillas 1 a 10 en las dos variantes de características (`--dry-run` muestra el plan sin ejecutar) |
| `agregar_semillas.py` | Resume la dispersión de las métricas entre semillas |
| `estabilidad_semillas.py` | Estabilidad frente a la semilla: dispersión, posición de la semilla 42 y pruebas de Wilcoxon pareadas con corrección de Holm |
| `seleccion_ciega.py` | Repite la elección de detector, clasificador y conjunto de características con un protocolo que excluye un tipo de ataque por pliegue sobre KDDTrain+, sin usar D2 |
| `cruce_detectores.py` | Cruce fila a fila de dos detectores de la etapa 1 sobre los ataques 0-day que la cascada no detecta |
| `degradacion_firmas.py` | Degradación de la etapa 2 desde la validación cruzada en D3 hasta los ataques de D2 |
| `aprendizaje_continuo.py` | Aprendizaje continuo de la etapa 2: versiones del sistema construidas sin un tipo de ataque y actualizadas con lotes crecientes de ese tipo (`--todos`, después `--agregar`; `--reconstruccion` mide la reconstrucción completa) |
| `representacion.py` | Representaciones alternativas de las características y clasificadores basados en distancia para la etapa 2, con varias reglas de rechazo a «desconocido» (`--agregar` produce la tabla final) |

### Figuras

`figura_nids_bloques.py`, `figura_tipos_ataque.py`, `figuras_cap4_memoria.py`,
`figuras_cap5_memoria.py`, `figura_paneles_cm.py` y `figura_aprendizaje_continuo.py` generan las
figuras de la memoria en `Resultados/figuras/` a partir de los datos y modelos ya generados.

### Módulos comunes

- `config.py`: rutas, semilla, variante de características y convenciones de clase.
- `evaluacion.py`: métricas binarias, multiclase y de 0-day por tipo, tiempos, figuras y
  escritura de las tablas de métricas con su esquema.

---

## Salidas

Todo se escribe en `Resultados/`: los subconjuntos procesados y sus metadatos, los modelos
entrenados (`Resultados/modelos/*.joblib`), las tablas de métricas (`metricas_*.csv`), los
informes de texto, las reglas extraídas del árbol de decisión y las figuras
(`Resultados/figuras/`).
