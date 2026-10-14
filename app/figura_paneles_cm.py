# -*- coding: utf-8 -*-
"""
figura_paneles_cm.py — Figuras de MEMORIA: las cuatro matrices de confusión de cada
etapa reunidas en un panel 2×2.

POR QUÉ EXISTE
--------------
Ocho matrices sueltas (cuatro binarias de la etapa 1 y cuatro multiclase de la
etapa 2), cuyas cifras ya reproducen las tablas de resultados, ocuparían ocho
páginas de la memoria. Se funden en dos figuras de cuatro paneles.

NO mide nada ni redibuja ninguna matriz: compone las PNG que ya emitieron
`anomalias.py` y `firmas.py` (semilla 42, variante de 54) y les pone una letra de
panel. Cero `fit`, cero inferencia, cero lectura de splits.

ENTRADA (Resultados/figuras/)
  anomalias_cm_{IsolationForest,OneClassSVM,LocalOutlierFactor,Autoencoder}_54.png
  firmas_cm_{DecisionTree,RandomForest,KNN,HistGradientBoosting}_54.png

SALIDA (Resultados/figuras/)
  fig-cm-anomalias-54-paneles.png
  fig-cm-firmas-54-paneles.png

Uso:
    python figura_paneles_cm.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

import config

PANELES = {
    "fig-cm-anomalias-54-paneles.png": ["anomalias_cm_IsolationForest_54.png",
                                        "anomalias_cm_OneClassSVM_54.png",
                                        "anomalias_cm_LocalOutlierFactor_54.png",
                                        "anomalias_cm_Autoencoder_54.png"],
    "fig-cm-firmas-54-paneles.png": ["firmas_cm_DecisionTree_54.png",
                                     "firmas_cm_RandomForest_54.png",
                                     "firmas_cm_KNN_54.png",
                                     "firmas_cm_HistGradientBoosting_54.png"],
}


def _fuente(tam):
    for nombre in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def componer(entradas, salida, margen=40):
    imgs = []
    for f in entradas:
        ruta = os.path.join(config.FIGURAS_DIR, f)
        if not os.path.exists(ruta):
            raise SystemExit("Falta %s: ejecuta antes anomalias.py / firmas.py" % ruta)
        imgs.append(Image.open(ruta).convert("RGB"))
    ancho = max(i.width for i in imgs)
    alto = max(i.height for i in imgs)
    lienzo = Image.new("RGB", (2 * ancho + 3 * margen, 2 * alto + 3 * margen), "white")
    dib = ImageDraw.Draw(lienzo)
    fuente = _fuente(int(alto * 0.05))
    for k, im in enumerate(imgs):
        fila, col = divmod(k, 2)
        x = margen + col * (ancho + margen) + (ancho - im.width) // 2
        y = margen + fila * (alto + margen) + (alto - im.height) // 2
        lienzo.paste(im, (x, y))
        dib.text((margen + col * (ancho + margen), margen + fila * (alto + margen) - margen // 2),
                 "(%s)" % "abcd"[k], fill="black", font=fuente)
    destino = os.path.join(config.FIGURAS_DIR, salida)
    lienzo.save(destino, dpi=(200, 200))
    print("done ->", destino, lienzo.size)
    return destino


if __name__ == "__main__":
    for salida, entradas in PANELES.items():
        componer(entradas, salida)
