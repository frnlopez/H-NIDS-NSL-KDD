# -*- coding: utf-8 -*-
"""
figura_nids_bloques.py — Figura conceptual del apartado 2.2.2 de la memoria: los
cinco bloques funcionales de un NIDS con la frontera de alcance de este trabajo
marcada encima.

QUÉ ES Y QUÉ NO ES
------------------
Es una figura CONCEPTUAL para el capítulo 2 (marco teórico). NO es el pipeline de
preprocesado. Lo que esta figura dibuja es la cadena canónica de bloques de un
NIDS —sensor, extracción de características, motor de análisis, generación de
alertas y consola— y, sobre ella, qué parte implementa el trabajo y qué parte
queda fuera.

CONTENIDO
---------
Los cinco bloques y los tres estados de alcance —fuera / parcial / núcleo— son
los de la tabla de correspondencia con este proyecto del apartado 2.2.2 de la
memoria. Si esa tabla cambia, esta figura hay que regenerarla.

El detalle interno del motor de análisis (etapa 1 de anomalías → etapa 2 de firmas,
con la clase `unknown` para lo anómalo no atribuible) es el que implementa
`hibrido.py`; se dibuja a nivel de caja, sin entrar en algoritmos ni en
hiperparámetros.

SALIDA
------
  Resultados/figuras/fig-nids-bloques.png       (300 dpi)

No se edita a mano; se regenera ejecutando este script.

REPRODUCIBILIDAD
----------------
El dibujo es determinista y no consume ningún generador aleatorio: no hay muestreo,
ni ajuste, ni datos. Por eso este script NO expone `--semilla` ni llama a
`config.fijar_semilla()` —no habría nada que sembrar— y tampoco escribe en ninguna
tabla de métricas. Usa `config.py` solo para las rutas y para `ensure_dirs()`.

Uso:
    python figura_nids_bloques.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # backend no interactivo: nunca bloquea (nada de plt.show)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import config


# --- Paleta de los tres estados de alcance ---------------------------------
# Coherente con el resto de figuras del proyecto, que van en tonos de azul
# (cmap "Blues" en las matrices de confusión): el bloque implementado es el azul
# saturado, el parcial un ámbar suave y lo que queda fuera, gris.
COLOR_NUCLEO = "#1f6fb4"       # dentro de alcance (núcleo del trabajo)
COLOR_NUCLEO_BORDE = "#12456f"
COLOR_PARCIAL = "#e8a33d"      # parcial
COLOR_PARCIAL_BORDE = "#a46c14"
COLOR_FUERA = "#d9d9d9"        # fuera de alcance
COLOR_FUERA_BORDE = "#8c8c8c"
COLOR_FRONTERA = "#12456f"

# Estados y su leyenda (el texto es el de la tabla del apartado 2.2.2).
ESTADOS = {
    "nucleo": (COLOR_NUCLEO, COLOR_NUCLEO_BORDE, "white"),
    "parcial": (COLOR_PARCIAL, COLOR_PARCIAL_BORDE, "black"),
    "fuera": (COLOR_FUERA, COLOR_FUERA_BORDE, "black"),
}

# Nombre de la figura de salida.
NOMBRE_FIGURA = "fig-nids-bloques.png"


class FiguraBloquesNIDS:
    """
    Genera el diagrama de bloques del NIDS con la frontera de alcance del trabajo.

    Una sola responsabilidad: dibujar y guardar. No lee datos, no entrena nada y
    no toca ninguna tabla de métricas.
    """

    # Los cinco bloques canónicos, en el orden de la cadena. Cada uno:
    #   (titulo, detalle, estado, etiqueta de alcance bajo la caja)
    BLOQUES = [
        ("1. Sensor / captura",
         "Obtiene el tráfico\ndel segmento vigilado\n(tap o puerto espejo)",
         "fuera",
         "Fuera de alcance\n(lo asume el dataset)"),
        ("2. Extracción de\ncaracterísticas",
         "Reensambla paquetes\nen conexiones y calcula\nsus atributos",
         "fuera",
         "Fuera de alcance\n(NSL-KDD ya entrega\nel vector)"),
        ("3. Motor de análisis",
         "",  # su detalle son las dos subcajas de la cascada
         "nucleo",
         "Núcleo del trabajo"),
        ("4. Generación de\nalertas",
         "Convierte el veredicto\nen un evento\ncon contexto",
         "parcial",
         "Parcial: se emite la\netiqueta por conexión,\nno el evento enriquecido"),
        ("5. Consola del\nanalista",
         "Presenta, agrupa\ny prioriza las alertas",
         "fuera",
         "Fuera de alcance"),
    ]

    # Geometría en coordenadas de datos (lienzo 0-106 x 0-100). Las anchuras están
    # ajustadas para que NINGÚN rótulo se salga de su caja: si se alarga un texto
    # de BLOQUES hay que volver a mirar la figura, no solo re-ejecutar.
    ANCHO_BLOQUE = 18.0
    HUECO = 4.0
    Y_CENTRO = 62.0          # eje de la cadena: por aquí pasan todas las flechas
    ALTO_BLOQUE = 22.0       # bloques normales
    ALTO_MOTOR = 46.0        # el motor es más alto: lleva la cascada dentro

    def __init__(self, nombre_figura=NOMBRE_FIGURA):
        self.nombre_figura = nombre_figura
        self.ruta_figura = os.path.join(config.FIGURAS_DIR, nombre_figura)

    # -- primitivas de dibujo ------------------------------------------------
    def _caja(self, ax, x, y, ancho, alto, estado, lw=1.8, ls="solid"):
        """Rectángulo coloreado según el estado de alcance."""
        relleno, borde, _ = ESTADOS[estado]
        ax.add_patch(Rectangle((x, y), ancho, alto, facecolor=relleno,
                               edgecolor=borde, linewidth=lw, linestyle=ls,
                               zorder=2))

    def _flecha(self, ax, x0, y0, x1, y1, color="#444444"):
        """Flecha recta entre dos puntos (une bloques consecutivos)."""
        ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.35,
                                head_width=1.8, head_length=1.6,
                                length_includes_head=True, color=color,
                                zorder=3))

    # -- composición ---------------------------------------------------------
    def _x_bloque(self, i):
        """Borde izquierdo del bloque i-ésimo."""
        return i * (self.ANCHO_BLOQUE + self.HUECO)

    def _dibujar_cadena(self, ax):
        """Los cinco bloques, sus rótulos y las flechas que los encadenan."""
        for i, (titulo, detalle, estado, alcance) in enumerate(self.BLOQUES):
            x = self._x_bloque(i)
            es_motor = (estado == "nucleo")
            alto = self.ALTO_MOTOR if es_motor else self.ALTO_BLOQUE
            y = self.Y_CENTRO - alto / 2.0
            self._caja(ax, x, y, self.ANCHO_BLOQUE, alto, estado,
                       lw=2.6 if es_motor else 1.8)

            _, _, color_texto = ESTADOS[estado]
            cx = x + self.ANCHO_BLOQUE / 2.0

            if es_motor:
                # El título va arriba del bloque alto; el interior lo ocupan las
                # dos etapas de la cascada.
                ax.text(cx, y + alto - 3.6, titulo, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color_texto,
                        zorder=4)
                self._dibujar_cascada(ax, x, y, alto)
            else:
                ax.text(cx, self.Y_CENTRO + 6.0, titulo, ha="center",
                        va="center", fontsize=10, fontweight="bold",
                        color=color_texto, zorder=4)
                ax.text(cx, self.Y_CENTRO - 3.4, detalle, ha="center",
                        va="center", fontsize=7.6, color=color_texto, zorder=4)

            # Rótulo de alcance, siempre bajo la caja y fuera de ella.
            ax.text(cx, self.Y_CENTRO - self.ALTO_MOTOR / 2.0 - 4.0, alcance,
                    ha="center", va="top", fontsize=8.4, style="italic",
                    color="#333333", zorder=4)

            # Flecha hacia el bloque siguiente.
            if i < len(self.BLOQUES) - 1:
                x_fin = x + self.ANCHO_BLOQUE
                self._flecha(ax, x_fin + 0.6, self.Y_CENTRO,
                             x_fin + self.HUECO - 0.6, self.Y_CENTRO)

    def _dibujar_cascada(self, ax, x, y, alto):
        """Las dos etapas encadenadas dentro del bloque «Motor de análisis»."""
        margen = 1.6
        ancho = self.ANCHO_BLOQUE - 2 * margen
        cx = x + self.ANCHO_BLOQUE / 2.0

        etapas = [
            ("Etapa 1\nDetección de anomalías",
             "modelo de lo normal\numbral percentil 95"),
            ("Etapa 2\nClasificación por firmas",
             "4 categorías de ataque\no bien «unknown»"),
        ]
        alto_etapa = 13.0
        y_sup = y + alto - 8.0 - alto_etapa      # etapa 1
        y_inf = y_sup - alto_etapa - 6.0         # etapa 2

        for (titulo, detalle), y_etapa in zip(etapas, (y_sup, y_inf)):
            ax.add_patch(Rectangle((x + margen, y_etapa), ancho, alto_etapa,
                                   facecolor="white", edgecolor="#12456f",
                                   linewidth=1.4, zorder=4))
            ax.text(cx, y_etapa + alto_etapa - 4.4, titulo, ha="center",
                    va="center", fontsize=8.0, fontweight="bold",
                    color="#12456f", zorder=5)
            ax.text(cx, y_etapa + 3.2, detalle, ha="center", va="center",
                    fontsize=7.0, color="#333333", zorder=5)

        # Flecha vertical entre las dos etapas: solo lo que la etapa 1 marca
        # como sospechoso llega a la etapa 2.
        self._flecha(ax, cx, y_sup - 0.6, cx, y_inf + alto_etapa + 0.6,
                     color="#12456f")
        ax.text(cx + 0.9, (y_sup + y_inf + alto_etapa) / 2.0, "sospechoso",
                ha="left", va="center", fontsize=7.0, color="#12456f",
                zorder=5)

    def _dibujar_frontera(self, ax):
        """Recuadro discontinuo que encierra lo implementado en este trabajo."""
        x0 = self._x_bloque(2) - 3.0
        x1 = self._x_bloque(3) + self.ANCHO_BLOQUE + 3.0
        y0 = self.Y_CENTRO - self.ALTO_MOTOR / 2.0 - 12.5
        y1 = self.Y_CENTRO + self.ALTO_MOTOR / 2.0 + 6.0
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=COLOR_FRONTERA, linewidth=2.2,
                               linestyle=(0, (6, 4)), zorder=1))
        ax.text((x0 + x1) / 2.0, y1 + 1.6,
                "Frontera de alcance de este TFG",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=COLOR_FRONTERA, zorder=4)

    def _dibujar_extremos(self, ax):
        """Entrada y salida de la cadena, tal y como las fija el apartado 2.2.2."""
        ax.text(-2.5, self.Y_CENTRO, "Tráfico\nde red", ha="right",
                va="center", fontsize=9, color="#333333")
        self._flecha(ax, -2.0, self.Y_CENTRO, -0.6, self.Y_CENTRO)
        x_fin = self._x_bloque(4) + self.ANCHO_BLOQUE
        self._flecha(ax, x_fin + 0.6, self.Y_CENTRO, x_fin + 2.0,
                     self.Y_CENTRO)
        ax.text(x_fin + 2.5, self.Y_CENTRO, "Analista", ha="left",
                va="center", fontsize=9, color="#333333")

    def _dibujar_leyenda(self, ax):
        """Leyenda de los tres estados de alcance."""
        items = [
            ("nucleo", "Dentro de alcance (implementado)"),
            ("parcial", "Parcial"),
            ("fuera", "Fuera de alcance"),
        ]
        y = 14.0
        x = 8.0
        for estado, texto in items:
            relleno, borde, _ = ESTADOS[estado]
            ax.add_patch(Rectangle((x, y), 3.4, 3.4, facecolor=relleno,
                                   edgecolor=borde, linewidth=1.4, zorder=2))
            ax.text(x + 4.4, y + 1.7, texto, ha="left", va="center",
                    fontsize=9, color="#222222")
            x += 32.0

        ax.text(53.0, 5.0,
                "Entrada: registros de conexiones ya resumidos en su vector de "
                "características (NSL-KDD).\nSalida: una etiqueta por conexión "
                "—normal, dos, probe, r2l, u2r o unknown—, no una respuesta "
                "ejecutada.",
                ha="center", va="center", fontsize=8.6, color="#333333")

    # -- API -----------------------------------------------------------------
    def generar(self):
        """Dibuja la figura y la guarda en `Resultados/figuras/`."""
        config.ensure_dirs()

        fig, ax = plt.subplots(figsize=(14, 7.5))
        ax.set_xlim(-15, 121)
        ax.set_ylim(0, 100)
        ax.axis("off")

        self._dibujar_frontera(ax)
        self._dibujar_cadena(ax)
        self._dibujar_extremos(ax)
        self._dibujar_leyenda(ax)

        ax.set_title(
            "Bloques funcionales de un NIDS y frontera de alcance de este TFG",
            fontsize=13, fontweight="bold", pad=14)

        fig.savefig(self.ruta_figura, dpi=300, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return self.ruta_figura


def main():
    config.setup_utf8()
    figura = FiguraBloquesNIDS()
    ruta_figura = figura.generar()
    print("Figura generada:")
    print("  " + ruta_figura)


if __name__ == "__main__":
    main()
