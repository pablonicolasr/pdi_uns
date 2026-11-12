"""
operaciones.py — TP3: Operaciones de luminancia

Workflow (el mismo del notebook):

    1. Normalizar los valores RGB del píxel (bytes -> 0..1)
    2. RGB -> YIQ
    3. Y' := f(Y)   (raíz, cuadrado, lineal a trozos, sigmoide o ecualización)
    4. Y'IQ -> R'G'B'   (la cromaticidad I, Q no se toca)
    5. Convertir R'G'B' a bytes y mostrar

Como en el TP0 y el TP1, se recorre la imagen con bucles (y, x) y se
trabaja píxel por píxel. El histograma, su acumulado y la tabla de
la ecualización también se calculan a mano. PySide6 solo se usa para
abrir y guardar el archivo.
"""

import math

import numpy as np
from PySide6.QtGui import QImage


# ---------------------------------------------------------------------------
# ARCHIVOS
# ---------------------------------------------------------------------------
def abrir_imagen(ruta: str) -> QImage | None:
    """Abre la imagen del archivo. Devuelve None si no se pudo abrir."""

    imagen = QImage(ruta)

    if imagen.isNull():
        return None

    return imagen


def guardar_imagen(imagen: QImage, ruta: str) -> bool:
    """Guarda la imagen; el formato sale de la extensión (.png, .jpg, .bmp...)."""

    return imagen.save(ruta)


# ---------------------------------------------------------------------------
# PASOS 1 Y 5 DEL WORKFLOW
# ---------------------------------------------------------------------------
def normalizar(img: np.ndarray) -> np.ndarray:
    """Paso 1: bytes 0..255 -> reales 0..1."""

    return img.astype(np.float32) / 255.0


def a_bytes(rgb: np.ndarray) -> np.ndarray:
    """Paso 5: reales 0..1 -> bytes 0..255 (lo que se sale de rango se limita)."""

    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def reducir(img: np.ndarray, ancho_max: int) -> np.ndarray:
    """
    Achica la imagen (bytes) para que su ancho no supere ancho_max,
    como read_img() del notebook (que la llevaba a 800 px de ancho).

    Se hace a mano con "vecino más cercano": el píxel (y, x) de la
    imagen nueva toma el píxel (y / escala, x / escala) de la original.
    """

    alto, ancho, canales = img.shape

    if ancho_max <= 0 or ancho <= ancho_max:
        return img

    escala = ancho_max / ancho

    nuevo_ancho = ancho_max
    nuevo_alto = max(1, int(alto * escala))

    nueva = np.zeros((nuevo_alto, nuevo_ancho, canales), dtype=img.dtype)

    for y in range(nuevo_alto):

        y_orig = min(int(y / escala), alto - 1)

        for x in range(nuevo_ancho):

            x_orig = min(int(x / escala), ancho - 1)

            nueva[y, x] = img[y_orig, x_orig]

    return nueva


# ---------------------------------------------------------------------------
# PASOS 2 Y 4: RGB <-> YIQ (mismas fórmulas que el TP1)
# ---------------------------------------------------------------------------
def rgb_a_yiq(rgb: np.ndarray) -> np.ndarray:
    """
    RGB normalizado -> YIQ, píxel por píxel.

        Y = 0.299    R + 0.587    G + 0.114    B
        I = 0.595716 R - 0.274453 G - 0.321263 B
        Q = 0.211456 R - 0.522591 G + 0.311135 B
    """

    height, width, _ = rgb.shape

    yiq_img = np.zeros_like(rgb)

    for y in range(height):

        for x in range(width):

            r, g, b = rgb[y, x]

            y_val = 0.299 * r + 0.587 * g + 0.114 * b
            i_val = 0.595716 * r - 0.274453 * g - 0.321263 * b
            q_val = 0.211456 * r - 0.522591 * g + 0.311135 * b

            yiq_img[y, x] = [y_val, i_val, q_val]

    return yiq_img


def yiq_a_rgb(yiq: np.ndarray) -> np.ndarray:
    """
    YIQ -> RGB, píxel por píxel. El resultado se limita a 0..1.

        R = Y + 0.9563 I + 0.6210 Q
        G = Y - 0.2721 I - 0.6474 Q
        B = Y - 1.1070 I + 1.7046 Q
    """

    height, width, _ = yiq.shape

    rgb_img = np.zeros_like(yiq)

    for y in range(height):

        for x in range(width):

            y_val, i_val, q_val = yiq[y, x]

            r = y_val + 0.9563 * i_val + 0.6210 * q_val
            g = y_val - 0.2721 * i_val - 0.6474 * q_val
            b = y_val - 1.1070 * i_val + 1.7046 * q_val

            rgb_img[y, x] = [r, g, b]

    return np.clip(rgb_img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# ESTADÍSTICAS DE LA LUMINANCIA (a mano)
# ---------------------------------------------------------------------------
def minimo_maximo_y(yiq: np.ndarray) -> tuple[float, float]:
    """Recorre la imagen y devuelve el menor y el mayor valor de Y."""

    height, width, _ = yiq.shape

    y_min = float(yiq[0, 0, 0])
    y_max = float(yiq[0, 0, 0])

    for y in range(height):

        for x in range(width):

            valor = yiq[y, x, 0]

            if valor < y_min:
                y_min = float(valor)

            if valor > y_max:
                y_max = float(valor)

    return y_min, y_max


def media_y(yiq: np.ndarray) -> float:
    """Promedio de Y: suma de todos los valores dividida la cantidad de píxeles."""

    height, width, _ = yiq.shape

    suma = 0.0

    for y in range(height):

        for x in range(width):

            suma += float(yiq[y, x, 0])

    return suma / (height * width)


def _nivel(y_val: float, niveles: int = 256) -> int:
    """Y (0..1) -> nivel entero 0..niveles-1 (casillero del histograma)."""

    nivel = int(y_val * (niveles - 1) + 0.5)

    return min(max(nivel, 0), niveles - 1)


def histograma_y(yiq: np.ndarray, niveles: int = 256) -> np.ndarray:
    """
    Histograma de la luminancia: cuántos píxeles hay en cada nivel de Y.

    Se recorre la imagen y se suma 1 en el casillero que corresponde
    al Y de cada píxel.
    """

    height, width, _ = yiq.shape

    hist = np.zeros(niveles, dtype=np.int64)

    for y in range(height):

        for x in range(width):

            hist[_nivel(yiq[y, x, 0], niveles)] += 1

    return hist


def histograma_acumulado(hist: np.ndarray) -> np.ndarray:
    """
    Histograma acumulado (CDF sin normalizar):

        acumulado[k] = hist[0] + hist[1] + ... + hist[k]
    """

    acumulado = np.zeros(len(hist), dtype=np.int64)

    suma = 0

    for k in range(len(hist)):

        suma += int(hist[k])

        acumulado[k] = suma

    return acumulado


# ---------------------------------------------------------------------------
# PASO 3: Y' := f(Y)
# Todas reciben la imagen YIQ y devuelven una copia con Y cambiada.
# ---------------------------------------------------------------------------
def aplicar_raiz(yiq: np.ndarray) -> np.ndarray:
    """
    Y' = raíz(Y)  ->  aclara la imagen ("low luminance" del notebook).

    Como 0 <= Y <= 1, la raíz levanta más los valores oscuros.
    """

    nueva = np.copy(yiq)

    height, width, _ = nueva.shape

    for y in range(height):

        for x in range(width):

            y_val = max(float(nueva[y, x, 0]), 0.0)

            nueva[y, x, 0] = math.sqrt(y_val)

    return nueva


def aplicar_cuadrado(yiq: np.ndarray) -> np.ndarray:
    """
    Y' = Y²  ->  oscurece la imagen ("high luminance" del notebook).
    """

    nueva = np.copy(yiq)

    height, width, _ = nueva.shape

    for y in range(height):

        for x in range(width):

            y_val = float(nueva[y, x, 0])

            nueva[y, x, 0] = y_val * y_val

    return nueva


def aplicar_lineal_a_trozos(
    yiq: np.ndarray,
    y_min: float,
    y_max: float
) -> np.ndarray:
    """
    Lineal a trozos (estiramiento de contraste):

        Y' = 0                              si Y <= y_min
        Y' = (Y - y_min) / (y_max - y_min)  si y_min < Y < y_max
        Y' = 1                              si Y >= y_max

    Con y_min y y_max iguales al mínimo y máximo de la imagen se
    obtiene lo mismo que en el notebook: el rango de Y se estira a 0..1.
    """

    nueva = np.copy(yiq)

    height, width, _ = nueva.shape

    rango = y_max - y_min

    for y in range(height):

        for x in range(width):

            y_val = float(nueva[y, x, 0])

            if rango <= 1e-8:
                # y_min == y_max: queda un umbral
                y_nuevo = 1.0 if y_val >= y_max else 0.0
            elif y_val <= y_min:
                y_nuevo = 0.0
            elif y_val >= y_max:
                y_nuevo = 1.0
            else:
                y_nuevo = (y_val - y_min) / rango

            nueva[y, x, 0] = y_nuevo

    return nueva


def aplicar_sigmoide(
    yiq: np.ndarray,
    k: float,
    x0: float | None = None
) -> np.ndarray:
    """
    Normalización con una curva sigmoide (curva "S"):

        1) Yn = (Y - Ymin) / (Ymax - Ymin)       (Y llevada a 0..1)
        2) Y' = 1 / (1 + e^(-k · (Yn - x0)))

    k  : pendiente. Más grande, más contraste alrededor de x0.
    x0 : punto medio de la curva. Si es None se usa la media de Yn.
    """

    nueva = np.copy(yiq)

    height, width, _ = nueva.shape

    y_min, y_max = minimo_maximo_y(yiq)

    rango = (y_max - y_min) or 1.0

    # paso 1: llevar Y a 0..1
    for y in range(height):

        for x in range(width):

            nueva[y, x, 0] = (float(nueva[y, x, 0]) - y_min) / rango

    if x0 is None:
        x0 = media_y(nueva)

    # paso 2: aplicar la curva
    for y in range(height):

        for x in range(width):

            y_norm = float(nueva[y, x, 0])

            nueva[y, x, 0] = 1.0 / (1.0 + math.exp(-k * (y_norm - x0)))

    return nueva


def tabla_ecualizacion(hist: np.ndarray) -> np.ndarray:
    """
    Tabla (LUT) de la ecualización: nivel viejo -> nivel nuevo (0..1).

        cdf[k]   = histograma acumulado
        cdf_min  = primer valor no nulo del acumulado
        nuevo[k] = (cdf[k] - cdf_min) / (N - cdf_min)

    Así los niveles se reparten de forma pareja entre 0 y 1.
    """

    cdf = histograma_acumulado(hist)

    total = int(cdf[-1])

    cdf_min = 0

    for valor in cdf:
        if valor > 0:
            cdf_min = int(valor)
            break

    tabla = np.zeros(len(hist), dtype=np.float32)

    denominador = total - cdf_min

    for k in range(len(hist)):

        if denominador <= 0:
            # imagen de un solo nivel: no hay nada que repartir
            tabla[k] = k / (len(hist) - 1)
        else:
            tabla[k] = max(0.0, (int(cdf[k]) - cdf_min) / denominador)

    return tabla


def aplicar_ecualizacion(yiq: np.ndarray, niveles: int = 256) -> np.ndarray:
    """
    Ecualización del histograma de Y:

        1) histograma de Y (256 niveles)
        2) tabla de ecualización a partir del acumulado
        3) a cada píxel se le asigna Y' = tabla[nivel de Y]
    """

    hist = histograma_y(yiq, niveles)

    tabla = tabla_ecualizacion(hist)

    nueva = np.copy(yiq)

    height, width, _ = nueva.shape

    for y in range(height):

        for x in range(width):

            nueva[y, x, 0] = tabla[_nivel(nueva[y, x, 0], niveles)]

    return nueva


# Métodos que ofrece la interfaz: clave -> (nombre, descripción)
METODOS = {
    "raiz": ("Raíz  (Y' = √Y)", "Aclara: levanta más los tonos oscuros."),
    "cuadrado": ("Cuadrado  (Y' = Y²)", "Oscurece: baja más los tonos medios."),
    "lineal": ("Lineal a trozos", "Estira el rango [Ymin, Ymax] a [0, 1]."),
    "sigmoide": ("Sigmoide", "Curva S: más contraste alrededor de x0."),
    "ecualizacion": ("Ecualización", "Reparte los niveles de Y de forma pareja."),
}
