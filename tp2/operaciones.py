"""
operaciones.py — TP2: Aritmética de píxeles

Operaciones entre dos imágenes A y B, píxel a píxel:

    1. Cuasi suma clampeada      (RGB y YIQ)
    2. Cuasi suma promediada     (RGB y YIQ)
    3. Cuasi suma "if-lighter"   (YIQ)
    4. Cuasi resta clampeada     (RGB)
    5. Cuasi resta promediada    (RGB)
    6. Cuasi resta "if-darker"   (YIQ)

Workflow de cada operación:

    1. Normalizar los bytes de A y B a 0..1
    2. (Solo las de YIQ) RGB -> YIQ
    3. Combinar A y B píxel a píxel
    4. (Solo las de YIQ) YIQ -> RGB
    5. Limitar a 0..1 y convertir a bytes

Igual que en el TP0 y el TP1, las operaciones recorren la imagen
con dos bucles (y, x) y trabajan sobre cada píxel. PySide6 solo se
usa para abrir y guardar el archivo.
"""

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
# PREPARACIÓN DE LAS IMÁGENES
# ---------------------------------------------------------------------------
def normalizar(img: np.ndarray) -> np.ndarray:
    """
    Paso 1 del workflow: bytes 0..255 -> reales 0..1.
    """

    return img.astype(np.float32) / 255.0


def a_bytes(rgb: np.ndarray) -> np.ndarray:
    """
    Último paso del workflow: reales 0..1 -> bytes 0..255.

    Los valores fuera de rango se limitan (clamp) antes de convertir.
    """

    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def reducir(img: np.ndarray, ancho_max: int) -> np.ndarray:
    """
    Achica la imagen (bytes) para que su ancho no supere ancho_max,
    manteniendo la proporción. Es el mismo criterio de read_img()
    del notebook, pero hecho a mano con "vecino más cercano":

        cada píxel (y, x) de la imagen nueva toma el píxel
        (y / escala, x / escala) de la original.

    Si la imagen ya es más chica, se devuelve sin cambios.
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


def recortar_al_tamano_comun(
    img_a: np.ndarray,
    img_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Las operaciones combinan el píxel (y, x) de A con el píxel (y, x)
    de B, así que las dos imágenes tienen que tener el mismo tamaño.

    Si no lo tienen, se recortan ambas al área común
    (alto mínimo × ancho mínimo), desde la esquina superior izquierda.
    """

    alto = min(img_a.shape[0], img_b.shape[0])
    ancho = min(img_a.shape[1], img_b.shape[1])

    return img_a[:alto, :ancho].copy(), img_b[:alto, :ancho].copy()


# ---------------------------------------------------------------------------
# ESPACIO YIQ (mismas fórmulas que el TP1)
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


def _promedio_ponderado(c1: float, c2: float, y1: float, y2: float) -> float:
    """
    Cromaticidad (I o Q) del resultado en las cuasi sumas YIQ:

        C3 = (Y1 · C1 + Y2 · C2) / (Y1 + Y2)

    Cada imagen aporta color en proporción a su luminancia: un píxel
    oscuro casi no "tiñe" el resultado. Si los dos píxeles son negros
    (Y1 + Y2 = 0) se usa el promedio simple para no dividir por cero.
    """

    suma_y = y1 + y2

    if suma_y <= 1e-12:
    
        return (c1 + c2) / 2

    return (y1 * c1 + y2 * c2) / suma_y


# ---------------------------------------------------------------------------
# 1 y 2) CUASI SUMAS EN RGB
# ---------------------------------------------------------------------------
def suma_rgb_clampeada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    C = A + B, canal por canal; si pasa de 1 se "clampea" a 1.

    Aclara la imagen y satura enseguida las zonas claras.
    """

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            r1, g1, b1 = img_a[y, x]
            r2, g2, b2 = img_b[y, x]

            r3 = r1 + r2 if r1 + r2 < 1 else 1.0
            g3 = g1 + g2 if g1 + g2 < 1 else 1.0
            b3 = b1 + b2 if b1 + b2 < 1 else 1.0

            nueva[y, x] = [r3, g3, b3]

    return nueva


def suma_rgb_promediada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    C = (A + B) / 2, canal por canal.

    Nunca se sale de rango: es una "mezcla" 50 % / 50 % de las dos.
    """

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            r1, g1, b1 = img_a[y, x]
            r2, g2, b2 = img_b[y, x]

            nueva[y, x] = [(r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2]

    return nueva


# ---------------------------------------------------------------------------
# 1, 2 y 3) CUASI SUMAS EN YIQ
# ---------------------------------------------------------------------------
def suma_yiq_clampeada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    En YIQ se separa luminancia de cromaticidad:

        Y3 = Y1 + Y2          (limitada a 1)
        I3 = (Y1 I1 + Y2 I2) / (Y1 + Y2)
        Q3 = (Y1 Q1 + Y2 Q2) / (Y1 + Y2)
    """

    yiq_a = rgb_a_yiq(img_a)
    yiq_b = rgb_a_yiq(img_b)

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            y1, i1, q1 = yiq_a[y, x]
            y2, i2, q2 = yiq_b[y, x]

            y3 = y1 + y2 if y1 + y2 < 1 else 1.0
            i3 = _promedio_ponderado(i1, i2, y1, y2)
            q3 = _promedio_ponderado(q1, q2, y1, y2)

            nueva[y, x] = [y3, i3, q3]

    return yiq_a_rgb(nueva)


def suma_yiq_promediada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Igual que la clampeada, pero la luminancia se promedia:

        Y3 = (Y1 + Y2) / 2
        I3, Q3 = promedio ponderado por la luminancia
    """

    yiq_a = rgb_a_yiq(img_a)
    yiq_b = rgb_a_yiq(img_b)

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            y1, i1, q1 = yiq_a[y, x]
            y2, i2, q2 = yiq_b[y, x]

            y3 = (y1 + y2) / 2
            i3 = _promedio_ponderado(i1, i2, y1, y2)
            q3 = _promedio_ponderado(q1, q2, y1, y2)

            nueva[y, x] = [y3, i3, q3]

    return yiq_a_rgb(nueva)


def suma_if_lighter(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    "If-lighter": en cada píxel gana el más claro.

    Se compara la luminancia Y y se copia el píxel completo (Y, I, Q)
    de la imagen ganadora, así el color no se mezcla:

        si Y1 >= Y2  ->  (Y1, I1, Q1)
        si no        ->  (Y2, I2, Q2)
    """

    yiq_a = rgb_a_yiq(img_a)
    yiq_b = rgb_a_yiq(img_b)

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            y1, i1, q1 = yiq_a[y, x]
            y2, i2, q2 = yiq_b[y, x]

            if y1 >= y2:
                nueva[y, x] = [y1, i1, q1]
            else:
                nueva[y, x] = [y2, i2, q2]

    return yiq_a_rgb(nueva)


# ---------------------------------------------------------------------------
# 4, 5 y 6) CUASI RESTAS
# ---------------------------------------------------------------------------
def resta_rgb_clampeada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    C = A - B, canal por canal; si da negativo se "clampea" a 0.

    Queda lo que A tiene "de más" respecto de B: sirve, por ejemplo,
    para ver diferencias entre dos fotos de la misma escena.
    """

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            r1, g1, b1 = img_a[y, x]
            r2, g2, b2 = img_b[y, x]

            r3 = r1 - r2 if r1 - r2 > 0 else 0.0
            g3 = g1 - g2 if g1 - g2 > 0 else 0.0
            b3 = b1 - b2 if b1 - b2 > 0 else 0.0

            nueva[y, x] = [r3, g3, b3]

    return nueva


def resta_rgb_promediada(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Como en el notebook: a A se le resta el promedio de A y B,

        C = A - (A + B) / 2 = (A - B) / 2

    canal por canal, y lo negativo se "clampea" a 0.
    """

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            r1, g1, b1 = img_a[y, x]
            r2, g2, b2 = img_b[y, x]

            r3 = r1 - (r1 + r2) / 2
            g3 = g1 - (g1 + g2) / 2
            b3 = b1 - (b1 + b2) / 2

            nueva[y, x] = [max(r3, 0.0), max(g3, 0.0), max(b3, 0.0)]

    return nueva


def resta_if_darker(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    "If-darker": en cada píxel gana el más oscuro (la menor Y),
    copiando el píxel completo (Y, I, Q) de esa imagen.
    """

    yiq_a = rgb_a_yiq(img_a)
    yiq_b = rgb_a_yiq(img_b)

    height, width, _ = img_a.shape

    nueva = np.zeros_like(img_a)

    for y in range(height):

        for x in range(width):

            y1, i1, q1 = yiq_a[y, x]
            y2, i2, q2 = yiq_b[y, x]

            if y1 <= y2:
                nueva[y, x] = [y1, i1, q1]
            else:
                nueva[y, x] = [y2, i2, q2]

    return yiq_a_rgb(nueva)


# ---------------------------------------------------------------------------
# CATÁLOGO PARA LA INTERFAZ
# ---------------------------------------------------------------------------
# grupo -> lista de (nombre de la función, texto del botón, fórmula)
OPERACIONES = {
    "Cuasi sumas": [
        ("suma_rgb_clampeada", "RGB clampeada", "C = min(A + B, 1)"),
        ("suma_rgb_promediada", "RGB promediada", "C = (A + B) / 2"),
        ("suma_yiq_clampeada", "YIQ clampeada",
         "Y = min(Ya + Yb, 1);  I, Q = promedio ponderado por Y"),
        ("suma_yiq_promediada", "YIQ promediada",
         "Y = (Ya + Yb) / 2;  I, Q = promedio ponderado por Y"),
        ("suma_if_lighter", "If-lighter",
         "gana el píxel de mayor Y (se copia su Y, I, Q)"),
    ],
    "Cuasi restas": [
        ("resta_rgb_clampeada", "RGB clampeada", "C = max(A - B, 0)"),
        ("resta_rgb_promediada", "RGB promediada", "C = max(A - (A + B) / 2, 0)"),
        ("resta_if_darker", "If-darker",
         "gana el píxel de menor Y (se copia su Y, I, Q)"),
    ],
}
