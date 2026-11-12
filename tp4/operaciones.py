"""
operaciones.py — TP4: Transformada de Fourier

1. Transformada de Fourier (TF) directa e inversa de imágenes.
2. Figuras geométricas para probarla: círculo, cuadrado, rectángulo,
   rombo y elipse (con desplazamiento y rotación, y varias en la misma
   imagen).
3. Guardar el espectro de amplitud en un bitmap, editarlo afuera y
   aplicar la TF inversa preservando la fase.

Workflow (el del notebook):

    1. Normalizar RGB (bytes -> 0..1) y pasar a YIQ
    2. TF 2D de la luminancia Y   (opcional: restarle antes la media)
    3. Centrar el espectro (llevar la frecuencia 0 al medio)
    4. Amplitud |F| y fase arg(F)  -> se muestran como imágenes
    5. TF inversa -> Y reconstruida; con la I y Q originales -> RGB

La TF no usa np.fft: se puede calcular con la suma directa de cuatro
bucles (la versión comentada del notebook), con bucles separando
columnas y filas, o con la matriz de la DFT armada a mano. El centrado
del espectro (fftshift) también se hace a mano. PySide6 solo se usa para abrir y guardar
archivos.
"""

import cmath
import json
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


def guardar_metadatos(meta: dict, ruta: str) -> None:
    """Guarda en JSON lo necesario para volver del bitmap a la amplitud."""

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def leer_metadatos(ruta: str) -> dict:
    """Lee el JSON guardado junto al bitmap de amplitud."""

    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PREPARACIÓN
# ---------------------------------------------------------------------------
def normalizar(img: np.ndarray) -> np.ndarray:
    """Bytes 0..255 -> reales 0..1."""

    return img.astype(np.float64) / 255.0


def a_bytes(rgb: np.ndarray) -> np.ndarray:
    """Reales 0..1 -> bytes 0..255 (lo que se sale de rango se limita)."""

    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def gris_a_rgb(gris: np.ndarray) -> np.ndarray:
    """Imagen de un canal (alto, ancho) -> (alto, ancho, 3) repitiendo el valor."""

    alto, ancho = gris.shape

    rgb = np.zeros((alto, ancho, 3), dtype=gris.dtype)

    for c in range(3):
        rgb[:, :, c] = gris

    return rgb


def reducir(img: np.ndarray, ancho_max: int) -> np.ndarray:
    """
    Achica la imagen (bytes) para que su ancho no supere ancho_max,
    como read_img() del notebook (que la llevaba a 1024 px de ancho).

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
# RGB <-> YIQ (mismas fórmulas que el TP1)
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
# FIGURAS GEOMÉTRICAS
# Fondo blanco (255) y figura negra (0), como en el notebook.
# ---------------------------------------------------------------------------
FIGURAS = {
    # tipo: (nombre, etiqueta del parámetro 1, etiqueta del parámetro 2 o None)
    "circulo": ("Círculo", "Radio", None),
    "cuadrado": ("Cuadrado", "Lado", None),
    "rectangulo": ("Rectángulo", "Ancho (lx)", "Alto (ly)"),
    "rombo": ("Rombo", "Semidiagonal x (dx)", "Semidiagonal y (dy)"),
    "elipse": ("Elipse", "Semieje x (a)", "Semieje y (b)"),
}


def esta_dentro(tipo: str, u: float, v: float, p1: float, p2: float) -> bool:
    """
    ¿El punto (u, v), medido desde el centro de la figura, cae dentro?

        círculo     u² + v² <= r²
        cuadrado    |u| <= lado/2  y  |v| <= lado/2
        rectángulo  |u| <= lx/2    y  |v| <= ly/2
        rombo       |u|/dx + |v|/dy <= 1
        elipse      u²/a² + v²/b² <= 1
    """

    if tipo == "circulo":
        return u * u + v * v <= p1 * p1

    if tipo == "cuadrado":
        return abs(u) <= p1 / 2 and abs(v) <= p1 / 2

    if tipo == "rectangulo":
        return abs(u) <= p1 / 2 and abs(v) <= p2 / 2

    if tipo == "rombo":
        return abs(u) / p1 + abs(v) / p2 <= 1

    if tipo == "elipse":
        return (u * u) / (p1 * p1) + (v * v) / (p2 * p2) <= 1

    raise ValueError(f"Figura desconocida: {tipo}")


def dibujar_figura(
    tipo: str,
    tamano: int,
    p1: float,
    p2: float = 0.0,
    desplazamiento_x: int = 0,
    desplazamiento_y: int = 0,
    angulo: float = 0.0,
    base: np.ndarray | None = None,
) -> np.ndarray:
    """
    Dibuja una figura negra sobre fondo blanco en una imagen
    cuadrada de tamano × tamano (gris, uint8).

    El centro de la figura queda en el centro de la imagen más el
    desplazamiento. El ángulo (en grados) la rota: para cada píxel
    se calculan sus coordenadas (u, v) en el sistema de la figura

        dx = x - cx,  dy = y - cy
        u  =  dx · cos(θ) + dy · sin(θ)
        v  = -dx · sin(θ) + dy · cos(θ)

    y se pinta si esta_dentro(u, v).

    Si se pasa una imagen base, la figura se agrega encima:
    así se arman imágenes con dos o más objetos.
    """

    if base is None:
        img = np.full((tamano, tamano), 255, dtype=np.uint8)
    else:
        img = np.copy(base)

    alto, ancho = img.shape

    cx = ancho // 2 + desplazamiento_x
    cy = alto // 2 + desplazamiento_y

    theta = math.radians(angulo)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for y in range(alto):

        dy = y - cy

        for x in range(ancho):

            dx = x - cx

            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t

            if esta_dentro(tipo, u, v, p1, p2):
                img[y, x] = 0

    return img


# ---------------------------------------------------------------------------
# TRANSFORMADA DISCRETA DE FOURIER (DFT) 2D, armada a mano
#
# Hay tres formas de calcular la misma suma, de la más "a mano" a la más
# rápida. Las tres dan el mismo resultado (salvo redondeo):
#
#   "directa"     cuatro bucles, la fórmula tal cual (lo que está
#                 comentado en el notebook). Cuesta (M·N)² términos:
#                 solo sirve para imágenes chicas (64 × 64 ya son
#                 16 millones de términos).
#   "separable"   la DFT 2D es la DFT 1D de cada columna y después de
#                 cada fila. Cuesta M·N·(M + N) términos.
#   "matricial"   lo mismo que la separable, pero cada DFT 1D es un
#                 producto por la matriz de la DFT. Es la más rápida.
#
# Ninguna usa np.fft. Normalización como en el notebook:
#   "numpy"    la directa no divide; la inversa divide por M·N
#   "forward"  la directa divide por M·N; la inversa no divide
# ---------------------------------------------------------------------------
METODOS_DFT = {
    "matricial": "Matricial (rápida)",
    "separable": "Separable con bucles (filas y columnas)",
    "directa": "Suma directa, 4 bucles (la del notebook)",
}


def terminos_dft(alto: int, ancho: int, metodo: str) -> int:
    """
    Cantidad de términos de la suma (multiplicaciones complejas) que
    hace cada método para una transformada de alto × ancho.
    """

    if metodo == "directa":
        return (alto * ancho) ** 2

    return alto * ancho * (alto + ancho)


def _avisar(progreso, hecho: int, total: int) -> None:
    """Informa el avance (0..1) a la interfaz, si pidió que se le avise."""

    if progreso is not None:
        progreso(hecho / total)


def dft2_directa(y: np.ndarray, inversa: bool = False, progreso=None) -> np.ndarray:
    """
    DFT 2D con la fórmula tal cual, sin normalizar (es la versión
    comentada del notebook):

        F(u, v) = Σ_i Σ_j  Y(i, j) · e^(∓2πi (u·i/M + v·j/N))

    Signo "-" para la directa y "+" para la inversa. Para cada uno de
    los M·N valores de salida se recorren los M·N valores de entrada.
    """

    height, width = y.shape

    signo = 1.0 if inversa else -1.0

    salida = np.zeros((height, width), dtype=np.complex128)

    two_pi = 2.0 * math.pi

    for u in range(height):

        for v in range(width):

            suma = 0.0 + 0.0j

            for i in range(height):

                ux = (u * i) / height

                for j in range(width):

                    theta = ux + (v * j) / width

                    suma += y[i][j] * cmath.exp(signo * 1j * two_pi * theta)

            salida[u, v] = suma

        _avisar(progreso, u + 1, height)

    return salida


def dft1(x, inversa: bool = False) -> np.ndarray:
    """
    DFT de una dimensión (un vector de n valores), sin normalizar:

        X(u) = Σ_k  x(k) · e^(∓2πi · u·k / n)
    """

    n = len(x)

    signo = 1.0 if inversa else -1.0

    salida = np.zeros(n, dtype=np.complex128)

    for u in range(n):

        suma = 0.0 + 0.0j

        for k in range(n):

            angulo = 2.0 * math.pi * u * k / n

            suma += x[k] * complex(math.cos(angulo), signo * math.sin(angulo))

        salida[u] = suma

    return salida


def dft2_separable(y: np.ndarray, inversa: bool = False, progreso=None) -> np.ndarray:
    """
    DFT 2D separable, sin normalizar:

        1) DFT 1D de cada columna  (recorre los M valores de la columna)
        2) DFT 1D de cada fila del resultado anterior

    Da lo mismo que la directa porque e^(-2πi (u·i/M + v·j/N)) es el
    producto e^(-2πi u·i/M) · e^(-2πi v·j/N).
    """

    height, width = y.shape

    columnas = np.zeros((height, width), dtype=np.complex128)

    for j in range(width):

        columnas[:, j] = dft1(y[:, j], inversa)

        _avisar(progreso, j + 1, width + height)

    salida = np.zeros((height, width), dtype=np.complex128)

    for i in range(height):

        salida[i, :] = dft1(columnas[i, :], inversa)

        _avisar(progreso, width + i + 1, width + height)

    return salida


def matriz_dft(n: int, inversa: bool = False) -> np.ndarray:
    """
    Matriz n × n de la DFT de una dimensión:

        W[u, k] = e^(-2πi · u·k / n)     (directa)
        W[u, k] = e^(+2πi · u·k / n)     (inversa)

    Como e^(-2πi·m/n) se repite cada n, primero se calculan las n
    "raíces" r[m] = cos(2πm/n) - i·sin(2πm/n) y después cada elemento
    es W[u, k] = r[(u·k) mod n].
    """

    signo = 1.0 if inversa else -1.0

    raices = np.zeros(n, dtype=np.complex128)

    for m in range(n):

        angulo = 2.0 * math.pi * m / n

        raices[m] = complex(math.cos(angulo), signo * math.sin(angulo))

    w = np.zeros((n, n), dtype=np.complex128)

    k = np.arange(n)

    for u in range(n):

        # fila u: los exponentes u·k para todos los k a la vez
        w[u, :] = raices[(u * k) % n]

    return w


def dft2_matricial(y: np.ndarray, inversa: bool = False, progreso=None) -> np.ndarray:
    """
    DFT 2D separable escrita con matrices, sin normalizar:

        F = W_M · Y · W_N

    W_M · Y hace la DFT 1D de cada columna y multiplicar a derecha por
    W_N (que es simétrica) la de cada fila.
    """

    height, width = y.shape

    salida = matriz_dft(height, inversa) @ y @ matriz_dft(width, inversa)

    _avisar(progreso, 1, 1)

    return salida


_CALCULOS = {
    "directa": dft2_directa,
    "separable": dft2_separable,
    "matricial": dft2_matricial,
}


def dft2(
    y: np.ndarray,
    metodo: str = "matricial",
    norm: str = "numpy",
    progreso=None
) -> np.ndarray:
    """
    DFT 2D directa de la luminancia (alto M × ancho N):

        F(u, v) = Σ_y Σ_x  Y(y, x) · e^(-2πi (u·y/M + v·x/N))

    Con norm="forward" además se divide por M·N (así F(0, 0) es el
    promedio de Y); con norm="numpy" no se divide (F(0, 0) es la suma).
    """

    f = _CALCULOS[metodo](y, False, progreso)

    if norm == "forward":
        f = f / (y.shape[0] * y.shape[1])

    return f


def idft2(
    f: np.ndarray,
    metodo: str = "matricial",
    norm: str = "numpy",
    progreso=None
) -> np.ndarray:
    """
    DFT inversa 2D:

        Y(y, x) = Σ_u Σ_v  F(u, v) · e^(+2πi (u·y/M + v·x/N))

    dividida por M·N si norm="numpy" (si fue "forward", la directa ya
    dividió). Se devuelve la parte real: la imaginaria es casi cero
    (redondeo) mientras el espectro conserve la simetría conjugada de
    una imagen real.
    """

    y = _CALCULOS[metodo](f, True, progreso)

    if norm == "numpy":
        y = y / (f.shape[0] * f.shape[1])

    return np.real(y)


def centrar_espectro(f: np.ndarray) -> np.ndarray:
    """
    Lleva la frecuencia (0, 0) de la esquina al centro (el "fftshift").

    Es intercambiar los cuadrantes: el elemento (u, v) pasa a la
    posición ((u + M//2) mod M, (v + N//2) mod N).
    """

    alto, ancho = f.shape

    nueva = np.zeros_like(f)

    for u in range(alto):

        nueva[(u + alto // 2) % alto, :] = f[u, :]

    resultado = np.zeros_like(f)

    for v in range(ancho):

        resultado[:, (v + ancho // 2) % ancho] = nueva[:, v]

    return resultado


def descentrar_espectro(f: np.ndarray) -> np.ndarray:
    """
    Deshace centrar_espectro (el "ifftshift"): el elemento (u, v)
    vuelve a la posición ((u - M//2) mod M, (v - N//2) mod N).
    """

    alto, ancho = f.shape

    nueva = np.zeros_like(f)

    for u in range(alto):

        nueva[(u - alto // 2) % alto, :] = f[u, :]

    resultado = np.zeros_like(f)

    for v in range(ancho):

        resultado[:, (v - ancho // 2) % ancho] = nueva[:, v]

    return resultado


# ---------------------------------------------------------------------------
# AMPLITUD Y FASE
# ---------------------------------------------------------------------------
def amplitud_y_fase(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa el espectro complejo F = a + b·i en:

        amplitud  A = √(a² + b²)
        fase      φ = atan2(b, a)      (entre -π y π)
    """

    a = np.real(f)
    b = np.imag(f)

    amplitud = np.sqrt(a * a + b * b)
    fase = np.arctan2(b, a)

    return amplitud, fase


def desde_amplitud_y_fase(amplitud: np.ndarray, fase: np.ndarray) -> np.ndarray:
    """
    Vuelve a armar el espectro complejo:

        F = A · (cos φ + i · sin φ)
    """

    return amplitud * np.cos(fase) + 1j * amplitud * np.sin(fase)


def amplitud_a_bitmap(amplitud: np.ndarray, log: bool = True):
    """
    Amplitud -> imagen gris de bytes, para verla o editarla.

        L = log(1 + A)                 (comprime el rango dinámico)
        bytes = 255 · (L - Lmin) / (Lmax - Lmin)

    Devuelve (bytes, Lmin, Lmax): con Lmin y Lmax se puede volver
    del bitmap a la amplitud (ver bitmap_a_amplitud).
    """

    l = np.log1p(amplitud) if log else np.array(amplitud, dtype=np.float64)

    l_min = float(l.min())
    l_max = float(l.max())

    rango = (l_max - l_min) or 1.0

    normalizada = (l - l_min) / rango

    return (normalizada * 255.0 + 0.5).astype(np.uint8), l_min, l_max


def bitmap_a_amplitud(
    bitmap: np.ndarray,
    l_min: float,
    l_max: float,
    log: bool = True
) -> np.ndarray:
    """
    Camino inverso de amplitud_a_bitmap:

        L = bytes / 255 · (Lmax - Lmin) + Lmin
        A = e^L - 1        (si se guardó en escala logarítmica)

    Si el bitmap tiene 3 canales (el editor lo guardó en color),
    se usa el promedio de los tres.
    """

    if bitmap.ndim == 3:
        bitmap = (bitmap[:, :, 0].astype(np.float64)
                  + bitmap[:, :, 1] + bitmap[:, :, 2]) / 3

    l = bitmap.astype(np.float64) / 255.0 * (l_max - l_min) + l_min

    if log:
        return np.expm1(l)

    return l


def amplitud_editada(
    amplitud_original: np.ndarray,
    bitmap_editado: np.ndarray,
    l_min: float,
    l_max: float,
    log: bool = True
) -> np.ndarray:
    """
    Amplitud a usar en la reconstrucción a partir del bitmap editado.

    El bitmap tiene solo 256 niveles: aunque no se edite nada, volver
    de él a la amplitud no da exactamente la original (se pierde
    precisión). Por eso, en cada píxel:

        - si el byte del bitmap es igual al que se guardó, ese píxel no
          se editó y se usa la amplitud original exacta;
        - si cambió, se usa la amplitud que sale del bitmap editado.

    Así la reconstrucción muestra solo el efecto de lo que se editó.
    """

    if bitmap_editado.ndim == 3:
        bitmap_editado = ((bitmap_editado[:, :, 0].astype(np.float64)
                           + bitmap_editado[:, :, 1] + bitmap_editado[:, :, 2]) / 3 + 0.5).astype(np.uint8)

    bitmap_original, _, _ = amplitud_a_bitmap(amplitud_original, log)

    desde_bitmap = bitmap_a_amplitud(bitmap_editado, l_min, l_max, log)

    alto, ancho = amplitud_original.shape

    resultado = np.copy(amplitud_original)

    for u in range(alto):

        for v in range(ancho):

            if bitmap_editado[u, v] != bitmap_original[u, v]:
                resultado[u, v] = desde_bitmap[u, v]

    return resultado


def fase_a_bitmap(fase: np.ndarray) -> np.ndarray:
    """Fase (-π..π) -> bytes (0..255), para verla como imagen gris."""

    return ((fase + math.pi) / (2 * math.pi) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# WORKFLOW COMPLETO
# ---------------------------------------------------------------------------
def transformada_directa(
    rgb: np.ndarray,
    restar_media: bool = False,
    centrar: bool = True,
    metodo: str = "matricial",
    norm: str = "numpy",
    progreso=None
) -> dict:
    """
    Pasos 1 a 3 del workflow sobre una imagen RGB normalizada.

    Devuelve un diccionario con todo lo necesario para la inversa:

        "F"      espectro complejo (centrado si centrar=True)
        "media"  lo que se le restó a Y (0 si restar_media=False)
        "I", "Q" cromaticidad original (se reusa al reconstruir)
        "centrado", "restar_media", "metodo", "norm"
    """

    yiq = rgb_a_yiq(rgb)

    y = yiq[:, :, 0]

    media = 0.0

    if restar_media:
        # sin la media, el pico de F(0, 0) no tapa al resto del espectro
        media = float(y.sum() / y.size)
        y = y - media

    f = dft2(y, metodo, norm, progreso)

    if centrar:
        f = centrar_espectro(f)

    return {
        "F": f,
        "media": media,
        "I": yiq[:, :, 1].copy(),
        "Q": yiq[:, :, 2].copy(),
        "centrado": centrar,
        "restar_media": restar_media,
        "metodo": metodo,
        "norm": norm,
    }


def transformada_inversa(
    datos: dict,
    f: np.ndarray | None = None,
    progreso=None
) -> np.ndarray:
    """
    Paso 5: TF inversa de la luminancia y vuelta a RGB, con el mismo
    método y la misma normalización que se usaron en la directa.

    Si se pasa f, se usa ese espectro (por ejemplo, el armado con una
    amplitud editada); si no, el de datos["F"].
    """

    if f is None:
        f = datos["F"]

    if datos["centrado"]:
        f = descentrar_espectro(f)

    y = idft2(f, datos["metodo"], datos["norm"], progreso) + datos["media"]

    alto, ancho = y.shape

    yiq = np.zeros((alto, ancho, 3))

    yiq[:, :, 0] = y
    yiq[:, :, 1] = datos["I"]
    yiq[:, :, 2] = datos["Q"]

    return yiq_a_rgb(yiq)
