"""
operaciones.py — TP1: Espacios cromáticos

"""

import numpy as np
from PySide6.QtGui import QImage

# MATRICES DE CONVERSIÓN
# RGB -> YIQ   (rgbToYIQ)
#   Y = 0.299    R + 0.587    G + 0.114    B
#   I = 0.595716 R - 0.274453 G - 0.321263 B
#   Q = 0.211456 R - 0.522591 G + 0.311135 B
M_RGB_A_YIQ = np.array([
    [0.299,     0.587,     0.114],
    [0.595716, -0.274453, -0.321263],
    [0.211456, -0.522591,  0.311135],
])

# YIQ -> RGB   (yiqToRGB)
#   R = Y + 0.9563 I + 0.6210 Q
#   G = Y - 0.2721 I - 0.6474 Q
#   B = Y - 1.1070 I + 1.7046 Q
M_YIQ_A_RGB = np.array([
    [1.0,  0.9563,  0.6210],
    [1.0, -0.2721, -0.6474],
    [1.0, -1.1070,  1.7046],
])

# Límites de los pasos 5 y 6 (adjust_luminance_saturation)
Y_MAX = 1.0
I_MAX = 0.5957
Q_MAX = 0.5226



# ARCHIVOS
def abrir_imagen(ruta: str) -> QImage:
    """Abre la imagen del archivo. Devuelve None si no se pudo abrir."""
    imagen = QImage(ruta)
    if imagen.isNull():
        return None
    return imagen


def guardar_imagen(imagen: QImage, ruta: str) -> bool:
    """Guarda la imagen; el formato sale de la extensión de la ruta (.png, .jpg, .bmp...)."""
    return imagen.save(ruta)


# PASO 1 y PASO 8 DEL WORKFLOW
def normalizar(img: np.ndarray) -> np.ndarray:
   
    return img.astype(np.float32) / 255.0


def a_bytes(rgb: np.ndarray) -> np.ndarray:
    """
    """
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# PESTAÑA "CANALES RGB"
def extraer_canal(rgb: np.ndarray, indice: int) -> np.ndarray:
   
    return rgb[:, :, indice]


def intercambiar_canales(rgb: np.ndarray) -> np.ndarray:

    nueva = np.copy(rgb)
    
    heigth, width, channels = nueva.shape
    
    for y in range(heigth):
    
        for x in range(width):
        
            r, g, b = nueva[y, x]
            
            change_r = b
            
            change_g = r
            
            change_b = g    
            
            nueva[y, x] = [change_r, change_g, change_b]
    
    return nueva


# PESTAÑA "LUMINANCIA / SATURACIÓN (YIQ)"  (punto 3 del TP)
def rgb_a_yiq(rgb: np.ndarray) -> np.ndarray:
    
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


def ajustar_yiq(yiq: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    """
    nueva = np.copy(yiq)
    
    height, width, _ = nueva.shape
    
    for y in range(height):
    
        for x in range(width):
        
            y_val, i_val, q_val = nueva[y, x]

            # Aplicar coeficientes
            y_val *= a
            i_val *= b
            q_val *= b

            # Clipping de los valores
            y_val = min(y_val, 1.0)
            i_val = np.clip(i_val, -I_MAX, I_MAX)
            q_val = np.clip(q_val, -Q_MAX, Q_MAX)

            nueva[y, x] = [y_val, i_val, q_val]

    return nueva


def yiq_a_rgb(yiq: np.ndarray) -> np.ndarray:
    """
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

    return np.clip(rgb_img, 0, 1)


# PESTAÑA "MAPAS CROMÁTICOS"
# Las mismas paletas que en el notebook: cmaps = ["gray", "rainbow", "viridis"]
PALETAS = ["gris", "arcoiris", "viridis"]

# Nombre en la interfaz -> nombre del colormap de matplotlib (como en el notebook)
_CMAPS_MATPLOTLIB = {
    "gris": "gray", 
    "arcoiris": "rainbow", 
    "viridis": "viridis"
 }

# Respaldo por si matplotlib no está instalado: colores de control + interpolación lineal
_CONTROL = {
    "gris":     [(0.0, (0, 0, 0)), (1.0, (1, 1, 1))],
    "arcoiris": [(0.0, (0.5, 0, 1)), (0.2, (0, 0.3, 1)), (0.4, (0, 1, 1)),
                 (0.6, (0.5, 1, 0.3)), (0.8, (1, 0.7, 0)), (1.0, (1, 0, 0))],
    "viridis":  [(0.0, (0.267, 0.005, 0.329)), (0.25, (0.229, 0.322, 0.546)),
                 (0.5, (0.128, 0.567, 0.551)), (0.75, (0.369, 0.789, 0.383)),
                 (1.0, (0.993, 0.906, 0.144))],
}


def luminancia(rgb: np.ndarray) -> np.ndarray:
    """
    Dato 2D a partir de una imagen a color = canal Y de YIQ.
    """
    return np.clip(rgb_a_yiq(rgb)[:, :, 0], 0.0, 1.0)


def aplicar_paleta(datos: np.ndarray, paleta: str) -> np.ndarray:
    """
    Dato 2D normalizado a 0..1 -> imagen RGB normalizada (alto, ancho, 3).
    """
    datos = np.clip(np.asarray(datos, dtype=np.float64), 0.0, 1.0)

    try:
        import matplotlib
        cmap = matplotlib.colormaps[_CMAPS_MATPLOTLIB[paleta]]
        return cmap(datos)[:, :, :3]              # cmap devuelve RGBA; se descarta A
    except ImportError:
        pass

    # Sin matplotlib: interpolación lineal entre colores de control, canal por canal
    puntos = _CONTROL[paleta]
    xs = [p for p, _ in puntos]
    salida = np.empty(datos.shape + (3,))
    for c in range(3):
        salida[:, :, c] = np.interp(datos, xs, [color[c] for _, color in puntos])
    return salida
