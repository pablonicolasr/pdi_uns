"""
operaciones.py

Operaciones básicas de Procesamiento Digital de Imágenes.

La imagen se maneja como un QImage de PySide6.

El objetivo de este módulo es trabajar directamente con los píxeles
de una imagen para comprender los fundamentos del procesamiento
digital de imágenes.
"""

from PySide6.QtGui import QImage, QColor

import numpy as np


# 1) ABRIR UNA IMAGEN DESDE ARCHIVO
#    -> botón "Abrir imagen..."

def abrir_imagen(ruta: str) -> QImage | None:
    """
    Recibe la ruta del archivo elegido por el usuario.

    Devuelve:
        QImage:
            Imagen cargada y convertida al formato RGB32.

        None:
            Si la imagen no pudo ser cargada.

    La conversión a QImage.Format_RGB32 permite trabajar con
    una representación homogénea de los píxeles,
    independientemente del formato original del archivo.
    """

    imagen = QImage(ruta)

    if imagen.isNull():
        return None

    imagen = imagen.convertToFormat(
        QImage.Format.Format_RGB32
    )

    return imagen


# 2) GUARDAR LA IMAGEN A UN ARCHIVO
#    -> botones "Guardar" / "Guardar como..."

def guardar_imagen(
    imagen: QImage,
    ruta: str
) -> bool:
    """
    Recibe una imagen y una ruta destino.

    La extensión de la ruta determina normalmente
    el formato utilizado:

        .png
        .jpg
        .jpeg
        .bmp

    Devuelve:
        True:
            Si la imagen pudo guardarse.

        False:
            Si ocurrió un error.
    """

    if imagen is None:
        return False

    if imagen.isNull():
        return False

    return imagen.save(ruta)


def qimage_a_numpy(imagen: QImage) -> np.ndarray:
    """
    Convierte un QImage en un ndarray de NumPy
    con forma:

        (alto, ancho, 3)

    y canales RGB.
    """

    imagen = imagen.convertToFormat(QImage.Format.Format_RGB888)

    ancho = imagen.width()
    alto = imagen.height()

    ptr = imagen.bits()

    array = np.frombuffer(
        ptr,
        dtype=np.uint8
    )

    array = array.reshape(
        alto,
        imagen.bytesPerLine()
    )

    array = array[:, :ancho * 3]

    array = array.reshape(
        alto,
        ancho,
        3
    )

    return array.copy()


def numpy_a_qimage(image: np.ndarray) -> QImage:

    alto, ancho, canales = image.shape

    bytes_por_linea = canales * ancho

    qimage = QImage(
        image.data,
        ancho,
        alto,
        bytes_por_linea,
        QImage.Format.Format_RGB888
    )

    return qimage.copy()


# 3a) LEER EL COLOR DE UN PIXEL
#     -> botón "Leer pixel"

def leer_pixel(
    imagen: QImage,
    x: int,
    y: int
) -> tuple[int, int, int]:
    """
    Devuelve el color del píxel ubicado en (x, y).

    El resultado es una tupla:

        (R, G, B)

    donde cada componente pertenece al intervalo:

        0 <= componente <= 255

    Ejemplo:

        (255, 0, 0)

    representa rojo puro.
    """
    
    arr = qimage_a_numpy(imagen)

    return tuple(map(int, arr[y, x, :3]))

# 3b) MODIFICAR EL COLOR DE UN PIXEL
#     -> botón "Modificar pixel"

def modificar_pixel(
    imagen: QImage,
    x: int,
    y: int,
    color: tuple[int, int, int]
) -> None:
    """
    Modifica directamente el píxel ubicado en (x, y).

    color debe ser una tupla:

        (R, G, B)

    Ejemplo:

        modificar_pixel(
            imagen,
            100,
            50,
            (255, 0, 0)
        )

    pinta de rojo el píxel (100, 50).
    """

    image = qimage_a_numpy(imagen)

    image[y, x] = color
    
    nueva_imagen = numpy_a_qimage(image)
    
    imagen.swap(nueva_imagen)


# 4) OPERACIÓN SOBRE TODA LA IMAGEN
#    -> botón "Aplicar operación"

def procesar_imagen(
    imagen: QImage
) -> QImage:
    """
    Se invierten los canales de los pixeles.

    Ejemplo:

        RGB original:
            (100, 50, 200)

        RGB negativo:
            (200, 100, 50)

    Se trabaja sobre una copia para conservar
    intacta la imagen original.
    """

    resultado = imagen.copy()
    
    resultado = qimage_a_numpy(resultado)

    heigth, width, channels = resultado.shape
    
    for y in range(heigth):
    
        for x in range(width):
        
            r, g, b = resultado[y, x]
            
            change_r = b
            
            change_g = r
            
            change_b = g    
            
            resultado[y, x] = [change_r, change_g, change_b] 

    return numpy_a_qimage(resultado)


def invertir_imagen(
    imagen: QImage
) -> QImage:
    """
    Se invierten los canales de los pixeles.

    Ejemplo:

        RGB original:
            (100, 50, 200)

        RGB negativo:
            (200, 100, 50)

    Se trabaja sobre una copia para conservar
    intacta la imagen original.
    """

    resultado = imagen.copy()
    
    resultado = qimage_a_numpy(resultado)

    heigth, width, channels = resultado.shape
    
    for y in range(heigth):
    
        for x in range(width):
        
            r, g, b = resultado[y, x]
            
            change_r = 255 - r
            
            change_g = 255 - g
            
            change_b = 255 - b   
            
            resultado[y, x] = [change_r, change_g, change_b] 

    return numpy_a_qimage(resultado)
