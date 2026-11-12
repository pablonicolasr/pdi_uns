"""
app.py — TP3 Operaciones de luminancia: interfaz gráfica (PySide6).

Ejecutar con:   python app.py
Requiere:       pip install PySide6 numpy

Las operaciones se implementan en operaciones.py; este archivo solo
arma la ventana, muestra las imágenes y dibuja los histogramas.
"""

import os
import sys
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QScrollArea, QPushButton,
    QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QFileDialog,
    QMessageBox, QSlider, QSpinBox, QDoubleSpinBox, QSplitter, QRadioButton,
    QButtonGroup, QCheckBox, QStackedWidget,
)

import operaciones as ops

FILTRO_IMAGENES = "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif *.ppm *.pgm *.tif *.tiff);;Todos (*)"


class FaltaImplementar(Exception):
    pass


# ---------------------------------------------------------------------------
# Conversión QImage <-> numpy (plomería, no es parte del TP)
# ---------------------------------------------------------------------------
def qimage_a_array(img: QImage) -> np.ndarray:
    """QImage -> ndarray uint8 (alto, ancho, 3) en RGB."""
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
    return buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3).copy()


def array_a_qimage(arr: np.ndarray) -> QImage:
    """ndarray uint8 (alto, ancho, 3) -> QImage (para mostrar o guardar)."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def validar(arr, nombre, forma=None, dtype_uint8=False):
    """Controla que una función de operaciones.py devolvió una imagen (alto, ancho, 3)."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{nombre}() debe devolver un np.ndarray (devolvió {type(arr).__name__}).")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{nombre}() devolvió forma {arr.shape}; se esperaba (alto, ancho, 3).")
    if forma is not None and arr.shape[:2] != forma[:2]:
        raise ValueError(f"{nombre}() devolvió tamaño {arr.shape[:2]}; se esperaba {forma[:2]}.")
    if dtype_uint8 and arr.dtype != np.uint8:
        raise TypeError(f"{nombre}() debe devolver dtype uint8 (devolvió {arr.dtype}).")
    return arr


def llamar(nombre, *args):
    """Llama a operaciones.<nombre>, traduciendo NotImplementedError."""
    try:
        return getattr(ops, nombre)(*args)
    except NotImplementedError:
        raise FaltaImplementar(nombre)


# ---------------------------------------------------------------------------
# Dibujo del histograma (a mano, sobre un arreglo de píxeles)
# ---------------------------------------------------------------------------
GRIS = np.array([150, 150, 150], np.uint8)
AZUL = np.array([30, 110, 220], np.uint8)
FONDO = np.array([255, 255, 255], np.uint8)


def dibujar_histogramas(hist_original, hist_resultado=None, ancho=320, alto=110):
    """
    Dibuja uno o dos histogramas superpuestos en una imagen (alto, ancho, 3):
    el original en gris y el resultado en azul (mezclado al 60 %).

    Cada nivel k ocupa la columna x = k · ancho / niveles, con una barra
    de altura proporcional a la raíz de su cantidad de píxeles: con la
    raíz, un nivel muy poblado (por ejemplo, el blanco saturado) no
    "aplasta" al resto del histograma.
    """
    lienzo = np.zeros((alto, ancho, 3), np.uint8)
    lienzo[:, :] = FONDO
    histos = [h for h in (hist_original, hist_resultado) if h is not None]
    maximo = max(float(np.sqrt(h.max())) for h in histos) or 1.0
    niveles = len(hist_original)

    for indice, (hist, color) in enumerate(zip(histos, (GRIS, AZUL))):
        for x in range(ancho):
            k = x * niveles // ancho
            barra = int(round(np.sqrt(hist[k]) / maximo * (alto - 1)))
            for y in range(alto - barra, alto):
                if indice == 0:
                    lienzo[y, x] = color
                else:
                    lienzo[y, x] = (0.6 * color + 0.4 * lienzo[y, x]).astype(np.uint8)
    return lienzo


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------
class LienzoImagen(QLabel):
    """Muestra una imagen con zoom y avisa qué píxel está bajo el mouse."""
    pixel_hover = Signal(int, int)

    def __init__(self, texto):
        super().__init__()
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)
        self._dims, self._zoom, self._texto = None, 1.0, texto
        self.mostrar(None, 1.0)

    def mostrar(self, arr, zoom):
        self._zoom = zoom
        if arr is None:
            self._dims = None
            self.clear()
            self.setText(self._texto)
            self.setStyleSheet("color: #999; padding: 12px;")
            self.adjustSize()
            return
        self.setStyleSheet("")
        h, w = arr.shape[:2]
        self._dims = (w, h)
        pix = QPixmap.fromImage(array_a_qimage(arr))
        if zoom != 1.0:
            pix = pix.scaled(max(1, int(w * zoom)), max(1, int(h * zoom)),
                             Qt.IgnoreAspectRatio,
                             Qt.FastTransformation if zoom > 1 else Qt.SmoothTransformation)
        self.setPixmap(pix)
        self.resize(pix.size())

    def mouseMoveEvent(self, e):
        if not self._dims:
            return
        p: QPoint = e.position().toPoint()
        x, y = int(p.x() / self._zoom), int(p.y() / self._zoom)
        if 0 <= x < self._dims[0] and 0 <= y < self._dims[1]:
            self.pixel_hover.emit(x, y)


class Panel(QWidget):
    """Título + lienzo con scroll."""
    def __init__(self, titulo, texto_vacio):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.titulo = QLabel(f"<b>{titulo}</b>")
        v.addWidget(self.titulo)
        self.lienzo = LienzoImagen(texto_vacio)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.lienzo)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea { background: #2b2b2b; }")
        v.addWidget(self.scroll, 1)


def spin(minimo, maximo, valor, paso=0.01, decimales=3):
    """QDoubleSpinBox ya configurado."""
    return QDoubleSpinBox(minimum=minimo, maximum=maximo, value=valor,
                          singleStep=paso, decimals=decimales)


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP3 — Operaciones de luminancia")
        self.resize(1400, 860)

        self.original = None      # uint8 (h, w, 3) — imagen de trabajo
        self.inicial = None       # uint8 — tal como se abrió (para Restaurar)
        self.resultado = None     # uint8 (h, w, 3)
        self.ruta = None
        self.yiq = None           # YIQ de la original (se calcula una vez)
        self.yiq_res = None       # Y'IQ del resultado
        self.hist = None          # histograma de Y de la original
        self.hist_res = None      # histograma de Y' del resultado
        self.stats = self.stats_res = ""   # textos con mín, máx y media de Y

        self._crear_acciones()
        self._crear_ui()
        self._actualizar_controles()

    # ------------------------------------------------------------ interfaz
    def _crear_acciones(self):
        A = lambda t, s, f: QAction(t, self, shortcut=s, triggered=f)
        self.act_abrir = A("Abrir imagen...", QKeySequence.Open, self.abrir)
        self.act_guardar = A("Guardar resultado como...", QKeySequence.Save, self.guardar)
        self.act_usar = A("Usar resultado como original", QKeySequence("Ctrl+U"), self.usar_resultado)
        self.act_restaurar = A("Restaurar imagen inicial", QKeySequence("Ctrl+R"), self.restaurar)
        self.act_salir = A("Salir", QKeySequence.Quit, self.close)

        m = self.menuBar().addMenu("&Archivo")
        m.addAction(self.act_abrir)
        m.addAction(self.act_guardar)
        m.addSeparator()
        m.addAction(self.act_salir)
        e = self.menuBar().addMenu("&Editar")
        e.addAction(self.act_usar)
        e.addAction(self.act_restaurar)

        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        for a in (self.act_abrir, self.act_guardar, self.act_usar, self.act_restaurar):
            tb.addAction(a)

    def _crear_ui(self):
        # --- paneles: original | resultado ---
        self.p_orig = Panel("Original", "Abrí una imagen para empezar")
        self.p_res = Panel("Resultado", "Elegí un método y tocá «Aplicar»")
        for p in (self.p_orig, self.p_res):
            p.lienzo.pixel_hover.connect(self._hover)
        for a, b in ((self.p_orig, self.p_res), (self.p_res, self.p_orig)):
            a.scroll.horizontalScrollBar().valueChanged.connect(b.scroll.horizontalScrollBar().setValue)
            a.scroll.verticalScrollBar().valueChanged.connect(b.scroll.verticalScrollBar().setValue)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.p_orig)
        split.addWidget(self.p_res)

        # --- panel lateral ---
        lateral = QWidget()
        lateral.setFixedWidth(360)
        lv = QVBoxLayout(lateral)

        g_info = QGroupBox("Imagen")
        fi = QFormLayout(g_info)
        self.lbl_archivo, self.lbl_tamano = QLabel("—"), QLabel("—")
        self.lbl_archivo.setWordWrap(True)
        fi.addRow("Archivo:", self.lbl_archivo)
        fi.addRow("Tamaño:", self.lbl_tamano)
        self.sp_ancho = QSpinBox(minimum=0, maximum=10000, value=800, singleStep=100, suffix=" px")
        self.sp_ancho.setSpecialValueText("sin reducir")
        self.sp_ancho.setToolTip("Las imágenes más anchas se reducen al abrirlas (como read_img del notebook).")
        fi.addRow("Ancho máx. al abrir:", self.sp_ancho)
        lv.addWidget(g_info)

        g_zoom = QGroupBox("Zoom")
        hz = QHBoxLayout(g_zoom)
        self.sl_zoom = QSlider(Qt.Horizontal, minimum=10, maximum=800, value=100)
        self.lbl_zoom = QLabel("100 %")
        self.lbl_zoom.setMinimumWidth(50)
        self.sl_zoom.valueChanged.connect(self._cambio_zoom)
        btn_ajustar = QPushButton("Ajustar")
        btn_ajustar.clicked.connect(self._zoom_ajustar)
        hz.addWidget(self.sl_zoom, 1)
        hz.addWidget(self.lbl_zoom)
        hz.addWidget(btn_ajustar)
        lv.addWidget(g_zoom)

        lv.addWidget(self._grupo_metodo())

        g_hist = QGroupBox("Histograma de Y, escala √  (gris: original · azul: resultado)")
        vh = QVBoxLayout(g_hist)
        self.lbl_hist = QLabel()
        self.lbl_hist.setFixedSize(320, 110)
        self.lbl_hist.setStyleSheet("border: 1px solid #ccc;")
        vh.addWidget(self.lbl_hist, 0, Qt.AlignHCenter)
        self.lbl_stats = QLabel("—")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.lbl_stats.setMinimumHeight(34)
        vh.addWidget(self.lbl_stats)
        lv.addWidget(g_hist)

        g_pix = QGroupBox("Píxel bajo el mouse")
        self.lbl_pixel = QLabel("—")
        self.lbl_pixel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_pixel.setStyleSheet("font-family: monospace;")
        QVBoxLayout(g_pix).addWidget(self.lbl_pixel)
        lv.addWidget(g_pix)
        lv.addStretch(1)

        central = QWidget()
        hl = QHBoxLayout(central)
        hl.addWidget(split, 1)
        hl.addWidget(lateral)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")

    def _grupo_metodo(self):
        """Radio buttons de los métodos y, debajo, los parámetros del elegido."""
        g = QGroupBox("Y' = f(Y)")
        v = QVBoxLayout(g)
        self.grupo_metodos = QButtonGroup(g)
        self.radios = {}
        for i, (clave, (nombre, descripcion)) in enumerate(ops.METODOS.items()):
            rb = QRadioButton(nombre)
            rb.setToolTip(descripcion)
            self.grupo_metodos.addButton(rb, i)
            self.radios[clave] = rb
            v.addWidget(rb)
        self.radios["raiz"].setChecked(True)

        # parámetros: una página por método (vacía si no tiene)
        self.pila = QStackedWidget()
        for clave in ops.METODOS:
            pagina = QWidget()
            f = QFormLayout(pagina)
            f.setContentsMargins(0, 4, 0, 0)
            if clave == "lineal":
                self.sp_ymin = spin(0, 1, 0.0)
                self.sp_ymax = spin(0, 1, 1.0)
                f.addRow("Ymin:", self.sp_ymin)
                f.addRow("Ymax:", self.sp_ymax)
                b = QPushButton("Usar mín. y máx. de la imagen")
                b.clicked.connect(self._usar_min_max)
                f.addRow(b)
            elif clave == "sigmoide":
                self.sp_k = spin(0.1, 100, 12.0, 0.5, 2)
                self.sp_x0 = spin(0, 1, 0.5)
                self.chk_media = QCheckBox("x0 = media de Y")
                self.chk_media.toggled.connect(lambda c: self.sp_x0.setEnabled(not c))
                f.addRow("k (pendiente):", self.sp_k)
                f.addRow("x0 (centro):", self.sp_x0)
                f.addRow(self.chk_media)
            else:
                f.addRow(QLabel(ops.METODOS[clave][1]))
            self.pila.addWidget(pagina)
        self.grupo_metodos.idToggled.connect(lambda i, on: on and self.pila.setCurrentIndex(i))
        v.addWidget(self.pila)

        self.btn_aplicar = QPushButton("Aplicar")
        self.btn_aplicar.setStyleSheet("font-weight: bold;")
        self.btn_aplicar.clicked.connect(self.aplicar)
        v.addWidget(self.btn_aplicar)
        return g

    # ------------------------------------------------------------ utilidades
    def _falta(self, nombre):
        QMessageBox.information(self, "Falta implementar",
                                f"La función <b>{nombre}()</b> de <i>operaciones.py</i> todavía no está implementada.")

    def _error(self, e):
        QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

    def _ejecutar(self, f):
        """Corre f() con cursor de espera, mide el tiempo y muestra los errores."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        t0 = time.perf_counter()
        try:
            f()
        except FaltaImplementar as e:
            QApplication.restoreOverrideCursor()
            self._falta(str(e))
            return False
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._error(e)
            return False
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Procesado en {time.perf_counter() - t0:.2f} s")
        return True

    def _actualizar_controles(self):
        hay = self.original is not None
        self.btn_aplicar.setEnabled(hay)
        self.act_restaurar.setEnabled(hay)
        self.act_guardar.setEnabled(self.resultado is not None)
        self.act_usar.setEnabled(self.resultado is not None)
        if hay:
            h, w = self.original.shape[:2]
            self.lbl_tamano.setText(f"{w} × {h} px")
            self.lbl_archivo.setText(os.path.basename(self.ruta) if self.ruta else "(resultado)")
        titulo = "TP3 — Operaciones de luminancia"
        self.setWindowTitle(titulo + (f" — {os.path.basename(self.ruta)}" if self.ruta else ""))

    def _redibujar(self):
        z = self.sl_zoom.value() / 100
        self.p_orig.lienzo.mostrar(self.original, z)
        self.p_res.lienzo.mostrar(self.resultado, z)

    def _cambio_zoom(self, val):
        self.lbl_zoom.setText(f"{val} %")
        self._redibujar()

    def _zoom_ajustar(self):
        if self.original is None:
            return
        h, w = self.original.shape[:2]
        vp = self.p_orig.scroll.viewport().size()
        z = min(vp.width() / w, vp.height() / h) * 100
        self.sl_zoom.setValue(int(max(10, min(800, z))))

    def _set_resultado(self, arr, titulo=""):
        self.resultado = arr
        self.p_res.titulo.setText(f"<b>Resultado</b>{' — ' + titulo if titulo else ''}")
        self._redibujar()
        self._actualizar_controles()
        self._mostrar_histograma()

    def _mostrar_histograma(self):
        if self.hist is None:
            self.lbl_hist.clear()
            self.lbl_stats.setText("—")
            return
        img = dibujar_histogramas(self.hist, self.hist_res)
        self.lbl_hist.setPixmap(QPixmap.fromImage(array_a_qimage(img)))
        lineas = ["Original:  " + self.stats]
        if self.yiq_res is not None and self.resultado is not None:
            lineas.append("Resultado: " + self.stats_res)
        self.lbl_stats.setText("\n".join(lineas))

    @staticmethod
    def _stats(yiq):
        """Texto con mínimo, máximo y media de Y (se calcula una vez por imagen)."""
        y_min, y_max = llamar("minimo_maximo_y", yiq)
        return f"mín {y_min:.2f} · máx {y_max:.2f} · media {llamar('media_y', yiq):.2f}"

    def _hover(self, x, y):
        lineas = [f"(x, y) = ({x}, {y})"]
        if self.original is not None and y < self.original.shape[0] and x < self.original.shape[1]:
            lineas.append("Original:  RGB(%3d, %3d, %3d)" % tuple(self.original[y, x]))
            lineas.append("           YIQ(%.3f, %+.3f, %+.3f)" % tuple(self.yiq[y, x]))
        if self.resultado is not None and y < self.resultado.shape[0] and x < self.resultado.shape[1]:
            lineas.append("Resultado: RGB(%3d, %3d, %3d)" % tuple(self.resultado[y, x]))
            if self.yiq_res is not None:
                lineas.append("           Y'IQ(%.3f, %+.3f, %+.3f)" % tuple(self.yiq_res[y, x]))
        self.lbl_pixel.setText("\n".join(lineas))

    def _metodo(self):
        for clave, rb in self.radios.items():
            if rb.isChecked():
                return clave

    def _usar_min_max(self):
        if self.yiq is None:
            return
        y_min, y_max = llamar("minimo_maximo_y", self.yiq)
        self.sp_ymin.setValue(y_min)
        self.sp_ymax.setValue(y_max)

    # ------------------------------------------------------------ archivo
    def abrir(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", FILTRO_IMAGENES)
        if ruta:
            self.cargar(ruta)

    def cargar(self, ruta):
        """Abre el archivo, lo reduce si es muy ancho y calcula su YIQ e histograma."""
        def f():
            img = llamar("abrir_imagen", ruta)
            if not isinstance(img, QImage) or img.isNull():
                raise ValueError("No se pudo abrir la imagen.")
            arr = validar(llamar("reducir", qimage_a_array(img), self.sp_ancho.value()), "reducir")
            self.original, self.inicial, self.ruta = arr, arr.copy(), ruta
            self._preparar_original()
        if self._ejecutar(f):
            self._zoom_ajustar()
            self._usar_min_max()

    def _preparar_original(self):
        """Pasos 1 y 2 sobre la original (se hacen una sola vez) + histograma."""
        rgb = validar(llamar("normalizar", self.original), "normalizar", self.original.shape)
        self.yiq = validar(llamar("rgb_a_yiq", rgb), "rgb_a_yiq", self.original.shape)
        self.hist = llamar("histograma_y", self.yiq)
        self.stats = self._stats(self.yiq)
        self.yiq_res = self.hist_res = None
        self._set_resultado(None)

    def guardar(self):
        if self.resultado is None:
            return
        base = os.path.splitext(self.ruta or "resultado")[0] + f"_{self._metodo()}.png"
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar resultado", base,
                                              "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;Todos (*)")
        if not ruta:
            return
        try:
            ok = llamar("guardar_imagen", array_a_qimage(self.resultado), ruta)
        except FaltaImplementar as e:
            return self._falta(str(e))
        except Exception as e:
            return self._error(e)
        if ok:
            self.statusBar().showMessage(f"Guardado: {ruta}")
        else:
            QMessageBox.warning(self, "Guardar", "No se pudo guardar la imagen.")

    def usar_resultado(self):
        """El resultado pasa a ser la original: permite encadenar métodos."""
        if self.resultado is not None:
            self.original = self.resultado.copy()
            self._ejecutar(self._preparar_original)
            self.statusBar().showMessage("El resultado ahora es la imagen original")

    def restaurar(self):
        if self.inicial is not None:
            self.original = self.inicial.copy()
            self._ejecutar(self._preparar_original)
            self.statusBar().showMessage("Imagen inicial restaurada")

    # ------------------------------------------------------------ workflow
    def aplicar(self):
        if self.yiq is None:
            return
        metodo = self._metodo()

        def f():
            sh = self.original.shape
            if metodo == "raiz":
                yiq2, titulo = llamar("aplicar_raiz", self.yiq), "Y' = √Y"
            elif metodo == "cuadrado":
                yiq2, titulo = llamar("aplicar_cuadrado", self.yiq), "Y' = Y²"
            elif metodo == "lineal":
                y_min, y_max = self.sp_ymin.value(), self.sp_ymax.value()
                yiq2 = llamar("aplicar_lineal_a_trozos", self.yiq, y_min, y_max)
                titulo = f"lineal a trozos [{y_min:.3f}, {y_max:.3f}]"
            elif metodo == "sigmoide":
                k = self.sp_k.value()
                x0 = None if self.chk_media.isChecked() else self.sp_x0.value()
                yiq2 = llamar("aplicar_sigmoide", self.yiq, k, x0)
                titulo = f"sigmoide k = {k:g}, x0 = {'media' if x0 is None else f'{x0:g}'}"
            else:
                yiq2, titulo = llamar("aplicar_ecualizacion", self.yiq), "ecualización"
            yiq2 = validar(yiq2, metodo, sh)                                         # paso 3
            rgb2 = validar(llamar("yiq_a_rgb", yiq2), "yiq_a_rgb", sh)               # paso 4
            out = validar(llamar("a_bytes", rgb2), "a_bytes", sh, dtype_uint8=True)  # paso 5
            self.yiq_res = yiq2
            self.hist_res = llamar("histograma_y", yiq2)
            self.stats_res = self._stats(yiq2)
            self._set_resultado(out, titulo)
        self._ejecutar(f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = VentanaPrincipal()
    v.show()
    sys.exit(app.exec())
