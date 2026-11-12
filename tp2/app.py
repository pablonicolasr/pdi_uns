"""
app.py — TP2 Aritmética de píxeles: interfaz gráfica (PySide6).

Ejecutar con:   python app.py
Requiere:       pip install PySide6 numpy

Las operaciones se implementan en operaciones.py; este archivo solo
arma la ventana, abre las dos imágenes y muestra el resultado.
"""

import os
import sys
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QScrollArea, QPushButton,
    QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QMessageBox, QSlider, QSpinBox, QSplitter,
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


def luminancia_byte(rgb) -> float:
    """Y de un píxel en bytes (solo para mostrarla al pasar el mouse)."""
    r, g, b = (float(c) / 255 for c in rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b


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


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP2 — Aritmética de píxeles")
        self.resize(1500, 820)

        self.img_a = None        # uint8 (h, w, 3) — imagen A
        self.img_b = None        # uint8 (h, w, 3) — imagen B
        self.inicial_a = None    # A tal como se abrió (para Restaurar)
        self.ruta_a = None
        self.ruta_b = None
        self.resultado = None    # uint8 (h, w, 3)

        self._crear_acciones()
        self._crear_ui()
        self._actualizar_controles()

    # ------------------------------------------------------------ interfaz
    def _crear_acciones(self):
        A = lambda t, s, f: QAction(t, self, shortcut=s, triggered=f)
        self.act_abrir_a = A("Abrir imagen A...", QKeySequence.Open, lambda: self.abrir("A"))
        self.act_abrir_b = A("Abrir imagen B...", QKeySequence("Ctrl+Shift+O"), lambda: self.abrir("B"))
        self.act_guardar = A("Guardar resultado como...", QKeySequence.Save, self.guardar)
        self.act_usar = A("Usar resultado como A", QKeySequence("Ctrl+U"), self.usar_resultado)
        self.act_restaurar = A("Restaurar A inicial", QKeySequence("Ctrl+R"), self.restaurar)
        self.act_salir = A("Salir", QKeySequence.Quit, self.close)

        m = self.menuBar().addMenu("&Archivo")
        for a in (self.act_abrir_a, self.act_abrir_b, self.act_guardar):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_salir)
        e = self.menuBar().addMenu("&Editar")
        e.addAction(self.act_usar)
        e.addAction(self.act_restaurar)

        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        for a in (self.act_abrir_a, self.act_abrir_b, self.act_guardar, self.act_usar, self.act_restaurar):
            tb.addAction(a)

    def _crear_ui(self):
        # --- paneles: A | B | Resultado ---
        self.p_a = Panel("Imagen A", "Abrí la imagen A")
        self.p_b = Panel("Imagen B", "Abrí la imagen B")
        self.p_res = Panel("Resultado", "Elegí una operación")
        paneles = (self.p_a, self.p_b, self.p_res)
        for p in paneles:
            p.lienzo.pixel_hover.connect(self._hover)
        # scroll sincronizado entre los tres paneles
        for a in paneles:
            for b in paneles:
                if a is not b:
                    a.scroll.horizontalScrollBar().valueChanged.connect(b.scroll.horizontalScrollBar().setValue)
                    a.scroll.verticalScrollBar().valueChanged.connect(b.scroll.verticalScrollBar().setValue)
        split = QSplitter(Qt.Horizontal)
        for p in paneles:
            split.addWidget(p)

        # --- panel lateral ---
        lateral = QWidget()
        lateral.setFixedWidth(360)
        lv = QVBoxLayout(lateral)

        g_img = QGroupBox("Imágenes")
        fi = QFormLayout(g_img)
        self.lbl_a, self.lbl_b = QLabel("—"), QLabel("—")
        self.lbl_a.setWordWrap(True)
        self.lbl_b.setWordWrap(True)
        fi.addRow("A:", self.lbl_a)
        fi.addRow("B:", self.lbl_b)
        self.sp_ancho = QSpinBox(minimum=0, maximum=10000, value=800, singleStep=100, suffix=" px")
        self.sp_ancho.setSpecialValueText("sin reducir")
        self.sp_ancho.setToolTip("Las imágenes más anchas se reducen al abrirlas (vecino más cercano). "
                                 "0 = no reducir.")
        fi.addRow("Ancho máx. al abrir:", self.sp_ancho)
        hb = QHBoxLayout()
        b_a = QPushButton("Abrir A...")
        b_a.clicked.connect(lambda: self.abrir("A"))
        b_b = QPushButton("Abrir B...")
        b_b.clicked.connect(lambda: self.abrir("B"))
        self.btn_swap = QPushButton("A ↔ B")
        self.btn_swap.setToolTip("Intercambiar A y B (importa en las restas)")
        self.btn_swap.clicked.connect(self.intercambiar)
        hb.addWidget(b_a)
        hb.addWidget(b_b)
        hb.addWidget(self.btn_swap)
        fi.addRow(hb)
        self.lbl_aviso = QLabel("")
        self.lbl_aviso.setWordWrap(True)
        self.lbl_aviso.setStyleSheet("color: #b36b00;")
        fi.addRow(self.lbl_aviso)
        lv.addWidget(g_img)

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

        # un botón por operación, agrupados como en operaciones.OPERACIONES
        self.btns_op = []
        for grupo, lista in ops.OPERACIONES.items():
            g = QGroupBox(grupo)
            grid = QGridLayout(g)
            for k, (funcion, texto, formula) in enumerate(lista):
                b = QPushButton(texto)
                b.setToolTip(formula)
                titulo = ("Suma " if funcion.startswith("suma") else "Resta ") + texto
                b.clicked.connect(lambda _=False, f=funcion, t=titulo, fo=formula:
                                  self.operar(f, t, fo))
                grid.addWidget(b, k // 2, k % 2)
                self.btns_op.append(b)
            lv.addWidget(g)

        self.lbl_formula = QLabel("La fórmula de cada operación aparece al pasar el mouse por su botón.")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setStyleSheet("color: #666;")
        lv.addWidget(self.lbl_formula)

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
        self.statusBar().showMessage("Abrí las imágenes A y B")

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
        hay_dos = self.img_a is not None and self.img_b is not None
        for b in self.btns_op:
            b.setEnabled(hay_dos)
        self.btn_swap.setEnabled(hay_dos)
        self.act_guardar.setEnabled(self.resultado is not None)
        self.act_usar.setEnabled(self.resultado is not None)
        self.act_restaurar.setEnabled(self.inicial_a is not None)

        def describir(img, ruta):
            if img is None:
                return "—"
            h, w = img.shape[:2]
            return f"{os.path.basename(ruta) if ruta else '(resultado)'} — {w} × {h} px"

        self.lbl_a.setText(describir(self.img_a, self.ruta_a))
        self.lbl_b.setText(describir(self.img_b, self.ruta_b))
        if hay_dos and self.img_a.shape[:2] != self.img_b.shape[:2]:
            h = min(self.img_a.shape[0], self.img_b.shape[0])
            w = min(self.img_a.shape[1], self.img_b.shape[1])
            self.lbl_aviso.setText(f"Tamaños distintos: se opera sobre el área común {w} × {h} px "
                                   "(esquina superior izquierda).")
        else:
            self.lbl_aviso.setText("")

    def _redibujar(self):
        z = self.sl_zoom.value() / 100
        self.p_a.lienzo.mostrar(self.img_a, z)
        self.p_b.lienzo.mostrar(self.img_b, z)
        self.p_res.lienzo.mostrar(self.resultado, z)

    def _cambio_zoom(self, val):
        self.lbl_zoom.setText(f"{val} %")
        self._redibujar()

    def _zoom_ajustar(self):
        ref = self.img_a if self.img_a is not None else self.img_b
        if ref is None:
            return
        h, w = ref.shape[:2]
        vp = self.p_a.scroll.viewport().size()
        z = min(vp.width() / w, vp.height() / h) * 100
        self.sl_zoom.setValue(int(max(10, min(800, z))))

    def _set_resultado(self, arr, titulo=""):
        self.resultado = arr
        self.p_res.titulo.setText(f"<b>Resultado</b>{' — ' + titulo if titulo else ''}")
        self._redibujar()
        self._actualizar_controles()

    def _hover(self, x, y):
        lineas = [f"(x, y) = ({x}, {y})"]
        for nombre, img in (("A", self.img_a), ("B", self.img_b), ("C", self.resultado)):
            if img is not None and y < img.shape[0] and x < img.shape[1]:
                px = img[y, x]
                lineas.append(f"{nombre}: RGB(%3d, %3d, %3d)  Y=%.3f" % (*px, luminancia_byte(px)))
        self.lbl_pixel.setText("\n".join(lineas))

    # ------------------------------------------------------------ archivo
    def abrir(self, cual):
        ruta, _ = QFileDialog.getOpenFileName(self, f"Abrir imagen {cual}", "", FILTRO_IMAGENES)
        if ruta:
            self.cargar(cual, ruta)

    def cargar(self, cual, ruta):
        """Abre el archivo con operaciones.abrir_imagen y lo reduce si es muy ancho."""
        try:
            img = llamar("abrir_imagen", ruta)
            if not isinstance(img, QImage) or img.isNull():
                QMessageBox.warning(self, "Abrir", "No se pudo abrir la imagen.")
                return
            arr = qimage_a_array(img)
            arr = validar(llamar("reducir", arr, self.sp_ancho.value()), "reducir")
        except FaltaImplementar as e:
            return self._falta(str(e))
        except Exception as e:
            return self._error(e)
        if cual == "A":
            self.img_a, self.inicial_a, self.ruta_a = arr, arr.copy(), ruta
        else:
            self.img_b, self.ruta_b = arr, ruta
        self._set_resultado(None)
        self._zoom_ajustar()
        self.statusBar().showMessage(f"Imagen {cual}: {ruta}")

    def guardar(self):
        if self.resultado is None:
            return
        base = os.path.splitext(self.ruta_a or "resultado")[0] + "_resultado.png"
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
        """El resultado pasa a ser A: permite encadenar operaciones."""
        if self.resultado is not None:
            self.img_a, self.ruta_a = self.resultado.copy(), None
            self._set_resultado(None)
            self.statusBar().showMessage("El resultado ahora es la imagen A")

    def restaurar(self):
        if self.inicial_a is not None:
            self.img_a = self.inicial_a.copy()
            self._set_resultado(None)
            self.statusBar().showMessage("Imagen A restaurada")

    def intercambiar(self):
        self.img_a, self.img_b = self.img_b, self.img_a
        self.ruta_a, self.ruta_b = self.ruta_b, self.ruta_a
        self._set_resultado(None)

    # ------------------------------------------------------------ operaciones
    def operar(self, funcion, titulo, formula):
        """Workflow: recortar al tamaño común -> normalizar -> operar -> bytes."""
        if self.img_a is None or self.img_b is None:
            return
        self.lbl_formula.setText(f"<b>{titulo}</b>: {formula}")

        def f():
            a, b = llamar("recortar_al_tamano_comun", self.img_a, self.img_b)
            na = validar(llamar("normalizar", a), "normalizar", a.shape)
            nb = validar(llamar("normalizar", b), "normalizar", b.shape)
            c = validar(llamar(funcion, na, nb), funcion, a.shape)
            out = validar(llamar("a_bytes", c), "a_bytes", a.shape, dtype_uint8=True)
            self._set_resultado(out, titulo)
        self._ejecutar(f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = VentanaPrincipal()
    v.show()
    sys.exit(app.exec())
