"""
app.py — TP1 Espacios cromáticos: interfaz gráfica (PySide6). No hace falta modificar este archivo.
Ejecutar con:   python app.py
Requiere:       pip install PySide6 numpy
Las operaciones se implementan en operaciones.py
"""

import os
import sys
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import QPixmap, QImage, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QScrollArea, QPushButton,
    QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QMessageBox, QSlider, QDoubleSpinBox, QTabWidget, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QDialog, QLineEdit, QSplitter, QSizePolicy,
)

import operaciones as ops

FILTRO_IMAGENES = "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif *.ppm *.pgm *.tif *.tiff);;Todos (*)"
FILTRO_DATASETS = "Datasets 2D (*.npy *.csv *.txt);;Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Todos (*)"


class FaltaImplementar(Exception):
    pass


# ---------------------------------------------------------------------------
# Conversión QImage <-> numpy (plomería, no es parte del TP)
# ---------------------------------------------------------------------------
def qimage_a_array(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=bpl * h)
    return buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3).copy()


def array_a_qimage(arr: np.ndarray) -> QImage:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def gris_a_bytes(d: np.ndarray, vmin=0.0, vmax=1.0) -> np.ndarray:
    """Visualización interna de un dato 2D en gris (sin usar operaciones.py)."""
    n = np.clip((np.asarray(d, float) - vmin) / ((vmax - vmin) or 1), 0, 1)
    return np.repeat((n * 255).round().astype(np.uint8)[..., None], 3, axis=-1)


def dataset_ejemplo(alto=360, ancho=720, semilla=0) -> np.ndarray:
    """Mapa de altitud sintético (metros): suma de 'montañas' gaussianas."""
    rng = np.random.default_rng(semilla)
    yy, xx = np.mgrid[0:alto, 0:ancho]
    z = np.zeros((alto, ancho))
    for _ in range(25):
        cx, cy = rng.uniform(0, ancho), rng.uniform(0, alto)
        s = rng.uniform(20, 110)
        z += rng.uniform(300, 2500) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    z += rng.normal(0, 40, z.shape)
    return np.clip(z, 0, None)


def validar(arr, nombre, ndim, forma=None, dtype_uint8=False):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{nombre}() debe devolver un np.ndarray (devolvió {type(arr).__name__}).")
    if arr.ndim != ndim:
        esp = "(alto, ancho, 3)" if ndim == 3 else "(alto, ancho)"
        raise ValueError(f"{nombre}() devolvió forma {arr.shape}; se esperaba {esp}.")
    if forma is not None and arr.shape[:2] != forma[:2]:
        raise ValueError(f"{nombre}() devolvió tamaño {arr.shape[:2]}; se esperaba {forma[:2]}.")
    if ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"{nombre}() devolvió {arr.shape[2]} canales; se esperaban 3.")
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
# Widgets
# ---------------------------------------------------------------------------
class LienzoImagen(QLabel):
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


class ControlCoef(QWidget):
    """Slider + spinbox para un coeficiente (0.00 .. 2.00) con botones de valores típicos."""
    cambiado = Signal(float)

    def __init__(self, presets):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal, minimum=0, maximum=200, value=100)
        self.spin = QDoubleSpinBox(minimum=0.0, maximum=2.0, singleStep=0.05, decimals=2, value=1.0)
        self.slider.valueChanged.connect(lambda val: self.spin.setValue(val / 100))
        self.spin.valueChanged.connect(self._spin)
        h.addWidget(self.slider, 1)
        h.addWidget(self.spin)
        v.addLayout(h)
        hp = QHBoxLayout()
        hp.setSpacing(3)
        for p in presets:
            b = QPushButton(f"{p:g}")
            b.setFixedWidth(42)
            b.clicked.connect(lambda _=False, p=p: self.spin.setValue(p))
            hp.addWidget(b)
        hp.addStretch(1)
        v.addLayout(hp)

    def _spin(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(round(val * 100))
        self.slider.blockSignals(False)
        self.cambiado.emit(val)

    def valor(self):
        return self.spin.value()

    def set_valor(self, v):
        self.spin.setValue(v)


class BarraColores(QWidget):
    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.lbl_min, self.lbl_max = QLabel("—"), QLabel("—")
        self.barra = QLabel()
        self.barra.setFixedHeight(18)
        self.barra.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.barra.setScaledContents(True)
        h.addWidget(self.lbl_min)
        h.addWidget(self.barra, 1)
        h.addWidget(self.lbl_max)

    def mostrar(self, rgb_bytes, vmin, vmax):
        self.barra.setPixmap(QPixmap.fromImage(array_a_qimage(rgb_bytes)))
        self.lbl_min.setText(f"{vmin:.4g}")
        self.lbl_max.setText(f"{vmax:.4g}")

    def limpiar(self):
        self.barra.clear()
        self.lbl_min.setText("—"); self.lbl_max.setText("—")


# ---------------------------------------------------------------------------
# Diálogo: grilla de combinaciones a × b (como en el notebook)
# ---------------------------------------------------------------------------
class DialogoGrilla(QDialog):
    def __init__(self, parent, img_bytes):
        super().__init__(parent)
        self.setWindowTitle("Grilla de combinaciones a × b")
        self.img = img_bytes
        self.resize(1100, 800)
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.ed_a = QLineEdit("0.4, 0.7, 1.0, 1.2, 1.5")
        self.ed_b = QLineEdit("0, 0.5, 1.0, 1.2, 1.5")
        self.ed_tam = QLineEdit("200")
        f.addRow("Valores de a (filas):", self.ed_a)
        f.addRow("Valores de b (columnas):", self.ed_b)
        f.addRow("Ancho de miniatura (px):", self.ed_tam)
        v.addLayout(f)
        btn = QPushButton("Generar grilla")
        btn.clicked.connect(self.generar)
        v.addWidget(btn)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        v.addWidget(self.area, 1)

    def _miniatura(self, ancho):
        h, w = self.img.shape[:2]
        if w <= ancho:
            return self.img
        q = array_a_qimage(self.img).scaledToWidth(ancho, Qt.SmoothTransformation)
        return qimage_a_array(q)

    def generar(self):
        try:
            a_vals = [float(s) for s in self.ed_a.text().replace(";", ",").split(",") if s.strip()]
            b_vals = [float(s) for s in self.ed_b.text().replace(";", ",").split(",") if s.strip()]
            ancho = int(self.ed_tam.text())
        except ValueError:
            QMessageBox.warning(self, "Grilla", "Revisá los valores: números separados por coma.")
            return
        mini = self._miniatura(ancho)
        cont = QWidget()
        g = QGridLayout(cont)
        for j, b in enumerate(b_vals):
            g.addWidget(QLabel(f"<b>b = {b:g}</b>"), 0, j + 1, Qt.AlignCenter)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            rgb = validar(llamar("normalizar", mini), "normalizar", 3, mini.shape)
            yiq = validar(llamar("rgb_a_yiq", rgb), "rgb_a_yiq", 3, mini.shape)
            for i, a in enumerate(a_vals):
                g.addWidget(QLabel(f"<b>a = {a:g}</b>"), i + 1, 0)
                for j, b in enumerate(b_vals):
                    y2 = validar(llamar("ajustar_yiq", yiq, a, b), "ajustar_yiq", 3, mini.shape)
                    r2 = validar(llamar("yiq_a_rgb", y2), "yiq_a_rgb", 3, mini.shape)
                    out = validar(llamar("a_bytes", r2), "a_bytes", 3, mini.shape, True)
                    lbl = QLabel()
                    lbl.setPixmap(QPixmap.fromImage(array_a_qimage(out)))
                    lbl.setToolTip(f"a = {a:g}, b = {b:g}")
                    g.addWidget(lbl, i + 1, j + 1)
                    QApplication.processEvents()
        except FaltaImplementar as e:
            QApplication.restoreOverrideCursor()
            return self.parent()._falta(str(e))
        except Exception as e:
            QApplication.restoreOverrideCursor()
            return self.parent()._error(e)
        QApplication.restoreOverrideCursor()
        self.area.setWidget(cont)


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class VentanaPrincipal(QMainWindow):
    TINTES = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TP1 — Espacios cromáticos")
        self.resize(1400, 820)

        self.original = None      # uint8 (h, w, 3) — imagen de trabajo
        self.inicial = None       # uint8 — tal como se abrió (para Restaurar)
        self.resultado = None     # uint8 (h, w, 3) — lo que se muestra a la derecha
        self.ruta = None
        self.dataset = None       # float (h, w) — pestaña mapas cromáticos
        self.dataset_nombre = None
        self.yiq_cache = None     # (yiq_original, yiq_ajustada) para inspección de pixels
        self.vista_extra = None   # dato 2D mostrado en "Resultado" (canal, Y/I/Q, dataset)

        self.timer_preview = QTimer(self, singleShot=True, interval=300)
        self.timer_preview.timeout.connect(self.aplicar_yiq)

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
        m.addAction(self.act_abrir); m.addAction(self.act_guardar)
        m.addSeparator(); m.addAction(self.act_salir)
        e = self.menuBar().addMenu("&Editar")
        e.addAction(self.act_usar); e.addAction(self.act_restaurar)

        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        for a in (self.act_abrir, self.act_guardar, self.act_usar, self.act_restaurar):
            tb.addAction(a)

    def _crear_ui(self):
        # --- paneles de imagen: original | resultado ---
        self.p_orig = Panel("Original", "Abrí una imagen para empezar")
        self.p_res = Panel("Resultado", "Acá aparece el resultado")
        for p in (self.p_orig, self.p_res):
            p.lienzo.pixel_hover.connect(self._hover)
        # scroll sincronizado
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
        lv.addWidget(g_info)

        g_zoom = QGroupBox("Zoom (ambos paneles)")
        hz = QHBoxLayout(g_zoom)
        self.sl_zoom = QSlider(Qt.Horizontal, minimum=10, maximum=800, value=100)
        self.lbl_zoom = QLabel("100 %"); self.lbl_zoom.setMinimumWidth(50)
        self.sl_zoom.valueChanged.connect(self._cambio_zoom)
        btn_ajustar = QPushButton("Ajustar")
        btn_ajustar.clicked.connect(self._zoom_ajustar)
        hz.addWidget(self.sl_zoom, 1); hz.addWidget(self.lbl_zoom); hz.addWidget(btn_ajustar)
        lv.addWidget(g_zoom)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_canales(), "Canales")
        self.tabs.addTab(self._tab_yiq(), "Lum. / Sat.")
        self.tabs.addTab(self._tab_mapas(), "Mapas")
        lv.addWidget(self.tabs, 1)

        g_pix = QGroupBox("Pixel bajo el mouse")
        self.lbl_pixel = QLabel("—")
        self.lbl_pixel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_pixel.setStyleSheet("font-family: monospace;")
        QVBoxLayout(g_pix).addWidget(self.lbl_pixel)
        lv.addWidget(g_pix)

        central = QWidget()
        hl = QHBoxLayout(central)
        hl.addWidget(split, 1)
        hl.addWidget(lateral)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")

    def _tab_canales(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Punto 1: ver cada canal por separado."))
        h = QHBoxLayout()
        self.btns_canal = []
        for i, n in enumerate("RGB"):
            b = QPushButton(f"Canal {n}")
            b.clicked.connect(lambda _=False, i=i: self.ver_canal(i))
            h.addWidget(b)
            self.btns_canal.append(b)
        v.addLayout(h)
        hm = QHBoxLayout()
        hm.addWidget(QLabel("Mostrar como:"))
        self.rb_gris, self.rb_tenido = QRadioButton("gris"), QRadioButton("teñido")
        self.rb_tenido.setChecked(True)
        grp = QButtonGroup(w); grp.addButton(self.rb_gris); grp.addButton(self.rb_tenido)
        hm.addWidget(self.rb_gris); hm.addWidget(self.rb_tenido); hm.addStretch(1)
        v.addLayout(hm)
        v.addSpacing(12)
        v.addWidget(QLabel("Punto 2: intercambiar canales."))
        self.btn_intercambiar = QPushButton("Intercambiar canales (R→G, G→B, B→R)")
        self.btn_intercambiar.clicked.connect(self.intercambiar)
        v.addWidget(self.btn_intercambiar)
        v.addStretch(1)
        return w

    def _tab_yiq(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g_a = QGroupBox("a — coeficiente de luminancia  (Y' = a·Y)")
        self.ctl_a = ControlCoef([0.4, 0.7, 1.0, 1.2, 1.5])
        QVBoxLayout(g_a).addWidget(self.ctl_a)
        g_b = QGroupBox("b — coeficiente de saturación  (I' = b·I, Q' = b·Q)")
        self.ctl_b = ControlCoef([0.0, 0.5, 1.0, 1.2, 1.5])
        QVBoxLayout(g_b).addWidget(self.ctl_b)
        v.addWidget(g_a); v.addWidget(g_b)
        self.ctl_a.cambiado.connect(self._coef_cambiado)
        self.ctl_b.cambiado.connect(self._coef_cambiado)

        hm = QHBoxLayout()
        hm.addWidget(QLabel("Mostrar:"))
        self.cb_vista_yiq = QComboBox()
        self.cb_vista_yiq.addItems(["Resultado R'G'B'", "Canal Y'", "Canal I'", "Canal Q'"])
        self.cb_vista_yiq.currentIndexChanged.connect(lambda _: self.yiq_cache and self._mostrar_yiq())
        hm.addWidget(self.cb_vista_yiq, 1)
        v.addLayout(hm)

        self.chk_auto = QCheckBox("Vista previa automática al mover los sliders")
        v.addWidget(self.chk_auto)
        self.btn_yiq = QPushButton("Aplicar a y b")
        self.btn_yiq.setStyleSheet("font-weight: bold;")
        self.btn_yiq.clicked.connect(self.aplicar_yiq)
        v.addWidget(self.btn_yiq)
        hb = QHBoxLayout()
        btn_reset = QPushButton("a = b = 1")
        btn_reset.clicked.connect(lambda: (self.ctl_a.set_valor(1.0), self.ctl_b.set_valor(1.0)))
        self.btn_grilla = QPushButton("Grilla a × b...")
        self.btn_grilla.clicked.connect(self.grilla)
        hb.addWidget(btn_reset); hb.addWidget(self.btn_grilla)
        v.addLayout(hb)
        v.addStretch(1)
        return w

    def _tab_mapas(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g_d = QGroupBox("Dataset 2D")
        vd = QVBoxLayout(g_d)
        self.lbl_dataset = QLabel("Sin dataset (se usa la luminancia de la imagen abierta)")
        self.lbl_dataset.setWordWrap(True)
        vd.addWidget(self.lbl_dataset)
        hd = QHBoxLayout()
        b1 = QPushButton("Abrir dataset...")
        b1.setToolTip(".npy, .csv/.txt (matriz de números) o una imagen")
        b1.clicked.connect(self.abrir_dataset)
        b2 = QPushButton("Ejemplo (altitud)")
        b2.clicked.connect(self.dataset_ejemplo)
        b3 = QPushButton("Usar imagen")
        b3.clicked.connect(self.dataset_de_imagen)
        hd.addWidget(b1); hd.addWidget(b2); hd.addWidget(b3)
        vd.addLayout(hd)
        v.addWidget(g_d)

        g_p = QGroupBox("Paleta")
        fp = QFormLayout(g_p)
        self.cb_paleta = QComboBox()
        self.cb_paleta.addItems(list(getattr(ops, "PALETAS", ["gris", "arcoiris"])))
        fp.addRow("Paleta:", self.cb_paleta)
        self.chk_percentil = QCheckBox("Mejorar contraste (percentiles 2–98)")
        fp.addRow(self.chk_percentil)
        v.addWidget(g_p)

        self.btn_paleta = QPushButton("Aplicar paleta")
        self.btn_paleta.setStyleSheet("font-weight: bold;")
        self.btn_paleta.clicked.connect(self.aplicar_paleta)
        v.addWidget(self.btn_paleta)
        self.cb_paleta.currentIndexChanged.connect(
            lambda _: self.vista_extra is not None and self.tabs.currentIndex() == 2 and self.aplicar_paleta())
        self.chk_percentil.toggled.connect(
            lambda _: self.vista_extra is not None and self.tabs.currentIndex() == 2 and self.aplicar_paleta())

        v.addWidget(QLabel("Barra de colores:"))
        self.barra = BarraColores()
        v.addWidget(self.barra)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------ utilidades
    def _falta(self, nombre):
        QMessageBox.information(self, "Falta implementar",
                                f"La función <b>{nombre}()</b> de <i>operaciones.py</i> todavía no está implementada.")

    def _error(self, e):
        QMessageBox.critical(self, "Error", f"{type(e).__name__}: {e}")

    def _ejecutar(self, f):
        """Corre f() con cursor de espera, mide tiempo y traduce errores. Devuelve True si salió bien."""
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

    def _normalizada(self):
        return validar(llamar("normalizar", self.original), "normalizar", 3, self.original.shape)

    def _a_bytes(self, rgb):
        return validar(llamar("a_bytes", rgb), "a_bytes", 3, rgb.shape, dtype_uint8=True)

    def _actualizar_controles(self):
        hay = self.original is not None
        for wd in (self.act_usar, self.act_restaurar, self.btn_intercambiar, self.btn_yiq,
                   self.btn_grilla, *self.btns_canal):
            wd.setEnabled(hay)
        self.act_guardar.setEnabled(self.resultado is not None)
        self.act_usar.setEnabled(self.resultado is not None)
        if hay:
            h, w = self.original.shape[:2]
            self.lbl_tamano.setText(f"{w} × {h} px")
            self.lbl_archivo.setText(os.path.basename(self.ruta) if self.ruta else "—")
        self.setWindowTitle("TP1 — Espacios cromáticos" + (f" — {os.path.basename(self.ruta)}" if self.ruta else ""))

    def _redibujar(self):
        z = self.sl_zoom.value() / 100
        self.p_orig.lienzo.mostrar(self._izquierda(), z)
        self.p_res.lienzo.mostrar(self.resultado, z)

    def _izquierda(self):
        if self.tabs.currentIndex() == 2 and self.dataset is not None and self.dataset_nombre != "imagen":
            return gris_a_bytes(self.dataset, self.dataset.min(), self.dataset.max())
        return self.original

    def _cambio_zoom(self, val):
        self.lbl_zoom.setText(f"{val} %")
        self._redibujar()

    def _zoom_ajustar(self):
        ref = self._izquierda()
        if ref is None:
            return
        h, w = ref.shape[:2]
        vp = self.p_orig.scroll.viewport().size()
        z = min(vp.width() / w, vp.height() / h) * 100
        self.sl_zoom.setValue(int(max(10, min(800, z))))

    def _set_resultado(self, arr, titulo, extra=None):
        self.resultado, self.vista_extra = arr, extra
        self.p_res.titulo.setText(f"<b>Resultado</b> — {titulo}")
        self._redibujar()
        self._actualizar_controles()

    def _hover(self, x, y):
        lineas = [f"(x, y) = ({x}, {y})"]
        izq = self._izquierda()
        if self.tabs.currentIndex() == 2 and self.dataset is not None and self.dataset_nombre != "imagen":
            if y < self.dataset.shape[0] and x < self.dataset.shape[1]:
                lineas.append(f"Dato:      {self.dataset[y, x]:.4g}")
        elif izq is not None and y < izq.shape[0] and x < izq.shape[1]:
            lineas.append("Original:  RGB(%3d, %3d, %3d)" % tuple(izq[y, x]))
        if self.resultado is not None and y < self.resultado.shape[0] and x < self.resultado.shape[1]:
            lineas.append("Resultado: RGB(%3d, %3d, %3d)" % tuple(self.resultado[y, x]))
        if self.yiq_cache and self.tabs.currentIndex() == 1:
            yo, ya = self.yiq_cache
            if y < yo.shape[0] and x < yo.shape[1]:
                lineas.append("YIQ:   (%.3f, %+.3f, %+.3f)" % tuple(yo[y, x]))
                lineas.append("Y'I'Q':(%.3f, %+.3f, %+.3f)" % tuple(ya[y, x]))
        self.lbl_pixel.setText("\n".join(lineas))

    # ------------------------------------------------------------ archivo
    def abrir(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", FILTRO_IMAGENES)
        if not ruta:
            return
        try:
            img = llamar("abrir_imagen", ruta)
        except FaltaImplementar as e:
            return self._falta(str(e))
        except Exception as e:
            return self._error(e)
        if not isinstance(img, QImage) or img.isNull():
            QMessageBox.warning(self, "Abrir", "No se pudo abrir la imagen.")
            return
        arr = qimage_a_array(img)
        self.original, self.inicial, self.ruta = arr, arr.copy(), ruta
        self.yiq_cache = None
        if self.dataset_nombre == "imagen":
            self.dataset = self.dataset_nombre = None
            self.lbl_dataset.setText("Sin dataset (se usa la luminancia de la imagen abierta)")
        self._set_resultado(None, "")
        self.p_res.titulo.setText("<b>Resultado</b>")
        self._actualizar_controles()
        self._zoom_ajustar()
        self.statusBar().showMessage(f"Abierta: {ruta}")

    def guardar(self):
        if self.resultado is None:
            return
        base = os.path.splitext(self.ruta or "resultado")[0] + "_resultado.png"
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
        if self.resultado is not None:
            self.original = self.resultado.copy()
            self.yiq_cache = None
            self._set_resultado(None, "")
            self.p_res.titulo.setText("<b>Resultado</b>")
            self.statusBar().showMessage("El resultado ahora es la imagen original (podés encadenar operaciones)")

    def restaurar(self):
        if self.inicial is not None:
            self.original = self.inicial.copy()
            self.yiq_cache = None
            self._set_resultado(None, "")
            self.p_res.titulo.setText("<b>Resultado</b>")
            self.statusBar().showMessage("Imagen inicial restaurada")

    # ------------------------------------------------------------ canales RGB
    def ver_canal(self, i):
        def f():
            rgb = self._normalizada()
            canal = validar(llamar("extraer_canal", rgb, i), "extraer_canal", 2, rgb.shape).astype(float)
            if self.rb_tenido.isChecked():
                vis = canal[..., None] * np.array(self.TINTES[i], float)
            else:
                vis = np.repeat(canal[..., None], 3, axis=-1)
            out = self._a_bytes(vis)
            self._set_resultado(out, f"canal {'RGB'[i]} ({'teñido' if self.rb_tenido.isChecked() else 'gris'})")
        self._ejecutar(f)

    def intercambiar(self):

        def f():

            if self.resultado is None:
                base = self.original
            else:
                base = self.resultado

            rgb = validar(
                llamar("normalizar", base),
                "normalizar",
                3,
                base.shape
            )

            nuevo = validar(
                llamar("intercambiar_canales", rgb),
                "intercambiar_canales",
                3,
                rgb.shape
            )

            self._set_resultado(
                self._a_bytes(nuevo),
                "canales intercambiados"
            )

        self._ejecutar(f)

    # ------------------------------------------------------------ YIQ
    def _coef_cambiado(self, _):
        if self.chk_auto.isChecked() and self.original is not None:
            self.timer_preview.start()

    def aplicar_yiq(self):
        if self.original is None:
            return
        a, b = self.ctl_a.valor(), self.ctl_b.valor()

        def f():
            sh = self.original.shape
            rgb = self._normalizada()                                                     # paso 1
            yiq = validar(llamar("rgb_a_yiq", rgb), "rgb_a_yiq", 3, sh)                    # paso 2
            yiq_copia = yiq.copy()
            yiq2 = validar(llamar("ajustar_yiq", yiq, a, b), "ajustar_yiq", 3, sh)        # pasos 3-6
            if not np.array_equal(yiq, yiq_copia):
                self.statusBar().showMessage("Ojo: ajustar_yiq modificó la imagen YIQ recibida")
            rgb2 = validar(llamar("yiq_a_rgb", yiq2), "yiq_a_rgb", 3, sh)                  # paso 7
            out = self._a_bytes(rgb2)                                                      # paso 8
            self.yiq_cache = (yiq_copia, yiq2)
            self._out_yiq = out
            self._ab = (a, b)
            self._mostrar_yiq()
        self._ejecutar(f)

    def _mostrar_yiq(self):
        a, b = self._ab
        vista = self.cb_vista_yiq.currentIndex()
        if vista == 0:
            return self._set_resultado(self._out_yiq, f"a = {a:g}, b = {b:g}")
        yiq2 = self.yiq_cache[1]
        rangos = {1: (0, 1), 2: (-0.5957, 0.5957), 3: (-0.5226, 0.5226)}
        lo, hi = rangos[vista]
        nombre = {1: "Y'", 2: "I'", 3: "Q'"}[vista]
        self._set_resultado(gris_a_bytes(yiq2[..., vista - 1], lo, hi),
                            f"canal {nombre} (gris, rango {lo:g}..{hi:g}) — a = {a:g}, b = {b:g}",
                            extra=yiq2[..., vista - 1])

    def grilla(self):
        if self.original is not None:
            DialogoGrilla(self, self.original).exec()

    # ------------------------------------------------------------ mapas cromáticos
    def _set_dataset(self, datos, nombre, descripcion):
        self.dataset, self.dataset_nombre = np.asarray(datos, float), nombre
        h, w = self.dataset.shape
        self.lbl_dataset.setText(f"{descripcion}\n{w} × {h}, valores {np.nanmin(self.dataset):.4g} .. "
                                 f"{np.nanmax(self.dataset):.4g}")
        self.barra.limpiar()
        self._set_resultado(None, "")
        self.p_res.titulo.setText("<b>Resultado</b>")
        self._zoom_ajustar()

    def dataset_ejemplo(self):
        self._set_dataset(dataset_ejemplo(), "ejemplo", "Ejemplo: mapa de altitud sintético (metros)")

    def dataset_de_imagen(self):
        if self.original is None:
            QMessageBox.information(self, "Mapas cromáticos", "Primero abrí una imagen.")
            return
        def f():
            lum = validar(llamar("luminancia", self._normalizada()), "luminancia", 2, self.original.shape)
            self._set_dataset(lum, "imagen", "Luminancia de la imagen abierta")
        self._ejecutar(f)

    def abrir_dataset(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir dataset 2D", "", FILTRO_DATASETS)
        if not ruta:
            return
        ext = os.path.splitext(ruta)[1].lower()
        try:
            if ext == ".npy":
                datos = np.load(ruta)
            elif ext in (".csv", ".txt"):
                try:
                    datos = np.loadtxt(ruta, delimiter=",")
                except ValueError:
                    datos = np.loadtxt(ruta)
            else:
                img = llamar("abrir_imagen", ruta)
                if not isinstance(img, QImage) or img.isNull():
                    raise ValueError("No se pudo abrir la imagen.")
                datos = validar(llamar("luminancia", llamar("normalizar", qimage_a_array(img))),
                                "luminancia", 2)
            datos = np.asarray(datos, float)
            if datos.ndim == 3 and datos.shape[2] == 1:
                datos = datos[..., 0]
            if datos.ndim != 2:
                raise ValueError(f"El dataset debe ser 2D (tiene forma {datos.shape}).")
        except FaltaImplementar as e:
            return self._falta(str(e))
        except Exception as e:
            return self._error(e)
        self._set_dataset(datos, "archivo", f"Archivo: {os.path.basename(ruta)}")

    def aplicar_paleta(self):
        if self.dataset is None:
            if self.original is None:
                QMessageBox.information(self, "Mapas cromáticos",
                                        "Cargá un dataset (o abrí una imagen y tocá «Usar imagen»).")
                return
            self.dataset_de_imagen()
            if self.dataset is None:
                return
        paleta = self.cb_paleta.currentText()

        def f():
            d = self.dataset
            if self.chk_percentil.isChecked():
                vmin, vmax = np.nanpercentile(d, [2, 98])
            else:
                vmin, vmax = np.nanmin(d), np.nanmax(d)
            norm = np.clip((d - vmin) / ((vmax - vmin) or 1), 0, 1)
            norm = np.nan_to_num(norm)
            rgb = validar(llamar("aplicar_paleta", norm, paleta), "aplicar_paleta", 3, d.shape)
            out = self._a_bytes(rgb)
            grad = np.tile(np.linspace(0, 1, 256), (8, 1))
            barra = self._a_bytes(validar(llamar("aplicar_paleta", grad, paleta), "aplicar_paleta", 3, grad.shape))
            self.barra.mostrar(barra, vmin, vmax)
            self._set_resultado(out, f"paleta «{paleta}»", extra=d)
        self._ejecutar(f)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = VentanaPrincipal()
    v.show()
    sys.exit(app.exec())
