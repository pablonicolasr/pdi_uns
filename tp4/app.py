"""
app.py — TP4 Transformada de Fourier: interfaz gráfica (PySide6).

Ejecutar con:   python app.py
Requiere:       pip install PySide6 numpy

Las operaciones se implementan en operaciones.py; este archivo solo
arma la ventana, muestra imagen, espectro y reconstrucción, y maneja
los archivos del espectro de amplitud.
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
    QMessageBox, QSlider, QSpinBox, QDoubleSpinBox, QSplitter, QComboBox,
    QCheckBox,
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


def llamar(nombre, *args, **kwargs):
    """Llama a operaciones.<nombre>, traduciendo NotImplementedError."""
    try:
        return getattr(ops, nombre)(*args, **kwargs)
    except NotImplementedError:
        raise FaltaImplementar(nombre)


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
        self.setWindowTitle("TP4 — Transformada de Fourier")
        self.resize(1550, 880)

        self.imagen = None          # uint8 (h, w, 3) — imagen de entrada
        self.figura = None          # uint8 (h, w) — figura en gris (para agregar más)
        self.ruta = None
        self.datos = None           # resultado de ops.transformada_directa
        self.amplitud = None        # |F|
        self.fase = None            # arg(F)
        self.espectro = None        # uint8 (h, w, 3) — lo que se ve en el panel del medio
        self.reconstruida = None    # uint8 (h, w, 3)

        self._crear_acciones()
        self._crear_ui()
        self._cambio_figura()
        self._actualizar_controles()

    # ------------------------------------------------------------ interfaz
    def _crear_acciones(self):
        A = lambda t, s, f: QAction(t, self, shortcut=s, triggered=f)
        self.act_abrir = A("Abrir imagen...", QKeySequence.Open, self.abrir)
        self.act_guardar_rec = A("Guardar reconstrucción...", QKeySequence.Save, self.guardar_reconstruccion)
        self.act_guardar_esp = A("Guardar vista del espectro...", QKeySequence(), self.guardar_espectro)
        self.act_salir = A("Salir", QKeySequence.Quit, self.close)

        m = self.menuBar().addMenu("&Archivo")
        for a in (self.act_abrir, self.act_guardar_rec, self.act_guardar_esp):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_salir)

        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        for a in (self.act_abrir, self.act_guardar_rec, self.act_guardar_esp):
            tb.addAction(a)

    def _crear_ui(self):
        # --- paneles: imagen | espectro | reconstrucción ---
        self.p_img = Panel("Imagen", "Abrí una imagen o dibujá una figura")
        self.p_esp = Panel("Espectro", "Tocá «TF directa»")
        self.p_rec = Panel("Reconstrucción", "Tocá «TF inversa»")
        paneles = (self.p_img, self.p_esp, self.p_rec)
        for p in paneles:
            p.lienzo.pixel_hover.connect(self._hover)
        for a in paneles:
            for b in paneles:
                if a is not b:
                    a.scroll.horizontalScrollBar().valueChanged.connect(b.scroll.horizontalScrollBar().setValue)
                    a.scroll.verticalScrollBar().valueChanged.connect(b.scroll.verticalScrollBar().setValue)
        split = QSplitter(Qt.Horizontal)
        for p in paneles:
            split.addWidget(p)
        split.setSizes([1000, 1000, 1000])   # tres paneles del mismo ancho

        # --- panel lateral ---
        lateral = QWidget()
        lateral.setFixedWidth(370)
        self.lateral = lateral
        lv = QVBoxLayout(lateral)

        lv.addWidget(self._grupo_imagen())
        lv.addWidget(self._grupo_figura())
        lv.addWidget(self._grupo_tf())
        lv.addWidget(self._grupo_amplitud())

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

    def _grupo_imagen(self):
        g = QGroupBox("Imagen")
        f = QFormLayout(g)
        self.lbl_archivo = QLabel("—")
        self.lbl_archivo.setWordWrap(True)
        f.addRow("Origen:", self.lbl_archivo)
        self.sp_ancho = QSpinBox(minimum=0, maximum=10000, value=1024, singleStep=128, suffix=" px")
        self.sp_ancho.setSpecialValueText("sin reducir")
        self.sp_ancho.setToolTip("Las imágenes más anchas se reducen al abrirlas (como read_img del notebook).")
        f.addRow("Ancho máx. al abrir:", self.sp_ancho)
        b = QPushButton("Abrir imagen...")
        b.clicked.connect(self.abrir)
        f.addRow(b)
        return g

    def _grupo_figura(self):
        g = QGroupBox("Figura geométrica (negra sobre blanco)")
        f = QFormLayout(g)
        self.cb_figura = QComboBox()
        for clave, (nombre, _, _) in ops.FIGURAS.items():
            self.cb_figura.addItem(nombre, clave)
        self.cb_figura.currentIndexChanged.connect(self._cambio_figura)
        f.addRow("Figura:", self.cb_figura)
        self.sp_tamano = QSpinBox(minimum=16, maximum=2048, value=512, singleStep=64, suffix=" px")
        f.addRow("Imagen N × N:", self.sp_tamano)
        self.lbl_p1, self.lbl_p2 = QLabel(), QLabel()
        self.sp_p1 = QDoubleSpinBox(minimum=1, maximum=2048, value=64, decimals=1)
        self.sp_p2 = QDoubleSpinBox(minimum=1, maximum=2048, value=32, decimals=1)
        f.addRow(self.lbl_p1, self.sp_p1)
        f.addRow(self.lbl_p2, self.sp_p2)
        hd = QHBoxLayout()
        self.sp_dx = QSpinBox(minimum=-1024, maximum=1024, value=0)
        self.sp_dy = QSpinBox(minimum=-1024, maximum=1024, value=0)
        hd.addWidget(QLabel("x"))
        hd.addWidget(self.sp_dx)
        hd.addWidget(QLabel("y"))
        hd.addWidget(self.sp_dy)
        f.addRow("Desplazamiento:", hd)
        self.sp_angulo = QDoubleSpinBox(minimum=-360, maximum=360, value=0, decimals=1, suffix=" °")
        f.addRow("Rotación:", self.sp_angulo)
        self.chk_agregar = QCheckBox("Agregar a la figura actual (varios objetos)")
        f.addRow(self.chk_agregar)
        b = QPushButton("Dibujar figura")
        b.clicked.connect(self.dibujar)
        f.addRow(b)
        return g

    def _grupo_tf(self):
        g = QGroupBox("Transformada de Fourier (de la luminancia Y)")
        v = QVBoxLayout(g)
        self.chk_media = QCheckBox("Restar la media de Y (quita el pico central)")
        self.chk_centrar = QCheckBox("Centrar el espectro (frecuencia 0 en el medio)")
        self.chk_centrar.setChecked(True)
        v.addWidget(self.chk_media)
        v.addWidget(self.chk_centrar)
        fm = QFormLayout()
        self.cb_metodo = QComboBox()
        for clave, nombre in ops.METODOS_DFT.items():
            self.cb_metodo.addItem(nombre, clave)
        self.cb_metodo.setToolTip("Las tres dan el mismo resultado. La suma directa es la fórmula "
                                  "tal cual (4 bucles): usala con figuras chicas: 32 × 32 tarda unos 2 s y 64 × 64, más de medio minuto.")
        self.cb_metodo.currentIndexChanged.connect(lambda _: self._actualizar_estimacion())
        fm.addRow("Cálculo:", self.cb_metodo)
        self.cb_norm = QComboBox()
        self.cb_norm.addItem("numpy (1/(M·N) en la inversa)", "numpy")
        self.cb_norm.addItem("forward (1/(M·N) en la directa)", "forward")
        self.cb_norm.setToolTip("Con «forward» |F| queda chico y la vista log(1 + |F|) se ve casi "
                                "igual que la lineal (más oscura).")
        fm.addRow("Normalización:", self.cb_norm)
        v.addLayout(fm)
        self.lbl_estimacion = QLabel("")
        self.lbl_estimacion.setWordWrap(True)
        self.lbl_estimacion.setStyleSheet("color: #666;")
        v.addWidget(self.lbl_estimacion)
        h = QHBoxLayout()
        self.btn_directa = QPushButton("TF directa")
        self.btn_directa.setStyleSheet("font-weight: bold;")
        self.btn_directa.clicked.connect(self.tf_directa)
        self.btn_inversa = QPushButton("TF inversa")
        self.btn_inversa.setStyleSheet("font-weight: bold;")
        self.btn_inversa.clicked.connect(self.tf_inversa)
        h.addWidget(self.btn_directa)
        h.addWidget(self.btn_inversa)
        v.addLayout(h)
        hm = QHBoxLayout()
        hm.addWidget(QLabel("Mostrar:"))
        self.cb_vista = QComboBox()
        self.cb_vista.addItems(["Amplitud, log(1 + |F|)", "Amplitud lineal", "Fase"])
        self.cb_vista.currentIndexChanged.connect(lambda _: self._mostrar_espectro())
        hm.addWidget(self.cb_vista, 1)
        v.addLayout(hm)
        return g

    def _grupo_amplitud(self):
        g = QGroupBox("Editar la amplitud (punto 3)")
        v = QVBoxLayout(g)
        lbl = QLabel("1) Guardá la amplitud, 2) editala con GIMP u otro editor (sin cambiar "
                     "el tamaño), 3) cargala: se reconstruye con la fase original.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #666;")
        v.addWidget(lbl)
        self.btn_guardar_amp = QPushButton("Guardar amplitud (PNG + JSON)...")
        self.btn_guardar_amp.clicked.connect(self.guardar_amplitud)
        self.btn_cargar_amp = QPushButton("Cargar amplitud editada y reconstruir...")
        self.btn_cargar_amp.clicked.connect(self.cargar_amplitud)
        v.addWidget(self.btn_guardar_amp)
        v.addWidget(self.btn_cargar_amp)
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
        self.lateral.setEnabled(False)   # que no se pueda tocar nada mientras calcula
        t0 = time.perf_counter()
        try:
            f()
        except FaltaImplementar as e:
            QApplication.restoreOverrideCursor()
            self.lateral.setEnabled(True)
            self._falta(str(e))
            return False
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.lateral.setEnabled(True)
            self._error(e)
            return False
        QApplication.restoreOverrideCursor()
        self.lateral.setEnabled(True)
        self.statusBar().showMessage(f"Procesado en {time.perf_counter() - t0:.2f} s")
        return True

    # cuántos términos de la suma por segundo hace Python, aprox. (para estimar el tiempo)
    TERMINOS_POR_SEGUNDO = {"directa": 5e5, "separable": 1e6, "matricial": 2e9}

    def _segundos_estimados(self, metodo):
        if self.imagen is None:
            return 0.0
        alto, ancho = self.imagen.shape[:2]
        return llamar("terminos_dft", alto, ancho, metodo) / self.TERMINOS_POR_SEGUNDO[metodo]

    def _actualizar_estimacion(self):
        """Muestra cuántos términos tiene la suma y cuánto tardaría con el método elegido."""
        if self.imagen is None:
            self.lbl_estimacion.setText("")
            return
        metodo = self.cb_metodo.currentData()
        alto, ancho = self.imagen.shape[:2]
        terminos = llamar("terminos_dft", alto, ancho, metodo)
        seg = self._segundos_estimados(metodo)
        tiempo = "menos de 1 s" if seg < 1 else (f"≈ {seg:.0f} s" if seg < 120 else f"≈ {seg / 60:.0f} min")
        self.lbl_estimacion.setText(f"{alto} × {ancho}: {terminos:,} términos por transformada "
                                    f"({tiempo}).".replace(",", "."))

    def _confirmar_tiempo(self, metodo):
        """Si la transformada va a tardar más de 20 s, pregunta antes de empezar."""
        seg = self._segundos_estimados(metodo)
        if seg <= 20:
            return True
        tiempo = f"{seg:.0f} segundos" if seg < 120 else f"{seg / 60:.0f} minutos"
        r = QMessageBox.question(
            self, "Puede tardar",
            f"Con «{ops.METODOS_DFT[metodo]}» esta transformada tarda unos {tiempo}.\n"
            "Para la suma directa conviene una figura de 32 × 32 (tarda unos 2 s).\n\n¿Calcular igual?")
        return r == QMessageBox.Yes

    def _progreso(self, que):
        """
        Función que operaciones.py llama con el avance (0..1): actualiza la
        barra de estado y deja que la ventana se redibuje mientras calcula.
        """
        def avisar(fraccion):
            self.statusBar().showMessage(f"{que}: {fraccion * 100:.0f} %")
            QApplication.processEvents()
        return avisar

    def _actualizar_controles(self):
        hay_img = self.imagen is not None
        hay_tf = self.datos is not None
        self.btn_directa.setEnabled(hay_img)
        self.btn_inversa.setEnabled(hay_tf)
        self.btn_guardar_amp.setEnabled(hay_tf)
        self.btn_cargar_amp.setEnabled(hay_tf)
        self.cb_vista.setEnabled(hay_tf)
        self.act_guardar_rec.setEnabled(self.reconstruida is not None)
        self.act_guardar_esp.setEnabled(self.espectro is not None)

    def _redibujar(self):
        z = self.sl_zoom.value() / 100
        self.p_img.lienzo.mostrar(self.imagen, z)
        self.p_esp.lienzo.mostrar(self.espectro, z)
        self.p_rec.lienzo.mostrar(self.reconstruida, z)

    def _cambio_zoom(self, val):
        self.lbl_zoom.setText(f"{val} %")
        self._redibujar()

    def _zoom_ajustar(self):
        if self.imagen is None:
            return
        h, w = self.imagen.shape[:2]
        # el panel más angosto manda, así las tres vistas entran sin scroll
        z = min(min(p.scroll.viewport().width() / w, p.scroll.viewport().height() / h)
                for p in (self.p_img, self.p_esp, self.p_rec)) * 100
        self.sl_zoom.setValue(int(max(10, min(800, z))))

    def _cambio_figura(self):
        """Cambia las etiquetas de los parámetros según la figura elegida."""
        nombre, p1, p2 = ops.FIGURAS[self.cb_figura.currentData()]
        self.lbl_p1.setText(p1 + ":")
        self.lbl_p2.setText((p2 or "—") + ":")
        self.sp_p2.setEnabled(p2 is not None)

    def _nueva_entrada(self, imagen, origen):
        """Cambia la imagen de entrada y descarta la TF anterior."""
        self.imagen = imagen
        self.lbl_archivo.setText(origen)
        self.datos = self.amplitud = self.fase = None
        self.espectro = self.reconstruida = None
        self.p_esp.titulo.setText("<b>Espectro</b>")
        self.p_rec.titulo.setText("<b>Reconstrucción</b>")
        self._redibujar()
        self._actualizar_controles()
        self._actualizar_estimacion()

    def _hover(self, x, y):
        lineas = [f"(x, y) = ({x}, {y})"]
        if self.imagen is not None and y < self.imagen.shape[0] and x < self.imagen.shape[1]:
            lineas.append("Imagen:  RGB(%3d, %3d, %3d)" % tuple(self.imagen[y, x]))
        if self.amplitud is not None and y < self.amplitud.shape[0] and x < self.amplitud.shape[1]:
            alto, ancho = self.amplitud.shape
            if self.datos["centrado"]:
                u, v = y - alto // 2, x - ancho // 2
            else:
                u, v = y, x
            lineas.append(f"Frecuencia (u, v) = ({u}, {v})")
            lineas.append(f"|F| = {self.amplitud[y, x]:.4g}   fase = {self.fase[y, x]:+.3f} rad")
        if self.reconstruida is not None and y < self.reconstruida.shape[0] and x < self.reconstruida.shape[1]:
            lineas.append("Recons.: RGB(%3d, %3d, %3d)" % tuple(self.reconstruida[y, x]))
        self.lbl_pixel.setText("\n".join(lineas))

    def _mostrar_espectro(self):
        """Arma la imagen del panel del medio según «Mostrar»."""
        if self.amplitud is None:
            return
        vista = self.cb_vista.currentIndex()
        if vista == 2:
            gris = llamar("fase_a_bitmap", self.fase)
            titulo = "fase (−π negro … +π blanco)"
        else:
            gris, _, _ = llamar("amplitud_a_bitmap", self.amplitud, log=(vista == 0))
            titulo = "amplitud, log(1 + |F|)" if vista == 0 else "amplitud lineal"
        self.espectro = llamar("gris_a_rgb", gris)
        self.p_esp.titulo.setText(f"<b>Espectro</b> — {titulo}")
        self._redibujar()
        self._actualizar_controles()

    # ------------------------------------------------------------ entrada
    def abrir(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", FILTRO_IMAGENES)
        if ruta:
            self.cargar(ruta)

    def cargar(self, ruta):
        def f():
            img = llamar("abrir_imagen", ruta)
            if not isinstance(img, QImage) or img.isNull():
                raise ValueError("No se pudo abrir la imagen.")
            arr = llamar("reducir", qimage_a_array(img), self.sp_ancho.value())
            self.figura, self.ruta = None, ruta
            h, w = arr.shape[:2]
            self._nueva_entrada(arr, f"{os.path.basename(ruta)} — {w} × {h} px")
        if self._ejecutar(f):
            self._zoom_ajustar()

    def dibujar(self):
        tipo = self.cb_figura.currentData()
        n = self.sp_tamano.value()
        base = None
        if self.chk_agregar.isChecked() and self.figura is not None and self.figura.shape == (n, n):
            base = self.figura

        def f():
            gris = llamar("dibujar_figura", tipo, n, self.sp_p1.value(), self.sp_p2.value(),
                          self.sp_dx.value(), self.sp_dy.value(), self.sp_angulo.value(), base)
            self.figura, self.ruta = gris, None
            origen = "varias figuras" if base is not None else ops.FIGURAS[tipo][0].lower()
            self._nueva_entrada(llamar("gris_a_rgb", gris), f"{origen} — {n} × {n} px")
        if self._ejecutar(f):
            self._zoom_ajustar()

    # ------------------------------------------------------------ transformada
    def tf_directa(self):
        if self.imagen is None:
            return

        metodo = self.cb_metodo.currentData()
        if not self._confirmar_tiempo(metodo):
            return

        def f():
            rgb = llamar("normalizar", self.imagen)
            self.datos = llamar("transformada_directa", rgb,
                                self.chk_media.isChecked(), self.chk_centrar.isChecked(),
                                metodo, self.cb_norm.currentData(), self._progreso("TF directa"))
            self.amplitud, self.fase = llamar("amplitud_y_fase", self.datos["F"])
            self.reconstruida = None
            self.p_rec.titulo.setText("<b>Reconstrucción</b>")
            self._mostrar_espectro()
        self._ejecutar(f)

    def tf_inversa(self):
        if self.datos is None:
            return
        if not self._confirmar_tiempo(self.datos["metodo"]):
            return

        def f():
            rgb = llamar("transformada_inversa", self.datos, None, self._progreso("TF inversa"))
            self.reconstruida = llamar("a_bytes", rgb)
            self.p_rec.titulo.setText("<b>Reconstrucción</b> — TF inversa")
            self._redibujar()
            self._actualizar_controles()
        self._ejecutar(f)

    # ------------------------------------------------------------ amplitud editable
    def guardar_amplitud(self):
        """Guarda log(1 + |F|) en 8 bits (PNG) y en un JSON el rango para volver atrás."""
        if self.datos is None:
            return
        base = os.path.splitext(self.ruta or "figura")[0] + "_amplitud.png"
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar amplitud", base, "PNG (*.png);;BMP (*.bmp)")
        if not ruta:
            return

        def f():
            bitmap, l_min, l_max = llamar("amplitud_a_bitmap", self.amplitud, log=True)
            if not llamar("guardar_imagen", array_a_qimage(llamar("gris_a_rgb", bitmap)), ruta):
                raise IOError("No se pudo guardar el bitmap.")
            alto, ancho = bitmap.shape
            meta = {
                "shape": [alto, ancho],
                "log_scale": True,
                "Lmin": l_min,
                "Lmax": l_max,
                "centrado": self.datos["centrado"],
                "restar_media": self.datos["restar_media"],
                "norm": self.datos["norm"],
                "nota": "Amplitud guardada como log(1 + |F|) llevada a 0..255. "
                        "Editá la imagen sin cambiarle el tamaño y volvé a cargarla.",
            }
            llamar("guardar_metadatos", meta, os.path.splitext(ruta)[0] + ".json")
        if self._ejecutar(f):
            self.statusBar().showMessage(f"Amplitud guardada: {ruta} (y su .json)")

    def cargar_amplitud(self):
        """Lee el bitmap editado, vuelve a la amplitud y reconstruye con la fase original."""
        if self.datos is None:
            return
        ruta, _ = QFileDialog.getOpenFileName(self, "Cargar amplitud editada", "", FILTRO_IMAGENES)
        if not ruta:
            return
        ruta_meta = os.path.splitext(ruta)[0] + ".json"
        if not os.path.exists(ruta_meta):
            ruta_meta, _ = QFileDialog.getOpenFileName(self, "Elegí el .json de esa amplitud", "",
                                                       "JSON (*.json)")
            if not ruta_meta:
                return
        if self._confirmar_tiempo(self.datos["metodo"]):
            self.reconstruir_desde(ruta, ruta_meta)

    def reconstruir_desde(self, ruta, ruta_meta):
        def f():
            meta = llamar("leer_metadatos", ruta_meta)
            img = llamar("abrir_imagen", ruta)
            if not isinstance(img, QImage) or img.isNull():
                raise ValueError("No se pudo abrir el bitmap.")
            bitmap = qimage_a_array(img)
            if list(bitmap.shape[:2]) != list(self.amplitud.shape):
                raise ValueError(f"El bitmap mide {bitmap.shape[1]} × {bitmap.shape[0]} y el espectro "
                                 f"{self.amplitud.shape[1]} × {self.amplitud.shape[0]}: no se puede usar.")
            if bool(meta.get("centrado", True)) != self.datos["centrado"]:
                raise ValueError("El bitmap se guardó con otro ajuste de «Centrar el espectro».")
            if meta.get("norm", "numpy") != self.datos["norm"]:
                raise ValueError("El bitmap se guardó con otra normalización.")
            amplitud = llamar("amplitud_editada", self.amplitud, bitmap, meta["Lmin"], meta["Lmax"],
                              log=bool(meta.get("log_scale", True)))
            f_nueva = llamar("desde_amplitud_y_fase", amplitud, self.fase)
            rgb = llamar("transformada_inversa", self.datos, f_nueva, self._progreso("TF inversa"))
            self.reconstruida = llamar("a_bytes", rgb)
            self.espectro = bitmap
            self.p_esp.titulo.setText(f"<b>Espectro</b> — amplitud editada ({os.path.basename(ruta)})")
            self.p_rec.titulo.setText("<b>Reconstrucción</b> — amplitud editada + fase original")
            self._redibujar()
            self._actualizar_controles()
        self._ejecutar(f)

    # ------------------------------------------------------------ guardar vistas
    def _guardar_array(self, arr, sufijo, titulo):
        base = os.path.splitext(self.ruta or "figura")[0] + sufijo
        ruta, _ = QFileDialog.getSaveFileName(self, titulo, base,
                                              "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;Todos (*)")
        if not ruta:
            return
        try:
            ok = llamar("guardar_imagen", array_a_qimage(arr), ruta)
        except Exception as e:
            return self._error(e)
        self.statusBar().showMessage(f"Guardado: {ruta}" if ok else "No se pudo guardar")

    def guardar_reconstruccion(self):
        if self.reconstruida is not None:
            self._guardar_array(self.reconstruida, "_reconstruida.png", "Guardar reconstrucción")

    def guardar_espectro(self):
        if self.espectro is not None:
            self._guardar_array(self.espectro, "_espectro.png", "Guardar vista del espectro")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = VentanaPrincipal()
    v.show()
    sys.exit(app.exec())
