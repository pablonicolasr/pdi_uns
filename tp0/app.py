"""
app.py — Interfaz gráfica (PySide6).
Ejecutar con:   python app.py
Las operaciones se implementan en operaciones.py
"""

import sys
import os

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QColor, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QScrollArea, QPushButton,
    QSpinBox, QGroupBox, QFormLayout, QVBoxLayout, QHBoxLayout, QFileDialog,
    QMessageBox, QColorDialog, QSlider, QFrame, QSizePolicy,
)

import operaciones as ops

FILTRO_IMAGENES = "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif *.ppm *.pgm *.tif *.tiff);;Todos (*)"


# Widgets auxiliares
class LienzoImagen(QLabel):
    """Muestra la imagen con zoom y emite la coordenada del pixel bajo el mouse."""
    pixel_clic = Signal(int, int)
    pixel_hover = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)
        self._imagen = None
        self._zoom = 1.0
        self.setText("Abrí una imagen para empezar")
        self.setStyleSheet("color: gray;")

    def mostrar(self, imagen: QImage | None, zoom: float):
        self._imagen, self._zoom = imagen, zoom
        if imagen is None or imagen.isNull():
            self.clear()
            self.setText("Abrí una imagen para empezar")
            self.adjustSize()
            return
        self.setStyleSheet("")
        pix = QPixmap.fromImage(imagen)
        if zoom != 1.0:
            pix = pix.scaled(int(imagen.width() * zoom), int(imagen.height() * zoom),
                             Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.setPixmap(pix)
        self.resize(pix.size())

    def _a_pixel(self, pos: QPoint):
        if self._imagen is None or self._imagen.isNull():
            return None
        x, y = int(pos.x() / self._zoom), int(pos.y() / self._zoom)
        if 0 <= x < self._imagen.width() and 0 <= y < self._imagen.height():
            return x, y
        return None

    def mousePressEvent(self, e):
        p = self._a_pixel(e.position().toPoint())
        if p and e.button() == Qt.LeftButton:
            self.pixel_clic.emit(*p)

    def mouseMoveEvent(self, e):
        p = self._a_pixel(e.position().toPoint())
        if p:
            self.pixel_hover.emit(*p)


class MuestraColor(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(48, 28)
        self.setFrameShape(QFrame.Box)
        self.set_color(None)

    def set_color(self, rgb):
    
        if rgb is None:
            self.setStyleSheet(
                "background: transparent;" 
                "border: 1px dashed gray;"
            )
        else:
        
            r, g, b = map(int, rgb)
            
            self.setStyleSheet(
                f"""
                background-color: rgb({r}, {g}, {b});
                border: 1px solid #444;
                """
            )


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Imágenes")
        self.resize(1100, 700)

        self.imagen: QImage | None = None      # imagen de trabajo
        self.original: QImage | None = None    # copia para "Restaurar"
        self.ruta: str | None = None
        self.modificada = False

        self._crear_acciones()
        self._crear_ui()
        self._actualizar_estado_controles()

    # ---------------- construcción de la interfaz ----------------
    def _crear_acciones(self):
        self.act_abrir = QAction("Abrir imagen...", self, shortcut=QKeySequence.Open, triggered=self.abrir)
        self.act_guardar = QAction("Guardar", self, shortcut=QKeySequence.Save, triggered=self.guardar)
        self.act_guardar_como = QAction("Guardar como...", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                                        triggered=self.guardar_como)
        self.act_restaurar = QAction("Restaurar original", self, shortcut=QKeySequence("Ctrl+R"),
                                     triggered=self.restaurar)
        self.act_salir = QAction("Salir", self, shortcut=QKeySequence.Quit, triggered=self.close)

        m = self.menuBar().addMenu("&Archivo")
        for a in (self.act_abrir, self.act_guardar, self.act_guardar_como):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_salir)
        self.menuBar().addMenu("&Editar").addAction(self.act_restaurar)

        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        for a in (self.act_abrir, self.act_guardar, self.act_guardar_como, self.act_restaurar):
            tb.addAction(a)

    def _crear_ui(self):
        # --- área de imagen ---
        self.lienzo = LienzoImagen()
        self.lienzo.pixel_clic.connect(self._clic_en_imagen)
        self.lienzo.pixel_hover.connect(
            lambda x, y: self.statusBar().showMessage(f"Pixel bajo el mouse: ({x}, {y})"))
        scroll = QScrollArea()
        scroll.setWidget(self.lienzo)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet("QScrollArea { background: #2b2b2b; }")

        # --- panel lateral ---
        panel = QVBoxLayout()

        # Info
        g_info = QGroupBox("Imagen")
        f = QFormLayout(g_info)
        self.lbl_archivo = QLabel("—")
        self.lbl_archivo.setWordWrap(True)
        self.lbl_tamano = QLabel("—")
        f.addRow("Archivo:", self.lbl_archivo)
        f.addRow("Tamaño:", self.lbl_tamano)
        panel.addWidget(g_info)

        # Zoom
        g_zoom = QGroupBox("Zoom")
        hz = QHBoxLayout(g_zoom)
        self.sl_zoom = QSlider(Qt.Horizontal, minimum=10, maximum=2000, value=100)
        self.lbl_zoom = QLabel("100 %")
        self.lbl_zoom.setMinimumWidth(50)
        self.sl_zoom.valueChanged.connect(self._cambio_zoom)
        hz.addWidget(self.sl_zoom)
        hz.addWidget(self.lbl_zoom)
        panel.addWidget(g_zoom)

        # Pixel
        g_pix = QGroupBox("Pixel")
        v = QVBoxLayout(g_pix)
        fx = QFormLayout()
        self.sp_x, self.sp_y = QSpinBox(), QSpinBox()
        fx.addRow("X:", self.sp_x)
        fx.addRow("Y:", self.sp_y)
        v.addLayout(fx)
        v.addWidget(QLabel("Tip: hacé clic en la imagen para elegir el pixel."))

        self.btn_leer = QPushButton("Leer pixel")
        self.btn_leer.clicked.connect(self.leer_pixel)
        v.addWidget(self.btn_leer)
        h = QHBoxLayout()
        h.addWidget(QLabel("Color actual:"))
        self.muestra_actual = MuestraColor()
        h.addWidget(self.muestra_actual)
        self.lbl_rgb_actual = QLabel("R — G — B —")
        h.addWidget(self.lbl_rgb_actual, 1)
        v.addLayout(h)
        panel.addWidget(g_pix)

        # Nuevo color
        g_col = QGroupBox("Nuevo color")
        v2 = QVBoxLayout(g_col)
        hc = QHBoxLayout()
        self.sp_r, self.sp_g, self.sp_b = (QSpinBox(maximum=255) for _ in range(3))
        self.sp_r.setValue(255)
        for nombre, sp in (("R", self.sp_r), ("G", self.sp_g), ("B", self.sp_b)):
            hc.addWidget(QLabel(nombre))
            hc.addWidget(sp)
            sp.valueChanged.connect(self._refrescar_muestra_nueva)
        v2.addLayout(hc)
        hc2 = QHBoxLayout()
        self.muestra_nueva = MuestraColor()
        btn_paleta = QPushButton("Elegir color...")
        btn_paleta.clicked.connect(self._elegir_color)
        btn_copiar = QPushButton("Usar color actual")
        btn_copiar.clicked.connect(self._copiar_color_actual)
        hc2.addWidget(self.muestra_nueva)
        hc2.addWidget(btn_paleta)
        hc2.addWidget(btn_copiar)
        v2.addLayout(hc2)
        self.btn_modificar = QPushButton("Modificar pixel")
        self.btn_modificar.setStyleSheet("font-weight: bold;")
        self.btn_modificar.clicked.connect(self.modificar_pixel)
        v2.addWidget(self.btn_modificar)
        panel.addWidget(g_col)
        self._refrescar_muestra_nueva()

        # Operación libre
        g_op = QGroupBox("Toda la imagen (opcional)")
        v3 = QVBoxLayout(g_op)
        self.btn_operacion = QPushButton("Invertir pixeles")
        self.btn_operacion.clicked.connect(self.aplicar_operacion)
        v3.addWidget(self.btn_operacion)
        panel.addWidget(g_op)

        
        # Invertir imagen
        g_inv = QGroupBox("Toda la imagen (Invertir imagen)")
        v3 = QVBoxLayout(g_inv)
        self.btn_operacion = QPushButton("Invertir imagen")
        self.btn_operacion.clicked.connect(self.invertir_imagen)
        v3.addWidget(self.btn_operacion)
        panel.addWidget(g_inv) 
        

        panel.addStretch(1)

        lateral = QWidget()
        lateral.setLayout(panel)
        lateral.setFixedWidth(320)

        central = QWidget()
        hl = QHBoxLayout(central)
        hl.addWidget(scroll, 1)
        hl.addWidget(lateral)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Listo")

    # ---------------- utilidades de interfaz ----------------
    def _hay_imagen(self):
        return self.imagen is not None and not self.imagen.isNull()

    def _actualizar_estado_controles(self):
        hay = self._hay_imagen()
        for w in (self.act_guardar, self.act_guardar_como, self.act_restaurar):
            w.setEnabled(hay)
        for w in (self.btn_leer, self.btn_modificar, self.btn_operacion, self.sp_x, self.sp_y):
            w.setEnabled(hay)
        if hay:
            self.sp_x.setRange(0, self.imagen.width() - 1)
            self.sp_y.setRange(0, self.imagen.height() - 1)
            self.lbl_tamano.setText(f"{self.imagen.width()} × {self.imagen.height()} px")
            self.lbl_archivo.setText(os.path.basename(self.ruta) if self.ruta else "(sin guardar)")
        titulo = "Procesamiento de Imágenes"
        if self.ruta:
            titulo += f" — {os.path.basename(self.ruta)}{' *' if self.modificada else ''}"
        self.setWindowTitle(titulo)

    def _redibujar(self):
        self.lienzo.mostrar(self.imagen, self.sl_zoom.value() / 100)

    def _cambio_zoom(self, valor):
        self.lbl_zoom.setText(f"{valor} %")
        self._redibujar()

    def _refrescar_muestra_nueva(self):
        self.muestra_nueva.set_color(self._color_nuevo())

    def _color_nuevo(self):
        return (self.sp_r.value(), self.sp_g.value(), self.sp_b.value())

    def _elegir_color(self):
        c = QColorDialog.getColor(QColor(*self._color_nuevo()), self, "Elegir color")
        if c.isValid():
            self.sp_r.setValue(c.red()); self.sp_g.setValue(c.green()); self.sp_b.setValue(c.blue())

    def _copiar_color_actual(self):
        rgb = getattr(self, "_rgb_actual", None)
        if rgb:
            self.sp_r.setValue(rgb[0]); self.sp_g.setValue(rgb[1]); self.sp_b.setValue(rgb[2])

    def _clic_en_imagen(self, x, y):
        self.sp_x.setValue(x)
        self.sp_y.setValue(y)
        self.leer_pixel(silencioso=True)

    def _falta(self, nombre):
        QMessageBox.information(self, "Falta implementar",
                                f"La función <b>{nombre}()</b> de <i>operaciones.py</i> todavía no está implementada.")

    def _error(self, nombre, exc):
        QMessageBox.critical(self, "Error", f"Error en {nombre}():\n{type(exc).__name__}: {exc}")

    # ---------------- acciones (llaman a operaciones.py) ----------------
    def abrir(self):
        if not self._confirmar_descartar():
            return
        ruta, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", FILTRO_IMAGENES)
        if not ruta:
            return
        try:
            img = ops.abrir_imagen(ruta)
        except NotImplementedError:
            return self._falta("abrir_imagen")
        except Exception as e:
            return self._error("abrir_imagen", e)
        if img is None or not isinstance(img, QImage) or img.isNull():
            QMessageBox.warning(self, "Abrir", "No se pudo abrir la imagen.")
            return
        self.imagen, self.original, self.ruta, self.modificada = img, img.copy(), ruta, False
        self.sl_zoom.setValue(100)
        self._redibujar()
        self._actualizar_estado_controles()
        self.statusBar().showMessage(f"Abierta: {ruta}")

    def guardar(self):
        if self.ruta:
            self._guardar_en(self.ruta)
        else:
            self.guardar_como()

    def guardar_como(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar imagen como", self.ruta or "imagen.png",
                                              "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;Todos (*)")
        if ruta:
            self._guardar_en(ruta)

    def _guardar_en(self, ruta):
        try:
            ok = ops.guardar_imagen(self.imagen, ruta)
        except NotImplementedError:
            return self._falta("guardar_imagen")
        except Exception as e:
            return self._error("guardar_imagen", e)
        if ok:
            self.ruta, self.modificada = ruta, False
            self._actualizar_estado_controles()
            self.statusBar().showMessage(f"Guardada: {ruta}")
        else:
            QMessageBox.warning(self, "Guardar", "No se pudo guardar la imagen.")

    def leer_pixel(self, silencioso=False):
        x, y = self.sp_x.value(), self.sp_y.value()
        try:
            r, g, b = ops.leer_pixel(self.imagen, x, y)
        except NotImplementedError:
            if not silencioso:
                self._falta("leer_pixel")
            self.statusBar().showMessage(f"Pixel seleccionado: ({x}, {y})")
            return
        except Exception as e:
            return self._error("leer_pixel", e)
        self._rgb_actual = (r, g, b)
        self.muestra_actual.set_color((r, g, b))
        self.lbl_rgb_actual.setText(f"R {r}  G {g}  B {b}")
        self.statusBar().showMessage(f"Pixel ({x}, {y}) = RGB({r}, {g}, {b})")

    def modificar_pixel(self):
        x, y = self.sp_x.value(), self.sp_y.value()
        try:
            ops.modificar_pixel(self.imagen, x, y, self._color_nuevo())
        except NotImplementedError:
            return self._falta("modificar_pixel")
        except Exception as e:
            return self._error("modificar_pixel", e)
        self.modificada = True
        self._redibujar()
        self._actualizar_estado_controles()
        self.leer_pixel(silencioso=True)

    def aplicar_operacion(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = ops.procesar_imagen(self.imagen)
        except NotImplementedError:
            QApplication.restoreOverrideCursor()
            return self._falta("procesar_imagen")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            return self._error("procesar_imagen", e)
        QApplication.restoreOverrideCursor()
        if isinstance(res, QImage) and not res.isNull():
            self.imagen = res
        self.modificada = True
        self._redibujar()
        self._actualizar_estado_controles()
    
    
    def invertir_imagen(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            res = ops.invertir_imagen(self.imagen)

            if isinstance(res, QImage) and not res.isNull():
                self.imagen = res
                self.modificada = True
                self._redibujar()
                self._actualizar_estado_controles()

        except NotImplementedError:
            return self._falta("invertir_imagen")

        except Exception as e:
            return self._error("invertir_imagen", e)

        finally:
            QApplication.restoreOverrideCursor()

    def restaurar(self):
        if self.original is not None:
            self.imagen = self.original.copy()
            self.modificada = False
            self._redibujar()
            self._actualizar_estado_controles()
            self.statusBar().showMessage("Imagen restaurada al original")

    def _confirmar_descartar(self):
        if not self.modificada:
            return True
        r = QMessageBox.question(self, "Cambios sin guardar",
                                 "La imagen tiene cambios sin guardar. ¿Descartarlos?")
        return r == QMessageBox.Yes

    def closeEvent(self, e):
        e.accept() if self._confirmar_descartar() else e.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    v = VentanaPrincipal()
    v.show()
    sys.exit(app.exec())
