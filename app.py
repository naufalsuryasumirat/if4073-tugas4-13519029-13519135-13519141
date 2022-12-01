import sys
import numpy as np
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import src.fruits_classification_cnn as cls_cnn_model
import src.fruits_classification_improc as improc
from PIL import Image, ImageQt
from matplotlib.cm import get_cmap
from PyQt5.QtWidgets import (
  QApplication, QPushButton, QMainWindow, QLabel, QTextEdit, QFileDialog)
from PyQt5.QtGui import QFont, QPixmap

class Window(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setStyleSheet('background-color: #F4F2E9')
    self.setWindowTitle('Klasifikasi Buah')
    self.setGeometry(0, 0, 1400, 800)
    self.setMinimumSize(1000, 800)
    # self.setFixedSize(1400, 800)

    # Labels
    self._lbl_title = QLabel('IF4073-Tugas4-Klasifikasi Buah', self)
    self._lbl_title.move(10, 10)
    self._lbl_title.resize(700, 35)
    self._lbl_title.setFont(QFont('AnyStyle', 20, 500))
    self._lbl_title.setStyleSheet('color: #372213;')
    
    self._lbl_input = QLabel('Input', self)
    self._lbl_input.move(10, 40)
    self._lbl_input.setFont(QFont('AnyStyle', 14, 30))
    self._lbl_input.setStyleSheet('color: #372213;')
    
    self._lbl_dl_info = QLabel('Deep Learning Model Prediction (VGG 16 Transfer Learning w/ ImageNet weights)', self)
    self._lbl_dl_info.move(10, 600)
    self._lbl_dl_info.resize(800, 35)
    self._lbl_dl_info.setFont(QFont('AnyStyle', 14, 100))
    self._lbl_dl_info.setStyleSheet('color: #372213;')
    
    self._lbl_dl_predict = QLabel('Predicted Label: ', self)
    self._lbl_dl_predict.move(10, 650)
    self._lbl_dl_predict.resize(150, 35)
    self._lbl_dl_predict.setFont(QFont('Roboto', 14, 10))
    self._lbl_dl_predict.setStyleSheet('color: #372213;')
    
    self._lbl_dl_predict_res = QLabel('test_label', self)
    self._lbl_dl_predict_res.move(180, 650)
    self._lbl_dl_predict_res.resize(400, 35)
    self._lbl_dl_predict_res.setFont(QFont('Courier', 12, 5))
    self._lbl_dl_predict_res.setStyleSheet('color: #372213;')
    self._lbl_dl_predict_res.setWordWrap(True)
    
    self._lbl_dl_confidence = QLabel('Confidence: ', self)
    self._lbl_dl_confidence.move(10, 675)
    self._lbl_dl_confidence.resize(100, 100)
    self._lbl_dl_confidence.setFont(QFont('Roboto', 14, 10))
    self._lbl_dl_confidence.setStyleSheet('color: #372213;')
    self._lbl_dl_confidence.setWordWrap(True)
    
    self._lbl_dl_confidence_res = QLabel('test_conf', self)
    self._lbl_dl_confidence_res.move(120, 675)
    self._lbl_dl_confidence_res.resize(700, 100)
    self._lbl_dl_confidence_res.setFont(QFont('Courier', 12, 5))
    self._lbl_dl_confidence_res.setStyleSheet('color: #372213;')
    self._lbl_dl_confidence_res.setWordWrap(True)
    
    # Buttons
    self._btn_load = QPushButton(self)
    self._btn_load.move(465, 13)
    self._btn_load.clicked.connect(self.load_image)
    self._btn_load.setFont(QFont('Courier', 14, 100))
    self._btn_load.setText('Load Image')
    self._btn_load.setFixedHeight(30)
    self._btn_load.setFixedWidth(150)
    self._btn_load.setStyleSheet('QPushButton { background-color: #372213; color: #F4F2E9; }'
                               'QPushButton::pressed { background-color: #F4F2E9; color: #372213; }')

    self._btn_process = QPushButton(self)
    self._btn_process.move(630, 13)
    self._btn_process.clicked.connect(self.predict)
    self._btn_process.setFont(QFont('Courier', 14, 100))
    self._btn_process.setText('Predict')
    self._btn_process.setFixedHeight(30)
    self._btn_process.setFixedWidth(150)
    self._btn_process.setStyleSheet('QPushButton { background-color: #372213; color: #F4F2E9; }'
                               'QPushButton::pressed { background-color: #F4F2E9; color: #372213; }')
    
    # Images
    self._img_input = QLabel(self)
    self._img_input.move(10, 70)
    
    self._img_segmt = QLabel(self)
    self._img_segmt.move(276, 70)
    
    self._img_grays = QLabel(self)
    self._img_grays.move(542, 70)
    
    self._img_imhog = QLabel(self)
    self._img_imhog.move(808, 70)
    
    self._cur_filepath = None

    self.show()
  
  def load_image(self):
    filepath = QFileDialog.getOpenFileName(None, 'OpenFile', '', 'Image file (*.jpg; *.png; *.jpeg; *.bmp)')
    if not filepath[0]: return
    self._cur_filepath = filepath[0]
    inp_pixmap = QPixmap(self._cur_filepath)
    inp_pixmap = inp_pixmap.scaled(64*4, 128*4)
    self._img_input.resize(inp_pixmap.width(), inp_pixmap.height())
    self._img_input.setPixmap(inp_pixmap)
    return
  
  def predict(self):
    if not self._cur_filepath: return
    label, confidence = cls_cnn_model.predict_fruit(self._cur_filepath)
    self._lbl_dl_predict_res.setText(f"{label}")
    self._lbl_dl_confidence_res.setText(
      f"{'; '.join([f'{lbl}={conf*100:.2f}%' for lbl, conf in confidence])}")
    
    img = improc.load_image(self._cur_filepath)
    segmented = improc.proc_segment(img)
    segmented_show = ImageQt.ImageQt(Image.fromarray(segmented, mode='RGB'))
    self._img_segmt.resize(64*4, 128*4)
    self._img_segmt.setPixmap(QPixmap.fromImage(segmented_show))
    
    grayscale = improc.gray_image(segmented)
    grayscale_show = ImageQt.ImageQt(Image.fromarray(np.uint8(grayscale*255), mode='L'))
    self._img_grays.resize(64*4, 128*4)
    self._img_grays.setPixmap(QPixmap.fromImage(grayscale_show))
    
    cm_viridis = get_cmap('viridis')
    fd, hog_im = improc.proc_hog(grayscale)
    hog_im = cm_viridis(hog_im)
    hog_show = ImageQt.ImageQt(Image.fromarray(np.uint8(hog_im*255)))
    self._img_imhog.resize(64*4, 128*4)
    self._img_imhog.setPixmap(QPixmap.fromImage(hog_show))
    
    return

if __name__ == '__main__':
  app = QApplication(sys.argv)
  window = Window()
  sys.exit(app.exec_())
    
