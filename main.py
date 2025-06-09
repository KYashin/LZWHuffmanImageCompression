import struct
import sys
import os
import cv2
import numpy as np
from time import time

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QGroupBox, QFrame
)
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication

import LZW_with_gray as LZW_gray
import LZW_with_color as LZW_color
import LZW_with_Huffman as LZW_Huffman


class CompressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LZW Image Compression")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QWidget {
                font-family: Segoe UI;
                font-size: 14px;
            }
            QPushButton {
                padding: 6px 12px;
                border-radius: 6px;
                background-color: #2e7d32;
                color: white;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QLabel#titleLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 8px;
            }
            QTextEdit {
                background-color: #f5f5f5;
                border-radius: 6px;
                padding: 6px;
            }
            QCheckBox {
                padding: 2px;
            }
            QGroupBox {
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 10px;
                margin: 4px;
            }
        """)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.file_path = None
        self.image = None
        self.output_dir_color = 'compressed_color_planes'
        self.restored_dir_color = 'RestoredImages'
        self.output_dir_gray = 'compressed_gray_planes'
        self.restored_dir_gray = 'RestoredGrayImages'

        # ЗАГОЛОВОК
        title = QLabel("LZW Image Compressor")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Изображение
        self.image_label = QLabel('There will be an image here')
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid #bbb; background-color: #eee;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Кнопки
        self.load_btn = QPushButton('Load image')
        self.compress_btn = QPushButton('Compress')

        # Настройки
        self.color_check = QCheckBox('Color image')

        options_box = QGroupBox('Settings')
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.color_check)
        options_box.setLayout(options_layout)

        # Лог
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        menubar = self.menuBar()
        edit_menu = menubar.addMenu('Edit')

        clear_log_action = QAction('Clear log', self)
        clear_log_action.triggered.connect(self.clear_log)
        edit_menu.addAction(clear_log_action)

        # Раскладка
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(options_box)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.compress_btn)
        right_layout.addWidget(self.log_output)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        content_layout = QHBoxLayout()
        content_layout.addLayout(left_layout)
        content_layout.addLayout(right_layout)
        main_layout.addLayout(content_layout)

        central_widget.setLayout(main_layout)

        self.color_check.setDisabled(True)
        self.compress_btn.setDisabled(True)

        self.load_btn.clicked.connect(self.load_image)
        self.compress_btn.clicked.connect(self.start_compression_log)

    def load_image(self):
        app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        images_dir = os.path.join(app_dir, 'Images')

        # Открыть диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
                self, "Open image", images_dir, "Image Files (*.jpg *.bmp *.jpeg *.webp *.tif *.tiff);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.log_output.append(f"Downloaded image: {file_path}\n")

            # Разблокируем кнопки и чекбоксы
            self.load_btn.setEnabled(True)
            self.compress_btn.setEnabled(True)
            self.color_check.setEnabled(True)

    def start_compression_log(self):
        if self.color_check.isChecked():
            self.log_output.append('The color image compression is started...\n')
        else:
            self.log_output.append('The grayscale image compression is started...\n')

        # Обновляем GUI
        QApplication.processEvents()

        # Запланировать вызов сжатия через 0 мс
        QTimer.singleShot(0, self.compress_image)

    def clear_log(self):
        self.log_output.clear()

    def compress_image(self):
        if self.color_check.isChecked():
            self.compress_color_image()
        else:
            self.compress_gray_image()

    def compress_color_image(self):
        self.image = cv2.imread(self.file_path, cv2.IMREAD_COLOR)
        start = time()
        channels = cv2.split(self.image)

        # Сохраняем bit-planes в .lzw файлы
        for ch_idx, channel in enumerate(channels):
            for i in range(8):
                plane = ((channel >> i) & 1).astype(np.uint8)
                bit_string = ''.join(str(b) for b in plane.flatten())

                compressed = LZW_color.lzw_compress(bit_string)
                packed = LZW_color.pack_12bit_codes(compressed)

                out_path = os.path.join(self.self.output_dir_color, f"ch{ch_idx}_plane{i}.lzw")
                with open(out_path, 'wb') as f:
                    f.write(struct.pack('II', *plane.shape))  # height, width
                    f.write(struct.pack('I', len(compressed)))  # кол-во 12-битных кодов
                    f.write(struct.pack('I', len(packed)))  # длина packed байт
                    f.write(struct.pack('I', len(bit_string)))  # длина строки бит
                    f.write(packed)

        restored_channels = []

        for ch_idx in range(3):
            restored_bit_planes = []

            for i in range(8):
                in_path = os.path.join(self.output_dir_color, f'ch{ch_idx}_plane{i}.lzw')
                with open(in_path, 'rb') as f:
                    height, width = struct.unpack('II', f.read(8))
                    shape = (height, width)

                    num_codes = struct.unpack('I', f.read(4))[0]
                    packed_len = struct.unpack('I', f.read(4))[0]
                    bit_string_len = struct.unpack('I', f.read(4))[0]

                    packed = f.read(packed_len)

                unpacked = LZW_color.unpack_12bit_codes(packed, num_codes)
                decompressed = LZW_color.lzw_decompress(unpacked)[:bit_string_len]
                restored = LZW_color.bit_string_to_image(decompressed, shape)
                restored_bit_planes.append(restored)

            restored_channel = np.zeros_like(restored_bit_planes[0], dtype=np.uint8)
            for i in range(8):
                restored_channel += (restored_bit_planes[i] << i)

            restored_channels.append(restored_channel)

        # === СОБИРАЕМ ЦВЕТНОЕ ИЗОБРАЖЕНИЕ ===
        restored_color = cv2.merge(restored_channels)
        cv2.imwrite(os.path.join(self.restored_dir_color, 'reconstructed_color.tif'), restored_color)

        original_size = self.image.nbytes
        compressed_size = sum(
            os.path.getsize(os.path.join(self.output_dir_color, f))
            for f in os.listdir(self.output_dir_color)
            if f.endswith('.lzw')
        )
        ratio = original_size / compressed_size
        end = time()

        self.log_output.append(f'Compression execution time: {(end - start):.6f} sec')
        self.log_output.append(f'Original image size: {original_size} bytes')
        self.log_output.append(f'Compressed planes size: {compressed_size} bytes')
        self.log_output.append(f'Compress ratio: {ratio:.11f}\n\n')

    def compress_gray_image(self):
        self.image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        start = time()

        for i in range(8):
            plane = ((self.image >> i) & 1).astype(np.uint8)
            bit_string = ''.join(str(b) for b in plane.flatten())

            compressed = LZW_gray.lzw_compress(bit_string)
            packed = LZW_gray.pack_12bit_codes(compressed)

            out_path = os.path.join(self.output_dir_gray, f"plane{i}.lzw")
            with open(out_path, 'wb') as f:
                f.write(struct.pack('II', *plane.shape))  # height, width
                f.write(struct.pack('I', len(compressed)))  # кол-во 12-битных кодов
                f.write(struct.pack('I', len(packed)))  # длина packed байт
                f.write(struct.pack('I', len(bit_string)))  # длина строки бит
                f.write(packed)

        # === ВОССТАНОВЛЕНИЕ ===

        restored_bit_planes = []

        for i in range(8):
            in_path = os.path.join(self.output_dir_gray, f"plane{i}.lzw")
            with open(in_path, 'rb') as f:
                height, width = struct.unpack('II', f.read(8))
                shape = (height, width)

                num_codes = struct.unpack('I', f.read(4))[0]
                packed_len = struct.unpack('I', f.read(4))[0]
                bit_string_len = struct.unpack('I', f.read(4))[0]

                packed = f.read(packed_len)

            unpacked = LZW_gray.unpack_12bit_codes(packed, num_codes)
            decompressed = LZW_gray.lzw_decompress(unpacked)[:bit_string_len]
            restored = LZW_gray.bit_string_to_image(decompressed, shape)
            restored_bit_planes.append(restored)

        # === СОБИРАЕМ ГРАДАЦИИ СЕРОГО ===

        restored_img = np.zeros_like(restored_bit_planes[0], dtype=np.uint8)
        for i in range(8):
            restored_img += (restored_bit_planes[i] << i)

        cv2.imwrite(os.path.join(self.restored_dir_gray, "reconstructed_gray.tif"), restored_img)

        original_size = self.image.nbytes
        compressed_size = sum(
            os.path.getsize(os.path.join(self.output_dir_gray, f))
            for f in os.listdir(self.output_dir_gray)
            if f.endswith('.lzw')
        )
        ratio = original_size / compressed_size
        end = time()

        self.log_output.append(f'Compression execution time: {(end - start):.6f} sec')
        self.log_output.append(f'Original image size: {original_size} bytes')
        self.log_output.append(f'Compressed planes size: {compressed_size} bytes')
        self.log_output.append(f'Compress ratio: {ratio:.11f}\n\n')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompressionApp()
    window.show()
    sys.exit(app.exec())