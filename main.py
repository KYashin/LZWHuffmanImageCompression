import sys
import os

# # Добавляем в sys.path корень проекта, чтобы видеть другие .py-файлы
# root_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(root_dir)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QGroupBox, QFrame
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt
import cv2
# import LZW_with_color
# import LZW_with_Huffman


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

        # ЗАГОЛОВОК
        title = QLabel("LZW Image Compressor with Adaptive Huffman")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Изображение
        self.image_label = QLabel("Здесь будет изображение")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 1px solid #bbb; background-color: #eee;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Кнопки
        self.load_btn = QPushButton("Загрузить")
        self.load_btn.clicked.connect(self.load_image)

        self.compress_btn = QPushButton("Сжать")
        # compress_btn.clicked.connect(self.)

        self.decompress_btn = QPushButton("Восстановить")
        self.save_btn = QPushButton("Сохранить")

        # Настройки
        self.color_check = QCheckBox("Цветное изображение")
        self.huffman_check = QCheckBox("Адаптивный Хаффман")

        options_box = QGroupBox("Настройки")
        options_layout = QVBoxLayout()
        options_layout.addWidget(self.color_check)
        options_layout.addWidget(self.huffman_check)
        options_box.setLayout(options_layout)

        # Лог
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        # Раскладка
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(options_box)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.compress_btn)
        right_layout.addWidget(self.decompress_btn)
        right_layout.addWidget(self.save_btn)
        right_layout.addWidget(self.log_output)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        content_layout = QHBoxLayout()
        content_layout.addLayout(left_layout)
        content_layout.addLayout(right_layout)
        main_layout.addLayout(content_layout)

        central_widget.setLayout(main_layout)

        self.color_check.setDisabled(True)
        self.huffman_check.setDisabled(True)
        self.decompress_btn.setDisabled(True)
        self.compress_btn.setDisabled(True)
        self.save_btn.setDisabled(True)

    def load_image(self):
        app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        root_dir = os.path.dirname(app_dir)
        images_dir = os.path.join(root_dir, 'Images')

        # Открыть диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
                self, "Открыть изображение", images_dir, "Image Files (*.png *.jpg *.bmp *.jpeg *.webp *.tif);;All Files (*)"
        )
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.image_label.setPixmap(pixmap)
            self.log_output.append(f"Загружено изображение: {file_path}")

            # Разблокируем кнопки и чекбоксы
            self.load_btn.setEnabled(True)
            self.compress_btn.setEnabled(True)
            self.decompress_btn.setEnabled(True)
            self.color_check.setEnabled(True)
            self.huffman_check.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CompressionApp()
    window.show()
    sys.exit(app.exec())
