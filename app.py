import sys
import os
import torch
import librosa
import soundfile as sf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QTextEdit, QWidget, QFileDialog, QFrame
)
from PyQt6.QtCore import QTimer, Qt, QPointF
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QPen
import pyaudio
import wave
import numpy as np
from inference import inference  # 导入你的推理函数

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform = np.zeros(32000)
        self.setMinimumHeight(80)
        self.setStyleSheet("background-color: #f8f9fa; border-radius: 10px;")
    def update_waveform(self, data):
        self.waveform = data
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h // 2
        if self.waveform is not None and len(self.waveform) > 0:
            step = max(1, len(self.waveform) // w)
            points = [QPointF(x, mid - self.waveform[x * step] * mid) for x in range(w)]
            pen = QPen(QColor(52, 152, 219), 2)
            painter.setPen(pen)
            for i in range(1, len(points)):
                painter.drawLine(points[i-1], points[i])

class AudioCaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio subtitle generator")
        self.setGeometry(100, 100, 800, 400)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
            QTextEdit {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 13px;
                background-color: #f8f9fa;
            }
        """)
        
        # 音频参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 32000  # 与模型要求的采样率一致
        self.CHUNK = 1024
        self.RECORD_SECONDS = 10  # 默认录制10秒
        self.is_recording = False
        self.frames = []
        
        # 初始化音频流
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # 创建UI
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title_label = QLabel("Automatic Audio Captioning")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 20px 0;
        """)
        main_layout.addWidget(title_label)
        
        # 控制按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # 录制按钮
        self.record_btn = QPushButton("Start to record")
        self.record_btn.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_btn)
        
        # 文件选择按钮
        self.file_btn = QPushButton("Select the audio file")
        self.file_btn.clicked.connect(self.open_audio_file)
        button_layout.addWidget(self.file_btn)
        
        main_layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("Ready - Please select the audio file or start recording")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #7f8c8d;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
        """)
        main_layout.addWidget(self.status_label)
        
        # 声波动图
        self.waveform_widget = WaveformWidget()
        main_layout.addWidget(self.waveform_widget)
        # 插入图片代替模型示意图和箭头
        image_label = QLabel()
        pixmap = QPixmap("/Users/huiyufei/Desktop/mini_audio_caption/图像2025-7-14 00.48.jpg")
        pixmap = pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(image_label)
        
        # 结果显示
        result_label = QLabel("Generating result:")
        result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-top: 10px;")
        main_layout.addWidget(result_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(150)
        self.result_text.setPlaceholderText("The audio description will be displayed here...")
        main_layout.addWidget(self.result_text)
        
        # 容器
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def toggle_recording(self):
        if not self.is_recording:
            # 开始录制
            self.is_recording = True
            self.record_btn.setText("stop recording")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            self.status_label.setText("Recording audio...")
            self.frames = []
            self.waveform_data = np.zeros(32000)
            
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
        else:
            # 停止录制
            self.is_recording = False
            self.record_btn.setText("Start Recording")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            self.status_label.setText("In processing audio...")
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # 保存临时文件并处理
            self.process_recorded_audio()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        # 实时更新波形
        audio_np = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        if hasattr(self, 'waveform_data'):
            self.waveform_data = np.concatenate([self.waveform_data, audio_np])[-32000:]
        else:
            self.waveform_data = audio_np
        self.waveform_widget.update_waveform(self.waveform_data)
        return (in_data, pyaudio.paContinue)
    
    def process_recorded_audio(self):
    # 保存为临时WAV文件
        temp_file = os.path.join(os.getcwd(), "temp_recording.wav")
    
        try:
        # 确保目录存在
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        
        # 写入音频数据
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        
        # 检查文件是否成功创建
            if not os.path.exists(temp_file):
                raise FileNotFoundError(f"Temporary files cannot be created: {temp_file}")
        
        # 调用模型推理
            self.status_label.setText("The audio content is being analyzed...")
            QApplication.processEvents()  # 强制刷新UI

            caption = inference(temp_file)  # 调用你的推理函数

            self.result_text.append(f"Description of the recorded audio:\n{caption}\n")
            self.status_label.setText("Completed!")
        
        except Exception as e:
            self.result_text.append(f"Error: {str(e)}")
            self.status_label.setText(f"Error: {str(e)}")
        
        finally:
        # 确保无论是否出错都尝试删除临时文件
           if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                self.result_text.append(f"Warning: Temporary files cannot be deleted: {str(e)}")
    
    def open_audio_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select the audio file", "", "音频文件 (*.wav *.mp3 *.flac *.m4a)"
        )
        
        if filepath:
            self.status_label.setText("The audio content is being analyzed...")
            QApplication.processEvents()  # 更新UI
            
            try:
                # 转换音频格式为模型需要的格式
                if not filepath.endswith('.wav'):
                    audio, sr = librosa.load(filepath, sr=32000)
                    temp_file = "temp_converted.wav"
                    sf.write(temp_file, audio, sr)
                    caption = inference(temp_file)
                    os.remove(temp_file)
                else:
                    caption = inference(filepath)
                
                self.result_text.append(f"File: {os.path.basename(filepath)}")
                self.result_text.append(f"Audio Description: {caption}\n")
                self.status_label.setText("Completed!")
            except Exception as e:
                self.result_text.append(f"Error: {str(e)}")
                self.status_label.setText("Error Handling")
    
    def closeEvent(self, event):
        # 清理资源
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioCaptionApp()
    window.show()
    sys.exit(app.exec())