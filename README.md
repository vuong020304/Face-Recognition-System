# 🎯 Face Recognition System

Hệ thống nhận diện khuôn mặt sử dụng InsightFace với giao diện đa trang hiện đại.

## ✨ Tính năng chính

- 📷 **Nhận diện từ ảnh** (JPG, PNG, BMP, GIF)
- 🌐 **Nhận diện từ URL** trực tiếp 
- 🎥 **Webcam real-time** và xử lý video
- 👥 **Quản lý Gallery** (thêm/xóa/sửa người)
- 🤖 **Smart Add Person** - tự động kiểm tra duplicate
- 📱 **Multi-page GUI** với navigation hiện đại

## 🚀 Cài đặt và chạy

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng
```bash
# GUI đa trang (khuyến nghị)
python run_multipage_gui.py

# Hoặc trực tiếp
python multipage_gui.py

# Console menu
python main.py
```

## 📁 Cấu trúc thư mục

```
Face-Detection/
├── 📁 face_core/           # Core modules
│   ├── detector.py         # Face detection & embedding
│   ├── gallery.py          # Gallery management  
│   └── recognizer.py       # Face recognition
├── 📁 demos/               # Demo functions
│   ├── image_demo.py       # Image recognition
│   ├── video_demo.py       # Video processing
│   ├── webcam_realtime_demo.py  # Real-time webcam
│   └── add_person_camera.py     # Smart add person
├── 📁 utils/               # Utilities
│   ├── image_utils.py      # Image processing
│   └── visualization.py    # Drawing & display
├── multipage_gui.py        # 🎯 Main GUI application
├── run_multipage_gui.py    # Launcher với auto-setup
├── main.py                 # Console interface
└── requirements.txt        # Dependencies
```

## 📱 Giao diện Multi-Page

### Navigation Pages:
- 🏠 **Home**: Gallery stats và overview
- 📷 **Image Recognition**: Chọn file ảnh + preview
- 🌐 **URL Recognition**: Nhập URL ảnh
- 🎥 **Webcam & Video**: Real-time và file processing
- 👥 **Gallery Manager**: Thêm/xóa người + gallery view
- ⚙️ **Settings**: Theme và recognition threshold

## 🔧 Dependencies chính

```txt
insightface>=0.7.3          # Face analysis engine
opencv-contrib-python>=4.5.0   # Computer vision
customtkinter>=5.2.0        # Modern GUI framework
numpy>=1.21.0              # Numerical computing
matplotlib>=3.5.0          # Image display
Pillow>=8.3.0              # Image processing
requests>=2.26.0           # HTTP requests
```

## 🛠️ Troubleshooting

| Vấn đề | Giải pháp |
|--------|----------|
| OpenCV GUI Error | `pip install opencv-contrib-python` |
| InsightFace Model Missing | Chạy `run_multipage_gui.py` |
| Import Error | Đảm bảo chạy từ thư mục gốc |
| Slow Performance | Enable CUDA GPU (tuỳ chọn) |

## 📊 Performance

- **Detection**: ~50-100ms/image
- **Accuracy**: >95% với gallery chất lượng tốt  
- **Webcam**: 15-30 FPS
- **Formats**: JPG, PNG, BMP, GIF, MP4, AVI

---

**🚀 Happy Face Recognition!** 🎯