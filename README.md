# 🎯 Face Recognition System

Hệ thống nhận diện khuôn mặt sử dụng InsightFace với giao diện đa trang hiện đại.

## ✨ Tính năng chính

- 📷 **Nhận diện từ ảnh** (JPG, PNG, BMP, GIF)
- 🌐 **Nhận diện từ URL** trực tiếp 
- 🎥 **Webcam real-time** với camera
- 👥 **Quản lý Gallery** (thêm/xóa/sửa người)
- 🤖 **Smart Add Person** - tự động thêm người
- 📱 **Multi-page GUI** với navigation hiện đại

## 🚀 Cài đặt và chạy

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng
```bash
# Ứng dụng
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
│   ├── webcam_realtime_demo.py  # Real-time webcam
│   └── add_person_camera.py     # Smart add person
├── 📁 utils/               # Utilities
│   ├── image_utils.py      # Image processing
│   └── visualization.py    # Drawing & display
├── multipage_gui.py        # 🎯 Main GUI application
├── main.py                 # Console interface
└── requirements.txt        # Dependencies
```

## 📱 Giao diện Multi-Page

### Navigation Pages:
- 🏠 **Home**: Gallery stats và overview
- 📷 **Image Recognition**: Chọn file ảnh + preview
- 🌐 **URL Recognition**: Nhập URL ảnh
- 🎥 **Webcam Recognition**: Real-time
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
- **Webcam**: 15-30 FPS với GPU
- **Formats**: JPG, PNG, BMP, GIF, MP4, AVI

---

**🚀 Happy Face Recognition!** 🎯

## 📋 Model & Embedding Comparison (Benchmark)

Use this table to record and compare runtime / matching quality for different detector models
and different embedding-comparison methods (cosine, euclid, ...). Fill values by
running the benchmark script or manual tests.

| Model pack | Embedding metric | Avg inference (ms) | Median (ms) | Avg match score (Img/Camera) | Best score | 
|-------|------------------:|-------------------:|-----------:|----------------:|---------:|---------:|-------|
| buffalo_l | cosine | 235.5/248.8 | 248.2 | 0.645/0.820 | 0.852 |
| Retina10GF/ms1mv2r50 | cosine | 172.1/172.9 | 173 | 0.562/0.812 | 0.845 |
| Retina10GF/glint360kr50 | cosine | 309.9/372.5 | 344.2 | 0.62/0.827 | 0.850 |
|-------|------------------:|-------------------:|-----------:|----------------:|---------:|---------:|-------|
| buffalo_m | cosine | 182.5/196.4 | 193.8 | 0.647/0.831 | 0.853 |
| SCRFD10GF/w600kr50 | cosine | 223.6/222.3 | 220.9 | 0.645/0.826 | 0.854 |
| SCRFD500MF/w600kr50 | cosine | 173.3/182.6 | 183.1 | 0.636/0.841 | 0.859 |
| buffalo_s | cosine | 65.1/54 | 47.8 | 0.559/0.793 | 0.817 |
| antelopev2 | cosine | 937.6/977.2 | 974.8 | 0.649/0.823 | 0.839 |
| Retina500MF/glint360kr100 | cosine | 1012.1/1277.8 | 891.7 | 0.648/0.803 | 0.830 |
| Retina2.5GF/glint360kr100 | cosine | 951.4/923.9 | 931.6 | 0.64/0.805 | 0.832 |
| Retina10GF/glint360kr100 | cosine | 948.3/992 | 989.9 | 0.649/0.814 | 0.845 |

