# Camera Settings
CAMERA_FPS = 30  # Target FPS cho camera
CAMERA_WIDTH = 1080  # Độ rộng camera
CAMERA_HEIGHT = 720  # Độ cao camera

# Face Detection & Recognition
RECOGNITION_INTERVAL = 0.3  # Nhận dạng mỗi 0.3 giây (ưu tiên accuracy)
RECOGNITION_THRESHOLD = 0.5  # Threshold cho nhận dạng
DEFAULT_THRESHOLD = 0.5  # Default threshold cho face recognition
DEFAULT_TOP_K = 3  # Số lượng kết quả top matches trả về
SIMILARITY_THRESHOLD = 0.95  # Threshold cho việc kiểm tra ảnh trùng lặp

# Model Settings
MODEL_NAME = "buffalo_l"  # Tên model face detection
CTX_ID = 0  # Context ID cho model
DET_SIZE = (640, 640)  # Kích thước detection

# Add Person Camera Settings
CAPTURE_INTERVAL = 3  # Chụp mỗi 3 giây
MAX_FACES_ALLOWED = 1  # Chỉ cho phép 1 khuôn mặt
WAIT_TIME_AFTER_INPUT = 2  # Đợi 2 giây sau khi nhập tên

# File Settings
GALLERY_PATH = 'face_gallery.pkl'  # Đường dẫn mặc định cho face gallery
SUPPORTED_IMAGE_EXT = "*.jpg *.jpeg *.png *.bmp *.gif"  # Định dạng ảnh hỗ trợ

# Sample Images
SAMPLE_IMAGES = [
    "https://picsum.photos/300/300?random=1",
    "https://picsum.photos/300/300?random=2",
]
DEFAULT_TEST_IMAGE = "test.jpg"
DEFAULT_TEST_URL = "https://picsum.photos/400/400?random=3"

# Messages
NO_FACE_MSG = "No face detected"
EMPTY_GALLERY_MSG = "Empty gallery"
UNKNOWN_LABEL = "Unknown"
