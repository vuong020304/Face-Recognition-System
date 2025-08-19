import cv2
import numpy as np
import requests
from io import BytesIO

def load_image_from_url(url):
    """Tải ảnh từ URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Lỗi khi tải ảnh từ URL: {str(e)}")
        return None

def resize_image(image, max_size=800):
    """Thay đổi kích thước ảnh giữ tỷ lệ"""
    h, w = image.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image