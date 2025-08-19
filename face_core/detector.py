import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    """Phát hiện và xử lý khuôn mặt"""
    
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        self.detector = FaceAnalysis(name=model_name)
        self.detector.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def detect_faces(self, image):
        """Phát hiện khuôn mặt từ ảnh"""
        # Load ảnh nếu đường dẫn
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Validate ảnh
        if not self._is_valid_image(image):
            return None, None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.get(image_rgb)
        return image_rgb, faces
    
    def _is_valid_image(self, image):
        """Kiểm tra ảnh hợp lệ"""
        return (image is not None and 
                isinstance(image, np.ndarray) and 
                len(image.shape) == 3 and 
                image.shape[2] == 3)
    
    def get_face_embedding(self, image, face=None):
        """Trích xuất embedding từ khuôn mặt"""
        if face is None:
            _, faces = self.detect_faces(image)
            if not faces or len(faces) == 0:
                return None
            face = max(faces, key=lambda x: x.det_score)
        
        embedding = face.embedding
        # Chuẩn hóa embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding