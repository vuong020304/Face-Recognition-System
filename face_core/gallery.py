import os
import pickle
import numpy as np
from datetime import datetime
from config import GALLERY_PATH, SIMILARITY_THRESHOLD

class FaceGalleryManager:
    """Quản lý thư viện khuôn mặt"""
    
    def __init__(self, detector, gallery_path=None):
        self.detector = detector
        self.gallery_path = gallery_path or GALLERY_PATH
        self.gallery = self._load_gallery() if os.path.exists(self.gallery_path) else {}
        
        if not self.gallery:
            print("Cảnh báo: Gallery trống. Sử dụng phương thức add_person để thêm người.")
        
    def _load_gallery(self):
        """Tải gallery từ file"""
        with open(self.gallery_path, 'rb') as f:
            return pickle.load(f)
    
    def save_gallery(self):
        """Lưu gallery hiện tại"""
        with open(self.gallery_path, 'wb') as f:
            pickle.dump(self.gallery, f)
    
    def add_person(self, name, image_path=None, image=None, similarity_threshold=SIMILARITY_THRESHOLD):
        """Thêm người vào gallery"""
        # Trích xuất embedding
        embedding = self.detector.get_face_embedding(image_path if image is None else image)
        if embedding is None:
            return False, "Không tìm thấy khuôn mặt"
        
        # Kiểm tra người đã tồn tại
        is_new_person = name not in self.gallery
        if is_new_person:
            self.gallery[name] = []
        
        # Kiểm tra duplicate (chỉ với người đã có)
        if not is_new_person:
            for existing_embedding in self.gallery[name]:
                similarity = np.dot(embedding, existing_embedding)
                if similarity >= similarity_threshold:
                    return False, f"Ảnh tương tự đã tồn tại cho {name} (similarity: {similarity:.3f})"
        
        # Thêm embedding
        self.gallery[name].append(embedding)
        self.save_gallery()
        
        # Thông báo phù hợp
        action = "Đã tạo mới" if is_new_person else "Đã thêm ảnh cho"
        total_count = len(self.gallery[name])
        return True, f"{action} {name} (tổng: {total_count} ảnh)"
    
    def remove_person(self, name):
        """Xóa người khỏi gallery"""
        if name in self.gallery:
            del self.gallery[name]
            self.save_gallery()
            return True, f"Đã xóa {name} khỏi gallery"
        return False, f"Không tìm thấy {name} trong gallery"
    
    def find_duplicates(self, threshold=0.95):
        """Tìm các embedding trùng lặp"""
        results = {}
        
        for name, embeddings in self.gallery.items():
            duplicates = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j])
                    if similarity >= threshold:
                        duplicates.append((i, j, similarity))
            
            if duplicates:
                results[name] = duplicates
        
        return results
    
    def remove_duplicate(self, name, index):
        """Xóa một embedding trùng lặp"""
        if name in self.gallery and 0 <= index < len(self.gallery[name]):
            self.gallery[name].pop(index)
            self.save_gallery()
            return True, f"Đã xóa embedding thứ {index} của {name}"
        return False, "Không thể xóa embedding"
    
    def get_all_people(self):
        """Lấy danh sách tất cả người trong gallery"""
        return list(self.gallery.keys())
    
    def get_person_count(self):
        """Đếm số người và số embeddings"""
        counts = {name: len(embs) for name, embs in self.gallery.items()}
        return counts
    
