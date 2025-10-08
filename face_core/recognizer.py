import numpy as np
from config import (
    DEFAULT_THRESHOLD, DEFAULT_TOP_K, NO_FACE_MSG,
    EMPTY_GALLERY_MSG, UNKNOWN_LABEL
)

class FaceRecognizer:
    """Nhận diện khuôn mặt"""
    
    def __init__(self, detector, gallery_manager, threshold=DEFAULT_THRESHOLD):
        self.detector = detector
        self.gallery_manager = gallery_manager
        self.threshold = threshold

    def recognize(self, image, top_k=DEFAULT_TOP_K):
        """Nhận diện khuôn mặt từ ảnh"""
        # Lấy embedding
        embedding = self._get_embedding(image)
        if embedding is None:
            return {"result": NO_FACE_MSG, "top_matches": []}
        
        # So khớp với gallery
        matches = self._match_with_gallery(embedding)
        if not matches:
            return {"result": EMPTY_GALLERY_MSG, "top_matches": []}
        
        # Trả về kết quả
        return self._format_results(matches, top_k)
    
    def _get_embedding(self, image):
        """Lấy embedding từ ảnh hoặc trả về nếu đã là embedding"""
        if isinstance(image, np.ndarray) and image.ndim == 1:
            return image
        return self.detector.get_face_embedding(image)
    
    #Cosine similarity
    def _match_with_gallery(self, embedding):
        """So khớp embedding với gallery"""
        matches = []
        for name, embeddings in self.gallery_manager.gallery.items():
            scores = [np.dot(embedding, e) for e in embeddings]
            best_score = max(scores) if scores else 0
            matches.append((name, best_score))
        return matches
    
    '''#Euclidean distance
    def _match_with_gallery(self, embedding, use_euclid=False):
        matches = []
        for name, embeddings in self.gallery_manager.gallery.items():
            if use_euclid:
                # Euclid -> lấy khoảng cách nhỏ nhất
                dists = [np.linalg.norm(embedding - e) for e in embeddings]
                if dists:
                    mind = min(dists)
                    # OPTION A: giữ dạng 'lớn hơn tốt' bằng cách đảo dấu
                    best_score = -mind
                    # OPTION B (thích hợp nếu muốn [0,1] similarity): best_score = 1.0 / (1.0 + mind)
                else:
                    best_score = -np.inf
            else:
                # Cosine similarity (embedding nên đã L2-normalized)
                scores = [np.dot(embedding, e) for e in embeddings]
                best_score = max(scores) if scores else 0.0

            matches.append((name, best_score))
        return matches'''
    
    def _format_results(self, matches, top_k):
        """Format kết quả cuối cùng"""
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:top_k]
        best_match, best_score = top_matches[0]
        
        result = best_match if best_score >= self.threshold else UNKNOWN_LABEL
        return {
            "result": result,
            "score": best_score,
            "top_matches": top_matches
        } 