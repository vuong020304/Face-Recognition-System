import os
import cv2
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer
from utils.image_utils import load_image_from_url
from utils.visualization import draw_faces, show_image

def init_sample_gallery(gallery_manager):
    """Khởi tạo gallery mẫu nếu gallery trống"""
    if gallery_manager.gallery:
        print("Gallery đã có dữ liệu, không cần khởi tạo mẫu.")
        return
    
    print("Khởi tạo gallery mẫu...")
    sample_images = [
        "https://picsum.photos/300/300?random=1",
        "https://picsum.photos/300/300?random=2"
    ]
    
    for i, url in enumerate(sample_images):
        img = load_image_from_url(url)
        if img is not None:
            success, msg = gallery_manager.add_person(f"Sample_Person_{i+1}", image=img)
            if success:
                print(msg)
    
    print(f"Đã khởi tạo gallery với {len(gallery_manager.gallery)} người")

def recognize_from_file(image_path, detector, recognizer):
    """Nhận diện khuôn mặt từ file ảnh và hiển thị kết quả"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return None
    
    img_rgb, faces = detector.detect_faces(img)
    if faces is None or len(faces) == 0:
        print("Không phát hiện khuôn mặt nào")
        show_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Không phát hiện khuôn mặt")
        return None
    
    results = []
    for i, face in enumerate(faces):
        embedding = detector.get_face_embedding(img_rgb, face)
        if embedding is not None:
            result = recognizer.recognize(embedding)
            results.append(result)
            print(f"Khuôn mặt {i+1}: {result['result']} (Score: {result.get('score', 0):.3f})")
            if result.get('top_matches'):
                print("Top matches:")
                for name, score in result['top_matches'][:3]:
                    print(f"  - {name}: {score:.3f}")
    
    # Vẽ kết quả lên ảnh và hiển thị
    img_with_results = draw_faces(img_rgb, faces, results)
    
    # Tạo title đơn giản dựa trên kết quả nhận diện
    if len(results) == 1:
        # Một khuôn mặt
        result = results[0]
        title = f"{result['result']} Score: {result.get('score', 0):.2f}"
    elif len(results) > 1:
        # Nhiều khuôn mặt - lấy người có score cao nhất
        best_result = max(results, key=lambda x: x.get('score', 0))
        title = f"{best_result['result']} Score: {best_result.get('score', 0):.2f} (+{len(results)-1} khác)"
    else:
        title = "Không nhận diện được"
    
    show_image(img_with_results, title)
    
    return results

def recognize_from_url(url, detector, recognizer):
    """Nhận diện khuôn mặt từ URL và hiển thị kết quả"""
    print(f"Đang tải ảnh từ: {url}")
    img = load_image_from_url(url)
    if img is None:
        print("Không thể tải ảnh từ URL")
        return None
    
    # Chuyển đổi ảnh sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt
    _, faces = detector.detect_faces(img)
    if faces is None or len(faces) == 0:
        print("Không phát hiện khuôn mặt nào")
        show_image(img_rgb, "Không phát hiện khuôn mặt")
        return None
    
    # Nhận diện khuôn mặt
    results = []
    for i, face in enumerate(faces):
        embedding = detector.get_face_embedding(img_rgb, face)
        if embedding is not None:
            result = recognizer.recognize(embedding)
            results.append(result)
            print(f"Khuôn mặt {i+1}: {result['result']} (Score: {result.get('score', 0):.3f})")
            if result.get('top_matches'):
                print("Top matches:")
                for name, score in result['top_matches'][:3]:
                    print(f"  - {name}: {score:.3f}")
    
    # Vẽ kết quả lên ảnh và hiển thị
    img_with_results = draw_faces(img_rgb, faces, results)
    
    # Tạo title đơn giản dựa trên kết quả nhận diện
    if len(results) == 1:
        # Một khuôn mặt
        result = results[0]
        title = f"{result['result']} Score: {result.get('score', 0):.2f}"
    elif len(results) > 1:
        # Nhiều khuôn mặt - lấy người có score cao nhất
        best_result = max(results, key=lambda x: x.get('score', 0))
        title = f"{best_result['result']} Score: {best_result.get('score', 0):.2f} (+{len(results)-1} khác)"
    else:
        title = "Không nhận diện được"
    
    show_image(img_with_results, title)
    
    return results

def image_recognition_demo():
    # Khởi tạo các đối tượng
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager)
    
    # Khởi tạo gallery mẫu nếu cần
    if not gallery_manager.gallery:
        choice = input("Gallery trống. Bạn có muốn khởi tạo gallery mẫu? (y/n): ")
        if choice.lower() == 'y':
            init_sample_gallery(gallery_manager)
    
    # Sử dụng functions đã định nghĩa ở top level
    
    # Thử nghiệm
    test_image = "test.jpg"  # Thay đổi thành đường dẫn ảnh của bạn
    if os.path.exists(test_image):
        print("Nhận diện từ file local:")
        recognize_from_file(test_image, detector, recognizer)
    
    test_url = "https://picsum.photos/400/400?random=3"
    print("Nhận diện từ URL:")
    recognize_from_url(test_url, detector, recognizer)

if __name__ == "__main__":
    image_recognition_demo()