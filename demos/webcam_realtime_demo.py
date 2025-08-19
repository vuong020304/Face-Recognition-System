import cv2
import time
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer

def webcam_realtime_demo():
    """Demo webcam với hiển thị realtime (cần OpenCV có GUI support)"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager)
    
    # Kiểm tra gallery
    if not gallery_manager.gallery:
        print("Cảnh báo: Gallery trống!")
        choice = input("Bạn có muốn khởi tạo gallery mẫu? (y/n): ")
        if choice.lower() == 'y':
            from demos.image_demo import init_sample_gallery
            init_sample_gallery(gallery_manager)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam")
        return
    
    print("Đang khởi động demo webcam realtime...")
    print("Nhấn 'q' để thoát, 's' để chụp ảnh")
    
    # Thiết lập FPS và kích thước
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    last_recognition_time = time.time()
    recognition_interval = 0.5  # Nhận diện mỗi 0.5 giây
    
    # Cache kết quả để tránh flickering
    cached_results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Nhận diện khuôn mặt với interval để tăng hiệu suất
            if current_time - last_recognition_time >= recognition_interval:
                img_rgb, faces = detector.detect_faces(frame)
                
                if faces is not None and len(faces) > 0:
                    cached_results = []
                    for face in faces:
                        embedding = detector.get_face_embedding(img_rgb, face)
                        if embedding is not None:
                            result = recognizer.recognize(embedding)
                            cached_results.append({
                                'bbox': face.bbox.astype(int),
                                'result': result
                            })
                
                last_recognition_time = current_time
            
            # Vẽ kết quả từ cache sử dụng function chuẩn
            if cached_results:
                from utils.visualization import draw_faces
                
                # Tạo faces và results từ cache
                faces = []
                results = []
                for cache_item in cached_results:
                    # Tạo fake face object với bbox
                    class FakeFace:
                        def __init__(self, bbox):
                            self.bbox = bbox
                    
                    faces.append(FakeFace(cache_item['bbox']))
                    results.append(cache_item['result'])
                
                frame = draw_faces(frame, faces, results)
            
            # Hiển thị thông tin hệ thống
            elapsed = current_time - start_time
            fps = frame_count / (elapsed + 0.1)  # Tránh chia cho 0
            info_text = f"FPS: {fps:.1f} | Faces: {len(cached_results)} | Gallery: {len(gallery_manager.gallery)}"
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hiển thị hướng dẫn
            cv2.putText(frame, "Press 'q': Quit, 's': Save screenshot", (10, frame.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition - Realtime', frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Chụp ảnh
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Đã lưu ảnh: {filename}")
                
    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {e}")
        
    finally:
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        print("Đã thoát webcam demo")

if __name__ == "__main__":
    try:
        webcam_realtime_demo()
    except Exception as e:
        print(f"Không thể chạy demo realtime: {e}")
        print("Hãy thử cài: pip install opencv-contrib-python")
