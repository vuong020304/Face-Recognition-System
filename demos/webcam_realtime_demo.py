import cv2
import time
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer

# ===== CONSTANTS =====
RECOGNITION_INTERVAL = 0.3  # Nhận dạng mỗi 0.3 giây (ưu tiên accuracy)
RECOGNITION_THRESHOLD = 0.6  # Threshold cho nhận dạng
CAMERA_FPS = 30  # Target FPS cho camera
CAMERA_WIDTH = 1080  # Độ rộng camera
CAMERA_HEIGHT = 720  # Độ cao camera

def webcam_realtime_demo():
    """Demo webcam với nhận dạng realtime - chỉ hiển thị FPS"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager, threshold=RECOGNITION_THRESHOLD)
    
    # Kiểm tra gallery
    if not gallery_manager.gallery:
        print("⚠️ [GALLERY] Gallery trống!")
        print("📝 Hãy thêm người vào gallery trước khi sử dụng chức năng này.")
        return
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ [CAMERA] Không thể mở webcam")
        return
    
    print("🎥 Đang khởi động demo webcam realtime...")
    print("👆 Nhấn 'q' để thoát")
    
    # Thiết lập camera properties
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    frame_count = 0
    start_time = time.time()
    last_recognition_time = time.time()
    
    # Cache kết quả để tránh flickering
    cached_results = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ [CAMERA] Không thể đọc frame từ webcam")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Nhận dạng với interval để ưu tiên accuracy
            if current_time - last_recognition_time >= RECOGNITION_INTERVAL:
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
                            
                            # In kết quả ra console
                            name = result['result']
                            score = result.get('score', 0)
                            print(f"👤 Detected: {name} (Score: {score:.3f})")
                else:
                    cached_results = []
                
                last_recognition_time = current_time
            
            # Vẽ kết quả lên frame
            display_frame = frame.copy()
            for result_info in cached_results:
                bbox = result_info['bbox']
                result = result_info['result']
                
                x1, y1, x2, y2 = bbox
                name = result['result']
                score = result.get('score', 0)
                
                # Chọn màu theo kết quả
                if name == "Unknown":
                    color = (0, 0, 255)  # Đỏ
                elif score > 0.7:
                    color = (0, 255, 0)  # Xanh lá
                elif score > 0.5:
                    color = (0, 255, 255)  # Vàng
                else:
                    color = (255, 165, 0)  # Cam
                
                # Vẽ bbox và label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({score:.2f})"
                cv2.putText(display_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Hiển thị FPS
            elapsed = current_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hiển thị hướng dẫn
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition - Realtime', display_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("\n⏹️ Dừng bởi người dùng")
    except Exception as e:
        print(f"❌ [ERROR] Lỗi trong quá trình chạy: {e}")
        
    finally:
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        
        # Hiển thị thống kê cuối
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Thống kê:")
        print(f"   ⏱️  Thời gian chạy: {total_time:.1f}s")
        print(f"   🎞️  Tổng frames: {frame_count}")
        print(f"   📈 FPS trung bình: {avg_fps:.1f}")
        print("✅ Đã thoát webcam demo")

if __name__ == "__main__":
    try:
        webcam_realtime_demo()
    except Exception as e:
        print(f"❌ [STARTUP] Không thể chạy demo realtime: {e}")
        print("🔧 Hãy thử cài: pip install opencv-contrib-python")