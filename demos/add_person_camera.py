import cv2
import os
import time
import numpy as np
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer

# ===== CONSTANTS =====
CAPTURE_INTERVAL = 3  # Chụp mỗi 3 giây
RECOGNITION_THRESHOLD = 0.6  # Threshold cho nhận dạng
MAX_FACES_ALLOWED = 1  # Chỉ cho phép 1 khuôn mặt
WAIT_TIME_AFTER_INPUT = 2  # Đợi 2 giây sau khi nhập tên

def smart_add_person_camera():
    """Thêm người thông minh - tự động nhận diện và hỏi tên khi cần"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager, threshold=RECOGNITION_THRESHOLD)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ [CAMERA] Không thể mở webcam")
        return False
    
    print("🤖 SMART ADD PERSON - Chụp tự động")
    print("=" * 50)
    print("📷 Camera sẽ tự động chụp ảnh mỗi 3 giây")
    print("🔍 Hệ thống sẽ kiểm tra người đã có chưa")
    print("✋ Nhấn 'q' để thoát")
    print()
    
    last_capture_time = 0
    capture_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ [CAMERA] Không thể đọc frame từ camera")
                break
            
            current_time = time.time()
            
            # Hiển thị frame với countdown
            display_frame = frame.copy()
            time_until_capture = CAPTURE_INTERVAL - (current_time - last_capture_time)
            
            if time_until_capture > 0:
                countdown = int(time_until_capture) + 1
                cv2.putText(display_frame, f"Next capture in: {countdown}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Capturing...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Captured: {capture_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Smart Add Person", display_frame)
            
            # Auto capture
            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                result = _process_auto_capture(frame, detector, gallery_manager, recognizer)
                if result:
                    capture_count += 1
                    print(f"📷 Capture #{capture_count}: {result}")
                last_capture_time = current_time
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Dừng bởi người dùng")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Hoàn thành - Đã chụp {capture_count} ảnh")
    
    return True

def _process_auto_capture(frame, detector, gallery_manager, recognizer):
    """Xử lý auto capture và quyết định thêm người"""
    # Detect faces
    img_rgb, faces = detector.detect_faces(frame)
    
    if not faces or len(faces) == 0:
        return "Không tìm thấy khuôn mặt"
    
    if len(faces) > MAX_FACES_ALLOWED:
        return f"Tìm thấy {len(faces)} khuôn mặt - Chỉ được {MAX_FACES_ALLOWED} người"
    
    # Get embedding của khuôn mặt
    face = faces[0]
    embedding = detector.get_face_embedding(img_rgb, face)
    if embedding is None:
        return "Không thể trích xuất embedding"
    
    # Nhận diện xem đã có chưa
    result = recognizer.recognize(embedding)
    person_name = result["result"]
    score = result.get("score", 0)
    
    if person_name != "Unknown":
        # Người đã có - thêm ảnh vào gallery
        success, msg = gallery_manager.add_person(person_name, image=img_rgb)
        if success:
            return f"✅ Thêm ảnh cho {person_name} (score: {score:.3f})"
        else:
            return f"❌ {msg}"
    else:
        # Người chưa có - hỏi tên
        cv2.destroyAllWindows()  # Tạm đóng camera window
        
        print(f"\n👤 NGƯỜI MỚI PHÁT HIỆN (confidence: {score:.3f})")
        new_name = input("Nhập tên người này (Enter để bỏ qua): ").strip()
        
        # Đợi một chút trước khi tiếp tục
        time.sleep(WAIT_TIME_AFTER_INPUT)
        
        if new_name:
            success, msg = gallery_manager.add_person(new_name, image=img_rgb)
            if success:
                return f"🎉 Tạo mới {new_name}"
            else:
                return f"❌ {msg}"
        else:
            return "⏭️ Bỏ qua người này"

def add_person_camera():
    """Thêm người bằng camera - chế độ thông minh"""
    return smart_add_person_camera()

if __name__ == "__main__":
    add_person_camera()