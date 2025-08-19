import cv2
import os
import time
import numpy as np
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer

def smart_add_person_camera():
    """Thêm người thông minh - tự động nhận diện và hỏi tên khi cần"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager, threshold=0.6)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam")
        return False
    
    print("🤖 SMART ADD PERSON - Chụp tự động")
    print("=" * 50)
    print("📷 Camera sẽ tự động chụp ảnh mỗi 3 giây")
    print("🔍 Hệ thống sẽ kiểm tra người đã có chưa")
    print("✋ Nhấn 'q' để thoát")
    print()
    
    capture_interval = 3  # Chụp mỗi 3 giây
    last_capture_time = 0
    capture_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Không thể đọc frame từ camera")
                break
            
            current_time = time.time()
            
            # Hiển thị frame với countdown
            display_frame = frame.copy()
            time_until_capture = capture_interval - (current_time - last_capture_time)
            
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
            if current_time - last_capture_time >= capture_interval:
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
    
    if len(faces) > 1:
        return "Tìm thấy nhiều khuôn mặt - Chỉ được 1 người"
    
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
        
        if new_name:
            success, msg = gallery_manager.add_person(new_name, image=img_rgb)
            if success:
                return f"🎉 Tạo mới {new_name}"
            else:
                return f"❌ {msg}"
        else:
            return "⏭️ Bỏ qua người này"

def add_person_camera_realtime():
    """Thêm người mới bằng camera với preview realtime"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    
    # Nhập tên người trước
    name = input("Nhập tên người cần thêm: ").strip()
    if not name:
        print("Tên không được để trống!")
        return False
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam")
        return False
    
    print(f"Đang chuẩn bị chụp ảnh cho: {name}")
    print("Nhấn 'SPACE' để chụp ảnh, 'q' để thoát")
    print("Hãy đặt khuôn mặt vào giữa màn hình...")
    
    captured = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Phát hiện khuôn mặt để hiển thị preview
            img_rgb, faces = detector.detect_faces(frame)
            
            # Vẽ bbox preview
            if faces:
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Hiển thị số khuôn mặt
                cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Hiển thị hướng dẫn
            cv2.putText(frame, "SPACE: Capture, Q: Quit", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hiển thị frame
            cv2.imshow(f'Add Person: {name}', frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space để chụp
                # Lưu ảnh tạm
                temp_filename = f"temp_capture_{int(time.time())}.jpg"
                cv2.imwrite(temp_filename, frame)
                
                # Thêm vào gallery
                success, msg = gallery_manager.add_person(name, image_path=temp_filename)
                print(msg)
                
                if success:
                    captured = True
                    print(f"Đã thêm {name} thành công!")
                
                # Xóa file tạm
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                break
                
    except Exception as e:
        print(f"Lỗi: {e}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return captured

def add_person_camera_auto():
    """Thêm người mới bằng camera với countdown tự động"""
    # Khởi tạo
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    
    # Nhập tên người trước
    name = input("Nhập tên người cần thêm: ").strip()
    if not name:
        print("Tên không được để trống!")
        return False
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam")
        return False
    
    print(f"Đang chuẩn bị chụp ảnh cho: {name}")
    print("Hãy đặt khuôn mặt vào giữa camera...")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"Chụp ảnh sau {i} giây...")
        time.sleep(1)
        
        # Đọc frame để kiểm tra khuôn mặt
        ret, frame = cap.read()
        if ret:
            _, faces = detector.detect_faces(frame)
            if faces:
                print(f"  ✓ Phát hiện {len(faces)} khuôn mặt")
            else:
                print("  ⚠ Không phát hiện khuôn mặt")
    
    # Chụp ảnh
    print("📸 CHỤP!")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Không thể chụp ảnh")
        return False
    
    # Lưu ảnh tạm
    temp_filename = f"capture_{name}_{int(time.time())}.jpg"
    cv2.imwrite(temp_filename, frame)
    print(f"Đã lưu ảnh: {temp_filename}")
    
    # Kiểm tra khuôn mặt
    _, faces = detector.detect_faces(frame)
    if not faces:
        print("❌ Không phát hiện khuôn mặt trong ảnh vừa chụp")
        print(f"Ảnh được lưu tại: {temp_filename} - bạn có thể thêm thủ công")
        return False
    
    if len(faces) > 1:
        print(f"⚠ Phát hiện {len(faces)} khuôn mặt. Sẽ lấy khuôn mặt lớn nhất")
    
    # Thêm vào gallery
    success, msg = gallery_manager.add_person(name, image_path=temp_filename)
    print(msg)
    
    if success:
        print(f"✅ Đã thêm {name} thành công!")
        # Xóa file tạm
        os.remove(temp_filename)
        return True
    else:
        print(f"Ảnh được lưu tại: {temp_filename} - bạn có thể thêm thủ công")
        return False

def add_person_camera():
    """Thêm người bằng camera - chế độ thông minh"""
    return smart_add_person_camera()

if __name__ == "__main__":
    add_person_camera()
