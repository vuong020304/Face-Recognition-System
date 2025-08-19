# demos/video_demo.py
import cv2
import os
import time
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer

def video_recognition_demo(video_path=None):
    """
    Demo nhận diện khuôn mặt từ video
    
    Args:
        video_path: Đường dẫn đến file video. Nếu None, sẽ sử dụng webcam.
    """
    # Khởi tạo các đối tượng
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager)
    
    # Mở video
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)
    else:
        print("Đường dẫn video không hợp lệ. Chuyển sang sử dụng webcam.")
        cap = cv2.VideoCapture(0)
        source_name = "Webcam"
    
    if not cap.isOpened():
        print(f"Không thể mở nguồn video: {source_name}")
        return
    
    # Thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Đang xử lý video: {source_name}")
    print(f"Kích thước: {frame_width}x{frame_height}, FPS: {fps}")
    print("Nhấn 'q' để thoát")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Xử lý mỗi 3 frame để tăng tốc độ
        if frame_count % 3 != 0:
            continue
        
        # Phát hiện và nhận diện khuôn mặt
        img_rgb, faces = detector.detect_faces(frame)
        
        if faces is not None and len(faces) > 0:
            results = []
            for face in faces:
                embedding = detector.get_face_embedding(img_rgb, face)
                if embedding is not None:
                    result = recognizer.recognize(embedding)
                    results.append(result)
            
            # Vẽ kết quả sử dụng function chuẩn
            from utils.visualization import draw_faces
            frame = draw_faces(frame, faces, results)
        
        # Hiển thị FPS
        elapsed_time = time.time() - start_time
        fps_real = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps_real:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Lưu frame thay vì hiển thị (tránh lỗi GUI)
        if frame_count % 30 == 0:  # Lưu mỗi 30 frames
            output_filename = f"video_frame_{frame_count}.jpg"
            cv2.imwrite(output_filename, frame)
            print(f"Đã lưu frame: {output_filename}")
        
        # Tự động thoát sau một số frame để tránh chạy mãi
        if frame_count > 300:  # Tối đa 300 frames
            print("Đã đạt giới hạn frames. Thoát...")
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Đã xử lý {frame_count} frames trong {elapsed_time:.2f} giây")
    print(f"FPS trung bình: {frame_count / elapsed_time:.2f}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    video_recognition_demo(video_path)