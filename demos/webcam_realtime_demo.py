import cv2
import time
import statistics
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer
from config import (
    RECOGNITION_INTERVAL, RECOGNITION_THRESHOLD,
    CAMERA_FPS, CAMERA_WIDTH, CAMERA_HEIGHT
)

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
    # Per-person score history for session summary
    per_person_scores = {}
    
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
                t0 = time.perf_counter()
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
                else:
                    cached_results = []

                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                # record sample for session-average
                if 'proc_samples' not in locals():
                    proc_samples = []
                proc_samples.append(elapsed_ms)

                # Print detection results with inference time on same line
                if cached_results:
                    for result_info in cached_results:
                        name = result_info['result']['result']
                        score = result_info['result'].get('score', 0)
                        print(f"👤 Detected: {name} (Score: {score:.3f}) - {elapsed_ms:.1f} ms")
                        # record per-person score for session summary
                        try:
                            per_person_scores.setdefault(name, []).append(float(score))
                        except Exception:
                            pass

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

        # If we recorded inference samples during the session, print summary
        if 'proc_samples' in locals() and proc_samples:
            count = len(proc_samples)
            avg_ms = sum(proc_samples) / count
            try:
                median_ms = statistics.median(proc_samples)
            except Exception:
                median_ms = sorted(proc_samples)[count//2]
            print(f"\n🧾 Inference time summary:")
            print(f"   - Samples: {count}")
            print(f"   - Avg: {avg_ms:.1f} ms")
            print(f"   - Median: {median_ms:.1f} ms")
        else:
            print("\n🧾 No inference samples recorded during session.")

        # Per-person average score summary (session)
        if per_person_scores:
            print("\n📋 Per-person score summary:")
            # compute avg and count, sort by avg desc then count desc
            summary = []
            for name, scores in per_person_scores.items():
                try:
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    best_score = max(scores) if scores else 0.0
                except Exception:
                    avg_score = 0.0
                    best_score = 0.0
                # name, count, avg, best
                summary.append((name, len(scores), avg_score, best_score))
            # sort by best_score desc, then avg_score desc, then count desc
            summary.sort(key=lambda x: (x[3], x[2], x[1]), reverse=True)
            for name, cnt, avg_score, best_score in summary:
                print(f"   - {name}: samples={cnt}, avg_score={avg_score:.3f}, best_score={best_score:.3f}")
        else:
            print("\n📋 No per-person scores recorded.")

        print("✅ Đã thoát webcam demo")

if __name__ == "__main__":
    try:
        webcam_realtime_demo()
    except Exception as e:
        print(f"❌ [STARTUP] Không thể chạy demo realtime: {e}")
        print("🔧 Hãy thử cài: pip install opencv-contrib-python")