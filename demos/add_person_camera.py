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
    """Thêm người thông minh - tự động nhận diện và trả về summary.

    Thu thập trong suốt phiên:
      - existing_adds: dict name -> count (ảnh đã thêm cho người có sẵn)
      - unknown_groups: list of groups, mỗi group {'embs': [...], 'images': [...]} (chưa gán tên)

    Trả về một dict summary khi người dùng nhấn 'q'.
    """
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager, threshold=RECOGNITION_THRESHOLD)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ [CAMERA] Không thể mở webcam")
        return None

    print("🤖 SMART ADD PERSON - Chụp tự động")
    print("=" * 50)
    print("📷 Camera sẽ tự động chụp ảnh mỗi 3 giây")
    print("🔍 Hệ thống sẽ kiểm tra người đã có chưa")
    print("✋ Nhấn 'q' để thoát")

    last_capture_time = 0
    capture_count = 0

    existing_adds = {}
    unknown_groups = []
    logs = []
    tmp_dir = "tmp_captures"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ [CAMERA] Không thể đọc frame từ camera")
                break

            current_time = time.time()
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

            if current_time - last_capture_time >= CAPTURE_INTERVAL:
                res = _process_auto_capture(frame, detector, gallery_manager, recognizer)
                if isinstance(res, dict):
                    capture_count += 1
                    status = res.get('status')
                    msg = res.get('message', '')
                    logs.append(msg)
                    # print status line in the requested format
                    print(f"📷 Capture #{capture_count}: {msg}")

                    if status == 'existing':
                        name = res.get('name')
                        if name:
                            existing_adds[name] = existing_adds.get(name, 0) + 1
                    elif status == 'unknown':
                        emb = res.get('embedding')
                        img = res.get('image')
                        placed = False

                        # Save image to temp file to avoid keeping large numpy arrays in summary
                        img_path = None
                        try:
                            if img is not None:
                                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                img_path = os.path.join(tmp_dir, f"capture_{int(time.time()*1000)}_{capture_count}.jpg")
                                cv2.imwrite(img_path, bgr)
                        except Exception:
                            img_path = None

                        if emb is not None:
                            for grp in unknown_groups:
                                mean_emb = grp.get('mean_emb')
                                if mean_emb is None:
                                    continue
                                num = float(np.dot(mean_emb, emb))
                                den = float(np.linalg.norm(mean_emb) * np.linalg.norm(emb) + 1e-8)
                                sim = num / den if den != 0 else 0.0
                                if sim >= 0.7:
                                    grp.setdefault('embs', []).append(emb)
                                    if img_path:
                                        grp.setdefault('images_paths', []).append(img_path)
                                    # update mean embedding
                                    grp['mean_emb'] = np.mean(grp['embs'], axis=0)
                                    placed = True
                                    break
                        if not placed:
                            new_grp = {
                                'embs': [emb] if emb is not None else [],
                                'mean_emb': emb if emb is not None else None,
                                'images_paths': [img_path] if img_path is not None else []
                            }
                            unknown_groups.append(new_grp)
                else:
                    capture_count += 1
                    msg = str(res)
                    logs.append(msg)
                    print(f"📷 Capture #{capture_count}: {msg}")

                last_capture_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️ Dừng bởi người dùng")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Build sanitized summary (exclude raw embeddings and numpy arrays)
    sanitized_groups = []
    for grp in unknown_groups:
        sanitized_groups.append({
            'thumb_path': grp.get('images_paths', [None])[0],
            'images_paths': grp.get('images_paths', []),
            'count': len(grp.get('images_paths', []))
        })

    summary = {
        'capture_count': capture_count,
        'existing_adds': existing_adds,
        'unknown_groups': sanitized_groups
    }

    return summary

def _process_auto_capture(frame, detector, gallery_manager, recognizer):
    """Xử lý auto capture và quyết định thêm người.

    Trả về dict:
      - {'status':'existing', 'name':..., 'message':...}
      - {'status':'unknown', 'embedding':..., 'image':..., 'message':...}
      - {'status':'no_face'|'multiple'|'error', 'message':...}
    """
    img_rgb, faces = detector.detect_faces(frame)

    if not faces or len(faces) == 0:
        return {"status": "no_face", "message": "Không tìm thấy khuôn mặt"}

    if len(faces) > MAX_FACES_ALLOWED:
        return {"status": "multiple", "message": f"Tìm thấy {len(faces)} khuôn mặt - Chỉ được {MAX_FACES_ALLOWED} người"}

    face = faces[0]
    embedding = detector.get_face_embedding(img_rgb, face)
    if embedding is None:
        return {"status": "error", "message": "Không thể trích xuất embedding"}

    result = recognizer.recognize(embedding)
    person_name = result.get("result")
    score = result.get("score", 0)

    if person_name and person_name != "Unknown":
        success, msg = gallery_manager.add_person(person_name, image=img_rgb)
        if success:
            return {"status": "existing", "name": person_name, "message": f"Thêm ảnh cho {person_name} (score: {score:.3f})"}
        else:
            return {"status": "error", "message": msg}
    else:
        return {"status": "unknown", "embedding": embedding, "image": img_rgb, "message": f"Người mới phát hiện (score: {score:.3f})"}

def add_person_camera():
    """Thêm người bằng camera - chế độ thông minh"""
    return smart_add_person_camera()

if __name__ == "__main__":
    add_person_camera()