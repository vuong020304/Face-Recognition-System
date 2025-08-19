import argparse
from demos.image_demo import image_recognition_demo, init_sample_gallery, recognize_from_file, recognize_from_url
from demos.video_demo import video_recognition_demo

def interactive_menu():
    """Menu tương tác cho hệ thống nhận diện khuôn mặt"""
    from face_core.detector import FaceDetector
    from face_core.gallery import FaceGalleryManager
    from face_core.recognizer import FaceRecognizer
    from utils.image_utils import load_image_from_url
    
    detector = FaceDetector()
    gallery_manager = FaceGalleryManager(detector)
    recognizer = FaceRecognizer(detector, gallery_manager)
    
    while True:
        print("\n===== FACE RECOGNITION SYSTEM =====")
        print("1. Nhận diện từ ảnh local")
        print("2. Nhận diện từ URL")
        print("3. Nhận diện từ webcam (realtime)")
        print("4. Nhận diện từ video")
        print("5. Thêm người vào gallery (ảnh/URL)")
        print("6. Thêm người bằng camera")
        print("7. Xem danh sách người trong gallery")
        print("8. Khởi tạo gallery mẫu")
        print("9. Xóa người khỏi gallery")
        print("0. Thoát")
        
        choice = input("\nChọn chức năng: ")
        
        if choice == "1":
            path = input("Nhập đường dẫn ảnh: ")
            if path:
                # Xóa dấu ngoặc kép nếu có
                path = path.strip('"').strip("'")
                recognize_from_file(path, detector, recognizer)
                
        elif choice == "2":
            url = input("Nhập URL ảnh: ")
            if url:
                recognize_from_url(url, detector, recognizer)
                
        elif choice == "3":
            try:
                from demos.webcam_realtime_demo import webcam_realtime_demo
                webcam_realtime_demo()
            except Exception as e:
                print(f"Lỗi khi chạy webcam realtime: {e}")
                print("Hãy cài: pip install opencv-contrib-python")
            
        elif choice == "4":
            path = input("Nhập đường dẫn video (Enter để dùng webcam): ")
            video_recognition_demo(path if path else None)
            
        elif choice == "5":
            name = input("Nhập tên người: ")
            path = input("Nhập đường dẫn ảnh hoặc URL: ")
            
            if not name or not path:
                print("Tên và đường dẫn không được để trống!")
                continue
            
            if path.startswith("http"):
                img = load_image_from_url(path)
                if img is not None:
                    success, msg = gallery_manager.add_person(name, image=img)
                else:
                    success, msg = False, "Không thể tải ảnh từ URL"
            else:
                success, msg = gallery_manager.add_person(name, image_path=path)
                
            print(msg)
            
        elif choice == "6":
            from demos.add_person_camera import add_person_camera
            add_person_camera()
            
        elif choice == "7":
            counts = gallery_manager.get_person_count()
            if not counts:
                print("Gallery trống!")
            else:
                print("\nDanh sách người trong gallery:")
                for name, count in counts.items():
                    print(f"- {name}: {count} embeddings")
                    
        elif choice == "8":
            init_sample_gallery(gallery_manager)
            
        elif choice == "9":
            name = input("Nhập tên người cần xóa: ")
            if name:
                success, msg = gallery_manager.remove_person(name)
                print(msg)
            
        elif choice == "0":
            break
            
        else:
            print("Lựa chọn không hợp lệ!")

def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, default='menu',
                        choices=['menu', 'image', 'webcam', 'video'],
                        help='Mode to run: menu, image, webcam, or video')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image or video file')
    
    args = parser.parse_args()
    
    if args.mode == 'menu':
        interactive_menu()
    elif args.mode == 'image':
        image_recognition_demo()
    elif args.mode == 'webcam':
        try:
            from demos.webcam_realtime_demo import webcam_realtime_demo
            webcam_realtime_demo()
        except Exception as e:
            print(f"Lỗi khi chạy webcam realtime: {e}")
    elif args.mode == 'video':
        video_recognition_demo(args.input)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()