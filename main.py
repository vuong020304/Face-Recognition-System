#!/usr/bin/env python3
"""
Face Recognition System - Main Application
🎯 Ứng dụng nhận dạng khuôn mặt với menu console tương tác
"""

import os
import sys
import argparse
import cv2
import time
import numpy as np
from pathlib import Path

from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer
from utils.image_utils import load_image_from_url
from utils.visualization import show_image

# ===== CONSTANTS =====
RECOGNITION_THRESHOLD = 0.6  # Threshold cho nhận dạng
DUPLICATE_THRESHOLD = 0.95   # Threshold cho duplicate detection
MAX_TOP_MATCHES = 3          # Số lượng top matches hiển thị
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}  # Định dạng ảnh hỗ trợ

def normalize_path(path_input):
    """Chuẩn hóa đường dẫn từ nhiều định dạng khác nhau"""
    if not path_input:
        return None
    
    # Xóa các dấu ngoặc kép, đơn, khoảng trắng thừa
    path = path_input.strip().strip('"').strip("'").strip()
    
    if not path:
        return None
    
    # Sử dụng pathlib để chuẩn hóa
    try:
        path_obj = Path(path)
        
        # Nếu là đường dẫn tương đối, resolve nó
        if not path_obj.is_absolute():
            path_obj = Path.cwd() / path_obj
        
        # Resolve để xử lý .. và .
        path_obj = path_obj.resolve()
        
        return str(path_obj)
    except Exception as e:
        print(f"⚠️  [PATH] Lỗi chuẩn hóa đường dẫn: {e}")
        return path

def is_valid_image_file(file_path):
    """Kiểm tra xem file có phải là ảnh hợp lệ không"""
    if not file_path or not os.path.exists(file_path):
        return False
    
    try:
        path_obj = Path(file_path)
        
        # Kiểm tra extension
        if path_obj.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
            return False
        
        # Kiểm tra kích thước file (không quá nhỏ)
        if path_obj.stat().st_size < 100:  # < 100 bytes
            return False
        
        return True
    except Exception:
        return False

def get_file_info(file_path):
    """Lấy thông tin file"""
    try:
        path_obj = Path(file_path)
        size_mb = path_obj.stat().st_size / (1024 * 1024)
        return {
            'name': path_obj.name,
            'size_mb': size_mb,
            'extension': path_obj.suffix.lower()
        }
    except Exception:
        return None

class FaceRecognitionApp:
    """Ứng dụng nhận dạng khuôn mặt chính"""
    
    def __init__(self):
        """Khởi tạo ứng dụng"""
        print("🔄 Đang khởi tạo Face Recognition System...")
        
        # Initialize core components
        self.detector = FaceDetector()
        self.gallery_manager = FaceGalleryManager(self.detector)
        self.recognizer = FaceRecognizer(self.detector, self.gallery_manager, threshold=RECOGNITION_THRESHOLD)
        
        print("✅ Hệ thống đã sẵn sàng!")
        self._show_status()
    
    def _show_status(self):
        """Hiển thị trạng thái hiện tại của gallery"""
        counts = self.gallery_manager.get_person_count()
        total_people = len(counts)
        total_images = sum(counts.values()) if counts else 0
        
        print(f"\n📊 Gallery Status: {total_people} người, {total_images} ảnh")
        if counts:
            print("👥 Danh sách:")
            for name, count in counts.items():
                print(f"   - {name}: {count} ảnh")
    
    def _show_menu(self):
        """Hiển thị menu chính"""
        print("\n" + "="*50)
        print("🎯 FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. 📷 Thêm người qua camera")
        print("2. 🖼️  Thêm người từ ảnh (file/URL)")
        print("3. 🎥 Nhận dạng realtime (camera)")
        print("4. 🔍 Nhận dạng từ ảnh")
        print("5. 👥 Xem danh sách gallery")
        print("6. 🗑️  Xóa người khỏi gallery")
        print("7. 🔄 Tìm và xóa duplicate")
        print("8. 📊 Thống kê gallery")
        print("0. ❌ Thoát")
        print("="*50)
    
    def _suggest_similar_files(self, target_path):
        """Gợi ý các file tương tự trong thư mục"""
        try:
            target_dir = Path(target_path).parent
            target_name = Path(target_path).stem.lower()
            
            if not target_dir.exists():
                return
            
            similar_files = []
            for file_path in target_dir.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS and
                    target_name in file_path.stem.lower()):
                    similar_files.append(str(file_path))
            
            if similar_files:
                print(f"🔍 Các file tương tự trong thư mục:")
                for i, file_path in enumerate(similar_files[:5], 1):
                    print(f"   {i}. {Path(file_path).name}")
        except Exception:
            pass
    
    def add_person_via_camera(self):
        """Thêm người qua camera"""
        print("\n📷 THÊM NGƯỜI QUA CAMERA")
        print("-" * 30)
        
        try:
            from demos.add_person_camera import smart_add_person_camera
            result = smart_add_person_camera()
            
            # Reload gallery sau khi thêm người qua camera
            self._reload_gallery()
            
            return result
        except Exception as e:
            print(f"❌ [CAMERA] Lỗi khi thêm người qua camera: {e}")
            return False
    
    def _reload_gallery(self):
        """Reload gallery từ file để đồng bộ với các thay đổi"""
        try:
            old_gallery = dict(self.gallery_manager.gallery)
            self.gallery_manager.gallery = self.gallery_manager._load_gallery()
            
            # Check for changes
            new_counts = self.gallery_manager.get_person_count()
            old_counts = {name: len(embs) for name, embs in old_gallery.items()}
            
            changes_detected = False
            for name, new_count in new_counts.items():
                old_count = old_counts.get(name, 0)
                if new_count != old_count:
                    changes_detected = True
                    print(f"🔄 [SYNC] {name}: {old_count} → {new_count} ảnh")
            
            if changes_detected:
                print("✅ Đã đồng bộ gallery thành công!")
        except Exception as e:
            print(f"⚠️  [SYNC] Lỗi khi reload gallery: {e}")
    
    def add_person_via_image(self):
        """Thêm người từ ảnh (file hoặc URL)"""
        print("\n🖼️  THÊM NGƯỜI TỪ ẢNH")
        print("-" * 30)
        
        name = input("Nhập tên người: ").strip()
        if not name:
            print("❌ Tên không được để trống!")
            return
        
        source = input("Nhập đường dẫn ảnh hoặc URL: ").strip()
        if not source:
            print("❌ Đường dẫn không được để trống!")
            return
        
        # Load image
        if source.startswith(("http://", "https://")):
            print("🌐 Đang tải ảnh từ URL...")
            image = load_image_from_url(source)
            if image is None:
                print("❌ [NETWORK] Không thể tải ảnh từ URL")
                return
        else:
            # Chuẩn hóa đường dẫn
            normalized_path = normalize_path(source)
            if not normalized_path:
                print("❌ [PATH] Đường dẫn không hợp lệ!")
                return
            
            print(f"📁 Đường dẫn chuẩn hóa: {normalized_path}")
            
            # Kiểm tra file tồn tại
            if not os.path.exists(normalized_path):
                print(f"❌ [FILE] File không tồn tại: {normalized_path}")
                # Gợi ý các file tương tự
                self._suggest_similar_files(normalized_path)
                return
            
            # Kiểm tra định dạng ảnh
            if not is_valid_image_file(normalized_path):
                print(f"❌ [FORMAT] File không phải là ảnh hợp lệ hoặc bị lỗi")
                print(f"📝 Các định dạng hỗ trợ: {', '.join(SUPPORTED_IMAGE_FORMATS)}")
                return
            
            # Hiển thị thông tin file
            file_info = get_file_info(normalized_path)
            if file_info:
                print(f"📊 File: {file_info['name']} ({file_info['size_mb']:.2f} MB)")
            
            # Đọc ảnh
            image = cv2.imread(normalized_path)
            if image is None:
                print("❌ [FILE] Không thể đọc file ảnh! File có thể bị hỏng.")
                return
        
        # Add to gallery
        success, msg = self.gallery_manager.add_person(name, image=image)
        if success:
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
    
    def realtime_recognition(self):
        """Nhận dạng realtime qua camera"""
        print("\n🎥 NHẬN DẠNG REALTIME")
        print("-" * 30)
        
        if not self.gallery_manager.gallery:
            print("⚠️  Gallery trống! Hãy thêm người trước.")
            return
        try:
            from demos.webcam_realtime_demo import webcam_realtime_demo
            webcam_realtime_demo()
        except Exception as e:
            print(f"❌ [CAMERA] Lỗi khi chạy nhận dạng realtime: {e}")
            print("🔧 Hãy thử: pip install opencv-contrib-python")
    
    def recognize_from_image(self):
        """Nhận dạng từ ảnh"""
        print("\n🔍 NHẬN DẠNG TỪ ẢNH")
        print("-" * 30)
        
        if not self.gallery_manager.gallery:
            print("⚠️  Gallery trống! Hãy thêm người trước.")
            return
        
        source = input("Nhập đường dẫn ảnh hoặc URL: ").strip()
        if not source:
            print("❌ Đường dẫn không được để trống!")
            return
        
        # Chuẩn hóa đường dẫn nếu không phải URL
        if not source.startswith(("http://", "https://")):
            normalized_source = normalize_path(source)
            if normalized_source:
                print(f"📁 Đường dẫn chuẩn hóa: {normalized_source}")
                source = normalized_source
        
        try:
            from demos.image_demo import recognize_from_source
            results = recognize_from_source(source, self.detector, self.recognizer)
            
            if results:
                print(f"\n✅ Hoàn thành nhận dạng {len(results)} khuôn mặt!")
            else:
                print("\n❌ Không thể nhận dạng được khuôn mặt nào!")
                
        except Exception as e:
            print(f"❌ [RECOGNITION] Lỗi khi nhận dạng: {e}")
    
    def show_gallery_list(self):
        """Hiển thị danh sách người trong gallery"""
        print("\n👥 DANH SÁCH GALLERY")
        print("-" * 30)
        
        counts = self.gallery_manager.get_person_count()
        if not counts:
            print("📭 Gallery trống!")
            return
        
        total_people = len(counts)
        total_images = sum(counts.values())
        
        print(f"📊 Tổng quan: {total_people} người, {total_images} ảnh\n")
        
        for i, (name, count) in enumerate(counts.items(), 1):
            print(f"{i:2}. {name:20} - {count} ảnh")
    
    def remove_person(self):
        """Xóa người khỏi gallery"""
        print("\n🗑️  XÓA NGƯỜI KHỎI GALLERY")
        print("-" * 30)
        
        counts = self.gallery_manager.get_person_count()
        if not counts:
            print("📭 Gallery trống!")
            return
        
        # Hiển thị danh sách
        print("Danh sách hiện tại:")
        for i, (name, count) in enumerate(counts.items(), 1):
            print(f"{i}. {name} ({count} ảnh)")
        
        # Nhập tên cần xóa
        name_to_remove = input("\nNhập tên người cần xóa: ").strip()
        if not name_to_remove:
            print("❌ Tên không được để trống!")
            return
        
        if name_to_remove not in counts:
            print(f"❌ Không tìm thấy '{name_to_remove}' trong gallery!")
            return
        
        # Xác nhận
        confirm = input(f"⚠️  Bạn có chắc muốn xóa '{name_to_remove}' ({counts[name_to_remove]} ảnh)? (y/N): ")
        if confirm.lower() != 'y':
            print("🚫 Đã hủy thao tác xóa")
            return
        
        # Thực hiện xóa
        success, msg = self.gallery_manager.remove_person(name_to_remove)
        if success:
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
    
    def find_and_remove_duplicates(self):
        """Tìm và gợi ý xóa duplicate"""
        print("\n🔄 TÌM VÀ XÓA DUPLICATE")
        print("-" * 30)
        
        if not self.gallery_manager.gallery:
            print("📭 Gallery trống!")
            return
        
        print("🔍 Đang tìm kiếm duplicate...")
        duplicates = self.gallery_manager.find_duplicates(threshold=DUPLICATE_THRESHOLD)
        
        if not duplicates:
            print("✅ Không tìm thấy duplicate nào!")
            return
        
        print(f"⚠️  Tìm thấy duplicate trong {len(duplicates)} người:")
        
        for name, dup_list in duplicates.items():
            print(f"\n👤 {name}:")
            for i, j, similarity in dup_list:
                print(f"   Ảnh {i} và {j}: {similarity:.3f}")
        
        # Auto-suggest removal
        print("\n🤖 Gợi ý tự động xóa:")
        total_removed = 0
        
        for name, dup_list in duplicates.items():
            print(f"\n👤 Đang xử lý {name}...")
            
            # Sort by similarity và chỉ giữ lại ảnh đầu tiên
            indices_to_remove = []
            for i, j, similarity in dup_list:
                # Luôn xóa index cao hơn để tránh shift
                indices_to_remove.append(max(i, j))
            
            # Remove duplicates và sort ngược để xóa từ index cao xuống thấp
            indices_to_remove = sorted(list(set(indices_to_remove)), reverse=True)
            
            for idx in indices_to_remove:
                confirm = input(f"   Xóa ảnh thứ {idx} của {name}? (Y/n): ")
                if confirm.lower() != 'n':
                    success, msg = self.gallery_manager.remove_duplicate(name, idx)
                    if success:
                        print(f"   ✅ {msg}")
                        total_removed += 1
                    else:
                        print(f"   ❌ {msg}")
        
        print(f"\n🎉 Hoàn thành! Đã xóa {total_removed} ảnh duplicate")
    
    def show_statistics(self):
        """Hiển thị thống kê gallery"""
        print("\n📊 THỐNG KÊ GALLERY")
        print("-" * 30)
        
        counts = self.gallery_manager.get_person_count()
        if not counts:
            print("📭 Gallery trống!")
            return
        
        total_people = len(counts)
        total_images = sum(counts.values())
        avg_images = total_images / total_people if total_people > 0 else 0
        
        print(f"👥 Tổng số người:     {total_people}")
        print(f"📸 Tổng số ảnh:       {total_images}")
        print(f"📊 TB ảnh/người:      {avg_images:.1f}")
        
        # Người có nhiều ảnh nhất
        max_person = max(counts.items(), key=lambda x: x[1])
        min_person = min(counts.items(), key=lambda x: x[1])
        
        print(f"🏆 Nhiều ảnh nhất:    {max_person[0]} ({max_person[1]} ảnh)")
        print(f"🥉 Ít ảnh nhất:       {min_person[0]} ({min_person[1]} ảnh)")
        
        # Kiểm tra duplicate
        duplicates = self.gallery_manager.find_duplicates(threshold=DUPLICATE_THRESHOLD)
        dup_count = sum(len(dup_list) for dup_list in duplicates.values())
        print(f"🔄 Duplicate pairs:   {dup_count}")
    
    def interactive_menu(self):
        """Menu tương tác chính"""
        print("🎉 Chào mừng đến với Face Recognition System!")
        
        while True:
            try:
                self._show_menu()
                choice = input("\n👆 Chọn chức năng (0-8): ").strip()

                if choice == "1":
                    self.add_person_via_camera()
                elif choice == "2":
                    self.add_person_via_image()
                elif choice == "3":
                    self.realtime_recognition()
                elif choice == "4":
                    self.recognize_from_image()
                elif choice == "5":
                    self.show_gallery_list()
                elif choice == "6":
                    self.remove_person()
                elif choice == "7":
                    self.find_and_remove_duplicates()
                elif choice == "8":
                    self.show_statistics()
                elif choice == "0":
                    print("\n👋 Cảm ơn bạn đã sử dụng Face Recognition System!")
                    break
                else:
                    print("❌ Lựa chọn không hợp lệ! Vui lòng chọn từ 0-8")

                # Hiển thị trạng thái sau mỗi thao tác
                if choice in ["1", "2", "6", "7"]:
                    # Reload gallery trước khi hiển thị status cho các thao tác có thể thay đổi gallery
                    if choice in ["1", "6", "7"]:  # Camera, remove, duplicate - có thể sử dụng instance khác
                        self._reload_gallery()
                    self._show_status()

            except KeyboardInterrupt:
                print("\n\n👋 Đã dừng ứng dụng. Cảm ơn bạn!")
                break
            except Exception as e:
                print(f"\n❌ [ERROR] Lỗi không mong muốn: {e}")
                print("🔄 Tiếp tục sử dụng ứng dụng...")

def main():
    """Entry point chính với command line arguments"""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, default='menu',
                        choices=['menu', 'webcam', 'image'],
                        help='Mode to run: menu (interactive), webcam (realtime), image (demo)')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input image file (for image mode)')
    
    args = parser.parse_args()
    
    try:
        app = FaceRecognitionApp()

        if args.mode == 'menu':
            app.interactive_menu()
        elif args.mode == 'webcam':
            app.realtime_recognition()
        elif args.mode == 'image':
            if args.input:
                # Quick image recognition với path normalization
                normalized_input = normalize_path(args.input)
                if not normalized_input or not os.path.exists(normalized_input):
                    print(f"❌ [FILE] File không tồn tại: {args.input}")
                    if normalized_input != args.input:
                        print(f"📁 Đã thử chuẩn hóa thành: {normalized_input}")
                    return

                from demos.image_demo import recognize_from_source
                recognize_from_source(normalized_input, app.detector, app.recognizer)
            else:
                app.recognize_from_image()
        else:
            print(f"❌ [MODE] Unknown mode: {args.mode}")

    except Exception as e:
        print(f"❌ [STARTUP] Lỗi khởi tạo ứng dụng: {e}")
        print("🔧 Hãy kiểm tra:")
        print("- Dependencies đã được cài đặt đầy đủ")
        print("- Camera có thể truy cập được")
        print("- Thư mục face_core/ và utils/ tồn tại")

if __name__ == "__main__":
    main()