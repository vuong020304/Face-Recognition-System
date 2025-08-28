"""
Multi-Page Face Recognition GUI
Giao diện đa trang với navigation
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
from PIL import Image
import os
import sys

# Import các module của hệ thống
from face_core.detector import FaceDetector
from face_core.gallery import FaceGalleryManager
from face_core.recognizer import FaceRecognizer
from demos.image_demo import recognize_from_source
from demos.add_person_camera import smart_add_person_camera
from demos.webcam_realtime_demo import webcam_realtime_demo

# Set appearance mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MultiPageFaceRecognitionApp:
    def __init__(self):
        # Initialize window
        self.root = ctk.CTk()
        self.root.title("🎯 Face Recognition System - Multi-Page")
        self.root.geometry("1400x900")
        
        # Initialize face recognition components
        self.detector = FaceDetector()
        self.gallery_manager = FaceGalleryManager(self.detector)
        self.recognizer = FaceRecognizer(self.detector, self.gallery_manager)
        
        # Variables
        self.current_page = "home"
        self.pages = {}
        
        self.setup_ui()
        self.show_page("home")
        
    def setup_ui(self):
        """Thiết lập giao diện đa trang"""
        
        # Main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        self.create_header()
        
        # Navigation
        self.create_navigation()
        
        # Content area
        self.content_frame = ctk.CTkFrame(self.main_container)
        self.content_frame.pack(fill="both", expand=True, pady=(20, 0))
        
        # Create all pages
        self.create_pages()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Tạo header"""
        header_frame = ctk.CTkFrame(self.main_container, height=80)
        header_frame.pack(fill="x", pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title với page indicator
        self.title_label = ctk.CTkLabel(
            header_frame,
            text="🎯 Face Recognition System",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.pack(side="left", padx=20, pady=20)
        
        # Current page indicator
        self.page_indicator = ctk.CTkLabel(
            header_frame,
            text="📍 Home",
            font=ctk.CTkFont(size=16),
            text_color="gray70"
        )
        self.page_indicator.pack(side="right", padx=20, pady=20)
    
    def create_navigation(self):
        """Tạo navigation bar"""
        nav_frame = ctk.CTkFrame(self.main_container, height=60)
        nav_frame.pack(fill="x", pady=(0, 20))
        nav_frame.pack_propagate(False)
        
        # Navigation buttons
        nav_buttons = [
            ("🏠 Home", "home", "blue"),
            ("📷 Image Recognition", "image", "green"), 
            ("🌐 URL Recognition", "url", "green"),
            ("🎥 Webcam Recognition", "webcam", "orange"),
            ("👥 Gallery Manager", "gallery", "orange"),
            ("⚙️ Settings", "settings", "gray")
        ]
        
        self.nav_buttons = {}
        
        for btn_text, page_id, color in nav_buttons:
            btn = ctk.CTkButton(
                nav_frame,
                text=btn_text,
                command=lambda p=page_id: self.show_page(p),
                width=200,
                height=40,
                font=ctk.CTkFont(size=14),
                fg_color=self.get_color(color),
                hover_color=self.get_hover_color(color)
            )
            btn.pack(side="left", padx=5, pady=10)
            self.nav_buttons[page_id] = btn
    
    def get_color(self, color_name):
        """Lấy màu theo tên"""
        colors = {
            "blue": "#1f538d",
            "green": "#2fa572", 
            "orange": "#ff6b35",
            "red": "#e74c3c",
            "gray": "#5c6c7c"
        }
        return colors.get(color_name, "#1f538d")
    
    def get_hover_color(self, color_name):
        """Lấy màu hover"""
        colors = {
            "blue": "#144670",
            "green": "#25835e",
            "orange": "#e55a2b", 
            "red": "#c0392b",
            "gray": "#4a5568"
        }
        return colors.get(color_name, "#144670")
    
    def create_pages(self):
        """Tạo tất cả các trang"""
        self.pages["home"] = self.create_home_page()
        self.pages["image"] = self.create_image_page()
        self.pages["url"] = self.create_url_page()
        self.pages["webcam"] = self.create_webcam_page()
        self.pages["gallery"] = self.create_gallery_page()
        self.pages["settings"] = self.create_settings_page()
    
    def create_home_page(self):
        """Tạo trang chủ"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Welcome section
        welcome_frame = ctk.CTkFrame(page)
        welcome_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Logo và welcome text
        welcome_label = ctk.CTkLabel(
            welcome_frame,
            text="🚀 Welcome to Face Recognition System\n\n" +
                 "🎯 Multi-Page Professional Interface\n" +
                 "📊 Navigate between different functions\n" +
                 "✨ Modern design with smooth transitions\n\n" +
                 "👆 Choose a function from the navigation bar above",
            font=ctk.CTkFont(size=18),
            justify="center"
        )
        welcome_label.pack(expand=True)
        
        # Quick stats
        stats_frame = ctk.CTkFrame(welcome_frame)
        stats_frame.pack(fill="x", padx=40, pady=40)
        
        # Gallery stats
        counts = self.gallery_manager.get_person_count()
        total_people = len(counts)
        total_images = sum(counts.values()) if counts else 0
        
        stats_label = ctk.CTkLabel(
            stats_frame,
            text=f"📊 Gallery Stats: {total_people} people, {total_images} images",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        stats_label.pack(pady=20)
        
        return page
    
    def create_image_page(self):
        """Tạo trang nhận diện ảnh"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(
            page,
            text="📷 Image Recognition",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Main content
        content_frame = ctk.CTkFrame(page)
        content_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Left: Controls
        left_frame = ctk.CTkFrame(content_frame, width=400)
        left_frame.pack(side="left", fill="y", padx=(20, 10), pady=20)
        left_frame.pack_propagate(False)
        
        # Instructions
        instructions = ctk.CTkLabel(
            left_frame,
            text="🎯 Select an image file to analyze\n\n" +
                 "✅ Supported formats: JPG, PNG, BMP, GIF\n" +
                 "🔍 System will detect and recognize faces\n" +
                 "📊 Results will show in a popup window",
            font=ctk.CTkFont(size=14),
            justify="left"
        )
        instructions.pack(padx=20, pady=30)
        
        # Select file button
        self.select_file_btn = ctk.CTkButton(
            left_frame,
            text="📁 Select Image File",
            command=self.select_and_process_image,
            width=300,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2fa572"
        )
        self.select_file_btn.pack(pady=20)
        
        # Processing status
        self.image_status_label = ctk.CTkLabel(
            left_frame,
            text="📋 Ready to process image",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.image_status_label.pack(pady=10)
        
        # Right: Preview area
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        # Preview label
        preview_title = ctk.CTkLabel(
            right_frame,
            text="🖼️ Image Preview",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        preview_title.pack(pady=20)
        
        # Preview area
        self.image_preview = ctk.CTkLabel(
            right_frame,
            text="No image selected\n\nClick 'Select Image File' to start",
            width=400,
            height=300,
            fg_color="gray20",
            corner_radius=10
        )
        self.image_preview.pack(pady=20, padx=20)
        
        return page
    
    def create_url_page(self):
        """Tạo trang nhận diện URL"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(
            page,
            text="🌐 URL Image Recognition",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Main content
        content_frame = ctk.CTkFrame(page)
        content_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Instructions
        instructions = ctk.CTkLabel(
            content_frame,
            text="🎯 Enter an image URL to analyze faces\n\n" +
                 "✅ Supports direct image links (jpg, png, etc.)\n" +
                 "🌐 Downloads and processes the image\n" +
                 "📊 Results displayed in popup window",
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        instructions.pack(pady=30)
        
        # URL input
        url_frame = ctk.CTkFrame(content_frame)
        url_frame.pack(pady=30)
        
        url_label = ctk.CTkLabel(
            url_frame,
            text="🔗 Image URL:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        url_label.pack(pady=(20, 10))
        
        self.url_entry = ctk.CTkEntry(
            url_frame,
            width=600,
            height=40,
            placeholder_text="https://example.com/image.jpg",
            font=ctk.CTkFont(size=14)
        )
        self.url_entry.pack(pady=10)
        
        # Process button
        process_url_btn = ctk.CTkButton(
            url_frame,
            text="🚀 Process URL",
            command=self.process_url_image,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2fa572"
        )
        process_url_btn.pack(pady=20)
        
        # Status
        self.url_status_label = ctk.CTkLabel(
            url_frame,
            text="📋 Enter URL and click Process",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.url_status_label.pack(pady=(10, 20))
        
        return page
    
    def create_webcam_page(self):
        """Tạo trang webcam"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(
            page,
            text="🎥 Webcam Recognition",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Main content
        content_frame = ctk.CTkFrame(page)
        content_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Webcam section
        webcam_frame = ctk.CTkFrame(content_frame)
        webcam_frame.pack(fill="x", padx=20, pady=20)
        
        webcam_title = ctk.CTkLabel(
            webcam_frame,
            text="📹 Realtime Webcam Recognition",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        webcam_title.pack(pady=20)
        
        webcam_desc = ctk.CTkLabel(
            webcam_frame,
            text="🎯 Start real-time face recognition using your webcam\n" +
                 "✅ Live face detection and recognition\n" +
                 "📊 Shows confidence scores and FPS",
            font=ctk.CTkFont(size=14)
        )
        webcam_desc.pack(pady=10)
        
        start_webcam_btn = ctk.CTkButton(
            webcam_frame,
            text="🚀 Start Webcam",
            command=self.start_webcam,
            width=250,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#ff6b35"
        )
        start_webcam_btn.pack(pady=20)
        
        return page
    
    def create_gallery_page(self):
        """Tạo trang quản lý gallery"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(
            page,
            text="👥 Gallery Management",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Main content
        content_frame = ctk.CTkFrame(page)
        content_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Left: Actions
        left_frame = ctk.CTkFrame(content_frame, width=400)
        left_frame.pack(side="left", fill="y", padx=(20, 10), pady=20)
        left_frame.pack_propagate(False)
        
        actions_title = ctk.CTkLabel(
            left_frame,
            text="⚡ Quick Actions",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        actions_title.pack(pady=20)
        
        # Action buttons
        actions = [
            ("👤 Add Person (Image)", self.add_person_image, "#2fa572"),
            ("🤖 Add Person (Camera)", self.add_person_camera, "#ff6b35"),
            ("🔄 Initialize Sample", self.init_sample_gallery, "#5c6c7c"),
            ("❌ Remove Person", self.remove_person, "#e74c3c")
        ]
        
        for btn_text, btn_command, btn_color in actions:
            btn = ctk.CTkButton(
                left_frame,
                text=btn_text,
                command=btn_command,
                width=300,
                height=45,
                font=ctk.CTkFont(size=14),
                fg_color=btn_color
            )
            btn.pack(pady=8, padx=20)
        
        # Right: Gallery view
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)
        
        gallery_title = ctk.CTkLabel(
            right_frame,
            text="📋 Gallery Overview",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        gallery_title.pack(pady=20)
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            right_frame,
            text="🔄 Refresh",
            command=self.refresh_gallery_view,
            width=100,
            height=30
        )
        refresh_btn.pack(pady=10)
        
        # Gallery list
        self.gallery_scrollable = ctk.CTkScrollableFrame(right_frame)
        self.gallery_scrollable.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.refresh_gallery_view()
        
        return page
    
    def create_settings_page(self):
        """Tạo trang settings"""
        page = ctk.CTkFrame(self.content_frame)
        
        # Title
        title = ctk.CTkLabel(
            page,
            text="⚙️ Settings & Configuration",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Main content
        content_frame = ctk.CTkFrame(page)
        content_frame.pack(fill="both", expand=True, padx=40, pady=(0, 40))
        
        # Theme settings
        theme_frame = ctk.CTkFrame(content_frame)
        theme_frame.pack(fill="x", padx=20, pady=20)
        
        theme_title = ctk.CTkLabel(
            theme_frame,
            text="🎨 Appearance",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        theme_title.pack(pady=20)
        
        # Theme selector
        theme_selector = ctk.CTkOptionMenu(
            theme_frame,
            values=["Dark", "Light", "System"],
            command=self.change_theme
        )
        theme_selector.pack(pady=10)
        
        # Recognition settings
        recog_frame = ctk.CTkFrame(content_frame)
        recog_frame.pack(fill="x", padx=20, pady=20)
        
        recog_title = ctk.CTkLabel(
            recog_frame,
            text="🎯 Recognition Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        recog_title.pack(pady=20)
        
        # Threshold slider
        threshold_label = ctk.CTkLabel(
            recog_frame,
            text="Recognition Threshold: 0.5",
            font=ctk.CTkFont(size=14)
        )
        threshold_label.pack(pady=10)
        
        self.threshold_slider = ctk.CTkSlider(
            recog_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            command=self.update_threshold
        )
        self.threshold_slider.set(0.5)
        self.threshold_slider.pack(pady=10)
        
        return page
    
    def create_status_bar(self):
        """Tạo status bar"""
        status_frame = ctk.CTkFrame(self.main_container, height=40)
        status_frame.pack(fill="x", pady=(20, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🟢 Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(side="left", padx=20, pady=10)
        
        # Current time
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        time_label = ctk.CTkLabel(
            status_frame,
            text=f"⏰ {current_time}",
            font=ctk.CTkFont(size=12)
        )
        time_label.pack(side="right", padx=20, pady=10)
    
    def show_page(self, page_id):
        """Hiển thị trang được chọn"""
        # Hide all pages
        for page in self.pages.values():
            page.pack_forget()
        
        # Show selected page
        if page_id in self.pages:
            self.pages[page_id].pack(fill="both", expand=True)
            self.current_page = page_id
            
            # Update page indicator
            page_names = {
                "home": "📍 Home",
                "image": "📍 Image Recognition", 
                "url": "📍 URL Recognition",
                "webcam": "📍 Webcam",
                "gallery": "📍 Gallery Management",
                "settings": "📍 Settings"
            }
            self.page_indicator.configure(text=page_names.get(page_id, "📍 Unknown"))
            
            # Update nav button colors
            self.update_nav_buttons(page_id)
    
    def update_nav_buttons(self, active_page):
        """Cập nhật màu nav buttons"""
        for page_id, btn in self.nav_buttons.items():
            if page_id == active_page:
                btn.configure(fg_color="#144670")  # Active color
            else:
                # Reset to original colors
                colors = {
                    "home": "#1f538d",
                    "image": "#2fa572",
                    "url": "#2fa572", 
                    "webcam": "#ff6b35",
                    "gallery": "#ff6b35",
                    "settings": "#5c6c7c"
                }
                btn.configure(fg_color=colors.get(page_id, "#1f538d"))
    
    # Event handlers
    def select_and_process_image(self):
        """Chọn và xử lý ảnh"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.image_status_label.configure(text="🔄 Processing...")
            self.select_file_btn.configure(state="disabled")
            
            # Show preview
            self.show_image_preview(file_path)
            
            # Process directly in main thread để tránh matplotlib threading issues
            try:
                results = recognize_from_source(file_path, self.detector, self.recognizer)
                
                if results:
                    self.image_status_label.configure(text=f"✅ Found {len(results)} faces")
                else:
                    self.image_status_label.configure(text="❌ No faces found")
                    
            except Exception as e:
                self.image_status_label.configure(text=f"❌ Error: {e}")
                messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            finally:
                self.select_file_btn.configure(state="normal")
    
    def show_image_preview(self, file_path):
        """Hiển thị preview ảnh"""
        try:
            # Load and resize image for preview
            image = Image.open(file_path)
            
            # Resize for preview
            preview_size = (350, 250)
            image.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            # Convert to CTk format
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            
            # Update preview
            self.image_preview.configure(image=photo, text="")
            
        except Exception as e:
            self.image_preview.configure(text=f"Preview error: {e}")
    

    
    def process_url_image(self):
        """Xử lý ảnh từ URL"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter an image URL")
            return
        
        self.url_status_label.configure(text="🔄 Loading from URL...")
        
        # Process URL directly in main thread để tránh matplotlib threading issues
        try:
            results = recognize_from_source(url, self.detector, self.recognizer)
            
            if results:
                self.url_status_label.configure(text=f"✅ Found {len(results)} faces")
            else:
                self.url_status_label.configure(text="❌ No faces found")
                
        except Exception as e:
            self.url_status_label.configure(text=f"❌ Error: {e}")
            messagebox.showerror("Error", f"Failed to process URL:\n{str(e)}")
    
    def start_webcam(self):
        """Khởi động webcam"""
        try:
            threading.Thread(target=webcam_realtime_demo).start()
            messagebox.showinfo("Webcam", "Webcam window opened separately")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot start webcam: {e}")
    
    def add_person_image(self):
        """Thêm người từ ảnh"""
        dialog = ctk.CTkInputDialog(text="Enter person name:", title="Add Person")
        name = dialog.get_input()
        
        if not name:
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            success, msg = self.gallery_manager.add_person(name, image_path=file_path)
            if success:
                messagebox.showinfo("Success", msg)
                self.refresh_gallery_view()
            else:
                messagebox.showerror("Error", msg)
    
    def add_person_camera(self):
        """Thêm người bằng camera"""
        try:
            # Run smart_add_person_camera in background and refresh gallery when done
            threading.Thread(target=self._add_person_camera_thread, daemon=True).start()
            messagebox.showinfo("Camera", "Smart Add Person camera opened (press 'q' in camera window to finish)")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot start camera: {e}")

    def _add_person_camera_thread(self):
        """Background runner for smart_add_person_camera that refreshes gallery after finish"""
        try:
            result = smart_add_person_camera()
            print("smart_add_person_camera returned:", result)
        except Exception as e:
            print("smart_add_person_camera error:", e)
            result = None

        # Ensure GUI refresh runs on main thread
        try:
            self.root.after(0, self.refresh_gallery_view)
        except Exception:
            # fallback
            self.refresh_gallery_view()
    
    def init_sample_gallery(self):
        """Khởi tạo gallery mẫu"""
        result = messagebox.askyesno("Confirm", "Initialize sample gallery?")
        if result:
            try:
                self.init_sample_gallery_local()
                messagebox.showinfo("Success", "Sample gallery initialized")
                self.refresh_gallery_view()
            except Exception as e:
                messagebox.showerror("Error", f"Cannot initialize gallery: {e}")
    
    def init_sample_gallery_local(self):
        """Initialize sample gallery với sample images"""
        # Tạo sample data directory nếu chưa có
        sample_dir = "sample_images"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            
        # Thông báo cho user thêm images vào folder
        messagebox.showinfo(
            "Sample Gallery", 
            f"Please add sample images to '{sample_dir}' folder\n"
            "Create subfolders for each person:\n"
            "- sample_images/person1/image1.jpg\n"
            "- sample_images/person2/image2.jpg\n\n"
            "Sample gallery structure created!"
        )
    
    def remove_person(self):
        """Xóa người"""
        dialog = ctk.CTkInputDialog(text="Enter person name to remove:", title="Remove Person")
        name = dialog.get_input()
        
        if name:
            result = messagebox.askyesno("Confirm", f"Remove '{name}' from gallery?")
            if result:
                success, msg = self.gallery_manager.remove_person(name)
                if success:
                    messagebox.showinfo("Success", msg)
                    self.refresh_gallery_view()
                else:
                    messagebox.showerror("Error", msg)
    
    def refresh_gallery_view(self):
        """Refresh gallery view"""
        # Clear existing widgets
        for widget in self.gallery_scrollable.winfo_children():
            widget.destroy()
        
        # Reload gallery manager from disk to ensure fresh data
        try:
            # Recreate gallery manager so it reloads saved data
            self.gallery_manager = FaceGalleryManager(self.detector)
            # Recreate recognizer so it picks up updated gallery embeddings
            self.recognizer = FaceRecognizer(self.detector, self.gallery_manager)
        except Exception:
            pass

        # Get gallery data
        counts = self.gallery_manager.get_person_count()
        
        if not counts:
            no_data_label = ctk.CTkLabel(
                self.gallery_scrollable,
                text="📭 No people in gallery\n\nUse 'Add Person' to start",
                font=ctk.CTkFont(size=14),
                text_color="gray70"
            )
            no_data_label.pack(pady=50)
        else:
            # Add people info
            for name, count in sorted(counts.items()):
                person_frame = ctk.CTkFrame(self.gallery_scrollable)
                person_frame.pack(fill="x", pady=5, padx=10)
                
                person_label = ctk.CTkLabel(
                    person_frame,
                    text=f"👤 {name}",
                    font=ctk.CTkFont(size=14, weight="bold")
                )
                person_label.pack(side="left", padx=20, pady=10)
                
                count_label = ctk.CTkLabel(
                    person_frame,
                    text=f"{count} images",
                    font=ctk.CTkFont(size=12),
                    text_color="gray70"
                )
                count_label.pack(side="right", padx=20, pady=10)
    
    def change_theme(self, new_theme):
        """Thay đổi theme"""
        ctk.set_appearance_mode(new_theme.lower())
    
    def update_threshold(self, value):
        """Cập nhật threshold"""
        # Update threshold if recognizer has this attribute
        if hasattr(self.recognizer, 'threshold'):
            self.recognizer.threshold = value
        
        # Update threshold label text
        threshold_label = None
        for widget in self.pages["settings"].winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkFrame):
                        for subchild in child.winfo_children():
                            if isinstance(subchild, ctk.CTkLabel) and "Threshold" in subchild.cget("text"):
                                subchild.configure(text=f"Recognition Threshold: {value:.2f}")
                                return
    

    
    def run(self):
        """Chạy ứng dụng"""
        self.root.mainloop()

def main():
    """Main function"""
    try:
        app = MultiPageFaceRecognitionApp()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure to install: pip install customtkinter")

if __name__ == "__main__":
    main()
