#!/usr/bin/env python3
"""
Multi-Page Face Recognition GUI Launcher
🎯 Professional multi-page interface với navigation
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Kiểm tra và cài đặt dependencies cần thiết"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check CustomTkinter
    try:
        import customtkinter
        print("✅ CustomTkinter found")
    except ImportError:
        missing_deps.append("customtkinter>=5.2.0")
    
    # Check InsightFace
    try:
        import insightface
        print("✅ InsightFace found")
    except ImportError:
        missing_deps.append("insightface>=0.7.3")
    
    # Check OpenCV
    try:
        import cv2
        print("✅ OpenCV found")
    except ImportError:
        missing_deps.append("opencv-contrib-python>=4.5.0")
    
    # Check other deps
    required_deps = ['numpy', 'matplotlib', 'PIL', 'requests']
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} found")
        except ImportError:
            if dep == 'PIL':
                missing_deps.append("Pillow>=8.3.0")
            else:
                missing_deps.append(f"{dep}>=1.0.0")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\n📦 Installing missing dependencies...")
        
        import subprocess
        try:
            for dep in missing_deps:
                print(f"Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print("✅ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("\n🔧 Please install manually:")
            print("pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies found!")
        return True

def initialize_models():
    """Khởi tạo models lần đầu nếu cần"""
    print("\n🤖 Initializing InsightFace models...")
    try:
        from insightface.app import FaceAnalysis
        # Initialize với buffalo_l model
        detector = FaceAnalysis(name='buffalo_l')
        detector.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace models initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("🔄 This may take a few minutes on first run...")
        return False

def main():
    """Main launcher function"""
    print("🚀 Starting Multi-Page Face Recognition GUI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    # Initialize models
    if not initialize_models():
        print("⚠️  Model initialization failed, but proceeding anyway...")
    
    # Start multi-page GUI
    print("\n🎯 Launching Multi-Page GUI...")
    try:
        from multipage_gui import MultiPageFaceRecognitionApp
        
        # Create and run app
        app = MultiPageFaceRecognitionApp()
        print("✅ GUI started successfully!")
        print("📱 Navigate between pages using the top navigation bar")
        print("🔄 Close this terminal when done")
        
        app.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Please ensure all files are in the correct location:")
        print("- multipage_gui.py")
        print("- face_core/ directory")
        print("- demos/ directory") 
        print("- utils/ directory")
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure you're in the correct directory")
        print("2. Check that all dependencies are installed")
        print("3. Try running: python multipage_gui.py directly")
        
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
