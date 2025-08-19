#!/usr/bin/env python3
"""
Modern Face Recognition GUI Launcher
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def install_customtkinter():
    """Install CustomTkinter if not available"""
    try:
        import customtkinter
        return True
    except ImportError:
        print("🔧 CustomTkinter not found. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter>=5.2.0"])
            print("✅ CustomTkinter installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install CustomTkinter")
            print("Please run manually: pip install customtkinter")
            return False

def main():
    """Main launcher function"""
    print("🚀 Starting Modern Face Recognition GUI...")
    
    # Check and install CustomTkinter if needed
    if not install_customtkinter():
        print("💡 Falling back to basic GUI...")
        try:
            from gui_app import main as basic_main
            basic_main()
        except Exception as e:
            print(f"❌ Error starting basic GUI: {e}")
        return
    
    # Start modern GUI
    try:
        from modern_gui import main as modern_main
        modern_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📦 Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
