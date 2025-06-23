#!/usr/bin/env python3
"""
Launch NBA Prediction System Interactive Demo
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def main():
    print("🏀 NBA Prediction System - Interactive Demo")
    print("=" * 50)
    print("🚀 Starting interactive web interface...")
    print("📱 The demo will open in your default browser")
    print("🔧 Server will run on http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], 
                      check=True, capture_output=True)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.serverAddress", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except subprocess.CalledProcessError:
        print("❌ Streamlit not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
            print("✅ Streamlit installed successfully!")
            print("🔄 Restarting demo...")
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                "streamlit_app.py",
                "--server.port", "8501"
            ])
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit. Please install manually:")
            print("   pip install streamlit")
            print("   Then run: streamlit run streamlit_app.py")
            
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        print("✅ Thank you for using the NBA Prediction System!")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        print("🔧 Try running manually: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 