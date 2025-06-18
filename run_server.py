#!/usr/bin/env python3
"""
Startup script for the TDS Virtual Teaching Assistant API
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("Checking requirements...")
    
    # Check if discourse_posts.json exists
    if not Path("discourse_posts.json").exists():
        print("❌ Error: discourse_posts.json not found!")
        print("   Please ensure the discourse posts data file is in the current directory.")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found!")
        return False
    
    print("✅ Required files found")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("Starting TDS Virtual Teaching Assistant API...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Import and run the app
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("TDS Virtual Teaching Assistant API")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()