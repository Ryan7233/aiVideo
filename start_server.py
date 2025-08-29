#!/usr/bin/env python3
"""
AI Video Clipper API 服务启动脚本
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import API_HOST, API_PORT

def main():
    """启动API服务"""
    print("🚀 Starting AI Video Clipper API Server...")
    print(f"📍 Host: {API_HOST}")
    print(f"🔢 Port: {API_PORT}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("="*50)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("output_data").mkdir(exist_ok=True)
    
    try:
        uvicorn.run(
            "api.main:app",
            host=API_HOST,
            port=API_PORT,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n⚠️  Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
