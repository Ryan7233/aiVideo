#!/bin/bash

# AI Video Clipper 项目设置脚本

set -e

echo "🚀 Setting up AI Video Clipper Project..."
echo "========================================"

# 检查Python版本
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE 'Python [0-9]+\.[0-9]+')
echo "Found: $python_version"

# 检查FFmpeg
echo "📋 Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -n1 | grep -oE 'ffmpeg version [0-9]+\.[0-9]+')
    echo "✅ Found: $ffmpeg_version"
else
    echo "❌ FFmpeg not found. Please install FFmpeg first:"
    echo "   macOS: brew install ffmpeg"
    echo "   Ubuntu: sudo apt install ffmpeg"
    echo "   CentOS: sudo yum install ffmpeg"
    exit 1
fi

# 创建虚拟环境
echo "🐍 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# 激活虚拟环境
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# 升级pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# 创建必要的目录
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p output_data
mkdir -p input_data
mkdir -p tests

# 设置权限
echo "🔐 Setting script permissions..."
chmod +x scripts/*.sh
chmod +x start_server.py

# 创建环境变量示例文件
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
GEMINI_API_BASE=http://localhost:8080
CUT_API_BASE=http://localhost:8081

# Video Processing
MIN_CLIP_DURATION=25
MAX_CLIP_DURATION=60
VIDEO_FPS=30
VIDEO_CRF=23
AUDIO_BITRATE=128k

# Storage
UPLOAD_BUCKET=clips
UPLOAD_BASE_URL=https://storage.example.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security
ALLOWED_VIDEO_EXTENSIONS=.mp4,.avi,.mov,.mkv
MAX_FILE_SIZE=500MB
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "========================================"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Place your video files in input_data/"
echo "3. Start the server: python start_server.py"
echo "4. Run the pipeline: python run_full_pipeline.py"
echo ""
echo "📚 Useful commands:"
echo "  Start server:   python start_server.py"
echo "  Run pipeline:   python run_full_pipeline.py"
echo "  Run tests:      pytest tests/"
echo "  Health check:   curl http://localhost:8000/health"
echo ""
