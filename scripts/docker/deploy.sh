#!/bin/bash
# AI Video Processing 部署脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查Docker和Docker Compose
check_requirements() {
    print_step "检查系统要求..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    print_message "Docker和Docker Compose已安装"
}

# 创建必要的目录
create_directories() {
    print_step "创建必要的目录..."
    
    mkdir -p input_data/downloads
    mkdir -p output_data
    mkdir -p logs
    mkdir -p nginx/ssl
    
    print_message "目录创建完成"
}

# 生成环境配置文件
setup_env() {
    print_step "设置环境配置..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_message "已从env.example创建.env文件，请根据需要修改配置"
        else
            print_warning ".env文件不存在，将使用默认配置"
        fi
    else
        print_message ".env文件已存在"
    fi
}

# 构建镜像
build_images() {
    print_step "构建Docker镜像..."
    
    docker-compose build --no-cache
    print_message "镜像构建完成"
}

# 启动服务
start_services() {
    print_step "启动服务..."
    
    # 启动基础服务
    docker-compose up -d redis minio
    print_message "基础服务已启动"
    
    # 等待Redis启动
    print_message "等待Redis启动..."
    sleep 10
    
    # 启动应用服务
    docker-compose up -d api worker flower
    print_message "应用服务已启动"
    
    # 可选：启动Nginx
    if [ "$1" = "--with-nginx" ]; then
        docker-compose up -d nginx
        print_message "Nginx已启动"
    fi
    
    # 可选：启动监控服务
    if [ "$1" = "--with-monitoring" ]; then
        docker-compose --profile monitoring up -d
        print_message "监控服务已启动"
    fi
}

# 检查服务状态
check_services() {
    print_step "检查服务状态..."
    
    sleep 15
    
    # 检查API服务
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_message "✅ API服务运行正常"
    else
        print_error "❌ API服务异常"
    fi
    
    # 检查Flower
    if curl -f http://localhost:5555 > /dev/null 2>&1; then
        print_message "✅ Flower监控运行正常"
    else
        print_error "❌ Flower监控异常"
    fi
    
    # 检查MinIO
    if curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        print_message "✅ MinIO存储运行正常"
    else
        print_error "❌ MinIO存储异常"
    fi
}

# 显示服务信息
show_info() {
    print_step "服务信息："
    echo ""
    echo "🚀 AI Video Processing 服务已启动"
    echo ""
    echo "📡 服务端点："
    echo "   • API服务:      http://localhost:8000"
    echo "   • API文档:      http://localhost:8000/docs"
    echo "   • 健康检查:     http://localhost:8000/health"
    echo "   • Flower监控:   http://localhost:5555 (使用 .env 中配置的账号)"
    echo "   • MinIO存储:    http://localhost:9001 (使用 .env 中配置的账号)"
    echo ""
    echo "🔧 管理命令："
    echo "   • 查看日志:     docker-compose logs -f [service]"
    echo "   • 重启服务:     docker-compose restart [service]"
    echo "   • 停止服务:     docker-compose down"
    echo "   • 查看状态:     docker-compose ps"
    echo ""
    echo "📁 重要目录："
    echo "   • 输入文件:     ./input_data/"
    echo "   • 输出文件:     ./output_data/"
    echo "   • 日志文件:     ./logs/"
    echo ""
}

# 主函数
main() {
    echo "🎬 AI Video Processing Docker部署脚本"
    echo "========================================"
    echo ""
    
    check_requirements
    create_directories
    setup_env
    build_images
    start_services "$1"
    check_services
    show_info
    
    print_message "🎉 部署完成！"
}

# 处理命令行参数
case "$1" in
    --help|-h)
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  --with-nginx      启用Nginx反向代理"
        echo "  --with-monitoring 启用Prometheus+Grafana监控"
        echo "  --help, -h        显示此帮助信息"
        echo ""
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
