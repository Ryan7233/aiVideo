#!/bin/bash
# AI Video Processing 管理脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 显示状态
show_status() {
    print_step "服务状态："
    docker-compose ps
    echo ""
    
    print_step "资源使用情况："
    docker stats --no-stream $(docker-compose ps -q) 2>/dev/null || echo "无运行中的容器"
}

# 查看日志
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        print_step "显示所有服务日志："
        docker-compose logs --tail=100 -f
    else
        print_step "显示 $service 服务日志："
        docker-compose logs --tail=100 -f "$service"
    fi
}

# 重启服务
restart_service() {
    local service=$1
    if [ -z "$service" ]; then
        print_step "重启所有服务..."
        docker-compose restart
    else
        print_step "重启 $service 服务..."
        docker-compose restart "$service"
    fi
    print_message "重启完成"
}

# 停止服务
stop_services() {
    print_step "停止所有服务..."
    docker-compose down
    print_message "服务已停止"
}

# 清理资源
cleanup() {
    print_step "清理Docker资源..."
    
    # 停止并删除容器
    docker-compose down --remove-orphans
    
    # 删除未使用的镜像
    docker image prune -f
    
    # 删除未使用的卷（谨慎使用）
    if [ "$1" = "--volumes" ]; then
        print_step "删除数据卷..."
        docker-compose down -v
        docker volume prune -f
    fi
    
    print_message "清理完成"
}

# 备份数据
backup_data() {
    local backup_dir="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_step "备份数据到 $backup_dir..."
    
    # 备份输出文件
    if [ -d "output_data" ]; then
        cp -r output_data "$backup_dir/"
        print_message "输出文件已备份"
    fi
    
    # 备份日志
    if [ -d "logs" ]; then
        cp -r logs "$backup_dir/"
        print_message "日志文件已备份"
    fi
    
    # 备份Redis数据
    docker-compose exec redis redis-cli BGSAVE
    docker cp $(docker-compose ps -q redis):/data/dump.rdb "$backup_dir/"
    print_message "Redis数据已备份"
    
    print_message "备份完成：$backup_dir"
}

# 更新服务
update_services() {
    print_step "更新服务..."
    
    # 拉取最新代码（如果是git仓库）
    if [ -d ".git" ]; then
        git pull
        print_message "代码已更新"
    fi
    
    # 重新构建镜像
    docker-compose build --no-cache
    print_message "镜像已重建"
    
    # 重启服务
    docker-compose up -d
    print_message "服务已更新"
}

# 健康检查
health_check() {
    print_step "执行健康检查..."
    
    local all_healthy=true
    
    # 检查API服务
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_message "✅ API服务健康"
    else
        print_error "❌ API服务异常"
        all_healthy=false
    fi
    
    # 检查Flower
    if curl -f http://localhost:5555 > /dev/null 2>&1; then
        print_message "✅ Flower监控健康"
    else
        print_error "❌ Flower监控异常"
        all_healthy=false
    fi
    
    # 检查Redis
    if docker-compose exec redis redis-cli ping | grep -q "PONG"; then
        print_message "✅ Redis服务健康"
    else
        print_error "❌ Redis服务异常"
        all_healthy=false
    fi
    
    # 检查MinIO
    if curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        print_message "✅ MinIO存储健康"
    else
        print_error "❌ MinIO存储异常"
        all_healthy=false
    fi
    
    if [ "$all_healthy" = true ]; then
        print_message "🎉 所有服务运行正常"
        return 0
    else
        print_error "⚠️  部分服务异常，请检查日志"
        return 1
    fi
}

# 显示帮助
show_help() {
    echo "AI Video Processing 管理脚本"
    echo ""
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  status              显示服务状态"
    echo "  logs [service]      查看日志（可指定服务名）"
    echo "  restart [service]   重启服务（可指定服务名）"
    echo "  stop                停止所有服务"
    echo "  cleanup [--volumes] 清理资源（--volumes删除数据卷）"
    echo "  backup              备份数据"
    echo "  update              更新服务"
    echo "  health              健康检查"
    echo "  help                显示此帮助"
    echo ""
    echo "服务名:"
    echo "  api                 API服务"
    echo "  worker              Celery Worker"
    echo "  flower              Flower监控"
    echo "  redis               Redis缓存"
    echo "  minio               MinIO存储"
    echo "  nginx               Nginx代理"
    echo ""
    echo "示例:"
    echo "  $0 status           # 显示所有服务状态"
    echo "  $0 logs api         # 查看API服务日志"
    echo "  $0 restart worker   # 重启Worker服务"
    echo "  $0 cleanup --volumes # 清理包括数据卷"
    echo ""
}

# 主函数
main() {
    case "$1" in
        status)
            show_status
            ;;
        logs)
            show_logs "$2"
            ;;
        restart)
            restart_service "$2"
            ;;
        stop)
            stop_services
            ;;
        cleanup)
            cleanup "$2"
            ;;
        backup)
            backup_data
            ;;
        update)
            update_services
            ;;
        health)
            health_check
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 检查是否在正确的目录
if [ ! -f "docker-compose.yml" ]; then
    print_error "请在项目根目录运行此脚本（包含docker-compose.yml的目录）"
    exit 1
fi

main "$@"
