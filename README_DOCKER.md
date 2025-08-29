# AI Video Processing - Docker部署版

## 🎯 项目概述

这是一个完整的AI视频处理平台，支持智能视频切片、场景检测、音频分析等功能。本版本提供了完整的Docker化部署方案，支持一键部署和扩容。

## ✨ 核心功能

### 🎬 智能视频处理
- **智能切片**：基于场景检测、音频能量、运动分析的多维度评分
- **格式转换**：支持9:16竖屏视频生成
- **字幕处理**：自动提取和烧录字幕
- **质量优化**：自动黑屏检测和音频优化

### 🔧 技术架构
- **FastAPI**：高性能异步API框架
- **Celery**：分布式任务队列
- **Redis**：消息队列和缓存
- **FFmpeg**：专业视频处理
- **Docker**：容器化部署

### 📊 监控和管理
- **Flower**：Celery任务监控
- **Prometheus + Grafana**：系统监控
- **MinIO**：对象存储
- **Nginx**：反向代理和负载均衡

## 🚀 快速部署

### 1. 环境要求
```bash
# 系统要求
- Docker >= 20.10
- Docker Compose >= 2.0
- 4GB+ RAM
- 20GB+ 磁盘空间
```

### 2. 一键部署
```bash
# 克隆项目
git clone <your-repo-url>
cd aiVideo

# 执行部署
chmod +x scripts/docker/deploy.sh
./scripts/docker/deploy.sh
```

### 3. 服务访问
| 服务 | 地址 | 用户名/密码 |
|------|------|-------------|
| 🎯 API服务 | http://localhost:8000 | - |
| 📚 API文档 | http://localhost:8000/docs | - |
| 🌸 任务监控 | http://localhost:5555 | admin/admin123 |
| 📦 对象存储 | http://localhost:9001 | minioadmin/minioadmin123 |

## 🎮 API使用示例

### 智能视频分析
```bash
curl -X POST http://localhost:8000/analyze_video \
  -H 'Content-Type: application/json' \
  -d '{"url":"file:///path/to/video.mp4"}'
```

### 智能切片生成
```bash
curl -X POST http://localhost:8000/auto_intro \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "file:///path/to/video.mp4",
    "min_sec": 15,
    "max_sec": 25,
    "smart_mode": true,
    "output": "output_data/smart_clip.mp4"
  }'
```

### 异步任务处理
```bash
# 提交任务
curl -X POST http://localhost:8000/tasks/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "file:///path/to/video.mp4",
    "min_sec": 20,
    "max_sec": 30
  }'

# 查询状态
curl http://localhost:8000/tasks/status/{task_id}
```

## 🛠️ 运维管理

### 服务管理
```bash
# 查看状态
./scripts/docker/manage.sh status

# 查看日志
./scripts/docker/manage.sh logs api

# 重启服务
./scripts/docker/manage.sh restart worker

# 健康检查
./scripts/docker/manage.sh health
```

### 数据备份
```bash
# 备份数据
./scripts/docker/manage.sh backup

# 清理资源
./scripts/docker/manage.sh cleanup
```

### 服务扩容
```bash
# 编辑docker-compose.yml
services:
  worker:
    deploy:
      replicas: 4  # 扩展到4个Worker实例

# 应用更改
docker-compose up -d --scale worker=4
```

## 🏗️ 架构说明

### 服务拓扑
```
                    ┌─────────────────┐
                    │   Nginx (80)    │
                    │  反向代理 + SSL   │
                    └─────────┬───────┘
                              │
                    ┌─────────┴───────┐
                    │  API (8000)     │
                    │ FastAPI + 智能AI │
                    └─────────┬───────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
  ┌─────────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐
  │ Worker1 (Celery)│ │ Worker2 (Celery)│ │  Redis (6379) │
  │   视频处理任务    │ │   视频处理任务    │ │ 消息队列+缓存   │
  └─────────────────┘ └─────────────────┘ └───────────────┘

  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐
  │ Flower (5555)   │ │  MinIO (9000)   │ │ Grafana (3000)│
  │   任务监控       │ │   对象存储       │ │   监控仪表板   │
  └─────────────────┘ └─────────────────┘ └───────────────┘
```

### 数据流程
1. **请求接收**：Nginx接收HTTP请求并路由到API
2. **任务分发**：API将长时间任务发送到Celery队列
3. **异步处理**：Worker从Redis队列获取任务并处理
4. **结果存储**：处理结果存储到MinIO，状态更新到Redis
5. **监控反馈**：Flower实时显示任务状态，Grafana展示系统指标

## 🔧 配置说明

### 环境变量
```bash
# 核心配置
API_HOST=0.0.0.0
API_PORT=8000
REDIS_HOST=redis
CELERY_BROKER_URL=redis://redis:6379/0

# 视频处理参数
VIDEO_FPS=30
VIDEO_CRF=23
MIN_CLIP_DURATION=10
MAX_CLIP_DURATION=60

# 存储配置
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
```

### 性能调优
```yaml
# Worker并发
services:
  worker:
    command: celery -A worker.celery_app worker --concurrency=4
    deploy:
      replicas: 2

# 资源限制
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## 📊 监控指标

### 系统监控
- **CPU使用率**：各容器CPU消耗
- **内存使用**：内存占用和泄漏检测
- **磁盘I/O**：存储读写性能
- **网络流量**：服务间通信状况

### 业务监控
- **任务队列长度**：待处理任务数量
- **处理成功率**：任务成功/失败比例
- **平均处理时间**：视频处理耗时统计
- **API响应时间**：接口性能指标

### 告警配置
```yaml
# Prometheus规则示例
groups:
  - name: ai-video-alerts
    rules:
    - alert: HighCPUUsage
      expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    - alert: TaskQueueTooLong
      expr: celery_queue_length > 100
```

## 🔐 安全配置

### 生产环境清单
- [ ] 修改所有默认密码
- [ ] 启用HTTPS/SSL
- [ ] 配置防火墙规则
- [ ] 设置资源限制
- [ ] 启用访问日志
- [ ] 配置备份策略

### SSL证书配置
```bash
# 生成自签名证书（测试用）
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem

# 启用HTTPS
./scripts/docker/deploy.sh --with-nginx
```

## 🚨 故障排除

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :8000
   
   # 查看详细错误
   ./scripts/docker/manage.sh logs api
   ```

2. **任务处理失败**
   ```bash
   # 检查Worker状态
   ./scripts/docker/manage.sh logs worker
   
   # 重启Worker
   ./scripts/docker/manage.sh restart worker
   ```

3. **磁盘空间不足**
   ```bash
   # 清理Docker资源
   ./scripts/docker/manage.sh cleanup
   
   # 查看磁盘使用
   df -h
   docker system df
   ```

### 性能问题
```bash
# 查看资源使用
docker stats

# 调整Worker并发
docker-compose up -d --scale worker=4

# 检查队列积压
curl http://localhost:5555/api/queues
```

## 📈 扩展开发

### 添加新的处理算法
1. 在`core/`目录添加新模块
2. 在`api/main.py`中注册新端点
3. 在`worker/tasks.py`中添加异步任务
4. 更新Docker镜像并重新部署

### 集成外部服务
```python
# 示例：集成云存储
from cloud_storage import upload_to_cloud

@app.post("/upload_cloud")
async def upload_to_cloud_storage(file: UploadFile):
    # 处理逻辑
    pass
```

## 📝 版本历史

### v1.0.0 (当前版本)
- ✅ 完整Docker化部署
- ✅ 智能视频切片算法
- ✅ 异步任务处理
- ✅ 监控和管理界面
- ✅ 负载均衡和高可用

### 路线图
- 🔄 Whisper语音识别集成
- 🔄 Web管理界面
- 🔄 用户认证系统
- 🔄 计费系统
- 🔄 平台发布集成

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 💬 支持

- 📧 邮件：support@ai-video.com
- 💬 社区：[GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 问题报告：[GitHub Issues](https://github.com/your-repo/issues)

---

**🎬 让AI为您的视频创作赋能！**
