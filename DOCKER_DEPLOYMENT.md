# Docker部署指南

## 🚀 快速开始

### 1. 环境准备

确保系统已安装以下软件：
- Docker (>= 20.10)
- Docker Compose (>= 2.0)

```bash
# 检查版本
docker --version
docker-compose --version
```

### 2. 一键部署

```bash
# 克隆项目
git clone <repository-url>
cd aiVideo

# 执行部署脚本
chmod +x scripts/docker/deploy.sh
./scripts/docker/deploy.sh
```

### 3. 访问服务

部署完成后，可以访问以下服务：

| 服务 | 地址 | 说明 |
|------|------|------|
| API服务 | http://localhost:8000 | 主要API接口 |
| API文档 | http://localhost:8000/docs | Swagger文档 |
| Flower监控 | http://localhost:5555 | Celery任务监控 |
| MinIO存储 | http://localhost:9001 | 对象存储管理 |

## 📋 部署选项

### 基础部署
```bash
./scripts/docker/deploy.sh
```

### 包含Nginx反向代理
```bash
./scripts/docker/deploy.sh --with-nginx
```

### 包含监控服务
```bash
./scripts/docker/deploy.sh --with-monitoring
```

## 🔧 服务管理

使用管理脚本进行日常运维：

```bash
chmod +x scripts/docker/manage.sh

# 查看服务状态
./scripts/docker/manage.sh status

# 查看日志
./scripts/docker/manage.sh logs          # 所有服务
./scripts/docker/manage.sh logs api     # 特定服务

# 重启服务
./scripts/docker/manage.sh restart      # 所有服务
./scripts/docker/manage.sh restart api  # 特定服务

# 健康检查
./scripts/docker/manage.sh health

# 备份数据
./scripts/docker/manage.sh backup

# 更新服务
./scripts/docker/manage.sh update
```

## 🏗️ 架构说明

### 服务组件

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (可选)                         │
│                  反向代理 + 负载均衡                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                  AI Video API                          │
│              FastAPI + 智能视频处理                      │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                  Celery Workers                        │
│                异步任务处理 (2实例)                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────┐
│                    Redis                               │
│              消息队列 + 结果存储                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   MinIO                                │
│                 对象存储服务                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Flower + Prometheus + Grafana              │
│                监控和可视化 (可选)                         │
└─────────────────────────────────────────────────────────┘
```

### 网络配置

- 自定义网络：`ai-video-network` (172.20.0.0/16)
- 服务间通信通过服务名解析
- 外部访问通过端口映射

### 数据持久化

| 卷名称 | 挂载点 | 说明 |
|--------|--------|------|
| redis_data | /data | Redis数据持久化 |
| minio_data | /data | MinIO对象存储 |
| prometheus_data | /prometheus | 监控数据 |
| grafana_data | /var/lib/grafana | 仪表板配置 |

## 🔧 配置说明

### 环境变量

复制并修改环境配置：
```bash
cp env.example .env
```

主要配置项：

```bash
# API配置
API_HOST=0.0.0.0
API_PORT=8000

# Redis配置
REDIS_HOST=redis
REDIS_PORT=6379

# Celery配置
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1

# MinIO配置
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
```

### 性能调优

#### Worker并发数
```yaml
# docker-compose.yml
deploy:
  replicas: 2  # Worker实例数量
```

#### 内存限制
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

#### 文件上传大小
```yaml
# nginx/nginx.conf
client_max_body_size 1G;
```

## 🐛 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 修改docker-compose.yml中的端口映射
   ports:
     - "8001:8000"  # 改为8001
   ```

2. **磁盘空间不足**
   ```bash
   # 清理Docker资源
   ./scripts/docker/manage.sh cleanup
   
   # 清理包括数据卷（谨慎使用）
   ./scripts/docker/manage.sh cleanup --volumes
   ```

3. **服务启动失败**
   ```bash
   # 查看详细日志
   ./scripts/docker/manage.sh logs [service_name]
   
   # 重启服务
   ./scripts/docker/manage.sh restart [service_name]
   ```

4. **健康检查失败**
   ```bash
   # 执行健康检查
   ./scripts/docker/manage.sh health
   
   # 手动测试API
   curl http://localhost:8000/health
   ```

### 日志查看

```bash
# 实时查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f api

# 查看最近100行日志
docker-compose logs --tail=100 api
```

### 数据恢复

```bash
# 从备份恢复
cp -r backup/20231201_120000/* ./

# 重启服务
docker-compose restart
```

## 🔐 安全配置

### 生产环境建议

1. **修改默认密码**
   ```bash
   # .env文件
   MINIO_ROOT_PASSWORD=your_secure_password
   FLOWER_BASIC_AUTH=your_user:your_password
   ```

2. **启用HTTPS**
   ```bash
   # 生成SSL证书
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem
   
   # 启用Nginx
   ./scripts/docker/deploy.sh --with-nginx
   ```

3. **网络隔离**
   ```yaml
   # 移除不必要的端口映射
   # ports:
   #   - "6379:6379"  # Redis不对外暴露
   ```

4. **资源限制**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
   ```

## 📊 监控配置

### 启用监控
```bash
./scripts/docker/deploy.sh --with-monitoring
```

### 访问监控服务
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

### 自定义指标

在API中添加Prometheus指标：
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
```

## 🚀 生产部署

### 硬件要求

| 组件 | 最小配置 | 推荐配置 |
|------|----------|----------|
| CPU | 2核 | 4核+ |
| 内存 | 4GB | 8GB+ |
| 存储 | 20GB | 100GB+ SSD |
| 网络 | 100Mbps | 1Gbps |

### 部署检查清单

- [ ] 环境变量配置正确
- [ ] SSL证书配置（如需要）
- [ ] 防火墙规则设置
- [ ] 数据备份策略
- [ ] 监控告警配置
- [ ] 日志轮转配置
- [ ] 资源限制设置

### 扩容策略

1. **水平扩容Worker**
   ```yaml
   deploy:
     replicas: 4  # 增加Worker实例
   ```

2. **API负载均衡**
   ```yaml
   api:
     deploy:
       replicas: 2
   ```

3. **Redis集群**
   ```yaml
   # 使用Redis Cluster或Sentinel
   ```

## 📝 更新日志

### v1.0.0
- 初始Docker化部署
- 支持API、Worker、Redis、MinIO
- 包含监控和反向代理配置
- 提供完整的管理脚本
