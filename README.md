# 🎬 AI Video Clipper - 智能视频切片平台

一个基于AI的智能视频处理平台，集成了**ASR语音识别**、**语义分析**和**智能选段**功能，能够自动从长视频中提取最有价值的片段。

## 🖥️ 可视化前端（New）

- 内置前端已集成并随 API 一起提供静态资源服务。
- 启动服务后，直接在浏览器访问：
  - Web UI: `http://127.0.0.1:8000/static/index.html`
  - API 文档: `http://127.0.0.1:8000/docs`

说明：当前根路径 `/` 用于健康检查与基础信息返回；前端通过 `/static` 提供。如果需要将首页改为直接渲染前端，可在 `api/main.py` 中移除或调整重复声明的根路由。

## 🌟 核心特性

### 🧠 ASR增强智能选段
- **多算法融合**：结合视觉分析、语义分析、音频分析的综合评分
- **语音识别集成**：基于Faster-Whisper的高精度ASR转录
- **智能候选生成**：质量片段、峰值时刻、内容驱动的三层选段策略
- **动态阈值优化**：根据内容质量自动调整选段标准

### 📊 深度语义分析
- **关键词提取**：TF-IDF算法 + 重要性修饰符
- **情感分析**：多维度情感评分（积极、消极、中性）
- **主题识别**：技术、商业、教育、娱乐、健康等领域分类
- **内容质量评分**：六维度综合评估系统

### 🎥 智能视频处理
- **9:16竖屏输出**：专为移动端优化的视频格式
- **高质量编码**：H.264编码，支持快速启动
- **批量处理**：支持生成多个不重叠的智能片段
- **实时预览**：提供详细的处理进度和质量评分

### ⚡ 高性能架构
- **异步任务队列**：Celery + Redis支持后台处理
- **RESTful API**：FastAPI现代异步Web框架
- **Docker部署**：完整的容器化解决方案
- **监控系统**：Flower + Prometheus + Grafana集成

## 🚀 快速开始

### 环境要求
- Python 3.11+
- FFmpeg
- Redis
- 8GB+ RAM（用于AI模型）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/your-username/aiVideo.git
cd aiVideo
```

2. **创建虚拟环境**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **启动服务**
```bash
# 启动Redis
redis-server

# 启动API服务器
./scripts/run_server.sh

# 启动Celery Worker
./scripts/start_worker.sh

# 启动监控面板（可选）
./scripts/start_flower.sh
```

启动后访问：
- Web UI: `http://127.0.0.1:8000/static/index.html`
- API 文档: `http://127.0.0.1:8000/docs`

或使用 Python 启动（开发模式 自动重载）：
```bash
python start_server.py
```

### Docker部署

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api
```

## 📖 API文档

### 前端相关（New）
- `POST /analyze_video`：视频下载/基础分析，返回本地 `video_path`
- `POST /video/multi_segment_clipping`：多片段智能剪辑与 9:16 合成，返回输出视频与片段信息
- `POST /upload/video`：上传本地视频到 `output_data/uploads`
- `POST /upload/photos`：批量上传图片到 `output_data/uploads`
- 静态资源：
  - `/static/*` 前端静态文件（`frontend/`）
  - `/output/*` 输出文件静态访问（`output_data/`）

### 语义分析
```bash
curl -X POST "http://127.0.0.1:8000/semantic/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is amazing AI technology!",
    "include_keywords": true,
    "include_sentiment": true,
    "include_topics": true,
    "include_quality": true
  }'
```

### ASR增强智能切片
```bash
curl -X POST "http://127.0.0.1:8000/smart_clipping/asr_enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "file:///path/to/video.mp4",
    "min_sec": 15,
    "max_sec": 25,
    "count": 2,
    "model_size": "base"
  }'
```

### 异步任务
```bash
# 提交任务
curl -X POST "http://127.0.0.1:8000/tasks/smart_clipping/enqueue" \
  -H "Content-Type: application/json" \
  -d '{"url": "file:///path/to/video.mp4", "count": 3}'

# 查询状态
curl "http://127.0.0.1:8000/tasks/status/{task_id}"
```

### 小红书与拼图/封面相关（New）
- `POST /xiaohongshu/generate_collage`：后端生成小红书风格拼图（返回 base64）
- `POST /xiaohongshu/render_page`：渲染单页（单图/拼图）
- `GET /xiaohongshu/layouts`：可用拼图布局
- `POST /cover/generate`：智能封面生成（配合选题、主题色）
- `GET /cover/templates`：可用封面模板
- `POST /image/decorations/smart`：智能装饰图片

## 🏗️ 项目结构

```
aiVideo/
├── api/                    # FastAPI应用
│   └── main.py            # 主API入口
├── frontend/               # 新增：内置前端（静态资源）
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── core/                   # 核心功能模块
│   ├── config.py          # 配置管理
│   ├── settings.py        # 环境设置
│   ├── smart_clipping.py  # 基础智能选段
│   ├── semantic_analysis.py  # 语义分析引擎
│   ├── asr_smart_clipping.py # ASR增强选段
│   └── whisper_asr.py     # Whisper ASR服务
│   ├── advanced_photo_ranking.py   # 新增：高级照片选优
│   ├── image_decorator.py          # 新增：图片智能装饰
│   ├── llm_service.py              # 新增：多LLM服务（OpenAI/Claude/Gemini/智谱/通义）
│   ├── smart_cover_generator.py    # 新增：智能封面生成
│   ├── advanced_collage_generator.py # 新增：高级拼图
│   ├── xiaohongshu_collage_generator.py # 新增：小红书拼图
│   └── xiaohongshu_publisher.py    # 新增：小红书发布集成
├── worker/                 # Celery异步任务
│   ├── celery_app.py      # Celery配置
│   └── tasks.py           # 任务定义
├── routers/                # API路由
│   └── tasks.py           # 任务相关路由
├── scripts/                # 部署脚本
│   ├── run_server.sh      # 启动API服务器
│   ├── start_worker.sh    # 启动Worker
│   └── start_flower.sh    # 启动监控
├── docker-compose.yml      # Docker编排
├── Dockerfile.api          # API服务镜像
├── Dockerfile.worker       # Worker服务镜像
└── requirements.txt        # Python依赖
```

## 🧪 测试示例

### 测试视频处理
```python
import requests

# 语义分析测试
response = requests.post("http://127.0.0.1:8000/semantic/analyze", json={
    "text": "This video explains machine learning concepts in an engaging way.",
    "include_keywords": True,
    "include_sentiment": True
})
print(response.json())

# 智能切片测试
response = requests.post("http://127.0.0.1:8000/smart_clipping/asr_enhanced", json={
    "url": "file:///path/to/your/video.mp4",
    "min_sec": 15,
    "max_sec": 30,
    "count": 2
})
print(response.json())
```

## 📊 性能指标

- **ASR准确率**：95%+（英文内容）
- **处理速度**：实时转录，2-5倍速处理
- **选段精度**：基于多维度评分的智能选择
- **并发支持**：支持多任务并行处理
- **内存使用**：2-4GB（取决于模型大小）

## 🔧 配置选项

### 环境变量
```bash
# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 视频处理参数
MIN_CLIP_DURATION=15
MAX_CLIP_DURATION=30
VIDEO_CRF=23
AUDIO_BITRATE=128k

# Celery配置
CELERY_BROKER_URL=redis://127.0.0.1:6379/0
CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1

# Whisper模型配置
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=auto
```

### LLM配置（New）
前端的文案生成、小红书内容等功能可以接入多种大模型。复制 `env_example.txt` 为 `.env` 并填写任意可用的 API Key（至少一种）：

```bash
# OpenAI
OPENAI_API_KEY=...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Claude
CLAUDE_API_KEY=...

# Google Gemini
GEMINI_API_KEY=...

# 智谱/通义
ZHIPU_API_KEY=...
QWEN_API_KEY=...
```

未配置时，后端会启用本地降级策略，仍可体验基础流程，但生成类能力会受限。

### 模型选择
- `tiny`：最快，适合实时处理
- `base`：平衡速度和精度（推荐）
- `small`：更高精度
- `medium`：高精度，需要更多资源
- `large`：最高精度，需要大量资源

## 🛠️ 开发指南

### 添加新的语义分析功能
1. 在 `core/semantic_analysis.py` 中添加新的分析方法
2. 更新API端点以支持新功能
3. 编写测试用例

### 扩展智能选段算法
1. 在 `core/asr_smart_clipping.py` 中实现新算法
2. 更新评分权重配置
3. 测试不同类型的视频内容

### 部署到生产环境
1. 使用Docker Compose进行容器化部署
2. 配置Nginx反向代理
3. 设置Prometheus + Grafana监控
4. 配置日志轮转和备份策略

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🆘 支持

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 📖 文档: [完整文档](docs/)
- 🐛 Bug报告: [Issues](https://github.com/your-username/aiVideo/issues)

## 🙏 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 强大的ASR模型
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - 高性能Whisper实现
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Celery](https://docs.celeryproject.org/) - 分布式任务队列
- [FFmpeg](https://ffmpeg.org/) - 多媒体处理框架

---

⭐ 如果这个项目对您有帮助，请给我们一个Star！
