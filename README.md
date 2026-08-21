# AI Video Clipper

一个本地优先的智能视频切片与小红书内容生产工具。当前主链路可以完成视频上传/安全下载、ASR 字幕、语义/画面/音频联合评分、多片段选择、9:16 转码与合并；照片排序、拼图、封面和文案生成作为扩展能力提供。

## 当前能力边界

| 能力 | 状态 | 说明 |
|---|---|---|
| 视频上传、远程下载 | 可用 | 分块写入；远程地址限制为公网 HTTP/HTTPS，并限制下载大小 |
| 多片段智能剪辑 | 可用 | 使用真实视频时长、画面变化、运动、音频 RMS 和可选 ASR 文本评分 |
| 9:16 转码、字幕烧录 | 可用 | 依赖 FFmpeg |
| Faster-Whisper ASR | 可用 | 首次使用会下载所选模型；可关闭内容分析以跳过 ASR |
| 照片排序、拼图、封面 | 可用/可降级 | CLIP 等可选模型缺失时使用基础策略 |
| LLM 文案 | 可用/可降级 | 配置有效 API Key 时调用模型，否则明确返回本地模板 |
| `/segment`、`/captions` | 演示接口 | 响应带 `mode: simulation`；正式流程使用主剪辑和 LLM 接口 |
| 云存储 `/upload` | 模拟 | 不会把文件上传到外部存储 |
| 小红书 OAuth、直发、统计 | 模拟 | 不会授权、发布或生成真实笔记 ID，请下载后手动发布 |

## 快速启动

要求：Python 3.11+、FFmpeg。只有使用 Celery 后台任务时才需要 Redis。

```bash
git clone https://github.com/Ryan7233/aiVideo.git
cd aiVideo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
python start_server.py
```

启动后访问：

- Web UI：<http://127.0.0.1:8000/>
- OpenAPI：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

运行时目录会自动创建。默认位于仓库下的 `input_data/`、`output_data/`、`logs/` 和 `models/`；可通过 `AIVIDEO_DATA_DIR` 把数据根目录移到其他位置。

## 主流程

1. 前端调用 `POST /upload/video` 上传本地视频，或通过 `POST /analyze_video` 下载公网视频。
2. `POST /video/multi_segment_clipping` 读取视频与可选字幕；没有字幕且启用内容分析时运行 Faster-Whisper。
3. `core/video_workflow.py` 将语义、画面与音频分数归一化，按开场/高潮/结尾约束选取互不重叠的片段。
4. FFmpeg 把片段转成 1080×1920 H.264/AAC 文件并合并，结果通过 `/output/...` 返回。

示例：

```bash
curl -F 'file=@sample.mp4' http://127.0.0.1:8000/upload/video

curl -X POST http://127.0.0.1:8000/video/multi_segment_clipping \
  -H 'Content-Type: application/json' \
  -d '{
    "video_path": "input_data/uploads/video_xxx.mp4",
    "topic": "核心主题",
    "target_segments": 3,
    "total_duration": 60,
    "enable_content_analysis": true,
    "asr_model_size": "base"
  }'
```

## 安全配置

默认只接受运行时受管目录内的本地文件，并拒绝回环、内网、链路本地等远程地址，避免路径穿越和 SSRF。除非处于可信的单机环境，不要设置 `ALLOW_UNSAFE_LOCAL_PATHS=true`。

生产环境建议至少设置：

```bash
AIVIDEO_API_KEY=生成一个足够长的随机值
CORS_ALLOWED_ORIGINS=https://your-domain.example
MAX_FILE_SIZE=524288000
MAX_IMAGE_FILE_SIZE=26214400
```

设置 `AIVIDEO_API_KEY` 后，受保护接口需要 `X-API-Key` 请求头；`/output` 产物也会被保护。内置静态前端目前不会自动保存或注入 API Key，因此生产部署应由反向代理完成认证和请求头注入，或使用受信 API 客户端。

所有 AI 服务密钥默认留空。不要把 `.env`、模型或运行时媒体提交到 Git。Docker Compose 中的 Flower 和 MinIO 已放入可选 profile，启用前必须替换示例凭据：

```bash
docker compose --profile observability up -d flower
docker compose --profile storage up -d minio
```

## 结构

```text
api/main.py                 FastAPI 路由与输入校验
core/runtime.py             运行时目录、路径边界、安全下载
core/video_workflow.py      主剪辑编排与 FFmpeg 合并
core/smart_clipping.py      真实画面/运动/音频测量
core/whisper_asr.py         Faster-Whisper 模型缓存与转录
core/semantic_analysis.py   文本语义评分
worker/tasks.py             Celery 后台任务
frontend/                   内置 Web UI
tests/                      API、安全边界和 FFmpeg 集成测试
```

## 测试与检查

```bash
pytest -q
python -m compileall -q api core routers worker tests
node --check frontend/script.js
bash -n scripts/*.sh
```

测试套件会生成一段短视频并实际执行 FFmpeg 主流程，不依赖仓库中的媒体样本。当前测试不评估 ASR 准确率或生产性能；模型精度、处理速度和内存占用取决于语言、视频质量、硬件和 Whisper 模型，不能用固定百分比承诺。

## Docker

```bash
docker compose up -d redis api worker
docker compose ps
docker compose logs -f api
```

生产部署仍需在反向代理层补齐 TLS、身份认证、访问日志脱敏、限流和产物生命周期清理。当前仓库适合本地使用和受控环境部署，不应在未配置认证的情况下直接暴露到公网。

## License

[MIT](LICENSE)
