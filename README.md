# AI Video Clipper

一个本地优先的智能视频切片与小红书内容生产工具。当前主链路可以完成视频上传/安全下载、ASR 字幕、语义/画面/音频联合评分、多片段选择、9:16 转码与合并；照片排序、拼图、封面和文案生成作为扩展能力提供。

## 当前能力边界

| 能力 | 状态 | 说明 |
|---|---|---|
| 视频上传、远程下载 | 可用 | 分块写入；远程地址限制为公网 HTTP/HTTPS，并限制下载大小 |
| 多片段智能剪辑 | 可用 | 使用真实视频时长、画面变化、运动、音频 RMS 和可选 ASR 文本评分 |
| 9:16 转码、字幕烧录 | 可用 | 依赖 FFmpeg |
| Faster-Whisper ASR | 可用 | 首次使用会下载所选模型；中文默认输出简体；可关闭内容分析以跳过 ASR |
| 语义评分 | 可用 | 优先调用 LLM 批量打分；未配置或调用失败时回退到 jieba 分词 + 词典规则 |
| 照片排序、拼图、封面 | 可用/可降级 | CLIP 等可选模型缺失时使用基础策略 |
| LLM 文案 | 可用/可降级 | 配置有效 API Key 时调用模型，否则明确返回本地模板 |
| 后台任务与轮询 | 可用 | `POST /jobs` 提交、`GET /jobs/{job_id}` 轮询；默认本机线程池，可切 Celery |
| 产物与任务清理 | 可用 | 按保留天数自动清理 `output_data`、下载缓存和任务记录 |
| `/segment`、`/captions` | 演示接口 | 响应带 `mode: simulation`；正式流程使用主剪辑和 LLM 接口 |
| 云存储 `/upload` | 模拟 | 不会把文件上传到外部存储 |
| 小红书 OAuth、直发、统计 | 模拟 | 不会授权、发布或生成真实笔记 ID，请下载后手动发布 |

## 快速启动

要求：Python 3.11–3.13、FFmpeg。只有使用 Celery 后台任务时才需要 Redis。
CI 会同时在 3.11 和 3.13 上跑测试。

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
3. `core/smart_clipping.py` 用一次 FFmpeg 解码同时采集场景变化、运动幅度（signalstats YDIF）
   和音频 RMS，结果通过 `metadata=print` 写入文件后解析。
4. `core/semantic_scoring.py` 把所有候选窗口**批量**送 LLM 打分（未配置时用词典规则）。
5. `core/video_workflow.py` 将语义、画面与音频分数归一化，按开场/高潮/结尾约束选取互不重叠的片段。
6. FFmpeg 把片段转成 1080×1920 H.264/AAC 文件并合并，结果通过 `/output/...` 返回。

### 语音识别

中文默认输出**简体**。Whisper 对普通话经常转写成繁体，而字幕会直接烧进视频，
`core/semantic_analysis.py` 里的情感和主题词典也全是简体，繁体输出会让匹配退化。
修法是在检测到中文时加一个极短的 `initial_prompt`。

这个提示必须短。它会通过 `condition_on_previous_text` 一直往后传，越长分段越粗
（34 秒素材、tiny 模型实测）：

| initial_prompt | 分段数 | 最长片段 | 繁体字符 |
|---|---|---|---|
| 无 | 12 | 4.7s | 24 |
| `简体`（当前默认） | 7 | 6.2s | 2 |
| `简体中文` | 4 | 10.8s | 2 |
| `以下是普通话的句子。` | 2 | 29.6s | 2 |

分段变粗不只是难看：每个候选窗口会拿到几乎一样的文本，语义评分就失去区分度，
烧录字幕也会变成几十秒一整块。`ASR_INITIAL_PROMPT` 可以覆盖，
`ASR_LANGUAGE_DETECTION=false` 可以关掉转录前的语言探测（约 0.15 秒）。

### 语义评分

判断"这段话值不值得剪"需要读懂意思，词典法做不到——它会把一句流畅、情绪饱满但没有信息量
的话排在有具体内容的话前面，这不是调权重能解决的。所以配置了 LLM API Key 时，候选窗口会
**批量**送模型打分并返回理由（写在 `selected_segments[].semantic_details` 里）。

回退是分层的，LLM 不会成为整条链路的单点故障：

- 未配置 Key、provider 报错、返回内容无法解析 → 整批回退到词典规则
- 模型只返回了部分片段的分数 → 缺的那几个单独回退，已返回的照常使用

`SEMANTIC_SCORING_MODE` 控制这个行为：`auto`（默认）、`llm`（强制，故障时报错）、
`rules`（完全不调用 LLM）。评分是高频调用，成本敏感的话建议给它配一个便宜的模型。

### 后台任务

剪辑要跑几分钟，因此内置 Web UI 走的是"提交任务 + 轮询"：连接断开或页面刷新都不会丢结果。
原有的同步接口保留不变，老客户端无需修改。

```bash
# 提交
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"multi_segment_clipping","params":{"video_path":"input_data/uploads/a.mp4","topic":"主题"}}'

# 轮询：done=true 后 result 里就是同步接口原来的返回体
curl http://127.0.0.1:8000/jobs/<job_id>

curl http://127.0.0.1:8000/jobs?limit=20     # 最近任务
curl http://127.0.0.1:8000/jobs/kinds        # 可用任务类型和当前后端
curl -X DELETE http://127.0.0.1:8000/jobs/<job_id>   # 尽力取消
```

`JOB_BACKEND=thread`（默认）用本机线程池，单机部署不需要 Redis；`JOB_BACKEND=celery`
把任务分发给 Celery worker，`docker compose` 里已经这样配置。两种后端写同一张 SQLite
任务表（`AIVIDEO_DB_PATH`），轮询接口不区分。取消是尽力而为：还没开始的任务不会开始，
已在运行的任务会被标记取消并丢弃结果，但不会杀掉正在跑的 FFmpeg。

### 产物清理

`output_data`、下载缓存和任务记录按保留天数自动清理，默认产物 7 天、上传 30 天、任务记录
30 天，每 6 小时扫一次。设成 `0` 关闭对应规则。也可以手动触发：

```bash
curl -X POST http://127.0.0.1:8000/admin/retention/sweep
```

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

并发方面，`MAX_CONCURRENT_MEDIA_JOBS` 限制同时运行的 FFmpeg 进程数（默认 CPU 核数的
一半）。**所有** FFmpeg 调用都经过这道闸门——主剪辑、黑屏/静音检测、封面抽帧、音频处理、
字幕提取和 ASR 音频转换；`tests/test_concurrency.py` 会静态检查有没有漏网的
`subprocess.run`。`ffprobe` 故意不限流：它只读几毫秒的元数据，排在长编码后面没有意义。
超出上限的请求会排队等待，而不是把机器上的 FFmpeg 进程数推到几十个。

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

## API 现状

`/docs` 里的接口按 `system / upload / video / asr / semantic / llm / pro / xiaohongshu /
media / jobs / tasks / admin` 分组。此前有 14 个路由在前端、测试和文档中都没有调用方，
按"**删掉假的和空的，保留并记录真的**"这条标准逐个过了一遍：

**已删除（4 个）** —— 它们要么返回编造的数据，要么什么都不做：

| 路由 | 原因 |
|---|---|
| `POST /xiaohongshu/authorize` | 伪造的 OAuth 会话 |
| `GET /xiaohongshu/profile` | 编造的账号资料和粉丝数 |
| `GET /xiaohongshu/note/{note_id}/stats` | 编造的笔记数据 |
| `POST /xiaohongshu/edit_text` | 空实现，直接返回"文案编辑成功" |

前三个依赖的是小红书并不开放的第三方接口，不存在变成真实功能的路径；保留它们只会让
调用方拿到看起来可信的假数字。`core/xiaohongshu_publisher.py` 里对应的三个方法一并删除。

**保留（10 个）** —— 有真实功能，只是仓库内暂时没人调用，已在此记录：

- `POST /auto_intro` —— **唯一**支持 yt-dlp 平台链接抓取的入口（其余路径只能直连下载公网文件）
- `POST /asr/transcribe`、`POST /asr/extract_audio`、`GET /asr/info` —— ASR 独立能力，
  异步版本在 `/tasks/asr/*`
- `POST /burnsub` —— 字幕烧录，与已测试的 `/cut916` 同类
- `POST /image/decorate`、`POST /image/decorations/smart` —— 图片滤镜/文字/贴纸
- `GET /collage/layouts`、`GET /cover/templates`、`GET /xiaohongshu/layouts` ——
  枚举对应 POST 接口的合法取值，客户端需要

顺带修了一个真实缺陷：`POST /image/decorations/smart` 原本把请求体当成 query 参数接收，
现在改成了正常的请求体模型。`tests/test_api_surface.py` 锁定了这套结论——被删的不会悄悄回来、
被留的不会悄悄消失、每个路由都有 tag、POST 不许用 query 传 body、路由总数只减不增。

### 图像处理依赖

不使用 OpenCV。它自带一套 FFmpeg 动态库，而 faster-whisper 依赖的 PyAV 自带另一个大版本，
同时加载会让一个进程里出现两份 `libavdevice`，macOS 明确警告重复的 Objective-C 类和
"mysterious crashes"。项目实际只用到 OpenCV 的 10 个调用，已全部改写到 `core/imaging.py`，
基于本来就有的 numpy / Pillow / SciPy。

和 OpenCV 的实测差异（FFmpeg 测试图案）：读图**完全一致**，灰度差 ≤1 级，Sobel 和拉普拉斯
方差在 5% 以内，边缘密度在 0.63–1.17 倍之间——调用方用它做照片之间的相对比较，不是绝对值。
`tests/test_imaging.py` 里有对照测试，装了 OpenCV 时会自动运行。

### 封面

封面配色可以设成 `theme="auto"`，从实际图片里聚类取色，而不是套用固定主题——此前不管照片
是什么内容，所有封面都是同一套粉色渐变。取色遵循内置主题的约定：`secondary` 是 `primary`
的同色系暗色（背景是 primary→secondary 渐变，用互补色会很突兀），互补色只用在小面积
点缀上。网格也会按图片数量自适应（4 张用 2×2，6 张用 2×3），此前固定 3×3，不足 9 张就会
在封面下方留一条空白带。

字体解析统一走 `core/fonts.py`，它会**实际渲染一个中文字符**来验证字体可用，而不是只检查
文件存在。原来的代码把 `/System/Library/Fonts/PingFang.ttc` 写死在第一位，而当前 macOS 上
这个路径并不存在，于是回退到没有中文字形的位图字体，标题渲染成几个看不清的像素；Linux 上
回退到 DejaVuSans，能打开但中文是方块。容器里没有中文字体时会打印安装提示
（`apt-get install -y fonts-noto-cjk`），也可以用 `AIVIDEO_CJK_FONT` 指定。

### 关于那两对"重复"模块

之前判断 `smart_cover_generator` / `smart_cover_design` 和两个 collage 生成器是重复实现，
**这个判断是错的**。实际情况：

- `smart_cover_design` 是**分析器**：从视频片段和照片里选最佳帧、分析画面特征、提取主色调、
  推荐设计方案
- `smart_cover_generator` 是**渲染器**：拿一组图片按布局和主题合成封面图

两者互补而非重复，真正的问题是它们没有串起来（分析器选的方案没有喂给渲染器）。这是个功能
缺口，不是清理项。两个 collage 生成器同理，且都被前端在用：`advanced_collage_generator`
是通用拼图（dynamic/grid/magazine/mosaic/creative），`xiaohongshu_collage_generator` 是
小红书专用的结构化版本。都不合并。

## 结构

```text
api/main.py                 FastAPI 路由与输入校验
core/config.py              唯一配置入口（环境变量 + .env）
core/runtime.py             运行时目录、路径边界、安全下载
core/video_workflow.py      主剪辑编排与 FFmpeg 合并
core/smart_clipping.py      真实画面/运动/音频测量
core/whisper_asr.py         Faster-Whisper 模型缓存与转录
core/text_tokenizer.py      中英文分词与词典匹配（jieba，缺失时降级）
core/semantic_scoring.py    候选片段语义评分（LLM 优先，词典规则兜底）
core/color_palette.py       从图片提取配色（封面渲染与设计共用）
core/fonts.py               字体解析（验证中文渲染能力，非仅检查文件存在）
requirements-gpu.txt        可选：torch / CLIP，主链路不需要
core/concurrency.py         FFmpeg 并发上限
core/jobs.py                后台任务提交（线程池 / Celery）
core/job_store.py           SQLite 任务记录
core/retention.py           产物与任务记录清理
routers/jobs.py             任务提交与轮询接口
core/semantic_analysis.py   文本语义评分
worker/tasks.py             Celery 后台任务
frontend/                   内置 Web UI
tests/                      API 表面、安全边界、中文评分、任务流和 FFmpeg 集成测试
```

## 测试与检查

```bash
pip install -r requirements-dev.txt   # 运行时依赖 + 测试依赖
pytest -q
python -m compileall -q api core routers worker tests
python -m pyflakes api core routers worker tests scripts *.py
node --check frontend/script.js
bash -n scripts/*.sh
```

Pyflakes 在 CI 里是**阻塞**检查（未定义名、未使用的导入和局部变量），目前是 0 条诊断。
之前那三个 `NameError`（`MAX_FILE_SIZE`、`ALLOWED_VIDEO_EXTENSIONS`、`datetime`）都属于
它能直接发现的类型。

测试套件会生成一段短视频并实际执行 FFmpeg 主流程，不依赖仓库中的媒体样本。
`tests/test_semantic_analysis.py` 锁定中文评分的基本不变量（有效内容必须高于口水话、
情感与主题词典必须能命中中文）；`tests/test_semantic_scoring.py` 用桩 provider 覆盖 LLM
评分和各级回退；`tests/test_event_loop_safety.py` 静态检查协程里不再直接调用 ASR /
FFmpeg 这类阻塞操作。当前测试不评估 ASR 准确率或生产性能；模型精度、
处理速度和内存占用取决于语言、视频质量、硬件和 Whisper 模型，不能用固定百分比承诺。

## Docker

```bash
docker compose up -d redis api worker
docker compose ps
docker compose logs -f api
```

生产部署仍需在反向代理层补齐 TLS、身份认证、访问日志脱敏和限流。当前仓库适合本地使用和受控环境部署，不应在未配置认证的情况下直接暴露到公网。

## License

[MIT](LICENSE)
