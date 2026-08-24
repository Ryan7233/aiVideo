# aiVideo

把一段长视频压成一份**剪辑清单**：哪几段值得剪、为什么、时间码是多少。

清单是主产物，9:16 成片是可选的。判断「哪十五秒值得剪」是专业剪辑软件替不了的部分；
把选出来的片段拼成视频，剪映做得比这里好——所以时间码和理由带走，执行交给你顺手的工具。

## 输出

一次分析产出三份东西：

| 格式 | 用途 |
|---|---|
| Markdown | 人读的清单：起止、时长、各维度分数、入选理由 |
| EDL (CMX3600) | 导入 Premiere / Resolve / Final Cut，片段直接落在时间线上 |
| SRT | 选中片段的字幕，按成片重新计时 |

勾选「同时渲染」才会额外产出 1080×1920 的 H.264 成片。清单几秒出，成片要等转码。

## 怎么选段

1. `POST /upload/video` 上传，或直接给平台链接（Bilibili / YouTube 等经 yt-dlp 解析；
   直链 `.mp4` 走带大小上限的下载器）。
2. Faster-Whisper 转写。中文默认输出简体，并按**词级时间戳重新分句**——Whisper 自己的
   分段跟着解码窗口走，连续口播经常整段几十秒返回，那样每个候选窗口拿到的文本都一样，
   语义评分就失去区分度。
3. 一次 FFmpeg 解码同时采集场景变化、运动幅度（signalstats YDIF）和音频 RMS。
4. 候选窗口批量送 LLM 打分并给出理由；未配置 API Key 时回退到 jieba 分词 + 词典规则。
5. 按开场 / 高潮 / 结尾约束选出互不重叠的片段，写出清单（可选渲染成片）。

## 快速启动

要求：Python 3.11–3.13、FFmpeg。只有 `JOB_BACKEND=celery` 时才需要 Redis。

```bash
git clone https://github.com/Ryan7233/aiVideo.git
cd aiVideo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env
python start_server.py
```

- Web UI：<http://127.0.0.1:8000/>
- OpenAPI：<http://127.0.0.1:8000/docs>

默认只监听回环地址。要对外提供服务，先设置 `AIVIDEO_API_KEY`，再用 compose 的
`production` profile 把 nginx 放在前面。

## 接口

```bash
# 提交（清单模式，不渲染）
curl -X POST http://127.0.0.1:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"kind":"multi_segment_clipping","params":{
        "video_path":"input_data/uploads/a.mp4",
        "topic":"主题","target_segments":3,"total_duration":60,"render":false}}'

# 轮询
curl http://127.0.0.1:8000/jobs/<job_id>
```

同步接口 `POST /video/multi_segment_clipping` 也在，参数相同。其余接口：`/analyze_video`、
`/asr/transcribe`、`/asr/extract_audio`、`/cut916`、`/burnsub`、`/semantic/analyze`。

## 配置

| 变量 | 说明 |
|---|---|
| `AIVIDEO_API_KEY` | 设置后除 `/ /health /info` 外都需要 `X-API-Key` |
| `API_HOST` | 默认 `127.0.0.1`；容器镜像里显式设为 `0.0.0.0` |
| `JOB_BACKEND` | `thread`（默认，无需 Redis）或 `celery` |
| `SEMANTIC_SCORING_MODE` | `auto` / `llm`（强制）/ `rules`（不调用 LLM） |
| `MAX_CONCURRENT_MEDIA_JOBS` | 同时运行的 FFmpeg 进程数，默认 CPU 核数一半 |
| `ASR_INITIAL_PROMPT` | 中文转写引导词，默认 `简体`（必须短，长提示会让分段变粗） |
| `OUTPUT_RETENTION_DAYS` 等 | 产物与任务记录保留天数，`0` 关闭该项 |

完整列表见 `env.example`。空值等价于未配置。

## 结构

```text
api/main.py                 FastAPI 路由与输入校验
core/video_workflow.py      主流程：选段编排与可选渲染
core/edl.py                 剪辑清单（Markdown / EDL / SRT）
core/whisper_asr.py         Faster-Whisper 转写
core/utterances.py          按词级时间戳重新分句
core/smart_clipping.py      一次解码采集画面 / 运动 / 音频
core/semantic_scoring.py    候选片段打分（LLM 优先，词典规则兜底）
core/semantic_analysis.py   中文语义评分
core/text_tokenizer.py      中英文分词与词典匹配
core/jobs.py                后台任务（线程池 / Celery）
core/job_store.py           SQLite 任务记录
core/runtime.py             路径边界、SSRF 校验、平台链接解析
core/concurrency.py         FFmpeg 并发上限
core/degradation.py         降级标记
frontend/                   单页 Web UI
tests/                      端到端与回归测试
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
python -m pyflakes api core worker tests scripts *.py
node --check frontend/app.js
```

弃用警告默认视为错误（见 `pytest.ini`）。Pyflakes 是 CI 的阻塞检查。

## 已知边界

- 没有发布能力。小红书没有开放的第三方发布接口，所以流程止于「导出，手动发」。
- 语义评分未配置 LLM 时是词典法。它能区分口水话和有内容的话，但判断不了
  「流畅但没信息量」——那需要真的读懂，配一个 API Key 才有。
- 并发闸门是进程内信号量。多个 Celery worker 时实际上限是 worker 数 × 该值。

## License

[MIT](LICENSE)
