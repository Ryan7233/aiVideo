/* aiVideo — submit a long video on the left, review its cut list on the right. */

const API = window.location.origin;
const KIND = 'multi_segment_clipping';
const POLL_MS = 1500;
const HISTORY_MS = 4000;
const POLL_TIMEOUT_MS = 60 * 60 * 1000;

const $ = (id) => document.getElementById(id);

const state = {
  source: 'file',
  file: null,          // the File the user picked, played back locally
  fileUrl: null,       // object URL for that File
  job: null,           // { id, params, result } currently on screen
  running: null,       // job id being polled
  accepted: new Set(), // indices of suggested segments the reviewer keeps
  manual: [],          // reviewer-added spans [{start_time, end_time}]
  inPoint: null,
  outPoint: null,
  stopAt: null,        // pause the player when it reaches this time
  activeKey: null,     // 's3' / 'm1' — which block/card is highlighted
  mediaUrl: null,      // object URL when the source had to be fetched with a key
};

const STEP_LABELS = {
  queued: '排队中',
  materializing: '准备素材',
  transcribing: '语音转写',
  measuring: '画面与音频测量',
  scoring: '语义评分',
  rendering: '渲染成片',
  exporting: '导出清单',
  clipping: '处理中',
  pending: '排队中',
  running: '处理中',
};

const STEP_NOTES = {
  transcribing: '语音转写是最久的一步；几分钟的视频通常 10–60 秒，首次运行还要下载模型。',
  measuring: '每 300 秒一块解码全片，采集场景变化、运动幅度和音频能量。',
  scoring: '候选窗口批量评分；配置了 LLM 时走模型，否则走词典规则。',
  rendering: '正在转码 9:16 成片，时长取决于片段总长。',
};

/* ── auth ────────────────────────────────────────────────── */
/* The server is unauthenticated until AIVIDEO_API_KEY is set, at which point
   every call but the page itself needs X-API-Key. Without this the UI loads
   and then 401s on upload, submit and poll alike. */
const KEY_STORAGE = 'aivideo.apiKey';

function apiKey() {
  try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; }
}

function setApiKey(value) {
  try { value ? localStorage.setItem(KEY_STORAGE, value) : localStorage.removeItem(KEY_STORAGE); }
  catch { /* private mode: the key just will not persist */ }
}

/** fetch with the key attached, and a clear signal when it is rejected. */
async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = apiKey();
  if (key) headers.set('X-API-Key', key);

  const response = await fetch(`${API}${path}`, { ...options, headers });
  if (response.status === 401) {
    revealAuth(key ? 'API Key 无效，请重新填写' : '');
    throw new Error('需要有效的 API Key');
  }
  return response;
}

function revealAuth(message) {
  $('auth').classList.remove('hidden');
  const note = $('auth-note');
  note.textContent = message || '存在本浏览器，随请求发送';
  note.className = `auth-note ${message ? 'bad' : ''}`;
  if (message) $('api-key').focus();
}

$('api-key-save').addEventListener('click', async () => {
  setApiKey($('api-key').value.trim());
  const note = $('auth-note');
  if (await serverAccepts()) {
    note.textContent = '已保存';
    note.className = 'auth-note good';
    loadHistory();
  } else {
    note.textContent = 'API Key 无效';
    note.className = 'auth-note bad';
  }
});

/** True when the current key (or no key) gets past the middleware. */
async function serverAccepts() {
  const headers = new Headers();
  const key = apiKey();
  if (key) headers.set('X-API-Key', key);
  try {
    const response = await fetch(`${API}/jobs/kinds`, { headers });
    return response.status !== 401;
  } catch {
    return true;   // network trouble is not an auth problem
  }
}

/* ── form: source tabs, file drop, sliders, persistence ──── */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => {
      const on = t === tab;
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', String(on));
    });
    state.source = tab.dataset.source;
    $('source-file').classList.toggle('hidden', state.source !== 'file');
    $('source-url').classList.toggle('hidden', state.source !== 'url');
  });
});

const drop = $('drop');
$('video-input').addEventListener('change', (e) => acceptFile(e.target.files[0]));
['dragenter', 'dragover'].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((type) =>
  drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => acceptFile(e.dataTransfer.files[0]));

function acceptFile(file) {
  if (!file) return;
  state.file = file;
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.fileUrl = URL.createObjectURL(file);
  $('drop-title').textContent = file.name;
  $('drop-hint').textContent = `${(file.size / 1048576).toFixed(1)} MB`;
}

[['w-sem', 'w-sem-v'], ['w-vis', 'w-vis-v'], ['w-aud', 'w-aud-v']].forEach(([input, out]) => {
  $(input).addEventListener('input', () => { $(out).textContent = $(input).value; });
});

const FORM_STORAGE = 'aivideo.form';
const FORM_FIELDS = ['topic', 'segments', 'duration', 'selection-mode', 'w-sem', 'w-vis', 'w-aud', 'asr-model', 'use-asr', 'render'];

function readForm() {
  const values = {};
  FORM_FIELDS.forEach((id) => {
    const el = $(id);
    values[id] = el.type === 'checkbox' ? el.checked : el.value;
  });
  return values;
}

function writeForm(values) {
  FORM_FIELDS.forEach((id) => {
    if (!(id in values)) return;
    const el = $(id);
    if (el.type === 'checkbox') el.checked = Boolean(values[id]);
    else el.value = values[id];
  });
  ['w-sem', 'w-vis', 'w-aud'].forEach((id) => { $(`${id}-v`).textContent = $(id).value; });
}

function rememberForm() {
  try { localStorage.setItem(FORM_STORAGE, JSON.stringify(readForm())); } catch { /* fine */ }
}

(function restoreForm() {
  try {
    const saved = JSON.parse(localStorage.getItem(FORM_STORAGE) || 'null');
    if (saved) writeForm(saved);
  } catch { /* fine */ }
})();

/** Fill the form from a previous job's params — "run this one again". */
function formFromParams(params) {
  writeForm({
    topic: params.topic || '',
    segments: params.target_segments ?? 3,
    duration: params.total_duration ?? 60,
    'selection-mode': params.selection_mode || 'highlights',
    'w-sem': params.semantic_weight ?? 0.5,
    'w-vis': params.visual_weight ?? 0.2,
    'w-aud': params.audio_weight ?? 0.3,
    'asr-model': params.asr_model_size || 'base',
    'use-asr': params.enable_content_analysis !== false,
    render: Boolean(params.render),
  });
}

/* ── stages ──────────────────────────────────────────────── */
function showStage(name) {
  ['empty', 'running', 'error', 'result'].forEach((stage) => {
    $(`stage-${stage}`).classList.toggle('hidden', stage !== name);
  });
}

function showError(message) {
  // A failed job's error carries the Python traceback after the first line;
  // the person needs the line, the traceback goes behind a fold.
  const [head, ...rest] = String(message || '失败').split('\n');
  const box = $('error-text');
  box.textContent = head;
  if (rest.length) {
    const details = document.createElement('details');
    details.innerHTML = '<summary>详细信息</summary>';
    const pre = document.createElement('pre');
    pre.textContent = rest.join('\n').trim();
    details.appendChild(pre);
    box.appendChild(details);
  }
  showStage('error');
}

/* ── run ─────────────────────────────────────────────────── */
$('run').addEventListener('click', run);

async function run() {
  const button = $('run');
  if (button.disabled) return;
  button.disabled = true;
  rememberForm();

  try {
    const videoPath = await resolveSource();
    const params = {
      video_path: videoPath,
      topic: $('topic').value.trim(),
      target_segments: Number($('segments').value),
      selection_mode: $('selection-mode').value,
      total_duration: Number($('duration').value),
      semantic_weight: Number($('w-sem').value),
      visual_weight: Number($('w-vis').value),
      audio_weight: Number($('w-aud').value),
      enable_content_analysis: $('use-asr').checked,
      asr_model_size: $('asr-model').value,
      render: $('render').checked,
    };

    setRunning('queued', 0);
    const jobId = await submitJob(params);
    state.running = jobId;
    loadHistory();
    const job = await pollJob(jobId);
    // The File the user just picked plays back instantly; keep it with the job.
    openJob(job, state.source === 'file' ? state.fileUrl : null);
  } catch (error) {
    console.error(error);
    showError(error.message || '失败');
  } finally {
    state.running = null;
    button.disabled = false;
    loadHistory();
  }
}

/** Upload the chosen file, or hand the URL straight to the backend. */
async function resolveSource() {
  if (state.source === 'url') {
    const url = $('video-url').value.trim();
    if (!url) throw new Error('请填写视频链接');
    return url;
  }
  if (!state.file) throw new Error('请先选择视频文件');

  setRunning('uploading', 0);
  $('running-step').textContent = '正在上传';
  const form = new FormData();
  form.append('file', state.file);
  const response = await request('/upload/video', { method: 'POST', body: form });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.status !== 'success') {
    throw new Error(body.detail || body.message || `上传失败 (${response.status})`);
  }
  return body.file.saved_path;
}

async function submitJob(params) {
  const submission = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: KIND, params }),
  });
  const submitted = await submission.json().catch(() => ({}));
  if (!submission.ok) throw new Error(detailOf(submitted) || `提交失败 (${submission.status})`);
  return submitted.job_id;
}

async function pollJob(jobId) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));

    const response = await request(`/jobs/${jobId}`);
    if (!response.ok) throw new Error(`无法查询任务状态 (${response.status})`);
    const job = await response.json();

    if (!job.done) {
      const step = (job.progress && job.progress.step) || job.status;
      setRunning(step, Math.round((Date.now() - startedAt) / 1000));
      continue;
    }
    if (job.status === 'succeeded') return job;
    if (job.status === 'cancelled') throw new Error('任务已取消');
    throw new Error(job.error || '任务失败');
  }
  throw new Error('任务超时');
}

function setRunning(step, elapsed) {
  showStage('running');
  $('running-step').textContent = STEP_LABELS[step] || step;
  $('running-elapsed').textContent = `${elapsed}s`;
  $('running-note').textContent = STEP_NOTES[step] || '';
}

$('cancel').addEventListener('click', async () => {
  if (!state.running) return;
  try { await request(`/jobs/${state.running}`, { method: 'DELETE' }); }
  catch (error) { console.error(error); }
});

function detailOf(body) {
  const detail = body && body.detail;
  if (!detail) return body && body.message;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || '').filter(Boolean).join('；');
  return detail.message || JSON.stringify(detail);
}

/* ── history ─────────────────────────────────────────────── */
let historyTimer = null;

async function loadHistory() {
  let jobs = [];
  try {
    const response = await request(`/jobs?kind=${KIND}&limit=40`);
    if (response.ok) jobs = (await response.json()).jobs || [];
  } catch { /* the list is a convenience; the page still works without it */ }

  const list = $('jobs');
  list.innerHTML = '';
  if (!jobs.length) list.innerHTML = '<li class="empty">还没有任务</li>';

  jobs.forEach((job) => {
    const li = document.createElement('li');
    li.className = `job ${job.status}${state.job && state.job.id === job.id ? ' current' : ''}`;
    const live = job.status === 'running' || job.status === 'pending';
    const step = live ? (STEP_LABELS[(job.progress && job.progress.step) || job.status] || job.status) : '';
    li.innerHTML = `
      <span class="dot"></span>
      <span>
        <div class="name">${escapeHtml(job.label || '（无主题）')}</div>
        <div class="when">${when(job.created_at)}${step ? ` · ${escapeHtml(step)}` : ''}${job.status === 'failed' ? ' · 失败' : ''}</div>
      </span>
      ${live ? '<button class="x" title="取消">✕</button>' : ''}`;
    li.addEventListener('click', () => openJobById(job.id));
    const cancel = li.querySelector('.x');
    if (cancel) cancel.addEventListener('click', async (event) => {
      event.stopPropagation();
      try { await request(`/jobs/${job.id}`, { method: 'DELETE' }); } catch { /* shown on refresh */ }
      loadHistory();
    });
    list.appendChild(li);
  });

  clearTimeout(historyTimer);
  if (jobs.some((job) => job.status === 'running' || job.status === 'pending')) {
    historyTimer = setTimeout(loadHistory, HISTORY_MS);
  }
}

$('history-refresh').addEventListener('click', loadHistory);

async function openJobById(jobId) {
  try {
    const response = await request(`/jobs/${jobId}`);
    if (!response.ok) throw new Error(`无法读取任务 (${response.status})`);
    const job = await response.json();
    if (job.status === 'succeeded') {
      openJob(job, null);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      state.job = { id: job.id, params: job.params, result: null };
      showError(job.error || (job.status === 'cancelled' ? '任务已取消' : '任务失败'));
      loadHistory();
    } else {
      state.running = job.id;
      setRunning((job.progress && job.progress.step) || job.status, 0);
      const done = await pollJob(job.id);
      openJob(done, null);
      state.running = null;
    }
  } catch (error) {
    showError(error.message || '打开失败');
  }
}

/* ── opening a finished job ──────────────────────────────── */
function openJob(job, localUrl) {
  state.job = { id: job.id, params: job.params || {}, result: job.result || {} };
  const segments = state.job.result.selected_segments || [];
  state.accepted = new Set(segments.map((_, index) => index));
  state.manual = [];
  state.inPoint = state.outPoint = null;
  state.stopAt = null;
  state.activeKey = null;

  formFromParams(state.job.params);
  renderResult(localUrl);
  showStage('result');
  loadHistory();
}

function renderResult(localUrl) {
  const { result, params } = state.job;
  const analysis = result.analysis || {};
  const transcription = analysis.transcription || {};

  $('result-title').textContent = params.topic ? `剪辑清单 · ${params.topic}` : '剪辑清单';
  $('result-meta').textContent = `${state.job.id.slice(0, 8)} · ${params.selection_mode === 'summary' ? '结构化摘要' : '精彩片段'}`;

  // Source playback: the File just picked (instant), else the server copy.
  const player = $('player');
  const note = $('player-note');
  player.pause();
  if (state.mediaUrl) { URL.revokeObjectURL(state.mediaUrl); state.mediaUrl = null; }
  note.classList.add('hidden');
  if (localUrl) {
    player.src = localUrl;
  } else if (result.source_video) {
    setPlayerSource(result.source_video);
  } else {
    player.removeAttribute('src');
    note.textContent = '这个任务的源视频不在服务器上，无法回放；片段时间码仍可导出。';
    note.classList.remove('hidden');
  }

  const source = {
    whisper: `语音识别 · ${transcription.segment_count || 0} 句`,
    provided_subtitle: '使用了提供的字幕',
    disabled: '未做语音识别',
    unavailable: '语音识别不可用，已按画面和音频选段',
  }[transcription.source] || transcription.source || '';

  const measurement = analysis.measurement || {};
  const duration = analysis.total_video_duration || 0;
  $('summary').innerHTML = [
    tile('素材时长', hms(duration)),
    tile('选出', `${(result.selected_segments || []).length} 段 / ${(analysis.total_output_duration || 0).toFixed(1)}s`),
    tile('候选窗口', analysis.analyzed_segments || 0),
    tile('测量覆盖', duration ? `${Math.min(100, Math.round(100 * (measurement.measured_seconds || 0) / duration))}%` : '—'),
    tile('转写', source || '—'),
  ].join('');

  $('notes').innerHTML = [
    ...(result.warnings || []).map((message) => `<p class="note warn">${escapeHtml(message)}</p>`),
    ...(result.decision_list_errors || []).map((message) => `<p class="note bad">导出失败：${escapeHtml(message)}</p>`),
    ...((measurement.errors || []).length ? [`<p class="note bad">${measurement.errors.length} 个区间测量失败，这些区间只按语义评分</p>`] : []),
  ].join('');

  renderExports();
  renderTimeline();
  renderSegments();
  renderReviewCount();
}

/** Give the player the server copy: a plain URL streams with range requests,
    but <video> cannot send X-API-Key, so behind a key the file is fetched
    whole and handed over as a blob. */
async function setPlayerSource(path) {
  const player = $('player');
  const note = $('player-note');
  if (!apiKey()) {
    player.src = `${API}/${path}`;
    player.addEventListener('error', () => {
      note.textContent = '源视频已被清理（超过保留天数），无法回放。';
      note.classList.remove('hidden');
    }, { once: true });
    return;
  }
  try {
    const response = await request(`/${path}`);
    if (!response.ok) throw new Error(String(response.status));
    state.mediaUrl = URL.createObjectURL(await response.blob());
    player.src = state.mediaUrl;
  } catch {
    note.textContent = '源视频已被清理（超过保留天数），无法回放。';
    note.classList.remove('hidden');
  }
}

/* ── exports ─────────────────────────────────────────────── */
function renderExports() {
  const { result } = state.job;
  const files = result.decision_list || {};
  const labels = { md: 'Markdown', edl: 'EDL', srt: 'SRT' };
  const targets = Object.entries(files).map(([kind, path]) => ({ path, label: labels[kind] || kind }));
  if (result.output_video) targets.unshift({ path: result.output_video, label: '成片 MP4', strong: true });

  // /output is behind the API key too, and <a download> cannot send a header,
  // so each file is fetched with it and handed over as a blob.
  const box = $('downloads');
  box.innerHTML = '';
  targets.forEach(({ path, label, strong }) => {
    const link = document.createElement('a');
    link.textContent = label;
    link.href = `${API}/${path}`;
    link.download = path.split('/').pop();
    if (strong) link.className = 'strong';
    link.addEventListener('click', async (event) => {
      event.preventDefault();
      try {
        const response = await request(`/${path}`);
        if (!response.ok) throw new Error(`下载失败 (${response.status})`);
        saveBlob(await response.blob(), link.download);
      } catch (error) {
        showToast(error.message || '下载失败');
      }
    });
    box.appendChild(link);
  });
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const temp = document.createElement('a');
  temp.href = url;
  temp.download = filename;
  temp.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('copy-md').addEventListener('click', async () => {
  const path = state.job && state.job.result.decision_list && state.job.result.decision_list.md;
  if (!path) return showToast('没有 Markdown 清单');
  try {
    const response = await request(`/${path}`);
    if (!response.ok) throw new Error(`读取失败 (${response.status})`);
    await navigator.clipboard.writeText(await response.text());
    showToast('已复制 Markdown');
  } catch (error) {
    showToast(error.message || '复制失败');
  }
});

$('export-labels').addEventListener('click', () => {
  if (!state.job) return;
  const segments = state.job.result.selected_segments || [];
  const accepted = [
    ...segments.filter((_, index) => state.accepted.has(index)),
    ...state.manual,
  ].map((span) => ({ start_time: round3(span.start_time), end_time: round3(span.end_time) }))
    .sort((a, b) => a.start_time - b.start_time);
  const rejected = segments.filter((_, index) => !state.accepted.has(index))
    .map((span) => ({ start_time: round3(span.start_time), end_time: round3(span.end_time) }));
  const labels = {
    accepted,
    rejected,
    source: state.job.result.source_video || state.job.params.video_path || null,
    job_id: state.job.id,
    topic: state.job.params.topic || '',
    exported_at: new Date().toISOString(),
  };
  const blob = new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' });
  saveBlob(blob, `labels_${state.job.id.slice(0, 8)}.json`);
});

/* ── timeline ────────────────────────────────────────────── */
function videoDuration() {
  const player = $('player');
  const measured = (state.job.result.analysis || {}).total_video_duration || 0;
  return (isFinite(player.duration) && player.duration > 0) ? player.duration : measured;
}

function renderTimeline() {
  const total = videoDuration();
  const analysis = state.job.result.analysis || {};
  const measurement = analysis.measurement || {};
  const segments = state.job.result.selected_segments || [];
  const pct = (seconds) => `${total ? Math.max(0, Math.min(100, 100 * seconds / total)) : 0}%`;

  const measured = $('tl-measured');
  const intervals = measurement.intervals || [];
  measured.innerHTML = intervals.map((interval) =>
    `<i style="left:${pct(interval.start)};width:${pct(interval.end - interval.start)}"></i>`).join('');
  $('timeline').classList.toggle('unmeasured-all', !intervals.length);

  const blocks = $('tl-blocks');
  blocks.innerHTML = '';
  const add = (key, span, cls, title) => {
    const block = document.createElement('i');
    block.className = `${cls}${state.activeKey === key ? ' active' : ''}`;
    block.style.left = pct(span.start_time);
    block.style.width = pct(span.end_time - span.start_time);
    block.title = title;
    block.dataset.key = key;
    block.addEventListener('click', (event) => {
      event.stopPropagation();
      playSpan(key, span.start_time, span.end_time);
    });
    blocks.appendChild(block);
  };
  segments.forEach((segment, index) => {
    const cls = `${typeClass(segment.type)}${state.accepted.has(index) ? '' : ' rejected'}`;
    add(`s${index}`, segment, cls, `#${index + 1} ${hms(segment.start_time)}–${hms(segment.end_time)}`);
  });
  state.manual.forEach((span, index) => {
    add(`m${index}`, span, 'manual', `手动 ${hms(span.start_time)}–${hms(span.end_time)}`);
  });
  if (state.inPoint !== null && state.outPoint !== null && state.outPoint > state.inPoint) {
    const pending = document.createElement('i');
    pending.className = 'pending';
    pending.style.left = pct(state.inPoint);
    pending.style.width = pct(state.outPoint - state.inPoint);
    blocks.appendChild(pending);
  }
}

$('timeline').addEventListener('click', (event) => {
  const total = videoDuration();
  if (!total) return;
  const rect = $('timeline').getBoundingClientRect();
  const ratio = (event.clientX - rect.left) / rect.width;
  const player = $('player');
  state.stopAt = null;
  setActive(null);
  player.currentTime = Math.max(0, Math.min(total, ratio * total));
});

function typeClass(type) {
  return type === 'intro' || type === 'conclusion' ? type : 'highlight';
}

/* ── player ──────────────────────────────────────────────── */
const player = $('player');

player.addEventListener('timeupdate', () => {
  const total = videoDuration();
  $('tl-playhead').style.left = total ? `${100 * player.currentTime / total}%` : '0';
});

// timeupdate fires only every ~250 ms, which overshoots a segment's end by
// more than a reviewer checking a boundary will tolerate; poll per frame.
function watchStop() {
  if (state.stopAt === null) return;
  if (player.currentTime >= state.stopAt - 0.02) {
    player.pause();
    player.currentTime = state.stopAt;
    state.stopAt = null;
    return;
  }
  requestAnimationFrame(watchStop);
}

player.addEventListener('loadedmetadata', () => { if (state.job) renderTimeline(); });

function playSpan(key, start, end) {
  if (!player.getAttribute('src')) return showToast('这个任务没有可回放的源视频');
  state.stopAt = end;
  setActive(key);
  player.currentTime = start;
  player.play().then(watchStop).catch(() => { /* autoplay policy: the user can press play */ });
  const card = document.querySelector(`.seg[data-key="${key}"]`);
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function setActive(key) {
  state.activeKey = key;
  document.querySelectorAll('.tl-blocks i[data-key], .seg[data-key]').forEach((el) => {
    el.classList.toggle('active', el.dataset.key === key);
  });
}

/* ── segments ────────────────────────────────────────────── */
function renderSegments() {
  const segments = state.job.result.selected_segments || [];
  const list = $('segment-list');
  list.innerHTML = '';

  segments.forEach((segment, index) => {
    const details = segment.semantic_details || {};
    const key = `s${index}`;
    const kept = state.accepted.has(index);
    const metric = (name) => segment.available_dimensions && segment.available_dimensions[name] === false
      ? '—' : Number(segment[`${name}_score`] || 0).toFixed(2);
    const why = details.reason || '';
    const text = segment.text || segment.preview_text || '';
    const long = text.length > 90;

    const li = document.createElement('li');
    li.className = `seg${kept ? '' : ' rejected'}${state.activeKey === key ? ' active' : ''}`;
    li.dataset.key = key;
    li.innerHTML = `
      <button class="play" title="播放这一段">▶</button>
      <div>
        <div class="head">
          <span class="range">${hms(segment.start_time)} → ${hms(segment.end_time)}</span>
          <span class="dur">${Number(segment.duration).toFixed(1)}s</span>
          <span class="badge ${typeClass(segment.type)}">${typeLabel(segment.type)}</span>
        </div>
        <div class="scores">
          <span>总分 <b>${Number(segment.score).toFixed(2)}</b></span>
          <span>语义 ${metric('semantic')}</span>
          <span>画面 ${metric('visual')}</span>
          <span>音频 ${metric('audio')}</span>
          ${segment.boundary_source === 'utterance' ? '<span>· 语句边界</span>' : ''}
        </div>
        <div class="bar"><i style="width:${Math.round(100 * Math.max(0, Math.min(1, segment.score)))}%"></i></div>
        ${why ? `<p class="why">${escapeHtml(why)}</p>` : ''}
        ${text ? `<p class="text${long ? ' clamp' : ''}">${escapeHtml(text)}</p>` : ''}
        ${long ? '<button class="more">展开全文</button>' : ''}
      </div>
      <div class="side">
        <label class="keep"><input type="checkbox" ${kept ? 'checked' : ''}> 采用</label>
        <div class="tools"><button class="ghost copy" title="复制时间码">复制</button></div>
      </div>`;

    li.querySelector('.play').addEventListener('click', () => playSpan(key, segment.start_time, segment.end_time));
    li.querySelector('.keep input').addEventListener('change', (event) => {
      event.target.checked ? state.accepted.add(index) : state.accepted.delete(index);
      li.classList.toggle('rejected', !event.target.checked);
      renderTimeline();
      renderReviewCount();
    });
    li.querySelector('.copy').addEventListener('click', () => {
      const line = `${hms(segment.start_time, true)} --> ${hms(segment.end_time, true)}`;
      navigator.clipboard.writeText(line).then(() => showToast(`已复制 ${line}`), () => showToast('复制失败'));
    });
    const more = li.querySelector('.more');
    if (more) more.addEventListener('click', () => {
      const p = li.querySelector('.text');
      const open = p.classList.toggle('clamp');
      more.textContent = open ? '展开全文' : '收起';
    });
    list.appendChild(li);
  });

  state.manual.forEach((span, index) => {
    const key = `m${index}`;
    const li = document.createElement('li');
    li.className = `seg manual${state.activeKey === key ? ' active' : ''}`;
    li.dataset.key = key;
    li.innerHTML = `
      <button class="play" title="播放这一段">▶</button>
      <div>
        <div class="head">
          <span class="range">${hms(span.start_time)} → ${hms(span.end_time)}</span>
          <span class="dur">${(span.end_time - span.start_time).toFixed(1)}s</span>
          <span class="badge manual">手动</span>
        </div>
        <p class="text">你标记的区间，会作为「认可片段」进入标注导出。</p>
      </div>
      <div class="side"><div class="tools"><button class="ghost remove">删除</button></div></div>`;
    li.querySelector('.play').addEventListener('click', () => playSpan(key, span.start_time, span.end_time));
    li.querySelector('.remove').addEventListener('click', () => {
      state.manual.splice(index, 1);
      if (state.activeKey === key) state.activeKey = null;
      renderTimeline();
      renderSegments();
      renderReviewCount();
    });
    list.appendChild(li);
  });
}

function typeLabel(type) {
  return { intro: '开场', conclusion: '结尾', highlight: '精彩', best_available: '补位' }[type] || type || '';
}

function renderReviewCount() {
  const total = (state.job.result.selected_segments || []).length;
  $('review-count').textContent = `采用 ${state.accepted.size}/${total} 段建议 + ${state.manual.length} 段手动`;
}

/* ── manual in/out marking ───────────────────────────────── */
function markIn() {
  if (!player.getAttribute('src')) return;
  state.inPoint = player.currentTime;
  if (state.outPoint !== null && state.outPoint <= state.inPoint) state.outPoint = null;
  renderMark();
}

function markOut() {
  if (!player.getAttribute('src')) return;
  state.outPoint = player.currentTime;
  if (state.inPoint !== null && state.inPoint >= state.outPoint) state.inPoint = null;
  renderMark();
}

function renderMark() {
  $('mark-in-v').textContent = state.inPoint === null ? '—' : hms(state.inPoint, true);
  $('mark-out-v').textContent = state.outPoint === null ? '—' : hms(state.outPoint, true);
  const ready = state.inPoint !== null && state.outPoint !== null && state.outPoint - state.inPoint >= 1;
  $('mark-add').disabled = !ready;
  renderTimeline();
}

$('mark-in').addEventListener('click', markIn);
$('mark-out').addEventListener('click', markOut);
$('mark-add').addEventListener('click', () => {
  if ($('mark-add').disabled) return;
  state.manual.push({ start_time: state.inPoint, end_time: state.outPoint });
  state.manual.sort((a, b) => a.start_time - b.start_time);
  state.inPoint = state.outPoint = null;
  renderMark();
  renderSegments();
  renderReviewCount();
  showToast('已添加手动区间');
});

document.addEventListener('keydown', (event) => {
  const tag = (event.target.tagName || '').toLowerCase();
  // Typing fields keep their keys; a focused checkbox (just ticked) does not.
  const typing = tag === 'select' || tag === 'textarea'
    || (tag === 'input' && event.target.type !== 'checkbox');
  if (typing || !state.job) return;
  if ($('stage-result').classList.contains('hidden')) return;
  if (event.key === 'i' || event.key === 'I') { markIn(); event.preventDefault(); }
  else if (event.key === 'o' || event.key === 'O') { markOut(); event.preventDefault(); }
  else if (event.key === ' ' && tag !== 'button' && tag !== 'video') {
    player.paused ? player.play().catch(() => {}) : player.pause();
    event.preventDefault();
  }
});

/* ── helpers ─────────────────────────────────────────────── */
const tile = (label, value) =>
  `<div class="tile"><span>${label}</span><b>${escapeHtml(String(value))}</b></div>`;

function hms(seconds, precise = false) {
  const total = Math.max(0, Number(seconds) || 0);
  const h = Math.floor(total / 3600);
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const s = Math.floor(total % 60);
  const base = `${h ? `${h}:` : ''}${m}:${String(s).padStart(2, '0')}`;
  if (!precise) return base;
  const ms = String(Math.round((total - Math.floor(total)) * 1000)).padStart(3, '0');
  return `${base}.${ms}`;
}

function round3(value) { return Math.round(Number(value) * 1000) / 1000; }

function when(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  if (isNaN(date)) return String(iso).slice(0, 16);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  const time = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  return sameDay ? time : `${date.getMonth() + 1}/${date.getDate()} ${time}`;
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

let toastTimer = null;
function showToast(message) {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('on'), 2200);
}

/* ── boot ────────────────────────────────────────────────── */
(async () => {
  if (!(await serverAccepts())) revealAuth('');
  loadHistory();
})();
