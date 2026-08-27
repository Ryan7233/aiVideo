/* aiVideo — submit a long video, get back a cut list. */

const API = window.location.origin;
const POLL_MS = 1500;
const POLL_TIMEOUT_MS = 60 * 60 * 1000;

const $ = (id) => document.getElementById(id);
const state = { source: 'file', file: null };

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

// Ask for a key only if this server actually wants one.
(async () => { if (!(await serverAccepts())) revealAuth(''); })();

/* ── source tabs ─────────────────────────────────────────── */
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

/* ── file picking, including drag and drop ───────────────── */
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
  $('drop-title').textContent = file.name;
  $('drop-hint').textContent = `${(file.size / 1048576).toFixed(1)} MB`;
}

/* ── weight sliders ──────────────────────────────────────── */
[['w-sem', 'w-sem-v'], ['w-vis', 'w-vis-v'], ['w-aud', 'w-aud-v']].forEach(([input, out]) => {
  $(input).addEventListener('input', () => { $(out).textContent = $(input).value; });
});

/* ── run ─────────────────────────────────────────────────── */
$('run').addEventListener('click', run);

async function run() {
  const button = $('run');
  if (button.disabled) return;
  button.disabled = true;
  $('result').classList.add('hidden');

  try {
    const videoPath = await resolveSource();
    say('已提交，正在分析…');

    const params = {
      video_path: videoPath,
      topic: $('topic').value.trim(),
      target_segments: Number($('segments').value),
      total_duration: Number($('duration').value),
      semantic_weight: Number($('w-sem').value),
      visual_weight: Number($('w-vis').value),
      audio_weight: Number($('w-aud').value),
      enable_content_analysis: $('use-asr').checked,
      asr_model_size: $('asr-model').value,
      render: $('render').checked,
    };

    const result = await runJob(params);
    render(result);
    say(`完成 · ${result.selected_segments.length} 个片段`, 'ok');
  } catch (error) {
    console.error(error);
    say(error.message || '失败', 'error');
  } finally {
    button.disabled = false;
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

  say('正在上传…');
  const form = new FormData();
  form.append('file', state.file);
  const response = await request('/upload/video', { method: 'POST', body: form });
  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.status !== 'success') {
    throw new Error(body.detail || body.message || `上传失败 (${response.status})`);
  }
  return body.file.saved_path;
}

/** Submit as a background job and poll. Falls back to the synchronous route. */
async function runJob(params) {
  const submission = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'multi_segment_clipping', params }),
  });

  if (submission.status === 404 || submission.status === 405) {
    const direct = await request('/video/multi_segment_clipping', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const body = await direct.json().catch(() => ({}));
    if (!direct.ok) throw new Error(detailOf(body) || `请求失败 (${direct.status})`);
    return body;
  }

  const submitted = await submission.json().catch(() => ({}));
  if (!submission.ok) throw new Error(detailOf(submitted) || `提交失败 (${submission.status})`);
  return pollJob(submitted.job_id);
}

async function pollJob(jobId) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));

    const response = await request(`/jobs/${jobId}`);
    if (!response.ok) throw new Error(`无法查询任务状态 (${response.status})`);
    const job = await response.json();

    if (!job.done) {
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      const step = (job.progress && job.progress.step) || job.status;
      say(`${step} · 已用时 ${elapsed}s`);
      continue;
    }
    if (job.status === 'succeeded') return job.result;
    if (job.status === 'cancelled') throw new Error('任务已取消');
    throw new Error(job.error || '任务失败');
  }
  throw new Error('任务超时');
}

function detailOf(body) {
  const detail = body && body.detail;
  if (!detail) return body && body.message;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || '').filter(Boolean).join('；');
  return detail.message || JSON.stringify(detail);
}

/* ── output ──────────────────────────────────────────────── */
function render(result) {
  const segments = result.selected_segments || [];
  const analysis = result.analysis || {};
  const quality = result.quality_metrics || {};
  const transcription = analysis.transcription || {};

  const source = {
    whisper: `语音识别 · ${transcription.segment_count || 0} 段`,
    provided_subtitle: '使用了提供的字幕',
    disabled: '未做语音识别',
    unavailable: '语音识别不可用，已按画面和音频选段',
  }[transcription.source] || transcription.source || '';

  $('summary').innerHTML = [
    tile('素材时长', `${(analysis.total_video_duration || 0).toFixed(1)}s`),
    tile('选出', `${segments.length} 段 / ${(analysis.total_output_duration || 0).toFixed(1)}s`),
    tile('候选窗口', analysis.analyzed_segments || 0),
    tile('综合分', (quality.overall_score || 0).toFixed(2)),
    source ? `<p class="note">${escapeHtml(source)}</p>` : '',
  ].join('');

  const body = $('clips').querySelector('tbody');
  body.innerHTML = segments.map((segment, index) => {
    const details = segment.semantic_details || {};
    const why = details.reason || segment.type || '';
    const text = segment.preview_text || '';
    return `<tr>
      <td>${index + 1}</td>
      <td class="num">${hms(segment.start_time)}</td>
      <td class="num">${hms(segment.end_time)}</td>
      <td class="num">${Number(segment.duration).toFixed(1)}s</td>
      <td class="num strong">${Number(segment.score).toFixed(2)}</td>
      <td class="num">${Number(segment.semantic_score).toFixed(2)}</td>
      <td class="num">${Number(segment.visual_score).toFixed(2)}</td>
      <td class="num">${Number(segment.audio_score).toFixed(2)}</td>
      <td>${why ? `<b>${escapeHtml(why)}</b>` : ''}${text ? `<span class="quote">${escapeHtml(text)}</span>` : ''}</td>
    </tr>`;
  }).join('');

  const files = result.decision_list || {};
  const labels = { md: '清单 Markdown', edl: 'EDL（导入剪辑软件）', srt: '字幕 SRT' };
  const targets = Object.entries(files).map(([kind, path]) => ({ path, label: labels[kind] || kind }));
  if (result.output_video) {
    targets.unshift({ path: result.output_video, label: '下载成片 MP4', strong: true });
  }

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
        const url = URL.createObjectURL(await response.blob());
        const temp = document.createElement('a');
        temp.href = url;
        temp.download = link.download;
        temp.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        say(error.message || '下载失败', 'error');
      }
    });
    box.appendChild(link);
  });

  const preview = $('preview');
  if (result.output_video) {
    // Same reason: give the player a blob it can read.
    request(`/${result.output_video}`)
      .then((response) => (response.ok ? response.blob() : null))
      .then((blob) => {
        if (!blob) return;
        preview.src = URL.createObjectURL(blob);
        preview.classList.remove('hidden');
      })
      .catch(() => { /* the download link still works */ });
  } else {
    preview.removeAttribute('src');
    preview.classList.add('hidden');
  }

  $('result').classList.remove('hidden');
  $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

const tile = (label, value) =>
  `<div class="tile"><span>${label}</span><b>${value}</b></div>`;

function hms(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const m = String(Math.floor(total / 60)).padStart(2, '0');
  const s = String(Math.floor(total % 60)).padStart(2, '0');
  return `${m}:${s}`;
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = String(value);
  return div.innerHTML;
}

function say(message, kind = '') {
  const el = $('status');
  el.textContent = message;
  el.className = `status ${kind}`;
}
