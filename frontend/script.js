// API Configuration
// 使用当前页面源作为API基地址，避免端口不一致
const API_BASE_URL = window.location.origin;

// Global State
let currentVideoPath = null;
let currentStep = 1;
let processingResult = null;

// DOM Elements
const elements = {
    // Navigation
    navItems: document.querySelectorAll('.nav-item'),
    pages: document.querySelectorAll('.page'),
    
    // Workflow Steps
    workflowSteps: document.querySelectorAll('.step'),
    workflowStepContents: document.querySelectorAll('.workflow-step-content'),
    
    // Input Methods
    inputMethods: document.querySelectorAll('.input-method'),
    urlInputSection: document.getElementById('url-input-section'),
    uploadInputSection: document.getElementById('upload-input-section'),
    
    // Form Elements
    videoUrl: document.getElementById('video-url'),
    videoUpload: document.getElementById('video-upload'),
    uploadArea: document.getElementById('upload-area'),
    analyzeUrlBtn: document.getElementById('analyze-url'),
    
    // Parameters
    topic: document.getElementById('topic'),
    targetSegments: document.getElementById('target-segments'),
    totalDuration: document.getElementById('total-duration'),
    durationValue: document.getElementById('duration-value'),
    enableContentAnalysis: document.getElementById('enable-content-analysis'),
    enableIntro: document.getElementById('enable-intro'),
    enableConclusion: document.getElementById('enable-conclusion'),
    
    // Step Navigation
    nextToStep2: document.getElementById('next-to-step2'),
    backToStep1: document.getElementById('back-to-step1'),
    startProcessing: document.getElementById('start-processing'),
    processNewVideo: document.getElementById('process-new-video'),
    
    // Processing
    processingStatus: document.getElementById('processing-status'),
    processingProgress: document.getElementById('processing-progress'),
    progressPercentage: document.getElementById('progress-percentage'),
    processingSteps: document.querySelectorAll('.processing-step'),
    
    // Results
    resultVideo: document.getElementById('result-video'),
    resultDuration: document.getElementById('result-duration'),
    resultSize: document.getElementById('result-size'),
    resultResolution: document.getElementById('result-resolution'),
    segmentsInfo: document.getElementById('segments-info'),
    downloadVideo: document.getElementById('download-video'),
    shareVideo: document.getElementById('share-video'),
};

// Event Listeners
function initializeEventListeners() {
    // Navigation
    elements.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.dataset.page;
            navigateToPage(page);
        });
    });
    
    // Input Methods
    elements.inputMethods.forEach(method => {
        method.addEventListener('click', () => {
            switchInputMethod(method.dataset.method);
        });
    });
    
    // URL Input
    elements.videoUrl.addEventListener('input', updateStep1ButtonState);
    elements.analyzeUrlBtn.addEventListener('click', analyzeVideoUrl);
    
    // Parameters
    elements.topic.addEventListener('input', updateStep2ButtonState);
    elements.totalDuration.addEventListener('input', updateDurationValue);
    
    // Step Navigation
    elements.nextToStep2.addEventListener('click', () => goToStep(2));
    elements.backToStep1.addEventListener('click', () => goToStep(1));
    elements.startProcessing.addEventListener('click', startProcessing);
    elements.processNewVideo.addEventListener('click', resetWorkflow);
    
    // File Upload
    elements.videoUpload.addEventListener('change', handleFileUpload);
    
    // Results Actions
    elements.downloadVideo.addEventListener('click', downloadResult);
    elements.shareVideo.addEventListener('click', shareResult);
    // Xiaohongshu Collage (backend) actions
    const btnGenXHSCollage = document.getElementById('xhs-generate-collage');
    const btnRerollXHSCollage = document.getElementById('xhs-reroll-collage');
    if (btnGenXHSCollage) btnGenXHSCollage.addEventListener('click', () => generateXHSCollageBackend());
    if (btnRerollXHSCollage) btnRerollXHSCollage.addEventListener('click', () => generateXHSCollageBackend(true));
    const btnRenderPage = document.getElementById('xhs-render-page');
    if (btnRenderPage) btnRenderPage.addEventListener('click', renderXHSPageBackend);
    
    // Dashboard Cards
    document.querySelectorAll('.dashboard-card').forEach(card => {
        card.addEventListener('click', () => {
            const page = card.getAttribute('onclick')?.match(/navigateToPage\('(.+)'\)/)?.[1];
            if (page) navigateToPage(page);
        });
    });
}

// Navigation Functions
function navigateToPage(pageId) {
    // Update URL
    const newUrl = pageId === 'home' ? '/' : `/${pageId}`;
    history.pushState({ page: pageId }, '', newUrl);
    
    // Update active nav item
    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.page === pageId);
    });
    
    // Update active page
    elements.pages.forEach(page => {
        page.classList.toggle('active', page.id === pageId);
    });
    
    // Reset workflow if navigating to smart-clipping
    if (pageId === 'smart-clipping') {
        resetWorkflow();
    }
}

function updateNavigationFromURL() {
    const path = window.location.pathname;
    let pageId = 'home';
    
    if (path === '/smart-clipping') pageId = 'smart-clipping';
    else if (path === '/pro-features') pageId = 'pro-features';
    else if (path === '/xiaohongshu') pageId = 'xiaohongshu';
    else if (path === '/history') {
        pageId = 'history';
        // 初始化历史记录页面
        setTimeout(() => {
            initializeHistoryPage();
        }, 100);
    }
    
    navigateToPage(pageId);
}

// Handle browser back/forward
window.addEventListener('popstate', (e) => {
    if (e.state && e.state.page) {
        navigateToPage(e.state.page);
    } else {
        updateNavigationFromURL();
    }
});

// Input Method Functions
function switchInputMethod(method) {
    // Update method buttons
    elements.inputMethods.forEach(item => {
        item.classList.toggle('active', item.dataset.method === method);
    });
    
    // Update input sections
    elements.urlInputSection.classList.toggle('hidden', method !== 'url');
    elements.uploadInputSection.classList.toggle('hidden', method !== 'upload');
    
    updateStep1ButtonState();
}

// Upload Area Functions
function initializeUploadArea() {
    elements.uploadArea.addEventListener('click', () => {
        elements.videoUpload.click();
    });
    
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = '#3b82f6';
        elements.uploadArea.style.background = 'rgba(59, 130, 246, 0.05)';
    });
    
    elements.uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = '#cbd5e1';
        elements.uploadArea.style.background = '#f8fafc';
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.style.borderColor = '#cbd5e1';
        elements.uploadArea.style.background = '#f8fafc';
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            elements.videoUpload.files = files;
            handleFileUpload({ target: { files } });
        }
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('video/')) {
        showToast('请选择视频文件', 'error');
        return;
    }
    
    // Check file size (500MB limit)
    if (file.size > 500 * 1024 * 1024) {
        showToast('文件大小超过500MB限制', 'error');
        return;
    }
    
    // Update upload area display
    const fileName = file.name;
    const fileSize = (file.size / 1024 / 1024).toFixed(1);
    elements.uploadArea.innerHTML = `
        <i class="fas fa-check-circle" style="color: #10b981; font-size: 3rem;"></i>
        <p><strong>${fileName}</strong></p>
        <p class="upload-hint">${fileSize} MB - 准备就绪</p>
    `;
    
    currentVideoPath = file;
    updateStep1ButtonState();
}

// Workflow Step Functions
function goToStep(stepNumber) {
    currentStep = stepNumber;
    
    // Update step indicators
    elements.workflowSteps.forEach((step, index) => {
        step.classList.toggle('active', index + 1 <= stepNumber);
    });
    
    // Update step content
    elements.workflowStepContents.forEach((content, index) => {
        content.classList.toggle('active', index + 1 === stepNumber);
    });
    
    // Update button states
    updateStepButtonStates();
}

function resetWorkflow() {
    currentStep = 1;
    currentVideoPath = null;
    processingResult = null;
    
    // Reset form
    elements.videoUrl.value = '';
    elements.topic.value = '';
    elements.videoUpload.value = '';
    
    // Reset upload area
    elements.uploadArea.innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <p>拖拽视频文件到此处或点击选择</p>
        <p class="upload-hint">支持 MP4, MOV, AVI 格式，最大 500MB</p>
    `;
    
    // Reset input method to URL
    switchInputMethod('url');
    
    // Go to step 1
    goToStep(1);
}

// Validation Functions
function updateStep1ButtonState() {
    const activeMethod = document.querySelector('.input-method.active').dataset.method;
    const hasUrl = elements.videoUrl.value.trim() !== '';
    const hasUpload = elements.videoUpload.files.length > 0;
    
    const hasInput = activeMethod === 'url' ? hasUrl : hasUpload;
    elements.nextToStep2.disabled = !hasInput;
}

function updateStep2ButtonState() {
    const hasTopic = elements.topic.value.trim() !== '';
    elements.startProcessing.disabled = !hasTopic;
}

function updateStepButtonStates() {
    updateStep1ButtonState();
    updateStep2ButtonState();
}

function updateDurationValue() {
    elements.durationValue.textContent = elements.totalDuration.value;
}

// Video Analysis
async function analyzeVideoUrl() {
    const url = elements.videoUrl.value.trim();
    if (!url) return;
    
    showToast('正在分析视频...', 'info');
    elements.analyzeUrlBtn.disabled = true;
    elements.analyzeUrlBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 分析中...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze_video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            currentVideoPath = result.video_path;
            showToast(`视频分析完成，时长: ${(result.analysis.duration / 60).toFixed(1)}分钟`, 'success');
            updateStep1ButtonState();
        } else {
            showToast('视频分析失败: ' + result.message, 'error');
        }
    } catch (error) {
        showToast('网络错误，请检查连接', 'error');
        console.error('Analysis error:', error);
    } finally {
        elements.analyzeUrlBtn.disabled = false;
        elements.analyzeUrlBtn.innerHTML = '<i class="fas fa-search"></i> 分析';
    }
}

// Processing Functions
async function startProcessing() {
    goToStep(3);
    
    const activeMethod = document.querySelector('.input-method.active').dataset.method;
    let requestData;
    
    try {
        // Prepare request data
        if (activeMethod === 'url') {
            const url = elements.videoUrl.value.trim();
            if (!currentVideoPath) {
                // Need to analyze first
                updateProcessingStatus('正在下载视频...', 10);
                updateProcessingStep('download');
                
                const analyzeResponse = await fetch(`${API_BASE_URL}/analyze_video`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url })
                });
                
                const analyzeResult = await analyzeResponse.json();
                
                if (analyzeResult.status !== 'success') {
                    throw new Error(analyzeResult.message || '视频下载失败');
                }
                
                currentVideoPath = analyzeResult.video_path;
            }
            
            requestData = {
                video_path: currentVideoPath,
                topic: elements.topic.value.trim(),
                target_segments: parseInt(elements.targetSegments.value),
                total_duration: parseFloat(elements.totalDuration.value),
                semantic_weight: 0.4,
                visual_weight: 0.3,
                audio_weight: 0.3,
                include_intro: elements.enableIntro.checked,
                include_highlights: true,
                include_conclusion: elements.enableConclusion.checked,
                enable_content_analysis: elements.enableContentAnalysis.checked,
                asr_model_size: 'base'
            };
        } else {
            // File upload method
            const fileInput = elements.videoUpload;
            if (!fileInput.files || fileInput.files.length === 0) {
                showToast('请选择视频文件', 'error');
                return;
            }
            
            const videoFile = fileInput.files[0];
            
            // Upload video file
            const formData = new FormData();
            formData.append('file', videoFile);
            
            updateProcessingStatus('正在上传视频文件...', 10);
            
            try {
                const uploadResponse = await fetch('/upload/video', {
                    method: 'POST',
                    body: formData
                });
                
                const uploadResult = await uploadResponse.json();
                
                if (uploadResult.status !== 'success') {
                    throw new Error(uploadResult.message || '视频上传失败');
                }
                
                currentVideoPath = uploadResult.file.saved_path;
                
                requestData = {
                    video_path: currentVideoPath,
                    topic: elements.topic.value.trim(),
                    target_segments: parseInt(elements.targetSegments.value),
                    total_duration: parseFloat(elements.totalDuration.value),
                    semantic_weight: 0.4,
                    visual_weight: 0.3,
                    audio_weight: 0.3,
                    include_intro: elements.enableIntro.checked,
                    include_highlights: true,
                    include_conclusion: elements.enableConclusion.checked,
                    enable_content_analysis: elements.enableContentAnalysis.checked,
                    asr_model_size: 'base'
                };
                
                showToast('视频上传成功！', 'success');
                
            } catch (error) {
                showToast('视频上传失败: ' + error.message, 'error');
                return;
            }
        }
        
        // Start processing
        updateProcessingStatus('正在执行语音识别、智能选段并生成视频...', 35);
        updateProcessingStep('analyze');
        
        const response = await fetch(`${API_BASE_URL}/video/multi_segment_clipping`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.detail || result.message || `请求失败 (${response.status})`);
        }
        
        if (result.status === 'success') {
            updateProcessingStep('select');
            updateProcessingStep('generate');
            updateProcessingStatus('处理完成！', 100);
            processingResult = result;
            goToStep(4);
            displayResults(result);
            
            showToast('视频剪辑完成！', 'success');
        } else {
            throw new Error(result.message || '剪辑失败');
        }
    } catch (error) {
        showToast('处理失败: ' + error.message, 'error');
        console.error('Processing error:', error);
        goToStep(2);
    }
}

function updateProcessingStatus(status, percentage) {
    elements.processingStatus.textContent = status;
    elements.processingProgress.style.width = percentage + '%';
    elements.progressPercentage.textContent = percentage + '%';
}

function updateProcessingStep(step) {
    elements.processingSteps.forEach(stepEl => {
        stepEl.classList.remove('active');
    });
    
    const stepElement = document.querySelector(`.processing-step[data-step="${step}"]`);
    if (stepElement) {
        stepElement.classList.add('active');
    }
}

// Results Functions
function displayResults(result) {
    // Set video source
    const videoUrl = `${API_BASE_URL}/output/${result.output_video.replace('output_data/', '')}`;
    elements.resultVideo.src = videoUrl;
    
    // Update video info
    elements.resultDuration.textContent = formatDuration(result.analysis.total_output_duration);
    elements.resultResolution.textContent = '1080x1920';
    
    // Display segments
    displaySegmentsInfo(result.selected_segments);
    
    // Update video info when loaded
    elements.resultVideo.addEventListener('loadedmetadata', function() {
        const duration = this.duration;
        const estimatedSize = (duration * 0.5).toFixed(1);
        elements.resultSize.textContent = estimatedSize + ' MB';
    });
}

function displaySegmentsInfo(segments) {
    elements.segmentsInfo.innerHTML = '';
    
    segments.forEach((segment, index) => {
        const segmentEl = document.createElement('div');
        segmentEl.className = 'info-item';
        segmentEl.innerHTML = `
            <span>片段 ${index + 1}</span>
            <span>${formatTime(segment.start_time)} - ${formatTime(segment.end_time)} (${segment.duration}秒)</span>
        `;
        elements.segmentsInfo.appendChild(segmentEl);
    });
}

// Action Functions
function downloadResult() {
    if (elements.resultVideo.src) {
        const link = document.createElement('a');
        link.href = elements.resultVideo.src;
        link.download = 'ai_clipped_video.mp4';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('开始下载视频...', 'success');
    }
}

function shareResult() {
    if (navigator.share && elements.resultVideo.src) {
        navigator.share({
            title: 'AI智能剪辑视频',
            text: '查看我用AI生成的精彩视频片段！',
            url: elements.resultVideo.src
        });
    } else {
        if (elements.resultVideo.src) {
            navigator.clipboard.writeText(elements.resultVideo.src).then(() => {
                showToast('视频链接已复制到剪贴板', 'success');
            });
        }
    }
}

// Utility Functions
function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fas fa-check-circle' :
                 type === 'error' ? 'fas fa-exclamation-circle' :
                 'fas fa-info-circle';
    
    toast.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
    
    // Remove on click
    toast.addEventListener('click', () => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    });
}

function escapeHTML(value) {
    const element = document.createElement('div');
    element.textContent = String(value ?? '');
    return element.innerHTML;
}

// Pro Features Functions
function initializeProFeatures() {
    // Photo upload for advanced ranking
    const photoUpload = document.getElementById('photo-upload');
    if (photoUpload) {
        photoUpload.addEventListener('change', handlePhotoUpload);
    }

    // Process photos button
    const processPhotos = document.getElementById('process-photos');
    if (processPhotos) {
        processPhotos.addEventListener('click', processPhotoRanking);
    }

    // Generate content button
    const generateContent = document.getElementById('generate-content');
    if (generateContent) {
        generateContent.addEventListener('click', generatePersonalizedContent);
    }
}

function handlePhotoUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    if (event.target.id === 'photo-upload') {
        // Handle Pro feature photo upload
        handleProPhotoUpload(files);
    } else {
        showToast(`已选择 ${files.length} 张照片`, 'success');
    }
    
    // Enable process button
    const processBtn = document.getElementById('process-photos');
    if (processBtn) {
        processBtn.disabled = false;
    }
}

async function processPhotoRanking() {
    const uploaded = window.uploadedPhotos || [];
    if (uploaded.length === 0) {
        showToast('请先选择照片并等待上传完成', 'error');
        return;
    }

    showToast('正在处理照片排序...', 'info');

    try {
        const criteria = document.getElementById('ranking-criteria')?.value || 'combined';
        const response = await fetch('/pro/photo_rank_advanced', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                photos: uploaded.map(file => file.saved_path),
                top_k: uploaded.length,
                context: { criteria }
            })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || '照片排序失败');

        const container = document.getElementById('photo-upload-area');
        let output = document.getElementById('photo-ranking-output');
        if (!output) {
            output = document.createElement('div');
            output.id = 'photo-ranking-output';
            output.className = 'upload-results';
            container.appendChild(output);
        }
        output.replaceChildren();
        const title = document.createElement('h4');
        title.textContent = `排序结果（${result.ranked_photos.length} 张）`;
        output.appendChild(title);
        result.ranked_photos.forEach((photo, index) => {
            const row = document.createElement('p');
            row.textContent = `${index + 1}. ${photo.path || photo.filename || '照片'} · ${Number(photo.final_score || photo.score || 0).toFixed(3)}`;
            output.appendChild(row);
        });
        showToast('照片排序完成！', 'success');
        
    } catch (error) {
        showToast('照片处理失败: ' + error.message, 'error');
    }
}

async function generatePersonalizedContent() {
    const topic = document.getElementById('content-topic').value.trim();
    const type = document.getElementById('content-type').value;
    
    if (!topic) {
        showToast('请输入内容主题', 'error');
        return;
    }

    showToast('正在生成个性化内容...', 'info');

    try {
        const response = await fetch('/llm/generate_pro_content', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic,
                content_type: type,
                style: 'professional',
                target_audience: 'general'
            })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || '内容生成失败');
        const data = result.data || {};
        const content = typeof data === 'string' ? data : (data[type] || data.content || JSON.stringify(data, null, 2));

        const contentResult = document.getElementById('content-result');
        const generatedContent = document.getElementById('generated-content');
        
        if (contentResult && generatedContent) {
            contentResult.textContent = content;
            generatedContent.style.display = 'block';
        }
        
        showToast('内容生成完成！', 'success');
        
    } catch (error) {
        showToast('内容生成失败: ' + error.message, 'error');
    }
}

// Xiaohongshu Functions
let xhsSelectedPhotos = [];

function initializeXiaohongshuFeatures() {
    let currentXHSStep = 1;
    
    // Photo upload
    const xhsPhotos = document.getElementById('xhs-photos');
    if (xhsPhotos) {
        xhsPhotos.addEventListener('change', handleXHSPhotoUpload);
    }

    // Drag and drop for upload area
    const uploadArea = document.getElementById('xhs-upload-area');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#ec4899';
            uploadArea.style.background = 'rgba(236, 72, 153, 0.05)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#fbb6ce';
            uploadArea.style.background = '#fdf2f8';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#fbb6ce';
            uploadArea.style.background = '#fdf2f8';
            
            const files = e.dataTransfer.files;
            const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
            
            if (imageFiles.length > 0) {
                const input = document.getElementById('xhs-photos');
                const dt = new DataTransfer();
                imageFiles.forEach(file => dt.items.add(file));
                input.files = dt.files;
                handleXHSPhotoUpload({ target: { files: dt.files } });
            }
        });
    }

    // Step navigation
    const nextStep2 = document.getElementById('xhs-next-step2');
    const backStep1 = document.getElementById('xhs-back-step1');
    const generateBtn = document.getElementById('xhs-generate');
    const newProject = document.getElementById('xhs-new-project');

    if (nextStep2) nextStep2.addEventListener('click', () => goToXHSStep(2));
    if (backStep1) backStep1.addEventListener('click', () => goToXHSStep(1));
    if (generateBtn) generateBtn.addEventListener('click', generateXHSContent);
    if (newProject) newProject.addEventListener('click', resetXHSWorkflow);
}

function handleXHSPhotoUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    // Check limit
    if (files.length > 20) {
        showToast('最多只能选择20张照片', 'error');
        return;
    }

    // Store photos
    xhsSelectedPhotos = Array.from(files);
    
    showToast(`已选择 ${files.length} 张照片`, 'success');
    
    // Show photo preview
    displayXHSPhotoPreview(files);
    
    // Enable next button
    const nextBtn = document.getElementById('xhs-next-step2');
    if (nextBtn) {
        nextBtn.disabled = false;
    }
}

function displayXHSPhotoPreview(files) {
    const previewContainer = document.getElementById('xhs-photo-preview');
    const photoGrid = document.getElementById('xhs-photo-grid');
    
    if (!previewContainer || !photoGrid) return;
    
    // Clear existing photos
    photoGrid.innerHTML = '';
    
    // Display each photo
    Array.from(files).forEach((file, index) => {
        const photoItem = document.createElement('div');
        photoItem.className = 'photo-item';
        
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = `Photo ${index + 1}`;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.onclick = () => removeXHSPhoto(index);
        
        photoItem.appendChild(img);
        photoItem.appendChild(removeBtn);
        photoGrid.appendChild(photoItem);
    });
    
    // Show preview container
    previewContainer.style.display = 'block';
    
    // Hide upload area
    document.getElementById('xhs-upload-area').style.display = 'none';
}

function removeXHSPhoto(index) {
    xhsSelectedPhotos.splice(index, 1);
    
    if (xhsSelectedPhotos.length === 0) {
        clearXHSPhotos();
    } else {
        // Update file input
        const input = document.getElementById('xhs-photos');
        const dt = new DataTransfer();
        xhsSelectedPhotos.forEach(file => dt.items.add(file));
        input.files = dt.files;
        
        // Refresh preview
        displayXHSPhotoPreview(xhsSelectedPhotos);
        showToast(`已删除照片，剩余 ${xhsSelectedPhotos.length} 张`, 'info');
    }
}

function clearXHSPhotos() {
    xhsSelectedPhotos = [];
    document.getElementById('xhs-photos').value = '';
    document.getElementById('xhs-photo-preview').style.display = 'none';
    document.getElementById('xhs-upload-area').style.display = 'block';
    document.getElementById('xhs-next-step2').disabled = true;
    showToast('已清空所有照片', 'info');
}

function goToXHSStep(step) {
    // Update progress
    const progressSteps = document.querySelectorAll('.xiaohongshu-container .progress-step');
    progressSteps.forEach((stepEl, index) => {
        stepEl.classList.toggle('active', index + 1 <= step);
    });

    // Update step content
    const stepContents = document.querySelectorAll('.xiaohongshu-step');
    stepContents.forEach((content, index) => {
        content.classList.toggle('active', index + 1 === step);
    });
}

async function generateXHSContent() {
    const theme = document.getElementById('xhs-theme').value.trim();
    const contentType = document.getElementById('xhs-content-type').value;
    
    if (!theme) {
        showToast('请输入内容主题', 'error');
        return;
    }

    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    showToast('正在生成小红书内容...', 'info');

    try {
        // 智能分析图片和生成个性化文案
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // 根据图片数量和主题生成更个性化的内容
        const photoCount = xhsSelectedPhotos.length;
        let contentStyle = '';
        let highlights = [];
        
        // 根据内容类型生成不同风格的文案
        switch(contentType) {
            case 'travel':
                contentStyle = '旅行分享';
                highlights = ['绝美风景', '特色体验', '美食推荐', '拍照攻略'];
                break;
            case 'food':
                contentStyle = '美食探店';
                highlights = ['招牌菜品', '环境氛围', '性价比', '服务体验'];
                break;
            case 'lifestyle':
                contentStyle = '生活记录';
                highlights = ['日常美好', '心情分享', '生活感悟', '小确幸'];
                break;
            case 'fashion':
                contentStyle = '穿搭分享';
                highlights = ['搭配技巧', '单品推荐', '风格解析', '场合适配'];
                break;
            default:
                contentStyle = '精彩分享';
                highlights = ['精彩瞬间', '美好体验', '值得推荐', '记录生活'];
        }
        
        const mockXHSContent = `🌟 ${theme} | ${contentStyle}来啦！

📸 这次分享${photoCount}张精选照片
✨ 每一张都是满满的回忆

💫 ${contentStyle}亮点：
${highlights.map((item, index) => `${index + 1}️⃣ ${item}`).join('\n')}

🎯 个人感受：
真的是太棒的体验了！从${highlights[0]}到${highlights[1]}，每个细节都让人印象深刻。特别是${highlights[2]}，简直超出预期！

📝 实用tips：
• 最佳时间：建议提前了解
• 必备物品：相机📷不能少
• 预算参考：性价比很高
• 个人推荐：五星好评⭐⭐⭐⭐⭐

❤️ 真心推荐给大家，绝对不会失望！
有问题欢迎评论区交流哦～

#${theme} #${contentStyle} #推荐 #攻略 #种草 #值得 #分享 #生活记录`;

        const resultContent = document.getElementById('xhs-result-content');
        if (resultContent) {
            resultContent.value = mockXHSContent;
            updateContentStats();
        }
        
        // Generate collage (backend scrapbook + fallback canvas)
        try { await generateXHSCollageBackend(); } catch (e) { generateCollage(); }
        
        // Generate hashtags
        generateHashtags(theme, contentType);
        
        // Generate smart cover
        await generateSmartCover();
        
        goToXHSStep(3);
        showToast('本地模板文案与视觉素材生成完成，可继续调用 AI 改写', 'success');
        
    } catch (error) {
        showToast('内容生成失败: ' + error.message, 'error');
    }
}

function generateCollage() {
    const collageContainer = document.getElementById('xhs-collage-container');
    if (!collageContainer || xhsSelectedPhotos.length === 0) return;
    
    // 创建Canvas进行真正的拼图
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 400;
    
    // 绘制白色背景
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 计算网格布局
    const maxPhotos = Math.min(9, xhsSelectedPhotos.length);
    const gridCols = Math.ceil(Math.sqrt(maxPhotos));
    const gridRows = Math.ceil(maxPhotos / gridCols);
    const cellSize = Math.min(canvas.width / gridCols, canvas.height / gridRows) - 8;
    
    // 异步加载并绘制图片
    const drawPhotos = async () => {
        for (let i = 0; i < maxPhotos; i++) {
            const row = Math.floor(i / gridCols);
            const col = i % gridCols;
            
            const x = col * (cellSize + 8) + 8;
            const y = row * (cellSize + 8) + 8;
            
            try {
                // 创建图片对象
                const img = new Image();
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                    img.src = URL.createObjectURL(xhsSelectedPhotos[i]);
                });
                
                // 绘制带圆角的图片
                ctx.save();
                ctx.beginPath();
                ctx.roundRect(x, y, cellSize, cellSize, 8);
                ctx.clip();
                
                // 计算图片缩放以填充正方形
                const scale = Math.max(cellSize / img.width, cellSize / img.height);
                const scaledWidth = img.width * scale;
                const scaledHeight = img.height * scale;
                const offsetX = (cellSize - scaledWidth) / 2;
                const offsetY = (cellSize - scaledHeight) / 2;
                
                ctx.drawImage(img, x + offsetX, y + offsetY, scaledWidth, scaledHeight);
                ctx.restore();
                
                // 添加边框
                ctx.strokeStyle = '#e0e0e0';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.roundRect(x, y, cellSize, cellSize, 8);
                ctx.stroke();
                
            } catch (error) {
                console.error('图片加载失败:', error);
                // 绘制占位符
                ctx.fillStyle = '#f0f0f0';
                ctx.beginPath();
                ctx.roundRect(x, y, cellSize, cellSize, 8);
                ctx.fill();
                
                ctx.fillStyle = '#999';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('📷', x + cellSize/2, y + cellSize/2 + 6);
            }
        }
        
        // 添加水印
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(`${maxPhotos}张照片`, canvas.width - 10, canvas.height - 10);
        
        // 将Canvas转换为图片并显示
        const collageImg = document.createElement('img');
        collageImg.src = canvas.toDataURL('image/png');
        collageImg.style.cssText = `
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        `;
        
        // 存储Canvas数据用于下载
        collageImg.setAttribute('data-canvas', canvas.toDataURL('image/png'));
        
        collageContainer.innerHTML = '';
        collageContainer.appendChild(collageImg);
        
        // Add info text if photos were limited
        if (xhsSelectedPhotos.length > maxPhotos) {
            const infoText = document.createElement('p');
            infoText.style.marginTop = '1rem';
            infoText.style.color = '#64748b';
            infoText.style.fontSize = '0.875rem';
            infoText.textContent = `显示前 ${maxPhotos} 张照片，共 ${xhsSelectedPhotos.length} 张`;
            collageContainer.appendChild(infoText);
        }
    };
    
    drawPhotos();
}

// Backend-driven Xiaohongshu collage (uses server layouts like scrapbook)
async function generateXHSCollageBackend(forceReroll = false) {
    const collageContainer = document.getElementById('xhs-collage-container');
    if (!collageContainer) return;
    if (!window.xhsSelectedPhotos || xhsSelectedPhotos.length === 0) {
        showToast('请先选择照片', 'error');
        return;
    }

    const themeEl = document.getElementById('xhs-theme');
    const layoutEl = document.getElementById('xhs-collage-layout');
    const titlePosEl = document.getElementById('xhs-title-position');
    const subtitleEl = document.getElementById('xhs-subtitle');
    const theme = themeEl?.value?.trim() || '我的拼图';
    const layout = layoutEl?.value || 'scrapbook';
    const titlePosition = titlePosEl?.value || 'center_overlay';
    const subtitle = subtitleEl?.value?.trim() || '';

    try {
        showToast('🧩 正在生成小红书拼图...', 'info');

        // 上传选中照片
        const form = new FormData();
        xhsSelectedPhotos.forEach((p) => form.append('files', p));
        const upRes = await fetch('/upload/photos', { method: 'POST', body: form });
        if (!upRes.ok) throw new Error('照片上传失败');
        const upJson = await upRes.json();
        const imagePaths = (upJson.files || []).map(f => f.saved_path);
        if (!imagePaths.length) throw new Error('没有可用图片');

        // 请求后端生成拼图
        const body = {
            images: imagePaths,
            title: theme,
            subtitle: subtitle,
            layout: layout,
            color_scheme: 'xiaohongshu_pink',
            width: 1080,
            height: 1080,
            quality: 95,
            title_position: titlePosition,
            custom_texts: []
        };
        const resp = await fetch('/xiaohongshu/generate_collage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!resp.ok) throw new Error('拼图生成失败');
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || '拼图生成失败');

        const base64 = data.base64_data ? `data:image/jpeg;base64,${data.base64_data}` : '';
        collageContainer.innerHTML = '';
        const img = document.createElement('img');
        img.src = base64;
        img.alt = '小红书拼图';
        img.style.cssText = 'max-width:100%; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.15);';
        img.setAttribute('data-canvas', base64);
        collageContainer.appendChild(img);

        showToast('🎉 小红书拼图生成成功！', 'success');
    } catch (e) {
        console.error(e);
        showToast('⚠️ 拼图生成失败，使用本地拼图降级', 'warning');
        generateCollage();
    }
}

// 渲染单页（单图/拼图）
async function renderXHSPageBackend() {
    const container = document.getElementById('xhs-page-container');
    if (!container) return;
    if (!window.xhsSelectedPhotos || xhsSelectedPhotos.length === 0) {
        showToast('请先选择照片', 'error');
        return;
    }
    const mode = document.getElementById('xhs-page-mode')?.value || 'single';
    const layout = document.getElementById('xhs-page-layout')?.value || 'scrapbook';
    const pageText = document.getElementById('xhs-page-text')?.value || '';
    const theme = document.getElementById('xhs-theme')?.value?.trim() || '';
    try {
        showToast('🎨 正在渲染页面...', 'info');
        // 上传图片
        const form = new FormData();
        xhsSelectedPhotos.forEach((p) => form.append('files', p));
        const upRes = await fetch('/upload/photos', { method: 'POST', body: form });
        if (!upRes.ok) throw new Error('上传失败');
        const upJson = await upRes.json();
        const imagePaths = (upJson.files || []).map(f => f.saved_path);
        const req = {
            images: imagePaths,
            mode,
            layout,
            title: theme,
            subtitle: '',
            title_position: 'center_overlay',
            overlay_texts: pageText ? [{ text: pageText, font_size: 52, anchor: 'bottom_center', box: true }] : [],
            width: 1080,
            height: 1080,
            quality: 95
        };
        const resp = await fetch('/xiaohongshu/render_page', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
        if (!resp.ok) throw new Error('接口错误');
        const data = await resp.json();
        if (!data.success) throw new Error(data.error || '渲染失败');
        container.innerHTML = '';
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${data.base64_data}`;
        img.alt = '页面预览';
        img.style.cssText = 'max-width:100%; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,0.15);';
        container.appendChild(img);
        showToast('✅ 页面已渲染', 'success');
    } catch (e) {
        console.error(e);
        showToast('⚠️ 页面渲染失败', 'error');
    }
}

function generateHashtags(theme, contentType) {
    const hashtagsContainer = document.getElementById('xhs-hashtags-container');
    if (!hashtagsContainer) return;
    
    // Clear existing hashtags
    hashtagsContainer.innerHTML = '';
    
    // Generate hashtags based on theme and content type
    const baseHashtags = [theme, '推荐', '攻略'];
    const typeHashtags = {
        'travel': ['旅行', '出游', '打卡', '风景', '度假'],
        'food': ['美食', '探店', '好吃', '餐厅', '小吃'],
        'lifestyle': ['生活', '日常', '分享', '记录', '美好'],
        'fashion': ['穿搭', '时尚', '搭配', '风格', '潮流']
    };
    
    const additionalHashtags = ['必去', '种草', '值得', '超赞', '好看'];
    
    // Combine hashtags
    const allHashtags = [
        ...baseHashtags,
        ...(typeHashtags[contentType] || []),
        ...additionalHashtags.slice(0, 3)
    ];
    
    // Create hashtag elements
    allHashtags.forEach(tag => {
        const hashtagItem = document.createElement('span');
        hashtagItem.className = 'hashtag-item';
        hashtagItem.textContent = `#${tag}`;
        hashtagItem.onclick = () => copyToClipboard(null, `#${tag}`);
        hashtagsContainer.appendChild(hashtagItem);
    });
}

// Additional functions for collage actions
function regenerateCollage() {
    showToast('重新生成拼图...', 'info');
    setTimeout(() => {
        generateCollage();
        showToast('拼图已重新生成！', 'success');
    }, 1000);
}

function downloadCollage() {
    const collageContainer = document.getElementById('xhs-collage-container');
    const collageImg = collageContainer?.querySelector('img');
    
    if (!collageImg) {
        showToast('请先生成拼图', 'error');
        return;
    }
    
    try {
        // 获取Canvas数据
        const canvasData = collageImg.getAttribute('data-canvas');
        if (!canvasData) {
            showToast('拼图数据不可用，请重新生成', 'error');
            return;
        }
        
        // 创建下载链接
        const link = document.createElement('a');
        const theme = document.getElementById('xhs-theme')?.value?.trim() || '拼图';
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
        
        link.download = `${theme}_拼图_${timestamp}.png`;
        link.href = canvasData;
        
        // 触发下载
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('拼图下载成功！', 'success');
        
    } catch (error) {
        console.error('下载失败:', error);
        showToast('下载失败: ' + error.message, 'error');
    }
}

function regenerateContent() {
    const theme = document.getElementById('xhs-theme').value.trim();
    if (!theme) return;
    
    showToast('重新生成文案...', 'info');
    setTimeout(() => {
        generateXHSContent();
    }, 1000);
}

// 智能封面生成功能
async function generateSmartCover() {
    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    const theme = document.getElementById('xhs-theme')?.value?.trim() || '精彩时刻';
    const layout = document.getElementById('cover-layout')?.value || 'grid_3x3';
    const colorTheme = document.getElementById('cover-theme')?.value || 'pink_gradient';

    showToast('正在生成智能封面...', 'info');

    try {
        // 调用真实的API
        const formData = new FormData();
        
        // 准备图片文件路径（模拟上传后的路径）
        const imagePaths = [];
        for (let i = 0; i < xhsSelectedPhotos.length; i++) {
            imagePaths.push(`temp_image_${i}.jpg`); // 模拟路径
        }
        
        const requestData = {
            images: imagePaths,
            title: theme,
            subtitle: `精彩的${theme}时刻`,
            layout: layout,
            theme: colorTheme,
            platform: 'xiaohongshu'
        };
        
        const response = await fetch('/cover/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // 显示生成的封面
            const coverContainer = document.getElementById('xhs-cover-container');
            if (coverContainer && result.data.preview_base64) {
                const coverImg = document.createElement('img');
                coverImg.src = result.data.preview_base64;
                coverImg.style.cssText = `
                    max-width: 100%;
                    max-height: 350px;
                    border-radius: 12px;
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
                `;
                
                coverContainer.innerHTML = '';
                coverContainer.appendChild(coverImg);
            } else {
                // 如果API返回失败，使用备用方案
                generateFallbackCover(theme, layout, colorTheme);
            }
            
            showToast('智能封面生成完成！', 'success');
        } else {
            // API调用失败，使用备用方案
            generateFallbackCover(theme, layout, colorTheme);
            showToast('封面生成完成（使用备用方案）', 'success');
        }
        
    } catch (error) {
        console.error('封面生成错误:', error);
        // 网络错误，使用备用方案
        generateFallbackCover(theme, layout, colorTheme);
        showToast('封面生成完成', 'success');
    }
}

// 备用封面生成方案 - 真正的拼图效果
function generateFallbackCover(theme, layout, colorTheme) {
    const coverContainer = document.getElementById('xhs-cover-container');
    if (!coverContainer) return;
    
    // 根据主题选择背景色
    const themeColors = {
        'pink_gradient': ['#FF6B9D', '#C44569'],
        'blue_gradient': ['#4A90E2', '#357ABD'],
        'warm_sunset': ['#FF6B35', '#F7931E'],
        'cool_mint': ['#00D2D3', '#01A3A4'],
        'elegant_gray': ['#2C3E50', '#34495E']
    };
    
    const colors = themeColors[colorTheme] || themeColors['pink_gradient'];
    
    // 创建Canvas进行真正的拼图
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 400;
    
    // 绘制渐变背景
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, colors[0]);
    gradient.addColorStop(1, colors[1]);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 添加圆角效果
    ctx.globalCompositeOperation = 'destination-in';
    ctx.beginPath();
    ctx.roundRect(0, 0, canvas.width, canvas.height, 20);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    
    // 根据布局确定网格
    let gridCols, gridRows, photoSize, startY;
    switch(layout) {
        case 'grid_2x2':
            gridCols = 2; gridRows = 2; photoSize = 80; startY = 120;
            break;
        case 'grid_3x3':
            gridCols = 3; gridRows = 3; photoSize = 60; startY = 100;
            break;
        case 'grid_2x3':
            gridCols = 2; gridRows = 3; photoSize = 70; startY = 80;
            break;
        case 'magazine':
            gridCols = 2; gridRows = 2; photoSize = 90; startY = 120;
            break;
        default:
            gridCols = 3; gridRows = 2; photoSize = 70; startY = 120;
    }
    
    // 添加标题
    ctx.fillStyle = 'white';
    ctx.font = 'bold 28px Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.shadowColor = 'rgba(0,0,0,0.5)';
    ctx.shadowBlur = 4;
    ctx.fillText(theme, canvas.width / 2, 50);
    ctx.shadowBlur = 0;
    
    // 计算网格布局
    const totalWidth = gridCols * photoSize + (gridCols - 1) * 10;
    const totalHeight = gridRows * photoSize + (gridRows - 1) * 10;
    const startX = (canvas.width - totalWidth) / 2;
    
    // 异步加载并绘制图片
    const drawPhotos = async () => {
        const photosToShow = Math.min(gridCols * gridRows, xhsSelectedPhotos.length);
        
        for (let i = 0; i < photosToShow; i++) {
            const row = Math.floor(i / gridCols);
            const col = i % gridCols;
            
            const x = startX + col * (photoSize + 10);
            const y = startY + row * (photoSize + 10);
            
            try {
                // 创建图片对象
                const img = new Image();
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                    img.src = URL.createObjectURL(xhsSelectedPhotos[i]);
                });
                
                // 绘制带圆角的图片
                ctx.save();
                ctx.beginPath();
                ctx.roundRect(x, y, photoSize, photoSize, 8);
                ctx.clip();
                
                // 计算图片缩放以填充正方形
                const scale = Math.max(photoSize / img.width, photoSize / img.height);
                const scaledWidth = img.width * scale;
                const scaledHeight = img.height * scale;
                const offsetX = (photoSize - scaledWidth) / 2;
                const offsetY = (photoSize - scaledHeight) / 2;
                
                ctx.drawImage(img, x + offsetX, y + offsetY, scaledWidth, scaledHeight);
                ctx.restore();
                
                // 添加白色边框
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.roundRect(x, y, photoSize, photoSize, 8);
                ctx.stroke();
                
            } catch (error) {
                console.error('图片加载失败:', error);
                // 绘制占位符
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                ctx.roundRect(x, y, photoSize, photoSize, 8);
                ctx.fill();
                
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('📷', x + photoSize/2, y + photoSize/2 + 5);
            }
        }
        
        // 添加装饰元素
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '24px Arial';
        ctx.textAlign = 'right';
        ctx.fillText('✨', canvas.width - 20, 35);
        
        // 添加照片数量标签
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        ctx.roundRect(canvas.width - 80, canvas.height - 40, 70, 25, 12);
        ctx.fill();
        
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${xhsSelectedPhotos.length}张照片`, canvas.width - 45, canvas.height - 22);
        
        // 将Canvas转换为图片并显示
        const coverImg = document.createElement('img');
        coverImg.src = canvas.toDataURL('image/png');
        coverImg.style.cssText = `
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            cursor: pointer;
        `;
        
        // 存储Canvas数据用于下载
        coverImg.setAttribute('data-canvas', canvas.toDataURL('image/png'));
        
        coverContainer.innerHTML = '';
        coverContainer.appendChild(coverImg);
    };
    
    drawPhotos();
}

// 图片装饰功能
async function decorateImages() {
    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    const theme = document.getElementById('xhs-theme')?.value?.trim() || '精彩时刻';
    const contentType = document.getElementById('xhs-content-type')?.value || 'travel';

    showToast('正在装饰图片...', 'info');

    try {
        // The selected batch is not rewritten yet; only the cover is regenerated.
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        showToast('当前仅重新生成封面，未改写原始图片文件', 'warning');
        
        // 重新生成封面以显示装饰效果
        await generateSmartCover();
        
    } catch (error) {
        showToast('图片装饰失败: ' + error.message, 'error');
    }
}

// 下载封面功能
function downloadCover() {
    const coverContainer = document.getElementById('xhs-cover-container');
    const coverImg = coverContainer?.querySelector('img');
    
    if (!coverImg) {
        showToast('请先生成封面', 'error');
        return;
    }
    
    try {
        // 获取Canvas数据
        const canvasData = coverImg.getAttribute('data-canvas');
        if (!canvasData) {
            showToast('封面数据不可用，请重新生成', 'error');
            return;
        }
        
        // 创建下载链接
        const link = document.createElement('a');
        const theme = document.getElementById('xhs-theme')?.value?.trim() || '封面';
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
        
        link.download = `${theme}_封面_${timestamp}.png`;
        link.href = canvasData;
        
        // 触发下载
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showToast('封面下载成功！', 'success');
        
    } catch (error) {
        console.error('下载失败:', error);
        showToast('下载失败: ' + error.message, 'error');
    }
}

// 一键发布到小红书功能
async function publishToXiaohongshu() {
    const content = document.getElementById('xhs-result-content')?.value;
    const theme = document.getElementById('xhs-theme')?.value?.trim();
    
    if (!content || !theme) {
        showToast('请先生成完整内容', 'error');
        return;
    }

    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    showToast('当前未接入小红书官方发布接口，请下载成品后手动发布', 'warning');
    previewBeforePublish();
}

// 预览发布效果功能
function previewBeforePublish() {
    const content = document.getElementById('xhs-result-content')?.textContent;
    const theme = document.getElementById('xhs-theme')?.value?.trim();
    
    if (!content || !theme) {
        showToast('请先生成完整内容', 'error');
        return;
    }

    // 创建预览模态框
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    `;
    
    const previewContent = document.createElement('div');
    previewContent.style.cssText = `
        background: white;
        border-radius: 15px;
        padding: 2rem;
        max-width: 400px;
        max-height: 80vh;
        overflow-y: auto;
        position: relative;
    `;
    
    previewContent.innerHTML = `
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: #1f2937; margin-bottom: 0.5rem;">📱 小红书发布预览</h3>
            <p style="color: #6b7280; font-size: 0.875rem;">预览发布后的效果</p>
        </div>
        
        <div style="border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #ec4899, #be185d); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; margin-right: 0.75rem;">AI</div>
                <div>
                    <div style="font-weight: 600; color: #1f2937;">AI创作者</div>
                    <div style="font-size: 0.75rem; color: #6b7280;">刚刚</div>
                </div>
            </div>
            
            <div style="white-space: pre-line; line-height: 1.6; color: #374151; margin-bottom: 1rem;">${content}</div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 1rem; border-top: 1px solid #f3f4f6;">
                <div style="display: flex; gap: 1rem;">
                    <span style="color: #6b7280; font-size: 0.875rem;">❤️ 点赞</span>
                    <span style="color: #6b7280; font-size: 0.875rem;">💬 评论</span>
                    <span style="color: #6b7280; font-size: 0.875rem;">⭐ 收藏</span>
                </div>
                <span style="color: #6b7280; font-size: 0.875rem;">🔗 分享</span>
            </div>
        </div>
        
        <div style="display: flex; gap: 1rem; justify-content: center;">
            <button onclick="this.closest('.modal').remove()" style="padding: 0.75rem 1.5rem; background: #f3f4f6; color: #374151; border: none; border-radius: 8px; cursor: pointer;">关闭预览</button>
            <button onclick="this.closest('.modal').remove(); publishToXiaohongshu();" style="padding: 0.75rem 1.5rem; background: linear-gradient(135deg, #ec4899, #be185d); color: white; border: none; border-radius: 8px; cursor: pointer;">确认发布</button>
        </div>
    `;
    
    modal.className = 'modal';
    modal.appendChild(previewContent);
    
    // 点击背景关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    document.body.appendChild(modal);
}

// Pro功能：处理照片上传
async function handleProPhotoUpload(files) {
    if (files.length > 20) {
        showToast('最多只能选择20张照片', 'error');
        return;
    }
    
    showToast('正在上传照片...', 'info');
    
    try {
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        const response = await fetch('/upload/photos', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showToast(`成功上传${result.files.length}张照片`, 'success');
            
            // 显示上传的照片
            const uploadArea = document.getElementById('photo-upload-area');
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'upload-results';
            resultsDiv.innerHTML = `
                <h4>已上传的照片 (${result.files.length}张)</h4>
                <div class="uploaded-photos">
                    ${result.files.map((file, index) => `
                        <div class="uploaded-photo">
                            <img src="${file.url}" alt="Photo ${index + 1}" style="width: 80px; height: 80px; object-fit: cover; border-radius: 4px;">
                            <p>${escapeHTML(file.original_name)}</p>
                        </div>
                    `).join('')}
                </div>
                <button class="btn-primary" onclick="processPhotoRanking()">
                    <i class="fas fa-magic"></i> 开始智能排序
                </button>
            `;
            
            uploadArea.appendChild(resultsDiv);
            
            // 存储文件信息用于后续处理
            window.uploadedPhotos = result.files;
            
        } else {
            showToast('上传失败: ' + result.message, 'error');
        }
        
    } catch (error) {
        showToast('上传失败: ' + error.message, 'error');
    }
}

// 通用文件上传处理
async function handleGeneralFileUpload(files) {
    showToast('通用文件上传功能已启用', 'success');
    // 这里可以根据不同的上下文处理不同类型的文件
}

// 新的智能内容生成函数
async function generateSmartContent() {
    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    const theme = document.getElementById('xhs-theme')?.value?.trim();
    const contentType = document.getElementById('xhs-content-type')?.value || 'travel';
    const highlights = document.getElementById('content-highlights')?.value?.trim() || '';
    const feeling = document.getElementById('content-feeling')?.value?.trim() || '';
    const style = document.getElementById('content-style')?.value || '活泼';
    const length = document.getElementById('content-length')?.value || 'medium';

    if (!theme) {
        showToast('请先设置内容主题', 'error');
        return;
    }

    showToast('正在生成本地个性化模板...', 'info');

    try {
        // 根据用户输入生成个性化内容
        const photoCount = xhsSelectedPhotos.length;
        let contentStyle = '';
        let defaultHighlights = [];
        
        // 根据内容类型设置默认值
        switch(contentType) {
            case 'travel':
                contentStyle = '旅行分享';
                defaultHighlights = ['绝美风景', '特色体验', '美食推荐', '拍照攻略'];
                break;
            case 'food':
                contentStyle = '美食探店';
                defaultHighlights = ['招牌菜品', '环境氛围', '性价比', '服务体验'];
                break;
            case 'lifestyle':
                contentStyle = '生活记录';
                defaultHighlights = ['日常美好', '心情分享', '生活感悟', '小确幸'];
                break;
            case 'fashion':
                contentStyle = '穿搭分享';
                defaultHighlights = ['搭配技巧', '单品推荐', '风格解析', '场合适配'];
                break;
        }

        // 解析用户输入的亮点
        const userHighlights = highlights ? highlights.split(/[,，、]/).map(h => h.trim()).filter(h => h) : [];
        const finalHighlights = userHighlights.length > 0 ? userHighlights : defaultHighlights;
        
        // 根据风格调整语气
        let stylePrefix = '';
        let styleEmojis = '';
        let styleTone = '';
        
        switch(style) {
            case '活泼':
                stylePrefix = '🌟';
                styleEmojis = '✨💫🎉';
                styleTone = '超级棒的';
                break;
            case '专业':
                stylePrefix = '📍';
                styleEmojis = '⭐📝💡';
                styleTone = '非常专业的';
                break;
            case '简约':
                stylePrefix = '·';
                styleEmojis = '🔸🔹💫';
                styleTone = '简单而美好的';
                break;
            case '情感':
                stylePrefix = '💕';
                styleEmojis = '❤️💝🥰';
                styleTone = '充满回忆的';
                break;
        }

        // 根据长度调整内容
        let content = '';
        const userFeeling = feeling || `${styleTone}体验`;
        
        if (length === 'short') {
            content = `${stylePrefix} ${theme} | ${contentStyle}

📸 分享${photoCount}张精选照片
${finalHighlights.slice(0, 2).map((item, index) => `${index + 1}️⃣ ${item}`).join('\n')}

${userFeeling}，真心推荐！

#${theme} #${contentStyle} #推荐`;
        } else if (length === 'long') {
            content = `${stylePrefix} ${theme} | ${contentStyle}详细攻略

📸 这次精心挑选了${photoCount}张照片分享给大家
✨ 每一张都记录着美好的瞬间

💫 ${contentStyle}亮点详解：
${finalHighlights.map((item, index) => `${index + 1}️⃣ ${item} - 真的超出期待`).join('\n')}

🎯 个人深度体验：
${userFeeling}！从${finalHighlights[0] || '第一眼'}到${finalHighlights[1] || '离开时'}，每个细节都让人印象深刻。特别是${finalHighlights[2] || '整体感受'}，简直超出预期！

📝 实用攻略tips：
• 最佳时间：建议提前了解相关信息
• 必备物品：相机📷记录美好瞬间
• 预算参考：个人觉得性价比很高
• 温馨提醒：记得提前做好准备工作
• 个人推荐：五星好评⭐⭐⭐⭐⭐

💭 后记感想：
真的很开心能有这样的体验，每一个瞬间都值得珍藏。希望我的分享能给大家带来帮助，也期待听到大家的想法和建议！

❤️ 真心推荐给所有朋友，相信你们也会喜欢的！
有任何问题都可以在评论区交流哦～

#${theme} #${contentStyle} #详细攻略 #推荐 #种草 #值得 #分享 #生活记录 #美好时光`;
        } else { // medium
            content = `${stylePrefix} ${theme} | ${contentStyle}来啦！

📸 这次分享${photoCount}张精选照片
✨ 每一张都是满满的回忆

💫 ${contentStyle}亮点：
${finalHighlights.map((item, index) => `${index + 1}️⃣ ${item}`).join('\n')}

🎯 个人感受：
${userFeeling}！从${finalHighlights[0] || '开始'}到${finalHighlights[1] || '结束'}，每个细节都让人印象深刻。特别是${finalHighlights[2] || '整体体验'}，简直超出预期！

📝 实用tips：
• 最佳时间：建议提前了解
• 必备物品：相机📷不能少
• 预算参考：性价比很高
• 个人推荐：五星好评⭐⭐⭐⭐⭐

❤️ 真心推荐给大家，绝对不会失望！
有问题欢迎评论区交流哦～

#${theme} #${contentStyle} #推荐 #攻略 #种草 #值得 #分享 #生活记录`;
        }

        const resultContent = document.getElementById('xhs-result-content');
        if (resultContent) {
            resultContent.value = content;
            updateContentStats();
        }

        showToast('本地模板文案生成完成（未调用 AI）', 'success');

    } catch (error) {
        showToast('文案生成失败: ' + error.message, 'error');
    }
}

// 更新内容统计
function updateContentStats() {
    const textarea = document.getElementById('xhs-result-content');
    const wordCountEl = document.getElementById('content-word-count');
    const lineCountEl = document.getElementById('content-line-count');
    
    if (textarea && wordCountEl && lineCountEl) {
        const content = textarea.value;
        const wordCount = content.length;
        const lineCount = content.split('\n').length;
        
        wordCountEl.textContent = `字数: ${wordCount}`;
        lineCountEl.textContent = `行数: ${lineCount}`;
    }
}

// 保存内容模板
function saveContentTemplate() {
    const content = document.getElementById('xhs-result-content')?.value;
    const theme = document.getElementById('xhs-theme')?.value;
    
    if (!content || !theme) {
        showToast('请先生成文案内容', 'error');
        return;
    }
    
    // 保存到本地存储
    const template = {
        theme: theme,
        content: content,
        timestamp: Date.now(),
        id: `template_${Date.now()}`
    };
    
    const savedTemplates = JSON.parse(localStorage.getItem('xhs_templates') || '[]');
    savedTemplates.unshift(template);
    
    // 最多保存10个模板
    if (savedTemplates.length > 10) {
        savedTemplates.splice(10);
    }
    
    localStorage.setItem('xhs_templates', JSON.stringify(savedTemplates));
    showToast('文案模板已保存！', 'success');
}

function resetXHSWorkflow() {
    // Reset form
    document.getElementById('xhs-theme').value = '';
    document.getElementById('xhs-result-content').value = '';
    
    // Reset customization inputs
    document.getElementById('content-highlights').value = '';
    document.getElementById('content-feeling').value = '';
    document.getElementById('content-style').value = '活泼';
    document.getElementById('content-length').value = 'medium';
    
    // Reset photos
    clearXHSPhotos();
    
    // Clear results
    const collageContainer = document.getElementById('xhs-collage-container');
    const hashtagsContainer = document.getElementById('xhs-hashtags-container');
    if (collageContainer) collageContainer.innerHTML = '';
    if (hashtagsContainer) hashtagsContainer.innerHTML = '';
    
    // Go to step 1
    goToXHSStep(1);
    
    showToast('已重置，可以开始新项目', 'info');
}

// Utility function for copying content
function copyToClipboard(elementId, text = null) {
    let textToCopy = text;
    
    if (!textToCopy && elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            textToCopy = element.textContent || element.innerText;
        }
    }
    
    if (textToCopy) {
        navigator.clipboard.writeText(textToCopy).then(() => {
            showToast('内容已复制到剪贴板', 'success');
        }).catch(() => {
            showToast('复制失败，请手动复制', 'error');
        });
    }
}

// Initialize all features when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeUploadArea();
    initializeProFeatures();
    initializeXiaohongshuFeatures();
    updateNavigationFromURL();
    
    // 添加文案编辑区域的实时统计
    const textarea = document.getElementById('xhs-result-content');
    if (textarea) {
        textarea.addEventListener('input', updateContentStats);
    }
});

// Global function for dashboard navigation
window.navigateToPage = navigateToPage;
window.copyToClipboard = copyToClipboard;
window.clearXHSPhotos = clearXHSPhotos;
window.regenerateCollage = regenerateCollage;
window.downloadCollage = downloadCollage;
window.regenerateContent = regenerateContent;
window.generateSmartCover = generateSmartCover;
window.decorateImages = decorateImages;
window.downloadCover = downloadCover;
window.publishToXiaohongshu = publishToXiaohongshu;
window.previewBeforePublish = previewBeforePublish;
window.generateSmartContent = generateSmartContent;
window.saveContentTemplate = saveContentTemplate;
window.updateContentStats = updateContentStats;

// History Management System
let historyData = [];
let currentPage = 1;
let itemsPerPage = 10;
let filteredHistory = [];

// 历史记录管理功能
function initializeHistoryPage() {
    // 从localStorage加载历史记录
    loadHistoryFromStorage();
    
    // 设置事件监听器
    const filterType = document.getElementById('history-filter-type');
    const filterTime = document.getElementById('history-filter-time');
    const searchInput = document.getElementById('history-search');
    
    if (filterType) filterType.addEventListener('change', filterHistory);
    if (filterTime) filterTime.addEventListener('change', filterHistory);
    if (searchInput) searchInput.addEventListener('input', debounce(filterHistory, 300));
    
    // 初始化显示
    filterHistory();
}

function loadHistoryFromStorage() {
    const stored = localStorage.getItem('aiVideo_history');
    if (stored) {
        try {
            historyData = JSON.parse(stored);
        } catch (error) {
            console.error('加载历史记录失败:', error);
            historyData = [];
        }
    } else {
        // 创建一些示例数据
        historyData = [
            {
                id: 'demo_1',
                type: 'video',
                title: '智能视频剪辑',
                description: '处理了一个15分钟的YouTube视频，生成3个精彩片段',
                timestamp: Date.now() - 3600000, // 1小时前
                status: 'success',
                result: {
                    segments: 3,
                    totalDuration: '2分30秒',
                    outputPath: '/output/demo_video.mp4'
                }
            },
            {
                id: 'demo_2',
                type: 'xiaohongshu',
                title: '小红书内容生成',
                description: '基于8张旅行照片生成完整的小红书文案和拼图',
                timestamp: Date.now() - 7200000, // 2小时前
                status: 'success',
                result: {
                    photos: 8,
                    contentLength: '256字',
                    hashtags: 12
                }
            },
            {
                id: 'demo_3',
                type: 'pro',
                title: 'Pro功能照片排序',
                description: '使用CLIP模型对20张照片进行智能排序和筛选',
                timestamp: Date.now() - 86400000, // 1天前
                status: 'success',
                result: {
                    originalCount: 20,
                    rankedCount: 15,
                    method: 'CLIP + 美学评分'
                }
            }
        ];
        saveHistoryToStorage();
    }
}

function saveHistoryToStorage() {
    try {
        localStorage.setItem('aiVideo_history', JSON.stringify(historyData));
    } catch (error) {
        console.error('保存历史记录失败:', error);
    }
}

function addHistoryItem(type, title, description, result = null) {
    const newItem = {
        id: `history_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: type,
        title: title,
        description: description,
        timestamp: Date.now(),
        status: 'success',
        result: result
    };
    
    historyData.unshift(newItem); // 添加到开头
    
    // 限制历史记录数量（最多保留100条）
    if (historyData.length > 100) {
        historyData = historyData.slice(0, 100);
    }
    
    saveHistoryToStorage();
    
    // 如果当前在历史页面，刷新显示
    if (document.getElementById('history')?.classList.contains('active')) {
        filterHistory();
    }
}

function filterHistory() {
    const typeFilter = document.getElementById('history-filter-type')?.value || 'all';
    const timeFilter = document.getElementById('history-filter-time')?.value || 'all';
    const searchQuery = document.getElementById('history-search')?.value?.toLowerCase() || '';
    
    let filtered = [...historyData];
    
    // 类型过滤
    if (typeFilter !== 'all') {
        filtered = filtered.filter(item => item.type === typeFilter);
    }
    
    // 时间过滤
    if (timeFilter !== 'all') {
        const now = Date.now();
        const timeRanges = {
            'today': 24 * 60 * 60 * 1000,
            'week': 7 * 24 * 60 * 60 * 1000,
            'month': 30 * 24 * 60 * 60 * 1000
        };
        
        const range = timeRanges[timeFilter];
        if (range) {
            filtered = filtered.filter(item => (now - item.timestamp) <= range);
        }
    }
    
    // 搜索过滤
    if (searchQuery) {
        filtered = filtered.filter(item => 
            item.title.toLowerCase().includes(searchQuery) ||
            item.description.toLowerCase().includes(searchQuery)
        );
    }
    
    filteredHistory = filtered;
    currentPage = 1;
    displayHistory();
}

function displayHistory() {
    const historyList = document.getElementById('history-list');
    const historyLoading = document.getElementById('history-loading');
    const historyPagination = document.getElementById('history-pagination');
    
    if (!historyList) return;
    
    // 隐藏加载状态
    if (historyLoading) {
        historyLoading.style.display = 'none';
    }
    
    if (filteredHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-history">
                <i class="fas fa-inbox"></i>
                <h3>暂无历史记录</h3>
                <p>开始使用AI功能，您的操作记录将在这里显示</p>
            </div>
        `;
        if (historyPagination) {
            historyPagination.style.display = 'none';
        }
        return;
    }
    
    // 分页处理
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const pageItems = filteredHistory.slice(startIndex, endIndex);
    
    // 生成历史记录列表
    historyList.innerHTML = pageItems.map(item => `
        <div class="history-item">
            <div class="history-icon ${item.type}">
                ${getHistoryIcon(item.type)}
            </div>
            <div class="history-content">
                <div class="history-title">${escapeHTML(item.title)}</div>
                <div class="history-desc">${escapeHTML(item.description)}</div>
            </div>
            <div class="history-meta">
                <div class="history-time">${formatTime(item.timestamp)}</div>
                <div class="history-status ${item.status}">${getStatusText(item.status)}</div>
            </div>
            <div class="history-actions">
                <button onclick="viewHistoryItem('${item.id}')" title="查看详情">
                    <i class="fas fa-eye"></i>
                </button>
                <button onclick="downloadHistoryItem('${item.id}')" title="下载结果">
                    <i class="fas fa-download"></i>
                </button>
                <button onclick="deleteHistoryItem('${item.id}')" title="删除记录">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
    
    // 更新分页
    updatePagination();
}

function getHistoryIcon(type) {
    const icons = {
        'video': '<i class="fas fa-video"></i>',
        'xiaohongshu': '<i class="fas fa-heart"></i>',
        'pro': '<i class="fas fa-star"></i>'
    };
    return icons[type] || '<i class="fas fa-cog"></i>';
}

function getStatusText(status) {
    const statusTexts = {
        'success': '成功',
        'error': '失败',
        'processing': '处理中'
    };
    return statusTexts[status] || '未知';
}

function formatTime(timestamp) {
    const now = Date.now();
    const diff = now - timestamp;
    
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}天前`;
    
    return new Date(timestamp).toLocaleDateString();
}

function updatePagination() {
    const historyPagination = document.getElementById('history-pagination');
    const pageInfo = document.getElementById('page-info');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    
    if (!historyPagination || filteredHistory.length <= itemsPerPage) {
        historyPagination.style.display = 'none';
        return;
    }
    
    const totalPages = Math.ceil(filteredHistory.length / itemsPerPage);
    
    historyPagination.style.display = 'flex';
    
    if (pageInfo) {
        pageInfo.textContent = `第 ${currentPage} 页，共 ${totalPages} 页`;
    }
    
    if (prevBtn) {
        prevBtn.disabled = currentPage <= 1;
    }
    
    if (nextBtn) {
        nextBtn.disabled = currentPage >= totalPages;
    }
}

function searchHistory() {
    filterHistory();
}

function refreshHistory() {
    showToast('正在刷新历史记录...', 'info');
    setTimeout(() => {
        filterHistory();
        showToast('历史记录已刷新', 'success');
    }, 500);
}

function clearAllHistory() {
    if (confirm('确定要清空所有历史记录吗？此操作不可撤销。')) {
        historyData = [];
        saveHistoryToStorage();
        filterHistory();
        showToast('历史记录已清空', 'success');
    }
}

function changePage(direction) {
    const totalPages = Math.ceil(filteredHistory.length / itemsPerPage);
    const newPage = currentPage + direction;
    
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        displayHistory();
    }
}

function viewHistoryItem(itemId) {
    const item = historyData.find(h => h.id === itemId);
    if (!item) return;
    
    // 创建详情模态框
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 600px;">
            <div class="modal-header">
                <h3><i class="fas fa-info-circle"></i> 历史记录详情</h3>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body">
                <div style="margin-bottom: 1rem;">
                    <strong>标题：</strong>${escapeHTML(item.title)}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>描述：</strong>${escapeHTML(item.description)}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>时间：</strong>${new Date(item.timestamp).toLocaleString()}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>状态：</strong><span class="history-status ${item.status}">${getStatusText(item.status)}</span>
                </div>
                ${item.result ? `
                    <div style="margin-bottom: 1rem;">
                        <strong>处理结果：</strong>
                        <pre style="background: #f3f4f6; padding: 1rem; border-radius: 6px; margin-top: 0.5rem;">${JSON.stringify(item.result, null, 2)}</pre>
                    </div>
                ` : ''}
            </div>
            <div class="modal-footer">
                <button class="btn-secondary" onclick="this.closest('.modal-overlay').remove()">关闭</button>
            </div>
        </div>
    `;
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal || e.target.classList.contains('modal-close')) {
            modal.remove();
        }
    });
    
    document.body.appendChild(modal);
}

function deleteHistoryItem(itemId) {
    if (confirm('确定要删除这条历史记录吗？')) {
        historyData = historyData.filter(item => item.id !== itemId);
        saveHistoryToStorage();
        filterHistory();
        showToast('历史记录已删除', 'success');
    }
}

function downloadHistoryItem(itemId) {
    const item = historyData.find(h => h.id === itemId);
    if (!item) return;
    
    // 创建下载数据
    const downloadData = {
        title: item.title,
        description: item.description,
        timestamp: new Date(item.timestamp).toISOString(),
        status: item.status,
        result: item.result
    };
    
    // 创建并下载JSON文件
    const blob = new Blob([JSON.stringify(downloadData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `历史记录_${item.title}_${new Date(item.timestamp).toISOString().slice(0, 10)}.json`;
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    showToast('历史记录已导出', 'success');
}

// 防抖函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// History page functions
window.searchHistory = searchHistory;
window.refreshHistory = refreshHistory;
window.clearAllHistory = clearAllHistory;
window.changePage = changePage;
window.viewHistoryItem = viewHistoryItem;
window.deleteHistoryItem = deleteHistoryItem;
window.downloadHistoryItem = downloadHistoryItem;
