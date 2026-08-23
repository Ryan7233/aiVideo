/**
 * LLM集成模块 - 真实大模型API调用
 */

// 新的智能内容生成函数 - 使用真实LLM API
async function generateSmartContentLLM() {
    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }

    const theme = document.getElementById('xhs-theme')?.value?.trim();
    if (!theme) {
        showToast('请输入主题', 'error');
        return;
    }

    const highlights = document.getElementById('content-highlights')?.value?.trim() || '';
    const feeling = document.getElementById('content-feeling')?.value?.trim() || '';
    const style = document.getElementById('content-style')?.value || '活泼';
    const length = document.getElementById('content-length')?.value || 'medium';
    
    showToast('🤖 AI正在分析照片并生成个性化内容...', 'info');
    
    try {
        // 首先分析照片内容（模拟图片描述）
        const photoDescriptions = xhsSelectedPhotos.map((photo, index) => {
            // 这里可以集成图像识别API，现在使用基础描述
            return `照片${index + 1}: ${photo.name || '精美图片'}`;
        });
        
        // 调用真实的LLM API
        const response = await fetch('/llm/generate_content', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                theme: theme,
                photo_descriptions: photoDescriptions,
                highlights: highlights,
                feeling: feeling,
                style: style,
                length: length,
                custom_requirements: '',
                content_type: 'xiaohongshu'
            })
        });
        
        if (!response.ok) {
            throw new Error(`API调用失败: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success' && result.data) {
            // 使用LLM生成的内容
            const generatedData = result.data;
            
            const resultContent = document.getElementById('xhs-result-content');
            if (resultContent) {
                // 组合完整内容
                let fullContent = `${generatedData.title}\n\n${generatedData.content}`;
                
                if (generatedData.engagement) {
                    fullContent += `\n\n${generatedData.engagement}`;
                }
                
                resultContent.value = fullContent;
                updateContentStats();
            }
            
            // 显示生成的标签
            if (generatedData.hashtags && generatedData.hashtags.length > 0) {
                displayGeneratedHashtags(generatedData.hashtags);
            }
            
            // 显示API状态
            if (result.status === 'fallback') {
                showToast('✅ 内容生成完成（使用备用模式）', 'success');
            } else {
                showToast('🎉 AI内容生成完成！', 'success');
            }
            
        } else {
            throw new Error(result.error || '内容生成失败');
        }
        
    } catch (error) {
        console.error('LLM内容生成错误:', error);
        showToast('⚠️ AI服务暂时不可用，使用备用生成器', 'warning');
        
        // 降级到本地生成
        await generateFallbackContentLLM(theme, highlights, feeling, style, length);
    }
}

// 备用内容生成（当LLM API不可用时）
async function generateFallbackContentLLM(theme, highlights, feeling, style, length) {
    const photoCount = xhsSelectedPhotos.length;
    
    const lengthMap = {
        'short': 100,
        'medium': 200,
        'long': 300
    };
    
    const targetLength = lengthMap[length] || 200;
    
    let generatedText = '';
    
    if (style === '活泼') {
        generatedText = `✨ ${theme} ✨

今天带着满满的期待来体验${theme}！${photoCount}张照片记录下了这次美好的时光～

${highlights ? `特别想说的是：${highlights} 💕` : ''}
${feeling ? `我的感受：${feeling} 🌈` : ''}

每一个瞬间都值得珍藏，每一张照片都有它独特的故事。这次的体验真的超出了我的期待！

推荐给所有想要感受美好的朋友们～

#${theme} #美好时光 #值得推荐 #生活记录 #分享快乐`;

    } else if (style === '专业') {
        generatedText = `${theme} - 详细体验报告

本次${theme}体验包含${photoCount}个重要场景的详细记录。

核心亮点：
${highlights ? `• ${highlights}` : '• 整体体验优秀'}
${feeling ? `• 个人评价：${feeling}` : '• 性价比突出'}

通过深度体验和多角度分析，可以确认这是一次高质量的选择。建议有类似需求的朋友可以考虑。

详细信息请参考图片内容。

#${theme} #专业评测 #深度体验 #推荐指数五星`;

    } else if (style === '简约') {
        generatedText = `${theme}

${photoCount}张图
${highlights || '很棒的体验'}
${feeling || '推荐'} ⭐⭐⭐⭐⭐

#${theme} #简约分享`;

    } else if (style === '情感') {
        generatedText = `💝 ${theme} - 温暖的回忆

这${photoCount}张照片，每一张都承载着特别的情感...

${highlights ? `让我印象最深的是：${highlights}。那种感动，真的很难用言语表达。` : ''}
${feeling ? `${feeling}，这种感觉会一直伴随着我。` : ''}

希望通过这些照片，能把这份美好和温暖传递给更多的人。生活中的每一个瞬间，都值得被珍惜。

愿你们也能感受到这份简单而纯真的快乐 ❤️

#${theme} #温暖时光 #情感共鸣 #美好回忆 #传递正能量`;
    }
    
    // 调整长度
    if (generatedText.length > targetLength * 1.2) {
        generatedText = generatedText.substring(0, targetLength) + '...';
    }
    
    const resultContent = document.getElementById('xhs-result-content');
    if (resultContent) {
        resultContent.value = generatedText;
        updateContentStats();
    }
    
    // 生成相关标签
    generateHashtags(theme, 'xiaohongshu');
}

// 显示生成的标签
function displayGeneratedHashtags(hashtags) {
    const hashtagsContainer = document.getElementById('xhs-hashtags-container');
    if (!hashtagsContainer || !hashtags || hashtags.length === 0) return;
    
    hashtagsContainer.innerHTML = '';
    
    hashtags.forEach(tag => {
        const hashtagElement = document.createElement('span');
        hashtagElement.className = 'hashtag-item';
        hashtagElement.textContent = tag.startsWith('#') ? tag : `#${tag}`;
        hashtagsContainer.appendChild(hashtagElement);
    });
}

// 高级拼图生成功能
async function generateAdvancedCollage() {
    if (xhsSelectedPhotos.length === 0) {
        showToast('请先上传照片', 'error');
        return;
    }
    
    const theme = document.getElementById('xhs-theme')?.value?.trim() || '我的拼图';
    const rawLayout = document.getElementById('cover-layout')?.value || 'dynamic';
    // 兼容UI中的旧选项，映射到后端支持的布局
    const layoutMap = {
        'grid_3x3': 'grid',
        'grid_2x2': 'grid',
        'collage_mixed': 'dynamic',
        'magazine': 'magazine',
        'polaroid': 'creative',
        'treemap': 'treemap',
        'dynamic': 'dynamic',
        'grid': 'grid',
        'mosaic': 'mosaic',
        'creative': 'creative'
    };
    const layout = layoutMap[rawLayout] || 'dynamic';
    const colorTheme = document.getElementById('cover-theme')?.value || 'pink_gradient';
    const coverExtraText = document.getElementById('cover-overlay-text')?.value || '';
    // 将UI颜色主题映射到后端支持的 style 与 color_scheme
    const styleMap = {
        'pink_gradient': 'modern',
        'blue_gradient': 'modern',
        'warm_sunset': 'vintage',
        'cool_mint': 'minimal',
        'elegant_gray': 'minimal'
    };
    const schemeMap = {
        'pink_gradient': 'vibrant',
        'blue_gradient': 'cool',
        'warm_sunset': 'warm',
        'cool_mint': 'cool',
        'elegant_gray': 'monochrome'
    };
    const mappedStyle = styleMap[colorTheme] || 'modern';
    const mappedScheme = schemeMap[colorTheme] || 'auto';
    
    showToast('🎨 正在生成高级拼图效果...', 'info');
    
    try {
        // 上传照片并获取路径
        const formData = new FormData();
        xhsSelectedPhotos.forEach((photo, index) => {
            formData.append('files', photo);
        });
        
        const uploadResponse = await fetch('/upload/photos', {
            method: 'POST',
            body: formData
        });
        
        if (!uploadResponse.ok) {
            throw new Error('照片上传失败');
        }
        
        const uploadResult = await uploadResponse.json();
        let imagePaths = uploadResult.files.map(file => file.saved_path);
        // 若选择固定网格，限制图片数量（2x2=4，3x3=9），以确保布局明显
        if (rawLayout === 'grid_2x2') imagePaths = imagePaths.slice(0, 4);
        if (rawLayout === 'grid_3x3') imagePaths = imagePaths.slice(0, 9);
        
        // 调用高级拼图生成API
        const collageResponse = await fetch('/collage/generate_advanced', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                images: imagePaths,
                title: theme,
                layout_type: layout, // dynamic, grid, magazine, mosaic, creative, treemap
                style: mappedStyle,  // modern, vintage, artistic, minimal
                color_scheme: mappedScheme, // auto/warm/cool/monochrome/vibrant
                canvas_size: [800, 800],
                add_effects: true,
                add_text_overlay: true,
                extra_text: coverExtraText,
                text_position: 'center'
            })
        });
        
        if (!collageResponse.ok) {
            throw new Error('拼图生成失败');
        }
        
        const collageResult = await collageResponse.json();
        
        if (collageResult.status === 'success') {
            // 显示生成的拼图
            const coverContainer = document.getElementById('xhs-cover-container');
            if (coverContainer) {
                coverContainer.innerHTML = `
                    <img src="${collageResult.collage_base64}" 
                         alt="高级拼图" 
                         style="max-width: 100%; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.15);"
                         data-canvas="${collageResult.collage_base64}">
                    <div class="collage-info">
                        <p>布局: ${collageResult.metadata.layout_type} | 风格: ${collageResult.metadata.style}</p>
                        <p>图片数量: ${collageResult.metadata.image_count} | 尺寸: ${collageResult.metadata.canvas_size.join('x')}</p>
                    </div>
                `;
            }
            
            showToast('🎉 高级拼图生成成功！', 'success');
            
        } else {
            throw new Error(collageResult.error || '拼图生成失败');
        }
        
    } catch (error) {
        console.error('高级拼图生成错误:', error);
        showToast('⚠️ 高级拼图服务暂时不可用，使用基础拼图', 'warning');
        
        // 降级到基础Canvas拼图
        // 回退时使用原始布局值，保障2x2/3x3等明确定义
        generateFallbackCollage(theme, rawLayout, colorTheme);
    }
}

// 基础Canvas拼图（降级方案）
function generateFallbackCollage(theme, layout, colorTheme) {
    const collageContainer = document.getElementById('xhs-cover-container');
    if (!collageContainer || xhsSelectedPhotos.length === 0) return;

    // 创建Canvas进行真正的拼图
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 400;
    canvas.height = 400;
    
    // 绘制渐变背景
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    
    if (colorTheme === 'pink_gradient') {
        gradient.addColorStop(0, '#ffecd2');
        gradient.addColorStop(1, '#fcb69f');
    } else if (colorTheme === 'blue_gradient') {
        gradient.addColorStop(0, '#a8edea');
        gradient.addColorStop(1, '#fed6e3');
    } else if (colorTheme === 'purple_gradient') {
        gradient.addColorStop(0, '#d299c2');
        gradient.addColorStop(1, '#fef9d7');
    } else {
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(1, '#f0f0f0');
    }
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 计算网格布局
    const maxPhotos = Math.min(9, xhsSelectedPhotos.length);
    let gridCols, gridRows, photoSize, startX, startY;
    
    if (layout === 'grid_2x2') {
        gridCols = 2; gridRows = 2; photoSize = 80;
    } else if (layout === 'grid_3x3') {
        gridCols = 3; gridRows = 3; photoSize = 60;
    } else if (layout === 'grid_2x3') {
        gridCols = 3; gridRows = 2; photoSize = 70;
    } else {
        gridCols = Math.ceil(Math.sqrt(maxPhotos));
        gridRows = Math.ceil(maxPhotos / gridCols);
        photoSize = Math.min(80, (canvas.width - 40) / gridCols);
    }
    
    startX = (canvas.width - (gridCols * (photoSize + 10) - 10)) / 2;
    startY = (canvas.height - (gridRows * (photoSize + 10) - 10)) / 2;
    
    // 异步加载并绘制图片
    const drawPhotos = async () => {
        for (let i = 0; i < maxPhotos; i++) {
            const row = Math.floor(i / gridCols);
            const col = i % gridCols;
            
            const x = startX + col * (photoSize + 10);
            const y = startY + row * (photoSize + 10);
            
            try {
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
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.roundRect(x, y, photoSize, photoSize, 8);
                ctx.stroke();
                
            } catch (error) {
                console.error('图片加载失败:', error);
                // 绘制占位符
                ctx.fillStyle = '#f0f0f0';
                ctx.beginPath();
                ctx.roundRect(x, y, photoSize, photoSize, 8);
                ctx.fill();
                
                ctx.fillStyle = '#999';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('📷', x + photoSize/2, y + photoSize/2 + 6);
            }
        }
        
        // 添加标题
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.font = 'bold 24px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(theme, canvas.width / 2, 40);
        
        // 添加装饰元素
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '20px Arial';
        ctx.fillText('✨', 50, 50);
        ctx.fillText('✨', canvas.width - 50, 50);
        
        // 添加照片数量标签
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
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
    };
    
    drawPhotos();
}

// 检查LLM服务状态
async function checkLLMStatus() {
    try {
        const response = await fetch('/llm/status');
        const status = await response.json();
        
        // 显示状态信息
        const statusElement = document.getElementById('llm-status');
        if (statusElement) {
            const statusClass = status.status === 'active' ? 'status-active' : 'status-fallback';
            statusElement.innerHTML = `
                <span class="${statusClass}">
                    ${status.status === 'active' ? '🟢' : '🟡'} 
                    ${status.message}
                </span>
            `;
        }
        
        return status;
        
    } catch (error) {
        console.error('获取LLM状态失败:', error);
        return { status: 'error', message: '服务状态未知' };
    }
}

// 初始化LLM功能
function initializeLLMFeatures() {
    // 检查服务状态
    checkLLMStatus();
    
    // 替换原有的生成函数
    if (typeof generateSmartContent !== 'undefined') {
        window.originalGenerateSmartContent = generateSmartContent;
        window.generateSmartContent = generateSmartContentLLM;
    }
    
    // 封面按钮不再被替换成拼图生成。这个替换原本是 generateSmartCover 传假路径
    // （temp_image_0.jpg）必然失败时的绕行方案；那个问题已经修好，而按钮写的是
    // "生成封面"，旁边就是"下载封面"，它应该真的去生成封面。
    // 高级拼贴现在有自己的按钮，走 /collage/generate_advanced。
    window.generateAdvancedCollage = generateAdvancedCollage;
    
    console.log('🤖 LLM功能已初始化');
}

// 页面加载时初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLLMFeatures);
} else {
    initializeLLMFeatures();
}
