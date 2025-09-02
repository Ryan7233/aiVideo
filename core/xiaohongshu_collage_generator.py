"""
小红书级别高质量拼图生成器
实现专业级图片拼接、文案编辑、高清导出功能
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
from pathlib import Path
import io
import base64
import math
import os
from dataclasses import dataclass


@dataclass
class CollageConfig:
    """拼图配置"""
    width: int = 2160  # 4K分辨率宽度
    height: int = 2160  # 4K分辨率高度
    quality: int = 95   # JPEG质量
    dpi: int = 300      # 打印质量DPI
    background_color: str = "#FFFFFF"
    border_width: int = 6
    corner_radius: int = 20
    shadow_offset: int = 8
    shadow_blur: int = 15
    text_margin: int = 60
    title_size: int = 120
    subtitle_size: int = 80
    watermark_opacity: int = 30
    # 新增：标题排版控制
    title_position: str = "top"  # top | center_overlay
    title_box: bool = True
    title_box_opacity: int = 160
    title_box_padding: int = 28
    title_max_width_ratio: float = 0.7
    # 字体自定义
    font_path_override: Optional[str] = None


@dataclass 
class TextConfig:
    """文案配置"""
    text: str
    font_size: int
    color: str
    position: Tuple[int, int]
    max_width: int
    font_weight: str = "normal"
    shadow: bool = True
    background: bool = False
    background_color: str = "#FFFFFF"
    background_opacity: int = 180


class XiaohongshuCollageGenerator:
    """小红书级别拼图生成器"""
    
    def __init__(self):
        self.output_dir = Path("output_data/xiaohongshu_collages")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 高质量字体路径
        self.font_paths = self._get_font_paths()
        
        # 专业布局模板
        self.layout_templates = {
            "magazine_style": self._magazine_layout,
            "grid_modern": self._grid_modern_layout,
            "story_flow": self._story_flow_layout,
            "featured_main": self._featured_main_layout,
            "artistic_collage": self._artistic_collage_layout,
            "minimal_clean": self._minimal_clean_layout,
            # 新增更具“拼贴感”的布局
            "scrapbook": self._scrapbook_layout
        }
        
        # 颜色方案
        self.color_schemes = {
            "xiaohongshu_pink": {
                "primary": "#FF2442",
                "secondary": "#FF6B9D", 
                "accent": "#FFB6C1",
                "background": "#FFFBFC",
                "text": "#2C2C2C",
                "shadow": "#E0E0E0"
            },
            "fresh_mint": {
                "primary": "#00C896",
                "secondary": "#7FDBCA",
                "accent": "#B8F2E6",
                "background": "#F8FFFD",
                "text": "#2D5A27",
                "shadow": "#D4E6F1"
            },
            "warm_sunset": {
                "primary": "#FF6B35",
                "secondary": "#F7931E", 
                "accent": "#FFD23F",
                "background": "#FFFAF5",
                "text": "#8B4513",
                "shadow": "#F0E68C"
            },
            "elegant_gray": {
                "primary": "#2C3E50",
                "secondary": "#34495E",
                "accent": "#95A5A6", 
                "background": "#F8F9FA",
                "text": "#2C3E50",
                "shadow": "#BDC3C7"
            }
        }
    
    def _get_font_paths(self) -> Dict[str, str]:
        """获取系统字体路径"""
        font_paths = {}
        
        # macOS 字体路径
        macos_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/PingFang.ttc", 
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Times.ttc"
        ]
        
        # 检查可用字体
        for font_path in macos_fonts:
            if os.path.exists(font_path):
                if "PingFang" in font_path:
                    font_paths["chinese"] = font_path
                elif "Helvetica" in font_path:
                    font_paths["english"] = font_path
                elif "Arial" in font_path:
                    font_paths["unicode"] = font_path
                elif "Times" in font_path:
                    font_paths["serif"] = font_path
        
        # 默认字体
        if not font_paths:
            font_paths["default"] = None
            
        return font_paths
    
    def _get_font(self, size: int, weight: str = "normal") -> ImageFont.FreeTypeFont:
        """获取字体对象"""
        try:
            # 用户自定义优先
            if getattr(self, 'font_paths', None) and isinstance(self.font_paths, dict):
                pass
            # 覆盖路径
            if hasattr(self, 'config_font_override') and self.config_font_override:
                try:
                    return ImageFont.truetype(self.config_font_override, size)
                except Exception:
                    pass
            # 优先使用中文字体
            if "chinese" in self.font_paths:
                path = self.font_paths["chinese"]
                if path.lower().endswith('.ttc'):
                    for idx in (0,1,2,3):
                        try:
                            return ImageFont.truetype(path, size, index=idx)
                        except Exception:
                            continue
                else:
                    return ImageFont.truetype(path, size)
            elif "unicode" in self.font_paths:
                return ImageFont.truetype(self.font_paths["unicode"], size)
            elif "english" in self.font_paths:
                return ImageFont.truetype(self.font_paths["english"], size)
            else:
                return ImageFont.load_default()
        except Exception as e:
            logger.warning(f"字体加载失败: {e}")
            return ImageFont.load_default()
    
    async def generate_xiaohongshu_collage(
        self,
        images: List[str],
        title: str,
        subtitle: str = "",
        layout: str = "magazine_style",
        color_scheme: str = "xiaohongshu_pink",
        custom_texts: List[TextConfig] = None,
        config: CollageConfig = None,
        overlay_texts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        生成小红书级别的高质量拼图
        
        Args:
            images: 图片路径列表
            title: 主标题
            subtitle: 副标题
            layout: 布局样式
            color_scheme: 颜色方案
            custom_texts: 自定义文案列表
            config: 拼图配置
            
        Returns:
            Dict: 生成结果包含图片路径和base64数据
        """
        try:
            if not config:
                config = CollageConfig()
            
            # 加载图片
            loaded_images = await self._load_and_process_images(images, config)
            if not loaded_images:
                raise ValueError("没有有效的图片")
            
            # 创建高质量画布
            canvas = Image.new('RGB', (config.width, config.height), config.background_color)
            
            # 获取颜色方案
            colors = self.color_schemes.get(color_scheme, self.color_schemes["xiaohongshu_pink"])
            
            # 应用布局
            layout_func = self.layout_templates.get(layout, self._magazine_layout)
            canvas = await layout_func(canvas, loaded_images, colors, config)
            
            # 添加主标题和副标题（支持居中覆盖）
            # 记录可选的覆盖字体路径
            self.config_font_override = config.font_path_override
            canvas = self._add_main_texts(canvas, title, subtitle, colors, config)
            
            # 添加自定义文案
            if custom_texts:
                for text_config in custom_texts:
                    canvas = self._add_custom_text(canvas, text_config)
            
            # 添加专业效果
            canvas = self._add_professional_effects(canvas, colors, config)
            
            # 叠加额外文案块（锚点定位）
            if overlay_texts:
                canvas = self._draw_text_blocks(canvas, overlay_texts, colors, config)

            # 保存高质量图片
            output_path, base64_data = await self._save_high_quality_image(canvas, config)
            
            logger.info(f"小红书级别拼图生成成功: {output_path}")
            
            return {
                "success": True,
                "image_path": str(output_path),
                "base64_data": base64_data,
                "width": config.width,
                "height": config.height,
                "file_size": output_path.stat().st_size if output_path.exists() else 0,
                "layout": layout,
                "color_scheme": color_scheme,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"小红书拼图生成失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": int(time.time())
            }

    def _draw_text_blocks(self, canvas: Image.Image, blocks: List[Dict[str, Any]], colors: Dict, config: CollageConfig) -> Image.Image:
        draw = ImageDraw.Draw(canvas)
        W, H = canvas.size
        for b in blocks:
            try:
                text = str(b.get('text', '') or '').strip()
                if not text:
                    continue
                font_size = int(b.get('font_size', 56))
                color = b.get('color', colors.get('text', '#2C2C2C'))
                anchor = b.get('anchor', 'center')  # top_left/top_center/center/bottom_center/...
                box = bool(b.get('box', True))
                max_ratio = float(b.get('max_width_ratio', 0.8))
                opacity = int(b.get('box_opacity', 150))
                pad = int(b.get('padding', 18))
                font = self._get_font(font_size, 'bold')
                
                max_w = int(W * max_ratio)
                # wrap
                def wrap(t: str) -> List[str]:
                    lines, line = [], ''
                    for ch in t:
                        test = line + ch
                        w = draw.textbbox((0,0), test, font=font)[2]
                        if w <= max_w:
                            line = test
                        else:
                            if line:
                                lines.append(line)
                            line = ch
                    if line:
                        lines.append(line)
                    return lines[:6]
                lines = wrap(text)
                if not lines:
                    continue
                line_h = draw.textbbox((0,0), '测', font=font)[3]
                total_h = len(lines) * line_h + (len(lines)-1)*8
                block_w = max(draw.textbbox((0,0), ln, font=font)[2] for ln in lines)
                bx = (W - block_w)//2
                by = {
                    'top_left': 40,
                    'top_center': 120,
                    'top_right': 40,
                    'center': (H-total_h)//2,
                    'bottom_center': H - total_h - 120,
                    'bottom_left': H - total_h - 120,
                    'bottom_right': H - total_h - 120
                }.get(anchor, (H-total_h)//2)
                # adjust x for left/right anchors
                if anchor.endswith('left'):
                    bx = 40
                elif anchor.endswith('right'):
                    bx = W - block_w - 40
                
                if box:
                    overlay = Image.new('RGBA', canvas.size, (0,0,0,0))
                    od = ImageDraw.Draw(overlay)
                    rect = [bx - pad, by - pad, bx + block_w + pad, by + total_h + pad]
                    od.rounded_rectangle(rect, radius=16, fill=(255,255,255,opacity))
                    canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(canvas)
                
                cy = by
                for ln in lines:
                    draw.text((bx, cy), ln, font=font, fill=color)
                    cy += line_h + 8
            except Exception:
                continue
        return canvas
    
    async def _load_and_process_images(self, image_paths: List[str], config: CollageConfig) -> List[Image.Image]:
        """加载和处理图片"""
        processed_images = []
        
        for path in image_paths:
            try:
                # 支持多种路径格式
                if path.startswith('output_data/') or path.startswith('/'):
                    full_path = Path(path)
                else:
                    full_path = Path("output_data") / path
                
                if not full_path.exists():
                    logger.warning(f"图片不存在: {full_path}")
                    continue
                
                # 加载图片
                img = Image.open(full_path)
                
                # 转换为RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 图片质量增强
                img = self._enhance_image_quality(img)
                
                processed_images.append(img)
                
            except Exception as e:
                logger.warning(f"图片加载失败: {path}, {str(e)}")
                continue
        
        return processed_images
    
    def _enhance_image_quality(self, img: Image.Image) -> Image.Image:
        """增强图片质量"""
        try:
            # 锐化
            sharpness_enhancer = ImageEnhance.Sharpness(img)
            img = sharpness_enhancer.enhance(1.2)
            
            # 对比度增强
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(1.1)
            
            # 色彩饱和度
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(1.1)
            
            return img
        except Exception as e:
            logger.warning(f"图片质量增强失败: {e}")
            return img
    
    async def _magazine_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """杂志风格布局"""
        if not images:
            return canvas
        
        canvas_width, canvas_height = canvas.size
        
        # 预留标题空间
        title_space = 300
        content_height = canvas_height - title_space
        
        if len(images) == 1:
            # 单图：大图展示
            img = images[0]
            target_width = int(canvas_width * 0.85)
            target_height = int(content_height * 0.8)
            
            img = self._resize_and_crop_smart(img, (target_width, target_height))
            img = self._add_rounded_corners(img, config.corner_radius)
            img = self._add_shadow(img, colors["shadow"], config.shadow_offset, config.shadow_blur)
            
            x = (canvas_width - target_width) // 2
            y = title_space + (content_height - target_height) // 2
            
            canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            
        elif len(images) == 2:
            # 双图：一大一小的不对称布局
            main_img, small_img = images[0], images[1]
            
            # 主图
            main_width = int(canvas_width * 0.65)
            main_height = int(content_height * 0.75)
            main_img = self._resize_and_crop_smart(main_img, (main_width, main_height))
            main_img = self._add_rounded_corners(main_img, config.corner_radius)
            main_img = self._add_shadow(main_img, colors["shadow"], config.shadow_offset, config.shadow_blur)
            
            # 小图
            small_width = int(canvas_width * 0.28)
            small_height = int(content_height * 0.4)
            small_img = self._resize_and_crop_smart(small_img, (small_width, small_height))
            small_img = self._add_rounded_corners(small_img, config.corner_radius // 2)
            small_img = self._add_shadow(small_img, colors["shadow"], config.shadow_offset // 2, config.shadow_blur // 2)
            
            # 布局
            main_x = 60
            main_y = title_space + 60
            small_x = canvas_width - small_width - 60
            small_y = title_space + content_height - small_height - 60
            
            canvas.paste(main_img, (main_x, main_y), main_img if main_img.mode == 'RGBA' else None)
            canvas.paste(small_img, (small_x, small_y), small_img if small_img.mode == 'RGBA' else None)
            
        else:
            # 多图：网格布局
            canvas = await self._grid_modern_layout(canvas, images, colors, config)
        
        return canvas
    
    async def _grid_modern_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """现代网格布局"""
        if not images:
            return canvas
        
        canvas_width, canvas_height = canvas.size
        title_space = 300
        content_height = canvas_height - title_space
        
        # 根据图片数量确定网格
        num_images = len(images)
        if num_images <= 4:
            grid_cols, grid_rows = 2, 2
        elif num_images <= 6:
            grid_cols, grid_rows = 3, 2
        elif num_images <= 9:
            grid_cols, grid_rows = 3, 3
        else:
            grid_cols, grid_rows = 4, 3
        
        # 计算单元格尺寸
        spacing = 30
        total_spacing_x = spacing * (grid_cols + 1)
        total_spacing_y = spacing * (grid_rows + 1)
        
        cell_width = (canvas_width - total_spacing_x) // grid_cols
        cell_height = (content_height - total_spacing_y) // grid_rows
        
        # 放置图片
        for i, img in enumerate(images[:grid_cols * grid_rows]):
            row = i // grid_cols
            col = i % grid_cols
            
            # 处理图片
            processed_img = self._resize_and_crop_smart(img, (cell_width, cell_height))
            processed_img = self._add_rounded_corners(processed_img, config.corner_radius)
            processed_img = self._add_shadow(processed_img, colors["shadow"], config.shadow_offset, config.shadow_blur)
            
            # 计算位置
            x = spacing + col * (cell_width + spacing)
            y = title_space + spacing + row * (cell_height + spacing)
            
            canvas.paste(processed_img, (x, y), processed_img if processed_img.mode == 'RGBA' else None)
        
        return canvas
    
    async def _story_flow_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """故事流布局"""
        # 实现故事流式的图片排列
        return await self._magazine_layout(canvas, images, colors, config)
    
    async def _featured_main_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """主图特色布局"""
        # 实现以主图为核心的布局
        return await self._magazine_layout(canvas, images, colors, config)
    
    async def _artistic_collage_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """艺术拼贴布局"""
        # 实现艺术风格的拼贴
        return await self._magazine_layout(canvas, images, colors, config)
    
    async def _minimal_clean_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """简约清新布局"""
        # 实现简约风格布局
        return await self._magazine_layout(canvas, images, colors, config)
    
    def _resize_and_crop_smart(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """智能缩放和裁剪"""
        target_width, target_height = target_size
        img_width, img_height = img.size
        
        # 计算缩放比例
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        scale = max(scale_x, scale_y)
        
        # 缩放图片
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 居中裁剪
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return img.crop((left, top, right, bottom))
    
    def _add_rounded_corners(self, img: Image.Image, radius: int) -> Image.Image:
        """添加圆角"""
        try:
            # 创建圆角蒙版
            mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius, fill=255)
            
            # 应用蒙版
            img = img.convert('RGBA')
            img.putalpha(mask)
            
            return img
        except Exception as e:
            logger.warning(f"圆角添加失败: {e}")
            return img

    def _add_shadow(self, img: Image.Image, shadow_color: str, offset: int, blur: int) -> Image.Image:
        """添加阴影效果"""
        try:
            # 创建阴影层
            shadow = Image.new('RGBA', (img.size[0] + offset * 2, img.size[1] + offset * 2), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            
            # 解析颜色
            if shadow_color.startswith('#'):
                shadow_rgb = tuple(int(shadow_color[i:i+2], 16) for i in (1, 3, 5))
            else:
                shadow_rgb = (200, 200, 200)
            
            # 绘制阴影
            shadow_draw.rounded_rectangle(
                [offset, offset, shadow.size[0] - offset, shadow.size[1] - offset],
                20, fill=shadow_rgb + (80,)
            )
            
            # 模糊阴影
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
            
            # 合成图片和阴影
            result = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
            result.paste(shadow, (0, 0))
            result.paste(img, (0, 0), img)
            
            return result
        except Exception as e:
            logger.warning(f"阴影添加失败: {e}")
            return img

    def _add_polaroid_border(self, img: Image.Image, border: int = 22, bottom_extra: int = 28) -> Image.Image:
        """为图片添加拍立得白边（底部加厚）。返回带透明通道的图像。"""
        try:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            new_w = img.width + border * 2
            new_h = img.height + border * 2 + bottom_extra
            canvas = Image.new('RGBA', (new_w, new_h), (255, 255, 255, 255))
            canvas.paste(img, (border, border), img)
            return canvas
        except Exception as e:
            logger.warning(f"拍立得边框添加失败: {e}")
            return img

    def _torn_edge_mask(self, size: Tuple[int, int], seed: int = 0, roughness: int = 12, blur: int = 3) -> Image.Image:
        """生成撕边Alpha遮罩。粗糙度越大边缘越不规则。"""
        try:
            import random
            random.seed(seed)
            w, h = size
            mask = Image.new('L', (w, h), 255)
            draw = ImageDraw.Draw(mask)
            # 在四边绘制不规则凹凸，模拟撕边
            def jitter_line(x0, y0, x1, y1, steps=60, amp=roughness):
                pts = []
                for i in range(steps + 1):
                    t = i / steps
                    x = int(x0 + (x1 - x0) * t)
                    y = int(y0 + (y1 - y0) * t)
                    if x0 == x1:  # vertical
                        x += int(random.uniform(-amp, amp))
                    else:         # horizontal
                        y += int(random.uniform(-amp, amp))
                    pts.append((x, y))
                return pts
            # 外轮廓路径（逆时针）
            top = jitter_line(0, 0, w, 0)
            right = jitter_line(w - 1, 0, w - 1, h)
            bottom = jitter_line(w, h - 1, 0, h - 1)
            left = jitter_line(0, h, 0, 0)
            poly = top + right + bottom + left
            mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(poly, fill=255)
            if blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(blur))
            return mask
        except Exception as e:
            logger.warning(f"撕边遮罩生成失败: {e}")
            return Image.new('L', size, 255)

    def _add_washi_tape(self, canvas: Image.Image, pos: Tuple[int, int], size: Tuple[int, int], angle: float = 0.0,
                         color: Tuple[int, int, int] = (245, 225, 170), alpha: int = 160, radius: int = 6) -> Image.Image:
        """在画布上添加一段半透明和纸胶带效果。返回合成后的画布。"""
        try:
            tape = Image.new('RGBA', size, (0, 0, 0, 0))
            tape_draw = ImageDraw.Draw(tape)
            tape_draw.rounded_rectangle([0, 0, size[0], size[1]], radius, fill=color + (alpha,))
            # 添加些许纹理条纹
            for x in range(0, size[0], 6):
                tape_draw.line([(x, 0), (x, size[1])], fill=(255, 255, 255, 18))
            if angle:
                tape = tape.rotate(angle, expand=True)
            out = canvas.convert('RGBA')
            out.paste(tape, pos, tape)
            return out
        except Exception as e:
            logger.warning(f"和纸胶带添加失败: {e}")
            return canvas

    async def _scrapbook_layout(self, canvas: Image.Image, images: List[Image.Image], colors: Dict, config: CollageConfig) -> Image.Image:
        """手帐拼贴风布局：随机小角度旋转、轻微重叠、拍立得白边、和纸胶带与接缝阴影。"""
        if not images:
            return canvas
        rng = np.random.default_rng(int(time.time()))
        W, H = canvas.size

        # 背景轻纹理 + 渐变
        canvas = self._add_gradient_overlay(canvas, colors)
        noise = Image.effect_noise((W, H), 6).convert('L')
        noise_colored = Image.merge('RGBA', (noise, noise, noise, Image.new('L', (W, H), 18)))
        canvas = Image.alpha_composite(canvas.convert('RGBA'), noise_colored).convert('RGB')

        # 预留标题区域
        title_space = 260
        area = (W, H - title_space)

        # 决定每张图片的目标尺寸（主次有别）
        n = len(images)
        base = min(W, H - title_space)
        sizes: List[Tuple[int, int]] = []
        for i in range(n):
            scale = 0.42 if i == 0 else rng.uniform(0.26, 0.34)
            w = int(base * scale)
            h = int(w * rng.uniform(0.9, 1.1))
            sizes.append((w, h))

        # 生成摆位网格的候选中心点
        cols = max(2, int(np.ceil(np.sqrt(n + 1))))
        rows = max(2, int(np.ceil((n + 1) / cols)))
        xs = np.linspace(int(W * 0.14), int(W * 0.86), cols)
        ys = np.linspace(title_space + int(area[1] * 0.1), title_space + int(area[1] * 0.9), rows)
        centers = [(int(x), int(y)) for y in ys for x in xs]
        rng.shuffle(centers)

        out = canvas.convert('RGBA')
        z = 0
        for i, img in enumerate(images[: len(centers)]):
            tw, th = sizes[i]
            proc = self._resize_and_crop_smart(img, (tw, th))
            # 可选撕边（小概率）
            if rng.random() < 0.35:
                mask = self._torn_edge_mask((proc.width, proc.height), seed=rng.integers(1, 1e9))
                proc = proc.convert('RGBA')
                proc.putalpha(mask)
            # 拍立得白边
            proc = self._add_polaroid_border(proc, border= rng.integers(16, 26), bottom_extra=rng.integers(20, 36))
            # 轻微旋转
            angle = float(rng.uniform(-6, 6))
            proc = proc.rotate(angle, expand=True)

            # 投影/接缝阴影
            shadow = Image.new('RGBA', (proc.width + 8, proc.height + 8), (0, 0, 0, 0))
            sd = ImageDraw.Draw(shadow)
            sd.rectangle([4, 4, shadow.width - 2, shadow.height - 2], fill=(0, 0, 0, 80))
            shadow = shadow.filter(ImageFilter.GaussianBlur(6))

            cx, cy = centers[i]
            x = int(cx - proc.width / 2 + rng.uniform(-20, 20))
            y = int(cy - proc.height / 2 + rng.uniform(-20, 20))

            # 先贴阴影再贴图
            out.paste(shadow, (x - 4, y - 4), shadow)
            out.paste(proc, (x, y), proc)

            # 随机胶带
            if rng.random() < 0.65:
                tape_len = int(proc.width * rng.uniform(0.35, 0.55))
                tape_w = rng.integers(26, 36)
                tx = x + rng.integers(10, max(12, proc.width - tape_len - 10))
                ty = y - rng.integers(8, 20)
                out = self._add_washi_tape(out, (tx, ty), (tape_len, int(tape_w)), angle=float(rng.uniform(-12, 12)))

            z += 1

        return out.convert('RGB')
    
    def _add_main_texts(self, canvas: Image.Image, title: str, subtitle: str, colors: Dict, config: CollageConfig) -> Image.Image:
        """添加主标题和副标题"""
        try:
            draw = ImageDraw.Draw(canvas)
            canvas_width, canvas_height = canvas.size
            
            def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
                if not text:
                    return []
                # 按字符宽度折行（适配中文无空格）
                lines = []
                line = ''
                for ch in text:
                    test = line + ch
                    w = draw.textbbox((0,0), test, font=font)[2]
                    if w <= max_width:
                        line = test
                    else:
                        if line:
                            lines.append(line)
                        line = ch
                if line:
                    lines.append(line)
                return lines

            if config.title_position == 'center_overlay' and title:
                # 居中覆盖：标题+可选副标题置于画面中央，带半透明底框
                title_font = self._get_font(config.title_size, "bold")
                subtitle_font = self._get_font(int(config.subtitle_size*0.9))
                max_width = int(canvas_width * config.title_max_width_ratio)
                title_lines = wrap_text(title, title_font, max_width)
                subtitle_lines = wrap_text(subtitle or '', subtitle_font, max_width)
                line_spacing = 10
                title_heights = sum(draw.textbbox((0,0), t, font=title_font)[3] - draw.textbbox((0,0), t, font=title_font)[1] for t in title_lines)
                subs_heights = sum(draw.textbbox((0,0), s, font=subtitle_font)[3] - draw.textbbox((0,0), s, font=subtitle_font)[1] for s in subtitle_lines)
                total_h = title_heights + (line_spacing if subtitle_lines and title_lines else 0) + subs_heights
                # 背景框
                if config.title_box:
                    pad = config.title_box_padding
                    # 计算最大行宽
                    line_widths = [draw.textbbox((0,0), t, font=title_font)[2] for t in title_lines] + [draw.textbbox((0,0), s, font=subtitle_font)[2] for s in subtitle_lines]
                    box_w = max(line_widths) + pad*2 if line_widths else max_width
                    box_h = total_h + pad*2
                    box_x = (canvas_width - box_w)//2
                    box_y = (canvas_height - box_h)//2
                    overlay = Image.new('RGBA', canvas.size, (0,0,0,0))
                    od = ImageDraw.Draw(overlay)
                    # 半透明白底
                    od.rounded_rectangle([box_x, box_y, box_x+box_w, box_y+box_h], radius=18, fill=(255,255,255,config.title_box_opacity))
                    canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(canvas)
                # 绘制文本
                cur_y = (canvas_height - total_h)//2
                for t in title_lines:
                    bbox = draw.textbbox((0,0), t, font=title_font)
                    tw = bbox[2]-bbox[0]
                    tx = (canvas_width - tw)//2
                    draw.text((tx+2, cur_y+2), t, fill=colors["shadow"], font=title_font)
                    draw.text((tx, cur_y), t, fill=colors["text"], font=title_font)
                    cur_y += bbox[3]-bbox[1]
                if subtitle_lines:
                    cur_y += line_spacing
                    for s in subtitle_lines:
                        bbox = draw.textbbox((0,0), s, font=subtitle_font)
                        sw = bbox[2]-bbox[0]
                        sx = (canvas_width - sw)//2
                        draw.text((sx+1, cur_y+1), s, fill=colors["shadow"], font=subtitle_font)
                        draw.text((sx, cur_y), s, fill=colors["primary"], font=subtitle_font)
                        cur_y += bbox[3]-bbox[1]
            else:
                # 顶部居中（原行为）
                if title:
                    title_font = self._get_font(config.title_size, "bold")
                    title_bbox = draw.textbbox((0, 0), title, font=title_font)
                    title_width = title_bbox[2] - title_bbox[0]
                    title_height = title_bbox[3] - title_bbox[1]
                    title_x = (canvas_width - title_width) // 2
                    title_y = 80
                    draw.text((title_x + 3, title_y + 3), title, fill=colors["shadow"], font=title_font)
                    draw.text((title_x, title_y), title, fill=colors["text"], font=title_font)
                if subtitle:
                    subtitle_font = self._get_font(config.subtitle_size)
                    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
                    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
                    subtitle_x = (canvas_width - subtitle_width) // 2
                    subtitle_y = (title_y + title_height + 30) if title else 160
                    draw.text((subtitle_x + 2, subtitle_y + 2), subtitle, fill=colors["shadow"], font=subtitle_font)
                    draw.text((subtitle_x, subtitle_y), subtitle, fill=colors["primary"], font=subtitle_font)
            
            return canvas
        except Exception as e:
            logger.warning(f"主文字添加失败: {e}")
            return canvas
    
    def _add_custom_text(self, canvas: Image.Image, text_config: TextConfig) -> Image.Image:
        """添加自定义文案"""
        try:
            draw = ImageDraw.Draw(canvas)
            font = self._get_font(text_config.font_size, text_config.font_weight)
            
            # 文字背景
            if text_config.background:
                text_bbox = draw.textbbox(text_config.position, text_config.text, font=font)
                bg_padding = 20
                bg_rect = [
                    text_bbox[0] - bg_padding,
                    text_bbox[1] - bg_padding, 
                    text_bbox[2] + bg_padding,
                    text_bbox[3] + bg_padding
                ]
                
                # 背景颜色
                bg_color = text_config.background_color
                if bg_color.startswith('#'):
                    bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                    bg_color = bg_rgb + (text_config.background_opacity,)
                
                draw.rounded_rectangle(bg_rect, 15, fill=bg_color)
            
            # 文字阴影
            if text_config.shadow:
                shadow_pos = (text_config.position[0] + 2, text_config.position[1] + 2)
                draw.text(shadow_pos, text_config.text, fill=(0, 0, 0, 100), font=font)
            
            # 主文字
            draw.text(text_config.position, text_config.text, fill=text_config.color, font=font)
            
            return canvas
        except Exception as e:
            logger.warning(f"自定义文字添加失败: {e}")
            return canvas
    
    def _add_professional_effects(self, canvas: Image.Image, colors: Dict, config: CollageConfig) -> Image.Image:
        """添加专业效果"""
        try:
            # 添加微妙的渐变背景
            canvas = self._add_gradient_overlay(canvas, colors)
            
            # 添加边框装饰
            canvas = self._add_decorative_border(canvas, colors, config)
            
            return canvas
        except Exception as e:
            logger.warning(f"专业效果添加失败: {e}")
            return canvas
    
    def _add_gradient_overlay(self, canvas: Image.Image, colors: Dict) -> Image.Image:
        """添加渐变叠加"""
        try:
            # 创建微妙的渐变叠加
            overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # 顶部到底部的微妙渐变
            for y in range(canvas.height):
                alpha = int(10 * (1 - y / canvas.height))  # 很微妙的渐变
                if alpha > 0:
                    draw.line([(0, y), (canvas.width, y)], fill=(255, 255, 255, alpha))
            
            # 合成
            canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay)
            return canvas.convert('RGB')
        except Exception as e:
            logger.warning(f"渐变叠加失败: {e}")
            return canvas
    
    def _add_decorative_border(self, canvas: Image.Image, colors: Dict, config: CollageConfig) -> Image.Image:
        """添加装饰边框"""
        try:
            draw = ImageDraw.Draw(canvas)
            width, height = canvas.size
            
            # 外边框
            border_width = config.border_width
            border_color = colors["accent"]
            
            # 解析颜色
            if border_color.startswith('#'):
                border_rgb = tuple(int(border_color[i:i+2], 16) for i in (1, 3, 5))
            else:
                border_rgb = (255, 182, 193)
            
            # 绘制边框
            for i in range(border_width):
                draw.rectangle([i, i, width - i - 1, height - i - 1], outline=border_rgb, width=1)
            
            return canvas
        except Exception as e:
            logger.warning(f"装饰边框添加失败: {e}")
            return canvas
    
    async def _save_high_quality_image(self, canvas: Image.Image, config: CollageConfig) -> Tuple[Path, str]:
        """保存高质量图片"""
        try:
            # 生成唯一文件名
            timestamp = int(time.time())
            filename = f"xiaohongshu_collage_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            output_path = self.output_dir / filename
            
            # 保存高质量JPEG
            canvas.save(
                output_path,
                "JPEG",
                quality=config.quality,
                dpi=(config.dpi, config.dpi),
                optimize=True
            )
            
            # 生成base64数据
            buffer = io.BytesIO()
            canvas.save(buffer, "JPEG", quality=config.quality)
            base64_data = base64.b64encode(buffer.getvalue()).decode()
            
            return output_path, base64_data
            
        except Exception as e:
            logger.error(f"图片保存失败: {e}")
            raise


# 全局实例
_xiaohongshu_collage_generator = None

def get_xiaohongshu_collage_generator() -> XiaohongshuCollageGenerator:
    """获取小红书拼图生成器实例"""
    global _xiaohongshu_collage_generator
    if _xiaohongshu_collage_generator is None:
        _xiaohongshu_collage_generator = XiaohongshuCollageGenerator()
    return _xiaohongshu_collage_generator
