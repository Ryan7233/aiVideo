"""
高级拼图生成服务 - 实现真正的拼图效果
"""
import os
import io
import json
import base64
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from loguru import logger

class AdvancedCollageGenerator:
    """高级拼图生成器"""
    
    def __init__(self):
        self.default_size = (800, 800)
        self.font_paths = self._get_available_fonts()
        
    def _get_available_fonts(self) -> List[str]:
        """获取可用字体（优先中文字体，避免中文乱码）"""
        font_paths: List[str] = []
        
        # 高优先级中文字体候选
        preferred_cn_fonts = [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            # Windows
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/arialuni.ttf",  # Arial Unicode
            # Linux 常见Noto字体
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
        ]
        # 其他英文字体（备用）
        secondary_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        for fp in preferred_cn_fonts + secondary_fonts:
            if os.path.exists(fp):
                font_paths.append(fp)
        return font_paths

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """按优先顺序加载最适合的中文字体，避免乱码。"""
        # 优先使用已经发现的可用字体路径
        candidates = list(self.font_paths)
        # 如果未找到，尝试常见中文字体路径
        if not candidates:
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/arialuni.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
        # 逐个尝试加载（.ttc 需要兼容索引）
        for path in candidates:
            try:
                if path.lower().endswith('.ttc'):
                    # 尝试多个索引（不同字族）
                    for idx in (0, 1, 2, 3):
                        try:
                            return ImageFont.truetype(path, size=size, index=idx)
                        except Exception:
                            continue
                else:
                    return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        # 最后回退
        return ImageFont.load_default()
    
    async def generate_advanced_collage(
        self,
        images: List[str],  # 图片路径列表
        title: str,
        layout_type: str = "dynamic",  # dynamic, grid, magazine, mosaic, creative
        style: str = "modern",  # modern, vintage, artistic, minimal
        color_scheme: str = "auto",  # auto, warm, cool, monochrome, vibrant
        canvas_size: Tuple[int, int] = (800, 800),
        add_effects: bool = True,
        add_text_overlay: bool = True,
        extra_text: str = "",
        text_position: str = "bottom"
    ) -> Dict[str, Any]:
        """生成高级拼图"""
        
        try:
            # 加载和预处理图片
            processed_images = await self._load_and_process_images(images)
            
            if not processed_images:
                raise ValueError("没有有效的图片可以处理")
            
            # 创建画布
            canvas = Image.new('RGB', canvas_size, (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            
            # 根据布局类型生成拼图
            if layout_type == "dynamic":
                canvas = await self._create_dynamic_layout(canvas, processed_images, title, style)
            elif layout_type == "grid":
                canvas = await self._create_grid_layout(canvas, processed_images, title, style)
            elif layout_type == "magazine":
                canvas = await self._create_magazine_layout(canvas, processed_images, title, style)
            elif layout_type == "mosaic":
                canvas = await self._create_mosaic_layout(canvas, processed_images, title, style)
            elif layout_type == "creative":
                canvas = await self._create_creative_layout(canvas, processed_images, title, style)
            elif layout_type == "treemap":
                canvas = await self._create_treemap_layout(canvas, processed_images, title, style)
            else:
                canvas = await self._create_grid_layout(canvas, processed_images, title, style)
            
            # 应用颜色方案
            canvas = await self._apply_color_scheme(canvas, color_scheme)
            
            # 添加效果
            if add_effects:
                canvas = await self._add_visual_effects(canvas, style)
            
            # 添加文字叠加
            if add_text_overlay:
                canvas = await self._add_text_overlay(canvas, title, style, extra_text=extra_text, position=text_position)
            
            # 转换为base64
            output_buffer = io.BytesIO()
            canvas.save(output_buffer, format='PNG', quality=95)
            output_buffer.seek(0)
            
            base64_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            return {
                "status": "success",
                "collage_base64": f"data:image/png;base64,{base64_image}",
                "metadata": {
                    "layout_type": layout_type,
                    "style": style,
                    "color_scheme": color_scheme,
                    "canvas_size": canvas_size,
                    "image_count": len(processed_images),
                    "title": title
                }
            }
            
        except Exception as e:
            logger.error(f"高级拼图生成失败: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_available": True
            }
    
    async def _load_and_process_images(self, image_paths: List[str]) -> List[Image.Image]:
        """加载和预处理图片"""
        processed_images = []
        
        for path in image_paths:
            try:
                if os.path.exists(path):
                    img = Image.open(path)
                    
                    # 转换为RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 基本尺寸标准化
                    max_size = 1000
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    processed_images.append(img)
                    
                else:
                    logger.warning(f"图片文件不存在: {path}")
                    
            except Exception as e:
                logger.error(f"处理图片失败 {path}: {str(e)}")
                continue
        
        return processed_images
    
    async def _create_dynamic_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建动态布局 - 根据图片数量和比例智能排列"""
        
        canvas_width, canvas_height = canvas.size
        img_count = len(images)
        
        # 为标题预留空间
        title_height = 80
        available_height = canvas_height - title_height
        
        # 创建渐变背景
        canvas = self._add_gradient_background(canvas, style)
        
        if img_count == 1:
            # 单图：居中显示，保持比例，添加装饰边框
            img = images[0]
            max_width = int(canvas_width * 0.75)
            max_height = int(available_height * 0.75)
            
            img = self._resize_keep_ratio(img, (max_width, max_height))
            img = self._add_rounded_corners(img, 25)
            img = self._add_border_effect(img, style)
            
            x = (canvas_width - img.width) // 2
            y = title_height + (available_height - img.height) // 2
            canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            
        elif img_count == 2:
            # 双图：智能布局选择
            img1, img2 = images[0], images[1]
            
            # 分析图片比例决定布局
            avg_ratio = (img1.width/img1.height + img2.width/img2.height) / 2
            
            if avg_ratio > 1.3:  # 横图为主
                # 上下排列，增加间距和效果
                spacing = 15
                img_height = (available_height - spacing * 3) // 2
                
                for i, img in enumerate([img1, img2]):
                    img = self._resize_and_crop(img, (canvas_width - 60, img_height))
                    img = self._add_rounded_corners(img, 20)
                    img = self._add_shadow(img, (3, 3))
                    
                    x = 30
                    y = title_height + spacing + i * (img_height + spacing)
                    canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            else:  # 竖图或方图
                # 左右排列，添加重叠效果
                img_width = (canvas_width - 90) // 2
                overlap = 20
                
                for i, img in enumerate([img1, img2]):
                    img = self._resize_and_crop(img, (img_width, available_height - 40))
                    img = self._add_rounded_corners(img, 18)
                    
                    if i == 1:  # 第二张图片添加阴影层次感
                        img = self._add_shadow(img, (5, 5))
                    
                    x = 30 + i * (img_width - overlap)
                    y = title_height + 20
                    canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
        
        elif img_count == 3:
            # 三图：创意三角形布局
            main_img = images[0]
            side_imgs = images[1:3]
            
            # 主图占上方中央
            main_size = int(min(canvas_width, available_height) * 0.45)
            main_img = self._resize_and_crop(main_img, (main_size, main_size))
            main_img = self._add_rounded_corners(main_img, 22)
            main_img = self._add_border_effect(main_img, style)
            
            main_x = (canvas_width - main_size) // 2
            main_y = title_height + 20
            canvas.paste(main_img, (main_x, main_y), main_img if main_img.mode == 'RGBA' else None)
            
            # 下方两张图呈扇形排列
            side_size = int(main_size * 0.75)
            side_spacing = main_size // 3
            
            for i, img in enumerate(side_imgs):
                img = self._resize_and_crop(img, (side_size, side_size))
                img = self._add_rounded_corners(img, 18)
                img = self._add_shadow(img, (2, 2))
                
                if i == 0:  # 左下
                    x = main_x - side_spacing
                    y = main_y + main_size - side_size + 30
                else:  # 右下
                    x = main_x + main_size - side_size + side_spacing
                    y = main_y + main_size - side_size + 30
                
                canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
        
        elif img_count >= 4:
            # 多图：增强网格布局
            canvas = await self._create_enhanced_grid_layout(canvas, images, title, style)
        
        return canvas
    
    async def _create_enhanced_grid_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建增强的网格布局"""
        
        canvas_width, canvas_height = canvas.size
        img_count = len(images)
        
        # 计算最优网格尺寸
        if img_count <= 4:
            cols, rows = 2, 2
        elif img_count <= 6:
            cols, rows = 3, 2
        elif img_count <= 9:
            cols, rows = 3, 3
        else:
            cols = 4
            rows = math.ceil(img_count / cols)
        
        # 为标题预留空间
        title_height = 80
        available_height = canvas_height - title_height
        
        # 计算单元格尺寸和间距
        spacing = 12
        total_spacing_h = (cols + 1) * spacing
        total_spacing_v = (rows + 1) * spacing
        
        cell_width = (canvas_width - total_spacing_h) // cols
        cell_height = (available_height - total_spacing_v) // rows
        
        # 随机化布局增加视觉趣味
        layout_variations = [
            "standard",  # 标准网格
            "staggered",  # 交错排列
            "circular",   # 圆形排列
            "overlap"     # 重叠效果
        ]
        
        layout_type = random.choice(layout_variations[:2])  # 先用前两种稳定的布局
        
        for i, img in enumerate(images[:cols * rows]):
            row = i // cols
            col = i % cols
            
            # 调整图片尺寸
            target_size = (cell_width, cell_height)
            
            # 为某些位置添加尺寸变化
            if layout_type == "staggered" and (i % 3 == 0):
                target_size = (int(cell_width * 1.1), int(cell_height * 1.1))
            
            img = self._resize_and_crop(img, target_size)
            
            # 添加圆角和效果
            corner_radius = 15 + (i % 3) * 3  # 变化的圆角
            img = self._add_rounded_corners(img, corner_radius)
            
            # 随机添加一些特殊效果
            if i % 4 == 0:  # 每4张图片添加阴影
                img = self._add_shadow(img, (3, 3))
            elif i % 5 == 0:  # 每5张图片添加边框
                img = self._add_border_effect(img, style)
            
            # 计算位置
            base_x = spacing + col * (cell_width + spacing)
            base_y = title_height + spacing + row * (cell_height + spacing)
            
            # 添加位置微调
            if layout_type == "staggered":
                offset_x = (i % 3 - 1) * 5  # -5, 0, 5 的偏移
                offset_y = (i % 2) * 8       # 0, 8 的偏移
                base_x += offset_x
                base_y += offset_y
            
            canvas.paste(img, (base_x, base_y), img if img.mode == 'RGBA' else None)
        
        return canvas
    
    async def _create_grid_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建网格布局"""
        
        canvas_width, canvas_height = canvas.size
        img_count = len(images)
        
        # 计算网格尺寸
        cols = math.ceil(math.sqrt(img_count))
        rows = math.ceil(img_count / cols)
        
        # 为标题预留空间
        title_height = 80
        available_height = canvas_height - title_height
        
        # 计算每个单元格的尺寸
        cell_width = (canvas_width - (cols + 1) * 10) // cols
        cell_height = (available_height - (rows + 1) * 10) // rows
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            
            # 调整图片尺寸
            img = self._resize_and_crop(img, (cell_width, cell_height))
            
            # 计算位置
            x = 10 + col * (cell_width + 10)
            y = title_height + 10 + row * (cell_height + 10)
            
            # 添加圆角效果
            img = self._add_rounded_corners(img, 15)
            
            canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
        
        return canvas
    
    async def _create_magazine_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建杂志风格布局"""
        
        canvas_width, canvas_height = canvas.size
        title_height = 80
        available_height = canvas_height - title_height
        
        if len(images) >= 3:
            # 主图 + 辅助图片的杂志布局
            main_img = images[0]
            
            # 主图占据左上角大部分空间
            main_width = int(canvas_width * 0.65)
            main_height = int(available_height * 0.7)
            main_img = self._resize_and_crop(main_img, (main_width, main_height))
            main_img = self._add_rounded_corners(main_img, 20)
            canvas.paste(main_img, (15, title_height + 15), main_img if main_img.mode == 'RGBA' else None)
            
            # 右侧小图竖向排列
            remaining_images = images[1:4]  # 最多3张辅助图
            side_width = canvas_width - main_width - 45
            side_height = (available_height - 60) // len(remaining_images)
            
            for i, img in enumerate(remaining_images):
                img = self._resize_and_crop(img, (side_width, side_height - 10))
                img = self._add_rounded_corners(img, 12)
                
                x = main_width + 30
                y = title_height + 15 + i * side_height
                
                canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            
            # 底部横向小图
            if len(images) > 4:
                bottom_images = images[4:7]
                bottom_width = (canvas_width - 60) // len(bottom_images)
                bottom_height = available_height - main_height - 45
                
                for i, img in enumerate(bottom_images):
                    img = self._resize_and_crop(img, (bottom_width - 10, bottom_height))
                    img = self._add_rounded_corners(img, 10)
                    
                    x = 15 + i * bottom_width
                    y = title_height + main_height + 30
                    
                    canvas.paste(img, (x, y), img if img.mode == 'RGBA' else None)
        
        return canvas
    
    async def _create_mosaic_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建马赛克拼图布局"""
        
        # 马赛克效果：不规则大小和位置的图片拼贴
        canvas_width, canvas_height = canvas.size
        title_height = 80
        available_height = canvas_height - title_height
        
        # 创建随机布局区域
        regions = self._generate_mosaic_regions(canvas_width, available_height, len(images))
        
        for i, (img, region) in enumerate(zip(images, regions)):
            x, y, width, height = region
            
            # 调整图片到区域大小
            img = self._resize_and_crop(img, (width, height))
            
            # 随机旋转角度（小角度）
            if random.random() > 0.7:  # 30%概率旋转
                angle = random.randint(-5, 5)
                img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
            
            # 添加阴影效果
            img = self._add_shadow(img)
            
            canvas.paste(img, (x, y + title_height), img if img.mode == 'RGBA' else None)
        
        return canvas
    
    async def _create_creative_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """创建创意布局 - 圆形、多边形等特殊形状"""
        
        canvas_width, canvas_height = canvas.size
        title_height = 80
        available_height = canvas_height - title_height
        
        # 圆形拼图布局
        center_x = canvas_width // 2
        center_y = title_height + available_height // 2
        
        if len(images) == 1:
            # 单张大圆形
            radius = min(canvas_width, available_height) // 3
            img = self._resize_and_crop(images[0], (radius * 2, radius * 2))
            img = self._create_circle_image(img, radius)
            
            canvas.paste(img, (center_x - radius, center_y - radius), img)
            
        elif len(images) <= 6:
            # 中心一个大圆，周围小圆环绕
            main_radius = min(canvas_width, available_height) // 5
            small_radius = main_radius // 2
            
            # 中心图
            main_img = self._resize_and_crop(images[0], (main_radius * 2, main_radius * 2))
            main_img = self._create_circle_image(main_img, main_radius)
            canvas.paste(main_img, (center_x - main_radius, center_y - main_radius), main_img)
            
            # 周围图片
            remaining_images = images[1:]
            angle_step = 360 / len(remaining_images)
            orbit_radius = main_radius * 2
            
            for i, img in enumerate(remaining_images):
                angle = math.radians(i * angle_step)
                x = int(center_x + orbit_radius * math.cos(angle) - small_radius)
                y = int(center_y + orbit_radius * math.sin(angle) - small_radius)
                
                img = self._resize_and_crop(img, (small_radius * 2, small_radius * 2))
                img = self._create_circle_image(img, small_radius)
                
                canvas.paste(img, (x, y), img)
        
        return canvas

    async def _create_treemap_layout(
        self, canvas: Image.Image, images: List[Image.Image], title: str, style: str
    ) -> Image.Image:
        """树地图布局：根据图片权重按区域切分，形成混合尺寸的自然拼贴。"""
        width, height = canvas.size
        title_h = 80
        pad = 12
        region = (pad, title_h + pad, width - 2 * pad, height - title_h - 2 * pad)

        # 背景渐变
        canvas = self._add_gradient_background(canvas, style)

        # 计算权重（使用原图面积，避免极小块）
        weights = [max(1, img.width * img.height) for img in images]
        total = sum(weights)
        weights = [w / total for w in weights]

        rects: List[Tuple[int, int, int, int, int]] = []

        def slice_rect(x: int, y: int, w: int, h: int, idxs: List[int]):
            if not idxs or w <= 0 or h <= 0:
                return
            if len(idxs) == 1:
                rects.append((x, y, w, h, idxs[0]))
                return
            # 沿长边切分，第一块按权重占比
            horizontal = w >= h
            sum_w = sum(weights[i] for i in idxs)
            first = idxs[0]
            ratio = max(0.08, min(0.92, weights[first] / sum_w))
            if horizontal:
                cut = max(60, int(w * ratio))
                rects.append((x, y, cut, h, first))
                slice_rect(x + cut + pad, y, max(0, w - cut - pad), h, idxs[1:])
            else:
                cut = max(60, int(h * ratio))
                rects.append((x, y, w, cut, first))
                slice_rect(x, y + cut + pad, w, max(0, h - cut - pad), idxs[1:])

        slice_rect(*region, list(range(len(images))))

        # 绘制
        for (rx, ry, rw, rh, idx) in rects:
            img = images[idx]
            tw, th = max(8, rw), max(8, rh)
            tile = self._resize_and_crop(img, (tw, th))
            tile = self._add_rounded_corners(tile, 12)
            if idx % 3 == 0:
                tile = self._add_shadow(tile, (3, 3))
            canvas.paste(tile, (rx, ry), tile if tile.mode == 'RGBA' else None)

        return canvas
    
    def _resize_keep_ratio(self, img: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
        """保持比例调整图片大小"""
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
    
    def _resize_and_crop(self, img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """调整大小并裁剪到目标尺寸"""
        target_width, target_height = target_size
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # 图片更宽，按高度缩放
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            # 图片更高，按宽度缩放
            new_width = target_width
            new_height = int(target_width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 居中裁剪
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return img.crop((left, top, right, bottom))
    
    def _add_rounded_corners(self, img: Image.Image, radius: int) -> Image.Image:
        """添加圆角"""
        # 创建圆角遮罩
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), img.size], radius=radius, fill=255)
        
        # 应用遮罩
        result = Image.new('RGBA', img.size, (255, 255, 255, 0))
        result.paste(img, (0, 0))
        result.putalpha(mask)
        
        return result
    
    def _create_circle_image(self, img: Image.Image, radius: int) -> Image.Image:
        """创建圆形图片"""
        # 创建圆形遮罩
        mask = Image.new('L', (radius * 2, radius * 2), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([(0, 0), (radius * 2, radius * 2)], fill=255)
        
        # 应用遮罩
        result = Image.new('RGBA', (radius * 2, radius * 2), (255, 255, 255, 0))
        result.paste(img, (0, 0))
        result.putalpha(mask)
        
        return result
    
    def _add_shadow(self, img: Image.Image, offset: Tuple[int, int] = (5, 5)) -> Image.Image:
        """添加阴影效果"""
        # 创建阴影层
        shadow = Image.new('RGBA', 
                          (img.width + offset[0], img.height + offset[1]), 
                          (255, 255, 255, 0))
        
        # 创建阴影
        shadow_img = Image.new('RGBA', img.size, (0, 0, 0, 100))
        shadow.paste(shadow_img, offset)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
        
        # 合并原图和阴影
        result = Image.new('RGBA', shadow.size, (255, 255, 255, 0))
        result.paste(shadow, (0, 0))
        result.paste(img, (0, 0), img if img.mode == 'RGBA' else None)
        
        return result
    
    def _generate_mosaic_regions(self, width: int, height: int, count: int) -> List[Tuple[int, int, int, int]]:
        """生成马赛克布局区域"""
        regions = []
        
        # 简化的马赛克算法：随机划分区域
        for i in range(count):
            # 随机大小（在合理范围内）
            min_size = min(width, height) // 4
            max_size = min(width, height) // 2
            
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            
            # 随机位置（确保不超出边界）
            x = random.randint(0, max(0, width - w))
            y = random.randint(0, max(0, height - h))
            
            regions.append((x, y, w, h))
        
        return regions
    
    async def _apply_color_scheme(self, canvas: Image.Image, color_scheme: str) -> Image.Image:
        """应用颜色方案"""
        if color_scheme == "auto":
            return canvas
        
        enhancer = ImageEnhance.Color(canvas)
        
        if color_scheme == "warm":
            # 增强暖色调
            canvas = enhancer.enhance(1.2)
            # 添加暖色滤镜
            overlay = Image.new('RGBA', canvas.size, (255, 200, 150, 30))
            canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
            
        elif color_scheme == "cool":
            # 增强冷色调
            canvas = enhancer.enhance(1.1)
            overlay = Image.new('RGBA', canvas.size, (150, 200, 255, 30))
            canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
            
        elif color_scheme == "monochrome":
            # 单色调
            canvas = canvas.convert('L').convert('RGB')
            
        elif color_scheme == "vibrant":
            # 鲜艳色彩
            canvas = enhancer.enhance(1.5)
        
        return canvas
    
    async def _add_visual_effects(self, canvas: Image.Image, style: str) -> Image.Image:
        """添加视觉效果"""
        if style == "vintage":
            # 复古效果
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.5))
            enhancer = ImageEnhance.Contrast(canvas)
            canvas = enhancer.enhance(0.9)
            
        elif style == "artistic":
            # 艺术效果
            canvas = canvas.filter(ImageFilter.EDGE_ENHANCE_MORE)
            
        elif style == "minimal":
            # 极简效果
            enhancer = ImageEnhance.Brightness(canvas)
            canvas = enhancer.enhance(1.1)
        
        return canvas
    
    def _add_gradient_background(self, canvas: Image.Image, style: str) -> Image.Image:
        """添加渐变背景"""
        width, height = canvas.size
        
        # 根据风格选择渐变色
        if style == "modern":
            colors = [(245, 247, 250), (255, 255, 255)]  # 浅灰到白
        elif style == "vintage":
            colors = [(250, 240, 230), (245, 235, 220)]  # 米色渐变
        elif style == "artistic":
            colors = [(240, 248, 255), (230, 230, 250)]  # 蓝紫渐变
        else:  # minimal
            colors = [(255, 255, 255), (248, 249, 250)]  # 纯白渐变
        
        # 创建垂直渐变
        for y in range(height):
            ratio = y / height
            r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            
            draw = ImageDraw.Draw(canvas)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return canvas
    
    def _add_border_effect(self, img: Image.Image, style: str) -> Image.Image:
        """添加边框效果"""
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 创建带边框的新图像
        border_width = 8 if style == "vintage" else 4
        new_size = (img.width + border_width * 2, img.height + border_width * 2)
        
        # 根据风格选择边框颜色
        if style == "modern":
            border_color = (255, 255, 255, 200)
        elif style == "vintage":
            border_color = (139, 69, 19, 180)  # 棕色
        elif style == "artistic":
            border_color = (75, 0, 130, 150)   # 紫色
        else:  # minimal
            border_color = (220, 220, 220, 180)
        
        bordered_img = Image.new('RGBA', new_size, border_color)
        bordered_img.paste(img, (border_width, border_width), img)
        
        return bordered_img

    async def _add_text_overlay(self, canvas: Image.Image, title: str, style: str, extra_text: str = "", position: str = "bottom") -> Image.Image:
        """添加文字叠加"""
        draw = ImageDraw.Draw(canvas)
        
        # 选择字体
        font_size = 42 if len(title) < 10 else 36
        font = None
        
        if self.font_paths:
            try:
                font = self._load_font(font_size)
            except Exception:
                pass
        
        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                return canvas  # 如果无法加载字体，跳过文字
        
        # 计算文字位置
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (canvas.width - text_width) // 2
        # 放置位置：顶部或底部或居中
        if position == "center":
            y = canvas.height // 2 - text_height - 10
        else:
            y = 25
        
        # 根据风格添加背景框
        if style in ["modern", "artistic"]:
            # 添加半透明背景框
            padding = 15
            box_coords = [
                x - padding, y - padding//2,
                x + text_width + padding, y + text_height + padding//2
            ]
            
            bg_color = (0, 0, 0, 100) if style == "artistic" else (255, 255, 255, 150)
            overlay = Image.new('RGBA', canvas.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rounded_rectangle(box_coords, radius=12, fill=bg_color)
            canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(canvas)
        
        # 添加文字阴影
        if style != "minimal":
            shadow_offset = 3 if style == "vintage" else 2
            shadow_color = (0, 0, 0, 100)
            draw.text((x + shadow_offset, y + shadow_offset), title, font=font, fill=shadow_color)
        
        # 添加主文字
        if style == "vintage":
            text_color = (139, 69, 19)  # 棕色
        elif style == "artistic":
            text_color = (255, 255, 255)  # 白色
        elif style == "minimal":
            text_color = (100, 100, 100)  # 灰色
        else:  # modern
            text_color = (50, 50, 50)  # 深灰
            
        draw.text((x, y), title, font=font, fill=text_color)
        
        # 额外长文案（自动换行，绘制在标题下方）
        if extra_text:
            wrap_width = int(canvas.width * 0.8)
            body_font_size = max(22, font_size - 8)
            try:
                body_font = self._load_font(body_font_size)
            except Exception:
                body_font = ImageFont.load_default()
            
            # 简单按字符宽度折行（兼容中文）
            def wrap(text: str) -> List[str]:
                lines, line = [], ''
                for ch in text:
                    test = line + ch
                    w = draw.textbbox((0,0), test, font=body_font)[2]
                    if w <= wrap_width:
                        line = test
                    else:
                        if line:
                            lines.append(line)
                        line = ch
                if line:
                    lines.append(line)
                return lines[:6]  # 控制最多6行，避免过长
            
            lines = wrap(extra_text.strip())
            if lines:
                # 背景框
                padding = 12
                line_h = draw.textbbox((0,0), '测', font=body_font)[3]
                total_h = len(lines) * line_h + (len(lines)-1) * 6 + padding*2
                box_w = wrap_width + padding*2
                bx = (canvas.width - box_w) // 2
                by = y + text_height + 16
                overlay = Image.new('RGBA', canvas.size, (0,0,0,0))
                od = ImageDraw.Draw(overlay)
                bg_alpha = 110 if style in ["modern","artistic"] else 140
                od.rounded_rectangle([bx, by, bx+box_w, by+total_h], radius=10, fill=(255,255,255,bg_alpha))
                canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(canvas)
                
                cy = by + padding
                for ln in lines:
                    lw = draw.textbbox((0,0), ln, font=body_font)[2]
                    lx = (canvas.width - lw) // 2
                    draw.text((lx, cy), ln, font=body_font, fill=(60,60,60))
                    cy += line_h + 6
        
        return canvas

# 全局服务实例
_collage_generator = None

def get_advanced_collage_generator() -> AdvancedCollageGenerator:
    """获取高级拼图生成器实例"""
    global _collage_generator
    if _collage_generator is None:
        _collage_generator = AdvancedCollageGenerator()
    return _collage_generator
