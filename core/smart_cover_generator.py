"""
智能封面生成服务
实现拼图封面生成、模板应用、文案叠加等功能
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from pathlib import Path
import io
import base64
import math


class SmartCoverGenerator:
    """智能封面生成器"""
    
    def __init__(self):
        self.output_dir = Path("output_data/covers")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 封面尺寸配置
        self.cover_sizes = {
            "xiaohongshu": (1080, 1080),  # 小红书方形
            "weibo": (1200, 900),         # 微博横版
            "douyin": (1080, 1920),       # 抖音竖版
            "instagram": (1080, 1080),    # Instagram方形
            "custom": (1200, 1200)        # 自定义
        }
        
        # 布局模板
        self.layout_templates = {
            "grid_2x2": {"rows": 2, "cols": 2, "spacing": 10},
            "grid_3x3": {"rows": 3, "cols": 3, "spacing": 8},
            "grid_2x3": {"rows": 2, "cols": 3, "spacing": 8},
            "collage_mixed": {"type": "mixed", "spacing": 10},
            "magazine": {"type": "magazine", "spacing": 15},
            "polaroid": {"type": "polaroid", "spacing": 20}
        }
        
        # 颜色主题
        self.color_themes = {
            "pink_gradient": {
                "primary": "#FF6B9D", "secondary": "#C44569", 
                "accent": "#F8B500", "background": "#FFF5F7"
            },
            "blue_gradient": {
                "primary": "#4A90E2", "secondary": "#357ABD",
                "accent": "#50C878", "background": "#F0F8FF"
            },
            "warm_sunset": {
                "primary": "#FF6B35", "secondary": "#F7931E",
                "accent": "#FFD23F", "background": "#FFF8F0"
            },
            "cool_mint": {
                "primary": "#00D2D3", "secondary": "#01A3A4",
                "accent": "#10AC84", "background": "#F0FFFF"
            },
            "elegant_gray": {
                "primary": "#2C3E50", "secondary": "#34495E",
                "accent": "#E74C3C", "background": "#F8F9FA"
            }
        }
    
    async def generate_cover(
        self,
        images: List[str],
        title: str,
        subtitle: str = "",
        layout: str = "grid_3x3",
        theme: str = "pink_gradient",
        platform: str = "xiaohongshu",
        custom_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        生成智能封面 - 真正的拼图效果
        
        Args:
            images: 图片路径列表
            title: 主标题
            subtitle: 副标题
            layout: 布局模板
            theme: 颜色主题
            platform: 目标平台
            custom_config: 自定义配置
            
        Returns:
            Dict: 生成结果
        """
        try:
            # 获取封面尺寸 - 提高分辨率
            cover_size = self.cover_sizes.get(platform, self.cover_sizes["xiaohongshu"])
            # 提高生成分辨率，确保图片质量
            high_res_size = (cover_size[0] * 2, cover_size[1] * 2)
            
            # 创建高分辨率封面画布
            cover = Image.new('RGB', high_res_size, (255, 255, 255))
            
            # 获取主题配色
            colors = self.color_themes.get(theme, self.color_themes["pink_gradient"])
            
            # 添加高质量渐变背景
            cover = self._add_premium_background(cover, colors, theme)
            
            # 加载和预处理图片
            processed_images = []
            for img_path in images:
                try:
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    # 图片质量增强
                    img = self._enhance_image_quality(img)
                    processed_images.append(img)
                except Exception as e:
                    logger.warning(f"图片加载失败: {img_path}, {str(e)}")
            
            if not processed_images:
                return {
                    "status": "error",
                    "message": "没有可用的图片"
                }
            
            # 应用真正的拼图布局
            cover = self._apply_collage_layout(cover, processed_images, layout, colors, theme)
            
            # 添加高质量标题
            cover = self._add_premium_titles(cover, title, subtitle, colors, platform)
            
            # 添加拼图装饰效果
            cover = self._add_collage_effects(cover, colors, theme)
            
            # 缩放回目标尺寸（保持高质量）
            final_cover = cover.resize(cover_size, Image.Resampling.LANCZOS)
            
            # 保存高质量封面
            timestamp = int(time.time())
            cover_path = self.output_dir / f"collage_cover_{timestamp}.png"
            final_cover.save(cover_path, "PNG", quality=98, optimize=True)
            
            # 生成高质量预览base64 - 不再缩小太多
            buffer = io.BytesIO()
            preview = final_cover.copy()
            # 保持更高的预览质量
            preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
            preview.save(buffer, format='PNG', quality=95)
            preview_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # 同时生成完整尺寸的base64
            full_buffer = io.BytesIO()
            final_cover.save(full_buffer, format='PNG', quality=98)
            full_base64 = base64.b64encode(full_buffer.getvalue()).decode()
            
            logger.info(f"高质量拼图封面生成成功: {cover_path}")
            return {
                "status": "success",
                "data": {
                    "cover_path": str(cover_path),
                    "preview_base64": f"data:image/png;base64,{preview_base64}",
                    "full_base64": f"data:image/png;base64,{full_base64}",
                    "width": cover_size[0],
                    "height": cover_size[1],
                    "layout": layout,
                    "theme": theme,
                    "platform": platform,
                    "image_count": len(processed_images),
                    "quality": "premium"
                },
                "message": "高质量拼图封面生成成功"
            }
            
        except Exception as e:
            logger.error(f"封面生成失败: {str(e)}")
            return {
                "status": "error",
                "message": f"生成失败: {str(e)}"
            }
    
    def _add_background(self, cover: Image.Image, colors: Dict[str, str], custom_config: Dict[str, Any] = None) -> Image.Image:
        """添加背景"""
        try:
            width, height = cover.size
            
            # 创建渐变背景
            gradient = Image.new('RGBA', (width, height))
            draw = ImageDraw.Draw(gradient)
            
            # 简单的线性渐变
            primary_color = self._hex_to_rgb(colors["primary"])
            secondary_color = self._hex_to_rgb(colors["secondary"])
            
            for y in range(height):
                # 计算当前行的颜色
                ratio = y / height
                r = int(primary_color[0] * (1 - ratio) + secondary_color[0] * ratio)
                g = int(primary_color[1] * (1 - ratio) + secondary_color[1] * ratio)
                b = int(primary_color[2] * (1 - ratio) + secondary_color[2] * ratio)
                
                draw.line([(0, y), (width, y)], fill=(r, g, b, 180))
            
            # 合并背景
            cover = Image.alpha_composite(cover, gradient)
            
            return cover
            
        except Exception as e:
            logger.warning(f"背景添加失败: {str(e)}")
            return cover
    
    def _apply_layout(self, cover: Image.Image, images: List[Image.Image], layout: str, colors: Dict[str, str]) -> Image.Image:
        """应用布局模板"""
        try:
            template = self.layout_templates.get(layout, self.layout_templates["grid_3x3"])
            cover_width, cover_height = cover.size
            
            if layout.startswith("grid_"):
                return self._apply_grid_layout(cover, images, template, colors)
            elif layout == "collage_mixed":
                return self._apply_mixed_layout(cover, images, template, colors)
            elif layout == "magazine":
                return self._apply_magazine_layout(cover, images, template, colors)
            elif layout == "polaroid":
                return self._apply_polaroid_layout(cover, images, template, colors)
            else:
                return self._apply_grid_layout(cover, images, template, colors)
                
        except Exception as e:
            logger.warning(f"布局应用失败: {str(e)}")
            return cover
    
    def _apply_grid_layout(self, cover: Image.Image, images: List[Image.Image], template: Dict[str, Any], colors: Dict[str, str]) -> Image.Image:
        """应用网格布局"""
        try:
            rows = template["rows"]
            cols = template["cols"]
            spacing = template["spacing"]
            
            cover_width, cover_height = cover.size
            
            # 计算图片区域（留出标题空间）
            title_space = 200
            grid_area_height = cover_height - title_space
            grid_area_y = title_space // 2
            
            # 计算每个格子的尺寸
            cell_width = (cover_width - spacing * (cols + 1)) // cols
            cell_height = (grid_area_height - spacing * (rows + 1)) // rows
            
            # 放置图片
            for i in range(min(len(images), rows * cols)):
                row = i // cols
                col = i % cols
                
                # 计算位置
                x = spacing + col * (cell_width + spacing)
                y = grid_area_y + spacing + row * (cell_height + spacing)
                
                # 调整图片尺寸
                img = images[i].copy()
                img = self._resize_and_crop(img, (cell_width, cell_height))
                
                # 添加圆角和边框
                img = self._add_rounded_corners(img, 20)
                img = self._add_border(img, colors["accent"], 3)
                
                # 粘贴到封面
                cover.paste(img, (x, y), img)
            
            return cover
            
        except Exception as e:
            logger.warning(f"网格布局失败: {str(e)}")
            return cover
    
    def _apply_mixed_layout(self, cover: Image.Image, images: List[Image.Image], template: Dict[str, Any], colors: Dict[str, str]) -> Image.Image:
        """应用混合拼贴布局"""
        try:
            cover_width, cover_height = cover.size
            spacing = template["spacing"]
            
            # 预定义的混合布局位置（相对坐标）
            positions = [
                {"x": 0.1, "y": 0.2, "w": 0.35, "h": 0.3, "rotation": -5},
                {"x": 0.55, "y": 0.15, "w": 0.3, "h": 0.4, "rotation": 3},
                {"x": 0.15, "y": 0.6, "w": 0.25, "h": 0.25, "rotation": -2},
                {"x": 0.6, "y": 0.65, "w": 0.3, "h": 0.2, "rotation": 4},
                {"x": 0.05, "y": 0.05, "w": 0.2, "h": 0.15, "rotation": -3}
            ]
            
            for i, img in enumerate(images[:len(positions)]):
                pos = positions[i]
                
                # 计算实际位置和尺寸
                x = int(pos["x"] * cover_width)
                y = int(pos["y"] * cover_height)
                w = int(pos["w"] * cover_width)
                h = int(pos["h"] * cover_height)
                
                # 调整图片
                resized_img = self._resize_and_crop(img, (w, h))
                
                # 旋转图片
                if pos.get("rotation", 0) != 0:
                    resized_img = resized_img.rotate(pos["rotation"], expand=True, fillcolor=(0, 0, 0, 0))
                
                # 添加阴影效果
                shadow_img = self._add_shadow(resized_img, offset=(5, 5), blur=10)
                
                # 粘贴到封面
                cover.paste(shadow_img, (x, y), shadow_img)
            
            return cover
            
        except Exception as e:
            logger.warning(f"混合布局失败: {str(e)}")
            return cover
    
    def _apply_magazine_layout(self, cover: Image.Image, images: List[Image.Image], template: Dict[str, Any], colors: Dict[str, str]) -> Image.Image:
        """应用杂志风格布局"""
        try:
            cover_width, cover_height = cover.size
            
            if len(images) > 0:
                # 主图片（大图）
                main_img = images[0]
                main_width = int(cover_width * 0.6)
                main_height = int(cover_height * 0.5)
                main_img = self._resize_and_crop(main_img, (main_width, main_height))
                
                # 添加杂志风格边框
                main_img = self._add_magazine_frame(main_img, colors)
                
                # 放置主图片
                main_x = (cover_width - main_width) // 2
                main_y = int(cover_height * 0.3)
                cover.paste(main_img, (main_x, main_y), main_img)
                
                # 小图片（如果有）
                if len(images) > 1:
                    small_size = int(cover_width * 0.15)
                    for i, img in enumerate(images[1:4]):  # 最多3张小图
                        small_img = self._resize_and_crop(img, (small_size, small_size))
                        small_img = self._add_rounded_corners(small_img, 10)
                        
                        # 放置在右侧
                        small_x = main_x + main_width + 20
                        small_y = main_y + i * (small_size + 10)
                        
                        if small_x + small_size <= cover_width:
                            cover.paste(small_img, (small_x, small_y), small_img)
            
            return cover
            
        except Exception as e:
            logger.warning(f"杂志布局失败: {str(e)}")
            return cover
    
    def _apply_polaroid_layout(self, cover: Image.Image, images: List[Image.Image], template: Dict[str, Any], colors: Dict[str, str]) -> Image.Image:
        """应用宝丽来风格布局"""
        try:
            cover_width, cover_height = cover.size
            
            # 宝丽来照片尺寸
            polaroid_width = int(cover_width * 0.25)
            polaroid_height = int(polaroid_width * 1.2)  # 宝丽来比例
            
            # 随机放置位置
            positions = [
                (0.1, 0.2, -8), (0.6, 0.15, 5), (0.2, 0.6, -3),
                (0.65, 0.65, 7), (0.05, 0.05, -5)
            ]
            
            for i, img in enumerate(images[:len(positions)]):
                pos_x, pos_y, rotation = positions[i]
                
                # 创建宝丽来框架
                polaroid = self._create_polaroid_frame(polaroid_width, polaroid_height, colors)
                
                # 调整图片到宝丽来内部尺寸
                inner_size = int(polaroid_width * 0.85)
                inner_img = self._resize_and_crop(img, (inner_size, inner_size))
                
                # 粘贴图片到宝丽来框架
                inner_x = (polaroid_width - inner_size) // 2
                inner_y = int(polaroid_width * 0.1)
                polaroid.paste(inner_img, (inner_x, inner_y))
                
                # 旋转宝丽来
                if rotation != 0:
                    polaroid = polaroid.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
                
                # 添加阴影
                polaroid = self._add_shadow(polaroid, offset=(3, 3), blur=8)
                
                # 放置到封面
                x = int(pos_x * cover_width)
                y = int(pos_y * cover_height)
                cover.paste(polaroid, (x, y), polaroid)
            
            return cover
            
        except Exception as e:
            logger.warning(f"宝丽来布局失败: {str(e)}")
            return cover
    
    def _add_titles(self, cover: Image.Image, title: str, subtitle: str, colors: Dict[str, str], platform: str) -> Image.Image:
        """添加标题"""
        try:
            draw = ImageDraw.Draw(cover)
            cover_width, cover_height = cover.size
            
            # 加载字体
            try:
                title_font = ImageFont.truetype("assets/fonts/title.ttf", 48)
                subtitle_font = ImageFont.truetype("assets/fonts/subtitle.ttf", 28)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
            
            # 主标题
            if title:
                # 计算标题位置
                title_bbox = draw.textbbox((0, 0), title, font=title_font)
                title_width = title_bbox[2] - title_bbox[0]
                title_height = title_bbox[3] - title_bbox[1]
                
                title_x = (cover_width - title_width) // 2
                title_y = 50
                
                # 添加标题背景
                bg_padding = 20
                draw.rectangle([
                    title_x - bg_padding, title_y - bg_padding,
                    title_x + title_width + bg_padding, title_y + title_height + bg_padding
                ], fill=(*self._hex_to_rgb(colors["primary"]), 200))
                
                # 绘制标题
                draw.text((title_x, title_y), title, font=title_font, fill="white")
            
            # 副标题
            if subtitle:
                subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
                subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
                
                subtitle_x = (cover_width - subtitle_width) // 2
                subtitle_y = title_y + title_height + 30
                
                draw.text((subtitle_x, subtitle_y), subtitle, font=subtitle_font, fill=colors["secondary"])
            
            return cover
            
        except Exception as e:
            logger.warning(f"标题添加失败: {str(e)}")
            return cover
    
    def _add_decorative_elements(self, cover: Image.Image, colors: Dict[str, str], theme: str) -> Image.Image:
        """添加装饰元素"""
        try:
            draw = ImageDraw.Draw(cover)
            cover_width, cover_height = cover.size
            
            # 添加角落装饰
            accent_color = self._hex_to_rgb(colors["accent"])
            
            # 左上角
            draw.ellipse([20, 20, 80, 80], fill=(*accent_color, 150))
            
            # 右下角
            draw.ellipse([cover_width-80, cover_height-80, cover_width-20, cover_height-20], fill=(*accent_color, 150))
            
            # 添加一些小装饰点
            for i in range(10):
                x = np.random.randint(0, cover_width)
                y = np.random.randint(0, cover_height)
                size = np.random.randint(3, 8)
                draw.ellipse([x, y, x+size, y+size], fill=(*accent_color, 100))
            
            return cover
            
        except Exception as e:
            logger.warning(f"装饰元素添加失败: {str(e)}")
            return cover
    
    # 辅助方法
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """将十六进制颜色转换为RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _resize_and_crop(self, img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """调整图片尺寸并裁剪"""
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]
        
        if img_ratio > target_ratio:
            # 图片更宽，以高度为准
            new_height = size[1]
            new_width = int(new_height * img_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 裁剪宽度
            left = (new_width - size[0]) // 2
            img = img.crop((left, 0, left + size[0], size[1]))
        else:
            # 图片更高，以宽度为准
            new_width = size[0]
            new_height = int(new_width / img_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 裁剪高度
            top = (new_height - size[1]) // 2
            img = img.crop((0, top, size[0], top + size[1]))
        
        return img
    
    def _add_rounded_corners(self, img: Image.Image, radius: int) -> Image.Image:
        """添加圆角"""
        try:
            # 创建圆角遮罩
            mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle([0, 0, img.size[0], img.size[1]], radius, fill=255)
            
            # 应用遮罩
            output = Image.new('RGBA', img.size, (0, 0, 0, 0))
            output.paste(img, (0, 0))
            output.putalpha(mask)
            
            return output
        except:
            return img
    
    def _add_border(self, img: Image.Image, color: str, width: int) -> Image.Image:
        """添加边框"""
        try:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            border_color = self._hex_to_rgb(color)
            
            for i in range(width):
                draw.rectangle([i, i, w-1-i, h-1-i], outline=border_color)
            
            return img
        except:
            return img
    
    def _add_shadow(self, img: Image.Image, offset: Tuple[int, int] = (5, 5), blur: int = 10) -> Image.Image:
        """添加阴影效果"""
        try:
            # 创建阴影
            shadow = Image.new('RGBA', (img.width + offset[0] + blur, img.height + offset[1] + blur), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            
            # 绘制阴影形状
            shadow_draw.rectangle([
                offset[0], offset[1],
                img.width + offset[0], img.height + offset[1]
            ], fill=(0, 0, 0, 100))
            
            # 模糊阴影
            shadow = shadow.filter(ImageFilter.GaussianBlur(blur//2))
            
            # 合并图片和阴影
            result = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
            result.paste(shadow, (0, 0), shadow)
            result.paste(img, (0, 0), img)
            
            return result
        except:
            return img
    
    def _add_magazine_frame(self, img: Image.Image, colors: Dict[str, str]) -> Image.Image:
        """添加杂志风格边框"""
        try:
            # 创建边框
            border_width = 20
            new_size = (img.width + border_width * 2, img.height + border_width * 2)
            framed = Image.new('RGBA', new_size, self._hex_to_rgb(colors["background"]))
            
            # 粘贴原图
            framed.paste(img, (border_width, border_width))
            
            # 添加装饰线条
            draw = ImageDraw.Draw(framed)
            accent_color = self._hex_to_rgb(colors["accent"])
            
            # 四个角的装饰线
            line_length = 30
            for corner in [(0, 0), (new_size[0]-line_length, 0), (0, new_size[1]-line_length), (new_size[0]-line_length, new_size[1]-line_length)]:
                draw.line([corner[0], corner[1], corner[0] + line_length, corner[1]], fill=accent_color, width=3)
                draw.line([corner[0], corner[1], corner[0], corner[1] + line_length], fill=accent_color, width=3)
            
            return framed
        except:
            return img
    
    def _create_polaroid_frame(self, width: int, height: int, colors: Dict[str, str]) -> Image.Image:
        """创建宝丽来相框"""
        try:
            # 创建白色背景
            polaroid = Image.new('RGBA', (width, height), (255, 255, 255, 255))
            
            # 添加底部文字区域（宝丽来特色）
            draw = ImageDraw.Draw(polaroid)
            text_area_height = int(height * 0.2)
            
            # 可以在这里添加一些装饰或文字
            # draw.rectangle([10, height - text_area_height, width - 10, height - 10], outline=(200, 200, 200))
            
            return polaroid
        except:
            return Image.new('RGBA', (width, height), (255, 255, 255, 255))

    def _add_premium_background(self, cover: Image.Image, colors: Dict[str, str], theme: str) -> Image.Image:
        """添加高质量渐变背景"""
        try:
            width, height = cover.size
            
            # 创建多层渐变背景
            gradient = Image.new('RGB', (width, height))
            
            # 根据主题创建不同的渐变效果
            if theme == "pink_gradient":
                # 粉色系多层渐变
                for y in range(height):
                    ratio = y / height
                    # 三层渐变混合
                    r1, g1, b1 = self._hex_to_rgb(colors["primary"])
                    r2, g2, b2 = self._hex_to_rgb(colors["secondary"])
                    r3, g3, b3 = self._hex_to_rgb(colors["accent"])
                    
                    # 复杂的渐变计算
                    if ratio < 0.5:
                        t = ratio * 2
                        r = int(r1 * (1 - t) + r2 * t)
                        g = int(g1 * (1 - t) + g2 * t)
                        b = int(b1 * (1 - t) + b2 * t)
                    else:
                        t = (ratio - 0.5) * 2
                        r = int(r2 * (1 - t) + r3 * t * 0.3 + r1 * t * 0.7)
                        g = int(g2 * (1 - t) + g3 * t * 0.3 + g1 * t * 0.7)
                        b = int(b2 * (1 - t) + b3 * t * 0.3 + b1 * t * 0.7)
                    
                    # 绘制渐变线
                    draw = ImageDraw.Draw(gradient)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            else:
                # 其他主题的简单渐变
                primary = self._hex_to_rgb(colors["primary"])
                secondary = self._hex_to_rgb(colors["secondary"])
                
                for y in range(height):
                    ratio = y / height
                    r = int(primary[0] * (1 - ratio) + secondary[0] * ratio)
                    g = int(primary[1] * (1 - ratio) + secondary[1] * ratio)
                    b = int(primary[2] * (1 - ratio) + secondary[2] * ratio)
                    
                    draw = ImageDraw.Draw(gradient)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))
            
            # 添加纹理效果
            gradient = self._add_texture_overlay(gradient, theme)
            
            return gradient
            
        except Exception as e:
            logger.warning(f"高质量背景添加失败: {str(e)}")
            return cover

    def _add_texture_overlay(self, img: Image.Image, theme: str) -> Image.Image:
        """添加纹理叠加效果"""
        try:
            width, height = img.size
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # 添加细微的噪点纹理
            import random
            for _ in range(width * height // 100):  # 1%的像素点
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                alpha = random.randint(10, 30)
                color = (255, 255, 255, alpha) if random.random() > 0.5 else (0, 0, 0, alpha)
                draw.point((x, y), fill=color)
            
            # 合并纹理
            result = img.convert('RGBA')
            result = Image.alpha_composite(result, overlay)
            return result.convert('RGB')
            
        except Exception as e:
            logger.warning(f"纹理叠加失败: {str(e)}")
            return img

    def _enhance_image_quality(self, img: Image.Image) -> Image.Image:
        """增强图片质量"""
        try:
            # 锐化处理
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.2)
            
            # 对比度增强
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)
            
            # 饱和度增强
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.15)
            
            # 亮度微调
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
            
            return img
        except Exception as e:
            logger.warning(f"图片质量增强失败: {str(e)}")
            return img

    def _apply_collage_layout(self, cover: Image.Image, images: List[Image.Image], layout: str, colors: Dict[str, str], theme: str) -> Image.Image:
        """应用真正的拼图布局"""
        try:
            if layout == "grid_3x3":
                return self._create_puzzle_grid(cover, images, 3, 3, colors, theme)
            elif layout == "grid_2x3":
                return self._create_puzzle_grid(cover, images, 2, 3, colors, theme)
            elif layout == "collage_mixed":
                return self._create_artistic_collage(cover, images, colors, theme)
            elif layout == "magazine":
                return self._create_magazine_collage(cover, images, colors, theme)
            else:
                return self._create_puzzle_grid(cover, images, 3, 3, colors, theme)
                
        except Exception as e:
            logger.warning(f"拼图布局应用失败: {str(e)}")
            return cover

    def _create_puzzle_grid(self, cover: Image.Image, images: List[Image.Image], rows: int, cols: int, colors: Dict[str, str], theme: str) -> Image.Image:
        """创建真正的拼图网格效果"""
        try:
            cover_width, cover_height = cover.size
            
            # 为标题预留更多空间
            title_space = int(cover_height * 0.15)
            available_height = cover_height - title_space
            grid_y_start = title_space
            
            # 计算网格尺寸，添加更大的间距
            gap = int(min(cover_width, cover_height) * 0.02)  # 2%的间距
            
            cell_width = (cover_width - gap * (cols + 1)) // cols
            cell_height = (available_height - gap * (rows + 1)) // rows
            
            # 创建拼图片段
            for i in range(min(len(images), rows * cols)):
                row = i // cols
                col = i % cols
                
                # 计算位置
                x = gap + col * (cell_width + gap)
                y = grid_y_start + gap + row * (cell_height + gap)
                
                # 处理图片
                img = images[i].copy()
                img = self._resize_and_crop(img, (cell_width, cell_height))
                
                # 创建拼图片段效果
                puzzle_piece = self._create_puzzle_piece(img, i, theme, colors)
                
                # 添加3D效果
                puzzle_piece = self._add_3d_effect(puzzle_piece, colors)
                
                # 粘贴到封面
                if puzzle_piece.mode == 'RGBA':
                    cover.paste(puzzle_piece, (x, y), puzzle_piece)
                else:
                    cover.paste(puzzle_piece, (x, y))
            
            return cover
            
        except Exception as e:
            logger.warning(f"拼图网格创建失败: {str(e)}")
            return cover

    def _create_puzzle_piece(self, img: Image.Image, index: int, theme: str, colors: Dict[str, str]) -> Image.Image:
        """创建拼图片段效果"""
        try:
            width, height = img.size
            
            # 创建带圆角的基础形状
            radius = min(width, height) // 10
            mask = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle([0, 0, width, height], radius=radius, fill=255)
            
            # 应用圆角遮罩
            result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            result.paste(img, (0, 0))
            result.putalpha(mask)
            
            # 添加拼图特有的边框效果
            border_width = max(2, min(width, height) // 100)
            draw = ImageDraw.Draw(result)
            border_color = self._hex_to_rgb(colors["accent"])
            
            # 绘制渐变边框
            for i in range(border_width):
                alpha = int(255 * (1 - i / border_width))
                border_rgba = (*border_color, alpha)
                draw.rounded_rectangle(
                    [i, i, width - 1 - i, height - 1 - i], 
                    radius=radius - i, 
                    outline=border_rgba, 
                    width=1
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"拼图片段创建失败: {str(e)}")
            return img.convert('RGBA')

    def _add_3d_effect(self, img: Image.Image, colors: Dict[str, str]) -> Image.Image:
        """添加3D立体效果"""
        try:
            width, height = img.size
            
            # 创建阴影
            shadow_offset = max(3, min(width, height) // 50)
            shadow = Image.new('RGBA', (width + shadow_offset * 2, height + shadow_offset * 2), (0, 0, 0, 0))
            
            # 绘制多层阴影
            shadow_draw = ImageDraw.Draw(shadow)
            for i in range(shadow_offset):
                alpha = int(50 * (1 - i / shadow_offset))
                shadow_draw.rounded_rectangle([
                    shadow_offset + i, shadow_offset + i,
                    width + shadow_offset + i, height + shadow_offset + i
                ], radius=min(width, height) // 10, fill=(0, 0, 0, alpha))
            
            # 模糊阴影
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=shadow_offset // 2))
            
            # 合并图片和阴影
            result = Image.new('RGBA', shadow.size, (0, 0, 0, 0))
            result.paste(shadow, (0, 0), shadow)
            result.paste(img, (0, 0), img)
            
            return result
            
        except Exception as e:
            logger.warning(f"3D效果添加失败: {str(e)}")
            return img

    def _create_artistic_collage(self, cover: Image.Image, images: List[Image.Image], colors: Dict[str, str], theme: str) -> Image.Image:
        """创建艺术拼贴效果"""
        try:
            cover_width, cover_height = cover.size
            
            # 艺术化的不规则布局
            layouts = [
                # 主图 + 小图组合
                {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.6, "rotation": -2, "z": 3},
                {"x": 0.65, "y": 0.15, "w": 0.25, "h": 0.35, "rotation": 5, "z": 2},
                {"x": 0.55, "y": 0.6, "w": 0.35, "h": 0.25, "rotation": -3, "z": 1},
                {"x": 0.05, "y": 0.85, "w": 0.2, "h": 0.1, "rotation": 8, "z": 2},
                {"x": 0.75, "y": 0.05, "w": 0.2, "h": 0.15, "rotation": -5, "z": 1},
            ]
            
            # 按z-index排序，先绘制底层
            sorted_items = []
            for i, img in enumerate(images[:len(layouts)]):
                layout = layouts[i]
                sorted_items.append((layout["z"], i, img, layout))
            
            sorted_items.sort(key=lambda x: x[0])
            
            for z, i, img, layout in sorted_items:
                # 计算实际尺寸
                w = int(layout["w"] * cover_width)
                h = int(layout["h"] * cover_height)
                x = int(layout["x"] * cover_width)
                y = int(layout["y"] * cover_height)
                
                # 处理图片
                processed_img = self._resize_and_crop(img, (w, h))
                processed_img = self._create_artistic_frame(processed_img, colors, theme, z)
                
                # 旋转
                if layout.get("rotation", 0) != 0:
                    processed_img = processed_img.rotate(
                        layout["rotation"], 
                        expand=True, 
                        fillcolor=(0, 0, 0, 0)
                    )
                
                # 粘贴到封面
                if processed_img.mode == 'RGBA':
                    cover.paste(processed_img, (x, y), processed_img)
                else:
                    cover.paste(processed_img, (x, y))
            
            return cover
            
        except Exception as e:
            logger.warning(f"艺术拼贴创建失败: {str(e)}")
            return cover

    def _create_artistic_frame(self, img: Image.Image, colors: Dict[str, str], theme: str, z_level: int) -> Image.Image:
        """创建艺术相框效果"""
        try:
            width, height = img.size
            
            # 根据层级创建不同的框架效果
            if z_level == 3:  # 主图
                frame_width = max(8, width // 30)
                frame_color = colors["primary"]
            elif z_level == 2:  # 次要图
                frame_width = max(6, width // 40)
                frame_color = colors["secondary"]
            else:  # 装饰图
                frame_width = max(4, width // 50)
                frame_color = colors["accent"]
            
            # 创建带框架的图片
            framed_size = (width + frame_width * 2, height + frame_width * 2)
            framed = Image.new('RGBA', framed_size, (*self._hex_to_rgb(frame_color), 255))
            
            # 粘贴原图
            framed.paste(img, (frame_width, frame_width))
            
            # 添加内阴影效果
            inner_shadow = Image.new('RGBA', framed_size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(inner_shadow)
            
            # 绘制内阴影
            for i in range(frame_width // 2):
                alpha = int(100 * (1 - i / (frame_width // 2)))
                shadow_draw.rectangle([
                    frame_width - i, frame_width - i,
                    width + frame_width + i, height + frame_width + i
                ], outline=(0, 0, 0, alpha))
            
            # 合并内阴影
            framed = Image.alpha_composite(framed, inner_shadow)
            
            # 添加外部阴影
            framed = self._add_3d_effect(framed, colors)
            
            return framed
            
        except Exception as e:
            logger.warning(f"艺术相框创建失败: {str(e)}")
            return img.convert('RGBA')

    def _add_premium_titles(self, cover: Image.Image, title: str, subtitle: str, colors: Dict[str, str], platform: str) -> Image.Image:
        """添加高质量标题"""
        try:
            draw = ImageDraw.Draw(cover)
            cover_width, cover_height = cover.size
            
            # 动态字体大小
            title_font_size = max(48, cover_width // 15)
            subtitle_font_size = max(28, cover_width // 25)
            
            # 尝试加载字体
            try:
                # 尝试系统字体
                import platform
                system = platform.system()
                if system == "Darwin":  # macOS
                    title_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", title_font_size)
                    subtitle_font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", subtitle_font_size)
                elif system == "Windows":
                    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", title_font_size)
                    subtitle_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", subtitle_font_size)
                else:  # Linux
                    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_font_size)
                    subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", subtitle_font_size)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
            
            # 主标题
            if title:
                # 计算标题位置
                title_bbox = draw.textbbox((0, 0), title, font=title_font)
                title_width = title_bbox[2] - title_bbox[0]
                title_height = title_bbox[3] - title_bbox[1]
                
                title_x = (cover_width - title_width) // 2
                title_y = cover_height // 20
                
                # 创建标题背景效果
                bg_padding = title_height // 3
                bg_coords = [
                    title_x - bg_padding, title_y - bg_padding,
                    title_x + title_width + bg_padding, title_y + title_height + bg_padding
                ]
                
                # 渐变背景
                bg_color = (*self._hex_to_rgb(colors["primary"]), 200)
                draw.rounded_rectangle(bg_coords, radius=bg_padding, fill=bg_color)
                
                # 添加标题阴影
                shadow_offset = max(2, title_font_size // 20)
                draw.text((title_x + shadow_offset, title_y + shadow_offset), title, 
                         font=title_font, fill=(0, 0, 0, 100))
                
                # 绘制主标题
                draw.text((title_x, title_y), title, font=title_font, fill="white")
            
            # 副标题
            if subtitle:
                subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
                subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
                
                subtitle_x = (cover_width - subtitle_width) // 2
                subtitle_y = title_y + title_height + cover_height // 30
                
                # 副标题背景
                sub_bg_padding = subtitle_font_size // 4
                draw.rounded_rectangle([
                    subtitle_x - sub_bg_padding, subtitle_y - sub_bg_padding,
                    subtitle_x + subtitle_width + sub_bg_padding, subtitle_y + subtitle_font_size + sub_bg_padding
                ], radius=sub_bg_padding, fill=(*self._hex_to_rgb(colors["secondary"]), 150))
                
                draw.text((subtitle_x, subtitle_y), subtitle, font=subtitle_font, fill="white")
            
            return cover
            
        except Exception as e:
            logger.warning(f"高质量标题添加失败: {str(e)}")
            return cover

    def _add_collage_effects(self, cover: Image.Image, colors: Dict[str, str], theme: str) -> Image.Image:
        """添加拼图装饰效果"""
        try:
            draw = ImageDraw.Draw(cover)
            cover_width, cover_height = cover.size
            
            # 添加拼图连接线效果
            accent_color = (*self._hex_to_rgb(colors["accent"]), 100)
            
            # 在图片之间添加连接线
            line_width = max(2, min(cover_width, cover_height) // 200)
            
            # 水平连接线
            for y in [cover_height // 3, cover_height * 2 // 3]:
                draw.line([(cover_width // 10, y), (cover_width * 9 // 10, y)], 
                         fill=accent_color, width=line_width)
            
            # 垂直连接线
            for x in [cover_width // 3, cover_width * 2 // 3]:
                draw.line([(x, cover_height // 5), (x, cover_height * 4 // 5)], 
                         fill=accent_color, width=line_width)
            
            # 添加装饰性元素
            self._add_decorative_dots(cover, colors, theme)
            
            return cover
            
        except Exception as e:
            logger.warning(f"拼图装饰效果添加失败: {str(e)}")
            return cover

    def _add_decorative_dots(self, cover: Image.Image, colors: Dict[str, str], theme: str) -> Image.Image:
        """添加装饰点"""
        try:
            draw = ImageDraw.Draw(cover)
            cover_width, cover_height = cover.size
            
            # 在角落添加装饰圆点
            dot_size = max(10, min(cover_width, cover_height) // 50)
            accent_color = (*self._hex_to_rgb(colors["accent"]), 150)
            
            # 四个角落的装饰
            corners = [
                (dot_size, dot_size),  # 左上
                (cover_width - dot_size * 3, dot_size),  # 右上
                (dot_size, cover_height - dot_size * 3),  # 左下
                (cover_width - dot_size * 3, cover_height - dot_size * 3)  # 右下
            ]
            
            for x, y in corners:
                draw.ellipse([x, y, x + dot_size * 2, y + dot_size * 2], fill=accent_color)
            
            return cover
            
        except Exception as e:
            logger.warning(f"装饰点添加失败: {str(e)}")
            return cover

    def _create_magazine_collage(self, cover: Image.Image, images: List[Image.Image], colors: Dict[str, str], theme: str) -> Image.Image:
        """创建杂志风格拼贴"""
        try:
            cover_width, cover_height = cover.size
            
            if len(images) == 0:
                return cover
            
            # 杂志风格：主图 + 小图网格
            main_img = images[0]
            
            # 主图占据左侧大部分空间
            main_width = int(cover_width * 0.6)
            main_height = int(cover_height * 0.7)
            main_x = int(cover_width * 0.05)
            main_y = int(cover_height * 0.2)
            
            # 处理主图
            main_processed = self._resize_and_crop(main_img, (main_width, main_height))
            main_processed = self._create_artistic_frame(main_processed, colors, theme, 3)
            
            # 粘贴主图
            if main_processed.mode == 'RGBA':
                cover.paste(main_processed, (main_x, main_y), main_processed)
            else:
                cover.paste(main_processed, (main_x, main_y))
            
            # 处理剩余图片（右侧小图网格）
            remaining_images = images[1:5]  # 最多4张小图
            if remaining_images:
                grid_x = main_x + main_width + int(cover_width * 0.05)
                grid_width = cover_width - grid_x - int(cover_width * 0.05)
                
                # 2x2网格
                cell_size = min(grid_width // 2 - 10, main_height // 2 - 10)
                
                for i, img in enumerate(remaining_images):
                    row = i // 2
                    col = i % 2
                    
                    x = grid_x + col * (cell_size + 10)
                    y = main_y + row * (cell_size + 10)
                    
                    # 处理小图
                    small_img = self._resize_and_crop(img, (cell_size, cell_size))
                    small_img = self._create_artistic_frame(small_img, colors, theme, 1)
                    
                    # 粘贴小图
                    if small_img.mode == 'RGBA':
                        cover.paste(small_img, (x, y), small_img)
                    else:
                        cover.paste(small_img, (x, y))
            
            return cover
            
        except Exception as e:
            logger.warning(f"杂志风格拼贴创建失败: {str(e)}")
            return cover


# 全局实例
_smart_cover_generator = None


def get_smart_cover_generator() -> SmartCoverGenerator:
    """获取智能封面生成器实例"""
    global _smart_cover_generator
    if _smart_cover_generator is None:
        _smart_cover_generator = SmartCoverGenerator()
    return _smart_cover_generator
