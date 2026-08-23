"""
图片智能装饰服务
实现图片添加小图标、文案叠加、滤镜效果等功能
"""

import time
from typing import Dict, List, Any, Tuple
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from pathlib import Path
from core.runtime import OUTPUT_DIR
import io
import base64


class ImageDecorator:
    """图片装饰服务"""
    
    def __init__(self):
        self.icons_path = Path("assets/icons")
        self.fonts_path = Path("assets/fonts")
        self.templates_path = Path("assets/templates")
        
        # 确保资源目录存在
        self.icons_path.mkdir(parents=True, exist_ok=True)
        self.fonts_path.mkdir(parents=True, exist_ok=True)
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # 预设颜色方案
        self.color_schemes = {
            "warm": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            "cool": ["#74B9FF", "#0984E3", "#6C5CE7", "#A29BFE", "#FD79A8"],
            "nature": ["#00B894", "#00CEC9", "#6C5CE7", "#FDCB6E", "#E17055"],
            "elegant": ["#2D3436", "#636E72", "#DDD", "#74B9FF", "#FD79A8"],
            "vibrant": ["#FF3838", "#FF9F43", "#10AC84", "#5F27CD", "#00D2D3"]
        }
        
        # 预设文案样式
        self.text_styles = {
            "title": {"size": 48, "weight": "bold", "color": "#2D3436"},
            "subtitle": {"size": 32, "weight": "normal", "color": "#636E72"},
            "caption": {"size": 24, "weight": "normal", "color": "#74B9FF"},
            "tag": {"size": 20, "weight": "bold", "color": "#FFFFFF", "bg": "#FF6B6B"},
            "watermark": {"size": 16, "weight": "normal", "color": "#DDD", "opacity": 0.7}
        }
        
        # 预设图标类型
        self.icon_types = {
            "location": "📍", "heart": "❤️", "star": "⭐", "fire": "🔥",
            "camera": "📷", "food": "🍽️", "travel": "✈️", "shopping": "🛍️",
            "beauty": "💄", "fashion": "👗", "fitness": "💪", "coffee": "☕"
        }
    
    def decorate_image(
        self,
        image_path: str,
        decorations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        装饰图片
        
        Args:
            image_path: 图片路径
            decorations: 装饰配置
            
        Returns:
            Dict: 装饰结果
        """
        try:
            # 加载图片
            image = Image.open(image_path)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # 创建绘图对象
            draw = ImageDraw.Draw(image)
            
            # 应用滤镜
            if decorations.get("filter"):
                image = self._apply_filter(image, decorations["filter"])
                draw = ImageDraw.Draw(image)
            
            # 添加文案
            if decorations.get("texts"):
                for text_config in decorations["texts"]:
                    self._add_text(draw, image.size, text_config)
            
            # 添加图标
            if decorations.get("icons"):
                for icon_config in decorations["icons"]:
                    self._add_icon(draw, image.size, icon_config)
            
            # 添加装饰元素
            if decorations.get("elements"):
                for element_config in decorations["elements"]:
                    self._add_element(draw, image.size, element_config)
            
            # 保存装饰后的图片
            output_path = str(OUTPUT_DIR / f"decorated_{int(time.time())}.png")
            image.save(output_path, "PNG")
            
            # 转换为base64用于前端预览
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info(f"图片装饰完成: {image_path} -> {output_path}")
            return {
                "status": "success",
                "data": {
                    "decorated_path": output_path,
                    "preview_base64": f"data:image/png;base64,{image_base64}",
                    "width": image.size[0],
                    "height": image.size[1]
                },
                "message": "装饰完成"
            }
            
        except Exception as e:
            logger.error(f"图片装饰失败: {str(e)}")
            return {
                "status": "error",
                "message": f"装饰失败: {str(e)}"
            }
    
    def _apply_filter(self, image: Image.Image, filter_type: str) -> Image.Image:
        """应用滤镜效果"""
        try:
            if filter_type == "blur":
                return image.filter(ImageFilter.GaussianBlur(radius=2))
            elif filter_type == "sharpen":
                return image.filter(ImageFilter.SHARPEN)
            elif filter_type == "vintage":
                # 复古效果
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(0.8)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                return image
            elif filter_type == "bright":
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(1.2)
            elif filter_type == "warm":
                # 暖色调
                r, g, b, a = image.split()
                r = ImageEnhance.Brightness(r).enhance(1.1)
                g = ImageEnhance.Brightness(g).enhance(1.05)
                return Image.merge('RGBA', (r, g, b, a))
            else:
                return image
        except Exception as e:
            logger.warning(f"滤镜应用失败: {str(e)}")
            return image
    
    def _add_text(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], text_config: Dict[str, Any]):
        """添加文案"""
        try:
            text = text_config.get("text", "")
            position = text_config.get("position", "center")
            style = text_config.get("style", "title")
            
            # 获取样式配置
            style_config = self.text_styles.get(style, self.text_styles["title"])
            font_size = text_config.get("size", style_config["size"])
            color = text_config.get("color", style_config["color"])
            
            # 尝试加载字体（如果没有则使用默认字体）
            try:
                font = ImageFont.truetype("assets/fonts/default.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # 计算文本位置
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if position == "center":
                x = (image_size[0] - text_width) // 2
                y = (image_size[1] - text_height) // 2
            elif position == "top":
                x = (image_size[0] - text_width) // 2
                y = 50
            elif position == "bottom":
                x = (image_size[0] - text_width) // 2
                y = image_size[1] - text_height - 50
            elif position == "top-left":
                x, y = 50, 50
            elif position == "top-right":
                x = image_size[0] - text_width - 50
                y = 50
            elif position == "bottom-left":
                x = 50
                y = image_size[1] - text_height - 50
            elif position == "bottom-right":
                x = image_size[0] - text_width - 50
                y = image_size[1] - text_height - 50
            else:
                x, y = position if isinstance(position, tuple) else (100, 100)
            
            # 添加背景（如果配置了）
            if style_config.get("bg"):
                bg_color = style_config["bg"]
                padding = 20
                draw.rectangle([
                    x - padding, y - padding,
                    x + text_width + padding, y + text_height + padding
                ], fill=bg_color)
            
            # 绘制文本
            draw.text((x, y), text, font=font, fill=color)
            
        except Exception as e:
            logger.warning(f"文案添加失败: {str(e)}")
    
    def _add_icon(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], icon_config: Dict[str, Any]):
        """添加图标"""
        try:
            icon_type = icon_config.get("type", "heart")
            position = icon_config.get("position", "top-right")
            size = icon_config.get("size", 40)
            
            # 获取图标字符
            icon_char = self.icon_types.get(icon_type, "❤️")
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("assets/fonts/emoji.ttf", size)
            except:
                font = ImageFont.load_default()
            
            # 计算位置
            if position == "top-right":
                x, y = image_size[0] - size - 20, 20
            elif position == "top-left":
                x, y = 20, 20
            elif position == "bottom-right":
                x, y = image_size[0] - size - 20, image_size[1] - size - 20
            elif position == "bottom-left":
                x, y = 20, image_size[1] - size - 20
            else:
                x, y = position if isinstance(position, tuple) else (20, 20)
            
            # 绘制图标
            draw.text((x, y), icon_char, font=font, fill="#FF6B6B")
            
        except Exception as e:
            logger.warning(f"图标添加失败: {str(e)}")
    
    def _add_element(self, draw: ImageDraw.Draw, image_size: Tuple[int, int], element_config: Dict[str, Any]):
        """添加装饰元素"""
        try:
            element_type = element_config.get("type", "border")
            
            if element_type == "border":
                # 添加边框
                color = element_config.get("color", "#FF6B6B")
                width = element_config.get("width", 5)
                draw.rectangle([0, 0, image_size[0]-1, image_size[1]-1], outline=color, width=width)
                
            elif element_type == "corner":
                # 添加圆角装饰
                color = element_config.get("color", "#74B9FF")
                size = element_config.get("size", 50)
                positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
                for pos in positions:
                    if pos == "top-left":
                        draw.ellipse([0, 0, size, size], fill=color)
                    elif pos == "top-right":
                        draw.ellipse([image_size[0]-size, 0, image_size[0], size], fill=color)
                    elif pos == "bottom-left":
                        draw.ellipse([0, image_size[1]-size, size, image_size[1]], fill=color)
                    elif pos == "bottom-right":
                        draw.ellipse([image_size[0]-size, image_size[1]-size, image_size[0], image_size[1]], fill=color)
                        
            elif element_type == "gradient":
                # 添加渐变遮罩
                overlay = Image.new('RGBA', image_size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # 简单的线性渐变效果
                for i in range(image_size[1]):
                    alpha = int(255 * (i / image_size[1]) * 0.3)
                    overlay_draw.line([(0, i), (image_size[0], i)], fill=(0, 0, 0, alpha))
                
                # 合并到原图
                base_image = Image.new('RGBA', image_size, (255, 255, 255, 0))
                base_image.paste(overlay, (0, 0), overlay)
                
        except Exception as e:
            logger.warning(f"装饰元素添加失败: {str(e)}")
    
    def generate_smart_decorations(
        self,
        theme: str,
        content_type: str,
        mood: str = "vibrant"
    ) -> Dict[str, Any]:
        """
        智能生成装饰配置
        
        Args:
            theme: 主题
            content_type: 内容类型
            mood: 情绪风格
            
        Returns:
            Dict: 装饰配置
        """
        try:
            # 根据主题和类型生成智能装饰
            decorations = {
                "filter": self._get_smart_filter(content_type, mood),
                "texts": self._get_smart_texts(theme, content_type),
                "icons": self._get_smart_icons(content_type),
                "elements": self._get_smart_elements(mood)
            }
            
            return {
                "status": "success",
                "data": decorations,
                "message": "智能装饰生成完成"
            }
            
        except Exception as e:
            logger.error(f"智能装饰生成失败: {str(e)}")
            return {
                "status": "error",
                "message": f"生成失败: {str(e)}"
            }
    
    def _get_smart_filter(self, content_type: str, mood: str) -> str:
        """获取智能滤镜"""
        filter_map = {
            "travel": {"vibrant": "bright", "elegant": "vintage", "warm": "warm"},
            "food": {"vibrant": "warm", "elegant": "vintage", "cool": "bright"},
            "lifestyle": {"vibrant": "bright", "elegant": "vintage", "warm": "warm"},
            "fashion": {"vibrant": "sharpen", "elegant": "vintage", "cool": "bright"}
        }
        return filter_map.get(content_type, {}).get(mood, "bright")
    
    def _get_smart_texts(self, theme: str, content_type: str) -> List[Dict[str, Any]]:
        """获取智能文案"""
        texts = []
        
        # 主标题
        texts.append({
            "text": f"✨ {theme}",
            "position": "top",
            "style": "title"
        })
        
        # 类型标签
        type_labels = {
            "travel": "🌍 旅行分享",
            "food": "🍽️ 美食探店",
            "lifestyle": "💫 生活记录",
            "fashion": "👗 时尚穿搭"
        }
        
        if content_type in type_labels:
            texts.append({
                "text": type_labels[content_type],
                "position": "bottom-left",
                "style": "tag"
            })
        
        return texts
    
    def _get_smart_icons(self, content_type: str) -> List[Dict[str, Any]]:
        """获取智能图标"""
        icon_map = {
            "travel": [{"type": "location", "position": "top-right"}],
            "food": [{"type": "food", "position": "top-right"}],
            "lifestyle": [{"type": "heart", "position": "top-right"}],
            "fashion": [{"type": "fashion", "position": "top-right"}]
        }
        return icon_map.get(content_type, [{"type": "heart", "position": "top-right"}])
    
    def _get_smart_elements(self, mood: str) -> List[Dict[str, Any]]:
        """获取智能装饰元素"""
        element_map = {
            "vibrant": [{"type": "border", "color": "#FF6B6B", "width": 8}],
            "elegant": [{"type": "corner", "color": "#74B9FF", "size": 30}],
            "warm": [{"type": "gradient"}],
            "cool": [{"type": "border", "color": "#74B9FF", "width": 5}]
        }
        return element_map.get(mood, [])


# 全局实例
_image_decorator = None


def get_image_decorator() -> ImageDecorator:
    """获取图片装饰服务实例"""
    global _image_decorator
    if _image_decorator is None:
        _image_decorator = ImageDecorator()
    return _image_decorator
