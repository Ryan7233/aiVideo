"""
智能封面设计服务
帧选 + 文案叠字模板（自动取主色、加描边）
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import colorsys
from collections import Counter
import subprocess
import tempfile

logger = logging.getLogger(__name__)

# 尝试导入PIL用于图像处理
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL不可用，封面生成功能将受限")


class SmartCoverDesigner:
    """智能封面设计器"""
    
    def __init__(self):
        # 封面设计配置
        self.cover_config = {
            'target_size': (1080, 1920),  # 9:16 竖屏
            'safe_area': {
                'top': 0.1,      # 顶部安全区域
                'bottom': 0.15,  # 底部安全区域
                'left': 0.05,    # 左侧安全区域
                'right': 0.05    # 右侧安全区域
            },
            'text_zones': {
                'title': {'y_range': (0.15, 0.35), 'max_lines': 2},
                'subtitle': {'y_range': (0.75, 0.85), 'max_lines': 1}
            }
        }
        
        # 字体配置
        self.font_config = {
            'title': {
                'size_range': (60, 120),
                'weight': 'bold',
                'preferred_fonts': ['PingFang SC', 'Helvetica', 'Arial']
            },
            'subtitle': {
                'size_range': (32, 48),
                'weight': 'normal',
                'preferred_fonts': ['PingFang SC', 'Helvetica', 'Arial']
            }
        }
        
        # 色彩方案模板
        self.color_schemes = {
            'warm': {
                'primary': (255, 87, 34),    # 橙色
                'secondary': (255, 193, 7),  # 黄色
                'accent': (255, 255, 255),   # 白色
                'shadow': (0, 0, 0, 128)     # 半透明黑色
            },
            'cool': {
                'primary': (33, 150, 243),   # 蓝色
                'secondary': (76, 175, 80),  # 绿色
                'accent': (255, 255, 255),   # 白色
                'shadow': (0, 0, 0, 128)
            },
            'nature': {
                'primary': (76, 175, 80),    # 绿色
                'secondary': (139, 195, 74), # 浅绿色
                'accent': (255, 255, 255),   # 白色
                'shadow': (0, 0, 0, 128)
            },
            'elegant': {
                'primary': (96, 125, 139),   # 蓝灰色
                'secondary': (158, 158, 158), # 灰色
                'accent': (255, 255, 255),   # 白色
                'shadow': (0, 0, 0, 128)
            }
        }
        
        # 设计模板
        self.design_templates = {
            'minimal': {
                'background_overlay': 0.3,
                'text_background': True,
                'gradient_overlay': False,
                'decorative_elements': False
            },
            'vibrant': {
                'background_overlay': 0.5,
                'text_background': True,
                'gradient_overlay': True,
                'decorative_elements': True
            },
            'clean': {
                'background_overlay': 0.2,
                'text_background': False,
                'gradient_overlay': False,
                'decorative_elements': False
            }
        }
    
    def generate_smart_cover(self, clips: List[Dict], photos: List[Dict], 
                           title: str, style: str = '治愈') -> Dict:
        """
        生成智能封面
        
        Args:
            clips: 视频片段列表
            photos: 照片列表
            title: 标题文本
            style: 风格类型
            
        Returns:
            封面设计结果
        """
        try:
            logger.info(f"开始生成智能封面 - 标题: {title[:20]}..., 风格: {style}")
            
            # 1. 选择最佳帧或照片
            best_frame_info = self._select_best_frame(clips, photos)
            
            # 2. 分析图像特征
            image_analysis = self._analyze_image_features(best_frame_info)
            
            # 3. 确定设计方案
            design_scheme = self._determine_design_scheme(image_analysis, style)
            
            # 4. 提取主色调
            color_palette = self._extract_color_palette(best_frame_info, design_scheme)
            
            # 5. 设计文案布局
            text_layout = self._design_text_layout(title, color_palette, design_scheme)
            
            # 6. 生成封面图像
            cover_result = self._generate_cover_image(
                best_frame_info, text_layout, color_palette, design_scheme
            )
            
            logger.info("智能封面生成完成")
            return cover_result
            
        except Exception as e:
            logger.error(f"智能封面生成失败: {e}")
            return self._generate_fallback_cover(clips, title)
    
    def _select_best_frame(self, clips: List[Dict], photos: List[Dict]) -> Dict:
        """选择最佳帧或照片"""
        try:
            candidates = []
            
            # 从视频片段中提取关键帧
            for clip in clips:
                video_path = clip.get('output_path', '')
                if Path(video_path).exists():
                    frames = self._extract_key_frames(video_path, clip)
                    candidates.extend(frames)
            
            # 添加照片候选
            for photo in photos:
                if Path(photo.get('path', '')).exists():
                    candidates.append({
                        'type': 'photo',
                        'path': photo['path'],
                        'score': photo.get('final_score', 0.5),
                        'source': 'photo_ranking'
                    })
            
            if not candidates:
                logger.warning("没有找到有效的封面候选")
                return {}
            
            # 评估候选帧/照片
            scored_candidates = []
            for candidate in candidates:
                score = self._evaluate_cover_candidate(candidate)
                candidate['cover_score'] = score
                scored_candidates.append(candidate)
            
            # 选择最佳候选
            best_candidate = max(scored_candidates, key=lambda x: x['cover_score'])
            
            logger.info(f"选择最佳封面素材: {best_candidate['type']}, 分数: {best_candidate['cover_score']:.2f}")
            return best_candidate
            
        except Exception as e:
            logger.error(f"选择最佳帧失败: {e}")
            return {}
    
    def _extract_key_frames(self, video_path: str, clip_info: Dict) -> List[Dict]:
        """从视频中提取关键帧"""
        try:
            frames = []
            
            # 在视频的几个关键位置提取帧
            duration = clip_info.get('duration', 10)
            positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # 相对位置
            
            for i, pos in enumerate(positions):
                timestamp = duration * pos
                frame_path = self._extract_frame_at_time(video_path, timestamp, i)
                
                if frame_path and Path(frame_path).exists():
                    frames.append({
                        'type': 'video_frame',
                        'path': frame_path,
                        'timestamp': timestamp,
                        'position': pos,
                        'source_video': video_path,
                        'source': 'video_extraction'
                    })
            
            return frames
            
        except Exception as e:
            logger.error(f"提取关键帧失败: {e}")
            return []
    
    def _extract_frame_at_time(self, video_path: str, timestamp: float, index: int) -> Optional[str]:
        """在指定时间提取视频帧"""
        try:
            output_dir = Path("output_data/cover_frames")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            frame_filename = f"frame_{Path(video_path).stem}_{index:02d}.jpg"
            frame_path = output_dir / frame_filename
            
            # 使用FFmpeg提取帧
            cmd = [
                "ffmpeg", "-y", "-ss", f"{timestamp:.2f}",
                "-i", video_path, "-vframes", "1",
                "-q:v", "2", str(frame_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and frame_path.exists():
                return str(frame_path)
            else:
                logger.warning(f"帧提取失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"提取视频帧失败: {e}")
            return None
    
    def _evaluate_cover_candidate(self, candidate: Dict) -> float:
        """评估封面候选的质量"""
        try:
            if not PIL_AVAILABLE:
                return candidate.get('score', 0.5)
            
            image_path = candidate['path']
            image = Image.open(image_path)
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 基础评分
            base_score = candidate.get('score', 0.5)
            
            # 尺寸适配性评分
            width, height = image.size
            aspect_ratio = width / height
            target_ratio = self.cover_config['target_size'][0] / self.cover_config['target_size'][1]
            
            ratio_score = 1.0 - abs(aspect_ratio - target_ratio) / target_ratio
            ratio_score = max(0, ratio_score)
            
            # 图像质量评分
            quality_score = self._evaluate_image_quality(image)
            
            # 文本区域适宜性评分
            text_area_score = self._evaluate_text_area_suitability(image)
            
            # 综合评分
            final_score = (
                base_score * 0.4 +
                ratio_score * 0.2 +
                quality_score * 0.2 +
                text_area_score * 0.2
            )
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"评估封面候选失败: {e}")
            return candidate.get('score', 0.5)
    
    def _evaluate_image_quality(self, image: Image.Image) -> float:
        """评估图像质量"""
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 计算清晰度（基于梯度）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)
            
            # 计算亮度分布
            brightness = gray.mean()
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # 计算对比度
            contrast = gray.std()
            contrast_score = min(contrast / 80, 1.0)
            
            quality_score = (sharpness_score * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)
            return max(0, min(quality_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _evaluate_text_area_suitability(self, image: Image.Image) -> float:
        """评估文本区域的适宜性"""
        try:
            width, height = image.size
            img_array = np.array(image)
            
            # 检查标题区域的复杂度
            title_zone = self.cover_config['text_zones']['title']['y_range']
            title_y_start = int(height * title_zone[0])
            title_y_end = int(height * title_zone[1])
            
            title_region = img_array[title_y_start:title_y_end, :]
            
            # 计算区域的视觉复杂度
            gray_region = cv2.cvtColor(title_region, cv2.COLOR_RGB2GRAY)
            complexity = cv2.Laplacian(gray_region, cv2.CV_64F).var()
            
            # 复杂度适中最适合放文字
            optimal_complexity = 500
            complexity_score = 1.0 - abs(complexity - optimal_complexity) / optimal_complexity
            complexity_score = max(0, complexity_score)
            
            # 检查亮度均匀性
            brightness_std = gray_region.std()
            uniformity_score = max(0, 1.0 - brightness_std / 50)
            
            suitability_score = complexity_score * 0.6 + uniformity_score * 0.4
            return max(0, min(suitability_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _analyze_image_features(self, frame_info: Dict) -> Dict:
        """分析图像特征"""
        try:
            if not frame_info or not PIL_AVAILABLE:
                return {}
            
            image_path = frame_info['path']
            image = Image.open(image_path)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 提取主要颜色
            dominant_colors = self._extract_dominant_colors(image)
            
            # 分析亮度分布
            brightness_analysis = self._analyze_brightness_distribution(image)
            
            # 检测图像内容类型
            content_type = self._detect_content_type(image)
            
            # 分析构图特点
            composition_analysis = self._analyze_composition(image)
            
            return {
                'dominant_colors': dominant_colors,
                'brightness': brightness_analysis,
                'content_type': content_type,
                'composition': composition_analysis,
                'image_size': image.size
            }
            
        except Exception as e:
            logger.error(f"图像特征分析失败: {e}")
            return {}
    
    def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """提取主要颜色"""
        try:
            # 缩小图像以提高处理速度
            image_small = image.resize((150, 150))
            
            # 转换为numpy数组
            img_array = np.array(image_small)
            pixels = img_array.reshape(-1, 3)
            
            # 使用K-means聚类找出主要颜色
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # 按出现频率排序
            labels = kmeans.labels_
            label_counts = Counter(labels)
            
            sorted_colors = []
            for label, count in label_counts.most_common():
                color = tuple(colors[label])
                sorted_colors.append(color)
            
            return sorted_colors
            
        except Exception as e:
            logger.warning(f"主要颜色提取失败，使用默认颜色: {e}")
            return [(128, 128, 128), (64, 64, 64), (192, 192, 192)]
    
    def _analyze_brightness_distribution(self, image: Image.Image) -> Dict:
        """分析亮度分布"""
        try:
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            return {
                'mean': float(img_array.mean()),
                'std': float(img_array.std()),
                'min': int(img_array.min()),
                'max': int(img_array.max()),
                'is_dark': img_array.mean() < 80,
                'is_bright': img_array.mean() > 180,
                'high_contrast': img_array.std() > 60
            }
            
        except Exception:
            return {'mean': 128, 'std': 50, 'is_dark': False, 'is_bright': False, 'high_contrast': False}
    
    def _detect_content_type(self, image: Image.Image) -> str:
        """检测图像内容类型"""
        try:
            # 简单的内容类型检测
            img_array = np.array(image)
            
            # 检测主要颜色分布
            dominant_colors = self._extract_dominant_colors(image, 3)
            
            # 基于颜色判断内容类型
            green_score = sum(1 for r, g, b in dominant_colors if g > r and g > b)
            blue_score = sum(1 for r, g, b in dominant_colors if b > r and b > g)
            warm_score = sum(1 for r, g, b in dominant_colors if r > 150 and (r - g) > 30)
            
            if green_score >= 2:
                return 'nature'
            elif blue_score >= 2:
                return 'sky_water'
            elif warm_score >= 2:
                return 'warm_tones'
            else:
                return 'general'
                
        except Exception:
            return 'general'
    
    def _analyze_composition(self, image: Image.Image) -> Dict:
        """分析构图特点"""
        try:
            width, height = image.size
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            
            # 分析边缘分布
            edge_density_top = np.sum(edges[:height//3, :]) / (width * height // 3)
            edge_density_middle = np.sum(edges[height//3:2*height//3, :]) / (width * height // 3)
            edge_density_bottom = np.sum(edges[2*height//3:, :]) / (width * height // 3)
            
            return {
                'edge_distribution': {
                    'top': float(edge_density_top),
                    'middle': float(edge_density_middle),
                    'bottom': float(edge_density_bottom)
                },
                'main_subject_area': 'top' if edge_density_top > edge_density_middle else 'middle',
                'complexity': float(np.sum(edges) / (width * height))
            }
            
        except Exception:
            return {'edge_distribution': {'top': 0.1, 'middle': 0.1, 'bottom': 0.1}, 'main_subject_area': 'middle', 'complexity': 0.1}
    
    def _determine_design_scheme(self, image_analysis: Dict, style: str) -> Dict:
        """确定设计方案"""
        try:
            # 基于风格选择基础方案
            style_mapping = {
                '治愈': 'minimal',
                '专业': 'clean',
                '踩雷': 'vibrant'
            }
            
            base_template = style_mapping.get(style, 'minimal')
            design_scheme = self.design_templates[base_template].copy()
            
            # 基于图像特征调整
            if image_analysis:
                brightness = image_analysis.get('brightness', {})
                content_type = image_analysis.get('content_type', 'general')
                
                # 调整背景遮罩
                if brightness.get('is_dark', False):
                    design_scheme['background_overlay'] = max(0.1, design_scheme['background_overlay'] - 0.2)
                elif brightness.get('is_bright', False):
                    design_scheme['background_overlay'] = min(0.6, design_scheme['background_overlay'] + 0.2)
                
                # 基于内容类型调整
                if content_type == 'nature':
                    design_scheme['gradient_overlay'] = True
                    design_scheme['color_scheme'] = 'nature'
                elif content_type == 'sky_water':
                    design_scheme['color_scheme'] = 'cool'
                elif content_type == 'warm_tones':
                    design_scheme['color_scheme'] = 'warm'
                else:
                    design_scheme['color_scheme'] = 'elegant'
            
            return design_scheme
            
        except Exception as e:
            logger.error(f"设计方案确定失败: {e}")
            return self.design_templates['minimal']
    
    def _extract_color_palette(self, frame_info: Dict, design_scheme: Dict) -> Dict:
        """提取色彩方案"""
        try:
            # 获取预设色彩方案
            scheme_name = design_scheme.get('color_scheme', 'elegant')
            base_colors = self.color_schemes[scheme_name].copy()
            
            # 如果有图像，基于图像调整颜色
            if frame_info and PIL_AVAILABLE:
                image = Image.open(frame_info['path'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                dominant_colors = self._extract_dominant_colors(image, 3)
                
                if dominant_colors:
                    # 使用最主要的颜色作为主色调
                    main_color = dominant_colors[0]
                    
                    # 调整色彩饱和度和亮度
                    adjusted_color = self._adjust_color_for_text(main_color)
                    base_colors['primary'] = adjusted_color
                    
                    # 生成互补色作为辅助色
                    complement_color = self._generate_complement_color(main_color)
                    base_colors['secondary'] = complement_color
            
            return base_colors
            
        except Exception as e:
            logger.error(f"色彩方案提取失败: {e}")
            return self.color_schemes['elegant']
    
    def _adjust_color_for_text(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """调整颜色以适合文本显示"""
        try:
            r, g, b = color
            
            # 转换到HSV色彩空间
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            # 增加饱和度和亮度以提高可读性
            s = min(1.0, s * 1.2)
            v = max(0.4, min(0.8, v))
            
            # 转换回RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            
            return (int(r * 255), int(g * 255), int(b * 255))
            
        except Exception:
            return color
    
    def _generate_complement_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """生成互补色"""
        try:
            r, g, b = color
            
            # 转换到HSV色彩空间
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            # 生成互补色（色相偏移180度）
            complement_h = (h + 0.5) % 1.0
            
            # 调整饱和度和亮度
            complement_s = max(0.3, s * 0.8)
            complement_v = max(0.3, v * 0.9)
            
            # 转换回RGB
            r, g, b = colorsys.hsv_to_rgb(complement_h, complement_s, complement_v)
            
            return (int(r * 255), int(g * 255), int(b * 255))
            
        except Exception:
            return (128, 128, 128)
    
    def _design_text_layout(self, title: str, color_palette: Dict, design_scheme: Dict) -> Dict:
        """设计文案布局"""
        try:
            # 分析标题长度和内容
            title_length = len(title)
            
            # 确定字体大小
            if title_length <= 8:
                font_size = 100
                max_lines = 1
            elif title_length <= 16:
                font_size = 80
                max_lines = 2
            else:
                font_size = 60
                max_lines = 2
            
            # 确定文本颜色
            text_color = self._determine_text_color(color_palette, design_scheme)
            
            # 确定描边和背景
            stroke_color, stroke_width = self._determine_stroke(text_color, color_palette)
            background_config = self._determine_text_background(design_scheme, color_palette)
            
            return {
                'title': {
                    'text': title,
                    'font_size': font_size,
                    'max_lines': max_lines,
                    'color': text_color,
                    'stroke_color': stroke_color,
                    'stroke_width': stroke_width,
                    'position': 'center_top',
                    'background': background_config
                }
            }
            
        except Exception as e:
            logger.error(f"文案布局设计失败: {e}")
            return {
                'title': {
                    'text': title,
                    'font_size': 80,
                    'max_lines': 2,
                    'color': (255, 255, 255),
                    'stroke_color': (0, 0, 0),
                    'stroke_width': 3,
                    'position': 'center_top'
                }
            }
    
    def _determine_text_color(self, color_palette: Dict, design_scheme: Dict) -> Tuple[int, int, int]:
        """确定文本颜色"""
        # 优先使用白色，在浅色背景上使用深色
        primary_color = color_palette.get('primary', (128, 128, 128))
        
        # 计算亮度
        r, g, b = primary_color
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        
        if brightness > 150:
            return (40, 40, 40)  # 深色文本
        else:
            return (255, 255, 255)  # 白色文本
    
    def _determine_stroke(self, text_color: Tuple[int, int, int], color_palette: Dict) -> Tuple[Tuple[int, int, int], int]:
        """确定描边颜色和宽度"""
        if text_color == (255, 255, 255):
            # 白色文本用黑色描边
            return (0, 0, 0), 4
        else:
            # 深色文本用白色描边
            return (255, 255, 255), 3
    
    def _determine_text_background(self, design_scheme: Dict, color_palette: Dict) -> Dict:
        """确定文本背景"""
        if design_scheme.get('text_background', False):
            bg_color = color_palette.get('shadow', (0, 0, 0, 128))
            return {
                'enabled': True,
                'color': bg_color,
                'padding': 20,
                'border_radius': 10
            }
        else:
            return {'enabled': False}
    
    def _generate_cover_image(self, frame_info: Dict, text_layout: Dict, 
                            color_palette: Dict, design_scheme: Dict) -> Dict:
        """生成最终封面图像"""
        try:
            if not frame_info or not PIL_AVAILABLE:
                return self._generate_text_only_cover(text_layout, color_palette)
            
            # 加载背景图像
            background_image = Image.open(frame_info['path'])
            if background_image.mode != 'RGB':
                background_image = background_image.convert('RGB')
            
            # 调整到目标尺寸
            target_size = self.cover_config['target_size']
            cover_image = self._resize_and_crop_image(background_image, target_size)
            
            # 应用背景效果
            cover_image = self._apply_background_effects(cover_image, design_scheme, color_palette)
            
            # 添加文本
            cover_image = self._add_text_to_cover(cover_image, text_layout)
            
            # 保存封面
            output_path = self._save_cover_image(cover_image, frame_info)
            
            return {
                'cover_path': output_path,
                'source_frame': frame_info,
                'design_scheme': design_scheme,
                'color_palette': color_palette,
                'text_layout': text_layout,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"封面图像生成失败: {e}")
            return self._generate_fallback_cover([], text_layout['title']['text'])
    
    def _resize_and_crop_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """调整图像尺寸并裁剪"""
        try:
            target_width, target_height = target_size
            target_ratio = target_width / target_height
            
            original_width, original_height = image.size
            original_ratio = original_width / original_height
            
            if original_ratio > target_ratio:
                # 原图更宽，按高度缩放
                new_height = target_height
                new_width = int(original_width * target_height / original_height)
            else:
                # 原图更高，按宽度缩放
                new_width = target_width
                new_height = int(original_height * target_width / original_width)
            
            # 缩放图像
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 居中裁剪
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            cropped_image = resized_image.crop((left, top, right, bottom))
            
            return cropped_image
            
        except Exception as e:
            logger.error(f"图像尺寸调整失败: {e}")
            return image
    
    def _apply_background_effects(self, image: Image.Image, design_scheme: Dict, color_palette: Dict) -> Image.Image:
        """应用背景效果"""
        try:
            result_image = image.copy()
            
            # 应用遮罩
            overlay_alpha = design_scheme.get('background_overlay', 0.3)
            if overlay_alpha > 0:
                overlay = Image.new('RGBA', image.size, (0, 0, 0, int(255 * overlay_alpha)))
                result_image = Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
            
            # 应用渐变遮罩
            if design_scheme.get('gradient_overlay', False):
                gradient = self._create_gradient_overlay(image.size, color_palette)
                result_image = Image.alpha_composite(result_image.convert('RGBA'), gradient).convert('RGB')
            
            # 应用模糊效果（仅在文本区域）
            if design_scheme.get('blur_text_areas', False):
                result_image = self._apply_selective_blur(result_image)
            
            return result_image
            
        except Exception as e:
            logger.error(f"背景效果应用失败: {e}")
            return image
    
    def _create_gradient_overlay(self, size: Tuple[int, int], color_palette: Dict) -> Image.Image:
        """创建渐变遮罩"""
        try:
            width, height = size
            gradient = Image.new('RGBA', size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(gradient)
            
            primary_color = color_palette.get('primary', (128, 128, 128))
            
            # 创建从上到下的渐变
            for y in range(height):
                alpha = int(100 * (y / height) * 0.5)  # 最大50%透明度
                color = (*primary_color, alpha)
                draw.line([(0, y), (width, y)], fill=color)
            
            return gradient
            
        except Exception:
            return Image.new('RGBA', size, (0, 0, 0, 0))
    
    def _apply_selective_blur(self, image: Image.Image) -> Image.Image:
        """应用选择性模糊"""
        try:
            # 在标题区域应用轻微模糊
            height = image.size[1]
            title_zone = self.cover_config['text_zones']['title']['y_range']
            
            title_start = int(height * title_zone[0])
            title_end = int(height * title_zone[1])
            
            # 提取标题区域
            title_region = image.crop((0, title_start, image.size[0], title_end))
            
            # 应用模糊
            blurred_region = title_region.filter(ImageFilter.GaussianBlur(radius=2))
            
            # 合并回原图
            result = image.copy()
            result.paste(blurred_region, (0, title_start))
            
            return result
            
        except Exception:
            return image
    
    def _add_text_to_cover(self, image: Image.Image, text_layout: Dict) -> Image.Image:
        """在封面上添加文本"""
        try:
            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)
            
            title_config = text_layout['title']
            
            # 尝试加载字体
            font = self._load_font(title_config['font_size'])
            
            # 处理文本换行
            text_lines = self._wrap_text(title_config['text'], font, image.size[0] * 0.9, title_config['max_lines'])
            
            # 计算文本位置
            text_position = self._calculate_text_position(image.size, text_lines, font, title_config['position'])
            
            # 绘制文本背景（如果需要）
            if title_config.get('background', {}).get('enabled', False):
                self._draw_text_background(draw, text_lines, text_position, font, title_config['background'])
            
            # 绘制文本（带描边）
            self._draw_text_with_stroke(
                draw, text_lines, text_position, font,
                title_config['color'], title_config['stroke_color'], title_config['stroke_width']
            )
            
            return result_image
            
        except Exception as e:
            logger.error(f"文本添加失败: {e}")
            return image
    
    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """加载字体"""
        try:
            # 尝试加载系统字体
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:/Windows/Fonts/simhei.ttf",  # Windows
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux
            ]
            
            for font_path in font_paths:
                try:
                    if Path(font_path).exists():
                        return ImageFont.truetype(font_path, size)
                except Exception:
                    continue
            
            # 回退到默认字体
            return ImageFont.load_default()
            
        except Exception:
            return ImageFont.load_default()
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: float, max_lines: int) -> List[str]:
        """文本换行"""
        try:
            if max_lines == 1:
                return [text]
            
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
                
                if text_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        lines.append(word)
                    
                    if len(lines) >= max_lines:
                        break
            
            if current_line and len(lines) < max_lines:
                lines.append(current_line)
            
            return lines[:max_lines]
            
        except Exception:
            return [text]
    
    def _calculate_text_position(self, image_size: Tuple[int, int], text_lines: List[str], 
                               font: ImageFont.ImageFont, position: str) -> List[Tuple[int, int]]:
        """计算文本位置"""
        try:
            width, height = image_size
            positions = []
            
            # 计算每行文本的尺寸
            line_heights = []
            line_widths = []
            
            for line in text_lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                line_widths.append(line_width)
                line_heights.append(line_height)
            
            total_text_height = sum(line_heights) + (len(text_lines) - 1) * 10  # 行间距
            
            # 确定起始Y位置
            if position == 'center_top':
                title_zone = self.cover_config['text_zones']['title']['y_range']
                zone_center_y = height * (title_zone[0] + title_zone[1]) / 2
                start_y = int(zone_center_y - total_text_height / 2)
            else:
                start_y = int(height * 0.2)
            
            # 计算每行位置
            current_y = start_y
            for i, (line_width, line_height) in enumerate(zip(line_widths, line_heights)):
                x = int((width - line_width) / 2)  # 居中
                positions.append((x, current_y))
                current_y += line_height + 10  # 行间距
            
            return positions
            
        except Exception:
            return [(100, 100)] * len(text_lines)
    
    def _draw_text_background(self, draw: ImageDraw.ImageDraw, text_lines: List[str], 
                            positions: List[Tuple[int, int]], font: ImageFont.ImageFont, bg_config: Dict):
        """绘制文本背景"""
        try:
            if not bg_config.get('enabled', False):
                return
            
            padding = bg_config.get('padding', 20)
            bg_color = bg_config.get('color', (0, 0, 0, 128))
            border_radius = bg_config.get('border_radius', 10)
            
            # 计算背景区域
            min_x = min(pos[0] for pos in positions) - padding
            max_x = max(pos[0] + font.getbbox(line)[2] - font.getbbox(line)[0] 
                       for pos, line in zip(positions, text_lines)) + padding
            
            min_y = min(pos[1] for pos in positions) - padding
            max_y = max(pos[1] + font.getbbox(line)[3] - font.getbbox(line)[1] 
                       for pos, line in zip(positions, text_lines)) + padding
            
            # 绘制圆角矩形背景
            if len(bg_color) == 4:  # RGBA
                # 创建临时图像用于透明度
                temp_img = Image.new('RGBA', (max_x - min_x, max_y - min_y), bg_color)
                # 这里简化处理，直接绘制矩形
                draw.rectangle([min_x, min_y, max_x, max_y], fill=bg_color[:3])
            else:
                draw.rectangle([min_x, min_y, max_x, max_y], fill=bg_color)
            
        except Exception as e:
            logger.error(f"文本背景绘制失败: {e}")
    
    def _draw_text_with_stroke(self, draw: ImageDraw.ImageDraw, text_lines: List[str], 
                             positions: List[Tuple[int, int]], font: ImageFont.ImageFont,
                             text_color: Tuple[int, int, int], stroke_color: Tuple[int, int, int], 
                             stroke_width: int):
        """绘制带描边的文本"""
        try:
            for line, (x, y) in zip(text_lines, positions):
                # 绘制描边
                if stroke_width > 0:
                    for dx in range(-stroke_width, stroke_width + 1):
                        for dy in range(-stroke_width, stroke_width + 1):
                            if dx * dx + dy * dy <= stroke_width * stroke_width:
                                draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
                
                # 绘制主文本
                draw.text((x, y), line, font=font, fill=text_color)
            
        except Exception as e:
            logger.error(f"文本绘制失败: {e}")
    
    def _save_cover_image(self, image: Image.Image, frame_info: Dict) -> str:
        """保存封面图像"""
        try:
            output_dir = Path("output_data/covers")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(datetime.now().timestamp())
            cover_filename = f"smart_cover_{timestamp}.jpg"
            cover_path = output_dir / cover_filename
            
            # 保存为高质量JPEG
            image.save(cover_path, 'JPEG', quality=95, optimize=True)
            
            logger.info(f"智能封面已保存: {cover_path}")
            return str(cover_path)
            
        except Exception as e:
            logger.error(f"封面保存失败: {e}")
            return ""
    
    def _generate_text_only_cover(self, text_layout: Dict, color_palette: Dict) -> Dict:
        """生成纯文本封面（回退方案）"""
        try:
            if not PIL_AVAILABLE:
                return {'success': False, 'error': 'PIL not available'}
            
            # 创建纯色背景
            target_size = self.cover_config['target_size']
            bg_color = color_palette.get('primary', (70, 130, 180))
            
            cover_image = Image.new('RGB', target_size, bg_color)
            
            # 添加文本
            cover_image = self._add_text_to_cover(cover_image, text_layout)
            
            # 保存
            output_path = self._save_cover_image(cover_image, {'type': 'text_only'})
            
            return {
                'cover_path': output_path,
                'source_frame': {'type': 'text_only'},
                'success': True
            }
            
        except Exception as e:
            logger.error(f"纯文本封面生成失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_fallback_cover(self, clips: List[Dict], title: str) -> Dict:
        """生成回退封面"""
        return {
            'cover_path': '',
            'source_frame': {},
            'success': False,
            'error': '封面生成失败，请检查输入数据',
            'fallback_info': {
                'title': title,
                'clips_count': len(clips),
                'suggestion': '可以尝试提供更多照片或视频片段'
            }
        }


# 全局服务实例
smart_cover_designer = None

def get_smart_cover_designer() -> SmartCoverDesigner:
    """获取智能封面设计器实例"""
    global smart_cover_designer
    if smart_cover_designer is None:
        smart_cover_designer = SmartCoverDesigner()
    return smart_cover_designer
