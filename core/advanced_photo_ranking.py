"""
高级照片选优服务
集成CLIP模型、美学评分、重复检测、主体跟随
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import hashlib
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

# 尝试导入CLIP相关库（可选依赖）
try:
    import torch
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
    logger.info("CLIP模型可用")
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP模型不可用，将使用传统方法")

# 尝试导入图像处理库
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV不可用，某些功能将受限")


class AdvancedPhotoRankingService:
    """高级照片选优服务"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        
        # 美学评分权重配置
        self.aesthetic_weights = {
            'composition': 0.25,      # 构图质量
            'color_harmony': 0.20,    # 色彩和谐度
            'lighting': 0.20,         # 光照质量
            'subject_clarity': 0.15,  # 主体清晰度
            'visual_interest': 0.10,  # 视觉兴趣点
            'technical_quality': 0.10 # 技术质量
        }
        
        # 主题相关性关键词
        self.topic_keywords = {
            'food': ['food', 'meal', 'dish', 'restaurant', 'cuisine', '美食', '餐厅', '菜品'],
            'landscape': ['landscape', 'scenery', 'mountain', 'sea', 'sky', '风景', '山', '海'],
            'architecture': ['building', 'architecture', 'temple', 'bridge', '建筑', '寺庙', '桥'],
            'portrait': ['person', 'people', 'face', 'selfie', '人物', '自拍', '合影'],
            'lifestyle': ['lifestyle', 'daily', 'street', 'urban', '生活', '街景', '城市']
        }
        
        # 初始化CLIP模型
        if CLIP_AVAILABLE:
            self._initialize_clip_model()
    
    def _initialize_clip_model(self):
        """初始化CLIP模型"""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info(f"CLIP模型已加载到设备: {self.device}")
        except Exception as e:
            logger.error(f"CLIP模型加载失败: {e}")
            self.clip_model = None
    
    def rank_photos_advanced(self, photos: List[str], top_k: int = 15, 
                           context: Dict = None) -> List[Dict]:
        """
        高级照片选优排序
        
        Args:
            photos: 照片路径列表
            top_k: 返回前k张照片
            context: 上下文信息（主题、风格等）
            
        Returns:
            排序后的照片信息
        """
        try:
            if not photos:
                return []
            
            logger.info(f"开始高级照片选优，共 {len(photos)} 张照片")
            
            # 1. 基础信息提取
            photo_features = []
            for i, photo_path in enumerate(photos):
                if not Path(photo_path).exists():
                    logger.warning(f"照片不存在: {photo_path}")
                    continue
                
                features = self._extract_photo_features(photo_path, i, context)
                if features:
                    photo_features.append(features)
            
            if not photo_features:
                logger.warning("没有有效的照片特征")
                return []
            
            # 2. CLIP语义分析（如果可用）
            if self.clip_model and CLIP_AVAILABLE:
                photo_features = self._add_clip_features(photo_features, context)
            
            # 3. 美学质量评分
            photo_features = self._calculate_aesthetic_scores(photo_features)
            
            # 4. 重复检测
            photo_features = self._detect_duplicates(photo_features)
            
            # 5. 主体一致性分析
            photo_features = self._analyze_subject_consistency(photo_features)
            
            # 6. 综合评分和排序
            ranked_photos = self._calculate_final_scores(photo_features, context)
            
            # 7. 多样性优化
            final_selection = self._optimize_diversity(ranked_photos, top_k)
            
            logger.info(f"高级照片选优完成，返回 {len(final_selection)} 张照片")
            return final_selection
            
        except Exception as e:
            logger.error(f"高级照片选优失败: {e}")
            # 回退到基础版本
            from core.xiaohongshu_pipeline import get_photo_ranking_service
            basic_service = get_photo_ranking_service()
            return basic_service.rank_photos(photos, top_k)
    
    def _extract_photo_features(self, photo_path: str, index: int, context: Dict = None) -> Optional[Dict]:
        """提取照片基础特征"""
        try:
            if not CV2_AVAILABLE:
                # 没有OpenCV时的简化版本
                return {
                    'photo_id': f'photo_{index:03d}',
                    'path': photo_path,
                    'index': index,
                    'file_size': Path(photo_path).stat().st_size,
                    'basic_score': random.uniform(0.3, 0.8)
                }
            
            # 读取图像
            image = cv2.imread(photo_path)
            if image is None:
                logger.warning(f"无法读取图像: {photo_path}")
                return None
            
            height, width = image.shape[:2]
            
            # 基础特征
            features = {
                'photo_id': f'photo_{index:03d}',
                'path': photo_path,
                'index': index,
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'file_size': Path(photo_path).stat().st_size,
                'image_data': image,  # 保存用于后续分析
            }
            
            # 计算图像哈希（用于重复检测）
            features['image_hash'] = self._calculate_image_hash(image)
            
            # 颜色特征
            features['color_features'] = self._extract_color_features(image)
            
            # 纹理特征
            features['texture_features'] = self._extract_texture_features(image)
            
            # 边缘特征
            features['edge_features'] = self._extract_edge_features(image)
            
            return features
            
        except Exception as e:
            logger.error(f"提取照片特征失败 {photo_path}: {e}")
            return None
    
    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """计算图像感知哈希"""
        try:
            # 缩放到8x8
            small = cv2.resize(image, (8, 8))
            # 转为灰度
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # 计算平均值
            avg = gray.mean()
            # 生成哈希
            hash_str = ''
            for i in range(8):
                for j in range(8):
                    hash_str += '1' if gray[i, j] > avg else '0'
            return hash_str
        except Exception:
            return str(random.randint(10000000, 99999999))
    
    def _extract_color_features(self, image: np.ndarray) -> Dict:
        """提取颜色特征"""
        try:
            # 颜色直方图
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # 主要颜色
            dominant_color = image.reshape(-1, 3).mean(axis=0)
            
            # 颜色分布
            color_std = image.reshape(-1, 3).std(axis=0)
            
            return {
                'dominant_color': dominant_color.tolist(),
                'color_variance': color_std.tolist(),
                'brightness': float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()),
                'contrast': float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).std())
            }
        except Exception:
            return {'brightness': 128, 'contrast': 50}
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict:
        """提取纹理特征"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 计算梯度
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 纹理复杂度
            texture_complexity = np.sqrt(grad_x**2 + grad_y**2).mean()
            
            return {
                'complexity': float(texture_complexity),
                'gradient_variance': float(np.var(np.sqrt(grad_x**2 + grad_y**2)))
            }
        except Exception:
            return {'complexity': 50, 'gradient_variance': 25}
    
    def _extract_edge_features(self, image: np.ndarray) -> Dict:
        """提取边缘特征"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 边缘密度
            edge_density = np.sum(edges > 0) / edges.size
            
            return {
                'edge_density': float(edge_density),
                'edge_count': int(np.sum(edges > 0))
            }
        except Exception:
            return {'edge_density': 0.1, 'edge_count': 1000}
    
    def _add_clip_features(self, photo_features: List[Dict], context: Dict = None) -> List[Dict]:
        """添加CLIP语义特征"""
        if not self.clip_model or not CLIP_AVAILABLE:
            return photo_features
        
        try:
            logger.info("开始CLIP语义分析...")
            
            # 准备文本查询
            text_queries = [
                "a beautiful landscape photo",
                "delicious food photography", 
                "architectural photography",
                "portrait photography",
                "street photography",
                "high quality photo",
                "aesthetically pleasing image"
            ]
            
            # 如果有上下文，添加相关查询
            if context:
                city = context.get('city', '')
                style = context.get('style', '')
                if city:
                    text_queries.append(f"beautiful {city} scenery")
                if style == '美食':
                    text_queries.extend(["restaurant food", "local cuisine"])
                elif style == '风景':
                    text_queries.extend(["scenic view", "natural landscape"])
            
            # 编码文本
            text_tokens = clip.tokenize(text_queries).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 处理每张照片
            for photo in photo_features:
                try:
                    # 加载和预处理图像
                    image = Image.open(photo['path']).convert('RGB')
                    image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        image_features = self.clip_model.encode_image(image_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # 计算相似度
                    similarities = (image_features @ text_features.T).squeeze(0)
                    
                    # 保存CLIP特征
                    photo['clip_features'] = {
                        'similarities': similarities.cpu().numpy().tolist(),
                        'max_similarity': float(similarities.max()),
                        'avg_similarity': float(similarities.mean()),
                        'aesthetic_score': float(similarities[5]),  # "high quality photo"
                        'content_scores': {
                            'landscape': float(similarities[0]),
                            'food': float(similarities[1]),
                            'architecture': float(similarities[2]),
                            'portrait': float(similarities[3]),
                            'street': float(similarities[4])
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"CLIP分析失败 {photo['path']}: {e}")
                    # 设置默认值
                    photo['clip_features'] = {
                        'aesthetic_score': 0.5,
                        'max_similarity': 0.5,
                        'content_scores': {}
                    }
            
            return photo_features
            
        except Exception as e:
            logger.error(f"CLIP特征提取失败: {e}")
            return photo_features
    
    def _calculate_aesthetic_scores(self, photo_features: List[Dict]) -> List[Dict]:
        """计算美学质量评分"""
        try:
            for photo in photo_features:
                aesthetic_score = 0.0
                
                # 1. 构图质量 (基于三分法则、黄金比例等)
                composition_score = self._evaluate_composition(photo)
                aesthetic_score += composition_score * self.aesthetic_weights['composition']
                
                # 2. 色彩和谐度
                color_score = self._evaluate_color_harmony(photo)
                aesthetic_score += color_score * self.aesthetic_weights['color_harmony']
                
                # 3. 光照质量
                lighting_score = self._evaluate_lighting(photo)
                aesthetic_score += lighting_score * self.aesthetic_weights['lighting']
                
                # 4. 主体清晰度
                clarity_score = self._evaluate_subject_clarity(photo)
                aesthetic_score += clarity_score * self.aesthetic_weights['subject_clarity']
                
                # 5. 视觉兴趣点
                interest_score = self._evaluate_visual_interest(photo)
                aesthetic_score += interest_score * self.aesthetic_weights['visual_interest']
                
                # 6. 技术质量
                technical_score = self._evaluate_technical_quality(photo)
                aesthetic_score += technical_score * self.aesthetic_weights['technical_quality']
                
                # 如果有CLIP美学评分，融合进来
                if 'clip_features' in photo:
                    clip_aesthetic = photo['clip_features'].get('aesthetic_score', 0.5)
                    aesthetic_score = aesthetic_score * 0.7 + clip_aesthetic * 0.3
                
                photo['aesthetic_score'] = min(aesthetic_score, 1.0)
                photo['aesthetic_details'] = {
                    'composition': composition_score,
                    'color': color_score,
                    'lighting': lighting_score,
                    'clarity': clarity_score,
                    'interest': interest_score,
                    'technical': technical_score
                }
            
            return photo_features
            
        except Exception as e:
            logger.error(f"美学评分计算失败: {e}")
            # 设置默认分数
            for photo in photo_features:
                photo['aesthetic_score'] = 0.5
            return photo_features
    
    def _evaluate_composition(self, photo: Dict) -> float:
        """评估构图质量"""
        try:
            # 基于长宽比的构图评分
            aspect_ratio = photo.get('aspect_ratio', 1.0)
            
            # 黄金比例 (1.618) 和常见比例的偏好
            golden_ratio = 1.618
            common_ratios = [1.0, 4/3, 3/2, 16/9, golden_ratio]
            
            # 找到最接近的比例
            ratio_scores = [1.0 / (1.0 + abs(aspect_ratio - ratio)) for ratio in common_ratios]
            composition_score = max(ratio_scores)
            
            # 如果有边缘特征，考虑线条和结构
            if 'edge_features' in photo:
                edge_density = photo['edge_features'].get('edge_density', 0.1)
                # 适中的边缘密度通常表示良好的构图
                edge_score = 1.0 - abs(edge_density - 0.15) / 0.15
                composition_score = composition_score * 0.7 + max(0, edge_score) * 0.3
            
            return min(composition_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _evaluate_color_harmony(self, photo: Dict) -> float:
        """评估色彩和谐度"""
        try:
            color_features = photo.get('color_features', {})
            
            # 基于颜色方差的和谐度
            color_variance = color_features.get('color_variance', [50, 50, 50])
            avg_variance = sum(color_variance) / len(color_variance)
            
            # 适中的方差通常表示良好的色彩平衡
            harmony_score = 1.0 - abs(avg_variance - 40) / 40
            harmony_score = max(0, min(harmony_score, 1.0))
            
            # 考虑亮度
            brightness = color_features.get('brightness', 128)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            return harmony_score * 0.7 + brightness_score * 0.3
            
        except Exception:
            return 0.5
    
    def _evaluate_lighting(self, photo: Dict) -> float:
        """评估光照质量"""
        try:
            color_features = photo.get('color_features', {})
            
            brightness = color_features.get('brightness', 128)
            contrast = color_features.get('contrast', 50)
            
            # 适中的亮度和对比度
            brightness_score = 1.0 - abs(brightness - 128) / 128
            contrast_score = min(contrast / 80, 1.0)  # 高对比度通常更好
            
            lighting_score = brightness_score * 0.6 + contrast_score * 0.4
            return max(0, min(lighting_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _evaluate_subject_clarity(self, photo: Dict) -> float:
        """评估主体清晰度"""
        try:
            texture_features = photo.get('texture_features', {})
            edge_features = photo.get('edge_features', {})
            
            # 纹理复杂度表示细节丰富度
            complexity = texture_features.get('complexity', 50)
            complexity_score = min(complexity / 100, 1.0)
            
            # 边缘密度表示清晰度
            edge_density = edge_features.get('edge_density', 0.1)
            edge_score = min(edge_density / 0.2, 1.0)
            
            clarity_score = complexity_score * 0.6 + edge_score * 0.4
            return max(0, min(clarity_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _evaluate_visual_interest(self, photo: Dict) -> float:
        """评估视觉兴趣点"""
        try:
            # 基于纹理变化和边缘分布
            texture_features = photo.get('texture_features', {})
            edge_features = photo.get('edge_features', {})
            
            gradient_variance = texture_features.get('gradient_variance', 25)
            edge_count = edge_features.get('edge_count', 1000)
            
            # 适度的变化表示有趣的视觉内容
            variance_score = min(gradient_variance / 50, 1.0)
            edge_score = min(edge_count / 5000, 1.0)
            
            interest_score = variance_score * 0.5 + edge_score * 0.5
            return max(0, min(interest_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _evaluate_technical_quality(self, photo: Dict) -> float:
        """评估技术质量"""
        try:
            # 基于文件大小和分辨率
            file_size = photo.get('file_size', 0)
            width = photo.get('width', 0)
            height = photo.get('height', 0)
            
            # 分辨率评分
            pixel_count = width * height
            resolution_score = min(pixel_count / (1920 * 1080), 1.0)
            
            # 文件大小评分（适中的大小通常表示良好的质量）
            size_mb = file_size / (1024 * 1024)
            size_score = min(size_mb / 5, 1.0) if size_mb > 0 else 0.3
            
            technical_score = resolution_score * 0.7 + size_score * 0.3
            return max(0, min(technical_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _detect_duplicates(self, photo_features: List[Dict]) -> List[Dict]:
        """检测重复照片"""
        try:
            logger.info("开始重复检测...")
            
            # 计算图像哈希相似度
            hash_groups = defaultdict(list)
            
            for photo in photo_features:
                image_hash = photo.get('image_hash', '')
                if image_hash:
                    hash_groups[image_hash].append(photo)
            
            # 标记重复照片
            for photo in photo_features:
                image_hash = photo.get('image_hash', '')
                duplicates = hash_groups.get(image_hash, [])
                
                photo['is_duplicate'] = len(duplicates) > 1
                photo['duplicate_group_size'] = len(duplicates)
                
                # 如果是重复组，选择最好的一张
                if len(duplicates) > 1:
                    best_photo = max(duplicates, key=lambda p: p.get('aesthetic_score', 0))
                    photo['is_best_in_group'] = (photo['photo_id'] == best_photo['photo_id'])
                else:
                    photo['is_best_in_group'] = True
            
            return photo_features
            
        except Exception as e:
            logger.error(f"重复检测失败: {e}")
            for photo in photo_features:
                photo['is_duplicate'] = False
                photo['is_best_in_group'] = True
            return photo_features
    
    def _analyze_subject_consistency(self, photo_features: List[Dict]) -> List[Dict]:
        """分析主体一致性"""
        try:
            if not self.clip_model or len(photo_features) < 2:
                for photo in photo_features:
                    photo['subject_consistency'] = 0.8
                return photo_features
            
            logger.info("开始主体一致性分析...")
            
            # 如果有CLIP特征，计算内容相似度
            content_vectors = []
            for photo in photo_features:
                clip_features = photo.get('clip_features', {})
                content_scores = clip_features.get('content_scores', {})
                if content_scores:
                    vector = [
                        content_scores.get('landscape', 0),
                        content_scores.get('food', 0),
                        content_scores.get('architecture', 0),
                        content_scores.get('portrait', 0),
                        content_scores.get('street', 0)
                    ]
                    content_vectors.append(vector)
                else:
                    content_vectors.append([0.2, 0.2, 0.2, 0.2, 0.2])
            
            # 计算每张照片与其他照片的平均相似度
            for i, photo in enumerate(photo_features):
                similarities = []
                for j, other_vector in enumerate(content_vectors):
                    if i != j:
                        similarity = cosine_similarity([content_vectors[i]], [other_vector])[0][0]
                        similarities.append(similarity)
                
                photo['subject_consistency'] = np.mean(similarities) if similarities else 0.5
            
            return photo_features
            
        except Exception as e:
            logger.error(f"主体一致性分析失败: {e}")
            for photo in photo_features:
                photo['subject_consistency'] = 0.5
            return photo_features
    
    def _calculate_final_scores(self, photo_features: List[Dict], context: Dict = None) -> List[Dict]:
        """计算最终综合评分"""
        try:
            for photo in photo_features:
                # 基础分数权重
                weights = {
                    'aesthetic': 0.4,
                    'clip_quality': 0.2,
                    'subject_consistency': 0.15,
                    'technical': 0.1,
                    'uniqueness': 0.1,
                    'context_relevance': 0.05
                }
                
                # 美学分数
                aesthetic_score = photo.get('aesthetic_score', 0.5)
                
                # CLIP质量分数
                clip_features = photo.get('clip_features', {})
                clip_score = clip_features.get('max_similarity', 0.5)
                
                # 主体一致性
                consistency_score = photo.get('subject_consistency', 0.5)
                
                # 技术质量
                technical_score = photo['aesthetic_details'].get('technical', 0.5) if 'aesthetic_details' in photo else 0.5
                
                # 唯一性（非重复）
                uniqueness_score = 1.0 if photo.get('is_best_in_group', True) else 0.3
                
                # 上下文相关性
                context_score = self._calculate_context_relevance(photo, context)
                
                # 综合评分
                final_score = (
                    aesthetic_score * weights['aesthetic'] +
                    clip_score * weights['clip_quality'] +
                    consistency_score * weights['subject_consistency'] +
                    technical_score * weights['technical'] +
                    uniqueness_score * weights['uniqueness'] +
                    context_score * weights['context_relevance']
                )
                
                photo['final_score'] = final_score
                photo['score_breakdown'] = {
                    'aesthetic': aesthetic_score,
                    'clip_quality': clip_score,
                    'consistency': consistency_score,
                    'technical': technical_score,
                    'uniqueness': uniqueness_score,
                    'context': context_score
                }
            
            # 按最终分数排序
            photo_features.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 添加排名
            for i, photo in enumerate(photo_features):
                photo['rank'] = i + 1
            
            return photo_features
            
        except Exception as e:
            logger.error(f"最终评分计算失败: {e}")
            return photo_features
    
    def _calculate_context_relevance(self, photo: Dict, context: Dict = None) -> float:
        """计算上下文相关性"""
        if not context:
            return 0.5
        
        try:
            relevance_score = 0.5
            
            # 基于CLIP内容分析
            clip_features = photo.get('clip_features', {})
            content_scores = clip_features.get('content_scores', {})
            
            # 根据上下文调整
            style = context.get('style', '')
            city = context.get('city', '')
            
            if style == '美食' and 'food' in content_scores:
                relevance_score = content_scores['food']
            elif style == '风景' and 'landscape' in content_scores:
                relevance_score = content_scores['landscape']
            elif style == '建筑' and 'architecture' in content_scores:
                relevance_score = content_scores['architecture']
            
            return min(relevance_score, 1.0)
            
        except Exception:
            return 0.5
    
    def _optimize_diversity(self, ranked_photos: List[Dict], top_k: int) -> List[Dict]:
        """优化多样性选择"""
        try:
            if len(ranked_photos) <= top_k:
                return ranked_photos
            
            selected = []
            remaining = ranked_photos.copy()
            
            # 首先选择最高分的照片
            selected.append(remaining.pop(0))
            
            # 然后基于多样性选择剩余照片
            while len(selected) < top_k and remaining:
                best_candidate = None
                best_diversity_score = -1
                
                for candidate in remaining:
                    # 计算与已选照片的多样性
                    diversity_score = self._calculate_diversity_score(candidate, selected)
                    
                    # 综合质量分数和多样性分数
                    combined_score = candidate['final_score'] * 0.7 + diversity_score * 0.3
                    
                    if combined_score > best_diversity_score:
                        best_diversity_score = combined_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            # 重新排序选中的照片
            for i, photo in enumerate(selected):
                photo['final_rank'] = i + 1
            
            return selected
            
        except Exception as e:
            logger.error(f"多样性优化失败: {e}")
            return ranked_photos[:top_k]
    
    def _calculate_diversity_score(self, candidate: Dict, selected: List[Dict]) -> float:
        """计算多样性分数"""
        try:
            if not selected:
                return 1.0
            
            diversity_scores = []
            
            # 基于CLIP特征的多样性
            candidate_clip = candidate.get('clip_features', {}).get('content_scores', {})
            
            for selected_photo in selected:
                selected_clip = selected_photo.get('clip_features', {}).get('content_scores', {})
                
                if candidate_clip and selected_clip:
                    # 计算内容向量的余弦距离
                    candidate_vector = [candidate_clip.get(k, 0) for k in ['landscape', 'food', 'architecture', 'portrait', 'street']]
                    selected_vector = [selected_clip.get(k, 0) for k in ['landscape', 'food', 'architecture', 'portrait', 'street']]
                    
                    similarity = cosine_similarity([candidate_vector], [selected_vector])[0][0]
                    diversity = 1.0 - similarity  # 相似度越低，多样性越高
                    diversity_scores.append(diversity)
                else:
                    diversity_scores.append(0.5)
            
            return np.mean(diversity_scores)
            
        except Exception:
            return 0.5


# 全局服务实例
advanced_photo_service = None

def get_advanced_photo_service() -> AdvancedPhotoRankingService:
    """获取高级照片选优服务实例"""
    global advanced_photo_service
    if advanced_photo_service is None:
        advanced_photo_service = AdvancedPhotoRankingService()
    return advanced_photo_service
