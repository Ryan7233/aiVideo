"""
产物打包导出服务
支持对象存储上传和CDN分发
"""

import json
import logging
import zipfile
import shutil
from typing import List, Dict, Optional
from pathlib import Path
from core.runtime import OUTPUT_DIR
from datetime import datetime
import hashlib
import uuid

logger = logging.getLogger(__name__)


class ExportService:
    """导出服务"""
    
    def __init__(self):
        self.export_base_path = OUTPUT_DIR / "exports"
        self.export_base_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的导出格式
        self.supported_formats = {
            'json': self._export_json,
            'zip': self._export_zip,
            'folder': self._export_folder
        }
    
    def export_xiaohongshu_content(self, pipeline_result: Dict, 
                                  export_format: str = "zip",
                                  include_source: bool = False) -> Dict:
        """
        导出小红书内容产物
        
        Args:
            pipeline_result: 流水线处理结果
            export_format: 导出格式 (json/zip/folder)
            include_source: 是否包含源文件
            
        Returns:
            导出结果信息
        """
        try:
            # 生成导出ID
            export_id = self._generate_export_id(pipeline_result)
            export_path = self.export_base_path / export_id
            export_path.mkdir(parents=True, exist_ok=True)
            
            # 收集所有产物
            assets = self._collect_assets(pipeline_result, include_source)
            
            # 生成元数据
            metadata = self._generate_metadata(pipeline_result, assets, export_id)
            
            # 根据格式导出
            if export_format in self.supported_formats:
                export_result = self.supported_formats[export_format](
                    export_path, assets, metadata
                )
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            # 生成分享链接（简化版）
            share_urls = self._generate_share_urls(export_result, export_id)
            
            return {
                'export_id': export_id,
                'export_format': export_format,
                'export_path': str(export_result['path']),
                'file_size': export_result['size'],
                'asset_count': len(assets),
                'share_urls': share_urls,
                'metadata': metadata,
                'created_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"导出失败: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
    
    def _generate_export_id(self, pipeline_result: Dict) -> str:
        """生成导出ID"""
        # 基于内容生成唯一ID
        content_hash = hashlib.md5(
            json.dumps(pipeline_result, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:8]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"xhs_export_{timestamp}_{content_hash}"
    
    def _collect_assets(self, pipeline_result: Dict, include_source: bool) -> List[Dict]:
        """收集所有产物文件"""
        assets = []
        
        # 视频片段
        clips = pipeline_result.get('clips', [])
        for clip in clips:
            if 'output_path' in clip and Path(clip['output_path']).exists():
                assets.append({
                    'type': 'video',
                    'category': 'clip',
                    'path': clip['output_path'],
                    'filename': Path(clip['output_path']).name,
                    'size': Path(clip['output_path']).stat().st_size,
                    'clip_index': clip.get('clip_index', 0),
                    'duration': clip.get('duration', 0)
                })
        
        # 字幕文件
        subtitles = pipeline_result.get('subtitles', {})
        for srt_file in subtitles.get('srt_files', []):
            if Path(srt_file).exists():
                assets.append({
                    'type': 'subtitle',
                    'category': 'srt',
                    'path': srt_file,
                    'filename': Path(srt_file).name,
                    'size': Path(srt_file).stat().st_size
                })
        
        for ass_file in subtitles.get('ass_files', []):
            if Path(ass_file).exists():
                assets.append({
                    'type': 'subtitle',
                    'category': 'ass',
                    'path': ass_file,
                    'filename': Path(ass_file).name,
                    'size': Path(ass_file).stat().st_size
                })
        
        # 封面图片
        cover = pipeline_result.get('cover', {})
        best_cover = cover.get('best_cover', {})
        if best_cover.get('source') and Path(best_cover['source']).exists():
            assets.append({
                'type': 'image',
                'category': 'cover',
                'path': best_cover['source'],
                'filename': Path(best_cover['source']).name,
                'size': Path(best_cover['source']).stat().st_size,
                'cover_type': best_cover.get('type', 'unknown')
            })
        
        # 精选照片
        photos = pipeline_result.get('photos_ranked', [])
        for photo in photos[:5]:  # 只包含前5张
            if Path(photo['path']).exists():
                assets.append({
                    'type': 'image',
                    'category': 'photo',
                    'path': photo['path'],
                    'filename': Path(photo['path']).name,
                    'size': Path(photo['path']).stat().st_size,
                    'photo_rank': photo.get('rank', 0),
                    'photo_score': photo.get('score', 0)
                })
        
        # 源文件（如果需要）
        if include_source:
            source_video = pipeline_result.get('source_video', '')
            if source_video and Path(source_video).exists():
                assets.append({
                    'type': 'video',
                    'category': 'source',
                    'path': source_video,
                    'filename': Path(source_video).name,
                    'size': Path(source_video).stat().st_size
                })
        
        return assets
    
    def _generate_metadata(self, pipeline_result: Dict, assets: List[Dict], 
                          export_id: str) -> Dict:
        """生成元数据"""
        return {
            'export_info': {
                'export_id': export_id,
                'created_at': datetime.now().isoformat(),
                'generator': 'AI Video Clipper - XHS Pipeline',
                'version': '1.0.0'
            },
            'content_info': {
                'title': pipeline_result.get('draft', {}).get('title', ''),
                'city': pipeline_result.get('storyline', {}).get('city', ''),
                'style': pipeline_result.get('storyline', {}).get('style', ''),
                'duration_estimate': pipeline_result.get('storyline', {}).get('duration_estimate', ''),
                'clip_count': len([a for a in assets if a['type'] == 'video' and a['category'] == 'clip']),
                'photo_count': len([a for a in assets if a['type'] == 'image' and a['category'] == 'photo'])
            },
            'draft_content': pipeline_result.get('draft', {}),
            'storyline': pipeline_result.get('storyline', {}),
            'processing_stats': {
                'total_assets': len(assets),
                'total_size_mb': sum(a['size'] for a in assets) / (1024 * 1024),
                'video_count': len([a for a in assets if a['type'] == 'video']),
                'image_count': len([a for a in assets if a['type'] == 'image']),
                'subtitle_count': len([a for a in assets if a['type'] == 'subtitle'])
            }
        }
    
    def _export_json(self, export_path: Path, assets: List[Dict], metadata: Dict) -> Dict:
        """导出为JSON格式"""
        # 创建JSON数据
        json_data = {
            'metadata': metadata,
            'assets': assets,
            'file_list': [asset['filename'] for asset in assets]
        }
        
        # 保存JSON文件
        json_file = export_path / 'xiaohongshu_content.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 复制资源文件
        for asset in assets:
            src_path = Path(asset['path'])
            dst_path = export_path / asset['filename']
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        
        return {
            'path': export_path,
            'size': sum(f.stat().st_size for f in export_path.iterdir() if f.is_file()),
            'main_file': str(json_file)
        }
    
    def _export_zip(self, export_path: Path, assets: List[Dict], metadata: Dict) -> Dict:
        """导出为ZIP格式"""
        zip_file = export_path.parent / f"{export_path.name}.zip"
        
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 添加元数据
            zf.writestr('metadata.json', json.dumps(metadata, ensure_ascii=False, indent=2))
            
            # 添加资源文件
            for asset in assets:
                src_path = Path(asset['path'])
                if src_path.exists():
                    # 根据类型创建目录结构
                    if asset['type'] == 'video':
                        arc_name = f"videos/{asset['filename']}"
                    elif asset['type'] == 'image':
                        arc_name = f"images/{asset['filename']}"
                    elif asset['type'] == 'subtitle':
                        arc_name = f"subtitles/{asset['filename']}"
                    else:
                        arc_name = asset['filename']
                    
                    zf.write(src_path, arc_name)
            
            # 添加小红书文案
            draft = metadata.get('draft_content', {})
            if draft:
                readme_content = self._generate_readme(draft, metadata)
                zf.writestr('小红书文案.txt', readme_content)
        
        # 清理临时目录
        if export_path.exists():
            shutil.rmtree(export_path)
        
        return {
            'path': zip_file,
            'size': zip_file.stat().st_size,
            'main_file': str(zip_file)
        }
    
    def _export_folder(self, export_path: Path, assets: List[Dict], metadata: Dict) -> Dict:
        """导出为文件夹格式"""
        # 创建子目录
        (export_path / 'videos').mkdir(exist_ok=True)
        (export_path / 'images').mkdir(exist_ok=True)
        (export_path / 'subtitles').mkdir(exist_ok=True)
        
        # 保存元数据
        with open(export_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 复制资源文件到对应目录
        for asset in assets:
            src_path = Path(asset['path'])
            if src_path.exists():
                if asset['type'] == 'video':
                    dst_path = export_path / 'videos' / asset['filename']
                elif asset['type'] == 'image':
                    dst_path = export_path / 'images' / asset['filename']
                elif asset['type'] == 'subtitle':
                    dst_path = export_path / 'subtitles' / asset['filename']
                else:
                    dst_path = export_path / asset['filename']
                
                shutil.copy2(src_path, dst_path)
        
        # 生成README
        draft = metadata.get('draft_content', {})
        if draft:
            readme_content = self._generate_readme(draft, metadata)
            with open(export_path / '小红书文案.txt', 'w', encoding='utf-8') as f:
                f.write(readme_content)
        
        return {
            'path': export_path,
            'size': sum(f.stat().st_size for f in export_path.rglob('*') if f.is_file()),
            'main_file': str(export_path / 'metadata.json')
        }
    
    def _generate_readme(self, draft: Dict, metadata: Dict) -> str:
        """生成README文件内容"""
        content = []
        
        # 标题
        title = draft.get('title', '小红书内容')
        content.append(f"📝 {title}")
        content.append("=" * 50)
        content.append("")
        
        # 正文
        body = draft.get('body', '')
        if body:
            content.append("📖 正文内容：")
            content.append(body)
            content.append("")
        
        # 话题标签
        hashtags = draft.get('hashtags', [])
        if hashtags:
            content.append("🏷️ 话题标签：")
            content.append(" ".join(hashtags))
            content.append("")
        
        # POI信息
        poi = draft.get('poi', [])
        if poi:
            content.append("📍 地点信息：")
            for p in poi:
                content.append(f"- {p.get('name', '')}")
            content.append("")
        
        # 生成信息
        content.append("🤖 生成信息：")
        content.append(f"生成时间：{metadata['export_info']['created_at']}")
        content.append(f"导出ID：{metadata['export_info']['export_id']}")
        content.append(f"城市：{metadata['content_info']['city']}")
        content.append(f"风格：{metadata['content_info']['style']}")
        content.append("")
        
        # 文件列表
        content.append("📁 文件说明：")
        content.append("- videos/: 视频片段文件")
        content.append("- images/: 图片和封面文件")
        content.append("- subtitles/: 字幕文件 (SRT/ASS格式)")
        content.append("- metadata.json: 详细元数据")
        
        return "\n".join(content)
    
    def _generate_share_urls(self, export_result: Dict, export_id: str) -> Dict:
        """生成分享链接（简化版）"""
        base_url = "https://example.com/exports"  # 实际部署时替换为真实域名
        
        return {
            'download_url': f"{base_url}/{export_id}/download",
            'preview_url': f"{base_url}/{export_id}/preview",
            'share_code': export_id[-8:].upper(),  # 8位分享码
            'expires_at': (datetime.now()).isoformat(),  # 简化版不设过期
            'qr_code_url': f"{base_url}/{export_id}/qr"
        }
    
    def cleanup_old_exports(self, days: int = 7) -> Dict:
        """清理旧的导出文件"""
        try:
            cleaned_count = 0
            total_size_freed = 0
            
            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
            
            for export_dir in self.export_base_path.iterdir():
                if export_dir.is_dir():
                    dir_time = export_dir.stat().st_mtime
                    if dir_time < cutoff_time:
                        size = sum(f.stat().st_size for f in export_dir.rglob('*') if f.is_file())
                        shutil.rmtree(export_dir)
                        cleaned_count += 1
                        total_size_freed += size
                
                elif export_dir.is_file() and export_dir.suffix == '.zip':
                    file_time = export_dir.stat().st_mtime
                    if file_time < cutoff_time:
                        size = export_dir.stat().st_size
                        export_dir.unlink()
                        cleaned_count += 1
                        total_size_freed += size
            
            return {
                'cleaned_count': cleaned_count,
                'size_freed_mb': total_size_freed / (1024 * 1024),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"清理导出文件失败: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局服务实例
export_service = None

def get_export_service() -> ExportService:
    """获取导出服务实例"""
    global export_service
    if export_service is None:
        export_service = ExportService()
    return export_service
