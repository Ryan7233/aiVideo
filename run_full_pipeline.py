import json
import requests
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration
from core.config import (
    GEMINI_API_BASE, CUT_API_BASE, MIN_CLIP_DURATION, MAX_CLIP_DURATION
)
from core.runtime import LOG_DIR, ensure_runtime_directories

# Configuration
SRC_VIDEO = os.getenv("SRC_VIDEO", "sample.mp4")
CLEAN_CLIPS_JSON = os.getenv("CLEAN_CLIPS_JSON", "clips_clean.json")
RESULTS_JSON = os.getenv("RESULTS_JSON", "results.json")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 300))

# Setup logging
ensure_runtime_directories()
logger.add(str(LOG_DIR / "pipeline.log"), rotation="10 MB", level="INFO")

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

class VideoClipperPipeline:
    """短视频自动切片工作流主类"""
    
    def __init__(self, src_video: str = SRC_VIDEO, clips_json: str = CLEAN_CLIPS_JSON):
        self.src_video = src_video
        self.clips_json = clips_json
        self.results_json = RESULTS_JSON
        self.final_artifacts = []
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """验证输入文件是否存在"""
        if not os.path.exists(self.src_video):
            raise PipelineError(f"源视频文件不存在: {self.src_video}")
        
        if not os.path.exists(self.clips_json):
            raise PipelineError(f"切片配置文件不存在: {self.clips_json}")
        
        if not os.path.exists('transcript.txt'):
            raise PipelineError("字幕文件不存在: transcript.txt")
    
    def _load_data(self) -> tuple[str, List[Dict]]:
        """加载字幕和切片配置数据"""
        try:
            # Load the full transcript for context
            with open('transcript.txt', 'r', encoding='utf-8') as f:
                full_transcript = f.read()
            
            # Load the cleaned clips data
            with open(self.clips_json, 'r', encoding='utf-8') as f:
                clips_data = json.load(f)["clips"]
            
            logger.info(f"Loaded transcript ({len(full_transcript)} chars) and {len(clips_data)} clips")
            return full_transcript, clips_data
            
        except Exception as e:
            raise PipelineError(f"数据加载失败: {str(e)}")
    
    def _call_api_with_retry(self, url: str, payload: Dict, max_retries: int = 3) -> Dict:
        """带重试机制的API调用"""
        for attempt in range(max_retries):
            try:
                logger.info(f"API call attempt {attempt + 1}: {url}")
                response = requests.post(
                    url, 
                    json=payload, 
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise PipelineError(f"API调用失败 ({url}): {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _generate_916_video(self, clip_info: Dict, clip_index: int) -> str:
        """生成9:16竖屏视频"""
        out_916_path = f"clip_{clip_index}_916.mp4"
        
        cut_payload = {
            "src": self.src_video,
            "start": clip_info["start"],
            "end": clip_info["end"],
            "out": out_916_path
        }
        
        try:
            result = self._call_api_with_retry(f"{CUT_API_BASE}/cut916", cut_payload)
            
            if result.get("code") != 0:
                raise PipelineError(f"视频生成失败: {result.get('stderr', 'Unknown error')}")
            
            logger.info(f"Successfully created 9:16 video: {out_916_path}")
            return out_916_path
            
        except Exception as e:
            # Clean up partial file if it exists
            if os.path.exists(out_916_path):
                os.remove(out_916_path)
            raise PipelineError(f"9:16视频生成失败: {str(e)}")
    
    def _generate_captions(self, clip_info: Dict, full_transcript: str) -> Dict:
        """生成AI文案"""
        caption_payload = {
            "transcript": full_transcript,
            "clip_text": clip_info.get("reason", "")
        }
        
        try:
            captions = self._call_api_with_retry(f"{GEMINI_API_BASE}/captions", caption_payload)
            logger.info(f"Generated captions: {captions.get('title', '')}")
            return captions
            
        except Exception as e:
            logger.warning(f"文案生成失败，使用默认值: {str(e)}")
            return {"title": "", "hashtags": [], "desc": ""}
    
    def _upload_video(self, video_path: str) -> Dict:
        """上传视频到云存储"""
        upload_payload = {"path": video_path}
        
        try:
            upload_info = self._call_api_with_retry(f"{CUT_API_BASE}/upload", upload_payload)
            logger.info(f"Successfully uploaded: {upload_info.get('url')}")
            return upload_info
            
        except Exception as e:
            logger.warning(f"上传失败，使用默认值: {str(e)}")
            return {"url": "", "key": ""}
    
    def _process_clip(self, clip_info: Dict, clip_index: int, full_transcript: str) -> Dict:
        """处理单个视频片段"""
        logger.info(f"Processing Clip #{clip_index} ({clip_info['start']} - {clip_info['end']})")
        
        try:
            # Step 1: Generate 9:16 video
            logger.info("Step 1: Generating 9:16 vertical video...")
            out_916_path = self._generate_916_video(clip_info, clip_index)
            
            # Step 2: Generate captions
            logger.info("Step 2: Generating AI captions...")
            captions = self._generate_captions(clip_info, full_transcript)
            
            # Step 3: Upload to cloud storage
            logger.info("Step 3: Uploading to cloud storage...")
            upload_info = self._upload_video(out_916_path)
            
            # Step 4: Consolidate results
            artifact = {
                "clip_index": clip_index,
                "final_url": upload_info.get("url"),
                "local_file": out_916_path,
                "title": captions.get("title"),
                "hashtags": captions.get("hashtags", []),
                "description": captions.get("desc"),
                "start_time": clip_info["start"],
                "end_time": clip_info["end"],
                "duration": clip_info.get("duration"),
                "reason": clip_info.get("reason"),
                "status": "success"
            }
            
            logger.info(f"Clip #{clip_index} processed successfully")
            return artifact
            
        except Exception as e:
            logger.error(f"Clip #{clip_index} processing failed: {str(e)}")
            return {
                "clip_index": clip_index,
                "start_time": clip_info["start"],
                "end_time": clip_info["end"],
                "status": "failed",
                "error": str(e)
            }
    
    def run(self) -> Dict[str, Any]:
        """运行完整的工作流"""
        logger.info("🚀 Starting AI Video Clipper Pipeline")
        
        try:
            # Load data
            full_transcript, clips_data = self._load_data()
            
            logger.info(f"Found {len(clips_data)} clips to process")
            
            # Process each clip
            for i, clip_info in enumerate(clips_data, 1):
                logger.info(f"\n--- Processing Clip #{i}/{len(clips_data)} ---")
                
                artifact = self._process_clip(clip_info, i, full_transcript)
                self.final_artifacts.append(artifact)
                
                # Progress update
                success_count = sum(1 for a in self.final_artifacts if a.get("status") == "success")
                logger.info(f"Progress: {success_count}/{len(self.final_artifacts)} clips completed successfully")
            
            # Generate summary
            success_count = sum(1 for a in self.final_artifacts if a.get("status") == "success")
            failed_count = len(self.final_artifacts) - success_count
            
            summary = {
                "total_clips": len(self.final_artifacts),
                "successful_clips": success_count,
                "failed_clips": failed_count,
                "success_rate": f"{(success_count/len(self.final_artifacts)*100):.1f}%" if self.final_artifacts else "0%",
                "artifacts": self.final_artifacts
            }
            
            # Write results
            with open(self.results_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Pipeline complete! Summary: {success_count}/{len(self.final_artifacts)} clips successful")
            logger.info(f"📄 Results saved to: {self.results_json}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise PipelineError(f"工作流执行失败: {str(e)}")

def main():
    """主函数"""
    ensure_runtime_directories()
    
    try:
        # Initialize and run pipeline
        pipeline = VideoClipperPipeline()
        results = pipeline.run()
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Total clips processed: {results['total_clips']}")
        print(f"Successful: {results['successful_clips']}")
        print(f"Failed: {results['failed_clips']}")
        print(f"Success rate: {results['success_rate']}")
        print(f"Results saved to: {RESULTS_JSON}")
        print("="*50)
        
        # Print successful clips
        successful_clips = [a for a in results['artifacts'] if a.get('status') == 'success']
        if successful_clips:
            print("\n📹 Successfully Generated Videos:")
            for clip in successful_clips:
                print(f"  • {clip.get('local_file', 'Unknown')} - {clip.get('title', 'No title')}")
        
        return 0
        
    except PipelineError as e:
        logger.error(f"Pipeline error: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        print("\n⚠️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
