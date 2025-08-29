import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

class TestAPIEndpoints:
    """API端点测试类"""
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_segment_invalid_transcript(self):
        """测试切片API - 无效字幕"""
        payload = {
            "transcript": "",
            "min_sec": 25,
            "max_sec": 60
        }
        response = client.post("/segment", json=payload)
        assert response.status_code == 422
    
    def test_segment_invalid_duration(self):
        """测试切片API - 无效时长"""
        payload = {
            "transcript": "测试字幕内容",
            "min_sec": -1,
            "max_sec": 60
        }
        response = client.post("/segment", json=payload)
        assert response.status_code == 422
    
    def test_segment_valid_request(self):
        """测试切片API - 有效请求"""
        payload = {
            "transcript": "测试字幕内容，包含时间戳和文本内容。",
            "min_sec": 25,
            "max_sec": 60
        }
        
        with patch('api.main.run_gemini') as mock_gemini:
            mock_gemini.return_value = '{"clips": [{"start": "00:15","end": "00:48","reason": "测试片段"}]}'
            
            response = client.post("/segment", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "clips" in data
            assert len(data["clips"]) > 0
    
    def test_captions_invalid_request(self):
        """测试文案API - 无效请求"""
        payload = {
            "topic": "",
            "transcript": "",
            "clip_text": ""
        }
        response = client.post("/captions", json=payload)
        assert response.status_code == 422
    
    def test_captions_valid_request(self):
        """测试文案API - 有效请求"""
        payload = {
            "topic": "AI技术",
            "transcript": "完整的字幕内容",
            "clip_text": "片段内容描述"
        }
        
        with patch('api.main.run_gemini') as mock_gemini:
            mock_gemini.return_value = '{"title": "测试标题","hashtags": ["#测试"],"desc": "测试描述"}'
            
            response = client.post("/captions", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "title" in data
            assert "hashtags" in data
            assert "desc" in data
    
    def test_cut916_invalid_file(self):
        """测试视频裁剪API - 无效文件"""
        payload = {
            "src": "nonexistent.mp4",
            "start": "00:10",
            "end": "00:30",
            "out": "output.mp4"
        }
        response = client.post("/cut916", json=payload)
        assert response.status_code == 422
    
    def test_cut916_invalid_time_format(self):
        """测试视频裁剪API - 无效时间格式"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"dummy video content")
            tmp_file_path = tmp_file.name
        
        try:
            payload = {
                "src": tmp_file_path,
                "start": "invalid:time",
                "end": "00:30",
                "out": "output.mp4"
            }
            response = client.post("/cut916", json=payload)
            assert response.status_code == 422
        finally:
            os.unlink(tmp_file_path)
    
    def test_cut916_valid_request(self):
        """测试视频裁剪API - 有效请求"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"dummy video content")
            tmp_file_path = tmp_file.name
        
        try:
            payload = {
                "src": tmp_file_path,
                "start": "00:10",
                "end": "00:30",
                "out": "output.mp4"
            }
            
            with patch('api.main.safe_run_ffmpeg') as mock_ffmpeg:
                mock_ffmpeg.return_value = {
                    "code": 0,
                    "stdout": "",
                    "stderr": "",
                    "duration": 1.5
                }
                
                response = client.post("/cut916", json=payload)
                assert response.status_code == 200
                
                data = response.json()
                assert data["out"] == "output.mp4"
                assert data["code"] == 0
        finally:
            os.unlink(tmp_file_path)
    
    def test_upload_invalid_file(self):
        """测试上传API - 无效文件"""
        payload = {
            "path": "nonexistent.mp4",
            "bucket": "test-bucket"
        }
        response = client.post("/upload", json=payload)
        assert response.status_code == 422
    
    def test_upload_valid_request(self):
        """测试上传API - 有效请求"""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(b"dummy video content")
            tmp_file_path = tmp_file.name
        
        try:
            payload = {
                "path": tmp_file_path,
                "bucket": "test-bucket"
            }
            
            response = client.post("/upload", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "url" in data
            assert "key" in data
            assert "size" in data
            assert data["bucket"] == "test-bucket"
        finally:
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    pytest.main([__file__])
