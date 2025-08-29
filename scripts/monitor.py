#!/usr/bin/env python3
"""
AI Video Clipper 监控脚本
用于监控服务状态、性能和资源使用情况
"""

import requests
import psutil
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import argparse

class ServiceMonitor:
    """服务监控类"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.logs_dir = Path("logs")
        
    def check_api_health(self) -> Dict[str, Any]:
        """检查API健康状态"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "response_time_ms": round(response_time, 2),
                    "timestamp": data.get("timestamp"),
                    "version": data.get("version")
                }
            else:
                return {
                    "status": "unhealthy",
                    "response_time_ms": round(response_time, 2),
                    "status_code": response.status_code
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time_ms": None
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统资源统计"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_ffmpeg_processes(self) -> Dict[str, Any]:
        """检查FFmpeg进程"""
        try:
            ffmpeg_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if 'ffmpeg' in proc.info['name'].lower():
                        ffmpeg_processes.append({
                            "pid": proc.info['pid'],
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_percent": proc.info['memory_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                "count": len(ffmpeg_processes),
                "processes": ffmpeg_processes
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_log_files(self) -> Dict[str, Any]:
        """检查日志文件状态"""
        try:
            if not self.logs_dir.exists():
                return {"error": "Logs directory not found"}
            
            log_files = {}
            for log_file in self.logs_dir.glob("*.log"):
                stat = log_file.stat()
                log_files[log_file.name] = {
                    "size_mb": round(stat.st_size / (1024**2), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            
            return {"log_files": log_files}
        except Exception as e:
            return {"error": str(e)}
    
    def check_output_files(self) -> Dict[str, Any]:
        """检查输出文件状态"""
        try:
            output_dir = Path("output_data")
            if not output_dir.exists():
                return {"error": "Output directory not found"}
            
            video_files = []
            total_size = 0
            
            for video_file in output_dir.glob("*.mp4"):
                stat = video_file.stat()
                size_mb = stat.st_size / (1024**2)
                total_size += size_mb
                
                video_files.append({
                    "name": video_file.name,
                    "size_mb": round(size_mb, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return {
                "video_count": len(video_files),
                "total_size_mb": round(total_size, 2),
                "videos": video_files
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_report(self) -> Dict[str, Any]:
        """生成完整监控报告"""
        timestamp = datetime.now().isoformat()
        
        report = {
            "timestamp": timestamp,
            "api_health": self.check_api_health(),
            "system_stats": self.get_system_stats(),
            "ffmpeg_processes": self.check_ffmpeg_processes(),
            "log_files": self.check_log_files(),
            "output_files": self.check_output_files()
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印监控报告"""
        print(f"\n{'='*60}")
        print(f"📊 AI Video Clipper 监控报告")
        print(f"📅 时间: {report['timestamp']}")
        print(f"{'='*60}")
        
        # API健康状态
        api_health = report['api_health']
        status_icon = "✅" if api_health['status'] == 'healthy' else "❌"
        print(f"\n🌐 API状态: {status_icon} {api_health['status']}")
        if api_health.get('response_time_ms'):
            print(f"   响应时间: {api_health['response_time_ms']}ms")
        if api_health.get('version'):
            print(f"   版本: {api_health['version']}")
        
        # 系统资源
        sys_stats = report['system_stats']
        if 'error' not in sys_stats:
            print(f"\n💻 系统资源:")
            print(f"   CPU使用率: {sys_stats['cpu_percent']}%")
            print(f"   内存使用率: {sys_stats['memory_percent']}% ({sys_stats['memory_used_gb']}GB / {sys_stats['memory_total_gb']}GB)")
            print(f"   磁盘使用率: {sys_stats['disk_percent']}% ({sys_stats['disk_used_gb']}GB / {sys_stats['disk_total_gb']}GB)")
        
        # FFmpeg进程
        ffmpeg = report['ffmpeg_processes']
        if 'error' not in ffmpeg:
            print(f"\n🎬 FFmpeg进程: {ffmpeg['count']} 个")
            for proc in ffmpeg['processes']:
                print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']}%, 内存 {proc['memory_percent']:.1f}%")
        
        # 日志文件
        logs = report['log_files']
        if 'error' not in logs:
            print(f"\n📝 日志文件:")
            for name, info in logs['log_files'].items():
                print(f"   {name}: {info['size_mb']}MB (修改时间: {info['modified']})")
        
        # 输出文件
        outputs = report['output_files']
        if 'error' not in outputs:
            print(f"\n📹 输出文件:")
            print(f"   视频数量: {outputs['video_count']}")
            print(f"   总大小: {outputs['total_size_mb']}MB")
            for video in outputs['videos'][:5]:  # 只显示前5个
                print(f"   {video['name']}: {video['size_mb']}MB")
            if len(outputs['videos']) > 5:
                print(f"   ... 还有 {len(outputs['videos']) - 5} 个文件")
        
        print(f"\n{'='*60}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Video Clipper 监控工具")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API服务地址")
    parser.add_argument("--interval", type=int, default=30, help="监控间隔（秒）")
    parser.add_argument("--output", help="输出报告到JSON文件")
    parser.add_argument("--once", action="store_true", help="只运行一次")
    
    args = parser.parse_args()
    
    monitor = ServiceMonitor(args.api_url)
    
    try:
        while True:
            # 生成报告
            report = monitor.generate_report()
            
            # 打印报告
            monitor.print_report(report)
            
            # 保存到文件
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"📄 报告已保存到: {args.output}")
            
            if args.once:
                break
            
            # 等待下次监控
            print(f"\n⏰ {args.interval}秒后进行下次监控...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n⚠️ 监控已停止")

if __name__ == "__main__":
    main()
