#!/usr/bin/env bash
set -euo pipefail

in="$1"       # 源视频路径
start="$2"    # 开始时间，支持 00:10 或 00:00:10
end="$3"      # 结束时间，支持 00:45 或 00:00:45
out="$4"      # 输出文件路径

# 将 mm:ss 规范成 hh:mm:ss（避免 ffmpeg 解析歧义）
norm_time () {
  local t="$1"
  if [[ "$t" =~ ^[0-9]{2}:[0-9]{2}$ ]]; then
    echo "00:$t"
  else
    echo "$t"
  fi
}

s=$(norm_time "$start")
e=$(norm_time "$end")

# 极速切片（不重编码）——可能不够“帧级精准”
ffmpeg -y -ss "$s" -to "$e" -i "$in" -c copy -avoid_negative_ts make_zero "$out"
echo "✅ Done: $out"
