#!/usr/bin/env bash
set -euo pipefail

in="$1"; start="$2"; end="$3"; out="$4"

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

# 计算段长（to - ss），这里用 ffmpeg 内部处理，直接给 -to 绝对时间也可
ffmpeg -y -ss "$s" -to "$e" -i "$in"   -vf "fps=30,scale=1280:-2"   -c:v libx264 -preset veryfast -crf 23   -c:a aac -b:a 128k   -movflags +faststart   -avoid_negative_ts make_zero   "$out"

echo "✅ Precise cut: $out"
