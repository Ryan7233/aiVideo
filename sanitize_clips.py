import json, sys

DUR = float(sys.argv[1])  # 视频总秒数，如 92.4
clips = json.load(sys.stdin)["clips"]
def to_sec(s):
    parts = s.split(':')
    if len(parts)==2:
        m, s = parts
        return int(m)*60 + int(s)
    h,m,s = parts
    return int(h)*3600+int(m)*60+int(float(s))
def to_mmss(x):
    x = max(0, int(x))
    return f"{x//60:02d}:{x%60:02d}"

out = []
last_end = -1
for c in clips:
    st, ed = to_sec(c["start"]), to_sec(c["end"])
    st = max(0, st)
    ed = min(int(DUR), ed)
    if ed - st < 8:   # 太短的段落丢弃（可调）
        continue
    if st < last_end: # 重叠过多，略微推后
        st = last_end
    if ed - st >= 8:
        out.append({"start": to_mmss(st), "end": to_mmss(ed), "reason": c.get("reason","")})
        last_end = ed
print(json.dumps({"clips": out}, ensure_ascii=False))
