#!/usr/bin/env bash
# Сборка финальных роликов со звуком.
#   ./build.sh           -> promo_sound.mp4 (48 с) и ad_18s.mp4 (18.6 с, рекламная нарезка)
# Нужны: promo.mp4 (node render.js), python3 + numpy + scipy, ffmpeg ($FFMPEG или из PATH).
set -euo pipefail
cd "$(dirname "$0")"
FF="${FFMPEG:-ffmpeg}"
LOUD="loudnorm=I=-14:TP=-1:LRA=9"          # громкость под соцсети

python3 sound.py full
python3 sound.py ad

# полный ролик
"$FF" -y -loglevel error -i promo.mp4 -i promo_audio.wav -map 0:v -map 1:a -c:v copy \
  -af "$LOUD" -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart promo_sound.mp4

# рекламная нарезка: куски из promo.mp4, стыки ровно в долю 120 BPM (см. AD_CUTS в sound.py)
CUTS=$(python3 -c "from sound import AD_CUTS; print(' '.join(f'{a}:{b}' for a, b in AD_CUTS))")
FILTER=""; LABELS=""; i=0
for c in $CUTS; do
  a=${c%%:*}; b=${c##*:}
  FILTER+="[0:v]trim=start=$a:end=$b,setpts=PTS-STARTPTS[v$i];"; LABELS+="[v$i]"; i=$((i+1))
done
FILTER+="${LABELS}concat=n=$i:v=1:a=0,fps=30[v]"
"$FF" -y -loglevel error -i promo.mp4 -i ad_audio.wav -filter_complex "$FILTER" -map "[v]" -map 1:a \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -af "$LOUD" -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart ad_18s.mp4

ls -la promo_sound.mp4 ad_18s.mp4
