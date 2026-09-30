#!/usr/bin/env bash
# Сборка финальных роликов со звуком.
#   ./build.sh [hard|deep|funk]   (по умолчанию hard)
#   hard -> promo_sound.mp4, ad_18s.mp4 ;  deep/funk -> promo_sound_<стиль>.mp4, ad_18s_<стиль>.mp4
# Нужны: promo.mp4 (node render.js), python3 + numpy + scipy, ffmpeg ($FFMPEG или из PATH).
set -euo pipefail
cd "$(dirname "$0")"
FF="${FFMPEG:-ffmpeg}"; PY="${PYTHON:-python3}"
STYLE="${1:-hard}"; SUF=""; [ "$STYLE" != hard ] && SUF="_$STYLE"
LOUD="loudnorm=I=-14:TP=-1:LRA=9"          # громкость под соцсети

"$PY" sound.py full "$STYLE"
"$PY" sound.py ad "$STYLE"

# полный ролик
"$FF" -y -loglevel error -i promo.mp4 -i "promo_audio$SUF.wav" -map 0:v -map 1:a -c:v copy \
  -af "$LOUD" -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "promo_sound$SUF.mp4"

# рекламная нарезка: куски из promo.mp4, стыки ровно в долю 120 BPM (см. AD_CUTS в sound.py)
CUTS=$("$PY" -c "from sound import AD_CUTS; print(' '.join(f'{a}:{b}' for a, b in AD_CUTS))")
FILTER=""; LABELS=""; i=0
for c in $CUTS; do
  a=${c%%:*}; b=${c##*:}
  FILTER+="[0:v]trim=start=$a:end=$b,setpts=PTS-STARTPTS[v$i];"; LABELS+="[v$i]"; i=$((i+1))
done
FILTER+="${LABELS}concat=n=$i:v=1:a=0,fps=30[v]"
"$FF" -y -loglevel error -i promo.mp4 -i "ad_audio$SUF.wav" -filter_complex "$FILTER" -map "[v]" -map 1:a \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -af "$LOUD" -c:a aac -b:a 256k -ar 48000 -shortest -movflags +faststart "ad_18s$SUF.mp4"

ls -la "promo_sound$SUF.mp4" "ad_18s$SUF.mp4"
