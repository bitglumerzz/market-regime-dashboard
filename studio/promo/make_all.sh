#!/usr/bin/env bash
# Полная сборка промо на своей машине — одной командой:
#   cd studio/promo && ./make_all.sh
# Результат: promo_sound.mp4 (48 с, со звуком) и ad_18s.mp4 (18.6 с, рекламная нарезка).
# Нужны: Node.js 18+, Python 3.9+. ffmpeg можно не ставить — подтянется из pip (imageio-ffmpeg).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> 1/4 зависимости Python (numpy, scipy, imageio-ffmpeg)"
python3 -m venv .venv 2>/dev/null || true
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -m pip install -q --upgrade pip >/dev/null
"$PY" -m pip install -q -r requirements.txt

echo "==> 2/4 зависимости Node (Playwright + Chromium)"
npm install --silent
[ "${SKIP_BROWSER_INSTALL:-0}" = 1 ] || npx playwright install chromium

export FFMPEG="${FFMPEG:-$(command -v ffmpeg || "$PY" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')}"
echo "    ffmpeg: $FFMPEG"

echo "==> 3/4 рендер видео покадрово (1449 кадров, 3–10 минут)"
node render.js "$@"

echo "==> 4/4 звук + сведение + рекламная нарезка"
PYTHON="$PY" ./build.sh
echo "Готово: $(pwd)/promo_sound.mp4 и $(pwd)/ad_18s.mp4"
