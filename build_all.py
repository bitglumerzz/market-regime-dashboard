#!/usr/bin/env python3
"""Сборка всего проекта REGIME AI одной командой — Windows, macOS, Linux.

    python build_all.py                  # всё: ролики (3 стиля звука), нарезки, интро для X, проверка бота
    python build_all.py --styles deep    # только один стиль звука
    python build_all.py --only bot       # только тесты бота   (варианты: promo, intro, bot)
    python build_all.py --fast           # проверочный прогон: 3 кадра вместо полного рендера

Нужны: Python 3.9+ и Node.js 18+ (npm). Всё остальное скрипт ставит сам в локальные папки проекта:
.venv (numpy, scipy, imageio-ffmpeg, aiogram…), node_modules (Playwright) и Chromium для рендера.
ffmpeg берётся из PATH, а если его нет — из pip-пакета imageio-ffmpeg.

Результаты:
    studio/promo/promo_sound.mp4, promo_sound_deep.mp4, promo_sound_funk.mp4   — 48 с со звуком
    studio/promo/ad_18s.mp4,      ad_18s_deep.mp4,      ad_18s_funk.mp4        — рекламные нарезки 18.6 с
    studio/intro/intro.mp4                                                     — интро для X, 15 с
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMO, INTRO, BOT = ROOT / "studio" / "promo", ROOT / "studio" / "intro", ROOT / "bot"
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"
LOUD = "loudnorm=I=-14:TP=-1:LRA=9"            # громкость под соцсети
STYLES = ("hard", "deep", "funk")


def say(msg: str) -> None:
    print(f"\n\033[1;33m==> {msg}\033[0m" if sys.stdout.isatty() else f"\n==> {msg}", flush=True)


def run(cmd: list, cwd: Path = ROOT, env: dict | None = None, capture: bool = False) -> str:
    cmd = [str(c) for c in cmd]
    print("   $", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True, capture_output=capture)
    return res.stdout if capture else ""


def fail(msg: str) -> None:
    print(f"\nОШИБКА: {msg}", file=sys.stderr)
    sys.exit(1)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def setup_python(need_bot: bool) -> Path:
    say("1. Python-окружение (.venv)")
    if sys.version_info < (3, 9):
        fail(f"нужен Python 3.9+, сейчас {sys.version.split()[0]}")
    if not venv_python().exists():
        run([sys.executable, "-m", "venv", VENV])
    py = venv_python()
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-q", "-r", PROMO / "requirements.txt"])
    if need_bot:
        run([py, "-m", "pip", "install", "-q", "-r", BOT / "requirements-dev.txt"])
    return py


def setup_node() -> str:
    say("2. Node.js, Playwright и Chromium")
    npm, npx, node = (shutil.which(x) for x in ("npm", "npx", "node"))
    if not (npm and npx and node):
        fail("не найден Node.js/npm. Установите Node.js 18+ с https://nodejs.org (LTS) и перезапустите терминал.")
    major = int(run([node, "--version"], capture=True).strip().lstrip("v").split(".")[0])
    if major < 18:
        fail(f"нужен Node.js 18+, сейчас {major}")
    run([npm, "install", "--no-fund", "--no-audit"], cwd=PROMO)
    if os.environ.get("SKIP_BROWSER_INSTALL") != "1":
        run([npx, "playwright", "install", "chromium"], cwd=PROMO)
    return node


def find_ffmpeg(py: Path) -> str:
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ff:
        ff = run([py, "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"], capture=True).strip()
    print(f"   ffmpeg: {ff}")
    return ff


def render(node: str, ff: str, page: Path | None, fast: bool) -> None:
    env = {**os.environ, "FFMPEG": ff}
    cmd = [node, "render.js"]
    if page:
        cmd += ["--page", page]
    if fast:
        cmd += ["--stills", "0.5,12.8,30" if page is None else "1,6,13", "--dir", (page.parent if page else PROMO)]
    t0 = time.time()
    run(cmd, cwd=PROMO, env=env)
    print(f"   готово за {time.time() - t0:.0f} с")


def build_sound(py: Path, ff: str, style: str) -> None:
    suf = "" if style == "hard" else f"_{style}"
    run([py, "sound.py", "full", style], cwd=PROMO)
    run([py, "sound.py", "ad", style], cwd=PROMO)
    video = PROMO / "promo.mp4"
    run([ff, "-y", "-loglevel", "error", "-i", video, "-i", PROMO / f"promo_audio{suf}.wav", "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-af", LOUD, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
         PROMO / f"promo_sound{suf}.mp4"])
    cuts = run([py, "-c", "from sound import AD_CUTS; print(' '.join(f'{a}:{b}' for a, b in AD_CUTS))"], cwd=PROMO, capture=True).split()
    parts, labels = [], ""
    for i, c in enumerate(cuts):
        a, b = c.split(":")
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v{i}]")
        labels += f"[v{i}]"
    flt = ";".join(parts) + f";{labels}concat=n={len(cuts)}:v=1:a=0,fps=30[v]"
    run([ff, "-y", "-loglevel", "error", "-i", video, "-i", PROMO / f"ad_audio{suf}.wav", "-filter_complex", flt,
         "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-af", LOUD, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest", "-movflags", "+faststart", PROMO / f"ad_18s{suf}.mp4"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("promo", "intro", "bot"), action="append", help="собрать только часть (можно несколько раз)")
    ap.add_argument("--styles", default=",".join(STYLES), help="стили звука через запятую: hard,deep,funk")
    ap.add_argument("--fast", action="store_true", help="проверочный прогон: несколько кадров вместо полного рендера")
    a = ap.parse_args()
    parts = set(a.only or ("promo", "intro", "bot"))
    styles = [s.strip() for s in a.styles.split(",") if s.strip()]
    if bad := [s for s in styles if s not in STYLES]:
        fail(f"неизвестный стиль: {', '.join(bad)} (есть: {', '.join(STYLES)})")

    t0 = time.time()
    py = setup_python(need_bot="bot" in parts)
    made: list[Path] = []
    if parts & {"promo", "intro"}:
        node = setup_node()
        ff = find_ffmpeg(py)
        if "promo" in parts:
            say("3. Промо-ролик: покадровый рендер (1449 кадров; 3–15 минут в зависимости от компьютера)")
            render(node, ff, None, a.fast)
            if not a.fast:
                say("4. Звук, сведение и рекламные нарезки")
                for st in styles:
                    print(f"   стиль: {st}")
                    build_sound(py, ff, st)
                    suf = "" if st == "hard" else f"_{st}"
                    made += [PROMO / f"promo_sound{suf}.mp4", PROMO / f"ad_18s{suf}.mp4"]
        if "intro" in parts:
            say("5. Интро для X (логотип и обложку положите в studio/intro/assets/logo.png и cover.jpg)")
            render(node, ff, INTRO / "index.html", a.fast)
            if not a.fast:
                made.append(INTRO / "intro.mp4")
    if "bot" in parts:
        say("6. Проверка Telegram-бота (тесты)")
        run([py, "-m", "pytest", "-q"], cwd=BOT)

    say(f"Готово за {(time.time() - t0) / 60:.1f} мин")
    for f in made:
        print(f"   {f.relative_to(ROOT)}  ({f.stat().st_size / 1e6:.1f} МБ)" if f.exists() else f"   НЕТ ФАЙЛА: {f}")
    if a.fast:
        print("   --fast: построены только проверочные кадры still_*.png в studio/promo и studio/intro")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        fail(f"команда завершилась с ошибкой (код {e.returncode}). Прокрутите вывод выше — там причина.")
    except KeyboardInterrupt:
        fail("прервано")
