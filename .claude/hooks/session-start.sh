#!/bin/bash
# Облачные сессии Claude Code: ставим зависимости лаборатории сигналов, чтобы тесты signal_lab шли без ручной установки.
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}"
python3 -m pip install -q -r signal_lab/requirements.txt
