#!/usr/bin/env bash
# 闸 C 的一键复验：从干净环境起，只照 README 说的做。
#   bash scripts/verify.sh          验现有产物
#   bash scripts/verify.sh --twice  另加一次重建，比对确定性
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "== 没有 .venv，按 README 建一个 =="
  python -m venv .venv
  PY=./.venv/Scripts/python.exe
  [ -x "$PY" ] || PY=./.venv/bin/python
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi

echo "== 单元断言 =="
PYTHONPATH=src "$PY" -m pytest tests -q

echo "== 产物断言 =="
PYTHONPATH=src "$PY" scripts/verify.py "$@"
