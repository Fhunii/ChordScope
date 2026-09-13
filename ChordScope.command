#!/bin/bash
cd "$(dirname "$0")" || exit 1
xattr -dr com.apple.quarantine "$PWD" 2>/dev/null || true
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "Python 3.10が必要です。ダウンロードページを開きます。"
  open "https://www.python.org/downloads/release/python-31019/"
  echo "インストール後、もう一度ChordScope.commandを開いてください。"
  read -r
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "初回セットアップ中です。数分かかることがあります。"
  python3.10 -m venv .venv || exit 1
  .venv/bin/python -m pip install --upgrade pip || exit 1
  .venv/bin/pip install -r requirements.txt || exit 1
  .venv/bin/pip install "basic-pitch>=0.4.0,<1" || exit 1
fi
echo "起動後もこのターミナルは閉じないでください。"
(sleep 2; open "http://localhost:3000") &
exec .venv/bin/python server.py
