#!/usr/bin/env bash
set -e
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/ida_gorev.py
