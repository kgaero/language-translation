#!/usr/bin/env bash
set -euo pipefail

pip install -q -r requirements.txt
python train.py
