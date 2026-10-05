#!/bin/sh
# Build the anonymised supplementary archive: the package, its tests and docs,
# the result files, the exported live-model completions and the figures.  The
# Refactory corpus is not included (it is public and is not ours to
# redistribute), nor are caches of offline completions, which regenerate.
set -e
cd "$(dirname "$0")/.."
python3 experiments/export_live_cache.py
OUT="${1:-dmas_supplementary.zip}"
rm -f "$OUT"
zip -qr "$OUT" \
  dmas LICENSE Makefile README.md pyproject.toml \
  src tests docs examples experiments \
  -x '*/__pycache__/*' -x '*.pyc' -x '*/.pytest_cache/*' -x '*.DS_Store'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
