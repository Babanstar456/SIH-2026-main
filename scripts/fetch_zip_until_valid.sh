#!/usr/bin/env bash
# Resume a zip download until the archive actually VALIDATES.
#
# Content-length is not trustworthy on every host - datashare.ed.ac.uk answers
# HEAD with a ~4 KB HTML interstitial, so a size-based completion check happily
# declares a 10%-downloaded file finished. Testing the zip's central directory
# is a completion check that cannot be fooled.
#
#   bash scripts/fetch_zip_until_valid.sh <url> <outfile> [max_tries]

URL="$1"; OUT="$2"; MAX="${3:-60}"
PY="C:/SIH26052_data/.venv/Scripts/python.exe"
[ -z "$URL" ] || [ -z "$OUT" ] && { echo "usage: <url> <out> [tries]"; exit 2; }

valid () {
  "$PY" -c "
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1])
    sys.exit(0 if z.namelist() else 1)
except Exception:
    sys.exit(1)
" "$1" 2>/dev/null
}

for i in $(seq 1 "$MAX"); do
  have=$( [ -f "$OUT" ] && stat -c %s "$OUT" 2>/dev/null || echo 0 )
  if [ "$have" -gt 0 ] && valid "$OUT"; then
    echo "[ok] $(basename "$OUT") is a complete, valid zip ($have bytes) after $((i-1)) resume(s)"
    touch "$OUT.done"; exit 0
  fi
  echo "[try $i/$MAX] $(basename "$OUT"): have ${have}B, not yet a valid archive - resuming"
  curl -sSL --fail --retry 3 --retry-delay 5 --retry-connrefused \
       --speed-time 60 --speed-limit 1024 -C - -o "$OUT" "$URL"
  sleep 3
done

echo "[FAIL] $(basename "$OUT") never validated after $MAX attempts"
exit 1
