#!/usr/bin/env bash
# Persistent resuming downloader for hosts that drop long transfers (Zenodo).
# Keeps resuming from the current byte offset until the file reaches its
# advertised Content-Length, or until MAX_TRIES is exhausted.
#
#   bash scripts/fetch_resume.sh <url> <outfile> [max_tries]

URL="$1"; OUT="$2"; MAX_TRIES="${3:-60}"
[ -z "$URL" ] || [ -z "$OUT" ] && { echo "usage: fetch_resume.sh <url> <out> [tries]"; exit 2; }

expected=$(curl -sIL --max-time 30 "$URL" | tr -d '\r' \
           | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{print v}')
[ -z "$expected" ] && { echo "could not determine size for $OUT"; expected=0; }

# Some hosts (datashare.ed.ac.uk) answer HEAD with a small HTML interstitial
# rather than the file, so a tiny Content-Length is a lie, not a target. Trusting
# it marks a truncated download "complete".
if [ "$expected" -gt 0 ] && [ "$expected" -lt 1048576 ]; then
  echo "ignoring implausible Content-Length ${expected} (likely an HTML interstitial)"
  expected=0
fi
echo "target size: ${expected:-unknown} bytes -> $OUT"

for i in $(seq 1 "$MAX_TRIES"); do
  have=$( [ -f "$OUT" ] && stat -c %s "$OUT" 2>/dev/null || echo 0 )
  if [ "$expected" -gt 0 ] && [ "$have" -ge "$expected" ]; then
    echo "[ok] $OUT complete ($have bytes) after $((i-1)) resume(s)"
    touch "$OUT.done"; exit 0
  fi
  pct=0; [ "$expected" -gt 0 ] && pct=$(( have * 100 / expected ))
  echo "[try $i/$MAX_TRIES] have ${have}B (${pct}%), resuming..."
  curl -sSL --fail --retry 3 --retry-delay 5 --retry-connrefused \
       --speed-time 60 --speed-limit 1024 \
       -C - -o "$OUT" "$URL" && { touch "$OUT.done"; \
       echo "[ok] $OUT complete"; exit 0; }
  sleep 5
done

echo "[FAIL] $OUT did not complete after $MAX_TRIES attempts"
exit 1
