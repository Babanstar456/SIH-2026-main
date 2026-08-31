#!/usr/bin/env bash
# Transient / impulsive-noise sources. Runs in parallel with download_tier1.sh.
RAW=/c/SIH26052_data/raw
LOG=/c/SIH26052_data/raw/_download_transients.log
mkdir -p "$RAW"
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

fetch () {  # fetch <url> <outfile>
  local url="$1" out="$RAW/$2"
  [ -f "$out.done" ] && { log "skip  $2"; return 0; }
  log "get   $2"
  if curl -sSL --fail --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$out" "$url"; then
    touch "$out.done"; log "ok    $2 ($(du -h "$out" | cut -f1))"
  else
    log "FAIL  $2"
  fi
}

# Real firearm recordings, outdoor range, multi-firearm/multi-orientation. CC BY 4.0.
# This is our primary gunshot source - ESC-50 has NO gun_shot class (only fireworks/thunderstorm).
fetch "https://zenodo.org/records/7004819/files/README.md?download=1" gunshot_edge_README.md
fetch "https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1" gunshot_edge.zip

# UrbanSound8K - open Zenodo mirror, no request form. CC BY-NC 4.0.
# Gives gun_shot (374), siren, engine_idling, jackhammer, drilling.
fetch "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1" UrbanSound8K.tar.gz

log "=== transient download pass complete ==="
