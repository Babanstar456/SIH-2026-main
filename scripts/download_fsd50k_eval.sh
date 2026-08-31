#!/usr/bin/env bash
# FSD50K EVAL split only (~6.2 GB), not the full dev set (~24.7 GB).
#
# Rationale: we need noise-source DIVERSITY, not a training corpus for a
# classifier. The eval split carries all 200 classes across ~10,200 clips, which
# is ample for the two categories measured to be thin - artillery (66 source
# clips) and rotor (72) - at a fifth of the download cost.
#
# Classes we care about: Explosion, Gunshot_and_gunfire, Fireworks, Boom,
# Aircraft, Engine, Siren.
#
# NOTE: the audio is a SPLIT zip (.z01 + .zip). Python's zipfile cannot read
# split archives; 7-Zip joins them transparently when pointed at the .zip.

RAW=/c/SIH26052_data/raw
LOG=$RAW/_download_fsd50k.log
mkdir -p "$RAW"
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

Z=https://zenodo.org/records/4060432/files

fetch () {  # fetch <filename> <expected_bytes|0>
  local name="$1" expect="${2:-0}" out="$RAW/$1"
  [ -f "$out.done" ] && { log "skip  $name"; return 0; }
  for i in $(seq 1 60); do
    have=$( [ -f "$out" ] && stat -c %s "$out" 2>/dev/null || echo 0 )
    if [ "$expect" -gt 0 ] && [ "$have" -ge "$expect" ]; then
      touch "$out.done"; log "ok    $name ($have bytes)"; return 0
    fi
    log "get   $name (try $i, have ${have}B)"
    if curl -sSL --fail --retry 3 --retry-delay 5 --retry-connrefused \
            --speed-time 60 --speed-limit 1024 -C - -o "$out" "$Z/$name?download=1"; then
      [ "$expect" -eq 0 ] && { touch "$out.done"; log "ok    $name"; return 0; }
    fi
    sleep 4
  done
  log "FAIL  $name"
}

# Metadata first - tiny, and it is what maps clips to classes.
fetch FSD50K.ground_truth.zip 0
fetch FSD50K.metadata.zip 0

# Audio: split archive, both parts required.
fetch FSD50K.eval_audio.z01 3221225472
fetch FSD50K.eval_audio.zip 0

log "=== fsd50k eval download complete ==="
ls -la "$RAW" | grep -i fsd | tee -a "$LOG"
