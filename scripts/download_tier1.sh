#!/usr/bin/env bash
# Tier-1 dataset downloads for SIH 26052.
# Resumable (curl -C -). Safe to re-run: files with a .done marker are skipped.
RAW=/c/SIH26052_data/raw
LOG=/c/SIH26052_data/raw/_download.log
mkdir -p "$RAW"

# OpenSLR mirrors, tried in order. NOTE: us.openslr.org does not resolve - do not add it back.
SLR_MIRRORS=(https://www.openslr.org https://openslr.trmal.net https://openslr.elda.org)

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

fetch () {  # fetch <url> <outfile>
  local url="$1" out="$RAW/$2"
  [ -f "$out.done" ] && { log "skip  $2 (already complete)"; return 0; }
  log "get   $2"
  if curl -sSL --fail --retry 5 --retry-delay 10 --retry-connrefused -C - -o "$out" "$url"; then
    touch "$out.done"; log "ok    $2 ($(du -h "$out" | cut -f1))"; return 0
  fi
  return 1
}

fetch_slr () {  # fetch_slr <resource_no> <filename>
  local out="$RAW/$2"
  [ -f "$out.done" ] && { log "skip  $2 (already complete)"; return 0; }
  for m in "${SLR_MIRRORS[@]}"; do
    log "get   $2  <- $m"
    if curl -sSL --fail --retry 3 --retry-delay 10 --retry-connrefused -C - -o "$out" "$m/resources/$1/$2"; then
      touch "$out.done"; log "ok    $2 ($(du -h "$out" | cut -f1))"; return 0
    fi
    log "      mirror failed, trying next"
  done
  log "FAIL  $2 - all mirrors exhausted"
}

# --- transients tier 1: ESC-50 (CC BY-NC 3.0), smallest, do first ---
fetch https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip esc50.zip

# --- clean speech (16 kHz FLAC, CC BY 4.0) ---
fetch_slr 12 dev-clean.tar.gz
fetch_slr 12 test-clean.tar.gz

# --- room impulse responses (OpenSLR 28) ---
fetch_slr 28 rirs_noises.zip

# --- clean speech, the big one ---
fetch_slr 12 train-clean-100.tar.gz

# --- steady background noise (MUSAN, CC BY 4.0) ---
fetch_slr 17 musan.tar.gz

log "=== tier-1 download pass complete ==="
ls -la "$RAW" | tee -a "$LOG"
