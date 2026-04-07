#!/bin/bash
# download_subs.sh — скачивает русские субтитры для списка YouTube-ID
# Использование: bash download_subs.sh id1 id2 ...

SUBS_DIR="$(dirname "$0")/subtitles"
mkdir -p "$SUBS_DIR"

ok=0; skip=0; fail=0

for id in "$@"; do
  # Уже есть?
  if ls "$SUBS_DIR/$id/"*.vtt 2>/dev/null | grep -q .; then
    echo "  [skip] $id (уже есть)"
    ((skip++))
    continue
  fi

  mkdir -p "$SUBS_DIR/$id"
  result=$(yt-dlp "https://www.youtube.com/watch?v=$id" \
    --write-sub --write-auto-sub --sub-lang ru \
    --skip-download --no-warnings \
    -o "$SUBS_DIR/$id/%(title)s" 2>&1)

  if ls "$SUBS_DIR/$id/"*.vtt 2>/dev/null | grep -q .; then
    echo "  [ok] $id"
    ((ok++))
  else
    rmdir "$SUBS_DIR/$id" 2>/dev/null
    if echo "$result" | grep -qi "private\|removed\|unavailable"; then
      echo "  [unavail] $id"
    else
      echo "  [no-ru-subs] $id"
    fi
    ((fail++))
  fi
done

echo ""
echo "✅ скачано=$ok  ⏭ уже было=$skip  ❌ нет субтитров/недоступно=$fail"
