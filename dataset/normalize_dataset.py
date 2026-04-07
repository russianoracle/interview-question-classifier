#!/usr/bin/env python3
"""
normalize_dataset.py — нормализует dataset.jsonl:
  1. Фильтрует ТОЛЬКО завершённые VTT-фрагменты без признаков вопроса
     (незавершённые — доверяем label_sample, там мог быть ? в следующем блоке)
  2. Downsample actionable до TARGET_RATIO × |non_actionable|
  3. Сохраняет dataset_normalized.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).parent
IN_JSONL = ROOT / "processed" / "dataset.jsonl"
OUT_JSONL = ROOT / "processed" / "dataset_normalized.jsonl"

TARGET_RATIO = 3  # actionable : non_actionable

QUESTION_RX = re.compile(
    r"[?？]|"
    r"\b(что|как|где|когда|зачем|почему|кто|какой|какая|какое|какие|"
    r"сколько|чем|каким образом|расскажи|расскажите|объясни|объясните|"
    r"опиши|опишите|назови|перечисли|сравни|реализуй|напиши|покажи)\b",
    re.IGNORECASE,
)

# Завершённое предложение заканчивается на . ? !
SENTENCE_END_RX = re.compile(r"[.?!]\s*$")


def is_vtt(sample: dict) -> bool:
    """True если источник — VTT-субтитры (не curated, не github repo)."""
    src = sample.get("source", "")
    return src != "curated" and "/" not in src


def should_drop(sample: dict) -> bool:
    """
    Убираем actionable только если выполнены ВСЕ три условия:
    - источник VTT (не curated / не github)
    - предложение завершено (финальный . ? !)  ← незавершённые не трогаем,
      там ? мог оказаться в следующем VTT-блоке
    - нет ни одного признака вопроса
    """
    text = sample["text"]
    return (
        is_vtt(sample)
        and bool(SENTENCE_END_RX.search(text))
        and not bool(QUESTION_RX.search(text))
    )


def main() -> None:
    samples = [
        json.loads(l)
        for l in IN_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Исходных записей: {len(samples)}")

    # ── 1. Разбиваем по классам ─────────────────────────────────────────────
    non_act = [s for s in samples if s["head2_label"] == "non_actionable"]
    act_raw = [s for s in samples if s["head2_label"] == "actionable"]

    # ── 2. Фильтр: завершённые VTT без вопроса ──────────────────────────────
    act_filtered = [s for s in act_raw if not should_drop(s)]
    act_dropped = len(act_raw) - len(act_filtered)
    print(f"actionable до фильтра: {len(act_raw)}")
    print(f"  → убрано завершённых VTT без признака вопроса: {act_dropped}")
    print(f"  → осталось: {len(act_filtered)}")
    print(f"non_actionable: {len(non_act)}")

    # ── 3. Downsample actionable ─────────────────────────────────────────────
    target_act = TARGET_RATIO * len(non_act)
    if len(act_filtered) > target_act:
        # Приоритет при сэмплировании: curated сохраняем полностью
        curated = [s for s in act_filtered if s["source"] == "curated"]
        vtt = [s for s in act_filtered if s["source"] != "curated"]
        vtt_need = target_act - len(curated)
        vtt_sampled = random.sample(vtt, min(max(vtt_need, 0), len(vtt)))
        act_final = curated + vtt_sampled
        print(
            f"Downsample: {len(act_filtered)} → {len(act_final)} "
            f"(curated={len(curated)}, vtt={len(vtt_sampled)})"
        )
    else:
        act_final = act_filtered
        print(f"Downsample не нужен: {len(act_final)} ≤ {target_act}")

    # ── 4. Итоговая выборка ──────────────────────────────────────────────────
    result = act_final + non_act
    random.shuffle(result)

    ratio = len(act_final) / max(1, len(non_act))
    print(
        f"\nИтог: {len(result)} записей  "
        f"(actionable={len(act_final)}, non_actionable={len(non_act)}, "
        f"ratio={ratio:.1f}:1)"
    )
    has_punct = sum(1 for s in result if s["has_punct"])
    mixed = sum(
        1 for s in result if s.get("head3_query_start") and s["head3_query_start"] > 0
    )
    print(f"has_punct={has_punct}, mixed(query_start>0)={mixed}")

    # ── 5. Сохранение ────────────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in result:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ Сохранено: {OUT_JSONL}")

    print("\n── actionable ──")
    for s in [s for s in result if s["head2_label"] == "actionable"][:8]:
        print(f"  {s['text'][:90]}")

    print("\n── non_actionable ──")
    for s in [s for s in result if s["head2_label"] == "non_actionable"][:8]:
        print(f"  {s['text'][:90]}")


if __name__ == "__main__":
    main()
