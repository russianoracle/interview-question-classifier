#!/usr/bin/env python3
"""
normalize_dataset.py — нормализует dataset.jsonl:
  1. Очищает мусор: код, URL, эмодзи, HTML, markdown-артефакты
  2. Для actionable из GitHub — оставляем только строки с явным признаком вопроса
  3. Фильтрует завершённые VTT-фрагменты без признаков вопроса
  4. Downsample actionable до TARGET_RATIO × |non_actionable|
  5. Сохраняет dataset_normalized.jsonl
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

# Мусорные паттерны
URL_RX = re.compile(r"https?://\S+")
EMOJI_RX = re.compile(r"[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2600-\u27BF]")
HTML_RX = re.compile(r"</?[a-zA-Z][^>]*>")
CODE_CHARS_RX = re.compile(
    r"[{};]|</|/>|::|\bpublic\b|\bprivate\b|\bstatic\b|\bvoid\b|\breturn\b|\bnull\b|^\s*//|^\s*/\*"
)
# Numbered list item: "147. Тестирование" или "+ Создать объект"
LIST_ITEM_RX = re.compile(r"^[\s]*([0-9]+\.\s+|\+\s+|\-\s+[А-ЯA-Z])")
# Строки с кодом: начинаются с символа программирования
CODE_LINE_RX = re.compile(
    r"^[\s]*(@|#include|import |from \w+ import|def |class |function |var |const |let |SELECT |INSERT |CREATE |ALTER |DROP )"
)


def is_vtt(sample: dict) -> bool:
    """True если источник — VTT-субтитры (не curated, не github repo)."""
    src = sample.get("source", "")
    return src != "curated" and "/" not in src


def is_github(sample: dict) -> bool:
    src = sample.get("source", "")
    return "/" in src and src != "curated"


def is_junk(text: str, min_words: int = 2) -> bool:
    """True если строка явно является мусором."""
    if URL_RX.search(text):
        return True
    if EMOJI_RX.search(text):
        return True
    if HTML_RX.search(text):
        return True
    if CODE_CHARS_RX.search(text):
        return True
    if CODE_LINE_RX.search(text):
        return True
    # Пункт списка без вопроса (например "+ Создать объект Statement")
    if LIST_ITEM_RX.match(text) and not QUESTION_RX.search(text):
        return True
    if len(text.split()) < min_words:
        return True
    return False


def should_drop_vtt(sample: dict) -> bool:
    """
    Убираем VTT actionable только если:
    - предложение завершено (. ? !)
    - нет ни одного признака вопроса
    """
    text = sample["text"]
    return bool(SENTENCE_END_RX.search(text)) and not bool(QUESTION_RX.search(text))


def main() -> None:
    samples = [
        json.loads(l)
        for l in IN_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Исходных записей: {len(samples)}")

    # ── 1. Разбиваем по классам ─────────────────────────────────────────────
    non_act_raw = [s for s in samples if s["head2_label"] == "non_actionable"]
    act_raw = [s for s in samples if s["head2_label"] == "actionable"]

    # ── 2. Очистка мусора (URL, код, HTML, эмодзи, короткие) ────────────────
    # actionable: мин. 4 слова (иначе обрезанный вопрос бесполезен)
    # non_actionable: мин. 2 слова (VTT-фрагменты изначально короткие)
    act_clean = [s for s in act_raw if not is_junk(s["text"], min_words=4)]
    non_act_clean = [s for s in non_act_raw if not is_junk(s["text"], min_words=2)]
    print(
        f"actionable:     {len(act_raw)} → после очистки мусора: {len(act_clean)} (−{len(act_raw) - len(act_clean)})"
    )
    print(
        f"non_actionable: {len(non_act_raw)} → после очистки мусора: {len(non_act_clean)} (−{len(non_act_raw) - len(non_act_clean)})"
    )

    # ── 3. GitHub actionable — оставляем только явные вопросы ───────────────
    def keep_actionable(s: dict) -> bool:
        # GitHub-источники часто содержат пункты без ?, доверяем только тем, у которых есть вопрос
        if is_github(s):
            return bool(QUESTION_RX.search(s["text"]))
        return True

    act_filtered = [s for s in act_clean if keep_actionable(s)]
    print(
        f"actionable после фильтра GitHub (только с признаком вопроса): {len(act_filtered)} (−{len(act_clean) - len(act_filtered)})"
    )

    # ── 4. VTT: убираем завершённые фрагменты без признака вопроса ──────────
    act_filtered2 = [s for s in act_filtered if not (is_vtt(s) and should_drop_vtt(s))]
    print(
        f"actionable после фильтра VTT (завершённые без ?): {len(act_filtered2)} (−{len(act_filtered) - len(act_filtered2)})"
    )
    act_filtered = act_filtered2
    non_act = non_act_clean
    print(f"non_actionable (чистые): {len(non_act)}")

    # ── 5. Downsample actionable ─────────────────────────────────────────────
    target_act = TARGET_RATIO * len(non_act)
    if len(act_filtered) > target_act:
        # Приоритет: curated → github (очищенные вопросы) → vtt
        curated = [s for s in act_filtered if s["source"] == "curated"]
        github = [s for s in act_filtered if is_github(s)]
        vtt = [s for s in act_filtered if is_vtt(s)]
        need = target_act - len(curated)
        # Сначала берём все github, потом добираем из vtt
        gh_take = min(len(github), max(need, 0))
        gh_sampled = (
            random.sample(github, gh_take) if gh_take < len(github) else list(github)
        )
        vtt_need = need - len(gh_sampled)
        vtt_sampled = random.sample(vtt, min(max(vtt_need, 0), len(vtt)))
        act_final = curated + gh_sampled + vtt_sampled
        print(
            f"Downsample: {len(act_filtered)} → {len(act_final)} "
            f"(curated={len(curated)}, github={len(gh_sampled)}, vtt={len(vtt_sampled)})"
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
