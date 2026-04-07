#!/usr/bin/env python3
"""
normalize_dataset.py — нормализует dataset.jsonl:

  1. Очищает мусор: код, URL, эмодзи, HTML, markdown-артефакты
  2. Для actionable из GitHub — оставляем только строки с явным признаком вопроса
  3. Удаляет СТАРЫЕ non_actionable (3-словные фрагменты — конфаундер по длине)
  4. Генерирует НОВЫЕ non_actionable из actionable-корпуса:
       завершённые предложения (≥6 слов, .!), без признаков вопроса/техники
       → реальные ответы и утверждения из интервью, длина сопоставима с actionable
  5. Балансирует 3:1 (actionable:non_actionable), curated и github в приоритете
  6. Сохраняет dataset_normalized.jsonl
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
TARGET_NACT = 8_000  # желаемое число non_actionable (добиваем из actioable-корпуса)

# ── Паттерны ──────────────────────────────────────────────────────────────────

QUESTION_RX = re.compile(
    r"[?？]|"
    r"\b(где|когда|зачем|почему|кто|какой|какая|какое|какие|"
    r"сколько|каким образом|расскажи|расскажите|объясни|объясните|"
    r"опиши|опишите|назови|перечисли|сравни|реализуй|напиши|покажи)\b",
    re.IGNORECASE,
)

# Отдельный паттерн для nact-фильтра: только однозначные признаки вопроса (? или imperative)
# "что" и "как" слишком широкие (ловят "то что", "как правило") — исключаем из nact-фильтра
QUESTION_STRICT_RX = re.compile(
    r"[?？]|"
    r"\b(где|когда|зачем|почему|кто|какой|какая|какое|какие|сколько|"
    r"расскажи|расскажите|объясни|объясните|опиши|опишите|назови|"
    r"перечисли|сравни|реализуй|напиши|покажи)\b",
    re.IGNORECASE,
)

TECH_RX = re.compile(
    r"\b(API|SDK|HTTP|SQL|ORM|JWT|REST|Git|Docker|Redis|Kafka|TCP|UDP|CI/CD|"
    r"алгоритм|паттерн|архитектур|микросервис|база данных|транзакц|индекс|кэш|"
    r"поток|синхрониз|конкурентн|асинхронн|протокол|сложность|O\(|"
    r"Python|Java|Swift|Kotlin|Golang|TypeScript|Rust|C\+\+)\b",
    re.IGNORECASE,
)

SENTENCE_END_RX = re.compile(r"[.!]\s*$")  # завершённое утверждение (не ?)

# Мусорные паттерны
URL_RX = re.compile(r"https?://\S+")
EMOJI_RX = re.compile(r"[\U00010000-\U0010ffff\U0001F300-\U0001F9FF\u2600-\u27BF]")
HTML_RX = re.compile(r"</?[a-zA-Z][^>]*>")
CODE_RX = re.compile(
    r"[{};]|</|/>|::|\bpublic\b|\bprivate\b|\bstatic\b|\bvoid\b|\breturn\b|\bnull\b|^\s*//|^\s*/\*"
)
CODE_LINE_RX = re.compile(
    r"^[\s]*(@|#include|import |from \w+ import|def |class |function |var |const |let |SELECT |INSERT |CREATE |ALTER |DROP )"
)
LIST_RX = re.compile(r"^[\s]*([0-9]+\.\s+|\+\s+|\-\s+[А-ЯA-Z])")


def is_vtt(sample: dict) -> bool:
    src = sample.get("source", "")
    return src != "curated" and "/" not in src


def is_github(sample: dict) -> bool:
    src = sample.get("source", "")
    return "/" in src and src != "curated"


def is_junk(text: str, min_words: int = 2) -> bool:
    if URL_RX.search(text):
        return True
    if EMOJI_RX.search(text):
        return True
    if HTML_RX.search(text):
        return True
    if CODE_RX.search(text):
        return True
    if CODE_LINE_RX.search(text):
        return True
    if LIST_RX.match(text) and not QUESTION_RX.search(text):
        return True
    if len(text.split()) < min_words:
        return True
    return False


def main() -> None:
    samples = [
        json.loads(l)
        for l in IN_JSONL.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    print(f"Исходных записей: {len(samples)}")

    # ── 1. Разбивка ──────────────────────────────────────────────────────────
    act_raw = [s for s in samples if s["head2_label"] == "actionable"]
    old_non_act = [s for s in samples if s["head2_label"] == "non_actionable"]
    print(f"Исходно: actionable={len(act_raw)}, non_actionable={len(old_non_act)}")
    print(
        f"  → УДАЛЯЕМ старые non_actionable (3-словные конфаундеры): {len(old_non_act)}"
    )

    # ── 2. Очистка мусора из actionable ──────────────────────────────────────
    act_clean = [s for s in act_raw if not is_junk(s["text"], min_words=4)]
    print(
        f"actionable после очистки мусора: {len(act_clean)} (−{len(act_raw) - len(act_clean)})"
    )

    # ── 3. GitHub: только явные вопросы ──────────────────────────────────────
    act_filtered = [
        s for s in act_clean if not is_github(s) or QUESTION_RX.search(s["text"])
    ]
    print(
        f"actionable после GitHub-фильтра: {len(act_filtered)} (−{len(act_clean) - len(act_filtered)})"
    )

    # ── 4. VTT: убираем завершённые без вопроса ───────────────────────────────
    def drop_vtt(s: dict) -> bool:
        return (
            is_vtt(s)
            and bool(SENTENCE_END_RX.search(s["text"]))
            and not QUESTION_RX.search(s["text"])
        )

    vtt_dropped = [s for s in act_filtered if is_vtt(s) and drop_vtt(s)]
    act_questions = [s for s in act_filtered if not drop_vtt(s)]
    print(
        f"actionable (вопросы): {len(act_questions)} (убрано завершённых VTT без ?: {len(vtt_dropped)})"
    )

    # ── 5. Генерируем non_actionable из VTT-ответов ──────────────────────────
    # Берём завершённые VTT-предложения без вопроса и без технических терминов,
    # длиной 6-30 слов — реальные ответы/утверждения из интервью
    def is_good_nact(s: dict) -> bool:
        text = s["text"]
        if not is_vtt(s):
            return False
        if is_junk(text, min_words=5):
            return False
        if len(text.split()) > 40:
            return False
        if not SENTENCE_END_RX.search(text):
            return False  # незавершённые — нет
        if QUESTION_STRICT_RX.search(text):
            return False  # с явным вопросом — нет
        # "что" и "как" без ? — оставляем (это утверждения: "то, что...", "как правило...")
        return True

    nact_pool = [s for s in act_raw if is_good_nact(s)]
    print(f"\nnon_actionable (ответы из VTT, 6-30 слов, завершённые): {len(nact_pool)}")

    # Сэмплируем нужное количество
    nact_count = min(TARGET_NACT, len(nact_pool))
    new_non_act_raw = random.sample(nact_pool, nact_count)

    # Меняем метку на non_actionable
    new_non_act = []
    for s in new_non_act_raw:
        entry = dict(s)
        entry["head2_label"] = "non_actionable"
        entry["head3_query_start"] = None
        new_non_act.append(entry)

    print(f"Выбрано non_actionable: {len(new_non_act)}")

    # Проверяем что перемаркированные не попадут обратно в actionable
    nact_texts = {s["text"] for s in new_non_act}
    act_questions = [s for s in act_questions if s["text"] not in nact_texts]
    print(f"actionable после удаления перемаркированных: {len(act_questions)}")

    # ── 6. Длина: проверяем нет ли конфаундера ────────────────────────────────
    import statistics

    act_lens = [len(s["text"].split()) for s in act_questions]
    nact_lens = [len(s["text"].split()) for s in new_non_act]
    print(f"\nПроверка длин (слов):")
    print(
        f"  actionable:     mean={statistics.mean(act_lens):.1f}  median={statistics.median(act_lens):.1f}"
    )
    print(
        f"  non_actionable: mean={statistics.mean(nact_lens):.1f}  median={statistics.median(nact_lens):.1f}"
    )

    # ── 7. Downsample actionable → 3:1 ───────────────────────────────────────
    target_act = TARGET_RATIO * len(new_non_act)
    if len(act_questions) > target_act:
        curated = [s for s in act_questions if s["source"] == "curated"]
        github = [s for s in act_questions if is_github(s)]
        vtt = [s for s in act_questions if is_vtt(s)]
        need = target_act - len(curated)
        gh_take = min(len(github), max(need, 0))
        gh_sampled = (
            random.sample(github, gh_take) if gh_take < len(github) else list(github)
        )
        vtt_need = need - len(gh_sampled)
        vtt_sampled = random.sample(vtt, min(max(vtt_need, 0), len(vtt)))
        act_final = curated + gh_sampled + vtt_sampled
        print(f"\nDownsample actionable: {len(act_questions)} → {len(act_final)}")
        print(
            f"  (curated={len(curated)}, github={len(gh_sampled)}, vtt={len(vtt_sampled)})"
        )
    else:
        act_final = act_questions
        print(f"\nDownsample не нужен: {len(act_final)} ≤ {target_act}")

    # ── 8. Финальная выборка ─────────────────────────────────────────────────
    result = act_final + new_non_act
    random.shuffle(result)

    ratio = len(act_final) / max(1, len(new_non_act))
    print(f"\nИтог: {len(result)} записей")
    print(
        f"  actionable={len(act_final)}, non_actionable={len(new_non_act)}, ratio={ratio:.1f}:1"
    )
    print(f"  has_punct={sum(1 for s in result if s['has_punct'])}")

    # ── 9. Сохранение ────────────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in result:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n✅ Сохранено: {OUT_JSONL}")

    # Примеры
    print("\n── actionable (с ?) ──")
    for s in [
        s for s in result if s["head2_label"] == "actionable" and "?" in s["text"]
    ][:5]:
        print(f"  {s['text'][:90]}")

    print("\n── non_actionable (ответы) ──")
    for s in [s for s in result if s["head2_label"] == "non_actionable"][:8]:
        print(f"  [{len(s['text'].split())}w] {s['text'][:90]}")


if __name__ == "__main__":
    main()
