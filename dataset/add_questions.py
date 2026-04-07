#!/usr/bin/env python3
"""
add_questions.py — обогащает датасет новыми вопросами:

1. Читает raw_new_questions.txt (без пунктуации)
2. Параллельно прогоняет через:
   a. Apple NaturalLanguage (NLTokenizer) — сегментация предложений
   b. PunctuationModel.mlpackage — восстановление пунктуации + капитализации
3. Фильтрует: оставляет только вопросы (заканчиваются на ? или содержат question words)
4. Дополняет head2/head3 метками
5. Аппендит в dataset.jsonl

Запуск: python3 add_questions.py
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
RAW_FILE = ROOT / "raw_new_questions.txt"
VOCAB_PATH = Path(
    "/Users/artemgusarov/Downloads/PROJECTS/AIssistant/AIssistant/punct_vocab.json"
)
MODEL_PATH = Path(
    "/Users/artemgusarov/Downloads/PROJECTS/AIssistant/AIssistant/PunctuationModel.mlpackage"
)
OUT_JSONL = ROOT / "processed" / "dataset.jsonl"

SEQ_LEN = 64
PUNCT_SUFFIX = ["", "", "", ",", ",", ",", ".", ".", ".", "?", "?", "?"]
CAP_MODE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

# ── Метки разметки ────────────────────────────────────────────────────────────
FILLER_PREFIX_RX = re.compile(
    r"^((?:(?:угу|ага|хорошо|окей|ок|ладно|понятно|ясно|да|нет|"
    r"отлично|супер|молодец|класс|интересно|хмм?|эм?)"
    r"[\s,\.!]*){1,3})",
    re.IGNORECASE,
)

QUESTION_RX = re.compile(
    r"[?？]|"
    r"\b(что|как|где|когда|зачем|почему|кто|какой|какая|какое|какие|"
    r"сколько|чем|каким образом|расскажите|расскажи|опишите|опиши|"
    r"назовите|перечислите|сравните|приведите|поведайте|вспомните)\b",
    re.IGNORECASE,
)


@dataclass
class Sample:
    text: str
    source: str
    head2_label: str
    head3_query_start: Optional[int]
    has_punct: bool


# ── Загрузка словаря и модели ─────────────────────────────────────────────────
print("Загружаю словарь и модель...")
vocab_list = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
TOKEN_TO_ID = {tok: i for i, tok in enumerate(vocab_list)}
MODEL = ct.models.MLModel(str(MODEL_PATH))
print(f"  vocab={len(TOKEN_TO_ID)}, model loaded ✅")


# ── WordPiece (зеркало PunctuationSession.swift) ──────────────────────────────
def word_piece(word: str) -> list[str]:
    remaining = word
    pieces: list[str] = []
    is_first = True
    oov_buf = ""
    while remaining:
        found = False
        end = len(remaining)
        while end > 0:
            cand = remaining[:end] if is_first else ("##" + remaining[:end])
            if cand in TOKEN_TO_ID:
                if oov_buf:
                    pieces.append(oov_buf)
                    oov_buf = ""
                pieces.append(remaining[:end])
                remaining = remaining[end:]
                is_first = False
                found = True
                break
            end -= 1
        if not found:
            oov_buf += remaining[0]
            remaining = remaining[1:]
    if oov_buf:
        pieces.append(oov_buf)
    return pieces or [word]


def encode(words: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ids = np.zeros(SEQ_LEN, dtype=np.int32)
    mask = np.zeros(SEQ_LEN, dtype=np.int32)
    ids[0] = 2
    mask[0] = 1  # [CLS]
    pos = 1
    word_starts: list[int] = []
    for word in words:
        if pos >= SEQ_LEN - 1:
            break
        pieces = word_piece(word)
        word_starts.append(pos)
        for i, piece in enumerate(pieces):
            if pos >= SEQ_LEN - 1:
                break
            tok = piece if i == 0 else "##" + piece
            ids[pos] = TOKEN_TO_ID.get(tok, 1)
            mask[pos] = 1
            pos += 1
    ids[pos] = 3
    mask[pos] = 1  # [SEP]
    return ids, mask, word_starts


def punct_restore(text: str) -> str:
    """Восстанавливает пунктуацию и капитализацию через PunctuationModel."""
    words = text.lower().split()
    if not words:
        return text

    ids, mask, word_starts = encode(words)

    # UNK ratio check
    unk = sum(1 for ws in word_starts if ids[ws] == 1)
    if words and unk / len(words) > 0.3:
        return text

    result = MODEL.predict(
        {
            "input_ids": ids.reshape(1, SEQ_LEN),
            "attention_mask": mask.reshape(1, SEQ_LEN),
        }
    )
    logits = np.array(result["logits"]).squeeze(0)  # (64, 12)

    out = []
    for wi, word in enumerate(words):
        if wi >= len(word_starts) or word_starts[wi] >= SEQ_LEN:
            out.append(word)
            continue
        label = int(np.argmax(logits[word_starts[wi]]))
        w = word
        if CAP_MODE[label] == 1:
            w = w[0].upper() + w[1:]
        elif CAP_MODE[label] == 2:
            w = w.upper()
        out.append(w + PUNCT_SUFFIX[label])
    return " ".join(out)


# ── Apple NL токенизация (сегментация предложений) ────────────────────────────
def nl_segment_sentences(text: str) -> list[str]:
    """Сегментирует текст на предложения через Apple NaturalLanguage."""
    try:
        import NaturalLanguage as NL  # PyObjC

        tokenizer = NL.NLTokenizer.alloc().initWithUnit_(NL.NLTokenUnitSentence)
        tokenizer.setString_(text)
        rng = (0, len(text))
        sentences = []
        tokenizer.enumerateTokensInRange_usingBlock_(
            rng,
            lambda r, _, stop: sentences.append(
                text[r.location : r.location + r.length]
            )
            or False,
        )
        return [s.strip() for s in sentences if s.strip()]
    except ImportError:
        # Fallback: простое разбиение по знакам препинания или переносам строки
        parts = re.split(r"(?<=[.!?])\s+|\n", text)
        return [p.strip() for p in parts if p.strip()]


def _make_sample(restored: str) -> Optional[Sample]:
    """Превращает восстановленный текст в Sample или None если не вопрос."""
    restored = restored.strip()
    if not restored:
        return None
    is_question = restored.endswith("?") or bool(QUESTION_RX.search(restored))
    if not is_question:
        return None
    words = restored.split()
    m = FILLER_PREFIX_RX.match(restored.lower())
    filler_words = len(m.group(1).split()) if m else 0
    if filler_words >= len(words):
        return None
    return Sample(
        text=restored,
        source="curated",
        head2_label="actionable",
        head3_query_start=filler_words if filler_words > 0 else 0,
        has_punct=True,
    )


def process_line(line: str) -> list[Sample]:
    """Обрабатывает одну строку: NL сегментация || punct restore (параллельно) → фильтр."""
    line = line.strip()
    if not line or line.startswith("#"):
        return []

    already_has_punct = bool(re.search(r"[,\.?!]", line))

    if already_has_punct:
        # Пунктуация уже есть — NL только для сегментации длинных реплик
        segments = nl_segment_sentences(line)
        return [s for seg in segments for s in [_make_sample(seg)] if s]

    # Запускаем NL-сегментацию и punct_restore независимо и параллельно
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        nl_fut = pool.submit(nl_segment_sentences, line)
        pr_fut = pool.submit(punct_restore, line)
        nl_segs = nl_fut.result()
        pr_text = pr_fut.result()

    # Если NL разбил на несколько предложений — punct_restore каждого сегмента
    if len(nl_segs) > 1:
        candidates = [punct_restore(seg) for seg in nl_segs]
    else:
        # Единственный сегмент — берём уже готовый pr_text (не вызываем повторно)
        candidates = [pr_text]

    return [s for cand in candidates for s in [_make_sample(cand)] if s]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    raw_lines = [
        l.strip()
        for l in RAW_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    print(f"Строк для обработки: {len(raw_lines)}")

    # Параллельная обработка (каждая строка → NL || punct независимо)
    results: list[Sample] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_line, line): line for line in raw_lines}
        for fut in concurrent.futures.as_completed(futures):
            results.extend(fut.result())

    print(f"Вопросов после фильтра: {len(results)}")

    # Дедупликация с существующим датасетом
    existing_texts: set[str] = set()
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                existing_texts.add(
                    re.sub(r"\s+", " ", d["text"].lower().rstrip("?.,!"))
                )
            except Exception:
                pass

    new_samples = [
        s
        for s in results
        if re.sub(r"\s+", " ", s.text.lower().rstrip("?.,!")) not in existing_texts
    ]
    print(f"Новых (не дублей): {len(new_samples)}")

    # Примеры
    for s in new_samples[:10]:
        print(f"  [{s.head3_query_start}] {s.text}")

    # Аппенд
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for s in new_samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    total = sum(1 for _ in OUT_JSONL.read_text().splitlines() if _.strip())
    print(f"\n✅ dataset.jsonl: {total} строк (+{len(new_samples)} новых)")


if __name__ == "__main__":
    main()
