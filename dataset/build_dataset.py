#!/usr/bin/env python3
"""
build_dataset.py — собирает датасет для 3-head модели из:
  1. VTT субтитров реальных IT-собеседований (YouTube)
  2. Markdown файлов с GitHub (вопросы для интервью)

VTT-парсинг: rolling-caption формат YouTube разворачивается через delta-извлечение,
склейку в сплошной поток и разбиение по границам предложений (.?!).
Незавершённые фрагменты прогоняются через Apple NL + PunctuationModel CoreML.

Выходной формат (dataset.jsonl):
  {
    "text":               str,
    "source":             str,
    "head2_label":        "actionable" | "non_actionable",
    "head3_query_start":  int | null,
    "has_punct":          bool
  }
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np

# ─── Пути ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
SUBS_DIR = ROOT / "subtitles"
GITHUB_DIR = ROOT / "github"
OUT_JSONL = ROOT / "processed" / "dataset.jsonl"
OUT_STATS = ROOT / "processed" / "stats.json"

VOCAB_PATH = Path(
    "/Users/artemgusarov/Downloads/PROJECTS/AIssistant/AIssistant/punct_vocab.json"
)
MODEL_PATH = Path(
    "/Users/artemgusarov/Downloads/PROJECTS/AIssistant/AIssistant/PunctuationModel.mlpackage"
)
SEQ_LEN = 64
PUNCT_SUFFIX = ["", "", "", ",", ",", ",", ".", ".", ".", "?", "?", "?"]
CAP_MODE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

# ─── Модели данных ────────────────────────────────────────────────────────────


@dataclass
class Sample:
    text: str
    source: str
    head2_label: str  # "actionable" | "non_actionable"
    head3_query_start: Optional[int]
    has_punct: bool


# ─── Константы разметки ───────────────────────────────────────────────────────

FILLER_EXACT = {
    "угу",
    "ага",
    "да",
    "нет",
    "ок",
    "окей",
    "хмм",
    "хм",
    "понятно",
    "хорошо",
    "ладно",
    "ясно",
    "понял",
    "поняла",
    "отлично",
    "супер",
    "интересно",
    "молодец",
    "класс",
    "да-да",
    "ну",
    "эм",
    "э",
}

FILLER_PREFIX_RX = re.compile(
    r"^((?:(?:угу|ага|хорошо|окей|ок|ладно|понятно|ясно|да|нет|"
    r"отлично|супер|молодец|класс|интересно|хмм?|эм?)"
    r"[\s,\.!]*){1,3})",
    re.IGNORECASE,
)

QUESTION_RX = re.compile(
    r"[?？]|"
    r"\b(что|как|где|когда|зачем|почему|кто|какой|какая|какое|какие|"
    r"сколько|чем|каким образом|расскажи|расскажите|объясни|объясните|"
    r"опиши|опишите|назови|перечисли|сравни|реализуй|напиши|покажи)\b",
    re.IGNORECASE,
)

TECH_RX = re.compile(
    r"\b(алгоритм|паттерн|архитектур|база|sql|nosql|api|rest|http|tcp|udp|"
    r"docker|kubernetes|git|sort|hash|дерев|граф|стек|очередь|"
    r"swift|kotlin|python|java|golang|rust|javascript|typescript|"
    r"deadlock|race|async|await|thread|memory|heap|cache|redis|kafka|"
    r"микросервис|монолит|solid|oop|паттерн|реализуй|напиши|покажи)\b",
    re.IGNORECASE,
)

BEHAVIORAL_RX = re.compile(
    r"\b(расскажи|расскажите|опиши|опишите|ваш опыт|ваши|почему вы|"
    r"зачем вы|как вы|ситуаци|конфликт|команд|коллег|достижени|"
    r"слабые|сильные|мотива|карьер|планируете|ожидани)\b",
    re.IGNORECASE,
)

SENTENCE_END_RX = re.compile(r"[.?!]\s*$")

# ─── PunctuationModel ─────────────────────────────────────────────────────────

print("Загружаю PunctuationModel...")
_vocab_list = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(_vocab_list)}
PUNCT_MODEL = ct.models.MLModel(str(MODEL_PATH))
print(f"  vocab={len(TOKEN_TO_ID)}, model loaded ✅")


def _word_piece(word: str) -> list[str]:
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


def _encode(words: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ids = np.zeros(SEQ_LEN, dtype=np.int32)
    mask = np.zeros(SEQ_LEN, dtype=np.int32)
    ids[0] = 2
    mask[0] = 1
    pos = 1
    word_starts: list[int] = []
    for word in words:
        if pos >= SEQ_LEN - 1:
            break
        pieces = _word_piece(word)
        word_starts.append(pos)
        for i, piece in enumerate(pieces):
            if pos >= SEQ_LEN - 1:
                break
            tok = piece if i == 0 else "##" + piece
            ids[pos] = TOKEN_TO_ID.get(tok, 1)
            mask[pos] = 1
            pos += 1
    ids[pos] = 3
    mask[pos] = 1
    return ids, mask, word_starts


def punct_restore(text: str) -> str:
    """Восстанавливает пунктуацию и капитализацию через PunctuationModel."""
    words = text.lower().split()
    if not words:
        return text
    ids, mask, word_starts = _encode(words)
    unk = sum(1 for ws in word_starts if ids[ws] == 1)
    if unk / len(words) > 0.3:
        return text
    result = PUNCT_MODEL.predict(
        {
            "input_ids": ids.reshape(1, SEQ_LEN),
            "attention_mask": mask.reshape(1, SEQ_LEN),
        }
    )
    logits = np.array(result["logits"]).squeeze(0)
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


def nl_segment_sentences(text: str) -> list[str]:
    """Сегментирует текст на предложения через Apple NaturalLanguage."""
    try:
        import NaturalLanguage as NL

        tokenizer = NL.NLTokenizer.alloc().initWithUnit_(NL.NLTokenUnitSentence)
        tokenizer.setString_(text)
        sentences: list[str] = []
        tokenizer.enumerateTokensInRange_usingBlock_(
            (0, len(text)),
            lambda r, _, stop: sentences.append(
                text[r.location : r.location + r.length]
            )
            or False,
        )
        return [s.strip() for s in sentences if s.strip()]
    except ImportError:
        parts = re.split(r"(?<=[.!?])\s+|\n", text)
        return [p.strip() for p in parts if p.strip()]


_CHUNK_SIZE = 40  # слов в окне для punct_restore


def restore_stream(text: str) -> list[str]:
    """
    Восстанавливает пунктуацию в произвольно длинном тексте:
    1. Снимаем пунктуацию
    2. Нарезаем по CHUNK_SIZE слов
    3. Каждый чанк → NL || punct_restore параллельно
    4. Склеиваем → режем по .?!
    """
    words = re.sub(r"[,\.?!]", "", text).split()
    if not words:
        return []

    chunks = [
        " ".join(words[i : i + _CHUNK_SIZE]) for i in range(0, len(words), _CHUNK_SIZE)
    ]

    def restore_chunk(chunk: str) -> str:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            nl_fut = pool.submit(nl_segment_sentences, chunk)
            pr_fut = pool.submit(punct_restore, chunk)
            nl_segs = nl_fut.result()
            pr_text = pr_fut.result()
        if len(nl_segs) > 1:
            return " ".join(punct_restore(seg) for seg in nl_segs)
        return pr_text

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        restored_chunks = list(pool.map(restore_chunk, chunks))

    full = " ".join(restored_chunks)
    sentences = re.split(r"(?<=[.?!])\s+", full)
    return [s.strip() for s in sentences if len(s.split()) >= 3]


# ─── VTT парсер ───────────────────────────────────────────────────────────────


def _extract_delta_stream(path: Path) -> str:
    """
    VTT → непрерывный текстовый поток через delta-извлечение.

    YouTube rolling-caption: каждый блок = предыдущий текст + новые слова.
    Три сценария:
      A. clean начинается с prev_clean → дельта = новые слова
      B. prev_clean заканчивается на clean → transition-блок (line 2 окна), пропуск
      C. иначе → новый контекст (смена говорящего / сцены)
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n{2,}", raw)
    stream_parts: list[str] = []
    prev_clean = ""

    for block in blocks:
        lines = block.strip().splitlines()
        if not any("-->" in l for l in lines):
            continue
        text_lines = [
            l
            for l in lines
            if "-->" not in l and not re.match(r"^\d+$", l) and l.strip()
        ]
        if not text_lines:
            continue  # пустой блок — не сброс контекста

        raw_text = " ".join(text_lines)
        clean = re.sub(r"<[^>]+>", "", raw_text)
        clean = re.sub(r"\[.*?\]", "", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if not clean:
            prev_clean = ""  # [музыка] → сброс контекста
            continue

        if prev_clean and clean.startswith(prev_clean):
            # A. Расширение — берём дельту
            delta = clean[len(prev_clean) :].strip()
        elif prev_clean and prev_clean.endswith(clean):
            # B. Transition-блок (line 2 предыдущего окна) — пропуск
            prev_clean = clean
            continue
        else:
            # C. Новый контекст
            delta = clean

        if delta:
            stream_parts.append(delta)
        prev_clean = clean

    return re.sub(r"\s+", " ", " ".join(stream_parts)).strip()


def parse_vtt(path: Path) -> list[str]:
    """
    VTT → список полных предложений.

    1. Delta-извлечение → непрерывный поток (без дублей rolling-window)
    2. Снимаем пунктуацию исходника, прогоняем через restore_stream (чанки 40 слов)
    """
    full_text = _extract_delta_stream(path)
    if not full_text:
        return []
    return restore_stream(full_text)


def has_punctuation(text: str) -> bool:
    return bool(re.search(r"[,\.?!]", text))


# ─── Логика разметки ─────────────────────────────────────────────────────────


def label_sample(text: str, source: str) -> Optional[Sample]:
    words = text.strip().split()
    if not words:
        return None

    lower = text.lower().strip()
    word_count = len(words)

    if lower.rstrip(".,!") in FILLER_EXACT:
        return Sample(
            text=text,
            source=source,
            head2_label="non_actionable",
            head3_query_start=None,
            has_punct=has_punctuation(text),
        )

    if word_count <= 3 and not QUESTION_RX.search(text) and not TECH_RX.search(text):
        return Sample(
            text=text,
            source=source,
            head2_label="non_actionable",
            head3_query_start=None,
            has_punct=has_punctuation(text),
        )

    m = FILLER_PREFIX_RX.match(lower)
    filler_prefix_words = 0
    if m:
        prefix = m.group(1).strip()
        filler_prefix_words = len(prefix.split())
        if filler_prefix_words >= word_count:
            return Sample(
                text=text,
                source=source,
                head2_label="non_actionable",
                head3_query_start=None,
                has_punct=has_punctuation(text),
            )

    has_q = bool(QUESTION_RX.search(text))
    has_t = bool(TECH_RX.search(text))
    has_b = bool(BEHAVIORAL_RX.search(text))

    if not (has_q or has_t or has_b):
        return None

    query_start = filler_prefix_words if filler_prefix_words > 0 else 0

    return Sample(
        text=text,
        source=source,
        head2_label="actionable",
        head3_query_start=query_start,
        has_punct=has_punctuation(text),
    )


# ─── GitHub markdown парсер ───────────────────────────────────────────────────


def extract_questions_from_markdown(text: str) -> list[str]:
    questions: list[str] = []
    for line in text.splitlines():
        clean = re.sub(r"[#*_`>\[\]()]", "", line).strip()
        clean = re.sub(r"\s+", " ", clean)
        if len(clean) < 8 or len(clean) > 300:
            continue
        if QUESTION_RX.search(clean) or (
            TECH_RX.search(clean) and len(clean.split()) >= 3
        ):
            questions.append(clean)
    return questions


# ─── Дедупликация ─────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("?.,!"))


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    samples: list[Sample] = []
    seen_normalized: set[str] = set()
    stats = {
        "vtt_files": 0,
        "vtt_sentences": 0,
        "github_files": 0,
        "github_questions": 0,
        "total_before_dedup": 0,
        "total_after_dedup": 0,
        "actionable": 0,
        "non_actionable": 0,
        "skipped_ambiguous": 0,
        "has_punct": 0,
    }

    def add(s: Sample) -> None:
        norm = normalize(s.text)
        if norm in seen_normalized or len(norm) < 4:
            return
        seen_normalized.add(norm)
        samples.append(s)
        stats["total_after_dedup"] += 1
        stats[s.head2_label] += 1
        if s.has_punct:
            stats["has_punct"] += 1

    # ── 1. VTT субтитры ──────────────────────────────────────────────────────
    print("📺 Обработка VTT субтитров...")
    vtt_files = sorted(SUBS_DIR.rglob("*.vtt"))
    for i, vtt_file in enumerate(vtt_files, 1):
        stats["vtt_files"] += 1
        source = vtt_file.parent.name

        print(f"  [{i}/{len(vtt_files)}] {source}...", end=" ", flush=True)
        sentences = parse_vtt(vtt_file)
        stats["vtt_sentences"] += len(sentences)
        print(f"{len(sentences)} предложений")

        for sent in sentences:
            stats["total_before_dedup"] += 1
            s = label_sample(sent, source)
            if s is None:
                stats["skipped_ambiguous"] += 1
                continue
            add(s)

    print(
        f"  VTT итого: {stats['vtt_files']} файлов, {stats['vtt_sentences']} предложений"
    )

    # ── 2. GitHub markdown ───────────────────────────────────────────────────
    print("📦 Обработка GitHub репо...")
    source = ""
    for md_file in sorted(GITHUB_DIR.glob("*.md")):
        stats["github_files"] += 1
        source = (
            md_file.stem.split("_")[0] + "/" + md_file.stem.split("_")[1]
            if "_" in md_file.stem
            else md_file.stem
        )
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        questions = extract_questions_from_markdown(content)
        stats["github_questions"] += len(questions)
        for q in questions:
            stats["total_before_dedup"] += 1
            s = label_sample(q, source)
            if s is None:
                stats["skipped_ambiguous"] += 1
                continue
            add(s)

    for txt_file in sorted(GITHUB_DIR.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        questions = extract_questions_from_markdown(content)
        for q in questions:
            stats["total_before_dedup"] += 1
            s = label_sample(q, source=txt_file.stem)
            if s is None:
                stats["skipped_ambiguous"] += 1
                continue
            add(s)

    print(
        f"  GitHub: {stats['github_files']} файлов, {stats['github_questions']} вопросов"
    )

    # ── 3. Сохранение ────────────────────────────────────────────────────────
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"""
╔══════════════════════════════════════════════╗
║              ДАТАСЕТ ГОТОВ                   ║
╠══════════════════════════════════════════════╣
║  Всего примеров (после дедупл.)  {stats["total_after_dedup"]:>7}       ║
║  ─────────────────────────────────────────── ║
║  actionable                      {stats["actionable"]:>7}       ║
║  non_actionable                  {stats["non_actionable"]:>7}       ║
║  пропущено (неоднозначно)        {stats["skipped_ambiguous"]:>7}       ║
║  ─────────────────────────────────────────── ║
║  с пунктуацией (Head 1 data)     {stats["has_punct"]:>7}       ║
╚══════════════════════════════════════════════╝
    """)
    print(f"📁 {OUT_JSONL}")

    print("\n── Примеры actionable ──")
    for s in [x for x in samples if x.head2_label == "actionable"][:8]:
        qs = f"query_start={s.head3_query_start}" if s.head3_query_start else "full"
        print(f"  [{qs}] {s.text[:80]}")

    print("\n── Примеры non_actionable ──")
    for s in [x for x in samples if x.head2_label == "non_actionable"][:8]:
        print(f"  {s.text[:80]}")


if __name__ == "__main__":
    main()
