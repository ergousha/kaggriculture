#!/usr/bin/env python3
"""Build the Jev claim corpus from this repo's prose. NO API CALLS, NO NETWORK.

    uv run python scripts/build_claims_corpus.py
    uv run python scripts/build_claims_corpus.py --stats-only
    uv run python scripts/build_claims_corpus.py --sample 12

Why a derived corpus instead of the raw markdown. Three documented properties of
jev-1.13 decide the design (TypeSafe docs read 2026-09-21):

  1. "Jev is not a calculator... implement any mathematical logic in code", and it
     "does not count reliably" (model-jaggedness/jev-1.13). So MEASUREMENT-grade
     numerals are replaced by typed placeholders before anything is sent, and the real
     values live in a local sidecar the model never sees. The model is asked what a
     claim MEANS; code decides whether its number still holds. A model that cannot see
     `$2,583` cannot be wrong about `$2,583`.

     Only measurement values are masked. Versions (`1.32.7`), dates, and small
     identifiers (`tier 6`, `day 40`, `v0.4.3`) are SEMANTIC content -- masking them
     destroys the meaning the model is there to judge, and they carry no arithmetic.

  2. "Large state full of irrelevant detail" degrades accuracy; "filter first". Code
     blocks, ASCII trees, collapsed <details> run history, badges and table rows are
     dropped, and only sentences that assert something checkable are kept.

  3. A `Choice` question accepts at most 255 options (cookbooks/semantic_find), and the
     budget is 32k tokens for `state` plus the longest question (models). The corpus is
     emitted in chunks respecting BOTH caps, each chunk a self-contained state.

Markdown here is hard-wrapped, so lines are reflowed into paragraphs BEFORE sentence
splitting -- splitting per line yields fragments ("It scores 60/60 against the reference
ladder where the") that no auditor can evaluate.

Output (both under logs/, which is gitignored):
  logs/claims_corpus.json  — what gets sent: chunks of `Cxxxx| <text>` lines
  logs/claims_sidecar.json — never sent: file:line anchors, raw numerals, section paths

The line-id format follows cookbooks/semantic_find: prefix every claim with a stable id
so a Choice question can point at one, using bare ids as options with `None`
descriptions because the state already carries the text.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SOURCES = ["README.md", "docs/experiments.md"]
OUT_CORPUS = HERE / "logs" / "claims_corpus.json"
OUT_SIDECAR = HERE / "logs" / "claims_sidecar.json"

CHOICE_MAX = 255  # model's hard cap on Choice options
CHARS_PER_CHUNK = 60_000  # keeps a chunk's state inside the 32k-token state budget

# One single-pass alternation. Order matters and is load-bearing:
# the KEEP_* groups match first and are emitted unchanged, so a version or date is
# consumed before any masking pattern can chew on its digits. A second pass over
# already-substituted text is what produced `<VER:<NUM:2>>`, so there is exactly one.
_TOKEN_RE = re.compile(
    r"""
      (?P<KEEP_VER>\b\d+\.\d+(?:\.\d+)+\b)
    | (?P<KEEP_DATE>\b\d{4}-\d{2}-\d{2}\b)
    | (?P<KEEP_SEMVERISH>\bv\d+(?:[._]\d+)*\b)
    | (?P<MONEY>[+\-−~]?\$\s?\d[\d,]*(?:\.\d+)?\s?[kKmM]?\b)
    | (?P<PCT>[+\-−~]?\d[\d,]*(?:\.\d+)?\s?%)
    | (?P<MULT>\b\d+(?:\.\d+)?x\b)
    | (?P<RATIO>\b\d[\d,]*\s?/\s?\d[\d,]*\b)
    | (?P<CORR>(?<![v\d.])[+\-−]?0\.\d+\b)
    | (?P<BIGNUM>\b(?:\d{1,3}(?:,\d{3})+|\d{4,})\b)
    """,
    re.VERBOSE,
)

CONCLUSION_WORDS = re.compile(
    r"\b(because|therefore|so that|which means|implies|proves|shows that|"
    r"the reason|hence|thus|explains|contradicts|falsifies|confirms|"
    r"dominates?|outperforms?|beats?|never|always|cannot|must)\b",
    re.IGNORECASE,
)
STALE_MARKERS = re.compile(
    r"\b(superseded|known-wrong|is wrong|was wrong|contradicted|falsified|"
    r"obsolete|no longer true|outdated|prescription was wrong)\b",
    re.IGNORECASE,
)


def clean_markdown(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # links -> label
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)  # bold
    s = re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", s)
    # Leading list/quote markers. Every alternative REQUIRES trailing whitespace:
    # without it `\d+\.` eats the `97.` from a line starting `**97.0%**` (leaving
    # `0%`), and `[-*+]` eats the minus from a leading `-0.05`, silently flipping the
    # sign on a correlation. Both were real, and both are silent.
    s = re.sub(r"^\s*(?:[>#]+\s*|[-*+]\s+|\d+\.\s+)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def iter_blocks(text: str):
    """Yield (line_map, section_path, is_stale_section, paragraph_text).

    Reflows hard-wrapped markdown into paragraphs and carries the enclosing heading
    trail, so a `#### Superseded: ...` heading marks every claim beneath it.

    `line_map` is [(char_offset_in_block, source_line_number), ...]. A reflowed block
    routinely spans a whole bullet list, so attributing every sentence in it to the
    block's first line puts citations 4-5 lines off target -- measured on this repo,
    it sent `README.md:216` to `README.md:212`. Citations are the entire output of
    this tool, so the offset map is load-bearing.
    """
    lines = text.splitlines()
    stack: list[tuple[int, str]] = []
    buf: list[tuple[int, str]] = []
    in_fence = in_details = False

    def flush():
        nonlocal buf
        if not buf:
            return None
        block_parts, line_map, off = [], [], 0
        for lineno, cleaned in buf:
            line_map.append((off, lineno))
            block_parts.append(cleaned)
            off += len(cleaned) + 1  # the joining space
        out = (
            line_map,
            " > ".join(h for _, h in stack),
            any(STALE_MARKERS.search(h) for _, h in stack),
            " ".join(block_parts),
        )
        buf = []
        return out

    for i, raw in enumerate(lines, start=1):
        s_line = raw.rstrip()
        low = s_line.lower()
        if s_line.lstrip().startswith("```"):
            out = flush()
            if out:
                yield out
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if "<details" in low:
            in_details = True
            continue
        if "</details" in low:
            in_details = False
            continue
        if in_details:  # collapsed superseded-run history, already retired by hand
            continue

        hm = re.match(r"^(#{1,6})\s+(.*)$", s_line)
        if hm:
            out = flush()
            if out:
                yield out
            depth = len(hm.group(1))
            stack = [(d, h) for d, h in stack if d < depth]
            stack.append((depth, clean_markdown(hm.group(2))))
            continue

        if (
            not s_line.strip()
            or s_line.lstrip().startswith(("|", "![", "<img", "<a ", "---", "==="))
            or re.match(r"^\s*[\w./-]+\s{2,}#", s_line)
        ):
            out = flush()
            if out:
                yield out
            continue

        buf.append((i, clean_markdown(s_line)))

    out = flush()
    if out:
        yield out


def line_at(line_map: list[tuple[int, int]], offset: int) -> int:
    """Source line number for a character offset inside a reflowed block."""
    idx = bisect.bisect_right([o for o, _ in line_map], offset) - 1
    return line_map[max(0, idx)][1]


def split_sentences(block: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z`\[(])", block)
    return [p.strip() for p in parts if p.strip()]


def mask(sentence: str) -> tuple[str, list[dict]]:
    """Single-pass replacement of measurement numerals with <TYPE:idx>."""
    values: list[dict] = []

    def _repl(m: re.Match[str]) -> str:
        kind = m.lastgroup or ""
        if kind.startswith("KEEP_"):
            return m.group(0)
        idx = len(values)
        values.append({"kind": kind, "raw": m.group(0).strip()})
        # Preserve a trailing space the money/pct pattern may have absorbed.
        tail = " " if m.group(0).endswith(" ") else ""
        return f"<{kind}:{idx}>{tail}"

    return _TOKEN_RE.sub(_repl, sentence), values


def build() -> tuple[dict, dict]:
    claims: list[dict] = []
    seen: set[str] = set()
    skipped_table_rows = 0
    dropped_fragment = 0

    for src in SOURCES:
        text = (HERE / src).read_text(encoding="utf-8")
        skipped_table_rows += sum(1 for ln in text.splitlines() if ln.lstrip().startswith("|"))
        for line_map, section, stale_sec, block in iter_blocks(text):
            cursor = 0
            for sent in split_sentences(block):
                found = block.find(sent, cursor)
                if found >= 0:
                    cursor = found + len(sent)
                start = line_at(line_map, found if found >= 0 else 0)
                if len(sent) < 40:
                    continue
                # A sentence that does not end in terminal punctuation is a reflow
                # fragment (trailing paragraph cut by a code fence or table).
                if not sent.endswith((".", "!", "?", '."', ".)", ".*", ".`")):
                    dropped_fragment += 1
                    continue
                has_num = bool(re.search(r"\d", sent))
                if not has_num and not CONCLUSION_WORDS.search(sent):
                    continue
                masked, values = mask(sent)
                key = hashlib.sha1(re.sub(r"\W+", "", masked.lower()).encode()).hexdigest()[:16]
                if key in seen:  # same figure restated across the two documents
                    continue
                seen.add(key)
                claims.append(
                    {
                        "text": masked,
                        "values": values,
                        "file": src,
                        "line": start,
                        "section": section,
                        "self_marked_stale": stale_sec or bool(STALE_MARKERS.search(sent)),
                        "kind": "numeric" if values else "conclusion",
                    }
                )

    for n, c in enumerate(claims):
        c["id"] = f"C{n:04d}"

    chunks: list[dict] = []
    cur: list[dict] = []
    cur_chars = 0
    for c in claims:
        line = f"{c['id']}| {c['text']}"
        if cur and (len(cur) >= CHOICE_MAX or cur_chars + len(line) > CHARS_PER_CHUNK):
            chunks.append({"claim_ids": [x["id"] for x in cur]})
            cur, cur_chars = [], 0
        cur.append(c)
        cur_chars += len(line) + 1
    if cur:
        chunks.append({"claim_ids": [x["id"] for x in cur]})

    by_id = {c["id"]: c for c in claims}
    for ch in chunks:
        ch["state"] = "\n".join(f"{i}| {by_id[i]['text']}" for i in ch["claim_ids"])

    corpus = {
        "note": "measurement numerals are masked as <TYPE:idx>; raw values live in the "
        "sidecar and are never sent. Versions, dates and small identifiers are "
        "intentionally left intact as semantic content.",
        "choice_option_cap": CHOICE_MAX,
        "chunks": chunks,
    }
    sidecar = {
        "claims": {
            c["id"]: {
                k: c[k]
                for k in ("file", "line", "section", "values", "kind", "self_marked_stale", "text")
            }
            for c in claims
        },
        "skipped_table_rows": skipped_table_rows,
        "dropped_reflow_fragments": dropped_fragment,
    }
    return corpus, sidecar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    corpus, sidecar = build()
    claims = sidecar["claims"]
    raw_words = sum(len((HERE / s).read_text(encoding="utf-8").split()) for s in SOURCES)
    cor_words = sum(len(ch["state"].split()) for ch in corpus["chunks"])
    raw_tok, cor_tok = int(raw_words * 1.35), int(cor_words * 1.35)
    numeric = sum(1 for c in claims.values() if c["kind"] == "numeric")

    print(f"sources           : {', '.join(SOURCES)}")
    print(
        f"claims extracted  : {len(claims)}  ({numeric} numeric, "
        f"{len(claims) - numeric} conclusion)"
    )
    print(
        f"self-marked stale : {sum(1 for c in claims.values() if c['self_marked_stale'])}"
        "   (labelled positives for Phase 5)"
    )
    print(f"table rows skipped: {sidecar['skipped_table_rows']}")
    print(f"reflow fragments  : {sidecar['dropped_reflow_fragments']} dropped")
    print(
        f"chunks            : {len(corpus['chunks'])} (max "
        f"{max(len(c['claim_ids']) for c in corpus['chunks'])} claims, cap {CHOICE_MAX})"
    )
    print(f"numerals masked   : {sum(len(c['values']) for c in claims.values())}")
    print()
    print(
        f"raw prose         : {raw_words:>7} words  ~{raw_tok:>7} tok  "
        f"${raw_tok / 1e9 * 42:.6f}/pass"
    )
    print(
        f"corpus            : {cor_words:>7} words  ~{cor_tok:>7} tok  "
        f"${cor_tok / 1e9 * 42:.6f}/pass"
    )
    print(
        f"reduction         : {(1 - cor_tok / raw_tok) * 100:.1f}% fewer input tokens "
        f"({raw_tok / cor_tok:.2f}x)"
    )

    if args.sample:
        print(f"\n=== random sample of {args.sample} ===")
        random.seed(7)
        for cid in random.sample(list(claims), args.sample):
            c = claims[cid]
            print(
                f"{cid} [{c['kind']:10}] {c['file']}:{c['line']}"
                f"{'  STALE' if c['self_marked_stale'] else ''}"
            )
            print(f"      {c['text'][:160]}")

    if not args.stats_only:
        OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
        OUT_CORPUS.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        OUT_SIDECAR.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT_CORPUS.relative_to(HERE)} and {OUT_SIDECAR.relative_to(HERE)}")


if __name__ == "__main__":
    main()
