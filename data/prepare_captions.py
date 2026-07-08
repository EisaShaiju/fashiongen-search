"""
prepare_captions.py
===================
Turn structured product metadata into training captions (JSONL of {id, caption}).

Built for the H&M `articles.csv` schema (product name, colour, department,
garment group, detail description) but the field mapping is easy to adapt to
DeepFashion attribute annotations.

    python -m data.prepare_captions \
        --articles data/raw/hm/articles.csv --out data/processed/hm_captions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def caption_from_row(row) -> str:
    parts = []
    color = str(row.get("colour_group_name", "")).strip()
    name = str(row.get("prod_name", "")).strip()
    group = str(row.get("garment_group_name", "")).strip()
    section = str(row.get("index_group_name", "")).strip()
    detail = str(row.get("detail_desc", "")).strip()

    lead = " ".join(x for x in [color.lower(), name.lower()] if x and x != "nan")
    parts.append(f"a product photo of a {lead}" if lead else "a product photo of a garment")
    extra = ", ".join(x.lower() for x in [group, section] if x and x != "nan")
    if extra:
        parts.append(extra)
    if detail and detail != "nan":
        parts.append(detail[:160])
    parts.append("studio shot, white background")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="article_id")
    args = ap.parse_args()

    df = pd.read_csv(args.articles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for _, row in df.iterrows():
            # H&M image filenames are zero-padded 10-digit article ids
            aid = str(row[args.id_col]).zfill(10)
            f.write(json.dumps({"id": aid, "caption": caption_from_row(row)}) + "\n")
            n += 1
    print(f"wrote {n} captions -> {args.out}")


if __name__ == "__main__":
    main()
