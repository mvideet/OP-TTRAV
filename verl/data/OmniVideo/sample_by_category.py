#!/usr/bin/env python3
"""
Sample from OmniVideo train/val JSON for sanity-check runs.
Preserves full item structure.

Modes:
  --n-total N: Randomly sample N items total (no category splitting).
  --n-per-category N: Sample up to N per content_parent_category (default).

Optional filters (mutually exclusive for sanity runs):
  --question-type X: Keep only items with this question_type (e.g. Reasoning).
  --content-parent-category X: Keep only items with this content_parent_category (e.g. Education).
"""
import argparse
import json
import random
from pathlib import Path


CATEGORY_KEY_PARENT = "content_parent_category"
CATEGORY_KEY_FINE = "content_fine_category"
QUESTION_TYPE_KEY = "question_type"
UNCATEGORIZED = "(Uncategorized)"


def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def sample_random(items: list, n_total: int, seed: int) -> list:
    """Randomly sample n_total items (or all if fewer available)."""
    rng = random.Random(seed)
    n = min(n_total, len(items))
    return rng.sample(items, n)


def sample_by_category(
    items: list,
    n_per_category: int,
    category_key: str,
    seed: int,
) -> list:
    by_cat = {}
    for item in items:
        cat = (item.get(category_key) or "").strip() or UNCATEGORIZED
        by_cat.setdefault(cat, []).append(item)
    rng = random.Random(seed)
    out = []
    for cat, group in sorted(by_cat.items()):
        n = min(n_per_category, len(group))
        out.extend(rng.sample(group, n))
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Sample N per category from OmniVideo JSON")
    parser.add_argument("--train", type=Path, default=Path("train.json"), help="Train JSON path")
    parser.add_argument("--test", type=Path, default=Path("test.json"), help="Test/val JSON path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: same as train)")
    parser.add_argument("--n-total", type=int, default=None,
                        help="If set, randomly sample this many items total (no category splitting)")
    parser.add_argument("--n-per-category", type=int, default=2, help="Max samples per category (used when --n-total not set)")
    parser.add_argument("--category", choices=["parent", "fine"], default="parent",
                        help="Category key for per-category mode (ignored when --n-total is set)")
    parser.add_argument("--question-type", type=str, default=None,
                        help="If set, keep only items with this question_type (e.g. Reasoning)")
    parser.add_argument("--content-parent-category", type=str, default=None,
                        help="If set, keep only items with this content_parent_category. Comma-separated for multiple (e.g. Education,Entertainment)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="sanity", help="Output suffix: train_<suffix>.json")
    args = parser.parse_args()

    out_dir = args.out_dir or args.train.resolve().parent
    category_key = CATEGORY_KEY_PARENT if args.category == "parent" else CATEGORY_KEY_FINE

    train_path = args.train if args.train.is_absolute() else Path(__file__).resolve().parent / args.train
    test_path = args.test if args.test.is_absolute() else Path(__file__).resolve().parent / args.test

    train_data = load_json(train_path) if train_path.exists() else []
    test_data = load_json(test_path) if test_path.exists() else []

    if args.question_type:
        train_data = [x for x in train_data if (x.get(QUESTION_TYPE_KEY) or "").strip() == args.question_type]
        test_data = [x for x in test_data if (x.get(QUESTION_TYPE_KEY) or "").strip() == args.question_type]
        if not train_data or not test_data:
            raise SystemExit(
                f"No items with question_type={args.question_type!r} found "
                f"(train: {len(train_data)}, test: {len(test_data)})"
            )
    elif args.content_parent_category:
        allowed = {c.strip() for c in args.content_parent_category.split(",") if c.strip()}
        train_data = [x for x in train_data if (x.get(CATEGORY_KEY_PARENT) or "").strip() in allowed]
        test_data = [x for x in test_data if (x.get(CATEGORY_KEY_PARENT) or "").strip() in allowed]
        if not train_data or not test_data:
            raise SystemExit(
                f"No items with content_parent_category in {allowed!r} found "
                f"(train: {len(train_data)}, test: {len(test_data)})"
            )

    if args.n_total is not None:
        train_sampled = sample_random(train_data, args.n_total, args.seed)
        test_sampled = sample_random(test_data, args.n_total, args.seed + 1)
        mode_info = f"n_total={args.n_total}"
    else:
        train_sampled = sample_by_category(train_data, args.n_per_category, category_key, args.seed)
        test_sampled = sample_by_category(test_data, args.n_per_category, category_key, args.seed + 1)
        mode_info = f"n_per_category={args.n_per_category}, key={category_key}"

    out_train = out_dir / f"train_{args.suffix}.json"
    out_test = out_dir / f"test_{args.suffix}.json"
    save_json(out_train, train_sampled)
    save_json(out_test, test_sampled)

    filter_info = ""
    if args.question_type:
        filter_info = f", question_type={args.question_type}"
    elif args.content_parent_category:
        filter_info = f", content_parent_category={args.content_parent_category}"
    print(f"Sampled {len(train_sampled)} train, {len(test_sampled)} test ({mode_info}{filter_info}, seed={args.seed})")
    print(f"  {out_train}")
    print(f"  {out_test}")


if __name__ == "__main__":
    main()
