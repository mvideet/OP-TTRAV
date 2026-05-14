"""
Arena-Hard pairwise judge with GPT-4o-mini, position-controlled.

Implements the canonical Arena-Hard methodology:
  - Pairwise comparison: our_model vs baseline_model (per prompt)
  - Uses Arena-Hard's official judge system prompt
  - Verdicts: A>>B / A>B / A=B / B>A / B>>A
  - Both orderings run (swap A/B) to neutralize position bias
  - Final metric: win-rate against baseline, position-controlled

Cost: GPT-4o-mini is ~$0.15/1M input + $0.60/1M output. With ~5K input tokens
per judge call × 2 orderings × N prompts ≈ ~$1 per 200-prompt eval.

Inputs are two JSONL files:
  --model_rollouts:    {uid, response} per line (our model's responses)
  --baseline_rollouts: {uid, response} per line (baseline model, e.g. GPT-4.1)

Usage:
  python verl/scripts/judge_arena_hard_pairwise.py \\
    --model_rollouts model_outputs.jsonl \\
    --baseline_rollouts baseline_outputs.jsonl \\
    --questions verl/data/ArenaHard-v2.0-TTRL/test.json \\
    --judge_model gpt-4o-mini-2024-07-18 \\
    --output ah_winrate_step300.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Arena-Hard official judge system prompt (verbatim from lmarena-ai/arena-hard-auto)
JUDGE_SYSTEM_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: \"My final verdict is tie: [[A=B]]\"."""


USER_TEMPLATE = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""


VERDICT_RE = re.compile(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]")

# Map verdict label -> score for "A wins" (signed). Higher = A wins more.
VERDICT_SCORE = {
    "A>>B": 1.0,
    "A>B": 0.5,
    "A=B": 0.0,
    "B>A": -0.5,
    "B>>A": -1.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_rollouts", required=True, help="JSONL: {uid, response}")
    p.add_argument("--baseline_rollouts", required=True, help="JSONL: {uid, response}")
    p.add_argument("--questions", required=True, help="JSON list with {id, prompt} per row")
    p.add_argument("--judge_model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--output", default="ah_winrate.json")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=2048,
                   help="Judge max output tokens (it writes its own answer + reasoning + verdict)")
    p.add_argument("--limit", type=int, default=None, help="Cap n pairs for testing")
    p.add_argument("--our_label", default="model", help="Label for our model in the output")
    p.add_argument("--baseline_label", default="baseline", help="Label for the baseline")
    p.add_argument("--step", type=int, default=None,
                   help="If --model_rollouts is a multi-step dump_rollouts JSONL, filter to this step only")
    return p.parse_args()


async def judge_one(client, judge_model, question, answer_a, answer_b, max_tokens):
    """Single GPT-4o-mini call. Returns (verdict_str_or_None, score_for_A)."""
    try:
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    question=question[:8000],  # truncate to fit context
                    answer_a=answer_a[:6000],
                    answer_b=answer_b[:6000],
                )},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        m = VERDICT_RE.search(raw)
        if not m:
            return None, 0.0
        verdict = m.group(1)
        return verdict, VERDICT_SCORE[verdict]
    except Exception as e:
        return None, 0.0


async def run_pairs(items, judge_model, concurrency, max_tokens):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(items)

    async def _worker(idx, q, a, b):
        async with sem:
            v, s = await judge_one(client, judge_model, q, a, b, max_tokens)
            results[idx] = (v, s)

    tasks = [_worker(i, item["question"], item["a"], item["b"])
             for i, item in enumerate(items)]
    t0 = time.time()
    for i, _ in enumerate(asyncio.as_completed(tasks), 1):
        await _
        if i % 50 == 0:
            print(f"  {i}/{len(items)} judged ({time.time()-t0:.0f}s)", flush=True)
    return results


def main():
    args = parse_args()

    # Load data
    print(f"Loading questions from {args.questions}...", flush=True)
    questions = {r["id"]: r["prompt"] for r in json.load(open(args.questions))}

    print(f"Loading model rollouts from {args.model_rollouts}"
          + (f" (filtering to step={args.step})" if args.step is not None else "") + "...", flush=True)
    model_resp = {}
    for line in open(args.model_rollouts):
        r = json.loads(line)
        if args.step is not None and r.get("step") != args.step:
            continue
        uid = r.get("id") or r.get("uid")
        rollouts = r.get("rollouts") or [r.get("response", "")]
        model_resp[uid] = rollouts[0] if rollouts else ""

    print(f"Loading baseline rollouts from {args.baseline_rollouts}...", flush=True)
    baseline_resp = {}
    for line in open(args.baseline_rollouts):
        r = json.loads(line)
        uid = r.get("id") or r.get("uid")
        # arena-hard model_answer format: messages -> assistant -> content -> answer
        text = r.get("response", "")
        if not text:
            msgs = r.get("messages") or []
            for m in msgs:
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, dict):
                        text = c.get("answer") or c.get("content") or ""
                    else:
                        text = str(c)
                    break
        baseline_resp[uid] = text

    # Build pair list: each uid -> 2 games (model as A, then model as B)
    uids = sorted(set(model_resp) & set(baseline_resp) & set(questions))
    print(f"Common uids: {len(uids)} (model={len(model_resp)}, baseline={len(baseline_resp)}, qs={len(questions)})", flush=True)
    if args.limit:
        uids = uids[: args.limit]

    pair_items = []
    for uid in uids:
        q = questions[uid]
        m = model_resp[uid]
        b = baseline_resp[uid]
        # Game 1: model as A
        pair_items.append({"uid": uid, "order": "model_as_A", "question": q, "a": m, "b": b})
        # Game 2: model as B (swap for position-control)
        pair_items.append({"uid": uid, "order": "model_as_B", "question": q, "a": b, "b": m})

    print(f"Total judge calls: {len(pair_items)} (= {len(uids)} prompts × 2 orderings)", flush=True)

    # Judge
    results = asyncio.run(run_pairs(pair_items, args.judge_model, args.concurrency, args.max_tokens))

    # Aggregate
    per_uid: Dict[str, Dict[str, Optional[Tuple[str, float]]]] = defaultdict(dict)
    n_parsed_ok = 0
    for item, res in zip(pair_items, results):
        v, s = res or (None, 0.0)
        # Convert to "our model wins" sign
        # When order=model_as_A: positive score for A = our model wins
        # When order=model_as_B: positive score for A = baseline wins, so flip
        if item["order"] == "model_as_A":
            our_score = s
        else:
            our_score = -s
        per_uid[item["uid"]][item["order"]] = (v, our_score)
        if v is not None:
            n_parsed_ok += 1

    # Per-prompt: average over the two orderings (position-controlled score)
    scored_uids = []
    n_wins, n_losses, n_ties = 0, 0, 0
    for uid in uids:
        results_for_uid = per_uid[uid]
        scores = [v[1] for v in results_for_uid.values() if v is not None]
        if len(scores) != 2:
            continue
        avg = sum(scores) / 2.0
        scored_uids.append(avg)
        if avg > 0:
            n_wins += 1
        elif avg < 0:
            n_losses += 1
        else:
            n_ties += 1

    if not scored_uids:
        print("ERROR: no valid pairs judged", flush=True)
        sys.exit(1)

    mean_score = sum(scored_uids) / len(scored_uids)  # in [-1, +1]
    # Convert to win-rate format: (wins + 0.5*ties) / total
    # But our scores can be +/- 0.5 or +/- 1 due to "significantly" vs "slightly".
    # Standard Arena-Hard reports: win-rate as a percentage where a tie counts 0.5.
    winrate_pct = 0.0
    for s in scored_uids:
        # Map [-1, +1] → [0, 1] win prob
        winrate_pct += (s + 1.0) / 2.0
    winrate_pct = 100.0 * winrate_pct / len(scored_uids)

    print(f"\n{'='*60}\nResults (judge={args.judge_model}, n={len(scored_uids)}):")
    print(f"  {args.our_label} vs {args.baseline_label}")
    print(f"  raw wins/ties/losses: {n_wins}/{n_ties}/{n_losses}")
    print(f"  mean score (range [-1,+1]): {mean_score:+.4f}")
    print(f"  win-rate (linearly scaled): {winrate_pct:.2f}%")
    print(f"  parse-rate: {n_parsed_ok}/{len(pair_items)} ({100*n_parsed_ok/len(pair_items):.1f}%)")

    out = {
        "judge_model": args.judge_model,
        "our_label": args.our_label,
        "baseline_label": args.baseline_label,
        "n_prompts": len(scored_uids),
        "n_judge_calls": len(pair_items),
        "n_parsed_ok": n_parsed_ok,
        "raw_wins": n_wins,
        "raw_ties": n_ties,
        "raw_losses": n_losses,
        "mean_score": mean_score,
        "winrate_pct": winrate_pct,
        "per_uid": {uid: {k: {"verdict": v[0], "our_score": v[1]} for k, v in d.items() if v is not None}
                    for uid, d in per_uid.items()},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote: {args.output}")


if __name__ == "__main__":
    main()
