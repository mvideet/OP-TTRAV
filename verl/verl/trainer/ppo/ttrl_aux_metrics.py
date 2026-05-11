"""
Auxiliary monitoring metrics for TTRL training (no training-time effect).

For each batch of rollouts we compute, against the *real* gold answer
(stashed in reward_model.ground_truth before the score_map overwrite):

  Deterministic (always on, ~free):
    - bleu_sentence    sacrebleu sentence-BLEU (single-reference)
    - rouge_l_f        ROUGE-L F1
    - exact_match      normalized strict match
    - contains_em      gold appears as a substring of the response
    - resp_len_tokens  whitespace token count of the response

  External judge (opt-in, costs ~$0.30 per 1k rollouts):
    - gpt_judge_score  GPT-4o-mini score 0-10 against gold, normalized to [0,1]

Env vars
--------
  TTRL_AUX_DETERMINISTIC      "1" to enable (default: "1")
  TTRL_AUX_GPT_JUDGE          "1" to enable GPT-4o-mini side-channel (default: "0")
  TTRL_AUX_GPT_MODEL          override model (default: "gpt-4o-mini-2024-07-18")
  TTRL_AUX_GPT_CONCURRENCY    parallel API calls (default: "8")
  TTRL_AUX_GPT_TIMEOUT        per-call timeout seconds (default: "15")

The metrics are attached to the per-prompt stats dict under keys prefixed
with "aux_", which compute_ttrl_metrics in ttrl_utils.py aggregates.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Deterministic metrics (BLEU / ROUGE-L / exact match)
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", (text or "").strip().lower())


def _safe_bleu(response: str, gold: str) -> float:
    """Single-reference sentence-BLEU, smoothed. Returns 0..1."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(response or "", [gold or ""]).score / 100.0
        return float(score)
    except Exception:
        return 0.0


_ROUGE_SCORER = None


def _get_rouge_scorer():
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def _safe_rouge_l(response: str, gold: str) -> float:
    try:
        sc = _get_rouge_scorer()
        out = sc.score(gold or "", response or "")
        return float(out["rougeL"].fmeasure)
    except Exception:
        return 0.0


def compute_deterministic_metrics(response: str, gold: str) -> Dict[str, float]:
    r_norm = _normalize(response)
    g_norm = _normalize(gold)
    exact = 1.0 if r_norm == g_norm and g_norm else 0.0
    contains = 1.0 if g_norm and g_norm in r_norm else 0.0
    return {
        "bleu_sentence": _safe_bleu(response, gold),
        "rouge_l_f": _safe_rouge_l(response, gold),
        "exact_match": exact,
        "contains_em": contains,
        "resp_len_tokens": float(len((response or "").split())),
    }


# ---------------------------------------------------------------------------
# Async GPT-4o-mini judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = (
    "You are grading a free-form response against a gold reference answer.\n"
    "Output a single integer from 0 to 10 — nothing else.\n"
    "10 = the response correctly conveys the gold answer.\n"
    "0  = the response is completely unrelated / wrong.\n"
    "Partial credit allowed in between.\n\n"
    "QUESTION: {question}\n\n"
    "GOLD ANSWER: {gold}\n\n"
    "RESPONSE: {response}\n\n"
    "Score (0-10):"
)


_SCORE_RE = re.compile(r"\b(10|[0-9])\b")


def _parse_score(raw: str) -> Optional[float]:
    if not raw:
        return None
    m = _SCORE_RE.search(raw.strip())
    if not m:
        return None
    try:
        n = int(m.group(1))
        return max(0.0, min(1.0, n / 10.0))
    except ValueError:
        return None


async def _judge_one(client, model: str, question: str, gold: str, response: str,
                     timeout: float, max_new_tokens: int = 8) -> Optional[float]:
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=(question or "")[:2000],
        gold=(gold or "")[:1000],
        response=(response or "")[:2000],
    )
    try:
        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        raw = completion.choices[0].message.content or ""
        return _parse_score(raw)
    except Exception:
        return None


async def _judge_batch_async(client, model: str, items: List[Dict[str, str]],
                              concurrency: int, timeout: float) -> List[Optional[float]]:
    sem = asyncio.Semaphore(concurrency)

    async def _run(item):
        async with sem:
            return await _judge_one(
                client, model,
                item.get("question", ""), item.get("gold", ""), item.get("response", ""),
                timeout,
            )

    return await asyncio.gather(*[_run(it) for it in items])


def score_with_gpt(items: List[Dict[str, str]],
                   model: str = "gpt-4o-mini-2024-07-18",
                   concurrency: int = 8,
                   timeout: float = 15.0) -> List[Optional[float]]:
    """Synchronous wrapper. items: [{question, gold, response}, ...]."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [None] * len(items)
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        scores = asyncio.run(_judge_batch_async(client, model, items, concurrency, timeout))
        return scores
    except Exception as e:
        print(f"[AUX_GPT_JUDGE] failed to dispatch ({type(e).__name__}: {e}); skipping",
              file=sys.stderr, flush=True)
        return [None] * len(items)


# ---------------------------------------------------------------------------
# Batch-level entrypoint
# ---------------------------------------------------------------------------

def compute_aux_metrics_for_batch(
    batch,
    model_outputs: List[str],
    num_prompts: int,
    n: int,
    stats_list: List[dict],
    questions: Optional[List[str]] = None,
    golds: Optional[List[str]] = None,
) -> None:
    """
    Compute auxiliary monitoring metrics and write per-prompt aggregates
    into `stats_list[i]` as new "aux_*" keys.

    Caller passes `questions` and `golds` (length=num_prompts) so this
    module doesn't have to know the dataset schema. If omitted, it tries
    common batch keys.
    """
    if not stats_list or len(stats_list) != num_prompts:
        return

    if questions is None or golds is None:
        questions = []
        golds = []
        for i in range(num_prompts):
            nb = batch[i].non_tensor_batch
            q = nb.get("question") or nb.get("prompt") or ""
            g = nb["reward_model"].get("original_gt", "")
            if isinstance(g, (list, tuple)):
                g = g[0] if g else ""
            questions.append(str(q) if q is not None else "")
            golds.append(str(g) if g is not None else "")

    deterministic_on = os.environ.get("TTRL_AUX_DETERMINISTIC", "1") == "1"
    gpt_on = os.environ.get("TTRL_AUX_GPT_JUDGE", "0") == "1"
    gpt_model = os.environ.get("TTRL_AUX_GPT_MODEL", "gpt-4o-mini-2024-07-18")
    gpt_concurrency = int(os.environ.get("TTRL_AUX_GPT_CONCURRENCY", "8"))
    gpt_timeout = float(os.environ.get("TTRL_AUX_GPT_TIMEOUT", "15"))

    if not deterministic_on and not gpt_on:
        return

    # Per-rollout deterministic metrics (cheap, always fine to compute).
    per_rollout_det: List[Dict[str, float]] = []
    if deterministic_on:
        for i in range(num_prompts):
            gold_i = golds[i]
            for j in range(n):
                resp = model_outputs[i * n + j]
                per_rollout_det.append(compute_deterministic_metrics(resp, gold_i))

    # Per-rollout GPT judge (opt-in).
    per_rollout_gpt: List[Optional[float]] = []
    if gpt_on:
        items = []
        for i in range(num_prompts):
            for j in range(n):
                items.append({
                    "question": questions[i],
                    "gold": golds[i],
                    "response": model_outputs[i * n + j],
                })
        t0 = time.time()
        per_rollout_gpt = score_with_gpt(items, model=gpt_model,
                                         concurrency=gpt_concurrency, timeout=gpt_timeout)
        valid = [s for s in per_rollout_gpt if s is not None]
        elapsed = time.time() - t0
        print(
            f"[AUX_GPT_JUDGE] judged {len(valid)}/{len(items)} rollouts in {elapsed:.1f}s "
            f"(model={gpt_model}, concurrency={gpt_concurrency})",
            file=sys.stderr, flush=True,
        )

    # Aggregate into per-prompt stats.
    for i in range(num_prompts):
        block_det = per_rollout_det[i * n : (i + 1) * n] if deterministic_on else None
        if block_det:
            stats_list[i]["aux_bleu_mean"] = _safe_mean([d["bleu_sentence"] for d in block_det])
            stats_list[i]["aux_rouge_l_mean"] = _safe_mean([d["rouge_l_f"] for d in block_det])
            stats_list[i]["aux_exact_match_mean"] = _safe_mean([d["exact_match"] for d in block_det])
            stats_list[i]["aux_contains_em_mean"] = _safe_mean([d["contains_em"] for d in block_det])
            stats_list[i]["aux_resp_len_tokens_mean"] = _safe_mean([d["resp_len_tokens"] for d in block_det])

        if gpt_on:
            block_gpt = per_rollout_gpt[i * n : (i + 1) * n]
            valid_gpt = [s for s in block_gpt if s is not None]
            if valid_gpt:
                stats_list[i]["aux_gpt_judge_mean"] = _safe_mean(valid_gpt)
                stats_list[i]["aux_gpt_judge_parse_rate"] = len(valid_gpt) / len(block_gpt)
            else:
                stats_list[i]["aux_gpt_judge_mean"] = float("nan")
                stats_list[i]["aux_gpt_judge_parse_rate"] = 0.0


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and x != x)]
    return float(sum(xs) / len(xs)) if xs else 0.0
