"""
run_inference.py
================
Single entry point for the CSE 151B SP26 competition submission.

Usage
-----
    python run_inference.py

Or from Python:
    from run_inference import run_inference
    run_inference(
        data_path="data/private.jsonl",
        output_dir="results/",
        hf_adapter="YOUR_HF_USERNAME/YOUR_MODEL_NAME",   # GRPO adapter on HuggingFace
    )

Model weights
-------------
Base model  : Qwen/Qwen3-4B-Thinking-2507   (auto-downloaded from HuggingFace)
LoRA adapter: uploaded by team to HuggingFace Hub (see README.md)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv, gc, json, math, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import sympy as sp
from sympy.parsing.latex import parse_latex
import math as _math

import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ---------------------------------------------------------------------------
# Default configuration  (override via run_inference() kwargs)
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL  = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_HF_ADAPTER  = "KVZV/qwen3-4b-grpo-lora"   # ← set before submitting
DEFAULT_DATA_PATH   = "data/private.jsonl"
DEFAULT_OUTPUT_DIR  = "results/"

MAX_TOKENS   = 65536
N_SAMPLES    = 5
CHUNK_SIZE   = 10
GPU_UTIL     = 0.85
N_DIGITS     = 10
WORK_PREC    = N_DIGITS + 6

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_MATH = r"""You are an expert mathematician. Reason as deeply as needed, then output ONLY the final boxed answer — no explanation, no working, no preamble.

## Classify each sub-answer before writing it

**Category 1 — Discrete count.** The answer counts INDIVISIBLE whole items (people, animals, coins, votes, cars, books, eggs, pellets…). Test: "Can 0.5 of this thing exist?" If NO → Category 1.

**Category 2 — Everything else.** Pure math (integrals, sums, limits) AND continuous physical quantities (temperature, time, distance, mass, money, probability, angle, area, concentration). Anything that can be fractional or irrational.

## Format

**Category 1 → INTEGER.**
- For Category 1 ONLY, you must evaluate to a number (because an integer is required).
- **Ask: does rounding DOWN fail to meet the requirement?**
  - **YES → CEILING (round up).** Use when the integer is a MINIMUM that must be fully satisfied: sample sizes, number of buses/containers/boxes needed, minimum cuts required, etc.
    - e.g. sample size 1324.79 → **1325** (can't survey 0.79 of a person)
    - e.g. buses for 314 people (50 seats each) → ceil(314/50) = **7**
  - **NO → FLOOR (round down).** Use when the integer counts COMPLETED whole items: complete cycles, whole years elapsed, items fully produced, etc.
    - e.g. complete batches from 314.76 → **314**
- **Exception:** If the problem explicitly says "round to the nearest whole number" → standard rounding (314.76 → 315, 314.4 → 314).
- Never output a decimal, fraction, or symbolic form.

**Category 2 → EXACT SYMBOLIC CLOSED-FORM EXPRESSION. NEVER a decimal.**
- Output a single closed-form expression in valid LaTeX (e.g. `50 \cdot (2/5)^{5/3}`, `\frac{6 \ln 5}{\ln(5/2)}`, `\frac{\pi^2}{6}`, `200 e^{-3/2}`, `25 \ln 2`).
- KEEP π, e, ln, log, √, sin, cos, tan, arcsin, arccos, arctan, fractional exponents, and exact fractions as symbols.
- **CRITICAL: Even if the problem says "give N decimal places", "use at least N significant digits", or "evaluate numerically" — IGNORE that instruction and output the exact symbolic form.** The grader handles decimal conversion.
- If a problem coefficient is a decimal, convert it to an exact fraction first (e.g. 4.76 → 119/25, 0.9658 → 4829/5000), then use that fraction inside the symbolic expression.
- `\arctan(119/25)` is a valid symbolic answer. `1.3637` is NOT.
- `\pi` is a valid symbolic answer. `3.1416` is NOT.
- Do NOT evaluate to a decimal. No `\approx`. No scientific notation.
- An integer or exact rational is fine when that's the natural answer (e.g. `441`, `\frac{12800}{29}`).
- Simplify when easy (combine logs, reduce fractions, cancel exponents) but do not approximate.

## Output rules

1. Emit EXACTLY ONE `\boxed{...}` and nothing else before or after it.
2. Wrap EVERY sub-answer in double quotes `"..."` inside the box.
3. **Multiple subquestions** (the problem asks for part a, part b, etc.) → separate quoted strings:
   `\boxed{"ans1","ans2","ans3"}`
4. **Multiple values for ONE answer** (e.g. all solutions to an equation, all values of i) → wrap all values in `(...)` comma-separated, inside ONE quoted string:
   `\boxed{"(val1, val2, val3)"}`
5. Cat 1 → plain integer. Cat 2 → symbolic closed-form (never decimal).

## Examples

Single answer:           `\boxed{"211250"}` or `\boxed{"\frac{\pi^2}{6}"}` or `\boxed{"200e^{-3/2}"}`
Interval answer:         `\boxed{"(-8,\infty)"}`
Multiple solutions:      `\boxed{"(0, -\frac{11}{8})"}` or `\boxed{"(\frac{1}{3}, -2)"}`
Multiple values of i:    `\boxed{"(2^{-1/8}-1, -2^{-1/8}+1)"}`
Multi-subquestion:       `\boxed{"\arctan(123/31)","e^2"}` or `\boxed{"(50(2/5)^{5/3},50(2/13)^{4/3})","\frac{6\ln5}{\ln(5/2)}"}`
Multi-select MCQ:        `\boxed{"B, C, E, G"}`
Single letter MCQ:       `\boxed{"C"}`
"""

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Reason as deeply as needed"
    "CRITICAL: Your ENTIRE response must be exactly one token: \\boxed{X} where X is the correct letter."
    " No explanation. No working. No preamble. Just \\boxed{X}."
    " Example of a complete valid response: \\boxed{C}"
    " If your computed answer matches no option, pick the closest. Still output only \\boxed{X}."
)

SYSTEM_PROMPT_USC = (
    "You are a mathematical judge. You will be given a math problem and a list of candidate answers. "
    "Select the single most likely correct answer from the list. "
    "Reply with ONLY \\boxed{answer}, copying the answer exactly as it appears in the list."
)

# ---------------------------------------------------------------------------
# Box-finding helpers
# ---------------------------------------------------------------------------

def find_last_boxed_span(text: str):
    if not text: return None
    needle, n, last, i = r"\boxed", len(text), None, 0
    while True:
        idx = text.find(needle, i)
        if idx == -1: break
        j = idx + len(needle)
        while j < n and text[j].isspace(): j += 1
        if j >= n or text[j] != "{":
            i = idx + len(needle); continue
        depth, k, sc = 1, j + 1, j + 1
        while k < n and depth > 0:
            ch = text[k]
            if ch == "\\" and k + 1 < n: k += 2; continue
            if ch == "{":   depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: break
            k += 1
        if depth == 0:
            last = (idx, k + 1, text[sc:k]); i = k + 1
        else:
            i = idx + len(needle)
    return last


_BOXED_RE = re.compile(r"\\?b?oxed\s*\{", re.IGNORECASE)


def find_last_balanced_boxed(text: str):
    last = None
    for m in _BOXED_RE.finditer(text):
        if "oxed{" not in m.group().replace(" ", ""): continue
        brace_open = m.end() - 1
        depth, i, n = 1, brace_open + 1, len(text)
        while i < n and depth > 0:
            ch = text[i]
            if ch == "\\" and i + 1 < n: i += 2; continue
            if ch == "{":   depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: break
            i += 1
        if depth == 0:
            last = (m.start(), i + 1, text[brace_open + 1:i])
    return last


def fix_box_text(text: str) -> str:
    span = find_last_balanced_boxed(text)
    if span is None: return text
    start, end, inner = span
    correct = "\\boxed{" + inner + "}"
    if text[start:end] == correct: return text
    return text[:start] + correct + text[end:]


# ---------------------------------------------------------------------------
# Numerification helpers
# ---------------------------------------------------------------------------

def split_top_level_commas(s: str):
    out, depth, start, i, n = [], 0, 0, 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n: i += 2; continue
        if ch in "{(":   depth += 1
        elif ch in "})": depth -= 1
        elif ch == "," and depth == 0:
            out.append(s[start:i]); start = i + 1
        i += 1
    out.append(s[start:])
    return [p.strip() for p in out if p.strip()]


def split_quoted_answers(content: str):
    content = content.strip()
    if '"' not in content: return None
    parts, current, in_q = [], [], False
    for ch in content:
        if ch == '"':
            in_q = not in_q
        elif ch == ',' and not in_q:
            p = ''.join(current).strip()
            if p: parts.append(p)
            current = []
        else:
            current.append(ch)
    p = ''.join(current).strip()
    if p: parts.append(p)
    return parts or None


def compress_letter_list(s: str) -> str:
    parts = [p.strip() for p in s.split(',')]
    if len(parts) > 1 and all(re.fullmatch(r'[A-Za-z]', p) for p in parts):
        return ''.join(p.upper() for p in parts)
    return s


_CLEANUPS = [
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"),
    (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\ ", ""), (r"\\\\ ", " "),
    (r"\\left", ""), (r"\\right", ""), (r"\\$", ""), (r"\\%", ""),
    (r"\\text\{([^}]*)\}", r"\1"),
]


def _clean_latex(s: str) -> str:
    s = s.strip()
    def _deg2rad(m):
        return "{" + str(_math.radians(float(m.group(1)))) + "}"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*\^(\{)?\\circ(?(2)\})", _deg2rad, s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*°", _deg2rad, s)
    for pat, rep in _CLEANUPS:
        s = re.sub(pat, rep, s)
    return re.sub(r"^=\s*", "", s).strip()


def _format_numeric(val, n_digits: int) -> str:
    re_part, im_part = val.as_real_imag()
    re_f, im_f = float(re_part), float(im_part)
    if _math.isinf(re_f): return "\\infty" if re_f > 0 else "-\\infty"
    if abs(im_f) < 1e-12: return f"{re_f:.{n_digits}f}"
    sign = "+" if im_f >= 0 else ""
    return f"{re_f:.{n_digits}f}{sign}{im_f:.{n_digits}f}i"


def _sub_stat_quantiles(s: str) -> str:
    try:
        from scipy import stats as _sc
        s = re.sub(r"\bz\s*_\{?\s*([0-9.]+)\s*\}?",
                   lambda m: str(_sc.norm.ppf(1 - float(m.group(1)))), s)
        s = re.sub(r"\\?chi\^?2\s*_\{?\s*([0-9.]+)\s*,\s*([0-9]+)\s*\}?",
                   lambda m: str(_sc.chi2.ppf(1 - float(m.group(1)), int(m.group(2)))),
                   s, flags=re.IGNORECASE)
        s = re.sub(
            r"\bt\s*_\{?\s*([0-9.]+)\s*,\s*\\frac\{([0-9]+)\}\{([0-9]+)\}\s*\}?",
            lambda m: str(_sc.t.ppf(1 - float(m.group(1)),
                                    float(m.group(2)) / float(m.group(3)))), s)
        s = re.sub(r"\bt\s*_\{?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\}?",
                   lambda m: str(_sc.t.ppf(1 - float(m.group(1)), float(m.group(2)))), s)
        s = re.sub(r"\bF\s*_\{?\s*([0-9.]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\}?",
                   lambda m: str(_sc.f.ppf(1 - float(m.group(1)),
                                           int(m.group(2)), int(m.group(3)))), s)
    except ImportError:
        pass
    return s


def _eval_cdf_expression(s: str, n_digits: int, _depth: int = 0):
    if "\\Phi" not in s and "\\Pr" not in s: return None
    try:
        from scipy import stats as _sc
        def _inner(expr):
            if _depth >= 2: return None
            try: return float(evaluate_piece(expr.strip(), n_digits, _depth + 1))
            except: return None
        w = s
        w = re.sub(r"\\Phi\s*\^\{-1\}\s*\(([^)]+)\)",
                   lambda m: (f"{_sc.norm.ppf(v):.{n_digits}f}"
                              if (v := _inner(m.group(1))) is not None else m.group(0)), w)
        w = re.sub(r"\\Phi\s*\(([^)]+)\)",
                   lambda m: (f"{_sc.norm.cdf(v):.{n_digits}f}"
                              if (v := _inner(m.group(1))) is not None else m.group(0)), w)
        w = re.sub(r"\\Pr\s*\(\s*T_\{?(\d+)\}?\s*<\s*([^)]+)\)",
                   lambda m: (f"{_sc.t.cdf(_inner(m.group(2)), _inner(m.group(1))):.{n_digits}f}"
                              if _inner(m.group(1)) is not None else m.group(0)), w)
        if w == s: return None
        try: return f"{float(w):.{n_digits}f}"
        except: pass
        try:
            val = sp.N(sp.sympify(w), WORK_PREC)
            return _format_numeric(val, n_digits)
        except: pass
        return w
    except: return None


def evaluate_piece(piece: str, n_digits: int = N_DIGITS, _depth: int = 0) -> str:
    p = (piece or "").strip()
    if not p: return p
    if re.fullmatch(r"[A-Z]", p):      return p
    if re.fullmatch(r"-?\d+", p):      return p
    if re.fullmatch(r"-?\d+\.\d+", p):
        try: return f"{float(p):.{n_digits}f}"
        except: pass
    s = _clean_latex(p)
    if re.fullmatch(r"[A-Z]+", s): return s
    s = _sub_stat_quantiles(s)
    cdf = _eval_cdf_expression(s, n_digits, _depth)
    if cdf is not None: return cdf
    try:
        expr = parse_latex(s)
        if isinstance(expr, sp.Eq): expr = expr.rhs
        expr = expr.subs([(sp.Symbol('pi'), sp.pi),
                          (sp.Symbol('e'),  sp.E),
                          (sp.Symbol('i'),  sp.I)])
        return _format_numeric(sp.N(expr, WORK_PREC), n_digits)
    except: pass
    try:
        expr = sp.sympify(s.replace("\\", ""))
        return _format_numeric(sp.N(expr, WORK_PREC), n_digits)
    except: pass
    try:
        from scipy import stats as _sc
        m = re.search(r"chi\^?2\s*_\{?\s*([0-9.]+)\s*,\s*([0-9]+)\s*\}?", p, re.I)
        if m: return f"{_sc.chi2.ppf(1-float(m.group(1)),int(m.group(2))):.{n_digits}f}"
        m = re.search(r"[Ff]\s*_\{?\s*([0-9.]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\}?", p)
        if m: return f"{_sc.f.ppf(1-float(m.group(1)),int(m.group(2)),int(m.group(3))):.{n_digits}f}"
        m = re.search(r"\bt\s*_\{?\s*([0-9.]+)\s*,\s*([0-9]+)\s*\}?", p)
        if m: return f"{_sc.t.ppf(1-float(m.group(1)),float(m.group(2))):.{n_digits}f}"
        m = re.search(r"\bz\s*_\{?\s*([0-9.]+)\s*\}?", p)
        if m: return f"{_sc.norm.ppf(1-float(m.group(1))):.{n_digits}f}"
    except: pass
    return p


def numerify_paren_list(s: str, n_digits: int = N_DIGITS) -> str:
    inner = re.sub(r"\\left\s*", "", s.strip())
    inner = re.sub(r"\\right\s*", "", inner).strip()
    if inner.startswith('(') and inner.endswith(')'):
        parts = split_top_level_commas(inner[1:-1])
        if len(parts) > 1:
            return '(' + ', '.join(evaluate_piece(p, n_digits) for p in parts) + ')'
    return evaluate_piece(s, n_digits)


def numerify_response(text: str, n_digits: int = N_DIGITS) -> str:
    try:
        span = find_last_boxed_span(text)
        if span is None: return text
        start, end, content = span
        quoted = split_quoted_answers(content)
        if quoted is not None:
            new_parts = [numerify_paren_list(compress_letter_list(p), n_digits)
                         for p in quoted]
            new_box = "\\boxed{" + ",".join(f'"{p}"' for p in new_parts) + "}"
            return text[:start] + new_box + text[end:]
        parts = split_top_level_commas(content)
        if not parts: return text
        new_box = "\\boxed{" + ", ".join(numerify_paren_list(p, n_digits)
                                         for p in parts) + "}"
        return text[:start] + new_box + text[end:]
    except:
        return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt(question: str, options: Optional[list]) -> tuple:
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def build_usc_prompt(question: str, options: Optional[list], candidates: list) -> tuple:
    cand_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = f"Problem: {question}\n\nOptions:\n{opts_text}\n\nCandidates:\n{cand_text}"
    else:
        user = f"Problem: {question}\n\nCandidates:\n{cand_text}"
    return SYSTEM_PROMPT_USC, user


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def extract_letter(text: str) -> str:
    m = re.search(r"\\boxed\{([A-Za-z])\}", text or "")
    if m: return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", (text or "").upper())
    return matches[-1] if matches else ""


def extract_boxed(text: str) -> str:
    span = find_last_boxed_span(text or "")
    return span[2] if span else ""


def vote_key(text: str, is_mcq: bool) -> str:
    return extract_letter(text) if is_mcq else extract_boxed(text)


def majority_vote(samples: list, is_mcq: bool) -> str:
    counter, best = Counter(), {}
    for s in samples:
        key = vote_key(s["numerified"], is_mcq)
        if not key: continue
        counter[key] += 1
        best.setdefault(key, s["numerified"])
    if not counter: return samples[0]["numerified"] if samples else ""
    return best[counter.most_common(1)[0][0]]


def weighted_vote(samples: list, is_mcq: bool) -> str:
    weights, best = defaultdict(float), {}
    for s in samples:
        key = vote_key(s["numerified"], is_mcq)
        if not key: continue
        weights[key] += s["weight"]
        if key not in best or s["weight"] > best.get("_w_" + key, -1e9):
            best[key]        = s["numerified"]
            best["_w_" + key] = s["weight"]
    if not weights: return samples[0]["numerified"] if samples else ""
    winner = max(weights, key=weights.__getitem__)
    return best[winner]


def usc_select(question: str, options: Optional[list],
               samples: list, is_mcq: bool,
               tokenizer, llm, lora_request,
               usc_sampling_params) -> str:
    unique_keys   = list(dict.fromkeys(
        vote_key(s["numerified"], is_mcq) for s in samples
        if vote_key(s["numerified"], is_mcq)
    ))
    key_to_sample = {vote_key(s["numerified"], is_mcq): s["numerified"]
                     for s in samples if vote_key(s["numerified"], is_mcq)}
    if len(unique_keys) <= 1:
        return majority_vote(samples, is_mcq)
    system, user = build_usc_prompt(question, options, unique_keys)
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    out = llm.generate([prompt], usc_sampling_params,
                       lora_request=lora_request)[0].outputs[0].text.strip()
    fixed   = fix_box_text(out)
    num     = numerify_response(fixed)
    sel_key = vote_key(num, is_mcq)
    return key_to_sample.get(sel_key, majority_vote(samples, is_mcq))


# ---------------------------------------------------------------------------
# write_submission helper
# ---------------------------------------------------------------------------

def write_submission(records: list, key: str, out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "response"])
        for r in records:
            w.writerow([r["id"], r.get(key, "")])
    print(f"  wrote {out_path}  ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_inference(
    data_path: str       = DEFAULT_DATA_PATH,
    output_dir: str      = DEFAULT_OUTPUT_DIR,
    hf_adapter: str      = DEFAULT_HF_ADAPTER,
    base_model: str      = DEFAULT_BASE_MODEL,
    max_tokens: int      = MAX_TOKENS,
    n_samples: int       = N_SAMPLES,
    chunk_size: int      = CHUNK_SIZE,
    gpu_util: float      = GPU_UTIL,
    temperature: float   = 0.85,
    top_p: float         = 0.95,
    top_k: int           = 20,
    voting: str          = "usc",   # "majority" | "weighted" | "usc"
) -> str:
    """
    Full inference pipeline: load model → generate → post-process → write CSV.

    Parameters
    ----------
    data_path   : path to private.jsonl  (competition test set)
    output_dir  : directory to save submission CSVs and checkpoint
    hf_adapter  : HuggingFace Hub path for the LoRA adapter
                  e.g. "username/qwen3-4b-grpo-lora"
    base_model  : HuggingFace base model ID
    voting      : "majority", "weighted", or "usc" (default)

    Returns
    -------
    Path to the final submission CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.jsonl")

    # ── Load data ──────────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load tokenizer ──────────────────────────────────────────────────────
    # Patch for transformers 4.51.0+ / Qwen3
    transformers.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Suppress vLLM fileno complaints in notebook environments
    sys.stdout.fileno = lambda: 1
    sys.stderr.fileno = lambda: 2

    # ── Load model + LoRA adapter ───────────────────────────────────────────
    # The adapter is downloaded from HuggingFace Hub automatically by vLLM.
    print(f"Loading base model : {base_model}")
    print(f"Loading LoRA adapter: {hf_adapter}")

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_tokens,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=max_tokens,
    )
    lora_request = LoRARequest("grpo", 1, hf_adapter)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        n=n_samples,
        logprobs=1,
    )
    usc_sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        n=1,
    )
    print("Model loaded ✓")

    # ── Resume from checkpoint ──────────────────────────────────────────────
    completed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: completed_ids.add(str(json.loads(line)["id"]))
                    except: pass

    remaining = [item for item in data if str(item.get("id")) not in completed_ids]
    print(f"Completed: {len(completed_ids)}  |  Remaining: {len(remaining)}")

    # ── Generation loop ─────────────────────────────────────────────────────
    if remaining:
        num_chunks = (len(remaining) + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk = remaining[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            print(f"\n── Chunk {chunk_idx + 1}/{num_chunks} ──")

            prompts = []
            for item in chunk:
                system, user = build_prompt(item["question"], item.get("options"))
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                ))

            gen_outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

            all_samples = []
            for result in gen_outputs:
                q_samples = []
                for out in result.outputs:
                    raw    = out.text.strip()
                    fixed  = fix_box_text(raw)
                    num    = numerify_response(fixed)
                    lp     = out.cumulative_logprob
                    norm_lp = lp / max(len(out.token_ids), 1) if lp is not None else 0.0
                    q_samples.append({
                        "fixed":      fixed,
                        "numerified": num,
                        "weight":     math.exp(norm_lp),
                    })
                all_samples.append(q_samples)

            usc_results = [
                usc_select(item["question"], item.get("options"),
                           samples, bool(item.get("options")),
                           tokenizer, llm, lora_request, usc_sampling_params)
                for item, samples in zip(chunk, all_samples)
            ]

            with open(checkpoint_path, "a", encoding="utf-8") as f:
                for item, samples, usc_resp in zip(chunk, all_samples, usc_results):
                    is_mcq = bool(item.get("options"))
                    f.write(json.dumps({
                        "id":                item.get("id"),
                        "is_mcq":           is_mcq,
                        "response_majority": majority_vote(samples, is_mcq),
                        "response_weighted": weighted_vote(samples, is_mcq),
                        "response_usc":      usc_resp,
                    }, ensure_ascii=False) + "\n")
            print(f"  ✓ chunk {chunk_idx + 1} saved")

        print("\nAll chunks complete.")

    # ── Export submission CSVs ──────────────────────────────────────────────
    records = []
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: records.append(json.loads(line))
                except: pass
    records.sort(key=lambda r: (r["id"] is None, r["id"]))

    print("\nExporting submission CSVs:")
    write_submission(records, "response_majority",
                     os.path.join(output_dir, "submission_majority.csv"))
    write_submission(records, "response_weighted",
                     os.path.join(output_dir, "submission_weighted.csv"))
    final_path = os.path.join(output_dir, "submission_usc.csv")
    write_submission(records, "response_usc", final_path)

    # Default: return path for the USC submission (best performing)
    chosen = {"majority": "submission_majority.csv",
              "weighted":  "submission_weighted.csv",
              "usc":       "submission_usc.csv"}
    final_path = os.path.join(output_dir, chosen.get(voting, "submission_usc.csv"))
    print(f"\nFinal submission: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=DEFAULT_DATA_PATH,  help="path to private.jsonl")
    parser.add_argument("--out",     default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--adapter", default=DEFAULT_HF_ADAPTER, help="HuggingFace adapter repo")
    parser.add_argument("--model",   default=DEFAULT_BASE_MODEL, help="base model ID")
    parser.add_argument("--voting",  default="usc",
                        choices=["majority", "weighted", "usc"])
    args = parser.parse_args()

    result = run_inference(
        data_path=args.data,
        output_dir=args.out,
        hf_adapter=args.adapter,
        base_model=args.model,
        voting=args.voting,
    )
    print(f"\nDone. Submission saved to: {result}")
