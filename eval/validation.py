"""
Validation module for DEVS code generation.

Pipeline:
  Phase 1  — MS4 Me compiles/executes the model locally (native DNL/SES syntax enforcement).
             Compilation failure → classified as incorrect (syntax error); evaluation stops.
  Phase 2a — Automated matching against the canonical solution (5 independent functions).
             If any function matches → correct, no further review.
             Otherwise → logs to canonical_mismatches.log and proceeds to Phase 2b.
  Phase 2b — LLM judge as a metric (kept as-is; always runs when Phase 2a does not match).
"""
import re
import json
import logging
import os
import subprocess
import tempfile
from typing import Callable, Optional, Tuple

logger = logging.getLogger("devs-eval")

# ── MS4 Me executable ──────────────────────────────────────────────────────────
# Override via env var MS4ME_PATH or MS4ME_COMPILE_ARGS (space-separated flags).
MS4ME_EXECUTABLE   = os.environ.get("MS4ME_PATH", "ms4me")
MS4ME_COMPILE_ARGS = os.environ.get("MS4ME_COMPILE_ARGS", "--compile").split()
MS4ME_TIMEOUT      = int(os.environ.get("MS4ME_TIMEOUT", "30"))  # seconds

# ── Canonical mismatch log ─────────────────────────────────────────────────────
MISMATCH_LOG_PATH = os.environ.get("DEVS_MISMATCH_LOG", "logs/canonical_mismatches.log")
_mismatch_logger_ready = False


def _mismatch_logger() -> logging.Logger:
    """Dedicated logger; writes only to canonical_mismatches.log (does not propagate)."""
    global _mismatch_logger_ready
    ml = logging.getLogger("devs-eval.mismatch")
    if not _mismatch_logger_ready:
        log_dir = os.path.dirname(MISMATCH_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(MISMATCH_LOG_PATH, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
        )
        ml.addHandler(fh)
        ml.setLevel(logging.WARNING)
        ml.propagate = False
        _mismatch_logger_ready = True
    return ml


SEMANTIC_JUDGE_SYSTEM = (
    "You are an expert on DEVS discrete-event specification code. "
    "Judge whether the generated code fulfills the user's request. "
    "Reply with ONLY a single JSON object, no markdown: "
    '{"valid": true or false, "reason": "brief explanation"}'
)


# ── Phase 1: MS4 Me compilation ───────────────────────────────────────────────

def compile_with_ms4me(code: str) -> Tuple[bool, str]:
    """
    Writes the code to a temporary file and invokes MS4 Me locally.
    Retorna (sucesso, mensagem_de_erro).
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dnl", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        cmd = [MS4ME_EXECUTABLE] + MS4ME_COMPILE_ARGS + [tmp_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=MS4ME_TIMEOUT,
        )

        if result.returncode == 0:
            return True, ""

        stderr = (result.stderr or result.stdout or "").strip()
        return False, stderr[:1000] or "MS4 Me returned non-zero exit code"

    except FileNotFoundError:
        msg = (
            f"MS4 Me executable not found: {MS4ME_EXECUTABLE!r}. "
            "Set the MS4ME_PATH environment variable to the correct path."
        )
        logger.error(msg)
        return False, msg
    except subprocess.TimeoutExpired:
        msg = f"MS4 Me timed out after {MS4ME_TIMEOUT}s"
        logger.error(msg)
        return False, msg
    except Exception as e:
        logger.error("compile_with_ms4me unexpected error: %s", e)
        return False, str(e)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Phase 2a: canonical matching functions ────────────────────────────────────

JACCARD_THRESHOLD = float(os.environ.get("DEVS_JACCARD_THRESHOLD", "0.85"))

# DNL/SES statement patterns
_STATEMENT_RE = re.compile(
    r"(?:"
    r"passivate\s+in\s+\w+(?:\s+for\s+time\s+[\d.]+)?"
    r"|hold\s+in\s+\w+(?:\s+for\s+time\s+[\d.]+)?"
    r"|from\s+\w+\s+go\s+to\s+\w+(?:\s+when\s+\w+)?"
    r"|(?:external|internal|output)\s+\w+"
    r")",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> set:
    """Split into lowercase alphanumeric tokens."""
    return set(t for t in re.split(r"[^\w]+", text.lower()) if t)


def match_exact(generated: str, canonical: str) -> Tuple[bool, str]:
    """Exact comparison after normalizing whitespace and case."""
    ok = _norm(generated) == _norm(canonical)
    return ok, "" if ok else "exact: normalised strings differ"


def match_token_set(generated: str, canonical: str) -> Tuple[bool, str]:
    """All tokens from the canonical solution must be present in the generated code."""
    missing = _tokenize(canonical) - _tokenize(generated)
    if not missing:
        return True, ""
    return False, f"token_set: missing tokens: {sorted(missing)[:20]}"


def match_jaccard(generated: str, canonical: str) -> Tuple[bool, str]:
    """Jaccard similarity on token sets (configurable threshold)."""
    gen_t = _tokenize(generated)
    can_t = _tokenize(canonical)
    union = gen_t | can_t
    if not union:
        return True, ""
    score = len(gen_t & can_t) / len(union)
    ok = score >= JACCARD_THRESHOLD
    return ok, "" if ok else f"jaccard: score={score:.3f} < threshold={JACCARD_THRESHOLD}"


def match_line_set(generated: str, canonical: str) -> Tuple[bool, str]:
    """Compares sets of normalized lines (order-insensitive)."""
    def line_set(text: str) -> set:
        return {_norm(l) for l in text.splitlines() if _norm(l)}

    missing = line_set(canonical) - line_set(generated)
    if not missing:
        return True, ""
    preview = sorted(missing)[:5]
    return False, f"line_set: missing lines: {preview}"


def match_statements(generated: str, canonical: str) -> Tuple[bool, str]:
    """Extracts DNL/SES statements and compares their sets."""
    def extract(text: str) -> set:
        return {_norm(m) for m in _STATEMENT_RE.findall(text)}

    can_stmts = extract(canonical)
    if not can_stmts:
        return True, "statements: no canonical statements found"
    missing = can_stmts - extract(generated)
    if not missing:
        return True, ""
    return False, f"statements: missing: {sorted(missing)[:5]}"


# Registry of all matching functions (order: strictest → most permissive)
CANONICAL_MATCHERS: list[Tuple[str, Callable]] = [
    ("exact",       match_exact),
    ("token_set",   match_token_set),
    ("jaccard",     match_jaccard),
    ("line_set",    match_line_set),
    ("statements",  match_statements),
]


def canonical_match(
    generated: str,
    canonical: str,
    context: str = "",
) -> Tuple[bool, dict]:
    """
    Executes all matchers.
    Returns (any_match, {name: {match, detail}}).
    If no match, logs to canonical_mismatches.log.
    """
    results: dict = {}
    any_match = False

    for name, fn in CANONICAL_MATCHERS:
        ok, detail = fn(generated, canonical)
        results[name] = {"match": ok, "detail": detail}
        if ok:
            any_match = True

    if not any_match:
        _mismatch_logger().warning(
            "MISMATCH%s\n"
            "  matchers : %s\n"
            "  canonical: %s\n"
            "  generated: %s",
            f" [{context}]" if context else "",
            {k: v["detail"] for k, v in results.items()},
            canonical[:400],
            generated[:400],
        )

    return any_match, results


# ── Phase 2b: LLM semantic judge (metric — mantido tal qual) ──────────────────

class ValidationResult:
    """Holds semantic validation results."""

    def __init__(self):
        self.syntax_valid    = False
        self.syntax_error    = ""
        self.semantic_valid  = False
        self.semantic_error  = ""
        self.llm_analysis    = ""

    def to_dict(self):
        return {
            "syntax_valid":   self.syntax_valid,
            "syntax_error":   self.syntax_error,
            "semantic_valid": self.semantic_valid,
            "semantic_error": self.semantic_error,
            "llm_analysis":   self.llm_analysis,
        }


class DESCodeValidator:
    """Wraps the LLM semantic judge."""

    @staticmethod
    def _parse_semantic_json(response: str) -> Tuple[Optional[bool], str]:
        """Extracts valid/reason from the judge response; None = could not be parsed."""
        if not response or not response.strip():
            return False, "Empty judge response"
        text = response.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None, text[:500]
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None, text[:500]
        valid  = data.get("valid")
        reason = str(data.get("reason", "")).strip()
        if isinstance(valid, str):
            valid = valid.lower() in ("true", "1", "yes")
        if valid is True:
            return True, reason
        if valid is False:
            return False, reason or "Judge marked invalid"
        return None, text[:500]

    @staticmethod
    def validate_semantics(
        prompt: str, code: str, llm_engine=None
    ) -> Tuple[bool, str, str]:
        """
        Verifica semanticamente via LLM judge.
        Retorna (is_valid, error_message, raw_llm_response_excerpt).
        """
        if not llm_engine:
            logger.warning("No LLM engine provided for semantic validation")
            return False, "Semantic judge not configured", ""

        validation_prompt = (
            f"User request (prompt):\n{prompt}\n\n"
            f"Generated DEVS code:\n{code}\n\n"
            "Does the code correctly implement what was requested "
            "(states, transitions, timing)?\n"
            "Answer using the required JSON schema only."
        )
        try:
            response = llm_engine(validation_prompt)
            parsed   = DESCodeValidator._parse_semantic_json(response)
            if parsed[0] is None:
                low = response.lower()
                if '"valid": true' in low or "'valid': true" in low:
                    return True, "", response[:2000]
                return False, parsed[1] or response[:500], response[:2000]
            ok, reason = parsed
            return ok, reason, response[:2000]
        except Exception as e:
            logger.error("Semantic validation failed: %s", str(e))
            return False, str(e), ""


def validate_generated_code(
    prompt: str, code: str, llm_engine=None
) -> ValidationResult:
    """Executa apenas o LLM judge (fase 2b isolada)."""
    result    = ValidationResult()
    validator = DESCodeValidator()

    sem_ok, sem_err, analysis = validator.validate_semantics(prompt, code, llm_engine)
    result.syntax_valid    = True   # only called after syntax has already passed
    result.semantic_valid  = sem_ok
    result.semantic_error  = sem_err
    result.llm_analysis    = analysis
    return result


# ── eval_pipeline (public contract used by eval.py) ──────────────────────────

def eval_pipeline(
    code: str,
    policy_file: str,
    original_prompt: str,
    semantic_judge_fn=None,
) -> dict:
    """
    Phase 1  → MS4 Me compiles the code locally.
    Phase 2a → automated matching against the canonical solution (all functions).
               If any matcher hits → success, no further review.
               Otherwise → logs mismatch and proceeds to Phase 2b.
    Phase 2b → LLM judge as a fallback metric.
    """
    # ── Phase 1 ──────────────────────────────────────────────────────────────
    syntax_ok, syntax_error = compile_with_ms4me(code)
    if not syntax_ok:
        return {
            "devs_plan_success":    False,
            "opa_evaluation_result": "Failure",
            "devs_plan_error":      syntax_error,
            "opa_evaluation_error": "",
            "notes":                "",
        }

    # ── Phase 2a ─────────────────────────────────────────────────────────────
    canonical   = policy_file or ""
    ctx         = original_prompt[:80]
    any_match, canon_results = canonical_match(code, canonical, context=ctx)

    if any_match:
        matched_by = next(k for k, v in canon_results.items() if v["match"])
        return {
            "devs_plan_success":    True,
            "opa_evaluation_result": "success",
            "devs_plan_error":      "",
            "opa_evaluation_error": "",
            "notes":                f"canonical match [{matched_by}]",
        }

    # ── Phase 2b ─────────────────────────────────────────────────────────────
    vr  = validate_generated_code(original_prompt, code, llm_engine=semantic_judge_fn)
    d   = vr.to_dict()
    sem_ok = d["semantic_valid"]

    matcher_summary = json.dumps({k: v["match"] for k, v in canon_results.items()})
    notes_parts = [f"canonical_results={matcher_summary}"]
    if d.get("llm_analysis"):
        notes_parts.append(d["llm_analysis"])
    if d.get("semantic_error"):
        notes_parts.append(d["semantic_error"])

    return {
        "devs_plan_success":    True,
        "opa_evaluation_result": "success" if sem_ok else "Failure",
        "devs_plan_error":      "",
        "opa_evaluation_error": d["semantic_error"],
        "notes":                " | ".join(notes_parts),
    }


def empty_code_error(notes: str = "") -> dict:
    """Same format as eval_pipeline, for rows where no code was extracted."""
    return {
        "devs_plan_success":    False,
        "opa_evaluation_result": "Failure",
        "devs_plan_error":      "Generated code is empty",
        "opa_evaluation_error": "",
        "notes":                notes,
    }
