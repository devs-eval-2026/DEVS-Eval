"""
Extracts the «code» portion from model output (fenced blocks) with fallbacks
for poorly formatted output (CodeLlama: ```python, plain prose, or raw DNL/SES lines).
"""

import re
from typing import List, Optional, Tuple

# Order: specific delimiters first; generic ``` at the end (as a final "else" case).
# Includes common variants (python/hcl/text) that CodeLlama emits instead of ```ses/```DNL.
DEFAULT_DELIMITERS: List[str] = [
    "```ses",
    "```json",
    "```SES",
    "```DNL",
    "```dnl",
    "```text",
    "```plaintext",
    "```xml",
]

# Lines that look like canonical DNL/SES (one line per transition, etc.)
_DEVS_LINE = re.compile(
    r"(?i)(^|\b)(passivate|hold(\s+in)?|to\s+start|from\s+\w+\s+go\s+to|"
    r"output\s+to|when\s+in|after\s+)"
)


def _fenced_code_fallback(text: str) -> Optional[Tuple[str, str]]:
    """First ```[lang] ... ``` block found, for any fence language."""
    m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", text, re.DOTALL)
    if not m:
        return None
    c = m.group(1).strip()
    if not c:
        return None
    answer = text[: m.start()].strip()
    return answer, c


def _devs_lines_fallback(text: str) -> Optional[Tuple[str, str]]:
    """No fences: collects lines containing DEVS keywords (very 'chatty' models)."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    picked = [ln.strip() for ln in lines if ln.strip() and _DEVS_LINE.search(ln)]
    if not picked:
        return None
    return text.strip(), "\n".join(picked)


def separate_answer_and_code(
    text, delimiters: Optional[List[str]] = None
) -> Tuple[str, str]:
    if text is None:
        return "", ""
    text = str(text)
    dlist = list(delimiters) if delimiters is not None else list(DEFAULT_DELIMITERS)

    answer, code = text.strip(), ""
    for delimiter in dlist:
        parts = text.split(delimiter)
        if len(parts) < 2:
            continue
        answer = parts[0].strip()
        code = parts[1].strip()
        code = code.rsplit("```", 1)[0].strip()
        if code:
            return answer, code

    fb = _fenced_code_fallback(text)
    if fb:
        return fb

    fb2 = _devs_lines_fallback(text)
    if fb2:
        return fb2

    return answer, code


# --- Messages for when the backend returns empty text (eval / eval_open; not Replicate-specific)
REPLICATE_EVAL_MODELS = frozenset(
    {
        "codellama-7b",
        "codellama-34b",
        "Wizardcoder33b",
    }
)


def empty_response_log_note(model: str) -> Tuple[str, str]:
    """
    Returns (diagnostic log line, note string for the LLM Notes column).
    """
    if model in REPLICATE_EVAL_MODELS:
        return (
            "Replicate: API key, 429 rate limit, insufficient credits, model version issue, or exception (see above, models._replicate_run_stream).",
            "Empty response; see Replicate / models._replicate_run_stream logs above",
        )
    if model in ("gpt-5.4", "gpt-5.3-codex"):
        return (
            "OpenAI: exception above (models.GPT5_4 / GPT5_3_Codex) or null response. Causes: API key, quota, network, null content, context limit (prompt too long with FSP).",
            "Empty response; see OpenAI errors in log (models.GPT5_4 or GPT5_3_Codex)",
        )
    if model == "gemini-2.5-flash":
        return (
            "Google Gemini: GOOGLE_API_KEY missing, quota exceeded, or null response (see above).",
            "Empty response; see Gemini / GOOGLE_API_KEY logs above",
        )
    return (
        "See backend exceptions in the console output above.",
        "Empty model response; see console logs above",
    )
