import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Optional
from urllib import error, request

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OFFLINE_MODEL = "No AI (offline dummy output)"
_NETWORK_MODELS = [
    "openrouter/polaris-alpha",
    "alibaba/tongyi-deepresearch-30b-a3b:free",
    "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-scout:free",
    "microsoft/mai-ds-r1:free",
    "minimax/minimax-m2:free",
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "moonshotai/kimi-k2:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-coder:free",
    "tngtech/deepseek-r1t2-chimera:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "z-ai/glm-4.5-air:free",
]
AVAILABLE_MODELS = [OFFLINE_MODEL, *_NETWORK_MODELS]
DEFAULT_MODEL = _NETWORK_MODELS[0]
_dummy_generation = 0
_dummy_counters: DefaultDict[tuple[str, int, int], int] = defaultdict(int)
_PROMPT_LOG_PATH = Path("prompt.log")
_prompt_log_lock = threading.Lock()
_force_new_prompt_task = True
_current_task_prompt_count = 0
_next_task_label: Optional[str] = None
_CONNECTION_LOG_PATH = Path("connection.log")
_connection_log_lock = threading.Lock()


def _reset_prompt_state_locked() -> None:
    global _force_new_prompt_task, _current_task_prompt_count, _next_task_label
    _force_new_prompt_task = True
    _current_task_prompt_count = 0
    _next_task_label = None


def reset_prompt_log() -> None:
    with _prompt_log_lock:
        _PROMPT_LOG_PATH.write_text("", encoding="utf-8")
        _reset_prompt_state_locked()


def reset_connection_log() -> None:
    with _connection_log_lock:
        _CONNECTION_LOG_PATH.write_text("", encoding="utf-8")


def start_prompt_session(label: str | None = None) -> None:
    global _force_new_prompt_task, _current_task_prompt_count, _next_task_label
    with _prompt_log_lock:
        _force_new_prompt_task = True
        _current_task_prompt_count = 0
        _next_task_label = label


def finish_prompt_session() -> None:
    global _force_new_prompt_task, _current_task_prompt_count, _next_task_label
    with _prompt_log_lock:
        _reset_prompt_state_locked()


def _ensure_prompt_header_locked() -> None:
    global _force_new_prompt_task, _current_task_prompt_count, _next_task_label
    if not _force_new_prompt_task:
        return
    size = _PROMPT_LOG_PATH.stat().st_size if _PROMPT_LOG_PATH.exists() else 0
    with _PROMPT_LOG_PATH.open("a", encoding="utf-8") as log:
        if size:
            log.write("=====\n")
    _force_new_prompt_task = False
    _next_task_label = None
    _current_task_prompt_count = 0


def _log_prompt_exchange(prompt: str, response_raw: str | None, error: str | None) -> None:
    prompt_text = prompt.strip()
    if not prompt_text:
        prompt_text = "<empty prompt>"
    response_text = (response_raw or "").strip()
    global _current_task_prompt_count
    prompt_timestamp = datetime.now().isoformat(timespec="seconds")
    with _prompt_log_lock:
        _ensure_prompt_header_locked()
        with _PROMPT_LOG_PATH.open("a", encoding="utf-8") as log:
            if _current_task_prompt_count:
                log.write("----\n")
            entry_no = _current_task_prompt_count + 1
            log.write(f"m1ndm4p [{prompt_timestamp}] Prompt {entry_no}:\n")
            log.write("------------------------------------------------------------\n")
            log.write(f"{prompt_text}\n")
            log.write("============================================================\n")
            response_timestamp = datetime.now().isoformat(timespec="seconds")
            log.write(f"{get_active_model()} [{response_timestamp}] Response {entry_no}:\n")
            log.write("------------------------------------------------------------\n")
            if error:
                log.write(f"<error> {error}\n")
            elif response_text:
                log.write(f"{response_text}\n")
            else:
                log.write("<empty>\n")
            log.write("============================================================\n")
        _current_task_prompt_count += 1


def _log_connection_event(status: str, model: str, detail: str | None = None) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    message = detail.strip() if detail else ""
    line = f"{timestamp}\t{status.upper()}\t{model}"
    if message:
        line = f"{line}\t{message}"
    with _connection_log_lock:
        with _CONNECTION_LOG_PATH.open("a", encoding="utf-8") as log:
            log.write(line + "\n")


def get_active_model() -> str:
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def set_active_model(model: str) -> None:
    os.environ["OPENROUTER_MODEL"] = model
    reset_dummy_counters()


def reset_dummy_counters() -> None:
    global _dummy_generation
    _dummy_generation += 1
    _dummy_counters.clear()


def _infer_level_from_context(node_title: str, context_markdown: str | None) -> int:
    if not context_markdown:
        return 1
    heading_pattern = re.compile(r"^(#+)\s+(.*)\s*$")
    for line in context_markdown.splitlines():
        match = heading_pattern.match(line)
        if not match:
            continue
        hashes, title = match.groups()
        if title.strip() == node_title.strip():
            return max(len(hashes), 1)
    return 1


def _dummy_child_title(level: int) -> str:
    key = (get_active_model(), _dummy_generation, level)
    _dummy_counters[key] += 1
    return f"Level {level}, Node {_dummy_counters[key]}"


def _sync_dummy_counter(level: int, context_markdown: str | None) -> None:
    key = (get_active_model(), _dummy_generation, level)
    baseline = 0
    if context_markdown:
        pattern = re.compile(rf"Level {level}, Node (\d+)")
        matches = [int(match) for match in pattern.findall(context_markdown)]
        if matches:
            baseline = max(matches)
    # Only update if the stored counter is lower than the baseline
    if _dummy_counters[key] < baseline:
        _dummy_counters[key] = baseline


def _normalize_context_text(context_markdown: str | None) -> Optional[str]:
    if not context_markdown:
        return None
    lines: list[str] = []
    previous_blank = False
    for raw_line in context_markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            lines.append("")
        else:
            previous_blank = False
            lines.append(stripped)
    normalized = "\n".join(lines).strip()
    return normalized or None


def _post_openrouter(
    model: str, messages: list[dict]
) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    if model == OFFLINE_MODEL:
        return None, "offline model", None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None, "missing api key", None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "m1ndm4p",
    }
    body = {
        "model": model,
        "messages": messages,
    }

    data = json.dumps(body).encode("utf-8")
    http_request = request.Request(
        OPENROUTER_API_URL,
        data=data,
        headers=headers,
        method="POST",
    )
    raw_payload: Optional[str] = None
    try:
        with request.urlopen(http_request, timeout=30) as response:
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                return None, f"HTTP {status}", None
            raw_payload = response.read().decode("utf-8")
    except (error.URLError, error.HTTPError, TimeoutError) as exc:
        return None, str(exc), raw_payload

    try:
        parsed = json.loads(raw_payload or "")
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}", raw_payload
    return parsed, None, raw_payload


def _extract_text(response: dict) -> Optional[str]:
    try:
        choices = response["choices"]
        if not choices:
            return None
        message = choices[0]["message"]
        return message.get("content")
    except (KeyError, TypeError):
        return None


def _call_openrouter(prompt: str) -> Optional[str]:
    model = get_active_model()
    response_payload, error, raw_payload = _post_openrouter(
        model, [{"role": "user", "content": prompt}]
    )
    if error:
        _log_connection_event("FAIL", model, error)
    else:
        _log_connection_event("SUCCESS", model)
    raw_for_log = raw_payload
    if raw_for_log is None and response_payload is not None:
        raw_for_log = json.dumps(response_payload, ensure_ascii=False)
    completion_text = _extract_text(response_payload) if response_payload else None
    _log_prompt_exchange(prompt, raw_for_log, error)
    if not response_payload:
        return None
    return completion_text


def _reset_level_counter(level: int) -> None:
    key = (get_active_model(), _dummy_generation, level)
    _dummy_counters[key] = 0


def generate_children(
    node_title: str,
    n: int,
    context_markdown: str | None = None,
    *,
    reset_counter: bool = False,
) -> list[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = get_active_model()
    uses_network = bool(api_key) and active_model != OFFLINE_MODEL
    level = _infer_level_from_context(node_title, context_markdown)
    if not uses_network or n <= 0:
        if reset_counter:
            _reset_level_counter(level)
        else:
            _sync_dummy_counter(level, context_markdown)
        return [_dummy_child_title(level) for _ in range(n)]

    context_text = _normalize_context_text(context_markdown)
    context = f"\n\nContext:\n{context_text}" if context_text else ""
    prompt = (
        "You are helping to brainstorm ideas for a mind map. "
        "Provide exactly {n} concise child node titles for the node titled '{title}'. "
        "Each title should feel natural but stay within roughly 50 characters—use a short phrase or even a single word if that best captures the idea. "
        "When a concrete supporting detail (a statistic, outcome, or specific example) would clarify the idea, emit that detail as the node instead of another abstract heading. "
        "List each title on its own line with no numbering or bullets.{context}"
    ).format(n=n, title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        if reset_counter:
            _reset_level_counter(level)
        else:
            _sync_dummy_counter(level, context_markdown)
        return [_dummy_child_title(level) for _ in range(n)]

    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if len(lines) < n:
        if reset_counter:
            _reset_level_counter(level)
        else:
            _sync_dummy_counter(level, context_markdown)
        lines.extend(_dummy_child_title(level) for _ in range(len(lines), n))
    return lines[:n]


def generate_children_auto(node_title: str, context_markdown: str | None = None) -> list[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = get_active_model()
    uses_network = bool(api_key) and active_model != OFFLINE_MODEL
    if not uses_network:
        return generate_children(
            node_title,
            3,
            context_markdown=context_markdown,
            reset_counter=True,
        )

    context_text = _normalize_context_text(context_markdown)
    context = f"\n\nContext:\n{context_text}" if context_text else ""
    prompt = (
        "Act as a seasoned lecturer distilling the key takeaway for '{title}'. "
        "Review the parent and its current children, then add anywhere from 0 to 10 new branches—but treat more than three as costly; exceed three only when an extra branch delivers indispensable new insight. "
        "Skip anything redundant or slogan-like. Each heading must stand alone and encode the concrete answer (e.g., '1992 Market Share Surge', 'Metzeler R&D Partner', 'ISO 9001 Certification'); never pose a question. "
        "Prefer ultra-short labels (single vivid words or ≤25 characters) and trust the human to branch further for detail. If no meaningful gaps remain, return nothing. "
        "List each title on its own line with no numbering, bullets, or filler.{context}"
    ).format(title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        return generate_children(
            node_title,
            3,
            context_markdown=context_markdown,
            reset_counter=True,
        )

    lines = [line.strip() for line in result.splitlines() if line.strip()]
    filtered = [line for line in lines if line.lower() != "none"]
    if not filtered:
        return generate_children(node_title, 3, context_markdown=context_markdown)
    return filtered[:6]


def generate_paragraph(node_title: str, context_markdown: str | None = None) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return f"A brief paragraph about {node_title}."

    context_text = _normalize_context_text(context_markdown)
    context = f"\n\nContext:\n{context_text}" if context_text else ""
    prompt = (
        "Write one informative sentence for the mind map node titled '{title}'. "
        "Default to a brief result/event that feels natural in a mind map, but you may use up to 200 characters when a bit more detail adds clear value. "
        "Do not exceed 200 characters, and avoid headings or lists.{context}"
    ).format(title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        return f"A brief paragraph about {node_title}."

    paragraph = " ".join(result.strip().split())
    if len(paragraph) > 200:
        paragraph = paragraph[:200].rstrip()
    return paragraph if paragraph else f"A brief paragraph about {node_title}."
