import json
import os
import re
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Literal, Optional
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
_EMOJI_FALLBACKS = ["✨", "💡", "🧠", "🌱", "🚀", "📝", "🎯", "📌"]


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


def _dummy_child_title(level: int) -> str:
    key = (get_active_model(), _dummy_generation, level)
    _dummy_counters[key] += 1
    return f"Level {level}, Node {_dummy_counters[key]}"


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
        "X-App-Name": "m1ndm4p",
        "X-App-URL": "https://github.com/renn-tv/m1ndm4p",
        "X-App-Icon": "https://dentro.de/ai/images/DENTROID_Logo.png",
        "HTTP-Referer": "https://github.com/renn-tv/m1ndm4p",
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


_AUTO_CHILD_LIMIT = 10


def _structured_prompt(
    node_path: list[str],
    depth: int,
    *,
    include_body: bool,
    mode: Literal["auto", "exact", "body"],
    exact_children: int | None,
    context_markdown: str | None,
    level_targets: list[int] | None = None,
) -> str:
    normalized_context = _normalize_context_text(context_markdown)
    context_block = f"\n\nKnown outline:\n{normalized_context}" if normalized_context else ""
    root_title = node_path[-1]
    breadcrumb = " > ".join(node_path)
    current_level = max(len(node_path), 1)
    max_heading_level = current_level + max(depth, 0)
    heading_instruction = (
        f"Begin with the heading level {current_level} (`{'#' * current_level}`) for '{root_title}'. "
        f"Any children must use heading levels {current_level + 1} through {max_heading_level}."
    )
    targets = list(level_targets) if level_targets else []
    if depth <= 0:
        child_instruction = "Do not add any child headings; focus only on the current node."
    elif targets:
        target = targets[0]
        child_instruction = (
            f"Add exactly {target} direct child headings (level {current_level + 1}) under '{root_title}'. "
            "Merge or split ideas so that each heading is meaningful, but do not add more or fewer entries."
        )
    elif mode == "exact" and exact_children is not None:
        child_instruction = (
            f"Add exactly {exact_children} direct child headings (level {current_level + 1}) under '{root_title}'. "
            "Skip numbering or bullets. Only add grandchildren if depth allows and they clarify those children."
        )
    else:
        child_instruction = (
            f"Add between 0 and {_AUTO_CHILD_LIMIT} concise child headings (level {current_level + 1}). "
            "Plan the entire subtree before you write: decide which ideas deserve deeper detail all the way to the requested depth, then emit everything in one response. "
            "Skip the level entirely if nothing fresh is worth saying, and keep every label under 40 characters—think 2–4 word Pareto summaries, never sentences. "
            "Only branch beyond the first level when doing so clearly improves understanding, and never exceed the requested depth."
        )

    if include_body:
        body_instruction = (
            "Only add sentences to leaf headings (those you do not expand further). For each leaf, add exactly one plain declarative sentence "
            "(≤160 characters) immediately below it, separated by a blank line. Never place a sentence under a heading that still receives child headings. "
            "Keep sentences factual—no hype, lists, or markdown embellishments."
        )
    else:
        body_instruction = "Do not include any narrative paragraphs or bullet lists—headings only."

    detail_instructions: list[str] = []
    if targets:
        for idx, target in enumerate(targets[1:], start=1):
            parent_level = current_level + idx
            child_level = parent_level + 1
            detail_instructions.append(
                f"Each level {parent_level} heading must include exactly {target} level {child_level} subheadings; "
                "use concise labels and only skip a slot if absolutely no supporting ideas exist."
            )
        if len(targets) < max(depth, 0):
            for offset in range(len(targets) + 1, max(depth, 0) + 1):
                parent_level = current_level + offset
                child_level = parent_level + 1
                detail_instructions.append(
                    f"Continue deepening the outline to level {child_level}; prefer completing depth {depth} wherever meaningful."
                )
    else:
        for offset in range(1, max(depth, 0)):
            parent_level = current_level + offset
            child_level = parent_level + 1
            detail_instructions.append(
                f"Before writing level {parent_level} headings, consider which need level {child_level} detail so the final outline reaches the intended depth; "
                f"whenever a concrete fact exists, add the level {child_level} child to reveal it. If absolutely nothing specific exists, leave the parent as a leaf, but prefer completing full depth when possible."
            )
    detail_instruction = "\n".join(detail_instructions)

    target_instruction = ""
    if targets:
        lines = []
        for idx, count in enumerate(targets, start=1):
            level_no = current_level + idx
            lines.append(f"- Level {level_no}: exactly {count} headings.")
        target_instruction = "Target counts per depth:\n" + "\n".join(lines)

    rules = (
        "Rules: output only Markdown headings (with optional single sentences as described), "
        "never emit numbered lists, bullets, code fences, or commentary. "
        f"Stop once you reach heading level {max_heading_level}. "
        "Never recreate ancestors or siblings outside the provided path."
    )

    return (
        "You are expanding a structured Markdown outline based on the existing content.\n"
        f"Current section path: {breadcrumb}\n"
        f"{heading_instruction}\n"
        f"{child_instruction}\n"
        f"{detail_instruction}\n"
        f"{target_instruction}\n"
        f"{body_instruction}\n"
        f"{rules}"
        f"{context_block}"
    )


def _dummy_subtree_markdown(
    node_path: list[str],
    depth: int,
    *,
    include_body: bool,
    exact_children: int | None,
    level_targets: list[int] | None = None,
) -> str:
    current_level = max(len(node_path), 1)
    root_title = node_path[-1]
    lines: list[str] = [f"{'#' * current_level} {root_title}"]
    if include_body:
        lines.extend(
            [
                "",
                f"Placeholder detail for {root_title}.",
            ]
        )

    def build(level: int, remaining_depth: int, *, first_level: bool, target_index: int) -> None:
        if remaining_depth <= 0:
            return
        if level_targets and target_index < len(level_targets):
            child_count = max(level_targets[target_index], 0)
        elif first_level and exact_children is not None:
            child_count = max(exact_children, 0)
        else:
            child_count = 3
        for _ in range(child_count):
            title = _dummy_child_title(level + 1)
            lines.append("")
            lines.append(f"{'#' * (level + 1)} {title}")
            if include_body:
                lines.append("")
                lines.append(f"{title} details pending.")
            build(level + 1, remaining_depth - 1, first_level=False, target_index=target_index + 1)

    build(current_level, depth, first_level=True, target_index=0)
    text = "\n".join(lines).strip()
    return f"{text}\n" if text else ""


def generate_structured_subtree(
    node_path: list[str],
    depth: int,
    context_markdown: str | None = None,
    *,
    include_body: bool = False,
    mode: Literal["auto", "exact", "body"] = "auto",
    exact_children: int | None = None,
    level_targets: list[int] | None = None,
) -> Optional[str]:
    if not node_path:
        raise ValueError("node_path must contain at least one title")
    if depth < 0:
        depth = 0
    depth = min(depth, 5)
    if mode == "exact" and (exact_children is None or depth == 0):
        raise ValueError("exact mode requires depth >= 1 and exact_children to be set")
    if mode == "body":
        depth = 0
        include_body = True

    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = get_active_model()
    uses_network = bool(api_key) and active_model != OFFLINE_MODEL

    prompt = _structured_prompt(
        node_path,
        depth,
        include_body=include_body,
        mode=mode,
        exact_children=exact_children,
        context_markdown=context_markdown,
        level_targets=level_targets,
    )
    if not uses_network:
        return _dummy_subtree_markdown(
            node_path,
            depth,
            include_body=include_body,
            exact_children=exact_children,
            level_targets=level_targets,
        )

    result = _call_openrouter(prompt)
    if result:
        return result.strip()
    return _dummy_subtree_markdown(
        node_path,
        depth,
        include_body=include_body,
        exact_children=exact_children,
    )


def generate_leaf_bodies(
    entries: list[dict[str, object]],
    context_markdown: str | None = None,
) -> list[str]:
    if not entries:
        return []

    normalized_context = _normalize_context_text(context_markdown)
    context_block = f"\n\nExisting mind map:\n{normalized_context}" if normalized_context else ""
    entry_lines: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        path = entry.get("path", [])
        if isinstance(path, list):
            breadcrumb = " > ".join(str(part) for part in path)
        else:
            breadcrumb = str(path)
        title = str(entry.get("title", "") or "").strip() or breadcrumb.split(" > ")[-1]
        level = entry.get("level")
        entry_lines.append(f"{idx}. Path: {breadcrumb}")
        entry_lines.append(f"   Title: {title}")
        if isinstance(level, int):
            entry_lines.append(f"   Heading level: {level}")
    entries_block = "\n".join(entry_lines)

    prompt = (
        "You are adding crisp factual sentences to existing leaf nodes in a mind map.\n"
        "For each entry listed below, write exactly one plain, declarative sentence (≤160 characters) that states the key fact for the final node in the path.\n"
        "Use knowledge grounded in the supplied context or well-established domain facts; never invent colorful language, metaphors, or marketing copy.\n"
        "Keep wording precise, avoid repeated phrases across entries, and never mention numbering or these instructions.\n"
        "Respond strictly as JSON: a list of objects with fields \"index\" (matching the entry number) and \"body\" (the sentence string). "
        "Example: [{\"index\":1,\"body\":\"<sentence>\"}]."
        f"\n\nEntries:\n{entries_block}"
        f"{context_block}"
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = get_active_model()
    uses_network = bool(api_key) and active_model != OFFLINE_MODEL
    if not uses_network:
        return [f"A brief note about {entry.get('title', 'this node')}." for entry in entries]

    result = _call_openrouter(prompt)
    if not result:
        return [f"A brief note about {entry.get('title', 'this node')}." for entry in entries]

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return [f"A brief note about {entry.get('title', 'this node')}." for entry in entries]

    responses: list[str] = ["" for _ in entries]
    records = parsed if isinstance(parsed, list) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        index = record.get("index")
        if not isinstance(index, int):
            continue
        if not 1 <= index <= len(entries):
            continue
        body = record.get("body")
        if not isinstance(body, str):
            continue
        cleaned = " ".join(body.split()).strip()
        if not cleaned:
            continue
        if len(cleaned) > 200:
            cleaned = cleaned[:200].rstrip()
        responses[index - 1] = cleaned

    fallback_template = "A brief note about {title}."
    for idx, text in enumerate(responses):
        if text:
            continue
        title = str(entries[idx].get("title", "this node"))
        responses[idx] = fallback_template.format(title=title)
    return responses


def _offline_emoji(text: str) -> str:
    if not text:
        return ""
    score = sum(ord(char) for char in text)
    return _EMOJI_FALLBACKS[score % len(_EMOJI_FALLBACKS)]


def suggest_emoji(text: str, path_hint: str | None = None) -> str:
    snippet = (text or "").strip()
    if not snippet:
        return ""
    path = (path_hint or "").strip()
    context_line = f"\nMind map path: {path}" if path else ""
    prompt = (
        "Choose a single emoji that matches the theme of this mind map heading or bullet.\n"
        "Respond with the emoji only. If nothing fits, respond with NONE.\n"
        f"Entry text: {snippet}{context_line}"
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    active_model = get_active_model()
    uses_network = bool(api_key) and active_model != OFFLINE_MODEL
    if not uses_network:
        return _offline_emoji(snippet)
    result = _call_openrouter(prompt)
    if not result:
        return ""
    emoji = result.strip()
    if not emoji or emoji.upper().startswith("NONE"):
        return ""
    token = emoji.split()[0]
    return token
