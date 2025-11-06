import json
import os
import re
from collections import defaultdict
from typing import DefaultDict, Optional
from urllib import error, request

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OFFLINE_MODEL = "No AI (offline dummy output)"
_NETWORK_MODELS = [
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


def _post_openrouter(model: str, messages: list[dict]) -> Optional[dict]:
    if model == OFFLINE_MODEL:
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

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
    try:
        with request.urlopen(http_request, timeout=30) as response:
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                return None
            payload = response.read().decode("utf-8")
    except (error.URLError, error.HTTPError, TimeoutError):
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


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
    response = _post_openrouter(model, [{"role": "user", "content": prompt}])
    if not response:
        return None
    return _extract_text(response)


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

    context = f"\n\nContext:\n{context_markdown}" if context_markdown else ""
    prompt = (
        "You are helping to brainstorm ideas for a mind map. "
        "Provide exactly {n} concise child node titles for the node titled '{title}'. "
        "Each title must be on its own line with no numbering or bullets.{context}"
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

    context = f"\n\nContext:\n{context_markdown}" if context_markdown else ""
    prompt = (
        "Based on the existing mind map, suggest a sensible set of child nodes for '{title}'. "
        "Return between 2 and 6 concise titles that add meaningful detail. "
        "List each title on its own line with no numbering, bullets, or commentary.{context}"
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

    context = f"\n\nContext:\n{context_markdown}" if context_markdown else ""
    prompt = (
        "Write a single concise paragraph (~200 characters) for the mind map node titled '{title}'. "
        "Avoid headings or lists and keep the tone informative.{context}"
    ).format(title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        return f"A brief paragraph about {node_title}."

    paragraph = " ".join(result.strip().split())
    return paragraph if paragraph else f"A brief paragraph about {node_title}."
