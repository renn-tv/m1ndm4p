import json
import os
from typing import Optional

try:
    import requests
except ImportError:  # pragma: no cover - dependency advisory
    requests = None
    print("The 'requests' library is required for OpenRouter access. Please install it to enable AI features.")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "minimax/minimax-chat"  # Default free tier model


def _post_openrouter(model: str, messages: list[dict]) -> Optional[dict]:
    if requests is None:
        return None

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(body), timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        return response.json()
    except ValueError:
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
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    response = _post_openrouter(model, [{"role": "user", "content": prompt}])
    if not response:
        return None
    return _extract_text(response)


def generate_children(node_title: str, n: int, context_markdown: str | None = None) -> list[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or n <= 0:
        return [f"{node_title} idea {i + 1}" for i in range(n)]

    context = f"\n\nContext:\n{context_markdown}" if context_markdown else ""
    prompt = (
        "You are helping to brainstorm ideas for a mind map. "
        "Provide exactly {n} concise child node titles for the node titled '{title}'. "
        "Each title must be on its own line with no numbering or bullets.{context}"
    ).format(n=n, title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        return [f"{node_title} idea {i + 1}" for i in range(n)]

    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if len(lines) < n:
        lines.extend(f"{node_title} idea {i + 1}" for i in range(len(lines), n))
    return lines[:n]


def generate_children_auto(node_title: str, context_markdown: str | None = None) -> list[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return generate_children(node_title, 3, context_markdown=context_markdown)

    context = f"\n\nContext:\n{context_markdown}" if context_markdown else ""
    prompt = (
        "Based on the existing mind map, suggest a sensible set of child nodes for '{title}'. "
        "Return between 2 and 6 concise titles that add meaningful detail. "
        "List each title on its own line with no numbering, bullets, or commentary.{context}"
    ).format(title=node_title, context=context)

    result = _call_openrouter(prompt)
    if not result:
        return generate_children(node_title, 3, context_markdown=context_markdown)

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
