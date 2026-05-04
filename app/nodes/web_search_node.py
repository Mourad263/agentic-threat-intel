"""Web search node for live cybersecurity information retrieval."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Protocol
from urllib import error, request

from dotenv import load_dotenv

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step, summarize_web_results

WebResult = dict[str, str]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DOTENV_PATH = _PROJECT_ROOT / ".env"

_MAX_QUERY_RESULTS = 3
_MAX_SEARCH_QUERIES = 1

_INCIDENT_SIGNALS = {
    "incident",
    "incidents",
    "campaign",
    "campaigns",
    "victim",
    "victims",
    "organization",
    "organizations",
    "sector",
    "hospital",
    "healthcare",
    "school",
    "education",
    "university",
    "government",
    "municipal",
    "enterprise",
    "manufacturer",
    "breach",
    "extortion",
    "disruption",
    "disrupted",
    "outage",
    "claimed",
    "law enforcement",
    "advisory",
    "attack",
    "attacks",
    "operations",
    "operations disrupted",
    "data leak",
    "data theft",
    "ransom demand",
    "leak site",
    "ransomware attack",
    "ransomware attacks",
    "lockbit",
    "blackcat",
    "alphv",
    "clop",
    "akira",
    "play ransomware",
    "black basta",
}

_GENERIC_SIGNALS = {
    "what is",
    "cybersecurity-101",
    "cybersecurity 101",
    "guide",
    "glossary",
    "definition",
    "definitions",
    "explained",
    "examples explained",
    "what you need to know",
    "types of",
    "overview",
    "beginner",
    "basics",
    "prevention tips",
    "checklist",
    "how to prevent",
    "complete guide",
    "ultimate guide",
    "ransomware meaning",
    "ransomware explained",
    "ransomware protection",
    "ransomware prevention",
}

_LISTICLE_PATTERN = re.compile(
    r"\b(top\s+\d+|\d+\s+(examples|attacks|types|ways|trends|cyber-attacks|cyberattacks))\b",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b(2024|2025|2026)\b")


class WebSearchProvider(Protocol):
    """Protocol for pluggable web search providers."""

    def search(self, query: str, max_results: int = _MAX_QUERY_RESULTS) -> list[WebResult]:
        """Return normalized web results."""


class TavilyWebSearchProvider:
    """Small Tavily integration that can be swapped later."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = _MAX_QUERY_RESULTS) -> list[WebResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "topic": "news",
            "time_range": "year",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_usage": True,
            "exclude_domains": [
                "wikipedia.org",
                "techtarget.com",
                "cloudflare.com",
                "nordvpn.com",
                "kaspersky.com/resource-center",
                "crowdstrike.com/cybersecurity-101",
                "fortinet.com/resources/cyberglossary",
            ],
        }

        body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        return _normalize_tavily_results(data.get("results", []))


def _load_project_env() -> None:
    """Load the repo .env file if present."""
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _normalize_tavily_results(items: list[dict]) -> list[WebResult]:
    """Normalize Tavily results into the shared-state schema."""
    results: list[WebResult] = []

    for item in items:
        results.append(
            {
                "title": (item.get("title") or "Untitled result").strip(),
                "url": (item.get("url") or "").strip(),
                "content": (
                    item.get("content")
                    or item.get("raw_content")
                    or ""
                ).strip(),
            }
        )

    return results


def get_web_search_provider() -> WebSearchProvider | None:
    """Return the configured provider, or None if no provider is configured."""
    _load_project_env()

    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    key_exists = bool(tavily_api_key)

    debug_print(f"[web_search_node] TAVILY_API_KEY exists: {key_exists}")

    if tavily_api_key:
        provider = TavilyWebSearchProvider(api_key=tavily_api_key)
        debug_print("[web_search_node] Tavily provider initialized: True")
        return provider

    debug_print(f"[web_search_node] Tavily provider initialized: False (.env path: {_DOTENV_PATH})")
    return None


def search_web(query: str, max_results: int = _MAX_QUERY_RESULTS) -> list[WebResult]:
    """Search the web and return normalized results."""
    provider = get_web_search_provider()

    if provider is None:
        raise RuntimeError(
            "[web_search_node] Tavily search unavailable: TAVILY_API_KEY is missing or empty. "
            f"Set it in the environment or in {_DOTENV_PATH}."
        )

    try:
        return provider.search(query=query, max_results=max_results)

    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        debug_print(f"[web_search_node] Tavily HTTP error: {exc.code} {exc.reason}")
        debug_print(f"[web_search_node] Tavily response body: {response_body}")
        return []

    except error.URLError as exc:
        debug_print(f"[web_search_node] Web search request failed: {exc}")
        return []

    except Exception as exc:
        debug_print(f"[web_search_node] Unexpected web search error: {exc}")
        return []


def format_web_results(results: list[WebResult]) -> str:
    """Convert web search results into a readable context block."""
    if not results:
        return ""

    formatted_results = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled result")
        url = result.get("url", "")
        content = result.get("content", "").strip()

        formatted_results.append(
            f"[Web Result {index}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

    return "\n\n---\n\n".join(formatted_results)


def _normalize_text(value: str) -> str:
    """Collapse whitespace for deterministic scoring and previews."""
    return re.sub(r"\s+", " ", value or "").strip()


def _build_search_queries(user_query: str, topic: str) -> list[str]:
    """Build topic-aware incident-focused Tavily queries."""
    normalized_query = user_query.lower()
    normalized_topic = _normalize_text(topic).lower()

    wants_recent = any(
        term in normalized_query
        for term in (
            "recent",
            "latest",
            "current",
            "today",
            "this week",
            "examples",
            "incidents",
            "campaigns",
        )
    )

    if "ddos" in normalized_topic or "denial of service" in normalized_topic or normalized_topic == "dos":
        base_queries = [
            "recent DDoS attack service disruption botnet traffic flood 2025 2026",
            "DDoS campaign HTTP flood DNS amplification victims 2025 2026",
        ]

    elif "brute" in normalized_topic or "credential" in normalized_topic or "password" in normalized_topic:
        base_queries = [
            "recent brute force attack password spraying credential stuffing 2025 2026",
            "recent credential stuffing campaign authentication attacks victims 2025 2026",
        ]

    elif "trojan" in normalized_topic or "rat" in normalized_topic or "backdoor" in normalized_topic:
        base_queries = [
            "recent trojan malware campaign backdoor remote access 2025 2026",
            "trojan malware loader payload command and control campaign 2025 2026",
        ]

    else:
        base_queries = [
            'ransomware "ransomware attack" victim disruption 2025 2026',
            "ransomware campaign extortion victims 2025 2026",
        ]

    queries = base_queries if wants_recent else base_queries[:1]

    if "healthcare" in normalized_query or "hospital" in normalized_query:
        queries.append(f"{normalized_topic} healthcare hospital cyber attack disruption 2025 2026")

    if "education" in normalized_query or "school" in normalized_query or "university" in normalized_query:
        queries.append(f"{normalized_topic} university school cyber attack disruption 2025 2026")

    if "government" in normalized_query or "public sector" in normalized_query:
        queries.append(f"{normalized_topic} government public sector cyber attack disruption 2025 2026")

    deduped_queries: list[str] = []
    seen: set[str] = set()

    for query in queries:
        normalized = _normalize_text(query)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped_queries.append(normalized)

    return deduped_queries[:_MAX_SEARCH_QUERIES]


def _count_matches(text: str, signals: set[str]) -> int:
    """Count substring matches for a small signal set."""
    lowered = text.lower()
    return sum(1 for signal in signals if signal in lowered)


def _score_result(result: WebResult) -> float:
    """Score results toward incident-grade intelligence and away from generic explainers."""
    title = _normalize_text(result.get("title", ""))
    url = _normalize_text(result.get("url", ""))
    content = _normalize_text(result.get("content", ""))
    combined = f"{title} {url} {content}".lower()

    score = 0.0

    score += _count_matches(title, _INCIDENT_SIGNALS) * 2.5
    score += _count_matches(content, _INCIDENT_SIGNALS) * 1.2
    score += _count_matches(url, _INCIDENT_SIGNALS) * 1.5

    if _YEAR_PATTERN.search(title):
        score += 1.5

    if _YEAR_PATTERN.search(content):
        score += 1.0

    generic_hits = _count_matches(title, _GENERIC_SIGNALS) * 3
    generic_hits += _count_matches(url, _GENERIC_SIGNALS) * 2
    generic_hits += _count_matches(content, _GENERIC_SIGNALS)

    score -= generic_hits

    if _LISTICLE_PATTERN.search(title):
        score -= 8.0

    if _LISTICLE_PATTERN.search(content) and _count_matches(content, _INCIDENT_SIGNALS) < 3:
        score -= 3.0

    bad_markers = (
        "top 10",
        "top ten",
        "roundup",
        "opinion",
        "sponsored",
        "press release",
        "what is",
        "explained",
        "guide",
        "checklist",
        "glossary",
    )

    if any(marker in combined for marker in bad_markers):
        score -= 5.0

    if len(content) > 250:
        score += 0.5

    trusted_sources = (
        "bleepingcomputer.com",
        "therecord.media",
        "securityweek.com",
        "darkreading.com",
        "cisa.gov",
        "ic3.gov",
        "chainalysis.com",
        "unit42.paloaltonetworks.com",
        "mandiant.com",
        "secureworks.com",
        "rapid7.com",
        "sophos.com",
        "infosecurity-magazine.com",
        "govtech.com",
        "scworld.com",
        "csoonline.com",
    )

    if any(source in url.lower() for source in trusted_sources):
        score += 3.0

    return score


def _dedupe_results(results: list[WebResult]) -> list[WebResult]:
    """Remove duplicate results by URL, then by normalized title."""
    deduped: list[WebResult] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for result in results:
        url = _normalize_text(result.get("url", "")).lower()
        title = _normalize_text(result.get("title", "")).lower()

        dedupe_key = url or title

        if not dedupe_key:
            continue

        if url and url in seen_urls:
            continue

        if title and title in seen_titles:
            continue

        if url:
            seen_urls.add(url)

        if title:
            seen_titles.add(title)

        deduped.append(result)

    return deduped


def _select_best_results(results: list[WebResult], max_results: int = _MAX_QUERY_RESULTS) -> list[WebResult]:
    """Keep the strongest incident-grade results while preserving the shared schema."""
    ranked = sorted(
        (
            (result, _score_result(result))
            for result in _dedupe_results(results)
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    filtered = [result for result, score in ranked if score > 0]

    if filtered:
        return filtered[:max_results]

    return [result for result, _score in ranked[:max_results]]


def web_search_node(state: AppState) -> AppState:
    """Retrieve live web results and return the updated shared state."""
    user_query = state.get("user_query", "").strip()
    topic = state.get("topic", "").strip()

    web_queries = _build_search_queries(user_query=user_query, topic=topic)

    debug_print("\n[web_search_node] Debug")
    debug_print(f"User query: {user_query}")
    debug_print(f"Topic: {topic}")
    debug_print(f"[web_search_node] queries being sent: {web_queries}")
    debug_print(f"[web_search_node] incoming web_results: {len(state.get('web_results', []))}")

    if not user_query:
        debug_print("[web_search_node] Empty user query. Skipping web search.")
        state["web_results"] = []

        record_execution_step(
            state,
            "web_search",
            title="Web Search",
            summary="Skipped live web search because the user query was empty.",
            details={
                "queries": web_queries,
                "web_results_count": 0,
                "results": [],
            },
        )

        return state

    aggregated_results: list[WebResult] = []

    for query in web_queries:
        query_results = search_web(query=query, max_results=_MAX_QUERY_RESULTS)

        debug_print(f"[web_search_node] results for query '{query}': {len(query_results)}")

        for index, result in enumerate(query_results, start=1):
            debug_print(
                f"[web_search_node] raw result {index}: "
                f"{result.get('title', 'Untitled')} | {result.get('url', '')}"
            )

        aggregated_results.extend(query_results)

    results = _select_best_results(aggregated_results, max_results=_MAX_QUERY_RESULTS)

    debug_print(f"[web_search_node] number of aggregated results returned: {len(aggregated_results)}")
    debug_print(f"[web_search_node] FINAL SAVE: {len(results)}")

    state["web_results"] = results

    debug_print(f"[web_search_node] STATE AFTER SAVE: {len(state.get('web_results', []))}")

    if not results:
        debug_print("[web_search_node] No web results found.")

        record_execution_step(
            state,
            "web_search",
            title="Web Search",
            summary="Executed live web search but no usable results were retained.",
            details={
                "queries": web_queries,
                "web_results_count": 0,
                "results": [],
            },
        )

        return state

    first_result = results[0]

    debug_print(
        "[web_search_node] first result: "
        f"{first_result.get('title', 'Untitled result')} | {first_result.get('url', '')}"
    )

    debug_print(f"[web_search_node] Retrieved {len(results)} web results.")

    preview = format_web_results(results)[:400].replace("\n", " ")
    debug_print(f"[web_search_node] Preview: {preview}...")

    record_execution_step(
        state,
        "web_search",
        title="Web Search",
        summary=f"Executed {len(web_queries)} live search queries and kept {len(results)} ranked results.",
        details={
            "queries": web_queries,
            "web_results_count": len(results),
            "results": summarize_web_results(results),
        },
    )

    return state