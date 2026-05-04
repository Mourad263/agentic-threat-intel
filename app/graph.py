"""LangGraph workflow wiring for the threat intelligence pipeline."""

from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from app.nodes.critic_node import critic_node
from app.nodes.planner_node import planner_node
from app.nodes.reviser_node import reviser_node
from app.nodes.retriever_node import retriever_node
from app.nodes.validator_node import validator_node
from app.nodes.web_search_node import web_search_node
from app.nodes.writer_node import writer_node
from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step

REQUIRED_SECTIONS = [
    "Overview",
    "Attack Explanation",
    "Recent Examples",
    "IOCs",
    "Detection",
    "Mitigation",
    "Limitations",
]

PLACEHOLDER_PATTERNS = [
    r"\[Web Result",
    r"<[^>]+>",
    r"\bTBD\b",
    r"\bplaceholder\b",
    r"\blorem ipsum\b",
]

SECTION_HEADER_PATTERN = re.compile(
    r"(?mi)^(Overview|Attack Explanation|Recent Examples|IOCs|Detection|Mitigation|Limitations)\s*:?\s*$"
)


def _topic_key(topic: str = "") -> str:
    """Normalize topic into a small internal key."""
    normalized = (topic or "").lower().strip()

    if "ddos" in normalized or "denial of service" in normalized or normalized == "dos":
        return "ddos"

    if "brute" in normalized or "credential" in normalized or "password" in normalized:
        return "brute_force"

    if "trojan" in normalized or "rat" in normalized or "backdoor" in normalized:
        return "trojan"

    return "ransomware"


def _decide_route(state: AppState) -> str:
    """Route questions to local RAG, web search, or both."""
    user_query = state.get("user_query", "").lower()
    plan = state.get("plan", "").lower()
    combined_text = f"{user_query} {plan}"

    web_keywords = {
        "recent",
        "current",
        "latest",
        "news",
        "cve",
        "breach",
        "campaign",
        "today",
        "this week",
        "active",
        "ongoing",
        "examples",
        "incidents",
    }

    rag_keywords = {
        "what is",
        "how does",
        "explain",
        "overview",
        "difference",
        "concept",
        "architecture",
        "workflow",
        "definition",
    }

    wants_web = any(keyword in combined_text for keyword in web_keywords)
    wants_rag = any(keyword in combined_text for keyword in rag_keywords)

    if wants_web and wants_rag:
        return "both"

    if wants_web:
        return "web"

    return "rag"


def _extract_section_headers(text: str) -> list[str]:
    """Return all recognized section headers in order."""
    if not text:
        return []

    return [match.group(1) for match in SECTION_HEADER_PATTERN.finditer(text)]


def _contains_all_required_sections(text: str) -> bool:
    """Check whether all required sections are present at least once."""
    headers = set(_extract_section_headers(text))
    return all(section in headers for section in REQUIRED_SECTIONS)


def _has_duplicate_headers(text: str) -> bool:
    """Check whether any required section header appears more than once."""
    headers = _extract_section_headers(text)
    seen: set[str] = set()

    for header in headers:
        if header in seen:
            return True

        seen.add(header)

    return False


def _contains_placeholder_text(text: str) -> bool:
    """Check for obvious placeholder or leaked internal reference text."""
    if not text:
        return True

    lowered = text.lower()

    if "final revised answer" in lowered:
        return True

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PLACEHOLDER_PATTERNS)


def _is_trivially_generic(text: str) -> bool:
    """Reject very weak drafts that are too short or too generic."""
    if not text:
        return True

    stripped = text.strip()

    if len(stripped) < 700:
        return True

    weak_phrases = [
        "ransomware is a type of malicious software",
        "monitor for suspicious activity",
        "use antivirus software",
        "keep software up to date",
    ]

    weak_hits = sum(1 for phrase in weak_phrases if phrase in stripped.lower())

    return weak_hits >= 3


def _extract_section(text: str, section_name: str) -> str:
    """Extract one section body by name."""
    match = re.search(
        rf"(?mis)^{re.escape(section_name)}\s*:?\s*(.*?)(?=^Overview\s*:?\s*$|^Attack Explanation\s*:?\s*$|^Recent Examples\s*:?\s*$|^IOCs\s*:?\s*$|^Detection\s*:?\s*$|^Mitigation\s*:?\s*$|^Limitations\s*:?\s*$|\Z)",
        text or "",
    )

    if not match:
        return ""

    return match.group(1).strip()


def _extract_recent_examples_section(text: str) -> str:
    """Extract the Recent Examples section body."""
    return _extract_section(text, "Recent Examples")


def _recent_examples_are_weak(text: str, topic: str = "") -> bool:
    """Return True only when Recent Examples are clearly weak."""
    recent_examples = _extract_recent_examples_section(text)

    if not recent_examples:
        return True

    normalized = re.sub(r"\s+", " ", recent_examples).strip().lower()
    key = _topic_key(topic)

    if len(normalized.split()) < 35:
        return True

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", recent_examples))

    if bullet_count < 1:
        return True

    hard_bad_signals = [
        "opinion ###",
        "home »",
        "read more",
        "subscribe",
        "newsletter",
        "advertisement",
        "sponsored",
        "privacy policy",
        "cookie",
        "glossary",
        "definition",
        "colonial pipeline",
        "jbs foods",
        "hollywood presbyterian",
        "baltimore city",
    ]

    if any(signal in normalized for signal in hard_bad_signals):
        return True

    title_dump_signals = [
        "| cso online",
        "top 10 cyber",
        "record number of ransomware victims and groups in 2025",
    ]

    if any(signal in normalized for signal in title_dump_signals):
        return True

    wrong_topic_markers = {
        "ddos": [
            "ransomware pressure",
            "ransomware operators",
            "ransomware activity",
            "ransomware deployment",
            "encryption event",
            "ransom demand",
            "data theft, not encryption",
            "shadow copy",
            "ransom-note",
        ],
        "brute_force": [
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
        ],
        "trojan": [
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
        ],
        "ransomware": [],
    }

    if any(marker in normalized for marker in wrong_topic_markers.get(key, [])):
        return True

    analyst_markers = [
        "this matters",
        "which shows",
        "which suggests",
        "highlighting",
        "reinforcing",
        "indicating",
        "demonstrating",
        "because",
    ]

    if not any(marker in normalized for marker in analyst_markers):
        return True

    operational_terms_by_topic = {
        "ddos": [
            "ddos",
            "denial of service",
            "traffic",
            "flood",
            "botnet",
            "requests",
            "packets",
            "bandwidth",
            "amplification",
            "outage",
            "disrupt",
            "latency",
            "service",
            "availability",
        ],
        "brute_force": [
            "brute force",
            "password",
            "password spray",
            "password spraying",
            "credential stuffing",
            "login",
            "authentication",
            "mfa",
            "failed",
            "account",
            "identity",
        ],
        "trojan": [
            "trojan",
            "malware",
            "backdoor",
            "remote access",
            "rat",
            "loader",
            "payload",
            "command-and-control",
            "c2",
            "persistence",
            "credential",
        ],
        "ransomware": [
            "extortion",
            "data theft",
            "disruption",
            "ransomware",
            "victim",
            "organization",
            "attack",
            "detection",
            "containment",
            "recovery",
            "demand",
            "leverage",
        ],
    }

    topic_terms = operational_terms_by_topic.get(key, operational_terms_by_topic["ransomware"])

    if sum(term in normalized for term in topic_terms) < 2:
        return True

    return False


def _section_has_enough_bullets(text: str, section_name: str, minimum: int) -> bool:
    """Check bullet count for a specific section."""
    body = _extract_section(text, section_name)

    if not body:
        return False

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", body))

    return bullet_count >= minimum


def _topic_section_terms(topic: str = "") -> dict[str, list[str]]:
    """Return topic-aware SOC quality terms."""
    key = _topic_key(topic)

    if key == "ddos":
        return {
            "ioc": [
                "traffic",
                "packets",
                "requests",
                "bandwidth",
                "syn",
                "udp",
                "dns",
                "http",
                "latency",
                "timeout",
                "dropped",
                "user agents",
                "distributed ip",
            ],
            "detection": [
                "traffic baselines",
                "packets per second",
                "requests per second",
                "bandwidth",
                "syn",
                "udp",
                "dns",
                "http",
                "waf",
                "cdn",
                "firewall",
                "latency",
                "timeout",
                "load balancer",
                "logs",
            ],
            "mitigation": [
                "cdn",
                "waf",
                "ddos protection",
                "rate limiting",
                "scrubbing",
                "filtering",
                "autoscaling",
                "caching",
                "isp",
                "provider",
                "playbook",
                "anycast",
            ],
        }

    if key == "brute_force":
        return {
            "ioc": [
                "failed login",
                "password spraying",
                "credential stuffing",
                "source ip",
                "impossible travel",
                "mfa",
                "login attempts",
                "authentication",
            ],
            "detection": [
                "authentication logs",
                "failed-login",
                "password-spraying",
                "source ip",
                "impossible travel",
                "mfa",
                "identity",
                "vpn",
                "cloud audit",
                "login velocity",
            ],
            "mitigation": [
                "mfa",
                "conditional access",
                "password",
                "lockout",
                "throttling",
                "legacy authentication",
                "ip reputation",
                "device trust",
                "adaptive authentication",
            ],
        }

    if key == "trojan":
        return {
            "ioc": [
                "child processes",
                "registry",
                "scheduled tasks",
                "services",
                "command-and-control",
                "domains",
                "credential theft",
                "dll",
                "payload",
                "persistence",
            ],
            "detection": [
                "process trees",
                "script execution",
                "persistence",
                "dns",
                "http",
                "beaconing",
                "endpoint",
                "edr",
                "injection",
                "credential",
            ],
            "mitigation": [
                "email security",
                "proxy",
                "dns filtering",
                "edr",
                "application control",
                "patch",
                "isolate",
                "revoke",
                "credentials",
                "script execution",
            ],
        }

    return {
        "ioc": [
            "powershell",
            "vssadmin",
            "wmic",
            "diskshadow",
            "psexec",
            "remote service",
            "smb",
            "ransom-note",
            "encryption",
            "shadow copy",
        ],
        "detection": [
            "file modification",
            "ransom-note",
            "encryption",
            "shadow copy",
            "powershell",
            "psexec",
            "smb",
            "rdp",
            "telemetry",
        ],
        "mitigation": [
            "segmentation",
            "least privilege",
            "mfa",
            "backup",
            "immutable",
            "restore",
            "edr",
            "siem",
            "incident-response",
            "playbook",
        ],
    }


def _soc_sections_are_strong_enough(text: str, topic: str = "") -> bool:
    """Lightweight topic-aware check for IOCs, Detection, and Mitigation strength."""
    normalized = text.lower()
    terms = _topic_section_terms(topic)

    ioc_hits = sum(term in normalized for term in terms["ioc"])
    detection_hits = sum(term in normalized for term in terms["detection"])
    mitigation_hits = sum(term in normalized for term in terms["mitigation"])

    if ioc_hits < 3:
        return False

    if detection_hits < 3:
        return False

    if mitigation_hits < 3:
        return False

    if not _section_has_enough_bullets(text, "IOCs", 3):
        return False

    if not _section_has_enough_bullets(text, "Detection", 3):
        return False

    if not _section_has_enough_bullets(text, "Mitigation", 3):
        return False

    return True


def _is_good_enough(draft_answer: str, topic: str = "") -> bool:
    """Lightweight rule-based gate to decide whether critic/reviser can be skipped."""
    if not draft_answer or not draft_answer.strip():
        return False

    if not _contains_all_required_sections(draft_answer):
        return False

    if _has_duplicate_headers(draft_answer):
        return False

    if _contains_placeholder_text(draft_answer):
        return False

    if _is_trivially_generic(draft_answer):
        return False

    if _recent_examples_are_weak(draft_answer, topic=topic):
        return False

    if not _soc_sections_are_strong_enough(draft_answer, topic=topic):
        return False

    return True


def _writer_quality_gate(state: AppState) -> str:
    """Route from writer directly to validator when the draft is already strong."""
    draft_answer = state.get("draft_answer", "") or ""
    topic = state.get("topic", "")
    fast_path = _is_good_enough(draft_answer, topic=topic)

    state["fast_path_used"] = fast_path

    if fast_path:
        debug_print("[fast_path] skipping critic/reviser")

        record_execution_step(
            state,
            "fast_path",
            title="Fast Path",
            summary="Writer draft passed topic-aware quality checks; skipped critic and reviser.",
            details={
                "decision": "skip_critic_reviser",
                "topic": topic,
                "draft_length": len(draft_answer),
                "required_sections_present": _contains_all_required_sections(draft_answer),
                "duplicate_headers": _has_duplicate_headers(draft_answer),
                "recent_examples_weak": _recent_examples_are_weak(draft_answer, topic=topic),
                "soc_sections_strong": _soc_sections_are_strong_enough(draft_answer, topic=topic),
            },
        )

        return "validator"

    debug_print("[fast_path] using full pipeline")

    record_execution_step(
        state,
        "fast_path",
        title="Fast Path",
        summary="Writer draft did not pass topic-aware quality checks; using full pipeline.",
        details={
            "decision": "use_full_pipeline",
            "topic": topic,
            "draft_length": len(draft_answer),
            "required_sections_present": _contains_all_required_sections(draft_answer),
            "duplicate_headers": _has_duplicate_headers(draft_answer),
            "has_placeholders": _contains_placeholder_text(draft_answer),
            "too_generic": _is_trivially_generic(draft_answer),
            "recent_examples_weak": _recent_examples_are_weak(draft_answer, topic=topic),
            "soc_sections_strong": _soc_sections_are_strong_enough(draft_answer, topic=topic),
        },
    )

    return "critic"


def router_node(state: AppState) -> AppState:
    """Decide which retrieval path to use."""
    route = _decide_route(state)

    debug_print(f"\n[router_node] Selected route: {route}")
    debug_print(f"[router_node] incoming web_results: {len(state.get('web_results', []))}")

    state["route"] = route

    record_execution_step(
        state,
        "router",
        title="Router",
        summary=f"Chose the `{route}` retrieval path.",
        details={
            "route": route,
            "user_query": state.get("user_query", ""),
            "plan_preview": state.get("plan", ""),
        },
    )

    return state


def build_graph():
    """Build and compile the threat-intelligence workflow.

    Flow:
    Planner -> Router -> Retriever -> Web Search -> Writer
    Then:
    - Strong draft: Writer -> Validator
    - Weak draft: Writer -> Critic -> Reviser -> Validator
    """
    graph = StateGraph(AppState)

    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("writer", writer_node)
    graph.add_node("validator", validator_node)

    graph.add_node("critic", critic_node)
    graph.add_node("reviser", reviser_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "web_search")
    graph.add_edge("web_search", "writer")

    graph.add_conditional_edges(
        "writer",
        _writer_quality_gate,
        {
            "validator": "validator",
            "critic": "critic",
        },
    )

    graph.add_edge("critic", "reviser")
    graph.add_edge("reviser", "validator")
    graph.add_edge("validator", END)

    return graph.compile()


def run_graph(state: AppState) -> AppState:
    """Run the compiled graph with the provided state."""
    return build_graph().invoke(state)