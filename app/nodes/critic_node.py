"""Critic node for concise reflection on the writer draft."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step
from app.utils.prompts import load_prompt

_NO_DRAFT_FALLBACK = (
    "Missing Sections:\n"
    "- No draft answer was produced.\n\n"
    "Reasoning:\n"
    "- Critic could not assess reasoning without a draft.\n\n"
    "Support:\n"
    "- Critic could not assess unsupported or vague claims.\n\n"
    "Recent Examples:\n"
    "- Critic could not verify whether recent examples were included.\n\n"
    "Clarity:\n"
    "- Critic could not assess clarity or structure."
)

_STRONG_DRAFT_FEEDBACK = (
    "Missing Sections:\n"
    "- None.\n\n"
    "Reasoning:\n"
    "- None.\n\n"
    "Support:\n"
    "- None.\n\n"
    "Recent Examples:\n"
    "- None.\n\n"
    "Clarity:\n"
    "- None."
)

_SECTION_PATTERN = re.compile(
    r"(?mi)^(Overview|Attack Explanation|Recent Examples|IOCs|Indicators of Compromise|IOCs\s*\(Indicators of Compromise\)|Detection|Mitigation|Limitations)\s*:?\s*$"
)


def get_critic_llm() -> ChatOllama:
    return ChatOllama(model="llama3.2", temperature=0.0)


def filter_valid_documents(documents):
    """Backward-compatible helper."""
    return [doc for doc in documents if getattr(doc, "page_content", "").strip()]


def _extract_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_PATTERN.finditer(text or ""))

    if not matches:
        return {}

    aliases = {
        "Indicators of Compromise": "IOCs",
        "IOCs (Indicators of Compromise)": "IOCs",
    }

    sections: dict[str, str] = {}

    for index, match in enumerate(matches):
        name = aliases.get(match.group(1), match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    return sections


def _recent_examples_weak(recent_examples: str, web_results: list[dict]) -> bool:
    normalized = re.sub(r"\s+", " ", recent_examples or "").strip().lower()

    if not normalized:
        return True

    if web_results and len(normalized.split()) < 80:
        return True

    bullet_count = len(re.findall(r"(?m)^\s*[-*]\s+", recent_examples or ""))

    if web_results and bullet_count < 2:
        return True

    weak_patterns = [
        "responds to many",
        "statistics",
        "percentage",
        "trend",
        "report",
        "didn't detect",
        "did not detect",
        "what is",
        "examples of",
        "guide",
        "glossary",
        "definition",
    ]

    if any(pattern in normalized for pattern in weak_patterns):
        return True

    analyst_markers = [
        "which shows",
        "which suggests",
        "this matters",
        "highlighting",
        "reinforcing",
        "indicating",
        "demonstrating",
    ]

    if web_results and not any(marker in normalized for marker in analyst_markers):
        return True

    return False


def _draft_is_strong_enough(draft_answer: str, web_results: list[dict]) -> bool:
    sections = _extract_sections(draft_answer)

    required = [
        "Overview",
        "Attack Explanation",
        "Recent Examples",
        "IOCs",
        "Detection",
        "Mitigation",
        "Limitations",
    ]

    if any(not sections.get(section, "").strip() for section in required):
        return False

    attack = sections["Attack Explanation"].lower()

    attack_groups = [
        ("initial access", "phishing", "credentials", "vulnerability", "exposed"),
        ("execution", "payload", "tooling", "script", "process"),
        ("privilege escalation", "admin", "elevat", "token", "credential"),
        ("lateral movement", "smb", "rdp", "psexec", "remote service"),
        ("shadow copy", "backup", "vssadmin", "wmic", "diskshadow"),
        ("encrypt", "extortion", "ransom", "data theft"),
    ]

    if sum(any(term in attack for term in group) for group in attack_groups) < 3:
        return False

    recent_examples = sections["Recent Examples"]

    if _recent_examples_weak(recent_examples, web_results):
        return False

    iocs = sections["IOCs"].lower()

    if sum(term in iocs for term in ("powershell", "vssadmin", "wmic", "psexec", "smb")) < 2:
        return False

    detection = sections["Detection"].lower()

    if sum(term in detection for term in ("powershell", "shadow copy", "smb", "rdp")) < 2:
        return False

    mitigation = sections["Mitigation"].lower()

    if sum(term in mitigation for term in ("segment", "backup", "mfa", "least privilege")) < 2:
        return False

    if len(draft_answer) <= 1500:
        return False

    return True


def _build_rule_based_feedback(draft_answer: str, web_results: list[dict]) -> str:
    sections = _extract_sections(draft_answer)

    feedback = {
        "Missing Sections": [],
        "Reasoning": [],
        "Support": [],
        "Recent Examples": [],
        "Clarity": [],
    }

    required = [
        "Overview",
        "Attack Explanation",
        "Recent Examples",
        "IOCs",
        "Detection",
        "Mitigation",
        "Limitations",
    ]

    missing = [section for section in required if not sections.get(section, "").strip()]

    if missing:
        feedback["Missing Sections"].append(f"Missing or empty sections: {', '.join(missing)}.")

    if _recent_examples_weak(sections.get("Recent Examples", ""), web_results):
        feedback["Recent Examples"].append(
            "Recent Examples are weak. Rewrite them as analyst-style bullets explaining what happened and why it matters. Avoid snippet fragments, generic reports, statistics-only bullets, and article-title dumps."
        )

    attack = sections.get("Attack Explanation", "").lower()

    if attack and sum(term in attack for term in ("initial access", "lateral", "encrypt", "extortion", "backup")) < 3:
        feedback["Reasoning"].append(
            "Attack Explanation needs clearer ransomware chain coverage: initial access, privilege escalation, lateral movement, backup tampering, encryption, and extortion."
        )

    iocs = sections.get("IOCs", "").lower()

    if iocs and sum(term in iocs for term in ("powershell", "vssadmin", "wmic", "psexec", "smb")) < 2:
        feedback["Support"].append(
            "IOCs are too generic. Include behavioral and tool-based indicators such as PowerShell, VSS deletion, PsExec, SMB fan-out, or ransom-note creation."
        )

    detection = sections.get("Detection", "").lower()

    if detection and sum(term in detection for term in ("powershell", "shadow copy", "smb", "rdp", "encryption")) < 2:
        feedback["Support"].append(
            "Detection needs more SOC-level monitoring logic for endpoint, identity, network, lateral movement, and encryption behavior."
        )

    mitigation = sections.get("Mitigation", "").lower()

    if mitigation and sum(term in mitigation for term in ("segment", "backup", "mfa", "least privilege")) < 2:
        feedback["Support"].append(
            "Mitigation needs stronger controls such as MFA, least privilege, segmentation, immutable backups, restoration testing, and incident-response readiness."
        )

    output_parts = []

    for section_name, items in feedback.items():
        output_parts.append(f"{section_name}:")
        if items:
            output_parts.extend(f"- {item}" for item in items)
        else:
            output_parts.append("- None.")
        output_parts.append("")

    return "\n".join(output_parts).strip()


def critic_node(state: AppState) -> AppState:
    user_query = state.get("user_query", "").strip()
    plan = state.get("plan", "").strip()
    draft_answer = state.get("draft_answer", "").strip()
    web_results = state.get("web_results", [])

    debug_print("\n[critic_node] Debug")
    debug_print(f"[critic_node] incoming web_results: {len(web_results)}")
    debug_print(f"Query: {user_query}")
    debug_print(f"Draft length: {len(draft_answer)}")

    if not draft_answer:
        state["critic_feedback"] = _NO_DRAFT_FALLBACK

        record_execution_step(
            state,
            "critic",
            title="Critic",
            summary="Skipped critique because no writer draft was available.",
            details={"critic_feedback": _NO_DRAFT_FALLBACK},
        )

        return state

    rule_feedback = _build_rule_based_feedback(draft_answer, web_results)

    if _draft_is_strong_enough(draft_answer, web_results):
        state["critic_feedback"] = _STRONG_DRAFT_FEEDBACK

        record_execution_step(
            state,
            "critic",
            title="Critic",
            summary="Skipped LLM critique because the draft passed strict quality gates.",
            details={
                "critic_feedback": _STRONG_DRAFT_FEEDBACK,
                "skipped_llm": True,
            },
        )

        return state

    if "Recent Examples:\n- Recent Examples are weak" in rule_feedback:
        state["critic_feedback"] = rule_feedback

        record_execution_step(
            state,
            "critic",
            title="Critic",
            summary="Used rule-based critique because Recent Examples failed strict quality checks.",
            details={
                "critic_feedback": rule_feedback,
                "skipped_llm": True,
            },
        )

        return state

    llm = get_critic_llm()

    system_prompt = load_prompt(
        "critic_prompt.txt",
        """
You are a cybersecurity threat-intelligence critic.

Review the draft answer and provide SHORT but STRICT feedback.

Focus ONLY on critical issues:
1. Missing or weak sections
2. Weak or generic Attack Explanation
3. Recent Examples quality
4. Weak IOCs
5. Weak Detection/Mitigation

Rules:
- Be strict.
- If Recent Examples look like snippets, generic trends, statistics, article titles, or vague reports, flag them.
- Only say "None" if the section is truly strong.
- Do NOT rewrite the answer.

Output format:

Missing Sections:
- ...

Reasoning:
- ...

Support:
- ...

Recent Examples:
- ...

Clarity:
- ...
        """,
    )

    human_prompt = f"""
Query:
{user_query}

Planner Output:
{plan[:250]}

Draft:
{draft_answer[:1500]}

Rule-based pre-check:
{rule_feedback}

Review whether the draft has weak or generic sections, especially Recent Examples, IOCs, Detection, and Mitigation.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    critic_feedback = (
        response.content if isinstance(response.content, str) else str(response.content)
    ).strip()

    if not critic_feedback:
        critic_feedback = rule_feedback

    state["critic_feedback"] = critic_feedback

    record_execution_step(
        state,
        "critic",
        title="Critic",
        summary="Reviewed the writer draft for critical weaknesses.",
        details={"critic_feedback": critic_feedback},
    )

    return state