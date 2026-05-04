"""Lightweight final-output validation and safe post-processing.

Topic-aware version:
- Prevents DDoS / brute-force / trojan answers from being repaired with ransomware defaults.
- Preserves the same validation_report structure used by the API/UI.
- Performs deterministic post-processing only; it does not call the LLM.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

REQUIRED_SECTIONS = [
    "Overview",
    "Attack Explanation",
    "Recent Examples",
    "IOCs",
    "Detection",
    "Mitigation",
    "Limitations",
]

SECTION_ALIASES = {
    "Overview": "Overview",
    "Attack Explanation": "Attack Explanation",
    "Recent Examples": "Recent Examples",
    "IOCs": "IOCs",
    "Indicators of Compromise": "IOCs",
    "IOCs (Indicators of Compromise)": "IOCs",
    "Detection": "Detection",
    "Mitigation": "Mitigation",
    "Limitations": "Limitations",
    "Sources": None,
}

SECTION_HEADER_PATTERN = re.compile(
    r"(?mi)^(Overview|Attack Explanation|Recent Examples|IOCs|Indicators of Compromise|IOCs\s*\(Indicators of Compromise\)|Detection|Mitigation|Limitations|Sources)\s*:?\s*$"
)

WEB_RESULT_REFERENCE_PATTERN = re.compile(
    r"\s*(?:\[(?:Web Result)\s*\d+\]|\((?:Web Result)\s*\d+\))"
)

WEAK_TITLE_DUMP_PATTERNS = (
    "examples of",
    "what is",
    "cybersecurity 101",
    "opinion ###",
    "image ",
    "by ",
    "mins ",
    "home »",
    "read more",
    "subscribe",
    "newsletter",
    "advertisement",
    "top 10",
    "| cso online",
)

STOP_ENTITY_PHRASES = {
    "Overview",
    "Attack Explanation",
    "Recent Examples",
    "Indicators of Compromise",
    "IOCs",
    "Detection",
    "Mitigation",
    "Limitations",
    "Sources",
    "Web Result",
}

ORG_SUFFIXES = (
    "bank",
    "hospital",
    "health",
    "healthcare",
    "medical center",
    "clinic",
    "university",
    "college",
    "school",
    "district",
    "agency",
    "department",
    "ministry",
    "corporation",
    "corp",
    "company",
    "co",
    "institute",
    "authority",
    "council",
    "group",
    "holdings",
    "systems",
    "services",
    "laboratories",
    "labs",
)

TECHNICAL_ENTITY_TERMS = {
    "powershell",
    "wmic",
    "psexec",
    "vssadmin",
    "diskshadow",
    "edr",
    "siem",
    "smb",
    "rdp",
    "ioc",
    "iocs",
    "detection",
    "mitigation",
    "ransom note",
    "ransom notes",
}

TECHNICAL_ENTITY_SUBSTRINGS = (
    "hash",
    "domain",
    "command",
    "detect",
    "mitigat",
    "technique",
    "tool",
    "powershell",
    "wmic",
    "psexec",
    "vssadmin",
    "diskshadow",
    "edr",
    "siem",
    "smb",
    "rdp",
    "ransom note",
)


def _normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_phrase(value: str) -> str:
    """Normalize text for approximate matching."""
    return re.sub(r"[^a-z0-9]+", " ", _normalize_whitespace(value).lower()).strip()


def _topic_key(topic: str = "") -> str:
    """Normalize topic into one of the supported validator keys."""
    normalized = _normalize_whitespace(topic).lower()

    if "ddos" in normalized or "denial of service" in normalized or normalized == "dos":
        return "ddos"

    if "brute" in normalized or "credential" in normalized or "password" in normalized:
        return "brute_force"

    if "trojan" in normalized or "rat" in normalized or "backdoor" in normalized:
        return "trojan"

    return "ransomware"


def _section_fallbacks(topic: str = "") -> dict[str, str]:
    """Return topic-aware grounded fallbacks used when output is malformed or weak."""
    key = _topic_key(topic)

    if key == "ddos":
        return {
            "Overview": (
                "DDoS is a cybersecurity threat where attackers attempt to overwhelm an online service, "
                "network, or application with excessive traffic so legitimate users cannot access it."
            ),
            "Attack Explanation": (
                "DDoS attacks use botnets, compromised servers, reflection or amplification techniques, "
                "and high traffic volume to exhaust bandwidth, connection tables, application resources, "
                "or upstream infrastructure. Common forms include SYN floods, UDP floods, DNS or NTP "
                "amplification, HTTP floods, and application-layer request abuse. The goal is usually "
                "service disruption, extortion pressure, distraction during another intrusion, or reputational damage."
            ),
            "Recent Examples": (
                "- Current reporting in this run is more trend-oriented than incident-specific, but it still shows "
                "DDoS activity affecting service availability, infrastructure resilience, and response readiness."
            ),
            "IOCs": (
                "- Sudden traffic spikes from many distributed IP addresses, regions, autonomous systems, or cloud providers.\n"
                "- Abnormal increases in SYN packets, UDP traffic, DNS queries, HTTP requests, or repeated requests to expensive application endpoints.\n"
                "- High error rates, connection exhaustion, increased latency, packet drops, or service timeouts during the same time window.\n"
                "- Repeated traffic patterns with suspicious user agents, malformed requests, spoofed sources, or unusual protocol ratios."
            ),
            "Detection": (
                "- Monitor traffic baselines for sudden spikes in packets per second, requests per second, bandwidth, connection attempts, or DNS query volume.\n"
                "- Alert on SYN floods, UDP floods, DNS or NTP amplification patterns, HTTP request floods, and abnormal traffic concentration against specific services.\n"
                "- Correlate load balancer, firewall, WAF, CDN, DNS, and application logs to distinguish legitimate flash traffic from hostile traffic.\n"
                "- Track service health metrics such as latency, timeout rates, dropped connections, upstream saturation, and backend CPU or memory pressure."
            ),
            "Mitigation": (
                "- Use CDN, WAF, DDoS protection, rate limiting, traffic scrubbing, and upstream provider filtering to absorb or block malicious traffic.\n"
                "- Apply autoscaling, caching, request throttling, and circuit breakers for application-layer resilience.\n"
                "- Harden DNS and edge infrastructure, restrict exposed services, and use anycast or redundant hosting where possible.\n"
                "- Maintain a DDoS response playbook with escalation paths to ISP, CDN, cloud provider, SOC, and incident-response teams."
            ),
            "Limitations": (
                "This answer is limited to the retrieved documents and live web results available in this run. "
                "DDoS attribution is often difficult because traffic may come from compromised systems, proxies, or spoofed infrastructure."
            ),
        }

    if key == "brute_force":
        return {
            "Overview": (
                "Brute force is an identity-focused attack where adversaries repeatedly attempt passwords, "
                "credentials, or authentication combinations to gain unauthorized access."
            ),
            "Attack Explanation": (
                "Brute-force attacks may involve high-volume password guessing, password spraying, credential stuffing, "
                "or automated login attempts against VPNs, email portals, cloud services, SSH, RDP, and web applications. "
                "Attackers often use leaked credential lists, rotating IP addresses, proxy infrastructure, and low-and-slow "
                "timing to avoid lockouts and detection."
            ),
            "Recent Examples": (
                "- Current reporting in this run is more trend-oriented than incident-specific, but it still shows credential "
                "and authentication attacks remain operationally relevant for defenders."
            ),
            "IOCs": (
                "- Repeated failed logins against one account, many accounts, or the same service from unusual IP addresses.\n"
                "- Password-spraying patterns where one or a few passwords are tried across many users.\n"
                "- Login attempts from anonymizers, cloud hosting providers, unusual geographies, or impossible-travel patterns.\n"
                "- Successful login shortly after repeated failures, followed by MFA changes, mailbox rules, token creation, or privilege changes."
            ),
            "Detection": (
                "- Monitor authentication logs for failed-login spikes, password-spraying patterns, and unusual source IP diversity.\n"
                "- Alert on successful login after repeated failures, impossible travel, new device fingerprints, and suspicious MFA behavior.\n"
                "- Correlate identity-provider, VPN, email, endpoint, and cloud audit logs to detect post-compromise activity.\n"
                "- Track lockout events, login velocity, anomalous user-agent strings, and access attempts outside normal working patterns."
            ),
            "Mitigation": (
                "- Enforce MFA, conditional access, strong password policies, breached-password blocking, and account lockout or throttling.\n"
                "- Disable legacy authentication and protect VPN, RDP, email, and admin portals with additional access controls.\n"
                "- Use IP reputation, geo-risk rules, device trust, and adaptive authentication to reduce automated login abuse.\n"
                "- Monitor privileged accounts closely and review suspicious mailbox rules, OAuth grants, and session tokens after suspected compromise."
            ),
            "Limitations": (
                "This answer is limited to the retrieved documents and live web results available in this run. "
                "Brute-force activity may blend into normal login noise without strong identity telemetry."
            ),
        }

    if key == "trojan":
        return {
            "Overview": (
                "A trojan is malware that disguises itself as legitimate software or content while performing malicious "
                "actions such as persistence, credential theft, surveillance, payload delivery, or remote access."
            ),
            "Attack Explanation": (
                "Trojan infections commonly start through phishing attachments, malicious downloads, cracked software, "
                "fake installers, or compromised websites. After execution, a trojan may establish persistence, contact "
                "command-and-control infrastructure, collect credentials, download additional payloads, inject into processes, "
                "or give attackers remote access to the system."
            ),
            "Recent Examples": (
                "- Current reporting in this run is more trend-oriented than incident-specific, but it still shows trojan "
                "and backdoor activity remains relevant for endpoint and SOC monitoring."
            ),
            "IOCs": (
                "- Suspicious child processes from Office files, scripts, installers, browsers, or archive utilities.\n"
                "- New persistence entries such as registry run keys, scheduled tasks, startup folder items, services, or modified shortcuts.\n"
                "- Command-and-control connections to unusual domains, IPs, ports, or newly registered infrastructure.\n"
                "- Credential theft behavior, process injection, suspicious DLL loading, or unexpected payload downloads."
            ),
            "Detection": (
                "- Monitor process trees for suspicious parent-child relationships, script execution, unsigned binaries, and abnormal persistence creation.\n"
                "- Alert on unusual outbound beaconing, rare domains, newly registered domains, and suspicious DNS or HTTP patterns.\n"
                "- Correlate endpoint telemetry with email, proxy, DNS, and identity logs to identify infection chains and post-compromise behavior.\n"
                "- Use EDR detections for credential dumping, injection, persistence, privilege escalation, and suspicious file writes."
            ),
            "Mitigation": (
                "- Block malicious attachments, risky file types, and known-bad domains through email security, proxy, DNS filtering, and EDR.\n"
                "- Restrict script execution, enforce application control, remove local admin rights, and patch commonly exploited software.\n"
                "- Isolate infected endpoints, revoke stolen credentials, remove persistence, and investigate lateral movement or additional payloads.\n"
                "- Train users against fake installers, cracked software, phishing attachments, and social-engineering delivery paths."
            ),
            "Limitations": (
                "This answer is limited to the retrieved documents and live web results available in this run. "
                "Trojan behavior varies significantly by malware family and campaign."
            ),
        }

    return {
        "Overview": (
            "This analysis summarizes ransomware behavior using only the retrieved documents and live web results available in this run."
        ),
        "Attack Explanation": (
            "Ransomware intrusions commonly begin with phishing, exposed remote services, stolen credentials, or exploitation of vulnerable systems. "
            "Operators execute payloads or administrative tooling, escalate privileges, move laterally with SMB, RDP, PsExec, WMI, or remote services, "
            "disrupt backups or delete shadow copies to block recovery, and then encrypt systems while applying extortion pressure through operational disruption or stolen-data leverage."
        ),
        "Recent Examples": (
            "- Current reporting in this run is more trend-oriented than incident-specific, but it still shows ransomware operations "
            "emphasizing disruption, recovery impairment, and extortion pressure."
        ),
        "IOCs": (
            "- Suspicious PowerShell execution used for staging, remote execution, or defense evasion.\n"
            "- Shadow copy deletion activity involving `vssadmin`, `wmic`, or `diskshadow` that interferes with recovery.\n"
            "- PsExec, remote service creation, or SMB administrative-share activity consistent with lateral rollout.\n"
            "- Ransom-note creation, burst encryption, or unusual outbound transfers consistent with extortion staging."
        ),
        "Detection": (
            "- Monitor mass file modifications, rapid rename bursts, ransom-note creation, and encryption spikes across endpoints and shared drives.\n"
            "- Alert on shadow copy deletion, backup tampering, suspicious PowerShell, `wmic`, `vssadmin`, `diskshadow`, PsExec, or remote service creation.\n"
            "- Track SMB fan-out, administrative share access, RDP pivots, scheduled tasks, and multi-host execution patterns that indicate lateral movement.\n"
            "- Correlate endpoint, identity, and network telemetry so exfiltration or encryption activity is triaged quickly."
        ),
        "Mitigation": (
            "- Enforce network segmentation across user networks, server tiers, identity systems, and backup infrastructure to constrain lateral movement.\n"
            "- Apply least privilege, reduce standing admin rights, require MFA for remote access, and tightly control privileged tooling and service accounts.\n"
            "- Maintain offline or immutable backups, separate backup credentials, and test restoration regularly.\n"
            "- Keep EDR, SIEM, containment workflows, and incident-response playbooks ready so ransomware activity can be isolated quickly."
        ),
        "Limitations": "This answer is limited to the retrieved documents and live web results available in this run.",
    }


def _strip_embedded_headers(body: str) -> str:
    """Remove nested schema headers from a section body."""
    return SECTION_HEADER_PATTERN.sub("", body or "").strip()


def _strip_web_result_references(text: str) -> tuple[str, bool]:
    """Remove internal web-result references from user-facing output."""
    cleaned = WEB_RESULT_REFERENCE_PATTERN.sub("", text or "")
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), cleaned.strip() != (text or "").strip()


def _dedupe_blocks(text: str) -> str:
    """Remove duplicate paragraphs or bullets while preserving order."""
    if not text.strip():
        return ""

    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]

    if len(parts) <= 1:
        lines = [line.rstrip() for line in text.splitlines()]
        deduped_lines: list[str] = []
        seen_lines: set[str] = set()

        for line in lines:
            key = _normalize_phrase(line)
            if key and key in seen_lines:
                continue

            if key:
                seen_lines.add(key)

            deduped_lines.append(line)

        return "\n".join(deduped_lines).strip()

    deduped_parts: list[str] = []
    seen_parts: set[str] = set()

    for part in parts:
        key = _normalize_phrase(part)
        if key in seen_parts:
            continue

        seen_parts.add(key)
        deduped_parts.append(part)

    return "\n\n".join(deduped_parts).strip()


def _extract_sections(text: str) -> tuple[dict[str, list[str]], dict[str, int], list[str]]:
    """Extract canonical section content and track raw header usage."""
    normalized_text = (text or "").strip()

    if not normalized_text:
        return {}, {}, []

    matches = list(SECTION_HEADER_PATTERN.finditer(normalized_text))

    if not matches:
        return {"Overview": [normalized_text]}, {}, []

    sections: dict[str, list[str]] = {section: [] for section in REQUIRED_SECTIONS}
    header_counts: dict[str, int] = {}
    normalized_headers: list[str] = []

    for index, match in enumerate(matches):
        raw_header = _normalize_whitespace(match.group(1))
        canonical_header = SECTION_ALIASES.get(raw_header)

        if raw_header != canonical_header and canonical_header:
            normalized_headers.append(f"{raw_header} -> {canonical_header}")

        if canonical_header:
            header_counts[canonical_header] = header_counts.get(canonical_header, 0) + 1

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        body = _dedupe_blocks(_strip_embedded_headers(normalized_text[start:end].strip()))

        if canonical_header and body:
            sections[canonical_header].append(body)

    return sections, header_counts, normalized_headers


def _split_bullets(content: str) -> list[str]:
    """Return bullet-like lines from a section."""
    bullets = []

    for raw_line in content.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(("-", "*")):
            bullets.append(line.lstrip("-* ").strip())

    if bullets:
        return bullets

    if content.strip():
        return [content.strip()]

    return []


def _title_matches_known_result(line: str, titles: list[str]) -> bool:
    """Detect bullets that mirror web result titles or vendor article headings."""
    normalized_line = _normalize_phrase(line)

    if not normalized_line:
        return False

    for title in titles:
        normalized_title = _normalize_phrase(title)

        if not normalized_title:
            continue

        if normalized_line == normalized_title:
            return True

        if normalized_line in normalized_title and len(normalized_line) >= 12:
            return True

        if normalized_title in normalized_line and len(normalized_title) >= 12:
            return True

        if SequenceMatcher(None, normalized_line, normalized_title).ratio() >= 0.92:
            return True

    return False


def _sanitize_recent_examples(content: str, web_results: list[dict[str, Any]]) -> tuple[str, bool]:
    """Remove weak title-dump bullets from Recent Examples only."""
    if not content.strip():
        return content, False

    titles = [_normalize_whitespace(result.get("title", "")) for result in web_results]
    removed = False
    kept_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        normalized_line = _normalize_phrase(line)

        if normalized_line and any(pattern in normalized_line for pattern in WEAK_TITLE_DUMP_PATTERNS):
            removed = True
            continue

        if line.startswith(("-", "*")) and _title_matches_known_result(line.lstrip("-* ").strip(), titles):
            removed = True
            continue

        kept_lines.append(raw_line.rstrip())

    sanitized = "\n".join(line for line in kept_lines if line.strip()).strip()

    if sanitized:
        return sanitized, removed

    return "", True


def _extract_entity_candidates(text: str) -> set[str]:
    """Extract likely victim-organization entities conservatively."""
    candidates: set[str] = set()

    patterns = [
        r"\b(?:[A-Z][A-Za-z&.\-']+)(?:\s+[A-Z][A-Za-z&.\-']+){0,4}\s+(?:Bank|Hospital|Health|Healthcare|Medical Center|Clinic|University|College|School|District|Agency|Department|Ministry|Corporation|Corp|Company|Institute|Authority|Council|Group|Holdings|Laboratories|Labs?)\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text or ""):
            candidate = _normalize_whitespace(match.group(0)).strip(" ,.;:()[]")

            if candidate in STOP_ENTITY_PHRASES:
                continue

            normalized = _normalize_phrase(candidate)

            if not normalized:
                continue

            lowered = normalized.lower()

            if lowered in TECHNICAL_ENTITY_TERMS:
                continue

            if any(term in lowered for term in TECHNICAL_ENTITY_SUBSTRINGS):
                continue

            if not any(suffix in lowered for suffix in ORG_SUFFIXES):
                continue

            candidates.add(candidate)

    return candidates


def _build_supported_entity_set(retrieved_docs: list[Any], web_results: list[dict[str, Any]]) -> set[str]:
    """Collect entity names explicitly present in retrieved or web evidence."""
    supported: set[str] = set()

    for doc in retrieved_docs or []:
        page_content = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}

        for candidate in _extract_entity_candidates(page_content):
            supported.add(_normalize_phrase(candidate))

        for value in metadata.values():
            if isinstance(value, str):
                for candidate in _extract_entity_candidates(value):
                    supported.add(_normalize_phrase(candidate))

    for result in web_results or []:
        for field in ("title", "content"):
            for candidate in _extract_entity_candidates(result.get(field, "") or ""):
                supported.add(_normalize_phrase(candidate))

    return supported


def _generalize_entity_name(entity: str) -> str:
    """Convert a specific organization into a safe generalized category."""
    lowered = entity.lower()

    if any(term in lowered for term in ("hospital", "health", "healthcare", "clinic", "medical center")):
        return "a healthcare organization"

    if any(term in lowered for term in ("bank", "financial", "credit union")):
        return "a financial institution"

    if any(term in lowered for term in ("university", "college", "school", "district")):
        return "an education-sector organization"

    if any(term in lowered for term in ("agency", "department", "ministry", "authority", "council")):
        return "a government organization"

    if any(term in lowered for term in ("manufacturer", "industrial")):
        return "an industrial organization"

    return "a targeted organization"


def _generalize_unsupported_entities(
    text: str,
    supported_entities: set[str],
) -> tuple[str, list[dict[str, str]]]:
    """Replace unsupported named organizations with safe generalized terms."""
    updated_text = text
    replacements: list[dict[str, str]] = []

    for entity in sorted(_extract_entity_candidates(text), key=len, reverse=True):
        normalized = _normalize_phrase(entity)

        if normalized in supported_entities:
            continue

        replacement = _generalize_entity_name(entity)
        updated_text = re.sub(rf"\b{re.escape(entity)}\b", replacement, updated_text)
        replacements.append({"entity": entity, "replacement": replacement})

    return updated_text, replacements


def _split_sentences(text: str) -> list[str]:
    """Split text into short sentence candidates."""
    normalized = _normalize_whitespace(text)

    if not normalized:
        return []

    protected = normalized.replace("St. ", "St<prd> ").replace("Dr. ", "Dr<prd> ")

    return [
        sentence.replace("<prd>", ".").strip(" -")
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
        if sentence.strip()
    ]


def _topic_operational_terms(topic: str = "") -> tuple[str, ...]:
    """Return topic-aware operational terms for recent example scoring."""
    key = _topic_key(topic)

    if key == "ddos":
        return (
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
        )

    if key == "brute_force":
        return (
            "brute force",
            "password spray",
            "password spraying",
            "credential stuffing",
            "login",
            "authentication",
            "mfa",
            "failed",
            "account",
            "vpn",
            "cloud",
            "identity",
        )

    if key == "trojan":
        return (
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
            "phishing",
        )

    return (
        "disrupt",
        "outage",
        "encrypt",
        "extortion",
        "restore",
        "lateral",
        "exfil",
        "shadow copy",
        "ransomware",
        "recovery",
        "victim",
        "organization",
        "hospital",
        "manufacturer",
        "healthcare",
        "financial",
    )


def _extract_key_sentences(
    content: str,
    *,
    topic: str = "",
    max_sentences: int = 2,
) -> list[str]:
    """Select operationally useful sentences from a web snippet."""
    terms = _topic_operational_terms(topic)
    scored: list[tuple[str, float]] = []

    for sentence in _split_sentences(content):
        lowered = sentence.lower()
        score = 0.0

        score += sum(1.0 for term in terms if term in lowered)

        if re.search(r"\b(2024|2025|2026)\b", sentence):
            score += 1.5

        if len(sentence) > 280:
            score -= 0.5

        scored.append((sentence, score))

    ranked = [
        sentence
        for sentence, score in sorted(scored, key=lambda item: item[1], reverse=True)
        if score > 0
    ]

    selected: list[str] = []
    seen: set[str] = set()

    for sentence in ranked:
        key = _normalize_phrase(sentence)

        if not key or key in seen:
            continue

        seen.add(key)
        selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return selected


def _build_recent_examples_from_web(
    web_results: list[dict[str, Any]],
    *,
    topic: str = "",
) -> str:
    """Build a concise, operationally useful Recent Examples section from live results."""
    defaults = _section_fallbacks(topic)

    if not web_results:
        return defaults["Recent Examples"]

    key = _topic_key(topic)
    terms = _topic_operational_terms(topic)
    bullets: list[str] = []

    wrong_topic_markers = {
        "ddos": (
            "ransomware pressure",
            "ransomware operators",
            "ransomware activity",
            "ransomware deployment",
            "encryption event",
            "ransom demand",
            "data theft, not encryption",
            "shadow copy",
            "ransom-note",
        ),
        "brute_force": (
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
            "ddos activity",
        ),
        "trojan": (
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
            "ddos activity",
        ),
        "ransomware": (),
    }

    for result in web_results[:4]:
        result_title = _normalize_whitespace(result.get("title", ""))

        for sentence in _extract_key_sentences(result.get("content", ""), topic=topic, max_sentences=3):
            lowered = sentence.lower()

            if _title_matches_known_result(sentence, [result_title]):
                continue

            # Remove tag/category dumps like:
            # "Android, botnet, CloudFlare, Cybercrime, cybersecurity..."
            if sentence.count(",") >= 5 and len(sentence.split()) < 18:
                continue

            # Prevent wrong-topic contamination, especially ransomware text inside DDoS output.
            if any(marker in lowered for marker in wrong_topic_markers.get(key, ())):
                continue

            if not any(term in lowered for term in terms) and not re.search(r"\b(2024|2025|2026)\b", sentence):
                continue

            detail = sentence.rstrip(".").strip()

            if not detail:
                continue

            if key == "ddos":
                bullet = (
                    f"- {detail}; this matters because DDoS activity can degrade availability, increase latency, "
                    "exhaust infrastructure resources, and require rapid coordination with CDN, WAF, ISP, or cloud-provider mitigation teams."
                )

            elif key == "brute_force":
                bullet = (
                    f"- {detail}; this matters because credential attacks can blend into normal login activity unless identity telemetry, MFA behavior, and source patterns are monitored."
                )

            elif key == "trojan":
                bullet = (
                    f"- {detail}; this matters because trojan activity can enable persistence, command-and-control, credential theft, or follow-on payload delivery."
                )

            else:
                if "extortion" in lowered or "exfil" in lowered or "data theft" in lowered:
                    bullet = (
                        f"- {detail}; this matters because it shows current operations pairing disruption with extortion leverage."
                    )
                elif "shadow copy" in lowered or "backup" in lowered or "recovery" in lowered:
                    bullet = (
                        f"- {detail}; this matters because it highlights attacker focus on blocking restoration and prolonging impact."
                    )
                else:
                    bullet = (
                        f"- {detail}; this matters because it reflects operational disruption defenders should expect before or during ransomware deployment."
                    )

            if bullet not in bullets:
                bullets.append(bullet)

            if len(bullets) >= 3:
                break

        if len(bullets) >= 3:
            break

    if bullets:
        return "\n".join(bullets[:3])


    return defaults["Recent Examples"]


def _score_section_content(section: str, content: str, *, topic: str = "") -> float:
    """Score section variants so the validator can keep the strongest version."""
    
    normalized = _normalize_whitespace(content).lower()

    if not normalized:
        return 0.0

    score = min(len(normalized) / 40.0, 8.0)

    if section in {"IOCs", "Detection", "Mitigation", "Recent Examples"}:
        score += len(_split_bullets(content)) * 1.5

    if section == "Recent Examples":
        score += sum(1.0 for term in _topic_operational_terms(topic) if term in normalized)

    return score


def _score_bullet(section: str, bullet: str, *, topic: str = "") -> float:
    """Score individual bullets for merged bullet-oriented sections."""
    normalized = _normalize_phrase(bullet)
    score = min(len(normalized.split()), 20)

    if section == "Recent Examples":
        score += sum(1 for term in _topic_operational_terms(topic) if term in normalized)

    if section in {"IOCs", "Detection", "Mitigation"}:
        score += sum(1 for term in _topic_section_terms(section, topic) if term in normalized)

    return float(score)


def _resolve_section_variants(
    section: str,
    variants: list[str],
    *,
    topic: str = "",
) -> tuple[str, bool]:
    """Resolve duplicate sections by keeping strongest content and merging when useful."""
    cleaned_variants = [_dedupe_blocks(variant.strip()) for variant in variants if variant and variant.strip()]

    if not cleaned_variants:
        return "", False

    if len(cleaned_variants) == 1:
        return cleaned_variants[0], False

    strongest = max(cleaned_variants, key=lambda item: _score_section_content(section, item, topic=topic))

    if section not in {"Recent Examples", "IOCs", "Detection", "Mitigation"}:
        return strongest, True

    bullet_candidates: list[str] = []
    seen: set[str] = set()

    for variant in sorted(
        cleaned_variants,
        key=lambda item: _score_section_content(section, item, topic=topic),
        reverse=True,
    ):
        for bullet in _split_bullets(variant):
            key = _normalize_phrase(bullet)

            if not key or key in seen:
                continue

            seen.add(key)
            bullet_candidates.append(bullet)

    ranked = sorted(
        bullet_candidates,
        key=lambda item: _score_bullet(section, item, topic=topic),
        reverse=True,
    )

    limits = {"Recent Examples": 3, "IOCs": 4, "Detection": 4, "Mitigation": 4}
    merged = "\n".join(f"- {bullet}" for bullet in ranked[: limits[section]]).strip()

    return merged or strongest, True


def _collect_context_text(
    final_text: str,
    retrieved_docs: list[Any],
    web_results: list[dict[str, Any]],
) -> str:
    """Combine available evidence into a searchable string for targeted repairs."""
    parts = [final_text]

    for doc in retrieved_docs or []:
        parts.append(getattr(doc, "page_content", "") or "")
        metadata = getattr(doc, "metadata", {}) or {}
        parts.extend(str(value) for value in metadata.values() if value)

    for result in web_results or []:
        parts.append(result.get("title", "") or "")
        parts.append(result.get("content", "") or "")

    return _normalize_whitespace(" ".join(part for part in parts if part)).lower()


def _topic_section_terms(section_name: str, topic: str = "") -> tuple[str, ...]:
    """Return topic-specific expected terms for section quality checks."""
    key = _topic_key(topic)

    if key == "ddos":
        if section_name == "Attack Explanation":
            return (
                "botnet",
                "traffic",
                "bandwidth",
                "connection",
                "application",
                "syn",
                "udp",
                "dns",
                "http",
                "amplification",
                "service disruption",
            )
        if section_name == "IOCs":
            return (
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
            )
        if section_name == "Detection":
            return (
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
            )
        if section_name == "Mitigation":
            return (
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
            )

    if key == "brute_force":
        if section_name == "Attack Explanation":
            return (
                "password",
                "password spraying",
                "credential stuffing",
                "login",
                "authentication",
                "vpn",
                "email",
                "cloud",
                "proxy",
                "lockout",
            )
        if section_name == "IOCs":
            return (
                "failed login",
                "password spraying",
                "credential stuffing",
                "source ip",
                "impossible travel",
                "mfa",
                "login attempts",
            )
        if section_name == "Detection":
            return (
                "authentication logs",
                "failed-login",
                "password-spraying",
                "source ip",
                "impossible travel",
                "mfa",
                "identity",
                "vpn",
                "cloud audit",
            )
        if section_name == "Mitigation":
            return (
                "mfa",
                "conditional access",
                "password",
                "lockout",
                "throttling",
                "legacy authentication",
                "ip reputation",
                "device trust",
            )

    if key == "trojan":
        if section_name == "Attack Explanation":
            return (
                "phishing",
                "download",
                "installer",
                "persistence",
                "command-and-control",
                "credentials",
                "payload",
                "remote access",
            )
        if section_name == "IOCs":
            return (
                "child processes",
                "registry",
                "scheduled tasks",
                "services",
                "command-and-control",
                "domains",
                "credential theft",
                "dll",
                "payload",
            )
        if section_name == "Detection":
            return (
                "process trees",
                "script execution",
                "persistence",
                "dns",
                "http",
                "beaconing",
                "endpoint",
                "edr",
                "injection",
            )
        if section_name == "Mitigation":
            return (
                "email security",
                "proxy",
                "dns filtering",
                "edr",
                "application control",
                "patch",
                "isolate",
                "revoke",
                "credentials",
            )

    # Ransomware defaults
    if section_name == "Attack Explanation":
        return (
            "phishing",
            "credentials",
            "rdp",
            "vpn",
            "payload",
            "privilege",
            "lateral",
            "smb",
            "psexec",
            "shadow copy",
            "backup",
            "encrypt",
            "extortion",
        )

    if section_name == "IOCs":
        return (
            "powershell",
            "vssadmin",
            "wmic",
            "diskshadow",
            "psexec",
            "remote service",
            "service creation",
            "ransom-note",
            "ransom note",
            "encryption",
            "shadow copy",
            "smb",
            "lateral movement",
            "outbound",
            "exfil",
            "scheduled task",
        )

    if section_name == "Detection":
        return (
            "file modification",
            "mass file",
            "rename",
            "extension",
            "ransom-note",
            "encryption",
            "shadow copy",
            "vssadmin",
            "wmic",
            "diskshadow",
            "powershell",
            "psexec",
            "remote service",
            "lateral",
            "smb",
            "rdp",
        )

    if section_name == "Mitigation":
        return (
            "network segmentation",
            "least privilege",
            "local admin",
            "privileged",
            "mfa",
            "offline",
            "immutable",
            "backup",
            "restore",
            "edr",
            "siem",
            "playbook",
            "incident-response",
        )

    return ()


def _content_mentions_wrong_topic(content: str, topic: str = "") -> bool:
    """Detect obvious cross-topic contamination."""
    normalized = _normalize_whitespace(content).lower()
    key = _topic_key(topic)

    ransomware_terms = (
        "ransomware",
        "encrypt",
        "ransom-note",
        "ransom note",
        "shadow copy",
        "vssadmin",
        "diskshadow",
        "psexec",
        "extortion",
        "data theft",
        "ransom demand",
    )

    ddos_terms = (
        "ddos",
        "denial of service",
        "traffic",
        "packets",
        "requests",
        "bandwidth",
        "syn flood",
        "udp flood",
        "dns amplification",
        "http flood",
        "botnet",
    )

    brute_terms = (
        "brute force",
        "password spraying",
        "credential stuffing",
        "failed login",
        "authentication",
        "mfa",
        "login attempts",
    )

    trojan_terms = (
        "trojan",
        "backdoor",
        "remote access",
        "persistence",
        "command-and-control",
        "c2",
        "payload",
        "process injection",
    )

    if key == "ddos":
        wrong_hits = sum(term in normalized for term in ransomware_terms)
        right_hits = sum(term in normalized for term in ddos_terms)
        return wrong_hits >= 2 and right_hits < 3

    if key == "brute_force":
        wrong_hits = sum(term in normalized for term in ransomware_terms)
        right_hits = sum(term in normalized for term in brute_terms)
        return wrong_hits >= 2 and right_hits < 3

    if key == "trojan":
        wrong_hits = sum(term in normalized for term in ransomware_terms)
        right_hits = sum(term in normalized for term in trojan_terms)
        return wrong_hits >= 2 and right_hits < 3

    return False


def _section_needs_topic_upgrade(section_name: str, content: str, topic: str = "") -> bool:
    """Detect weak or wrong-topic sections."""
    normalized = _normalize_whitespace(content).lower()

    if not normalized:
        return True

    if "missing information in available sources" in normalized:
        return True

    if _content_mentions_wrong_topic(content, topic):
        return True

    if section_name == "Attack Explanation":
        terms = _topic_section_terms("Attack Explanation", topic)
        return sum(1 for term in terms if term in normalized) < 3

    if section_name in {"IOCs", "Detection", "Mitigation"}:
        terms = _topic_section_terms(section_name, topic)
        bullet_count = len(_split_bullets(content))
        term_hits = sum(1 for term in terms if term in normalized)
        return bullet_count < 3 or term_hits < 3

    return False


def _recent_examples_need_upgrade(
    content: str,
    web_results: list[dict[str, Any]],
    *,
    topic: str = "",
) -> bool:
    """Return True when Recent Examples is generic, title-dumpy, wrong-topic, or weak."""
    normalized = _normalize_whitespace(content).lower()
    key = _topic_key(topic)

    if not normalized:
        return True

    # Hard wrong-topic protection
    if key == "ddos" and any(
        term in normalized
        for term in (
            "ransomware pressure",
            "ransomware operators",
            "ransomware activity",
            "encryption event",
            "ransom demand",
            "ransomware deployment",
            "data theft, not encryption",
        )
    ):
        return True

    if key == "brute_force" and any(
        term in normalized
        for term in (
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
        )
    ):
        return True

    if key == "trojan" and any(
        term in normalized
        for term in (
            "ransomware pressure",
            "ransomware operators",
            "shadow copy",
            "encryption event",
            "ransom-note",
        )
    ):
        return True

    if any(pattern in normalized for pattern in WEAK_TITLE_DUMP_PATTERNS):
        return True

    if web_results:
        terms = _topic_operational_terms(topic)

        if not any(term in normalized for term in terms):
            return True

        titles = [_normalize_whitespace(result.get("title", "")) for result in web_results]
        title_hits = sum(
            1 for line in _split_bullets(content)
            if _title_matches_known_result(line, titles)
        )

        if title_hits:
            return True

    return False


def validate_output(
    final_text: str,
    *,
    web_results: list[dict[str, Any]] | None = None,
    retrieved_docs: list[Any] | None = None,
    topic: str = "",
) -> tuple[str, dict[str, object]]:
    """Validate final output with targeted cleanup and topic-aware section repairs only."""
    web_results = web_results or []
    retrieved_docs = retrieved_docs or []
    defaults = _section_fallbacks(topic)

    raw_sections, header_counts, normalized_headers = _extract_sections(final_text)
    supported_entities = _build_supported_entity_set(retrieved_docs, web_results)

    duplicate_headers_removed = sorted(header for header, count in header_counts.items() if count > 1)
    sections_merged: list[str] = []
    missing_sections_filled: list[str] = []
    unsupported_entities_generalized: list[dict[str, str]] = []

    weak_iocs_removed: list[str] = []
    weak_title_dump_removed = False
    web_result_reference_leakage_removed = False
    iocs_strengthened = False
    recent_examples_strengthened = False
    attack_explanation_strengthened = False

    final_sections: dict[str, str] = {}

    for section in REQUIRED_SECTIONS:
        content, merged = _resolve_section_variants(section, raw_sections.get(section, []), topic=topic)

        if merged:
            sections_merged.append(section)

        if not content:
            content = defaults[section]
            missing_sections_filled.append(section)

        content, refs_removed = _strip_web_result_references(content)
        web_result_reference_leakage_removed = web_result_reference_leakage_removed or refs_removed

        if section == "Recent Examples":
            content, removed = _sanitize_recent_examples(content, web_results)
            weak_title_dump_removed = weak_title_dump_removed or removed

            if _recent_examples_need_upgrade(content, web_results, topic=topic):
                content = _build_recent_examples_from_web(web_results, topic=topic)
                recent_examples_strengthened = True

        content, replacements = _generalize_unsupported_entities(content, supported_entities)

        if replacements:
            unsupported_entities_generalized.extend(replacements)

        if section == "Attack Explanation" and _section_needs_topic_upgrade(section, content, topic):
            content = defaults["Attack Explanation"]
            attack_explanation_strengthened = True

        if section == "IOCs" and _section_needs_topic_upgrade(section, content, topic):
            content = defaults["IOCs"]
            iocs_strengthened = True

        if section in {"Detection", "Mitigation"} and _section_needs_topic_upgrade(section, content, topic):
            content = defaults[section]

        final_sections[section] = _dedupe_blocks(content) or defaults[section]

    final_answer = "\n\n".join(
        f"{section}\n{final_sections[section].strip()}"
        for section in REQUIRED_SECTIONS
    ).strip()

    applied_fixes: list[str] = []

    if duplicate_headers_removed:
        applied_fixes.append("duplicate_headers_removed")

    if sections_merged:
        applied_fixes.append("sections_merged")

    if normalized_headers:
        applied_fixes.append("normalized_headers")

    if missing_sections_filled:
        applied_fixes.append("missing_sections_filled")

    if weak_title_dump_removed:
        applied_fixes.append("weak_title_dump_removed")

    if web_result_reference_leakage_removed:
        applied_fixes.append("web_result_reference_leakage_removed")

    if unsupported_entities_generalized:
        applied_fixes.append("unsupported_entities_generalized")

    if weak_iocs_removed:
        applied_fixes.append("weak_iocs_removed")

    if iocs_strengthened:
        applied_fixes.append("iocs_strengthened")

    if recent_examples_strengthened:
        applied_fixes.append("recent_examples_strengthened")

    if attack_explanation_strengthened:
        applied_fixes.append("attack_explanation_strengthened")

    validation_report: dict[str, object] = {
        "applied_fixes": applied_fixes,
        "required_sections_present": all(f"{section}\n" in final_answer for section in REQUIRED_SECTIONS),
        "duplicate_headers_removed": duplicate_headers_removed,
        "sections_merged": sections_merged,
        "headers_normalized": normalized_headers,
        "missing_sections_filled": missing_sections_filled,
        "weak_title_dump_removed": weak_title_dump_removed,
        "web_result_reference_leakage_removed": web_result_reference_leakage_removed,
        "unsupported_entities_generalized": unsupported_entities_generalized,
        "weak_iocs_removed": weak_iocs_removed,
        "iocs_strengthened": iocs_strengthened,
        "recent_examples_strengthened": recent_examples_strengthened,
        "attack_explanation_strengthened": attack_explanation_strengthened,
    }

    return final_answer, validation_report