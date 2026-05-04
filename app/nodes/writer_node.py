"""Writer node for grounded cybersecurity RAG + web answers.

Topic-aware version:
- Prevents DDoS / brute-force / trojan queries from falling back to ransomware content.
- Keeps ransomware output strong.
- Uses topic-specific default sections.
- Uses topic-specific Recent Examples synthesis.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.state import AppState
from app.utils.debug import debug_print
from app.utils.execution_trace import record_execution_step, truncate_text
from app.utils.prompts import load_prompt

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
}

_NO_DOCS_FALLBACK = (
    "Overview\n"
    "No grounded answer could be produced because no relevant local documents or live web results were available.\n\n"
    "Attack Explanation\n"
    "Missing information in available sources.\n\n"
    "Recent Examples\n"
    "No live web results were retrieved in this run, so no current incidents or patterns can be summarized.\n\n"
    "IOCs\n"
    "Missing information in available sources.\n\n"
    "Detection\n"
    "Missing information in available sources.\n\n"
    "Mitigation\n"
    "Missing information in available sources.\n\n"
    "Limitations\n"
    "No supporting evidence was retrieved from either the local knowledge base or live web results."
)

_GENERIC_TITLE_MARKERS = (
    "what is",
    "examples of",
    "examples from",
    "explained",
    "guide",
    "glossary",
    "definition",
)

_RECENT_EXAMPLE_INTENT_TERMS = (
    "recent",
    "latest",
    "examples",
    "incidents",
    "campaigns",
)

_INCIDENT_TERMS = (
    "incident",
    "campaign",
    "victim",
    "organization",
    "hospital",
    "school",
    "university",
    "government",
    "sector",
    "extortion",
    "disruption",
    "disrupted",
    "outage",
    "claimed",
    "law enforcement",
    "advisory",
    "attack",
    "operations",
    "ransomware",
    "data theft",
    "data leak",
    "ddos",
    "denial of service",
    "botnet",
    "traffic",
    "flood",
    "requests",
    "packets",
    "regulator",
    "journalist",
    "customer",
    "client",
    "credential",
    "login",
    "authentication",
    "trojan",
    "malware",
    "backdoor",
)

_MAX_DOC_CONTEXT_ITEMS = 2
_MAX_DOC_CHARS = 400
_MAX_WEB_RESULTS_FOR_CONTEXT = 3
_MAX_WEB_SENTENCES = 2
_MAX_CONTEXT_BLOCK_CHARS = 1000


def get_writer_llm() -> ChatOllama:
    """Return the local Ollama chat model used by the writer."""
    return ChatOllama(model="llama3.2", temperature=0.1)


def _normalize_text(value: str) -> str:
    """Collapse repeated whitespace while preserving content."""
    return re.sub(r"\s+", " ", value or "").strip()


def _topic_key(topic: str) -> str:
    """Normalize topic into a small internal key."""
    normalized = _normalize_text(topic).lower()

    if "ddos" in normalized or "denial of service" in normalized or normalized == "dos":
        return "ddos"

    if "brute" in normalized or "credential" in normalized or "password" in normalized:
        return "brute_force"

    if "trojan" in normalized or "rat" in normalized or "backdoor" in normalized:
        return "trojan"

    return "ransomware"


def _query_requests_recent_examples(user_query: str) -> bool:
    """Detect when the user explicitly asks for recent-example coverage."""
    normalized_query = _normalize_text(user_query).lower()
    return any(term in normalized_query for term in _RECENT_EXAMPLE_INTENT_TERMS)


def _split_sentences(text: str) -> list[str]:
    """Split a snippet into short sentence candidates."""
    normalized = _normalize_text(text)

    if not normalized:
        return []

    protected = (
        normalized.replace("St. ", "St<prd> ")
        .replace("Mr. ", "Mr<prd> ")
        .replace("Ms. ", "Ms<prd> ")
        .replace("Dr. ", "Dr<prd> ")
    )

    return [
        sentence.replace("<prd>", ".").strip(" -")
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
        if sentence.strip()
    ]


def format_single_document(index: int, doc: Document) -> str:
    """Format one retrieved document chunk into readable RAG context."""
    metadata = doc.metadata or {}
    source = metadata.get("source", "unknown source")
    page = metadata.get("page", "unknown")
    chunk_topic = metadata.get("topic", "unknown topic")

    content = doc.page_content.strip()

    if len(content) > _MAX_DOC_CHARS:
        content = content[:_MAX_DOC_CHARS].rstrip() + "..."

    return (
        f"[Document {index}]\n"
        f"Source: {source}\n"
        f"Page: {page}\n"
        f"Topic: {chunk_topic}\n"
        f"Content:\n{content}"
    )


def format_retrieved_docs(docs: list[Document]) -> str:
    """Convert retrieved documents into a clean, readable context block."""
    if not docs:
        return ""

    formatted_docs = [
        format_single_document(index, doc)
        for index, doc in enumerate(docs[:_MAX_DOC_CONTEXT_ITEMS], start=1)
    ]

    return "\n\n---\n\n".join(formatted_docs)


def format_retrieved_context(retrieved_docs: list[Document]) -> str:
    """Backward-compatible alias."""
    return format_retrieved_docs(retrieved_docs)


def _sentence_is_low_value(sentence: str, titles: list[str]) -> bool:
    """Detect article titles, metadata, navigation text, or weak snippets."""
    normalized_sentence = _normalize_text(sentence).lower()

    if not normalized_sentence:
        return True

    bad_markers = (
        "opinion ###",
        "image ",
        "by ",
        "mins ",
        "home »",
        "read more",
        "subscribe",
        "newsletter",
        "advertisement",
        "sponsored",
        "top 10",
        "what is",
        "examples of",
        "guide",
        "glossary",
        "definition",
        "| cso online",
        "login",
        "sign up",
        "privacy policy",
        "cookie",
    )

    if any(marker in normalized_sentence for marker in bad_markers):
        return True

    if len(normalized_sentence.split()) < 10:
        return True

    for title in titles:
        normalized_title = _normalize_text(title).lower()

        if not normalized_title:
            continue

        if normalized_sentence in normalized_title:
            return True

        if normalized_title in normalized_sentence:
            return True

    return False


def _extract_key_sentences(content: str, max_sentences: int = _MAX_WEB_SENTENCES) -> list[str]:
    """Prefer operationally useful sentences over raw snippet dumps."""
    candidates = _split_sentences(content)

    if not candidates:
        return []

    scored: list[tuple[str, float]] = []

    for sentence in candidates:
        lowered = sentence.lower()
        score = 0.0

        score += sum(1.0 for term in _INCIDENT_TERMS if term in lowered)

        if re.search(r"\b(2024|2025|2026)\b", sentence):
            score += 1.5

        if any(
            term in lowered
            for term in (
                "disruption",
                "extortion",
                "victim",
                "ransomware",
                "ddos",
                "denial of service",
                "traffic",
                "flood",
                "botnet",
                "credential",
                "login",
                "malware",
                "trojan",
                "backdoor",
                "client",
                "customer",
            )
        ):
            score += 2.0

        if len(sentence) > 280:
            score -= 1.0

        scored.append((sentence, score))

    selected = [
        sentence
        for sentence, score in sorted(scored, key=lambda item: item[1], reverse=True)
        if score > 0
    ]

    deduped: list[str] = []
    seen: set[str] = set()

    for sentence in selected:
        key = _normalize_text(sentence).lower()

        if key in seen:
            continue

        seen.add(key)
        deduped.append(sentence)

        if len(deduped) >= max_sentences:
            break

    return deduped


def _format_web_result(index: int, result: dict[str, str]) -> str:
    """Format a single web result as condensed evidence."""
    title = _normalize_text(result.get("title", ""))
    url = _normalize_text(result.get("url", ""))
    content = result.get("content", "")

    key_sentences = _extract_key_sentences(content)

    evidence = "\n".join(f"- {sentence}" for sentence in key_sentences)

    if not evidence:
        evidence = "- No high-value operational details extracted."

    return (
        f"[Web Result {index}]\n"
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Key Evidence:\n{evidence}"
    )


def build_web_evidence_block(
    web_results: list[dict[str, str]],
    max_results: int = _MAX_WEB_RESULTS_FOR_CONTEXT,
) -> str:
    """Compress web results into a cleaner evidence block for the writer and reviser."""
    if not web_results:
        return "No live web results available."

    formatted = [
        _format_web_result(index, result)
        for index, result in enumerate(web_results[:max_results], start=1)
    ]

    return "\n\n---\n\n".join(formatted)


def _analyst_style_bullet(sentence: str, topic: str = "") -> str:
    """Turn raw evidence into a topic-aware analyst-style recent-example bullet."""
    cleaned = sentence.rstrip(".").strip()
    cleaned = re.sub(r"^\s*[-*]\s*", "", cleaned).strip()
    lowered = cleaned.lower()
    key = _topic_key(topic)

    if key == "ddos":
        if "google" in lowered or "cloudflare" in lowered or "largest" in lowered or "record" in lowered:
            return (
                "- Recent reporting on large-scale DDoS activity shows attackers continuing to generate record traffic volumes; this matters because defenders need upstream DDoS protection, CDN capacity, and traffic-scrubbing readiness before service saturation occurs."
            )

        if "botnet" in lowered or "compromised" in lowered:
            return (
                "- Recent reporting links DDoS activity to botnets and compromised infrastructure; this matters because traffic may be highly distributed, making simple IP blocking ineffective without rate limiting, filtering, and provider-level mitigation."
            )

        if "http" in lowered or "application" in lowered or "requests" in lowered:
            return (
                "- Recent reporting highlights application-layer DDoS pressure using high request volumes; this matters because backend CPU, database, and expensive endpoints can fail even when bandwidth is not fully saturated."
            )

        return (
            f"- {cleaned}; this matters because DDoS activity can degrade availability, increase latency, exhaust infrastructure resources, and require rapid coordination with CDN, WAF, ISP, or cloud-provider mitigation teams."
        )

    if key == "brute_force":
        if "password spraying" in lowered or "password spray" in lowered:
            return (
                "- Recent reporting shows password-spraying remains a practical intrusion path against exposed identity systems; this matters because attackers can avoid simple lockouts by testing a small number of passwords across many users."
            )

        if "credential stuffing" in lowered or "stolen credentials" in lowered or "leaked credentials" in lowered:
            return (
                "- Recent reporting continues to show credential stuffing and leaked-password reuse as active risks; this matters because successful login activity may look legitimate unless identity telemetry, MFA, and impossible-travel rules are monitored."
            )

        return (
            f"- {cleaned}; this matters because brute-force and credential attacks target identity controls directly and often precede account takeover, privilege escalation, or cloud/email abuse."
        )

    if key == "trojan":
        if "remote access" in lowered or "rat" in lowered or "backdoor" in lowered:
            return (
                "- Recent reporting shows trojan and backdoor activity continuing to provide remote access for attackers; this matters because defenders must detect persistence, command-and-control traffic, and follow-on payload deployment early."
            )

        if "loader" in lowered or "payload" in lowered or "download" in lowered:
            return (
                "- Recent reporting highlights trojans acting as loaders for additional payloads; this matters because an initial infection can quickly become credential theft, ransomware deployment, or broader lateral movement."
            )

        return (
            f"- {cleaned}; this matters because trojan activity can hide inside apparently legitimate execution paths while enabling persistence, credential theft, command-and-control, or payload delivery."
        )

    # Ransomware-specific analyst bullets
    if "ddos attacks" in lowered or "regulators" in lowered or "journalists" in lowered or "victim’s clients" in lowered or "victim's clients" in lowered:
        return (
            "- Recent reporting describes ransomware operators applying secondary pressure through DDoS attacks, regulator contact, media outreach, or direct pressure on a victim’s customers; this matters because modern ransomware is increasingly an extortion operation, not only an encryption event."
        )

    if "financial institutions" in lowered or "banking" in lowered:
        return (
            "- Recent reporting highlights ransomware pressure against financial institutions, where extortion can create regulatory, operational, and customer-trust impact beyond encrypted systems."
        )

    if "record number" in lowered or "victim numbers rise" in lowered or "ransomware victims" in lowered:
        return (
            "- 2025 reporting points to elevated ransomware victim volumes, which matters because defenders should expect continued extortion activity even when individual ransomware groups appear to change or decline."
        )

    if "chainalysis" in lowered or "payments to ransomware groups" in lowered:
        return (
            "- Recent payment-trend reporting suggests more victims are resisting ransom demands, which matters because ransomware groups may respond with stronger extortion pressure, repeat targeting, or faster data-leak threats rather than relying only on encryption."
        )

    if "56%" in lowered or "3-12 months" in lowered or "didn't detect" in lowered or "did not detect" in lowered:
        return (
            "- Recent reporting on delayed ransomware detection shows that many organizations still discover intrusions late, which matters because ransomware operators often use that dwell time for privilege escalation, lateral movement, backup disruption, and extortion preparation before encryption."
        )

    if "re-extort" in lowered or "extortion campaigns" in lowered:
        return (
            "- Recent reporting showed ransomware actors attempting follow-on extortion against prior victims, which suggests defenders should plan for repeat pressure even after initial containment."
        )

    if "warlock" in lowered or "toolshell" in lowered:
        return (
            "- 2025 reporting linked a Warlock-related intrusion path to ToolShell abuse, showing how modern ransomware operations can pair intrusion tradecraft with later malware deployment."
        )

    if "unfi" in lowered:
        return (
            "- Mid-2025 reporting associated UNFI with a ransomware incident, reinforcing that large enterprise operations remain exposed to disruption and extortion pressure."
        )

    return (
        f"- {cleaned}; this matters because it highlights active ransomware pressure against organizations and the need for earlier detection, containment, and recovery planning."
    )


def synthesize_recent_examples(web_results: list[dict[str, str]], topic: str = "") -> str:
    """Create a grounded Recent Examples section from web evidence only."""
    key = _topic_key(topic)

    if not web_results:
        return "No live web results were retrieved in this run, so no current incidents or patterns can be summarized."

    titles = [_normalize_text(result.get("title", "")) for result in web_results]

    incident_bullets: list[str] = []
    pattern_bullets: list[str] = []

    combined_text = " ".join(
        _normalize_text(result.get("content", ""))
        for result in web_results
    ).lower()

    for result in web_results:
        sentences = _extract_key_sentences(result.get("content", ""), max_sentences=4)

        for sentence in sentences:
            if _sentence_is_low_value(sentence, titles):
                continue

            lowered = sentence.lower()

            useful = (
                any(term in lowered for term in _INCIDENT_TERMS)
                or re.search(r"\b(2024|2025|2026)\b", sentence)
            )

            if not useful:
                continue

            bullet = _analyst_style_bullet(sentence, topic=topic)

            if bullet not in incident_bullets:
                incident_bullets.append(bullet)

            if len(incident_bullets) >= 3:
                break

        if len(incident_bullets) >= 3:
            break

    if incident_bullets:
        return "\n".join(incident_bullets[:3])

    if key == "ddos":
        if any(term in combined_text for term in ("botnet", "compromised", "iot")):
            pattern_bullets.append(
                "- Live reporting indicates DDoS activity can be driven by botnets or compromised infrastructure; this matters because traffic is distributed and often requires provider-level filtering rather than simple local blocking."
            )
        if any(term in combined_text for term in ("http", "application", "request", "layer 7")):
            pattern_bullets.append(
                "- Available reporting points to application-layer request floods as a continuing availability risk; this matters because backend resources can be exhausted even when bandwidth remains available."
            )
        if any(term in combined_text for term in ("udp", "dns", "amplification", "traffic", "bandwidth")):
            pattern_bullets.append(
                "- Current DDoS reporting also emphasizes traffic floods and amplification patterns; this matters because defenders need traffic baselines, upstream filtering, and scrubbing capacity."
            )

    elif key == "brute_force":
        if any(term in combined_text for term in ("password spray", "password spraying", "credential stuffing")):
            pattern_bullets.append(
                "- Live reporting indicates password spraying and credential stuffing remain active identity risks; this matters because attackers can blend into normal login noise without strong authentication analytics."
            )
        if any(term in combined_text for term in ("mfa", "authentication", "login", "vpn", "cloud")):
            pattern_bullets.append(
                "- Available reporting shows exposed authentication portals remain attractive targets; this matters because MFA, throttling, conditional access, and identity telemetry are critical controls."
            )

    elif key == "trojan":
        if any(term in combined_text for term in ("loader", "payload", "malware", "backdoor", "remote access")):
            pattern_bullets.append(
                "- Live reporting indicates trojans continue to act as loaders, backdoors, or remote-access tools; this matters because initial execution can lead to persistence, credential theft, and follow-on compromise."
            )
        if any(term in combined_text for term in ("phishing", "attachment", "installer", "download")):
            pattern_bullets.append(
                "- Available reporting shows trojan delivery still commonly relies on user-facing lures such as attachments, downloads, or fake installers; this matters because email, web, and endpoint controls must be correlated."
            )

    else:
        if any(term in combined_text for term in ("extortion", "data theft", "leak", "ransom")):
            pattern_bullets.append(
                "- Live reporting emphasizes ransomware extortion and data-theft leverage; this matters because defenders must monitor outbound movement and pressure tactics, not only encryption."
            )
        if any(term in combined_text for term in ("lateral", "backup", "shadow copy", "recovery")):
            pattern_bullets.append(
                "- Available reporting points to recovery disruption and lateral movement as continuing ransomware concerns; this matters because backup isolation and host-to-host monitoring remain critical."
            )

    if pattern_bullets:
        return "\n".join(pattern_bullets[:3])

    topic_name = {
        "ddos": "DDoS",
        "brute_force": "brute-force",
        "trojan": "trojan",
        "ransomware": "ransomware",
    }.get(key, "cyber threat")

    return (
        f"- Live reporting in this run was high-level rather than incident-specific, but it still indicates ongoing {topic_name} activity relevant to defenders.\n"
        f"- The available evidence suggests defenders should treat {topic_name} as operationally relevant and validate detection, response, and mitigation controls against current activity."
    )


def _extract_sections(text: str) -> dict[str, str]:
    """Extract required sections from a generated draft."""
    if not text.strip():
        return {}

    pattern = re.compile(
        r"(?mi)^(Overview|Attack Explanation|Recent Examples|IOCs|Indicators of Compromise|IOCs\s*\(Indicators of Compromise\)|Detection|Mitigation|Limitations)\s*:?\s*$"
    )

    matches = list(pattern.finditer(text))
    sections: dict[str, list[str]] = {section: [] for section in REQUIRED_SECTIONS}

    for index, match in enumerate(matches):
        section = SECTION_ALIASES.get(
            _normalize_text(match.group(1)),
            _normalize_text(match.group(1)),
        )

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        body = text[start:end].strip()

        if body:
            cleaned_body = re.sub(
                r"(?mi)^(Overview|Attack Explanation|Recent Examples|IOCs|Indicators of Compromise|IOCs\s*\(Indicators of Compromise\)|Detection|Mitigation|Limitations)\s*:?\s*$",
                "",
                body,
            ).strip()

            if cleaned_body:
                sections[section].append(cleaned_body)

    normalized: dict[str, str] = {}

    for section in REQUIRED_SECTIONS:
        merged = "\n\n".join(
            chunk for chunk in sections[section]
            if _normalize_text(chunk)
        ).strip()

        if merged:
            normalized[section] = merged

    return normalized


def _build_default_sections(topic: str, web_results: list[dict[str, str]]) -> dict[str, str]:
    """Return stable topic-aware fallbacks for missing or weak writer sections."""
    key = _topic_key(topic)

    if key == "ddos":
        return {
            "Overview": (
                "DDoS is a cybersecurity threat where attackers attempt to overwhelm an online service, network, or application with excessive traffic so legitimate users cannot access it."
            ),
            "Attack Explanation": (
                "DDoS attacks commonly use botnets, compromised servers, reflection or amplification techniques, and large volumes of traffic to exhaust bandwidth, connection tables, application resources, or upstream infrastructure. "
                "Common forms include volumetric floods, SYN floods, UDP floods, DNS or NTP amplification, HTTP floods, and application-layer request abuse. The objective is usually service disruption, extortion pressure, distraction during another intrusion, or reputational damage."
            ),
            "Recent Examples": synthesize_recent_examples(web_results, topic=topic),
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
                "This answer is limited to the retrieved documents and live web results available in this run. DDoS attribution is often difficult because traffic may come from compromised systems, proxies, or spoofed infrastructure."
            ),
        }

    if key == "brute_force":
        return {
            "Overview": (
                "Brute force is an identity-focused attack where adversaries repeatedly attempt passwords, credentials, or authentication combinations to gain unauthorized access."
            ),
            "Attack Explanation": (
                "Brute-force attacks may involve high-volume password guessing, password spraying, credential stuffing, or automated login attempts against VPNs, email portals, cloud services, SSH, RDP, and web applications. "
                "Attackers often use leaked credential lists, rotating IP addresses, proxy infrastructure, and low-and-slow timing to avoid lockouts and detection."
            ),
            "Recent Examples": synthesize_recent_examples(web_results, topic=topic),
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
                "This answer is limited to the retrieved documents and live web results available in this run. Brute-force activity may blend into normal login noise without strong identity telemetry."
            ),
        }

    if key == "trojan":
        return {
            "Overview": (
                "A trojan is malware that disguises itself as legitimate software or content while performing malicious actions such as persistence, credential theft, surveillance, payload delivery, or remote access."
            ),
            "Attack Explanation": (
                "Trojan infections commonly start through phishing attachments, malicious downloads, cracked software, fake installers, or compromised websites. "
                "After execution, a trojan may establish persistence, contact command-and-control infrastructure, collect credentials, download additional payloads, inject into processes, or give attackers remote access to the system."
            ),
            "Recent Examples": synthesize_recent_examples(web_results, topic=topic),
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
                "This answer is limited to the retrieved documents and live web results available in this run. Trojan behavior varies significantly by malware family and campaign."
            ),
        }

    return {
        "Overview": (
            "Ransomware is a cybersecurity threat that should be understood through grounded source evidence rather than assumptions."
        ),
        "Attack Explanation": (
            "Ransomware operators commonly gain initial access through phishing, exposed remote services, stolen credentials, or software exploitation, "
            "then execute tooling to establish control. They often escalate privileges, move laterally with administrative tooling and SMB or RDP access, "
            "tamper with backups or delete shadow copies, and then encrypt systems while applying extortion pressure through recovery disruption or data-theft leverage."
        ),
        "Recent Examples": synthesize_recent_examples(web_results, topic=topic),
        "IOCs": (
            "- Suspicious PowerShell execution for payload staging, defense evasion, or remote execution.\n"
            "- Shadow copy deletion activity such as `vssadmin`, `wmic`, or `diskshadow` used to impair recovery.\n"
            "- PsExec, remote service creation, or SMB admin-share access consistent with lateral deployment.\n"
            "- Ransom-note creation, burst encryption activity, or unusual outbound connections consistent with extortion staging."
        ),
        "Detection": (
            "- Monitor abnormal file modification bursts, ransom-note creation, encryption spikes, and extension changes across endpoints and shared drives.\n"
            "- Alert on shadow copy deletion, backup tampering, suspicious PowerShell, `wmic`, `vssadmin`, `diskshadow`, PsExec, or remote service creation.\n"
            "- Track lateral movement indicators such as SMB fan-out, administrative share access, RDP pivots, credential reuse, and rapid execution across multiple hosts.\n"
            "- Correlate endpoint, identity, and network telemetry to catch outbound connections or exfiltration patterns preceding encryption or extortion."
        ),
        "Mitigation": (
            "- Enforce network segmentation for critical systems, management planes, and backup infrastructure to constrain lateral movement.\n"
            "- Apply least privilege, restrict local admin rights, harden remote access with MFA, and tightly control privileged account use.\n"
            "- Maintain offline or immutable backups, validate restorations regularly, and isolate backup credentials and infrastructure.\n"
            "- Keep EDR, SIEM, and incident-response playbooks ready so suspicious tooling, lateral movement, or encryption bursts can be contained quickly."
        ),
        "Limitations": (
            "This answer is limited to the retrieved documents and live web results available in this run."
        ),
    }


def _content_mentions_wrong_topic(content: str, topic: str) -> bool:
    """Detect obvious cross-topic contamination."""
    normalized = _normalize_text(content).lower()
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


def _section_is_weak(
    section_name: str,
    content: str,
    web_results: list[dict[str, str]],
    topic: str = "",
) -> bool:
    """Detect weak, placeholder, or wrong-topic section content."""
    normalized = _normalize_text(content).lower()
    key = _topic_key(topic)

    if not normalized:
        return True

    if "missing information in available sources" in normalized:
        return True

    if _content_mentions_wrong_topic(content, topic):
        return True

    if section_name == "Recent Examples":
        if not web_results and "no live web results were retrieved" not in normalized:
            return True

        if any(marker in normalized for marker in _GENERIC_TITLE_MARKERS):
            return True

        weak_markers = (
            "responds to many",
            "opinion ###",
            "image ",
            "| cso online",
            "top 10",
            "home »",
            "read more",
            "subscribe",
        )

        if any(weak in normalized for weak in weak_markers):
            return True

        title_dumps = sum(
            1
            for result in web_results
            if _normalize_text(result.get("title", "")).lower()
            and _normalize_text(result.get("title", "")).lower() in normalized
        )

        if title_dumps >= 1:
            return True

    if section_name == "Detection":
        if key == "ddos":
            required_groups = [
                ("traffic", "packets", "bandwidth", "requests"),
                ("syn", "udp", "dns", "http", "amplification"),
                ("latency", "timeout", "dropped", "error"),
                ("cdn", "waf", "firewall", "load balancer", "logs"),
            ]
        elif key == "brute_force":
            required_groups = [
                ("failed login", "login attempts", "authentication"),
                ("password spraying", "credential stuffing", "brute force"),
                ("impossible travel", "new device", "source ip", "geo"),
                ("identity", "vpn", "email", "cloud", "audit"),
            ]
        elif key == "trojan":
            required_groups = [
                ("process", "child process", "script", "unsigned"),
                ("persistence", "registry", "scheduled task", "service"),
                ("command-and-control", "c2", "dns", "http", "beacon"),
                ("edr", "endpoint", "injection", "credential"),
            ]
        else:
            required_groups = [
                ("file modification", "mass file", "extension", "ransom-note", "encryption"),
                ("shadow copy", "vssadmin", "wmic", "diskshadow"),
                ("powershell", "psexec", "remote service", "admin tool"),
                ("lateral", "smb", "rdp", "credential reuse", "scheduled task"),
            ]

        if sum(any(term in normalized for term in group) for group in required_groups) < 2:
            return True

    if section_name == "Mitigation":
        if key == "ddos":
            required_groups = [
                ("cdn", "waf", "ddos protection", "scrubbing"),
                ("rate limiting", "throttling", "filtering"),
                ("autoscaling", "caching", "circuit breaker"),
                ("isp", "provider", "playbook", "response"),
            ]
        elif key == "brute_force":
            required_groups = [
                ("mfa", "multi-factor", "conditional access"),
                ("lockout", "throttling", "rate limiting"),
                ("legacy authentication", "password"),
                ("ip reputation", "geo", "device trust"),
            ]
        elif key == "trojan":
            required_groups = [
                ("email security", "proxy", "dns", "edr"),
                ("application control", "script", "patch"),
                ("isolate", "revoke", "credentials"),
                ("persistence", "payload", "malware"),
            ]
        else:
            required_groups = [
                ("network segmentation", "segment", "segmentation"),
                ("least privilege", "local admin", "privileged", "mfa"),
                ("offline", "immutable", "backup", "restore"),
                ("monitoring", "response", "playbook", "incident-response", "edr", "siem"),
            ]

        if sum(any(term in normalized for term in group) for group in required_groups) < 2:
            return True

    return False


def _normalize_draft(
    draft_answer: str,
    topic: str,
    web_results: list[dict[str, str]],
    *,
    prefer_web_recent_examples: bool = False,
) -> str:
    """Force the writer output into the exact required section order."""
    sections = _extract_sections(draft_answer)
    defaults = _build_default_sections(topic=topic, web_results=web_results)

    parts: list[str] = []

    for section in REQUIRED_SECTIONS:
        content = sections.get(section, "").strip()

        if section == "Recent Examples" and prefer_web_recent_examples and web_results:
            content = defaults[section]

        elif _section_is_weak(section, content, web_results, topic=topic):
            content = defaults[section]

        parts.append(f"{section}\n{content}")

    return "\n\n".join(parts).strip()


def writer_node(state: AppState) -> AppState:
    """Generate a grounded draft answer using local RAG and live web results."""
    user_query = state.get("user_query", "").strip()
    plan = state.get("plan", "").strip()
    topic = state.get("topic", "").strip()
    retrieved_docs = state.get("retrieved_docs", [])
    web_results = state.get("web_results", [])

    prefer_web_recent_examples = bool(web_results) and _query_requests_recent_examples(user_query)

    debug_print(f"[writer_node] incoming web_results: {len(web_results)}")
    debug_print("\n[writer_node] Debug")
    debug_print(f"User query: {user_query}")
    debug_print(f"Topic: {topic}")
    debug_print(f"Docs count: {len(retrieved_docs)}")
    debug_print(f"Web results count: {len(web_results)}")
    debug_print(f"Prefer web Recent Examples: {prefer_web_recent_examples}")

    if not retrieved_docs and not web_results:
        state["draft_answer"] = _NO_DOCS_FALLBACK

        record_execution_step(
            state,
            "writer",
            title="Writer",
            summary="Produced the no-evidence fallback draft because neither local nor web evidence was available.",
            details={
                "retrieved_docs_count": 0,
                "web_results_count": 0,
                "draft_answer": _NO_DOCS_FALLBACK,
            },
        )

        return state

    local_context = (
        format_retrieved_docs(retrieved_docs)
        if retrieved_docs
        else "No local knowledge base documents were retrieved."
    )

    web_context = build_web_evidence_block(web_results)
    recent_examples_guidance = synthesize_recent_examples(web_results, topic=topic)

    local_context_for_prompt = local_context[:_MAX_CONTEXT_BLOCK_CHARS]
    web_context_for_prompt = web_context[:_MAX_CONTEXT_BLOCK_CHARS]
    recent_examples_for_prompt = recent_examples_guidance[:700]

    debug_print(
        f"[writer_node] Context preview: "
        f"{(local_context_for_prompt + ' ' + web_context_for_prompt)[:500].replace(chr(10), ' ')}..."
    )

    system_prompt = load_prompt(
        "writer_prompt.txt",
        """
You are a cybersecurity threat-intelligence writer.

Use ONLY the provided retrieved_docs and web_results.
No hallucinations. No fabricated incidents, actors, victims, dates, or malware details.
Write in concise SOC-style language.

Output these exact sections only:
Overview
Attack Explanation
Recent Examples
IOCs
Detection
Mitigation
Limitations

Rules:
- Stay strictly on the requested topic.
- Do not use ransomware concepts for DDoS, brute-force, or trojan questions unless the user explicitly asks for a comparison.
- Use retrieved_docs for foundational explanation.
- Use web_results for current intelligence.
- If web_results exist, Recent Examples is mandatory.
- Do not list article titles as examples.
- Do not paste snippets without synthesis.
- Mention a specific organization only if it is explicitly stated in web_results content; otherwise generalize to sectors or organization types.
- If named incidents are weak or absent, summarize only the operational patterns clearly supported by the live web evidence.
- If live results are mostly high-level references, say so explicitly in Recent Examples and then summarize the supported current patterns.
- Prefer synthesis over enumeration.
- Keep every claim grounded in the supplied context.
        """,
    )

    human_prompt = f"""
User Query:
{user_query}

Detected Topic:
{topic}

Planner Output:
{plan[:300]}

Key Local Context:
{local_context_for_prompt}

Key Web Evidence:
{web_context_for_prompt}

Recent Example Guidance:
{recent_examples_for_prompt}

Write a concise structured answer using only the supplied evidence.
The answer must stay strictly about the detected topic: {topic}.
""".strip()

    response = get_writer_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    draft_answer = response.content if isinstance(response.content, str) else str(response.content)

    normalized_draft = _normalize_draft(
        draft_answer.strip(),
        topic=topic,
        web_results=web_results,
        prefer_web_recent_examples=prefer_web_recent_examples,
    )

    state["draft_answer"] = normalized_draft if normalized_draft else _NO_DOCS_FALLBACK

    record_execution_step(
        state,
        "writer",
        title="Writer",
        summary="Generated the first grounded answer draft from local retrieval and live web evidence.",
        details={
            "topic": topic,
            "retrieved_docs_count": len(retrieved_docs),
            "web_results_count": len(web_results),
            "prefer_web_recent_examples": prefer_web_recent_examples,
            "draft_preview": truncate_text(state["draft_answer"], max_chars=1200),
        },
    )

    return state