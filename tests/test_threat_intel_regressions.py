"""Focused regression coverage for threat-intelligence synthesis behavior."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.nodes.critic_node import _draft_is_strong_enough
from app.nodes.reviser_node import repair_recent_examples, validate_output
from app.nodes.web_search_node import _build_search_queries, _select_best_results
from app.nodes.writer_node import synthesize_recent_examples
from app.utils.app_mode import is_demo_mode
from app.utils.output_validator import validate_output as validate_output_with_report


class ThreatIntelRegressionTests(unittest.TestCase):
    """Cover the recent-examples and web-ranking regressions."""

    def test_query_builder_biases_toward_incident_search(self) -> None:
        queries = _build_search_queries(
            user_query="Explain ransomware and include recent examples",
            topic="Ransomware",
        )

        self.assertLessEqual(len(queries), 2)
        self.assertTrue(any("recent ransomware incidents" in query for query in queries))
        self.assertTrue(any("victims" in query for query in queries))
        self.assertTrue(all("Explain ransomware and include recent examples" not in query for query in queries))

    def test_demo_mode_only_activates_when_env_is_demo(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APP_MODE", None)
            self.assertFalse(is_demo_mode())

        with patch.dict(os.environ, {"APP_MODE": "demo"}, clear=False):
            self.assertTrue(is_demo_mode())

    def test_result_filtering_penalizes_generic_listicles(self) -> None:
        results = [
            {
                "title": "15 Examples of recent Ransomware Attacks",
                "url": "https://example.com/what-is-ransomware-explained",
                "content": "This guide explains what ransomware is and gives beginner examples explained for 2026.",
            },
            {
                "title": "Hospital ransomware incident disrupts imaging services in 2025",
                "url": "https://example.org/ransomware-hospital-incident-2025",
                "content": "A ransomware incident hit a hospital, caused operational disruption, and triggered law enforcement advisory activity.",
            },
        ]

        filtered = _select_best_results(results, max_results=2)

        self.assertEqual(filtered[0]["title"], "Hospital ransomware incident disrupts imaging services in 2025")

    def test_generic_web_results_become_pattern_based_recent_examples(self) -> None:
        web_results = [
            {
                "title": "15 Examples of recent Ransomware Attacks",
                "url": "https://example.com/examples",
                "content": (
                    "Recent ransomware reporting highlights phishing as an initial access vector, abuse of exposed RDP services, "
                    "and double extortion against healthcare and education organizations."
                ),
            },
            {
                "title": "26 Ransomware Examples Explained in 2026",
                "url": "https://example.com/explained",
                "content": (
                    "The reporting notes lateral movement before encryption, data theft for extortion pressure, "
                    "and disruption across hospitals and schools."
                ),
            },
        ]

        recent_examples = synthesize_recent_examples(web_results)

        self.assertNotIn("15 Examples of recent Ransomware Attacks", recent_examples)
        self.assertNotIn("26 Ransomware Examples Explained in 2026", recent_examples)
        self.assertTrue("phishing" in recent_examples.lower() or "remote access" in recent_examples.lower())
        self.assertIn("extortion", recent_examples.lower())

    def test_incident_rich_web_results_keep_supported_named_details(self) -> None:
        web_results = [
            {
                "title": "Akira campaign disrupts St. Margaret Hospital",
                "url": "https://example.org/akira-hospital",
                "content": (
                    "In 2025, Akira ransomware disrupted St. Margaret Hospital, delaying imaging and outpatient services while "
                    "the victim organization worked on restoration."
                ),
            },
            {
                "title": "LockBit claim linked to manufacturing outage",
                "url": "https://example.org/lockbit-manufacturer",
                "content": (
                    "LockBit operators claimed an attack against a regional manufacturer in 2025, with extortion pressure and "
                    "production disruption described in the incident reporting."
                ),
            },
        ]

        recent_examples = synthesize_recent_examples(web_results)

        self.assertIn("Akira", recent_examples)
        self.assertIn("St. Margaret Hospital", recent_examples)
        self.assertIn("LockBit", recent_examples)
        self.assertNotIn("Akira campaign disrupts St. Margaret Hospital", recent_examples)

    def test_missing_web_results_do_not_hallucinate(self) -> None:
        recent_examples = synthesize_recent_examples([])
        self.assertIn("No live web results were retrieved", recent_examples)

    def test_strong_draft_quality_gate_allows_fast_path(self) -> None:
        draft = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\n"
            "Initial access can occur through phishing, exposed services, stolen credentials, or vulnerable public-facing systems. "
            "Operators execute payloads and tooling, escalate privileges, move laterally with SMB, RDP, PsExec, or remote services, "
            "delete shadow copies with vssadmin or wmic, and then encrypt systems while applying extortion pressure and data-theft leverage.\n\n"
            "Recent Examples\n"
            "- In 2025, recent reporting described disruption against a healthcare organization and showed how extortion followed operational impact. [Web Result 1]\n\n"
            "IOCs\n"
            "- Suspicious PowerShell execution used to stage payloads or disable defenses.\n"
            "- vssadmin or wmic activity tied to shadow copy deletion.\n"
            "- PsExec or SMB-based remote execution preceding ransomware rollout.\n"
            "- Unusual outbound connections consistent with extortion staging.\n\n"
            "Detection\n"
            "- Monitor endpoint process execution, suspicious PowerShell, and shadow copy deletion activity.\n"
            "- Track file-system anomalies, ransom-note creation, SMB spread, and RDP pivots.\n\n"
            "Mitigation\n"
            "- Enforce segmentation, least privilege, MFA, offline or immutable backups, and incident-response readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        self.assertTrue(_draft_is_strong_enough(draft, web_results=[{"title": "Incident", "url": "https://example.org", "content": "Healthcare organization disruption and extortion."}]))

    def test_reviser_preserves_existing_good_recent_examples(self) -> None:
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n"
            "- In 2025, Akira ransomware disrupted a regional hospital's imaging operations and recovery workflow. [Web Result 1]\n"
            "- Recent reporting also describes extortion pressure against a manufacturer after operational disruption. [Web Result 2]\n\n"
            "IOCs\nGrounded IOCs.\n\n"
            "Detection\nGrounded detection.\n\n"
            "Mitigation\nGrounded mitigation.\n\n"
            "Limitations\nGrounded limitations."
        )
        web_results = [
            {
                "title": "Akira campaign disrupts St. Margaret Hospital",
                "url": "https://example.org/akira-hospital",
                "content": "Akira ransomware disrupted hospital services in 2025.",
            },
            {
                "title": "LockBit claim linked to manufacturing outage",
                "url": "https://example.org/lockbit-manufacturer",
                "content": "Extortion pressure followed the outage.",
            },
        ]

        repaired = repair_recent_examples(answer, web_results)

        self.assertEqual(repaired, answer)

    def test_validation_removes_duplicate_ioc_headers_and_keeps_single_schema(self) -> None:
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Sector-focused pattern summary. [Web Result 1]\n\n"
            "IOCs (Indicators of Compromise)\n- First IOC.\n\n"
            "IOCs\n- Second IOC.\n\n"
            "Detection\n- Monitor abnormal file modification patterns, shadow copy deletion, and PowerShell usage.\n"
            "- Track lateral movement over SMB and RDP.\n\n"
            "Mitigation\n- Enforce network segmentation, least privilege, offline backups, and monitoring readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        validated = validate_output(answer, web_results=[])

        self.assertEqual(validated.count("\nIOCs\n"), 1)
        self.assertNotIn("IOCs (Indicators of Compromise)", validated)

    def test_validation_replaces_unsupported_named_organizations_with_grounded_patterns(self) -> None:
        web_results = [
            {
                "title": "Financial sector ransomware activity rises",
                "url": "https://example.org/financial-sector",
                "content": (
                    "Recent ransomware reporting describes pressure on financial institutions, shadow copy deletion, "
                    "and lateral movement before encryption. It does not name specific banks."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Bank of America and Wells Fargo were named in recent ransomware activity. [Web Result 1]\n\n"
            "IOCs\n- Grounded IOC.\n\n"
            "Detection\n- update antivirus\n\n"
            "Mitigation\n- use strong passwords\n\n"
            "Limitations\nGrounded limitations."
        )

        validated = validate_output(answer, web_results=web_results)

        self.assertNotIn("Bank of America", validated)
        self.assertNotIn("Wells Fargo", validated)
        self.assertIn("financial institutions", validated.lower())
        self.assertIn("shadow copy", validated.lower())
        self.assertIn("network segmentation", validated.lower())

    def test_validation_preserves_explicitly_supported_named_organizations(self) -> None:
        web_results = [
            {
                "title": "Akira disrupts St. Margaret Hospital",
                "url": "https://example.org/st-margaret",
                "content": (
                    "Akira ransomware disrupted St. Margaret Hospital in 2025 and forced restoration work after service outages."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Akira disrupted St. Margaret Hospital in 2025 and affected clinical operations. [Web Result 1]\n\n"
            "IOCs\n- Grounded IOC.\n\n"
            "Detection\n- Monitor abnormal file modification patterns, shadow copy deletion, suspicious PowerShell usage, and lateral movement over SMB.\n\n"
            "Mitigation\n- Enforce network segmentation, least privilege, offline backups, and monitoring readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        validated = validate_output(answer, web_results=web_results)

        self.assertIn("St. Margaret Hospital", validated)
        self.assertIn("Akira", validated)

    def test_output_validator_generalizes_only_probable_victim_organizations(self) -> None:
        web_results = [
            {
                "title": "Healthcare sector ransomware reporting",
                "url": "https://example.org/healthcare-sector",
                "content": (
                    "Recent reporting describes ransomware pressure against healthcare organizations without naming a specific victim. "
                    "Detection guidance highlights suspicious PowerShell, PsExec, WMIC, vssadmin, diskshadow, EDR coverage, SIEM telemetry, "
                    "SMB lateral movement, RDP abuse, ransom notes, hashes, and malicious domains."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Example Regional Hospital was impacted in a recent ransomware incident. [Web Result 1]\n\n"
            "IOCs\n"
            "- Suspicious PowerShell execution\n"
            "- WMIC or PsExec activity\n"
            "- vssadmin and diskshadow usage\n"
            "- SHA256 hashes tied to payload staging\n"
            "- Malicious domains used for command-and-control\n\n"
            "Detection\n"
            "- Alert on Suspicious PowerShell, WMIC, PsExec, vssadmin, diskshadow, SMB fan-out, RDP pivots, ransom notes, hashes, and domains.\n\n"
            "Mitigation\n"
            "- Ensure EDR and SIEM coverage is in place, segment SMB and RDP exposure, maintain offline backups, and enforce least privilege.\n\n"
            "Limitations\nGrounded limitations."
        )

        final_answer, report = validate_output_with_report(answer, web_results=web_results)

        self.assertNotIn("Example Regional Hospital", final_answer)
        self.assertIn("a healthcare organization", final_answer.lower())
        self.assertIn("Suspicious PowerShell", final_answer)
        self.assertIn("Ensure EDR", final_answer)
        self.assertIn("WMIC", final_answer)
        self.assertIn("PsExec", final_answer)
        self.assertIn("vssadmin", final_answer)
        self.assertIn("diskshadow", final_answer)
        self.assertIn("SIEM", final_answer)
        self.assertIn("SMB", final_answer)
        self.assertIn("RDP", final_answer)
        self.assertIn("ransom notes", final_answer)
        self.assertIn("hashes", final_answer)
        self.assertIn("domains", final_answer)
        self.assertEqual(
            report["unsupported_entities_generalized"],
            [{"entity": "Example Regional Hospital", "replacement": "a healthcare organization"}],
        )

    def test_validation_rebuilds_missing_schema_sections(self) -> None:
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Recent Examples\n- Pattern summary grounded in live reporting. [Web Result 1]\n\n"
            "Mitigation\n- Enforce network segmentation, least privilege, offline backups, and monitoring readiness."
        )

        validated = validate_output(answer, web_results=[])

        for section in [
            "Overview",
            "Attack Explanation",
            "Recent Examples",
            "IOCs",
            "Detection",
            "Mitigation",
            "Limitations",
        ]:
            self.assertIn(f"{section}\n", validated)

    def test_validation_removes_examples_of_title_dump_from_recent_examples(self) -> None:
        web_results = [
            {
                "title": "15 Examples of recent Ransomware Attacks",
                "url": "https://example.org/examples",
                "content": (
                    "Recent reporting highlights phishing, shadow copy deletion, and lateral movement across financial institutions."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n15 Examples of recent Ransomware Attacks\n\n"
            "IOCs\n- Grounded IOC.\n\n"
            "Detection\n- update antivirus\n\n"
            "Mitigation\n- use strong passwords\n\n"
            "Limitations\nGrounded limitations."
        )

        validated = validate_output(answer, web_results=web_results)

        self.assertNotIn("15 Examples of recent Ransomware Attacks", validated)
        self.assertIn("financial institutions", validated.lower())
        self.assertIn("shadow copy", validated.lower())

    def test_output_validator_merges_duplicate_sections_and_reports_it(self) -> None:
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Pattern summary.\n\n"
            "IOCs\n- Suspicious PowerShell execution.\n\n"
            "Detection:\n- Monitor process execution.\n\n"
            "Mitigation\n- Keep backups.\n\n"
            "Detection\n"
            "- Monitor abnormal file modification patterns, shadow copy deletion, suspicious PowerShell, and SMB fan-out.\n"
            "- Alert on PsExec, remote service creation, and rapid multi-host execution.\n\n"
            "Mitigation:\n"
            "- Enforce network segmentation, least privilege, offline backups, and incident-response readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        final_answer, report = validate_output_with_report(answer, web_results=[])

        self.assertEqual(final_answer.count("\nDetection\n"), 1)
        self.assertEqual(final_answer.count("\nMitigation\n"), 1)
        self.assertIn("shadow copy deletion", final_answer.lower())
        self.assertIn("network segmentation", final_answer.lower())
        self.assertIn("Detection", report["duplicate_headers_removed"])
        self.assertIn("Mitigation", report["duplicate_headers_removed"])
        self.assertIn("Detection", report["sections_merged"])
        self.assertIn("Mitigation", report["sections_merged"])

    def test_output_validator_removes_web_refs_and_strengthens_iocs(self) -> None:
        web_results = [
            {
                "title": "Ransomware activity shows shadow copy deletion and lateral movement",
                "url": "https://example.org/ransomware-patterns",
                "content": (
                    "Recent reporting highlights suspicious PowerShell execution, vssadmin delete shadows, SMB lateral movement, "
                    "and unusual outbound traffic before extortion."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nGrounded explanation.\n\n"
            "Recent Examples\n- Reporting highlighted current ransomware disruption patterns. [Web Result 1]\n\n"
            "IOCs\n"
            "- phishing emails\n"
            "- generic file extensions\n"
            "- payment requests from unknown senders\n\n"
            "Detection\n- Monitor abnormal file modification patterns, shadow copy deletion, suspicious PowerShell usage, and lateral movement over SMB.\n\n"
            "Mitigation\n- Enforce network segmentation, least privilege, offline backups, and monitoring readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        final_answer, report = validate_output_with_report(answer, web_results=web_results)

        self.assertNotIn("[Web Result 1]", final_answer)
        self.assertNotIn("generic file extensions", final_answer.lower())
        self.assertIn("powershell", final_answer.lower())
        self.assertTrue(report["web_result_reference_leakage_removed"])
        self.assertTrue(report["iocs_strengthened"])
        self.assertTrue(report["weak_iocs_removed"])

    def test_output_validator_strengthens_attack_explanation_and_recent_examples(self) -> None:
        web_results = [
            {
                "title": "Ransomware operators increasingly pair extortion with recovery disruption",
                "url": "https://example.org/recovery-disruption",
                "content": (
                    "Current reporting is trend-oriented rather than tied to a single named victim, but it consistently describes lateral movement, "
                    "backup disruption, shadow copy deletion, and extortion pressure after initial access."
                ),
            }
        ]
        answer = (
            "Overview\nGrounded overview.\n\n"
            "Attack Explanation\nThis answer uses grounded source evidence from the retrieved documents available in this run.\n\n"
            "Recent Examples\n- 2026 ransomware statistics overview. (Web Result 1)\n\n"
            "IOCs\n- Suspicious PowerShell execution.\n\n"
            "Detection\n- Monitor abnormal file modification patterns, shadow copy deletion, suspicious PowerShell usage, and lateral movement over SMB.\n\n"
            "Mitigation\n- Enforce network segmentation, least privilege, offline backups, and monitoring readiness.\n\n"
            "Limitations\nGrounded limitations."
        )

        final_answer, report = validate_output_with_report(answer, web_results=web_results)

        self.assertIn("initial access", final_answer.lower())
        self.assertIn("shadow copies", final_answer.lower())
        self.assertIn("encrypt systems", final_answer.lower())
        self.assertNotIn("grounded source evidence", final_answer.lower())
        self.assertIn("trend-oriented", final_answer.lower())
        self.assertTrue(report["attack_explanation_strengthened"])
        self.assertTrue(report["recent_examples_strengthened"])


if __name__ == "__main__":
    unittest.main()
