#!/usr/bin/env python3
"""
Sentinel Enterprise Policy Downloader

Downloads public policy, regulatory guidance, standards guidance, and compliance
documents from official sources into isolated source folders with provenance
metadata. UC HR policies are intentionally excluded because they are handled by
download_uc_hr_policies.py.

Examples:
    python enterprise_policy_downloader.py --group hr
    python enterprise_policy_downloader.py --group all
    python enterprise_policy_downloader.py --source nist
    python enterprise_policy_downloader.py --list-sources

Important:
- This tool downloads only publicly accessible material.
- It does not bypass authentication, paywalls, licenses, or robots.txt.
- Different organizations remain separated. Do not treat all sources as having
  equal authority during retrieval or answer generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Sentinel-Enterprise-Policy-Downloader/1.0 "
    "(public compliance research; respectful request rate)"
)

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.5",
    }
)


@dataclass(frozen=True)
class Source:
    key: str
    group: str
    organization: str
    authority_type: str
    jurisdiction: str
    scope: str
    seed_urls: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    max_depth: int = 1
    max_documents: int = 100


# Official public sources. The patterns deliberately focus collection on policy,
# guidance, publication, framework, and compliance material.
SOURCES: dict[str, Source] = {
    # Human resources and workplace compliance
    "dol": Source(
        key="dol",
        group="hr",
        organization="U.S. Department of Labor",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="Federal labor standards, wages, leave, and employer compliance",
        seed_urls=(
            "https://www.dol.gov/agencies/whd/fact-sheets",
            "https://www.dol.gov/agencies/whd/resources",
        ),
        allowed_hosts=("dol.gov", "www.dol.gov"),
        include_patterns=("fact-sheet", "fact_sheets", "fmla", "flsa", "guidance", "publication", ".pdf"),
        exclude_patterns=("/news", "/contact", "/about", "/forms"),
        max_depth=2,
        max_documents=180,
    ),
    "eeoc": Source(
        key="eeoc",
        group="hr",
        organization="U.S. Equal Employment Opportunity Commission",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="Federal equal-employment, discrimination, harassment, retaliation, and accommodation guidance",
        seed_urls=(
            "https://www.eeoc.gov/eeoc-guidance",
            "https://www.eeoc.gov/regulations-and-guidelines",
            "https://www.eeoc.gov/laws",
        ),
        allowed_hosts=("eeoc.gov", "www.eeoc.gov"),
        include_patterns=("guidance", "laws", "regulation", "harassment", "discrimination", "retaliation", "accommodation", ".pdf"),
        exclude_patterns=("/newsroom", "/contact", "/field-office"),
        max_depth=2,
        max_documents=160,
    ),
    "osha": Source(
        key="osha",
        group="hr",
        organization="Occupational Safety and Health Administration",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="Workplace safety and health compliance",
        seed_urls=(
            "https://www.osha.gov/publications/",
            "https://www.osha.gov/publications/bytype/guidance-documents",
        ),
        allowed_hosts=("osha.gov", "www.osha.gov", "obis.osha.gov"),
        include_patterns=("publication", "guidance", "standard", "safety", "health", ".pdf"),
        exclude_patterns=("/news", "/contact"),
        max_depth=2,
        max_documents=180,
    ),
    "opm": Source(
        key="opm",
        group="hr",
        organization="U.S. Office of Personnel Management",
        authority_type="federal_workforce_policy",
        jurisdiction="United States",
        scope="Federal workforce policy, hiring, performance, leave, records, and employee relations",
        seed_urls=("https://www.opm.gov/policy-data-oversight/",),
        allowed_hosts=("opm.gov", "www.opm.gov"),
        include_patterns=("policy-data-oversight", "guidance", "handbook", "guide", ".pdf"),
        exclude_patterns=("/news", "/about-us", "/contact"),
        max_depth=2,
        max_documents=180,
    ),

    # Cybersecurity and governance
    "nist": Source(
        key="nist",
        group="cyber",
        organization="National Institute of Standards and Technology",
        authority_type="federal_standard_guidance",
        jurisdiction="United States",
        scope="Cybersecurity, privacy, risk management, controls, and AI governance",
        seed_urls=(
            "https://csrc.nist.gov/publications/sp800",
            "https://www.nist.gov/cyberframework",
            "https://www.nist.gov/itl/ai-risk-management-framework",
        ),
        allowed_hosts=("nist.gov", "www.nist.gov", "csrc.nist.gov", "nvlpubs.nist.gov"),
        include_patterns=("publication", "sp800", "cyberframework", "risk-management-framework", ".pdf"),
        exclude_patterns=("/news", "/events"),
        max_depth=2,
        max_documents=220,
    ),
    "cisa": Source(
        key="cisa",
        group="cyber",
        organization="Cybersecurity and Infrastructure Security Agency",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="Cybersecurity operations, zero trust, incident response, and resilience",
        seed_urls=(
            "https://www.cisa.gov/resources-tools/resources",
            "https://www.cisa.gov/zero-trust-maturity-model",
        ),
        allowed_hosts=("cisa.gov", "www.cisa.gov"),
        include_patterns=("resources", "guidance", "guide", "zero-trust", "incident", "security", ".pdf"),
        exclude_patterns=("/news-events", "/careers", "/contact"),
        max_depth=2,
        max_documents=180,
    ),
    "owasp": Source(
        key="owasp",
        group="cyber",
        organization="OWASP Foundation",
        authority_type="industry_guidance",
        jurisdiction="Global",
        scope="Application security, API security, verification, and maturity guidance",
        seed_urls=(
            "https://owasp.org/www-project-application-security-verification-standard/",
            "https://owasp.org/www-project-top-ten/",
            "https://owasp.org/www-project-api-security/",
            "https://owasp.org/www-project-samm/",
        ),
        allowed_hosts=("owasp.org", "www.owasp.org"),
        include_patterns=("project", "standard", "top-ten", "api-security", "samm", "download", ".pdf"),
        exclude_patterns=("/events", "/chapters"),
        max_depth=2,
        max_documents=100,
    ),

    # Privacy and data protection
    "hhs_hipaa": Source(
        key="hhs_hipaa",
        group="privacy",
        organization="U.S. Department of Health and Human Services",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="HIPAA privacy, security, breach notification, and health-information compliance",
        seed_urls=(
            "https://www.hhs.gov/hipaa/for-professionals/index.html",
            "https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html",
        ),
        allowed_hosts=("hhs.gov", "www.hhs.gov"),
        include_patterns=("hipaa", "privacy", "security", "breach", "guidance", ".pdf"),
        exclude_patterns=("/news", "/about"),
        max_depth=2,
        max_documents=180,
    ),
    "edpb": Source(
        key="edpb",
        group="privacy",
        organization="European Data Protection Board",
        authority_type="eu_regulatory_guidance",
        jurisdiction="European Union / EEA",
        scope="GDPR guidelines, recommendations, and best practices",
        seed_urls=("https://www.edpb.europa.eu/our-work-tools/general-guidance/guidelines-recommendations-best-practices_en",),
        allowed_hosts=("edpb.europa.eu", "www.edpb.europa.eu"),
        include_patterns=("guidelines", "recommendations", "best-practices", "files", ".pdf"),
        exclude_patterns=("/news", "/about-edpb"),
        max_depth=2,
        max_documents=180,
    ),
    "california_privacy": Source(
        key="california_privacy",
        group="privacy",
        organization="California Privacy Protection Agency",
        authority_type="state_regulatory_guidance",
        jurisdiction="California, United States",
        scope="California Consumer Privacy Act and California privacy regulations",
        seed_urls=("https://cppa.ca.gov/regulations/",),
        allowed_hosts=("cppa.ca.gov", "www.cppa.ca.gov"),
        include_patterns=("regulation", "ccpa", "cpra", "rulemaking", ".pdf"),
        exclude_patterns=("/meetings", "/careers"),
        max_depth=2,
        max_documents=100,
    ),
    "ftc_privacy": Source(
        key="ftc_privacy",
        group="privacy",
        organization="Federal Trade Commission",
        authority_type="federal_guidance",
        jurisdiction="United States",
        scope="Business privacy, data security, identity protection, and consumer compliance",
        seed_urls=("https://www.ftc.gov/business-guidance/privacy-security",),
        allowed_hosts=("ftc.gov", "www.ftc.gov"),
        include_patterns=("business-guidance", "privacy", "security", "data", ".pdf"),
        exclude_patterns=("/news-events", "/legal-library/browse/cases"),
        max_depth=2,
        max_documents=150,
    ),

    # Financial and payment compliance
    "sec": Source(
        key="sec",
        group="finance",
        organization="U.S. Securities and Exchange Commission",
        authority_type="federal_regulatory_guidance",
        jurisdiction="United States",
        scope="Securities compliance, disclosure, governance, and cybersecurity guidance",
        seed_urls=("https://www.sec.gov/rules-regulations",),
        allowed_hosts=("sec.gov", "www.sec.gov"),
        include_patterns=("rules-regulations", "guidance", "interpretation", "compliance", ".pdf"),
        exclude_patterns=("/newsroom", "/litigation"),
        max_depth=2,
        max_documents=160,
    ),
    "finra": Source(
        key="finra",
        group="finance",
        organization="Financial Industry Regulatory Authority",
        authority_type="self_regulatory_guidance",
        jurisdiction="United States",
        scope="Broker-dealer rules, regulatory notices, supervision, and compliance",
        seed_urls=("https://www.finra.org/rules-guidance/notices",),
        allowed_hosts=("finra.org", "www.finra.org"),
        include_patterns=("rules-guidance", "notices", "regulatory-notice", ".pdf"),
        exclude_patterns=("/media-center", "/careers"),
        max_depth=2,
        max_documents=180,
    ),
    "pci_ssc": Source(
        key="pci_ssc",
        group="finance",
        organization="PCI Security Standards Council",
        authority_type="industry_standard",
        jurisdiction="Global",
        scope="Publicly available payment-card security standards and guidance",
        seed_urls=("https://www.pcisecuritystandards.org/document_library/",),
        allowed_hosts=("pcisecuritystandards.org", "www.pcisecuritystandards.org"),
        include_patterns=("document_library", "document", "guidance", "standard", ".pdf"),
        exclude_patterns=("/training", "/assessors"),
        max_depth=2,
        max_documents=100,
    ),

    # Cloud compliance and architecture guidance
    "aws": Source(
        key="aws",
        group="cloud",
        organization="Amazon Web Services",
        authority_type="vendor_guidance",
        jurisdiction="Global",
        scope="Cloud security, governance, risk, architecture, and compliance",
        seed_urls=(
            "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html",
            "https://docs.aws.amazon.com/whitepapers/latest/aws-risk-and-compliance/welcome.html",
        ),
        allowed_hosts=("docs.aws.amazon.com",),
        include_patterns=("wellarchitected", "security-pillar", "risk-and-compliance", "whitepapers", ".pdf"),
        exclude_patterns=(),
        max_depth=2,
        max_documents=140,
    ),
    "microsoft": Source(
        key="microsoft",
        group="cloud",
        organization="Microsoft",
        authority_type="vendor_guidance",
        jurisdiction="Global",
        scope="Azure security, compliance, identity, governance, and Microsoft Purview",
        seed_urls=(
            "https://learn.microsoft.com/en-us/azure/security/fundamentals/",
            "https://learn.microsoft.com/en-us/compliance/",
        ),
        allowed_hosts=("learn.microsoft.com",),
        include_patterns=("/azure/security", "/compliance", "governance", "identity", "purview"),
        exclude_patterns=("/answers", "/training"),
        max_depth=2,
        max_documents=180,
    ),
    "google_cloud": Source(
        key="google_cloud",
        group="cloud",
        organization="Google Cloud",
        authority_type="vendor_guidance",
        jurisdiction="Global",
        scope="Cloud security, IAM, governance, privacy, and compliance",
        seed_urls=(
            "https://cloud.google.com/security",
            "https://cloud.google.com/docs/security",
        ),
        allowed_hosts=("cloud.google.com",),
        include_patterns=("/security", "/iam", "/compliance", "/architecture", "/governance"),
        exclude_patterns=("/blog", "/products"),
        max_depth=2,
        max_documents=180,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str, fallback: str = "document") -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"_+", "_", value).strip(" ._")
    return value[:180] or fallback


def normalized_url(url: str) -> str:
    clean, _fragment = urldefrag(url)
    return clean.rstrip("/") if clean.endswith("/") else clean


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def host_allowed(url: str, source: Source) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == allowed or host.endswith("." + allowed) for allowed in source.allowed_hosts)


def matches_source(url: str, text: str, source: Source) -> bool:
    haystack = f"{url} {text}".lower()
    if any(pattern.lower() in haystack for pattern in source.exclude_patterns):
        return False
    if not source.include_patterns:
        return True
    return any(pattern.lower() in haystack for pattern in source.include_patterns)


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            robots_url = f"{base}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
                self._cache[base] = parser
            except Exception:
                # On robots retrieval failure, stay conservative about crawling HTML.
                self._cache[base] = None
        parser = self._cache[base]
        return parser.can_fetch(USER_AGENT, url) if parser else False


ROBOTS = RobotsCache()


def request(url: str, timeout: int = 60) -> requests.Response:
    response = SESSION.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response


def content_kind(response: requests.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    path = urlparse(response.url).path.lower()
    if "application/pdf" in content_type or path.endswith(".pdf"):
        return "pdf"
    if "text/html" in content_type or "application/xhtml+xml" in content_type:
        return "html"
    return "other"


def html_title_and_text(html: bytes, fallback: str) -> tuple[str, str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else fallback
    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )
    return title, text, soup


def document_path(
    root: Path,
    source: Source,
    title: str,
    digest: str,
    extension: str,
) -> Path:
    folder = root / source.group / source.key
    folder.mkdir(parents=True, exist_ok=True)
    base = safe_filename(title)
    return folder / f"{base}_{digest[:10]}{extension}"


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "source_key",
        "group",
        "organization",
        "authority_type",
        "jurisdiction",
        "scope",
        "title",
        "source_url",
        "final_url",
        "document_type",
        "local_path",
        "content_type",
        "sha256",
        "bytes",
        "retrieved_at_utc",
        "status",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_source_metadata(root: Path, source: Source) -> None:
    folder = root / source.group / source.key
    folder.mkdir(parents=True, exist_ok=True)
    metadata = asdict(source)
    metadata["seed_urls"] = list(source.seed_urls)
    metadata["allowed_hosts"] = list(source.allowed_hosts)
    metadata["include_patterns"] = list(source.include_patterns)
    metadata["exclude_patterns"] = list(source.exclude_patterns)
    (folder / "_source_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def crawl_source(
    source: Source,
    root: Path,
    delay: float,
    save_html: bool,
    max_documents_override: int | None,
) -> list[dict[str, str]]:
    save_source_metadata(root, source)

    max_documents = max_documents_override or source.max_documents
    queue: deque[tuple[str, int, str]] = deque(
        (url, 0, source.organization) for url in source.seed_urls
    )
    visited: set[str] = set()
    known_hashes: set[str] = set()
    rows: list[dict[str, str]] = []
    saved_count = 0

    print(f"\n[{source.key}] {source.organization}")
    print(f"  Group: {source.group}; limit: {max_documents}")

    while queue and saved_count < max_documents:
        url, depth, anchor_text = queue.popleft()
        url = normalized_url(url)

        if url in visited or not host_allowed(url, source):
            continue
        visited.add(url)

        # Direct PDF links may still be fetched when the parent index was allowed.
        is_pdf_hint = urlparse(url).path.lower().endswith(".pdf")
        if not is_pdf_hint and not ROBOTS.allowed(url):
            print(f"  robots.txt skipped: {url}")
            continue

        row = {
            "source_key": source.key,
            "group": source.group,
            "organization": source.organization,
            "authority_type": source.authority_type,
            "jurisdiction": source.jurisdiction,
            "scope": source.scope,
            "title": anchor_text or url,
            "source_url": url,
            "final_url": "",
            "document_type": "",
            "local_path": "",
            "content_type": "",
            "sha256": "",
            "bytes": "0",
            "retrieved_at_utc": utc_now(),
            "status": "",
            "error": "",
        }

        try:
            response = request(url)
            kind = content_kind(response)
            row["final_url"] = response.url
            row["content_type"] = response.headers.get("content-type", "")
            row["document_type"] = kind

            if kind == "pdf":
                data = response.content
                digest = sha256_bytes(data)
                row["sha256"] = digest
                row["bytes"] = str(len(data))

                if digest in known_hashes:
                    row["status"] = "duplicate-skipped"
                else:
                    known_hashes.add(digest)
                    title = anchor_text or Path(urlparse(response.url).path).stem
                    path = document_path(root, source, title, digest, ".pdf")
                    path.write_bytes(data)
                    row["title"] = title
                    row["local_path"] = str(path.relative_to(root))
                    row["status"] = "downloaded"
                    saved_count += 1
                    print(f"  PDF  {saved_count:03d}: {path.name}")

                rows.append(row)

            elif kind == "html":
                title, extracted_text, soup = html_title_and_text(response.content, anchor_text or url)

                # Save substantive official guidance pages as HTML plus plain text.
                should_save = (
                    save_html
                    and depth > 0
                    and matches_source(response.url, title, source)
                    and len(extracted_text) >= 900
                )

                if should_save:
                    digest = sha256_bytes(response.content)
                    if digest not in known_hashes:
                        known_hashes.add(digest)
                        html_path = document_path(root, source, title, digest, ".html")
                        txt_path = html_path.with_suffix(".txt")
                        html_path.write_bytes(response.content)
                        txt_path.write_text(extracted_text, encoding="utf-8")
                        row["title"] = title
                        row["sha256"] = digest
                        row["bytes"] = str(len(response.content))
                        row["local_path"] = str(txt_path.relative_to(root))
                        row["status"] = "downloaded-html-as-text"
                        saved_count += 1
                        print(f"  HTML {saved_count:03d}: {txt_path.name}")
                        rows.append(row)

                if depth < source.max_depth:
                    for anchor in soup.find_all("a", href=True):
                        child_text = " ".join(anchor.get_text(" ", strip=True).split())
                        child_url = normalized_url(urljoin(response.url, anchor["href"]))
                        parsed = urlparse(child_url)
                        if parsed.scheme not in {"http", "https"}:
                            continue
                        if not host_allowed(child_url, source):
                            continue
                        if not matches_source(child_url, child_text, source):
                            continue
                        if child_url not in visited:
                            queue.append((child_url, depth + 1, child_text or title))

            else:
                row["status"] = "unsupported-content-skipped"
                rows.append(row)

        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print(f"  ERROR: {url}: {exc}", file=sys.stderr)

        time.sleep(delay)

    print(f"  Completed: {saved_count} documents saved; {len(visited)} URLs visited.")
    return rows


def selected_sources(args: argparse.Namespace) -> list[Source]:
    if args.source:
        unknown = [key for key in args.source if key not in SOURCES]
        if unknown:
            raise SystemExit(f"Unknown source(s): {', '.join(unknown)}")
        return [SOURCES[key] for key in args.source]

    groups = {source.group for source in SOURCES.values()}
    if args.group == "all":
        return list(SOURCES.values())
    if args.group not in groups:
        raise SystemExit(f"Unknown group: {args.group}")
    return [source for source in SOURCES.values() if source.group == args.group]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download public enterprise policy and compliance documents for Sentinel."
    )
    parser.add_argument(
        "--output",
        default="data/raw/enterprise_policy",
        help="Output directory (default: data/raw/enterprise_policy)",
    )
    parser.add_argument(
        "--group",
        choices=["all", "hr", "cyber", "privacy", "finance", "cloud"],
        default="all",
        help="Download one group or all groups (default: all)",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Download a specific source key; repeat for multiple sources",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Download only PDFs; do not save substantive HTML guidance pages",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Override maximum saved documents per source",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List configured sources and exit",
    )
    args = parser.parse_args()

    if args.list_sources:
        for key, source in SOURCES.items():
            print(f"{key:20} group={source.group:8} organization={source.organization}")
        return 0

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)

    chosen = selected_sources(args)
    print(f"Output directory: {root}")
    print(f"Selected sources: {', '.join(source.key for source in chosen)}")

    all_rows: list[dict[str, str]] = []
    for source in chosen:
        all_rows.extend(
            crawl_source(
                source=source,
                root=root,
                delay=max(args.delay, 0.25),
                save_html=not args.pdf_only,
                max_documents_override=args.max_documents,
            )
        )

    manifest_path = root / "manifest.csv"
    write_manifest(all_rows, manifest_path)

    downloaded = sum(row["status"].startswith("downloaded") for row in all_rows)
    errors = sum(row["status"] == "error" for row in all_rows)

    print("\nEnterprise policy download complete.")
    print(f"Documents saved: {downloaded}")
    print(f"Errors:          {errors}")
    print(f"Manifest:        {manifest_path}")
    print("\nKeep source_key, authority_type, jurisdiction, and scope metadata during ingestion.")

    # Partial source failures are expected on large public sites, so return success
    # when at least one document was downloaded.
    return 0 if downloaded > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
