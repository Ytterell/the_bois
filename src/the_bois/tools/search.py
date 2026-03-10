"""Tiered research tools — PyPI → GitHub → Web search.

DDG's default API backend returns garbage for niche technical queries
(e.g. "textual library" → textual.ai marketing spam).  So we go
straight to the source: PyPI for package metadata/README, GitHub for
repos/READMEs, and DDG *lite* backend as a last resort (lite actually
hits DuckDuckGo's HTML endpoint, which returns real results).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

# ── Package name extraction ─────────────────────────────────────────
# Words that are almost never a PyPI package on their own.
_STOP_WORDS = frozenset(
    {
        # English
        "a",
        "an",
        "the",
        "for",
        "with",
        "how",
        "to",
        "use",
        "in",
        "on",
        "and",
        "or",
        "of",
        "is",
        "it",
        "by",
        "from",
        "that",
        "this",
        "not",
        "but",
        "be",
        "do",
        "if",
        "my",
        "no",
        "so",
        "we",
        "up",
        "as",
        "at",
        "all",
        "can",
        "get",
        "has",
        "new",
        "any",
        "may",
        # Tech generics — useful as search keywords but not package names
        "library",
        "framework",
        "api",
        "widget",
        "layout",
        "responsive",
        "grid",
        "tui",
        "gui",
        "cli",
        "documentation",
        "docs",
        "tutorial",
        "example",
        "code",
        "programming",
        "python",
        "implementation",
        "module",
        "package",
        "function",
        "class",
        "method",
        "import",
        "install",
        "setup",
        "config",
        "configuration",
        "create",
        "build",
        "using",
        "based",
        "app",
        "application",
        "project",
        "system",
        "interface",
        "component",
        "components",
        "tool",
        "tools",
        "data",
        "file",
        "files",
        "test",
        "tests",
        "error",
        "errors",
        "fix",
        "terminal",
        "console",
        "screen",
        "display",
        "render",
        "style",
        "theme",
        "dark",
        "light",
        "color",
        "key",
        "keyboard",
        "shortcut",
        "shortcuts",
        "event",
        "events",
        "handler",
        "callback",
        "async",
        "sync",
        "web",
        "http",
        "server",
        "client",
        "request",
        "response",
    }
)


def _extract_package_candidates(query: str) -> list[str]:
    """Pull potential PyPI package names out of a research query.

    Returns de-duped candidates in order of likelihood.
    We try individual words AND hyphenated adjacent pairs
    (e.g. "flask cors" → ["flask", "cors", "flask-cors"]).
    """
    words = re.findall(r"[a-zA-Z0-9_-]+", query.lower())
    words = [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    candidates: list[str] = []
    seen: set[str] = set()

    # Individual words
    for w in words:
        if w not in seen:
            candidates.append(w)
            seen.add(w)

    # Adjacent hyphenated pairs (common pattern: flask-cors, duckduckgo-search)
    for i in range(len(words) - 1):
        pair = f"{words[i]}-{words[i + 1]}"
        if pair not in seen:
            candidates.append(pair)
            seen.add(pair)

    # resolve_pypi_name-style variants — add common transformations
    # so we find google-api-python-client from "googleapiclient" etc.
    extra: list[str] = []
    for w in list(candidates):
        for variant in (
            w.replace("_", "-"),
            f"python-{w}",
            f"py{w}",
        ):
            if variant not in seen and variant != w:
                extra.append(variant)
                seen.add(variant)
    candidates.extend(extra)

    return candidates


# ── Tier 1: PyPI ─────────────────────────────────────────────────────


async def _fetch_pypi(
    package_name: str, max_chars: int = 3000
) -> tuple[str, str | None]:
    """Fetch package info from PyPI JSON API.

    Returns (content, docs_url) — content is a formatted string with
    summary, docs URL, and code examples from the README (or empty
    string on miss/error).  docs_url is the documentation URL extracted
    from project_urls, or None if not found.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return "", None
            resp.raise_for_status()
            data = resp.json()

        info = data.get("info", {})
        summary = info.get("summary", "")
        home_page = info.get("home_page", "")
        project_urls = info.get("project_urls") or {}
        description = info.get("description", "")

        # ── C: Extract docs URL from project_urls ──
        docs_url: str | None = None
        for label, link in project_urls.items():
            if any(k in label.lower() for k in ("doc", "documentation")):
                docs_url = link
                break
        # Fallback: homepage on known docs-hosting domains
        if not docs_url and home_page:
            hp_host = urlparse(home_page).hostname or ""
            if any(d in hp_host for d in ("readthedocs", ".io", "docs.")):
                docs_url = home_page

        parts: list[str] = []
        parts.append(f"📦 {info.get('name', package_name)} — {summary}")

        # Useful links (docs, repo)
        for label, link in project_urls.items():
            if any(k in label.lower() for k in ("doc", "source", "repo", "home")):
                parts.append(f"  {label}: {link}")
        if home_page and home_page not in "\n".join(parts):
            parts.append(f"  Homepage: {home_page}")

        # Code examples from README
        code_blocks = re.findall(
            r"```(?:python|py)?\n(.*?)```",
            description,
            re.DOTALL,
        )
        if code_blocks:
            parts.append("\nCode examples from README:")
            chars_used = 0
            for block in code_blocks[:6]:
                block = block.strip()
                if chars_used + len(block) > max_chars // 2:
                    break
                parts.append(f"```\n{block}\n```")
                chars_used += len(block)

        # First chunk of the README for context
        if description:
            # Strip markdown badges/images
            clean = re.sub(r"!\[.*?\]\(.*?\)", "", description)
            clean = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", clean)
            clean = clean.strip()
            remaining = max_chars - sum(len(p) for p in parts)
            if remaining > 200:
                parts.append(f"\nREADME excerpt:\n{clean[:remaining]}")

        combined = "\n".join(parts)
        return combined[:max_chars], docs_url

    except Exception as e:
        log.debug("PyPI fetch failed for %s: %s", package_name, e)
        return "", None


# ── Tier 2: GitHub ───────────────────────────────────────────────────


async def _fetch_github(query: str, max_chars: int = 3000) -> str:
    """Search GitHub for repos matching the query, fetch top README.

    GitHub's search is much better than DDG for finding the right repo
    when you have a library name.  Returns formatted string or empty.
    """
    search_url = "https://api.github.com/search/repositories"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                search_url,
                params={"q": query, "sort": "stars", "per_page": 3},
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            # GitHub rate limits unauthenticated to 10 req/min — be cool
            if resp.status_code == 403:
                log.warning("GitHub API rate limited")
                return ""
            resp.raise_for_status()
            items = resp.json().get("items", [])

            if not items:
                return ""

            parts: list[str] = []
            chars_used = 0

            for repo in items[:2]:
                full_name = repo["full_name"]
                desc = repo.get("description", "")
                stars = repo.get("stargazers_count", 0)

                parts.append(f"\n🔗 {full_name} ({stars:,}⭐)\n  {desc}")

                # Fetch README
                try:
                    readme_resp = await client.get(
                        f"https://api.github.com/repos/{full_name}/readme",
                        headers={"Accept": "application/vnd.github.v3.raw"},
                    )
                    if readme_resp.status_code != 200:
                        continue
                    readme = readme_resp.text

                    # Extract code blocks
                    code_blocks = re.findall(
                        r"```(?:python|py)?\n(.*?)```",
                        readme,
                        re.DOTALL,
                    )
                    if code_blocks:
                        parts.append("  Code examples:")
                        for block in code_blocks[:4]:
                            block = block.strip()
                            if chars_used + len(block) > max_chars // 2:
                                break
                            parts.append(f"  ```\n  {block}\n  ```")
                            chars_used += len(block)

                except Exception:
                    pass  # README fetch is best-effort

            combined = "\n".join(parts)
            return combined[:max_chars]

    except Exception as e:
        log.debug("GitHub search failed for %s: %s", query, e)
        return ""


# ── Tier 3: Web search (DDG) ─────────────────────────────────────────

# ── N: Expanded dev domains (exact and wildcard) ──
_DEV_DOMAINS = frozenset(
    {
        "stackoverflow.com",
        "github.com",
        "docs.python.org",
        "pypi.org",
        "developer.mozilla.org",
        "medium.com",
        "dev.to",
        "realpython.com",
        "geeksforgeeks.org",
        "rust-lang.org",
        "go.dev",
        "npmjs.com",
        "crates.io",
        "learn.microsoft.com",
        "wiki.archlinux.org",
        # Docs-hosting domains (exact)
        "textualize.io",
        "fastapi.tiangolo.com",
        "typer.tiangolo.com",
        "flask.palletsprojects.com",
        "click.palletsprojects.com",
        "jinja.palletsprojects.com",
        "werkzeug.palletsprojects.com",
        "docs.djangoproject.com",
        "docs.pydantic.dev",
        "docs.sqlalchemy.org",
        "mkdocs.org",
    }
)

# Wildcard suffixes — any host ending with these gets a bonus.
_DEV_DOMAIN_WILDCARDS = (
    ".readthedocs.io",
    ".readthedocs.org",
    ".palletsprojects.com",
    ".tiangolo.com",
    ".textualize.io",
)


def _domain_score(url: str) -> int:
    """Score a URL by how likely it is to contain useful dev content."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return 0
    # Exact match
    for domain in _DEV_DOMAINS:
        if host == domain or host.endswith(f".{domain}"):
            return 2
    # Wildcard match (readthedocs etc.)
    for suffix in _DEV_DOMAIN_WILDCARDS:
        if host.endswith(suffix) or host == suffix.lstrip("."):
            return 2
    return 0


async def fetch_page_content(url: str, max_chars: int = 2000) -> str:
    """Fetch actual page content and extract useful code/text.

    Focuses on <code>, <pre> blocks and nearby headings.
    Falls back gracefully on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "the_bois/1.0"})
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        log.debug("Failed to fetch %s: %s", url, e)
        return ""

    parts: list[str] = []
    for pattern in (r"<pre[^>]*>(.*?)</pre>", r"<code[^>]*>(.*?)</code>"):
        for match in re.finditer(pattern, html, re.DOTALL | re.IGNORECASE):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if len(text) > 20:
                parts.append(text)

    if not parts:
        return ""

    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n... (truncated)"
    return combined


# Module-level circuit breaker for DDG — if it keeps failing, stop wasting time
_ddg_consecutive_failures: int = 0
_DDG_CIRCUIT_BREAKER_THRESHOLD: int = 3


async def _search_web(query: str, max_results: int = 5) -> str:
    """Search via DuckDuckGo + fetch top dev-domain page content."""
    global _ddg_consecutive_failures

    if _ddg_consecutive_failures >= _DDG_CIRCUIT_BREAKER_THRESHOLD:
        log.info(
            "DDG circuit breaker open (%d consecutive failures) — skipping web search",
            _ddg_consecutive_failures,
        )
        return ""

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Search unavailable: neither ddgs nor duckduckgo_search installed."

    def _search() -> list[dict]:
        import time
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Retry up to 2 times with shorter backoff (was 3 retries, 2s/4s).
        # DDG rate-limits are rarely transient — burning 30s on retries
        # when it's actually down just slows everything.
        for attempt in range(2):
            if attempt > 0:
                time.sleep(2)
                log.debug("DDG search retry %d/2 for: %s", attempt + 1, query[:80])
            with DDGS() as ddgs:
                for backend in ("duckduckgo", None):
                    try:
                        kw: dict = {"max_results": max_results}
                        if backend is not None:
                            kw["backend"] = backend
                        results = list(ddgs.text(query, **kw))
                        if results:
                            return results
                    except Exception as e:
                        label = backend or "auto"
                        log.warning("DDG %s backend failed: %s", label, e)
        log.warning("DDG search exhausted all retries for: %s", query[:80])
        return []

    results = await asyncio.to_thread(_search)
    if not results:
        _ddg_consecutive_failures += 1
        return ""

    # Reset circuit breaker on success
    _ddg_consecutive_failures = 0

    # Score and sort by relevance (dev domains first)
    scored = sorted(
        results,
        key=lambda r: _domain_score(r.get("href", "")),
        reverse=True,
    )

    formatted: list[str] = []
    for i, r in enumerate(scored, 1):
        formatted.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")

    text = "\n\n".join(formatted)

    # Fetch page content from top dev-domain results
    dev_urls = [r["href"] for r in scored if _domain_score(r.get("href", "")) > 0][:2]

    if dev_urls:
        for url in dev_urls:
            content = await fetch_page_content(url)
            if content:
                text += f"\n\n--- Content from {url} ---\n{content}"

    return text


# ── Tier 1.5: Docs-site crawler ──────────────────────────────────────

_LINK_SCORE_TERMS = frozenset(
    {
        "reference",
        "api",
        "classes",
        "modules",
        "widgets",
        "components",
        "events",
        "methods",
        "attributes",
        "guide",
        "tutorial",
        "quickstart",
    }
)


def _score_link(href: str, text: str, query_words: set[str]) -> int:
    """Score a link's relevance to the query for docs-site crawling."""
    score = 0
    text_lower = text.lower()
    href_lower = href.lower()
    # Query terms in link text / href
    for w in query_words:
        if w in text_lower:
            score += 3
        if w in href_lower:
            score += 2
    # Generic API-reference terms
    for t in _LINK_SCORE_TERMS:
        if t in text_lower or t in href_lower:
            score += 1
    return score


async def _fetch_docs_site(
    docs_url: str,
    query: str,
    max_chars: int = 5000,
) -> str:
    """Fetch actual documentation pages by crawling the docs site.

    1. Fetch landing page, extract navigation links.
    2. Score links by relevance to *query*.
    3. Fetch top 2-3 sub-pages and extract code blocks, headings, and
       definition-list entries (common in Sphinx/MkDocs API references).
    """
    query_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_]+", query)
        if w.lower() not in _STOP_WORDS and len(w) > 1
    }
    if not query_words:
        query_words = {"api", "reference"}

    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
        ) as client:
            landing = await client.get(
                docs_url,
                headers={"User-Agent": "the_bois/1.0"},
            )
            landing.raise_for_status()
            html = landing.text
    except Exception as e:
        log.debug("Docs-site fetch failed for %s: %s", docs_url, e)
        return ""

    # ── Extract and score links ──
    base_host = urlparse(docs_url).hostname or ""
    link_pattern = re.compile(
        r'<a\s[^>]*href=["\']([^"\'>]+)["\'][^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    scored_links: list[tuple[int, str, str]] = []  # (score, abs_url, text)
    seen_urls: set[str] = set()

    for href_raw, text_raw in link_pattern.findall(html):
        text_clean = re.sub(r"<[^>]+>", "", text_raw).strip()
        if not text_clean or not href_raw:
            continue
        # Resolve relative URLs
        abs_url = urljoin(docs_url, href_raw.split("#")[0])
        # Only same-domain links
        link_host = urlparse(abs_url).hostname or ""
        if link_host != base_host:
            continue
        if abs_url in seen_urls:
            continue
        seen_urls.add(abs_url)

        score = _score_link(href_raw, text_clean, query_words)
        if score > 0:
            scored_links.append((score, abs_url, text_clean))

    # Sort by score descending, take top 3
    scored_links.sort(key=lambda x: x[0], reverse=True)
    top_links = scored_links[:3]

    if not top_links:
        # Just extract content from the landing page itself
        content = _extract_docs_content(html, max_chars)
        if content:
            return f"[Documentation: {docs_url}]\n{content}"
        return ""

    # ── Fetch sub-pages ──
    parts: list[str] = []
    chars_used = 0
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
        ) as client:
            for _score, page_url, link_text in top_links:
                if chars_used >= max_chars:
                    break
                try:
                    resp = await client.get(
                        page_url,
                        headers={"User-Agent": "the_bois/1.0"},
                    )
                    if resp.status_code != 200:
                        continue
                    page_content = _extract_docs_content(
                        resp.text,
                        max_chars - chars_used,
                    )
                    if page_content:
                        section = f"\n--- {link_text} ({page_url}) ---\n{page_content}"
                        parts.append(section)
                        chars_used += len(section)
                except Exception:
                    continue
    except Exception as e:
        log.debug("Docs sub-page fetch failed: %s", e)

    if not parts:
        return ""

    return f"[Documentation from {base_host}]\n" + "\n".join(parts)


def _extract_docs_content(html: str, max_chars: int = 5000) -> str:
    """Extract useful documentation content from an HTML page.

    Targets: headings before code blocks, <pre>/<code> blocks, and
    <dt>/<dd> pairs (Sphinx/MkDocs API reference format).
    """
    parts: list[str] = []

    # Headings paired with following code blocks
    heading_code = re.finditer(
        r"<h[1-4][^>]*>(.*?)</h[1-4]>\s*(?:.*?)"
        r"(<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>)",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    for m in heading_code:
        heading = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        code = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if heading and code and len(code) > 15:
            parts.append(f"## {heading}\n```\n{code}\n```")

    # Standalone <pre> blocks
    for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL | re.IGNORECASE):
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if len(text) > 20 and text not in "\n".join(parts):
            parts.append(f"```\n{text}\n```")

    # <dt>/<dd> pairs (API reference format)
    dt_dd = re.finditer(
        r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    for m in dt_dd:
        term = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        desc = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if term:
            parts.append(f"{term}: {desc[:200]}")

    if not parts:
        return ""

    combined = "\n\n".join(parts)
    return combined[:max_chars]


# ── PyPI name resolution (for deps auto-install) ────────────────────


async def resolve_pypi_name(import_name: str) -> str | None:
    """Resolve a Python import name to the correct PyPI package name.

    Tries the obvious match first (import_name == pip_name), then common
    transformations (underscore→hyphen, python- prefix).  Returns the
    canonical pip-installable name or None if truly not found.

    This replaces any hardcoded import→package dict — always up to date,
    no maintenance needed.  Ain't nobody got time for that.
    """
    candidates = [
        import_name,  # textual, flask, httpx
        import_name.replace("_", "-"),  # google_cloud → google-cloud
        f"python-{import_name}",  # dateutil → python-dateutil
        f"py{import_name}",  # yaml → pyyaml (close enough)
    ]
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        c_lower = c.lower()
        if c_lower not in seen:
            seen.add(c_lower)
            unique.append(c)

    async with httpx.AsyncClient(timeout=8.0) as client:
        for name in unique:
            url = f"https://pypi.org/pypi/{name}/json"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    canonical = data.get("info", {}).get("name", name)
                    log.debug(
                        "PyPI resolve: '%s' → '%s' (tried '%s')",
                        import_name,
                        canonical,
                        name,
                    )
                    return canonical
            except Exception:
                continue

    log.debug("PyPI resolve: '%s' → not found", import_name)
    return None


# ── B: API-usage signal words (don't short-circuit on PyPI alone) ────
_API_USAGE_SIGNALS = frozenset(
    {
        "attribute",
        "method",
        "signature",
        "usage",
        "compose",
        "event",
        "handler",
        "lifecycle",
        "callback",
        "property",
        "parameter",
        "argument",
        "return",
        "example",
        "correct",
        "how",
    }
)


# ── Main entry point ─────────────────────────────────────────────────


async def web_search(
    query: str,
    max_results: int = 5,
    known_docs_urls: dict[str, str] | None = None,
) -> str:
    """Tiered research: PyPI → docs site → GitHub → web search.

    1. Extract potential package names, fetch from PyPI (authoritative).
    1.5. If PyPI returned a docs URL (or we have one from a prior research
         run via *known_docs_urls*), crawl the actual docs site.
    2. Search GitHub for repos, fetch READMEs.
    3. If tiers 1-2 yield nothing OR the query is about API usage,
       also run web search (DDG lite).

    Args:
        query: Free-text research query.
        max_results: Cap for DDG web-search hits.
        known_docs_urls: Mapping of package-name → docs URL discovered during
            an earlier research phase.  Used as fallback when PyPI doesn't
            return a docs link itself.
    """
    parts: list[str] = []
    docs_url: str | None = None

    # ── Tier 1: PyPI ──
    candidates = _extract_package_candidates(query)
    pypi_hits: list[str] = []

    # Fetch all candidates in parallel — much faster than sequential
    pypi_results = await asyncio.gather(
        *(_fetch_pypi(name) for name in candidates[:4]),
        return_exceptions=True,
    )
    for name, result in zip(candidates[:4], pypi_results):
        if isinstance(result, BaseException):
            log.debug("PyPI fetch failed for '%s': %s", name, result)
            continue
        content, found_docs_url = result
        if content:
            pypi_hits.append(content)
            log.info("PyPI hit for '%s'", name)
        if found_docs_url and not docs_url:
            docs_url = found_docs_url
    if pypi_hits:
        parts.append("[PyPI Package Info]\n" + "\n\n".join(pypi_hits))

    # Fall back to known docs URLs from prior research if PyPI didn't give us one
    if not docs_url and known_docs_urls:
        for candidate in candidates[:4]:
            if candidate in known_docs_urls:
                docs_url = known_docs_urls[candidate]
                log.info("Using known docs URL for '%s': %s", candidate, docs_url)
                break

    # ── Tier 1.5: Follow docs URL from PyPI or known URLs ──
    if docs_url:
        log.info("Following docs URL: %s", docs_url)
        docs_content = await _fetch_docs_site(docs_url, query)
        if docs_content:
            parts.append(docs_content)

    # ── Tier 2: GitHub ──
    github_info = await _fetch_github(query)
    if github_info:
        parts.append("[GitHub Repositories]\n" + github_info)

    # ── B: Check if query is about API usage ──
    query_lower = query.lower()
    is_api_query = any(w in query_lower for w in _API_USAGE_SIGNALS)

    # If we got authoritative sources AND this isn't an API-usage query,
    # return them without hitting web search.
    if parts and not is_api_query:
        return "\n\n---\n\n".join(parts)

    # ── Tier 3: Web search ──
    if not parts:
        log.info("No PyPI/GitHub hits for '%s', falling back to web search", query)
    elif is_api_query:
        log.info("API-usage query detected — also running web search for '%s'", query)
    web_results = await _search_web(query, max_results=max_results)
    if web_results:
        parts.append("[Web Search Results]\n" + web_results)

    if parts:
        return "\n\n---\n\n".join(parts)

    return f"No results found for: {query}"
