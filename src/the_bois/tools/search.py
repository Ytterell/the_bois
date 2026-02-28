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
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# ── Package name extraction ─────────────────────────────────────────
# Words that are almost never a PyPI package on their own.
_STOP_WORDS = frozenset({
    # English
    "a", "an", "the", "for", "with", "how", "to", "use", "in", "on",
    "and", "or", "of", "is", "it", "by", "from", "that", "this",
    "not", "but", "be", "do", "if", "my", "no", "so", "we", "up",
    "as", "at", "all", "can", "get", "has", "new", "any", "may",
    # Tech generics — useful as search keywords but not package names
    "library", "framework", "api", "widget", "layout", "responsive",
    "grid", "tui", "gui", "cli", "documentation", "docs", "tutorial",
    "example", "code", "programming", "python", "implementation",
    "module", "package", "function", "class", "method", "import",
    "install", "setup", "config", "configuration", "create", "build",
    "using", "based", "app", "application", "project", "system",
    "interface", "component", "components", "tool", "tools", "data",
    "file", "files", "test", "tests", "error", "errors", "fix",
    "terminal", "console", "screen", "display", "render", "style",
    "theme", "dark", "light", "color", "key", "keyboard", "shortcut",
    "shortcuts", "event", "events", "handler", "callback", "async",
    "sync", "web", "http", "server", "client", "request", "response",
})


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

    return candidates


# ── Tier 1: PyPI ─────────────────────────────────────────────────────

async def _fetch_pypi(package_name: str, max_chars: int = 3000) -> str:
    """Fetch package info from PyPI JSON API.

    Returns a formatted string with summary, docs URL, and code
    examples from the README — or empty string on miss/error.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return ""
            resp.raise_for_status()
            data = resp.json()

        info = data.get("info", {})
        summary = info.get("summary", "")
        home_page = info.get("home_page", "")
        project_urls = info.get("project_urls") or {}
        description = info.get("description", "")

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
            r"```(?:python|py)?\n(.*?)```", description, re.DOTALL,
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
        return combined[:max_chars]

    except Exception as e:
        log.debug("PyPI fetch failed for %s: %s", package_name, e)
        return ""


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

                parts.append(
                    f"\n🔗 {full_name} ({stars:,}⭐)"
                    f"\n  {desc}"
                )

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
                        r"```(?:python|py)?\n(.*?)```", readme, re.DOTALL,
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


# ── Tier 3: Web search (DDG lite backend) ────────────────────────────

_DEV_DOMAINS = frozenset({
    "stackoverflow.com", "github.com", "docs.python.org", "pypi.org",
    "developer.mozilla.org", "readthedocs.io", "medium.com", "dev.to",
    "realpython.com", "geeksforgeeks.org", "rust-lang.org", "go.dev",
    "npmjs.com", "crates.io", "learn.microsoft.com", "wiki.archlinux.org",
})


def _domain_score(url: str) -> int:
    """Score a URL by how likely it is to contain useful dev content."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return 0
    for domain in _DEV_DOMAINS:
        if host == domain or host.endswith(f".{domain}"):
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


async def _search_web(query: str, max_results: int = 5) -> str:
    """Search via DDG lite backend + fetch top dev-domain page content.

    The 'lite' backend actually hits DuckDuckGo's HTML endpoint which
    returns real results, unlike the default 'api' backend that returns
    garbage for technical queries.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Search unavailable: neither ddgs nor duckduckgo_search installed."

    def _search() -> list[dict]:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Try lite first (best results), then html fallback
        for backend in ("lite", "html"):
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(
                        query, max_results=max_results, backend=backend,
                    ))
                if results:
                    return results
            except Exception as e:
                log.warning("DDG %s backend failed: %s", backend, e)
        return []

    results = await asyncio.to_thread(_search)
    if not results:
        return ""

    # Score and sort by relevance (dev domains first)
    scored = sorted(
        results, key=lambda r: _domain_score(r.get("href", "")), reverse=True,
    )

    formatted: list[str] = []
    for i, r in enumerate(scored, 1):
        formatted.append(
            f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}"
        )

    text = "\n\n".join(formatted)

    # Fetch page content from top dev-domain results
    dev_urls = [
        r["href"] for r in scored
        if _domain_score(r.get("href", "")) > 0
    ][:2]

    if dev_urls:
        for url in dev_urls:
            content = await fetch_page_content(url)
            if content:
                text += f"\n\n--- Content from {url} ---\n{content}"

    return text


# ── Main entry point ─────────────────────────────────────────────────

async def web_search(query: str, max_results: int = 5) -> str:
    """Tiered research: PyPI → GitHub → web search.

    1. Extract potential package names, fetch from PyPI (authoritative).
    2. Search GitHub for repos, fetch READMEs.
    3. If tiers 1-2 yield nothing, fall back to web search (DDG lite).
    """
    parts: list[str] = []

    # ── Tier 1: PyPI ──
    candidates = _extract_package_candidates(query)
    pypi_hits: list[str] = []

    for name in candidates[:4]:  # don't spam PyPI with 20 requests
        info = await _fetch_pypi(name)
        if info:
            pypi_hits.append(info)
            log.info("PyPI hit for '%s'", name)
    if pypi_hits:
        parts.append("[PyPI Package Info]\n" + "\n\n".join(pypi_hits))

    # ── Tier 2: GitHub ──
    github_info = await _fetch_github(query)
    if github_info:
        parts.append("[GitHub Repositories]\n" + github_info)

    # If we got authoritative sources, return them
    if parts:
        return "\n\n---\n\n".join(parts)

    # ── Tier 3: Web search (fallback) ──
    log.info("No PyPI/GitHub hits for '%s', falling back to web search", query)
    web_results = await _search_web(query, max_results=max_results)
    if web_results:
        return web_results

    return f"No results found for: {query}"
