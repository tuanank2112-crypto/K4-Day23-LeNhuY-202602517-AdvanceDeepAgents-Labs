"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.HTTPStatusError, httpx.TransportError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise

            if attempt == attempts - 1:
                raise

            retry_after = None
            if isinstance(exc, RetryableError) and exc.retry_after is not None:
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                ra = exc.response.headers.get("retry-after") or exc.response.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass

            if retry_after is not None:
                delay = min(cap, float(retry_after))
            else:
                exp_delay = base * (2 ** attempt)
                jitter = random.uniform(0, 0.5 * exp_delay)
                delay = min(cap, exp_delay + jitter)

            time.sleep(delay)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


# ---- TODO 2: arXiv ----
_LAST_ARXIV_CALL = 0.0


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[a-zA-Z0-9\-]+", query)
        if not terms:
            return "NO RESULTS"
        search_query = " AND ".join(f"all:{t}" for t in terms)

        # Respect arXiv etiquette: at least 3 seconds between two arXiv calls
        now = time.time()
        elapsed = now - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.time()

        clamped_max = max(1, min(int(max_results), 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _do_get():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        xml_text = with_retry(_do_get, attempts=5, base=2.0, cap=60.0)

        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            entries = root.findall("entry")

        records = []
        for entry in entries:
            raw_id = entry.findtext("atom:id", "", ns) or entry.findtext("id", "")
            paper_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1].strip())
            if not paper_id:
                continue
            url = f"https://arxiv.org/abs/{paper_id}"
            raw_pub = entry.findtext("atom:published", "", ns) or entry.findtext("published", "")
            published = raw_pub[:10]
            raw_title = entry.findtext("atom:title", "", ns) or entry.findtext("title", "")
            title = _clean(raw_title)
            raw_summary = entry.findtext("atom:summary", "", ns) or entry.findtext("summary", "")
            summary = _clean(raw_summary)[:600]
            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date:
            params["date"] = date

        def _do_get():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_do_get, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper")
            if not isinstance(paper, dict):
                continue
            pid = paper.get("id")
            if not pid:
                continue
            title = _clean(paper.get("title") or item.get("title") or "")
            summary = _clean(paper.get("summary") or item.get("summary") or "")[:600]
            published = (paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = paper.get("upvotes") or item.get("upvotes") or 0
            github = paper.get("githubRepo") or item.get("githubRepo") or ""
            stars = paper.get("githubStars") or item.get("githubStars") or 0

            if keyword:
                kw = keyword.lower()
                if kw not in title.lower() and kw not in summary.lower():
                    continue

            records.append({
                "id": str(pid),
                "url": f"https://huggingface.co/papers/{pid}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": query, "limit": clamped_limit}

        def _do_get():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_do_get, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
            pid = paper.get("id")
            if not pid:
                continue
            title = _clean(paper.get("title") or "")
            raw_summary = paper.get("ai_summary") or paper.get("summary") or ""
            summary = _clean(raw_summary)[:600]
            published = (paper.get("publishedAt") or "")[:10]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars") or 0

            records.append({
                "id": str(pid),
                "url": f"https://huggingface.co/papers/{pid}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact_key(text: str) -> str:
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key and key in text:
        text = text.replace(key, "<REDACTED>")
    return text


def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    exa_key = (os.getenv("EXA_API_KEY") or "").strip()
    url = EXA_URL
    if exa_key:
        url = f"{EXA_URL}?exaApiKey={exa_key}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if exa_key:
        headers["Authorization"] = f"Bearer {exa_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _do_post():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text

            data = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    break
            if data is None:
                try:
                    data = resp.json()
                except Exception:
                    pass

            if not data or not isinstance(data, dict):
                raise RuntimeError("Invalid response from Exa MCP")

            if "error" in data:
                err_obj = data["error"]
                err_msg = str(err_obj.get("message", err_obj) if isinstance(err_obj, dict) else err_obj)
                if any(w in err_msg.lower() for w in ["rate limit", "rate_limit", "free mcp rate limit", "429"]):
                    raise RetryableError(f"Exa rate limit: {err_msg}", retry_after=2.0)
                raise RuntimeError(f"Exa error: {err_msg}")

            result = data.get("result", {})
            meta = result.get("_meta", {})
            if meta.get("rateLimited") or meta.get("rate_limited") or "rate limit" in str(meta).lower():
                raise RetryableError("Exa rate limited via _meta", retry_after=2.0)

            texts = []
            for item in result.get("content", []):
                if item.get("type") == "text":
                    t = item.get("text", "")
                    if "hit exa's free mcp rate limit" in t.lower():
                        raise RetryableError("Exa rate limit banner", retry_after=2.0)
                    texts.append(t)

            return "\n\n".join(texts).strip()

    return with_retry(_do_post, attempts=2, base=1.0, cap=3.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective.strip() or f"find comprehensive information about {query}"
        results_text = _call_exa_mcp("web_search_exa", {
            "query": query,
            "objective": obj,
            "numResults": max(1, min(int(num_results), 10)),
        })
        if not results_text:
            return "NO RESULTS"
        return results_text
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        results_text = _call_exa_mcp("web_fetch_exa", {
            "urls": [url],
        })
        if not results_text:
            return "NO RESULTS"
        return results_text[:12000]
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
