"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not isinstance(topic, str):
        return "topic"
    text = topic.strip().lower()
    # Collapse non-alphanumeric chars into hyphen
    slug = re.sub(r"[^\w]+", "-", text).strip("-")
    slug = slug[:60].strip("-")
    return slug if slug else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a deep research study on: '{topic}'.\n\n"
        f"Required workflow:\n"
        f"1. Plan with write_todos and split this topic into at least 3 distinct sub-questions.\n"
        f"2. Delegate at least 3 sub-questions to researcher subagents using the task tool (e.g. Subtopic 1 to arxiv_search, Subtopic 2 to hf_search_papers, Subtopic 3 to hf_daily_papers).\n"
        f"3. Researcher subagents return notes in messages. Use your write_file tool to save their notes into {WORKDIR}/research/notes/ (e.g. 01-foundations.md, 02-architectures.md, 03-recent.md).\n"
        f"4. Use write_file to save merged sources into {SOURCES_PATH}. Ensure it spans at least 3 source families (arxiv, hf-daily, hf-search).\n"
        f"5. Write the report body to {REPORT_PATH} using write_file following REPORT_TEMPLATE.md (cite [n] from all 3 source families, do NOT write ## References).\n"
        f"6. Run `python3 {FINALIZER_PATH}` using execute.\n"
        f"7. Run `python3 {VALIDATOR_PATH} {REPORT_PATH} {SOURCES_PATH}` using execute and verify OK.\n"
        f"8. Delegate 1-2 claims to citation-checker via task.\n"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = msg.tool_calls
        elif isinstance(msg, dict) and "tool_calls" in msg:
            tool_calls = msg["tool_calls"] or []

        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_counts[name] += 1

        meta = getattr(msg, "usage_metadata", None)
        if not meta and isinstance(msg, dict):
            meta = msg.get("usage_metadata")
        if meta and isinstance(meta, dict):
            input_tokens += meta.get("input_tokens", 0) or meta.get("prompt_tokens", 0) or 0
            output_tokens += meta.get("output_tokens", 0) or meta.get("completion_tokens", 0) or 0

    subagent_calls = tool_counts.get("task", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": int(subagent_calls),
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("report is missing or empty")

    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json is missing or empty")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise RuntimeError("sources.json must be a JSON list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)

    families = sorted({s.get("source") for s in sources_data if isinstance(s, dict) and s.get("source")})
    n_sources = len(sources_data)

    summary_info = summarize(messages, elapsed, model_name)
    meta = {
        "topic": topic,
        **summary_info,
        "n_sources": n_sources,
        "source_families": families,
    }

    md_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"

    sources_path.write_text(json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_bytes(report_bytes)

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model = make_model()
    # Configure rate limiter and max_retries to respect provider rate limits
    if hasattr(model, "rate_limiter"):
        from langchain_core.rate_limiters import InMemoryRateLimiter
        model.rate_limiter = InMemoryRateLimiter(requests_per_second=0.2, max_bucket_size=2)
    if hasattr(model, "max_retries"):
        model.max_retries = 10
    if hasattr(model, "root_client"):
        try:
            model.root_client.max_retries = 10
        except Exception:
            pass

    model_name = os.getenv("LAB_MODEL") or getattr(model, "model_name", None) or getattr(model, "model", "model")
    print(f"[*] Starting research for: {topic}", flush=True)
    start = time.monotonic()

    with open_sandbox() as backend:
        print("[*] Sandbox opened. Initializing workspace...", flush=True)
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        print("[*] Invoking Deep Agent workflow...", flush=True)
        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(topic)}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        print(f"[*] Agent completed in {elapsed:.1f}s. Processing outputs...", flush=True)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        try:
            saved_path = save_outputs(backend, topic, messages, elapsed, model_name)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr, flush=True)
            return 1

    print(f"[✓] Report saved to: {saved_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
