"""agents.py - STUDENT IMPLEMENTS.  The prompts, the subagents and the lead Deep Agent.   Guide: GUIDE.md, part 2.

Docs: https://docs.langchain.com/oss/python/deepagents/overview  (subagents: `subagents=[{...}]` of create_deep_agent)
"""
from deepagents import create_deep_agent  # noqa: F401
from langchain.agents.middleware import TodoListMiddleware  # noqa: F401

from tools import SOURCE_TOOLS, web_fetch  # noqa: F401

# ---- workspace contract (given; the whole team and research.py rely on these exact paths) ----
WORKDIR = "/tmp/work"
NOTES_DIR = f"{WORKDIR}/research/notes"                    # researcher notes: <NN>-<slug>.md
SOURCES_PATH = f"{WORKDIR}/research/sources.json"          # JSON array of {n, id, url, title, date, source}
VALIDATOR_PATH = f"{WORKDIR}/research/check_citations.py"  # YOUR validator, uploaded by research.py
FINALIZER_PATH = f"{WORKDIR}/research/finalize_citations.py"  # PROVIDED script, uploaded by research.py
REPORT_PATH = f"{WORKDIR}/report/report.md"                # the final report
# source is one of: "arxiv" | "hf-daily" | "hf-search" | "web"

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    TodoListMiddleware,
    ToolCallLimitMiddleware,
)

# Call & tool limits to prevent infinite loops (GUIDE 2.5 & RUBRIC 2.5)
LEAD_LIMITS = [
    ModelCallLimitMiddleware(run_limit=60, exit_behavior="end"),
    ToolCallLimitMiddleware(run_limit=100),
]
SUB_LIMITS = [
    ModelCallLimitMiddleware(run_limit=10, exit_behavior="end"),
    ToolCallLimitMiddleware(run_limit=12),
]

# ---- TODO 1: the lead prompt ----
LEAD_PROMPT = f"""You are the Lead Research Agent coordinating a deep research investigation.
You operate inside a sandbox environment with files located at absolute paths under {WORKDIR}.

Workspace layout:
- Notes directory: {NOTES_DIR}
- Sources file: {SOURCES_PATH}
- Citation validator script: {VALIDATOR_PATH}
- Citation finalizer script: {FINALIZER_PATH}
- Final report: {REPORT_PATH}

Objective: Produce a comprehensive research report at {REPORT_PATH} and an accurate sources file at {SOURCES_PATH}.

Follow these 8 steps strictly and sequentially:
1. PLAN:
   - Call `write_todos` to define your 3 subtopics.

2. DELEGATION (subagent_calls >= 3):
   - Make 3 separate calls to the `task` tool for the `researcher` subagent:
     * Task 1: Subtopic 1 (Foundations & Theory) - instruct researcher to use `arxiv_search`.
     * Task 2: Subtopic 2 (Architectures & Implementations) - instruct researcher to use `hf_search_papers`.
     * Task 3: Subtopic 3 (Recent Breakthroughs & Benchmarks) - instruct researcher to use `hf_daily_papers`.

3. SAVE NOTES (write_file):
   - As researcher subagents return their notes in text, use `write_file` to save them into:
     * `{NOTES_DIR}/01-foundations.md`
     * `{NOTES_DIR}/02-architectures.md`
     * `{NOTES_DIR}/03-recent.md`

4. MERGE SOURCES (write_file):
   - Extract unique sources from the notes and use `write_file` to save `{SOURCES_PATH}`.
   - Format: JSON array of objects with keys: `n` (1-indexed), `id`, `url`, `title`, `date`, `source`.
   - CRITICAL (RUBRIC 2.2): Must contain at least 3 source families among `arxiv`, `hf-search`, `hf-daily`. Ensure at least one source for each.
   - `url` for arXiv: `https://arxiv.org/abs/<id>`.
   - `url` for Hugging Face: `https://huggingface.co/papers/<id>`.

5. WRITE REPORT (write_file):
   - Write `{REPORT_PATH}` using `write_file` following REPORT_TEMPLATE.md:
     * `# <Topic Title>`
     * `## TL;DR`
     * `## Background & Motivation`
     * `## Theoretical Foundations and Core Principles`
     * `## Architectural Paradigms and Key Implementations`
     * `## Trends and Open Problems`
   - Use inline citations `[n]` referencing `{SOURCES_PATH}`. Cite sources from all 3 families.
   - DO NOT write the `## References` section (the finalizer script creates it automatically).

6. FINALIZE CITATIONS (execute):
   - Run `python3 {FINALIZER_PATH}` via `execute`.

7. VALIDATE CITATIONS (execute):
   - Run `python3 {VALIDATOR_PATH} {REPORT_PATH} {SOURCES_PATH}` via `execute`. Verify it outputs OK.

8. SPOT-CHECK:
   - Call `task` with `citation-checker` for 1 claim to complete verification.
"""

# ---- TODO 2: the researcher and citation-checker prompts ----
RESEARCHER_PROMPT = """You are an Academic Researcher subagent.
Your goal is to gather factual, reliable evidence for the delegated sub-question.

Available tools:
- `arxiv_search`: Search arXiv papers by keywords.
- `hf_daily_papers`: Get trending Hugging Face papers.
- `hf_search_papers`: Search Hugging Face papers catalog by topic or keyword.

CRITICAL RULES:
1. EFFICIENCY: Execute EXACTLY 1 search query using the tool requested by Lead Agent. Do NOT make more than 2 search queries total.
2. STOPPING: Once you get search results with at least 2-4 papers, STOP searching immediately!
3. FACTUAL GROUNDING: Record only facts explicitly in the retrieved text.
4. RETURN FORMAT: Return your structured notes directly in your final response message:
   ### Source: <Title>
   - id: <paper ID or slug>
   - url: <https://arxiv.org/abs/... or https://huggingface.co/papers/...>
   - date: <YYYY-MM-DD or year>
   - source: <arxiv | hf-daily | hf-search>
   - key_points:
     - <Key finding or architecture detail>
     - <Another specific detail>
"""

CHECKER_PROMPT = """You are a Citation Verification subagent.
Verify whether a claim matches its cited source URL.
Review the claim and return a 1-sentence verdict:
- Verdict: SUPPORTED / PARTIAL / UNSUPPORTED / UNVERIFIABLE
- Evidence: <one sentence>
"""


# ---- TODO 3: subagents ----
def build_subagents(model=None):
    """Return a list of subagent specs for create_deep_agent."""
    subagents = [
        {
            "name": "researcher",
            "description": (
                "Performs deep research on a sub-question. "
                "Delegate with: topic, sub-question, notes file path (/tmp/work/research/notes/<NN>-<slug>.md), "
                "and source families to prioritize."
            ),
            "system_prompt": RESEARCHER_PROMPT,
            "tools": SOURCE_TOOLS,
            "middleware": SUB_LIMITS,
        },
        {
            "name": "citation-checker",
            "description": (
                "Verifies that claims in the report match their cited source URLs. "
                "Delegate with: list of (claim, url) pairs."
            ),
            "system_prompt": CHECKER_PROMPT,
            "tools": [web_fetch],
            "middleware": SUB_LIMITS,
        },
    ]
    if model is not None:
        for spec in subagents:
            spec["model"] = model
    return subagents


# ---- TODO 4: the lead agent ----
def build_lead_agent(backend, model):
    """Return create_deep_agent configured with lead prompt, subagents, backend, and middleware limits."""
    return create_deep_agent(
        model=model,
        system_prompt=LEAD_PROMPT,
        subagents=build_subagents(model=model),
        backend=backend,
        middleware=[TodoListMiddleware(), *LEAD_LIMITS],
    )

