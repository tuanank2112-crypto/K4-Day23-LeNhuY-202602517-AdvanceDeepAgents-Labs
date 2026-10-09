"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
from collections import Counter
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_URL = re.compile(r"https?://[^\s()<>]+(?:\([^\s()<>]+\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not sources or not isinstance(sources, list):
        return ["no sources in sources.json"]

    source_by_n = {}
    seen_urls = set()
    for s in sources:
        if not isinstance(s, dict):
            problems.append(f"invalid source entry: {s!r}")
            continue
        n = s.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source {s} does not have integer n")
            continue
        if n in source_by_n:
            problems.append(f"duplicate source number n={n}")
        source_by_n[n] = s

        url = s.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid url {url!r}")
        else:
            if url in seen_urls:
                problems.append(f"duplicate url {url} in sources.json")
            seen_urls.add(url)

    # 3. Check heading ## References
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing heading '## References'")
        body = report_text
        ref_section = ""
    else:
        body = report_text[:matches[-1].start()]
        ref_section = report_text[matches[-1].end():]

    # 4. Check citations in body
    segments = _CODE.split(body)
    cited = set()
    for i, seg in enumerate(segments):
        if i % 2:
            continue
        for m in _GROUP.finditer(seg):
            for num in _group_numbers(m.group(1)):
                cited.add(num)

    for c in sorted(cited):
        if c not in source_by_n:
            problems.append(f"[{c}] cited in body but missing from sources.json")

    for n in sorted(source_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # 5 & 6. Check References section
    if matches:
        ref_lines = re.findall(r"(?m)^\[(\d+)\][ \t]*(.*)$", ref_section)
        ref_numbers = []
        for n_str, rest in ref_lines:
            ref_num = int(n_str)
            ref_numbers.append(ref_num)
            full_line = f"[{n_str}] {rest}"
            urls = _URL.findall(full_line)
            if len(urls) == 0:
                problems.append(f"reference line [{ref_num}] has no URL")
            elif len(urls) > 1:
                problems.append(f"reference line [{ref_num}] has multiple URLs (bundled sources)")
            else:
                line_url = urls[0]
                if ref_num in source_by_n:
                    expected_url = source_by_n[ref_num].get("url", "")
                    if line_url != expected_url:
                        problems.append(
                            f"reference line [{ref_num}] URL mismatch: expected {expected_url}, got {line_url}"
                        )

        # Check reference line counts and uniqueness
        ref_counter = Counter(ref_numbers)
        for n, count in ref_counter.items():
            if count > 1:
                problems.append(f"reference line [{n}] appears {count} times")
            if n not in source_by_n:
                problems.append(f"reference line [{n}] is not in sources.json")

        for n in sorted(source_by_n.keys()):
            if n not in ref_counter:
                problems.append(f"missing reference line for [{n}]")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
