"""
Phase 2b: Dialogue Fetcher.

Input:  data/raw/bugs_index.json  (BugsInPy parsed bugs)
Output: data/raw/issues.json      (bugs + GitHub issue dialogue)

Pipeline (per bug):
  1. Fix commit ko GitHub se fetch karo.
  2. Commit message mein issue reference (#NNN) dhundo.
  3. Us issue ka title, body, comments laao = dialogue.
  4. Existing changed_files ko ground-truth label rakho.

Run:
  python -m src.data_collection.dialogue_fetcher
"""

import os
import re
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from github import Github, GithubException
from tqdm import tqdm


# ---------- Setup ----------
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN or TOKEN.startswith("ghp_your"):
    raise SystemExit("ERROR: GITHUB_TOKEN missing in .env file!")

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

IN_PATH = Path("data/raw/bugs_index.json")
OUT_PATH = Path(CFG["paths"]["raw_issues"])

# Patterns to detect issue number in commit message
ISSUE_REF_PATTERNS = [
    re.compile(r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*[:#]?\s*#?(\d+)", re.IGNORECASE),
    re.compile(r"#(\d+)"),  # fallback: any #1234
]


def extract_issue_number(commit_message):
    """Commit msg mein se issue number nikalo (priority: 'fixes #N' > '#N')."""
    for pat in ISSUE_REF_PATTERNS:
        m = pat.search(commit_message)
        if m:
            return int(m.group(1))
    return None


def fetch_issue_dialogue(repo, issue_num):
    """Issue ka title + body + comments laao."""
    try:
        issue = repo.get_issue(issue_num)
    except GithubException:
        return None

    # Sanity: ye issue hai PR nahi
    if issue.pull_request is not None:
        return None

    turns = [{
        "author": issue.user.login if issue.user else "unknown",
        "role": "reporter",
        "text": (issue.title or "") + "\n\n" + (issue.body or ""),
    }]
    try:
        for c in issue.get_comments():
            turns.append({
                "author": c.user.login if c.user else "unknown",
                "role": "commenter",
                "text": c.body or "",
            })
    except GithubException:
        pass

    return {
        "issue_number": issue.number,
        "issue_url": issue.html_url,
        "title": issue.title,
        "dialogue": turns,
    }


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Run bugsinpy_parser first. Missing: {IN_PATH}")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        bugs = json.load(f)

    print(f"Loaded {len(bugs)} bugs from BugsInPy index.")

    gh = Github(TOKEN, per_page=100)
    print(f"GitHub rate limit: {gh.get_rate_limit().core.remaining}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    results = []
    done_keys = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
            done_keys = {(r["repo"], r["bug_id"]) for r in results}
        print(f"Resuming - {len(results)} already done.")

    # Cache repo objects (avoid refetching same repo)
    repo_cache = {}

    pbar = tqdm(bugs, desc="Fetching dialogues")
    no_issue = 0
    fetched = 0

    for bug in pbar:
        key = (bug["repo"], bug["bug_id"])
        if key in done_keys:
            continue

        try:
            if bug["repo"] not in repo_cache:
                repo_cache[bug["repo"]] = gh.get_repo(bug["repo"])
            repo = repo_cache[bug["repo"]]

            # 1. Commit fetch karo
            try:
                commit = repo.get_commit(bug["fix_commit"])
            except GithubException:
                continue
            msg = commit.commit.message or ""

            # 2. Issue number nikalo
            issue_num = extract_issue_number(msg)
            if not issue_num:
                no_issue += 1
                continue

            # 3. Issue dialogue lao
            dlg = fetch_issue_dialogue(repo, issue_num)
            if not dlg:
                no_issue += 1
                continue

            results.append({
                "repo":          bug["repo"],
                "bug_id":        bug["bug_id"],
                "fix_commit":    bug["fix_commit"],
                "commit_message": msg.strip().split("\n")[0][:200],
                "fix_files":     bug["changed_files"],
                **dlg,
            })
            fetched += 1
            pbar.set_postfix(fetched=fetched, no_issue=no_issue)

            # Crash-safe save every 20
            if fetched % 20 == 0:
                with open(OUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        except GithubException as e:
            if e.status == 403:
                print("\nRate limit hit. Sleeping 60s...")
                time.sleep(60)
            else:
                print(f"\nSkipped {bug['repo']}#{bug['bug_id']}: {e}")
            continue

    # Final save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pbar.close()
    print(f"\nDone. Fetched: {len(results)}  |  No-issue-ref: {no_issue}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()