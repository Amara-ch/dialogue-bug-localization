"""
Phase 2: GitHub Issue Scraper

Kya karta hai:
  1. Target repo (Flask) se closed issues uthata hai.
  2. Har issue ka title, body, comments collect karta hai (= dialogue).
  3. Issue se linked Pull Request / fix commit dhundta hai.
  4. PR ki changed Python files = ground truth labels.
  5. Sab data/raw/issues.json mein save karta hai.

Chalao:
  python -m src.data_collection.github_scraper
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


# ---------- Config + Auth ----------
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN or TOKEN.startswith("ghp_your"):
    raise SystemExit("ERROR: GITHUB_TOKEN missing in .env file!")

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

TARGET_REPO = CFG["target_repo"]
MAX_ISSUES = CFG["max_issues"]
LABELS = CFG["labels_filter"]
OUT_PATH = Path(CFG["paths"]["raw_issues"])


# ---------- Helpers ----------
PR_LINK_PATTERN = re.compile(r"(?:#|pull/|pull-request/)(\d+)")


def find_linked_pr(issue, repo):
    """Issue ke events + body + comments mein dhundo koi PR/commit jisne ise close kiya."""
    # 1. Events check karo
    try:
        for event in issue.get_events():
            if event.event == "closed" and event.commit_id:
                return {"type": "commit", "sha": event.commit_id}
    except GithubException:
        pass

    # 2. Issue text mein PR number dhundo
    text = (issue.title or "") + " " + (issue.body or "")
    try:
        for c in issue.get_comments():
            text += " " + (c.body or "")
    except GithubException:
        pass

    seen = set()
    for match in PR_LINK_PATTERN.finditer(text):
        pr_num = int(match.group(1))
        if pr_num in seen:
            continue
        seen.add(pr_num)
        try:
            pr = repo.get_pull(pr_num)
            if pr.merged:
                return {"type": "pr", "number": pr_num}
        except GithubException:
            continue
    return None


def get_changed_files(repo, link):
    """Linked PR ya commit se changed Python files ki list nikalo."""
    files = []
    try:
        if link["type"] == "pr":
            pr = repo.get_pull(link["number"])
            files = [f.filename for f in pr.get_files()]
        elif link["type"] == "commit":
            commit = repo.get_commit(link["sha"])
            files = [f.filename for f in commit.files]
    except GithubException:
        return []

    # sirf .py source files rakho, test files hata do
    return [
        f for f in files
        if f.endswith(".py")
        and not f.startswith("tests/")
        and "test_" not in f
        and "/tests/" not in f
    ]


def extract_dialogue(issue):
    """Issue ko multi-turn dialogue mein convert karo."""
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
    return turns


# ---------- Main scraper ----------
def main():
    gh = Github(TOKEN, per_page=100)
    repo = gh.get_repo(TARGET_REPO)
    print(f"Connected to {TARGET_REPO}. Rate limit: {gh.get_rate_limit().core.remaining}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    collected = []
    seen_ids = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            collected = json.load(f)
            seen_ids = {x["number"] for x in collected}
        print(f"Resuming - {len(collected)} issues already saved.")

    # Labels filter (empty = saari closed issues)
    if LABELS:
        issues_iter = repo.get_issues(state="closed", labels=LABELS,
                                      sort="created", direction="desc")
    else:
        issues_iter = repo.get_issues(state="closed",
                                      sort="created", direction="desc")

    pbar = tqdm(total=MAX_ISSUES, initial=len(collected), desc="Scraping")
    scanned = 0

    for issue in issues_iter:
        if len(collected) >= MAX_ISSUES:
            break
        if issue.pull_request is not None:   # skip PRs
            continue
        if issue.number in seen_ids:
            continue

        scanned += 1
        try:
            link = find_linked_pr(issue, repo)
            if not link:
                continue
            files = get_changed_files(repo, link)
            if not files:
                continue

            collected.append({
                "number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "created_at": issue.created_at.isoformat(),
                "dialogue": extract_dialogue(issue),
                "fix_files": files,
                "link_info": link,
            })
            pbar.update(1)
            pbar.set_postfix(scanned=scanned, kept=len(collected))

            # Crash-safe save
            if len(collected) % 25 == 0:
                with open(OUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(collected, f, indent=2, ensure_ascii=False)

        except GithubException as e:
            if e.status == 403:
                print("\nRate limit hit. Sleeping 60s...")
                time.sleep(60)
            else:
                print(f"\nSkipping #{issue.number}: {e}")
            continue

    # Final save
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)

    pbar.close()
    print(f"\nDone. Scanned {scanned}, kept {len(collected)} issues -> {OUT_PATH}")


if __name__ == "__main__":
    main()