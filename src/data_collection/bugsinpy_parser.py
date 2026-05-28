"""
Phase 2a: BugsInPy parser (offline, no internet).

Reads data/external/BugsInPy/projects/<proj>/bugs/<id>/
  - bug.info       -> fixed_commit_id
  - bug_patch.txt  -> changed files (from 'diff --git a/X b/X' lines)

Saves: data/raw/bugs_index.json
[
  {"repo": "pandas-dev/pandas", "bug_id": 1,
   "fix_commit": "e41ee47...", "changed_files": ["pandas/core/dtypes/common.py"]},
  ...
]

Run:
  python -m src.data_collection.bugsinpy_parser
"""

import re
import json
from pathlib import Path
from tqdm import tqdm

BUGSINPY_ROOT = Path("data/external/BugsInPy/projects")
OUT_PATH = Path("data/raw/bugs_index.json")

# BugsInPy project folder name -> actual GitHub repo
REPO_MAP = {
    "ansible":      "ansible/ansible",
    "black":        "psf/black",
    "cookiecutter": "cookiecutter/cookiecutter",
    "fastapi":      "tiangolo/fastapi",
    "httpie":       "httpie/httpie",
    "keras":        "keras-team/keras",
    "luigi":        "spotify/luigi",
    "matplotlib":   "matplotlib/matplotlib",
    "pandas":       "pandas-dev/pandas",
    "PySnooper":    "cool-RR/PySnooper",
    "sanic":        "sanic-org/sanic",
    "scrapy":       "scrapy/scrapy",
    "spacy":        "explosion/spaCy",
    "thefuck":      "nvbn/thefuck",
    "tornado":      "tornadoweb/tornado",
    "tqdm":         "tqdm/tqdm",
    "youtube-dl":   "ytdl-org/youtube-dl",
}

DIFF_LINE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)


def parse_bug_info(path):
    """bug.info -> dict with fixed_commit_id."""
    info = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip().strip('"')
    return info


def parse_patch_files(path):
    """bug_patch.txt -> list of changed .py files (no test files)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    files = set()
    for m in DIFF_LINE.finditer(text):
        # 'a' path = old file, 'b' path = new file (same usually)
        f = m.group(2)
        if (f.endswith(".py")
                and "/tests/" not in f
                and not f.startswith("tests/")
                and "test_" not in Path(f).name):
            files.add(f)
    return sorted(files)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_bugs = []
    skipped = 0

    project_dirs = [p for p in BUGSINPY_ROOT.iterdir() if p.is_dir()]
    for proj_dir in tqdm(project_dirs, desc="Projects"):
        proj_name = proj_dir.name
        repo = REPO_MAP.get(proj_name)
        if not repo:
            continue

        bugs_dir = proj_dir / "bugs"
        if not bugs_dir.exists():
            continue

        for bug_dir in sorted(bugs_dir.iterdir(),
                              key=lambda x: int(x.name) if x.name.isdigit() else 1e9):
            if not bug_dir.is_dir():
                continue
            info_file = bug_dir / "bug.info"
            patch_file = bug_dir / "bug_patch.txt"
            if not (info_file.exists() and patch_file.exists()):
                skipped += 1
                continue

            info = parse_bug_info(info_file)
            commit = info.get("fixed_commit_id", "").strip()
            files = parse_patch_files(patch_file)

            if not commit or not files:
                skipped += 1
                continue

            all_bugs.append({
                "repo":          repo,
                "bugsinpy_proj": proj_name,
                "bug_id":        int(bug_dir.name),
                "fix_commit":    commit,
                "changed_files": files,
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_bugs, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\nTotal bugs parsed: {len(all_bugs)}")
    print(f"Skipped:           {skipped}")
    print(f"Saved to:          {OUT_PATH}")

    # Per-repo counts
    from collections import Counter
    counts = Counter(b["repo"] for b in all_bugs)
    print("\nPer-repo counts:")
    for r, c in counts.most_common():
        print(f"  {c:4d}  {r}")


if __name__ == "__main__":
    main()