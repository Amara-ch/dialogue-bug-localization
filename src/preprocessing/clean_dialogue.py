"""
Phase 3: Dialogue cleaning + tokenization + train/val/test split.

Input:  data/raw/issues.json   (109 bugs with raw dialogue)
Output: data/processed/dataset.json
        {
          "splits": {"train": [...ids], "val": [...], "test": [...]},
          "samples": [
             {"id": "...", "repo": "...", "dialogue_text": "...",
              "input_ids": [...], "attention_mask": [...],
              "fix_files": [...], "all_repo_files": [...]},
             ...
          ]
        }

Run:
  python -m src.preprocessing.clean_dialogue
"""

import re
import json
import random
from pathlib import Path

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------- Config ----------
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

IN_PATH  = Path(CFG["paths"]["raw_issues"])
OUT_PATH = Path(CFG["paths"]["processed"])

BERT_NAME  = CFG["model"]["bert_name"]
MAX_TOKENS = CFG["model"]["max_dialogue_tokens"]
SEED       = CFG["training"]["seed"]
SPLITS     = (CFG["training"]["train_split"],
              CFG["training"]["val_split"],
              CFG["training"]["test_split"])


# ---------- Cleaning helpers ----------
URL_RE         = re.compile(r"https?://\S+|www\.\S+")
CODE_BLOCK_RE  = re.compile(r"```.*?```", re.DOTALL)        # fenced code
INLINE_CODE_RE = re.compile(r"`([^`]+)`")                    # inline `code`
HTML_TAG_RE    = re.compile(r"<[^>]+>")
QUOTE_LINE_RE  = re.compile(r"^>.*$", re.MULTILINE)         # GitHub quotes
EMOJI_SHORT_RE = re.compile(r":[a-z_]+:")                   # :smile:
WHITESPACE_RE  = re.compile(r"\s+")


def clean_text(text):
    """Single piece of text saaf karo."""
    if not text:
        return ""
    text = CODE_BLOCK_RE.sub(" [CODE] ", text)   # large code blocks placeholder
    text = INLINE_CODE_RE.sub(r"\1", text)        # keep inline code text
    text = QUOTE_LINE_RE.sub("", text)
    text = URL_RE.sub(" [URL] ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = EMOJI_SHORT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_dialogue_text(dialogue):
    """Multi-turn dialogue ko ek string mein convert karo."""
    parts = []
    for turn in dialogue:
        role = turn.get("role", "user").upper()
        cleaned = clean_text(turn.get("text", ""))
        if cleaned:
            parts.append(f"[{role}] {cleaned}")
    return " [SEP] ".join(parts)


# ---------- Candidate file pool ----------
def build_repo_file_pools(samples):
    """Har repo ka file pool: us repo ke saare fix_files ka union.

    Training mein har bug ke liye humein 'candidate files' chahiye —
    kuch positive (real fix files) + kuch negative (random other files).
    Yahan hum repo-level union banate hain.
    """
    pool = {}
    for s in samples:
        pool.setdefault(s["repo"], set()).update(s["fix_files"])
    return {r: sorted(files) for r, files in pool.items()}


# ---------- Main ----------
def main():
    random.seed(SEED)

    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run dialogue_fetcher first.")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        bugs = json.load(f)
    print(f"Loaded {len(bugs)} bugs.")

    # Load tokenizer
    print(f"Loading tokenizer: {BERT_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    # Build repo file pools
    repo_pools = build_repo_file_pools(bugs)
    for r, files in repo_pools.items():
        print(f"  {r}: {len(files)} unique fix-files in pool")

    # Process each bug
    samples = []
    for b in tqdm(bugs, desc="Cleaning + tokenizing"):
        text = build_dialogue_text(b["dialogue"])
        if not text or len(text) < 20:
            continue

        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_TOKENS,
            padding="max_length",
            return_attention_mask=True,
        )

        samples.append({
            "id":              f"{b['repo'].replace('/', '__')}__{b['bug_id']}",
            "repo":            b["repo"],
            "bug_id":          b["bug_id"],
            "issue_url":       b["issue_url"],
            "title":           b["title"],
            "dialogue_text":   text,
            "input_ids":       enc["input_ids"],
            "attention_mask":  enc["attention_mask"],
            "fix_files":       b["fix_files"],
            "candidate_files": repo_pools[b["repo"]],   # all files of that repo
        })

    print(f"\nKept {len(samples)} samples (>= 20 chars).")

    # Train/val/test split (stratified by repo so each split has diverse repos)
    by_repo = {}
    for s in samples:
        by_repo.setdefault(s["repo"], []).append(s["id"])

    train_ids, val_ids, test_ids = [], [], []
    for repo, ids in by_repo.items():
        random.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(n * SPLITS[0]))
        n_val   = max(0, int(n * SPLITS[1]))
        train_ids += ids[:n_train]
        val_ids   += ids[n_train:n_train + n_val]
        test_ids  += ids[n_train + n_val:]

    print(f"Split -> train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    out = {
        "config": {
            "bert_name":  BERT_NAME,
            "max_tokens": MAX_TOKENS,
            "seed":       SEED,
        },
        "splits": {
            "train": train_ids,
            "val":   val_ids,
            "test":  test_ids,
        },
        "samples": samples,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()