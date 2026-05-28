"""
Phase 4: Path-based code graph.

Reads:  data/processed/dataset.json  (har sample mein candidate_files hain)
Builds: NetworkX graph over UNIQUE files across all repos.

Edges:
  1. same-folder        weight = 1.0
  2. same parent folder weight = 0.5
  3. similar filename   weight = 0.7  (Jaccard on filename stems)

Saves:
  data/graphs/code_graph.pkl
    {
      "graph": nx.Graph,
      "node_to_id": {file_path: int},
      "id_to_node": {int: file_path},
      "repo_of_node": {file_path: repo},
    }

Run:
  python -m src.graph.build_graph
"""

import re
import json
import pickle
from pathlib import Path, PurePosixPath
from collections import defaultdict

import yaml
import networkx as nx
from tqdm import tqdm


with open("configs/config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

IN_PATH  = Path(CFG["paths"]["processed"])
OUT_PATH = Path(CFG["paths"]["code_graph"])


# ---------- Helpers ----------
TOKEN_RE = re.compile(r"[a-zA-Z]+")


def filename_tokens(path):
    """Filename stem ko lowercase tokens mein todo. e.g. 'parsers_v2.py' -> {parsers, v}"""
    stem = PurePosixPath(path).stem.lower()
    return set(TOKEN_RE.findall(stem))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------- Main ----------
def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Run Phase 3 first.")

    with open(IN_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)

    # 1. Collect unique files (scoped per repo to avoid collisions across repos)
    #    Node key = "repo::path"  (so pandas/foo.py != keras/foo.py)
    nodes = set()
    repo_of_node = {}
    files_per_repo = defaultdict(list)

    for s in ds["samples"]:
        repo = s["repo"]
        for f in s["candidate_files"]:
            key = f"{repo}::{f}"
            nodes.add(key)
            repo_of_node[key] = repo
            files_per_repo[repo].append(key)

    nodes = sorted(nodes)
    node_to_id = {n: i for i, n in enumerate(nodes)}
    id_to_node = {i: n for n, i in node_to_id.items()}
    print(f"Unique file nodes: {len(nodes)} across {len(files_per_repo)} repos")

    # 2. Build graph
    G = nx.Graph()
    for n in nodes:
        path = n.split("::", 1)[1]
        G.add_node(n,
                   repo=repo_of_node[n],
                   path=path,
                   tokens=list(filename_tokens(path)))

    edges_added = 0

    # 3. Edges within each repo only (cross-repo files unrelated)
    for repo, repo_nodes in tqdm(files_per_repo.items(), desc="Edges per repo"):
        # Group by folder
        folder_groups   = defaultdict(list)
        parent_groups   = defaultdict(list)
        for n in repo_nodes:
            path = PurePosixPath(n.split("::", 1)[1])
            folder = str(path.parent)
            parent = str(path.parent.parent) if path.parent != PurePosixPath(".") else "."
            folder_groups[folder].append(n)
            parent_groups[parent].append(n)

        # Edge type 1: same folder
        for folder, members in folder_groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    G.add_edge(members[i], members[j], weight=1.0, type="same_folder")
                    edges_added += 1

        # Edge type 2: same parent folder (only if not already same folder)
        for parent, members in parent_groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if G.has_edge(members[i], members[j]):
                        continue
                    G.add_edge(members[i], members[j], weight=0.5, type="same_parent")
                    edges_added += 1

        # Edge type 3: filename token similarity within repo
        tokens_cache = {n: filename_tokens(n.split("::", 1)[1]) for n in repo_nodes}
        for i in range(len(repo_nodes)):
            for j in range(i + 1, len(repo_nodes)):
                a, b = repo_nodes[i], repo_nodes[j]
                if G.has_edge(a, b):
                    continue
                sim = jaccard(tokens_cache[a], tokens_cache[b])
                if sim >= 0.5:
                    G.add_edge(a, b, weight=0.7 * sim, type="name_sim")
                    edges_added += 1

    print(f"Total edges added: {edges_added}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Avg degree: {sum(dict(G.degree()).values()) / max(1, G.number_of_nodes()):.2f}")

    # 4. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "graph":         G,
            "node_to_id":    node_to_id,
            "id_to_node":    id_to_node,
            "repo_of_node":  repo_of_node,
        }, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()