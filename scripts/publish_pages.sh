#!/usr/bin/env bash
# Publishes the inspector (web/, with both trace indexes) to the gh-pages
# branch, served by GitHub Pages. The branch holds a single commit that every
# publish replaces, so the ~50 MB of trace data never piles up in history and
# main stays light. Build the data first: make index, make index CORE=pipe,
# make web.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

test -f web/data/runs.json || { echo "web/data is missing: run make web first" >&2; exit 1; }

# Every chunk an index promises must be there, or the page breaks on zoom.
python3 - <<'PY'
import json, math, pathlib, sys
for run in json.load(open("web/data/runs.json"))["runs"]:
    base = pathlib.Path("web") / run["path"]
    idx = json.load(open(base / "index.json"))
    want = math.ceil(idx["cycles"] / idx["chunk_cycles"])
    have = len(list((base / "chunks").glob("*.bin")))
    if have != want:
        sys.exit(f"{base}: {have} chunks, the index needs {want}")
    print(f"{base}: {idx['cycles']} cycles, {want} chunks")
PY

remote="$(git remote get-url origin)"
source_rev="$(git rev-parse --short HEAD)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -r web/. "$tmp/"
touch "$tmp/.nojekyll"   # serve files as they are, no Jekyll processing

git -C "$tmp" init -q -b gh-pages
git -C "$tmp" add -A
git -C "$tmp" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" \
  commit -q -F - <<MSG
Publish the inspector built from ${source_rev}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
git -C "$tmp" push -q -f "$remote" gh-pages
echo "published web/ from ${source_rev} to the gh-pages branch"
