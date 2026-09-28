#!/usr/bin/env bash
# Build the Delta patched tree from code/constituent-study-20260922.
#
#   code/delta/apply.sh <build-dir> [last-patch-number]
#
# Copies code/constituent-study-20260922 to <build-dir>/code after checking its content hash,
# commits it in a throwaway git repository inside <build-dir> (git apply --3way needs the
# blobs), copies newmods/ into <build-dir>/code/newmods/, then applies patches/NNNN-*.patch in
# order. Patches 0023 and 0024 import newmods. The module tests in tests/ are copied to
# <build-dir>/code/tests/ at the end. An optional last-patch-number stops early (e.g. 0006; the
# tests are then not copied). Never touches this repository.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE="$HERE/../constituent-study-20260922"
EXPECT=ad3e295ccfd24971415332074c7a7251253609eca2c36d5850e92a88ba132c74
OUT=${1:?usage: apply.sh <build-dir> [last-patch-number]}
LAST=${2:-9999}

# Content hash of the base tree: sha256 over sorted "path sha256" lines, one per file.
GOT=$(cd "$BASE" && python3 -c '
import hashlib, os
rows = []
for root, _, files in os.walk("."):
    for f in files:
        p = os.path.join(root, f)[2:]
        if "__pycache__" in p.split(os.sep):
            continue
        rows.append(p + " " + hashlib.sha256(open(p, "rb").read()).hexdigest())
print(hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest())')
[ "$GOT" = "$EXPECT" ] || { echo "BASE_TREE_HASH_MISMATCH $GOT" >&2; exit 1; }
[ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT")" ] || { echo "build dir $OUT is not empty" >&2; exit 1; }
mkdir -p "$OUT/code"
cp -R "$BASE"/. "$OUT/code/"
find "$OUT/code" -name __pycache__ -prune -exec rm -r {} +
cd "$OUT"
G="git -c user.name=apply -c user.email=apply@example.invalid"
git init -q
$G add -A
$G commit -qm "base: code/constituent-study-20260922"
mkdir -p code/newmods
cp "$HERE"/newmods/*.py code/newmods/
$G add -A
$G commit -qm "newmods"
for p in "$HERE"/patches/[0-9][0-9][0-9][0-9]-*.patch; do
  n=$(basename "$p" | cut -c1-4)
  [ "$((10#$n))" -le "$((10#$LAST))" ] || break
  git apply --3way "$p"
  $G add -A
  $G commit -qm "$(basename "$p" .patch)"
  echo "APPLIED $(basename "$p")"
done
if [ "$LAST" = 9999 ]; then
  mkdir -p code/tests
  cp "$HERE"/tests/*.py code/tests/
  echo "TESTS_COPIED"
fi
echo "APPLY_ALL_PASS $OUT/code"
