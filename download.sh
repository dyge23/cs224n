#!/bin/bash
# Download all CS224N assignments.
#
#   A1-A3: Winter 2026 offering (web.stanford.edu/class/cs224n/assignments_w26)
#   A4-A5: Winter 2024 offering (cs224n.1244), which does not include the
#          LLM Evals assignment: A4 = Neural Machine Translation,
#          A5 = Self-Supervised Learning and Fine-tuning with Transformers
#
# Assignments already present on disk are skipped, so re-running is safe.

set -e

W26="https://web.stanford.edu/class/cs224n/assignments_w26"
W24="https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/assignments"

# fetch <url> <file> -- download a single file unless it already exists
fetch() {
    if [ -e "$2" ]; then
        echo "skip (already present): $2"
        return
    fi
    echo "download: $2"
    aria2c -x 8 -c -d . -o "$2" "$1"
}

# fetch_zip <url> <dir> -- download and extract an assignment zip into <dir>
fetch_zip() {
    if [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
        echo "skip (already present): $2/"
        return
    fi
    echo "download + extract: $2/"
    mkdir -p "$2"
    tmp="$(mktemp -d)"
    aria2c -x 8 -c -d "$tmp" -o a.zip "$1"
    unzip -o -q "$tmp/a.zip" -d "$2"
    rm -f "$tmp/a.zip"
    rmdir "$tmp" 2>/dev/null || true
    rm -rf "$2/__MACOSX"
    # flatten a single top-level wrapper dir (e.g. "student/")
    if [ -z "$(find "$2" -maxdepth 1 -type f 2>/dev/null)" ] && \
       [ "$(find "$2" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" -eq 1 ]; then
        wrapper="$(find "$2" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
        shopt -s dotglob
        mv "$wrapper"/* "$2"/
        shopt -u dotglob
        rmdir "$wrapper"
    fi
}

# ---- Winter 2026: A1-A3 ----
fetch_zip "$W26/a1.zip" a1
fetch_zip "$W26/a2.zip" a2
fetch_zip "$W26/a3.zip" a3
fetch "$W26/a2.pdf" a2.pdf
fetch "$W26/a3.pdf" a3.pdf

# ---- Winter 2024: A4-A5 ----
fetch_zip "$W24/a4_student_code.zip" a4
fetch "$W24/a4_student_handout.pdf" a4.pdf
fetch_zip "$W24/a5_student_code.zip" a5
fetch "$W24/a5_student_handout.pdf" a5.pdf

echo "Done."
