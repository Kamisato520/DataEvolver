#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
latexmk -pdf main.tex
