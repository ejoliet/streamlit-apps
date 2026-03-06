#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
streamlit run ssc-doc-portal/app.py > run.log 2>&1

echo "Streamlit exited. Logs captured in $(pwd)/run.log"
