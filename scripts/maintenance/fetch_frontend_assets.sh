#!/bin/bash
set -euo pipefail

# Refresh local frontend assets used by cluster_editor_vue.py.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATIC_DIR="${ROOT_DIR}/static"

VUE_VERSION="${CLUSTER_EDITOR_VUE_VERSION:-3.3.9}"
THREE_VERSION="${CLUSTER_EDITOR_THREE_VERSION:-0.164.0}"

mkdir -p "${STATIC_DIR}"

curl -fsSL "https://cdn.jsdelivr.net/npm/vue@${VUE_VERSION}/dist/vue.esm-browser.prod.js" -o "${STATIC_DIR}/vue.esm-browser.prod.js"
curl -fsSL "https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.module.js" -o "${STATIC_DIR}/three.module.js"
curl -fsSL "https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/examples/jsm/controls/TrackballControls.js" -o "${STATIC_DIR}/TrackballControls.js"

echo "Updated local frontend assets in ${STATIC_DIR}"
ls -lh "${STATIC_DIR}"
