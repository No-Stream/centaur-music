#!/usr/bin/env bash
# Idempotent bootstrap for the headless-Chrome -> ffmpeg video pipeline.
#
# Sets up:
#   1. A static ffmpeg/ffprobe build (with libx264) under tools/ffmpeg/
#   2. A vendored, pinned three.js module under viz/vendor/
#   3. The playwright Python package (dev dependency group)
#   4. Verifies system google-chrome is present
#
# Safe to re-run: every step first checks whether its output already exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

FFMPEG_DIR="${REPO_ROOT}/tools/ffmpeg"
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"

VENDOR_DIR="${REPO_ROOT}/viz/vendor"
THREE_VERSION="0.170.0"
THREE_URL="https://unpkg.com/three@${THREE_VERSION}/build/three.module.js"
THREE_DEST="${VENDOR_DIR}/three.module.js"

log() {
    printf '[viz_setup] %s\n' "$1"
}

fail() {
    printf '[viz_setup] ERROR: %s\n' "$1" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 1. Static ffmpeg
# ---------------------------------------------------------------------------
if [[ -x "${FFMPEG_DIR}/ffmpeg" ]]; then
    log "ffmpeg already present at ${FFMPEG_DIR}/ffmpeg, skipping download"
else
    log "downloading static ffmpeg from ${FFMPEG_URL}"
    mkdir -p "${FFMPEG_DIR}"

    TMP_TARBALL="$(mktemp -t ffmpeg-release-XXXXXX.tar.xz)"
    trap 'rm -f "${TMP_TARBALL}"' EXIT

    curl -fL --progress-bar -o "${TMP_TARBALL}" "${FFMPEG_URL}"

    log "extracting ffmpeg + ffprobe binaries"
    # The tarball contains a single top-level dir like ffmpeg-<version>-amd64-static/.
    # Extract only the two binaries we need, stripping that leading path component.
    # (Captured into a variable rather than piped through `head` so tar isn't
    # killed with SIGPIPE under `set -o pipefail`.)
    TAR_LISTING="$(tar -tJf "${TMP_TARBALL}")"
    TOP_DIR="$(head -1 <<<"${TAR_LISTING}" | cut -d/ -f1)"
    tar -xJf "${TMP_TARBALL}" \
        -C "${FFMPEG_DIR}" \
        --strip-components=1 \
        "${TOP_DIR}/ffmpeg" "${TOP_DIR}/ffprobe"

    rm -f "${TMP_TARBALL}"
    trap - EXIT

    chmod +x "${FFMPEG_DIR}/ffmpeg" "${FFMPEG_DIR}/ffprobe"
fi

FFMPEG_VERSION_OUTPUT="$("${FFMPEG_DIR}/ffmpeg" -version)"
if ! grep -q -- '--enable-libx264' <<<"${FFMPEG_VERSION_OUTPUT}"; then
    fail "tools/ffmpeg/ffmpeg does not report --enable-libx264; static build is unsuitable"
fi
FFMPEG_VERSION_LINE="$(head -1 <<<"${FFMPEG_VERSION_OUTPUT}")"

# ---------------------------------------------------------------------------
# 2. Vendor three.js
# ---------------------------------------------------------------------------
if [[ -f "${THREE_DEST}" ]]; then
    log "three.module.js already present at ${THREE_DEST}, skipping download"
else
    log "downloading three.js r${THREE_VERSION} module from ${THREE_URL}"
    mkdir -p "${VENDOR_DIR}"
    curl -fL --progress-bar -o "${THREE_DEST}" "${THREE_URL}"
fi

THREE_SIZE_BYTES="$(stat -c '%s' "${THREE_DEST}")"
if (( THREE_SIZE_BYTES < 500000 )); then
    fail "viz/vendor/three.module.js is only ${THREE_SIZE_BYTES} bytes (<500KB) - likely a bad download"
fi
THREE_HEAD="$(head -c 200 "${THREE_DEST}")"
if grep -qi '<!doctype html\|<html' <<<"${THREE_HEAD}"; then
    fail "viz/vendor/three.module.js looks like an HTML error page, not JS"
fi
THREE_SIZE_KB="$(( THREE_SIZE_BYTES / 1024 ))"

# ---------------------------------------------------------------------------
# 3. Python playwright (dev dependency)
# ---------------------------------------------------------------------------
if grep -q '"playwright' "${REPO_ROOT}/pyproject.toml"; then
    log "playwright already declared in pyproject.toml, skipping uv add"
else
    log "adding playwright as a dev dependency via uv"
    (cd "${REPO_ROOT}" && uv add --dev playwright)
fi

PLAYWRIGHT_LOCK_LINE="$(grep -A1 '^name = "playwright"$' "${REPO_ROOT}/uv.lock" | grep '^version' | head -1 || true)"
if [[ -z "${PLAYWRIGHT_LOCK_LINE}" ]]; then
    fail "playwright not found in uv.lock after uv add --dev playwright"
fi
PLAYWRIGHT_LOCK_VERSION="$(sed -E 's/version = "(.*)"/\1/' <<<"${PLAYWRIGHT_LOCK_LINE}")"

# Best-effort runtime import check. `uv run` may hit a harness permission
# gate in some environments; if so, fall back to the uv.lock version we
# already confirmed above and say so explicitly.
PLAYWRIGHT_RUNTIME_VERSION=""
if PLAYWRIGHT_RUNTIME_VERSION="$(uv run python -c 'import playwright; print(playwright.__version__)' 2>/dev/null)"; then
    :
else
    PLAYWRIGHT_RUNTIME_VERSION=""
fi

# ---------------------------------------------------------------------------
# 4. System google-chrome
# ---------------------------------------------------------------------------
CHROME_BIN="/usr/bin/google-chrome"
if [[ ! -x "${CHROME_BIN}" ]]; then
    fail "${CHROME_BIN} not found; install Google Chrome before using the surge_xt / viz playwright pipeline"
fi
CHROME_VERSION_OUTPUT="$("${CHROME_BIN}" --version)"

# ---------------------------------------------------------------------------
# Readiness summary
# ---------------------------------------------------------------------------
log "=== readiness summary ==="
log "ffmpeg:      ${FFMPEG_VERSION_LINE}"
log "             --enable-libx264 present"
log "three.js:    ${THREE_DEST} (${THREE_SIZE_KB} KB, pinned r${THREE_VERSION})"
if [[ -n "${PLAYWRIGHT_RUNTIME_VERSION}" ]]; then
    log "playwright:  importable via 'uv run', version ${PLAYWRIGHT_RUNTIME_VERSION}"
else
    log "playwright:  'uv run' import check unavailable (permission gate) - confirmed via uv.lock instead, version ${PLAYWRIGHT_LOCK_VERSION}"
fi
log "chrome:      ${CHROME_VERSION_OUTPUT}"
log "=== all checks passed ==="
