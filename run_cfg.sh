#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./run_cfg.sh import ELF_PATH [OUT_DIR] [normal|fast]
#   ./run_cfg.sh reuse PROGRAM_NAME [OUT_DIR] [normal|fast]
#
# Backward-compatible import form:
#   ./run_cfg.sh ELF_PATH [OUT_DIR] [normal|fast]
#
# Examples:
#   ./run_cfg.sh import /workspace/bin/app /workspace/bin/app_cfg normal
#   ./run_cfg.sh reuse app /workspace/bin/app_cfg normal
#   PROJECT_DIR=/workspace/ghidra_projects PROJECT_NAME=elf.cfg \
#       ./run_cfg.sh reuse app /workspace/bin/app_cfg normal
#
# import: import/overwrite the ELF and run Ghidra Auto Analysis before extraction.
# reuse:  open an existing project program, skip Auto Analysis, and only extract.
# Every setting can also be supplied through an environment variable.
GHIDRA_HOME="${GHIDRA_HOME:-/workspace/tools/ghidra_12.0.3_PUBLIC}"
PROJECT_DIR="${PROJECT_DIR:-/workspace}"
PROCESSOR="${PROCESSOR:-}"

usage() {
    printf '%s\n' \
        "usage:" \
        "  $0 import ELF_PATH [OUT_DIR] [normal|fast]" \
        "  $0 reuse PROGRAM_NAME [OUT_DIR] [normal|fast]" \
        "  $0 ELF_PATH [OUT_DIR] [normal|fast]  # legacy import form" \
        "" \
        "environment overrides:" \
        "  GHIDRA_HOME, PROJECT_DIR, PROJECT_NAME, SCRIPT_DIR, OUT_DIR," \
        "  MODE, PROCESSOR (import only), ELF_PATH, PROGRAM_NAME"
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

case "$1" in
    import|reuse)
        OPERATION="$1"
        shift
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        # Preserve the original interface: a leading ELF path means import.
        OPERATION="import"
        ;;
esac

if (( $# > 3 )); then
    echo "error: too many arguments" >&2
    usage >&2
    exit 2
fi

case "${OPERATION}" in
    import)
        ELF_PATH="${1:-${ELF_PATH:-}}"
        if [[ -z "${ELF_PATH}" ]]; then
            echo "error: import mode requires ELF_PATH" >&2
            usage >&2
            exit 2
        fi
        ELF_BASENAME="$(basename -- "${ELF_PATH}")"
        PROGRAM_NAME="${PROGRAM_NAME:-${ELF_BASENAME}}"
        PROJECT_NAME="${PROJECT_NAME:-${ELF_BASENAME}_cfg}"
        DEFAULT_OUT_DIR="${ELF_PATH}_cfg_output"
        ;;
    reuse)
        PROGRAM_NAME="${1:-${PROGRAM_NAME:-}}"
        if [[ -z "${PROGRAM_NAME}" ]]; then
            echo "error: reuse mode requires PROGRAM_NAME" >&2
            usage >&2
            exit 2
        fi
        PROGRAM_BASENAME="$(basename -- "${PROGRAM_NAME}")"
        PROJECT_NAME="${PROJECT_NAME:-${PROGRAM_BASENAME}_cfg}"
        DEFAULT_OUT_DIR="${PROGRAM_BASENAME}_cfg_output"
        ;;
esac

OUT_DIR="${2:-${OUT_DIR:-${DEFAULT_OUT_DIR}}}"
MODE="${3:-${MODE:-normal}}"

# Keep extract_cfg.py beside this launcher.  SCRIPT_DIR may be overridden when
# the Ghidra script is stored elsewhere.
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${RUN_DIR}}"
SCRIPT_NAME="extract_cfg.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
PYGHIDRA_RUN="${GHIDRA_HOME}/support/pyghidraRun"

case "${MODE}" in
    normal|fast)
        ;;
    *)
        echo "error: mode must be 'normal' or 'fast', got '${MODE}'" >&2
        exit 2
        ;;
esac

if [[ ! -x "${PYGHIDRA_RUN}" ]]; then
    echo "error: pyghidraRun is not executable: ${PYGHIDRA_RUN}" >&2
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "error: Ghidra script does not exist: ${SCRIPT_PATH}" >&2
    exit 1
fi

case "${OPERATION}" in
    import)
        if [[ ! -f "${ELF_PATH}" ]]; then
            echo "error: ELF file does not exist: ${ELF_PATH}" >&2
            exit 1
        fi
        mkdir -p -- "${PROJECT_DIR}" "${OUT_DIR}"
        ;;
    reuse)
        PROJECT_REP="${PROJECT_DIR}/${PROJECT_NAME}.rep"
        PROJECT_GPR="${PROJECT_DIR}/${PROJECT_NAME}.gpr"
        if [[ ! -d "${PROJECT_REP}" || ! -f "${PROJECT_GPR}" ]]; then
            echo "error: existing Ghidra project was not found" >&2
            echo "  expected directory: ${PROJECT_REP}" >&2
            echo "  expected file:      ${PROJECT_GPR}" >&2
            exit 1
        fi
        mkdir -p -- "${OUT_DIR}"
        ;;
esac

command=(
    "${PYGHIDRA_RUN}"
    --headless
    "${PROJECT_DIR}"
    "${PROJECT_NAME}"
)

case "${OPERATION}" in
    import)
        command+=( -import "${ELF_PATH}" -overwrite )
        if [[ -n "${PROCESSOR}" ]]; then
            command+=( -processor "${PROCESSOR}" )
        fi
        ;;
    reuse)
        command+=( -process "${PROGRAM_NAME}" -noanalysis )
        ;;
esac

command+=( -scriptPath "${SCRIPT_DIR}" )

# -postScript expects the script name rather than its absolute path.  OUT_DIR
# and MODE are passed to extract_cfg.py through getScriptArgs().
command+=( -postScript "${SCRIPT_NAME}" "${OUT_DIR}" "${MODE}" )

echo "Ghidra CFG extraction"
echo "  operation: ${OPERATION}"
if [[ "${OPERATION}" == "import" ]]; then
    echo "  ELF:       ${ELF_PATH}"
else
    echo "  program:   ${PROGRAM_NAME}"
fi
echo "  output:    ${OUT_DIR}"
echo "  project:   ${PROJECT_DIR}/${PROJECT_NAME}"
echo "  script:    ${SCRIPT_PATH}"
echo "  mode:      ${MODE}"
if [[ "${OPERATION}" == "import" && -n "${PROCESSOR}" ]]; then
    echo "  processor: ${PROCESSOR}"
fi

exec "${command[@]}"
