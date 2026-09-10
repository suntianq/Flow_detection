#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./run_diff.sh import OLD_BINARY NEW_BINARY [OUT_DIR]
#   ./run_diff.sh reuse OLD_PROGRAM NEW_PROGRAM [OUT_DIR]
#
# Shorthand import form:
#   ./run_diff.sh OLD_BINARY NEW_BINARY [OUT_DIR]
#
# Examples:
#   ./run_diff.sh import /workspace/v1/app /workspace/v2/app /workspace/app_diff
#   ./run_diff.sh reuse old/app new/app /workspace/app_diff
#   PROJECT_DIR=/workspace/ghidra_projects PROJECT_NAME=app_versions \
#       ./run_diff.sh reuse old/app new/app /workspace/app_diff
#
# import:
#   Import/overwrite OLD and NEW into separate folders in one Ghidra project,
#   run Auto Analysis for both, then execute binary_diff.py with OLD as the
#   current program.  Separate folders make identical basenames safe.
#
# reuse:
#   Open two already-analyzed Programs in an existing project and run only the
#   diff.  Program arguments are project paths, for example old/app or new/app.
#
# Every setting can also be supplied through an environment variable.
GHIDRA_HOME="${GHIDRA_HOME:-/workspace/tools/ghidra_12.0.3_PUBLIC}"
PROJECT_DIR="${PROJECT_DIR:-/workspace}"
PROCESSOR="${PROCESSOR:-}"
OLD_PROJECT_FOLDER="${OLD_PROJECT_FOLDER:-old}"
NEW_PROJECT_FOLDER="${NEW_PROJECT_FOLDER:-new}"

usage() {
    printf '%s\n' \
        "usage:" \
        "  $0 import OLD_BINARY NEW_BINARY [OUT_DIR]" \
        "  $0 reuse OLD_PROGRAM NEW_PROGRAM [OUT_DIR]" \
        "  $0 OLD_BINARY NEW_BINARY [OUT_DIR]  # shorthand import form" \
        "" \
        "reuse program names are Ghidra project paths, for example:" \
        "  old/app  new/app" \
        "" \
        "environment overrides:" \
        "  GHIDRA_HOME, PROJECT_DIR, PROJECT_NAME, SCRIPT_DIR, OUT_DIR," \
        "  PROCESSOR (import only), OLD_BINARY, NEW_BINARY," \
        "  OLD_PROGRAM, NEW_PROGRAM, OLD_PROJECT_FOLDER, NEW_PROJECT_FOLDER"
}

normalize_project_path() {
    local value="$1"
    value="${value//\\//}"
    value="${value#/}"
    value="${value%/}"
    printf '%s' "${value}"
}

validate_project_path() {
    local label="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "error: ${label} must not be empty" >&2
        exit 2
    fi
    if [[ "/${value}/" == *"/../"* || "/${value}/" == *"/./"* ]]; then
        echo "error: ${label} must not contain '.' or '..' path components: ${value}" >&2
        exit 2
    fi
}

split_program_path() {
    local program_path="$1"
    PROGRAM_BASENAME="${program_path##*/}"
    if [[ "${program_path}" == */* ]]; then
        PROGRAM_FOLDER="${program_path%/*}"
    else
        PROGRAM_FOLDER=""
    fi
}

print_command() {
    printf '  command:'
    printf ' %q' "$@"
    printf '\n'
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
        OLD_BINARY="${1:-${OLD_BINARY:-}}"
        NEW_BINARY="${2:-${NEW_BINARY:-}}"
        if [[ -z "${OLD_BINARY}" || -z "${NEW_BINARY}" ]]; then
            echo "error: import mode requires OLD_BINARY and NEW_BINARY" >&2
            usage >&2
            exit 2
        fi

        OLD_BASENAME="$(basename -- "${OLD_BINARY}")"
        NEW_BASENAME="$(basename -- "${NEW_BINARY}")"
        PROJECT_NAME="${PROJECT_NAME:-${OLD_BASENAME}_diff}"
        DEFAULT_OUT_DIR="$(dirname -- "${OLD_BINARY}")/${OLD_BASENAME}_vs_${NEW_BASENAME}_diff_output"

        OLD_PROJECT_FOLDER="$(normalize_project_path "${OLD_PROJECT_FOLDER}")"
        NEW_PROJECT_FOLDER="$(normalize_project_path "${NEW_PROJECT_FOLDER}")"
        validate_project_path "OLD_PROJECT_FOLDER" "${OLD_PROJECT_FOLDER}"
        validate_project_path "NEW_PROJECT_FOLDER" "${NEW_PROJECT_FOLDER}"

        OLD_PROGRAM="${OLD_PROJECT_FOLDER}/${OLD_BASENAME}"
        NEW_PROGRAM="${NEW_PROJECT_FOLDER}/${NEW_BASENAME}"
        if [[ "${OLD_PROGRAM}" == "${NEW_PROGRAM}" ]]; then
            echo "error: OLD and NEW resolve to the same Ghidra project path: ${OLD_PROGRAM}" >&2
            echo "       use different OLD_PROJECT_FOLDER/NEW_PROJECT_FOLDER values" >&2
            exit 2
        fi
        ;;
    reuse)
        OLD_PROGRAM="$(normalize_project_path "${1:-${OLD_PROGRAM:-}}")"
        NEW_PROGRAM="$(normalize_project_path "${2:-${NEW_PROGRAM:-}}")"
        if [[ -z "${OLD_PROGRAM}" || -z "${NEW_PROGRAM}" ]]; then
            echo "error: reuse mode requires OLD_PROGRAM and NEW_PROGRAM" >&2
            usage >&2
            exit 2
        fi
        validate_project_path "OLD_PROGRAM" "${OLD_PROGRAM}"
        validate_project_path "NEW_PROGRAM" "${NEW_PROGRAM}"
        if [[ "${OLD_PROGRAM}" == "${NEW_PROGRAM}" ]]; then
            echo "error: OLD_PROGRAM and NEW_PROGRAM must be different" >&2
            exit 2
        fi

        split_program_path "${OLD_PROGRAM}"
        OLD_BASENAME="${PROGRAM_BASENAME}"
        OLD_PROGRAM_FOLDER="${PROGRAM_FOLDER}"
        split_program_path "${NEW_PROGRAM}"
        NEW_BASENAME="${PROGRAM_BASENAME}"
        NEW_PROGRAM_FOLDER="${PROGRAM_FOLDER}"

        PROJECT_NAME="${PROJECT_NAME:-${OLD_BASENAME}_diff}"
        DEFAULT_OUT_DIR="${OLD_BASENAME}_vs_${NEW_BASENAME}_diff_output"
        ;;
esac

OUT_DIR="${3:-${OUT_DIR:-${DEFAULT_OUT_DIR}}}"

# Keep binary_diff.py beside this launcher.  SCRIPT_DIR may be overridden when
# the Ghidra script is stored elsewhere.
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-${RUN_DIR}}"
SCRIPT_NAME="binary_diff.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"
PYGHIDRA_RUN="${GHIDRA_HOME}/support/pyghidraRun"

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
        if [[ ! -f "${OLD_BINARY}" ]]; then
            echo "error: OLD binary does not exist: ${OLD_BINARY}" >&2
            exit 1
        fi
        if [[ ! -f "${NEW_BINARY}" ]]; then
            echo "error: NEW binary does not exist: ${NEW_BINARY}" >&2
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

echo "Ghidra binary CFG diff"
echo "  operation:   ${OPERATION}"
if [[ "${OPERATION}" == "import" ]]; then
    echo "  OLD binary:  ${OLD_BINARY}"
    echo "  NEW binary:  ${NEW_BINARY}"
fi
echo "  OLD program: ${OLD_PROGRAM}"
echo "  NEW program: ${NEW_PROGRAM}"
echo "  output:      ${OUT_DIR}"
echo "  project:     ${PROJECT_DIR}/${PROJECT_NAME}"
echo "  script:      ${SCRIPT_PATH}"
if [[ "${OPERATION}" == "import" && -n "${PROCESSOR}" ]]; then
    echo "  processor:   ${PROCESSOR}"
fi

if [[ "${OPERATION}" == "import" ]]; then
    old_import_command=(
        "${PYGHIDRA_RUN}"
        --headless
        "${PROJECT_DIR}"
        "${PROJECT_NAME}/${OLD_PROJECT_FOLDER}"
        -import "${OLD_BINARY}"
        -overwrite
    )
    new_import_command=(
        "${PYGHIDRA_RUN}"
        --headless
        "${PROJECT_DIR}"
        "${PROJECT_NAME}/${NEW_PROJECT_FOLDER}"
        -import "${NEW_BINARY}"
        -overwrite
    )
    if [[ -n "${PROCESSOR}" ]]; then
        old_import_command+=( -processor "${PROCESSOR}" )
        new_import_command+=( -processor "${PROCESSOR}" )
    fi

    echo "[1/3] Importing and analyzing OLD"
    print_command "${old_import_command[@]}"
    "${old_import_command[@]}"

    echo "[2/3] Importing and analyzing NEW"
    print_command "${new_import_command[@]}"
    "${new_import_command[@]}"

    OLD_PROGRAM_FOLDER="${OLD_PROJECT_FOLDER}"
fi

# AnalyzeHeadless selects a project folder through PROJECT_NAME/folder and then
# -process receives the basename within that folder.  binary_diff.py receives
# NEW_PROGRAM as a root-relative project path and locates it recursively.
OLD_PROJECT_SPEC="${PROJECT_NAME}"
if [[ -n "${OLD_PROGRAM_FOLDER:-}" ]]; then
    OLD_PROJECT_SPEC+="/${OLD_PROGRAM_FOLDER}"
fi

diff_command=(
    "${PYGHIDRA_RUN}"
    --headless
    "${PROJECT_DIR}"
    "${OLD_PROJECT_SPEC}"
    -process "${OLD_BASENAME}"
    -noanalysis
    -scriptPath "${SCRIPT_DIR}"
    -postScript "${SCRIPT_NAME}" "${NEW_PROGRAM}" "${OUT_DIR}"
)

if [[ "${OPERATION}" == "import" ]]; then
    echo "[3/3] Comparing OLD and NEW CFGs"
else
    echo "[1/1] Comparing OLD and NEW CFGs"
fi
print_command "${diff_command[@]}"
exec "${diff_command[@]}"
