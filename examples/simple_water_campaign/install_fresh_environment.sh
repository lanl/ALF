#!/bin/bash
# Independent current-release environment; no personal site-packages or edits.
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo 'Usage: bash install_fresh_environment.sh ENV_PATH TORCH_INDEX base|torchsim|alchemi [PYTHON]' >&2
    exit 2
fi
alf_env_path=$1
alf_torch_index=$2
alf_stack=$3
alf_python=${4:-python3.12}
case "$alf_stack" in
    base) alf_extras='tests,full' ;;
    torchsim) alf_extras='tests,full,torchsim' ;;
    alchemi) alf_extras='tests,full,alchemi' ;;
    *) echo "Unknown stack: $alf_stack" >&2; exit 2 ;;
esac
if [ -e "$alf_env_path" ]; then
    echo "Environment already exists; choose a fresh path: $alf_env_path" >&2
    exit 2
fi
alf_source=$(cd -- "$(dirname -- "$0")/../.." && pwd)
"$alf_python" -m venv "$alf_env_path"
alf_env_python="$alf_env_path/bin/python"
"$alf_env_python" -m pip install --upgrade pip
"$alf_env_python" -m pip install --upgrade torch --index-url "$alf_torch_index"
"$alf_env_python" -m pip install --upgrade -e "$alf_source[$alf_extras]" 'hippynn[full]' pytest-timeout
"$alf_env_python" -m pip check
"$alf_env_python" -m pip freeze > "$alf_env_path/requirements-frozen.txt"
git -C "$alf_source" rev-parse HEAD > "$alf_env_path/alf-source-revision.txt"
