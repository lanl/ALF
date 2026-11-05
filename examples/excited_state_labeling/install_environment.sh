#!/bin/bash
# Install into new independent paths only. Resolve current dependencies once,
# then freeze the environment and official upstream source identity.
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo 'Usage: bash install_environment.sh gpu4pyscf|pyseqm ENV_PATH PYTHON SOURCE_PATH' >&2
    exit 2
fi
inc8_backend=$1
inc8_env=$2
inc8_python=$3
inc8_source=$4
if [ -e "$inc8_env" ]; then
    echo "Refusing to alter existing environment: $inc8_env" >&2
    exit 2
fi
"$inc8_python" -m venv "$inc8_env"
inc8_exec="$inc8_env/bin/python"
"$inc8_exec" -m pip install --upgrade pip
case "$inc8_backend" in
    gpu4pyscf)
        "$inc8_exec" -m pip install --upgrade gpu4pyscf-cuda13x 'cupy-cuda13x[ctk]' pyscf
        ;;
    pyseqm)
        git clone --depth 1 https://github.com/lanl/PYSEQM.git "$inc8_env/upstream-PYSEQM"
        git -C "$inc8_env/upstream-PYSEQM" rev-parse HEAD > "$inc8_env/pyseqm-revision.txt"
        "$inc8_exec" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu132
        "$inc8_exec" -m pip install "$inc8_env/upstream-PYSEQM"
        ;;
    *) echo 'Unknown backend' >&2; exit 2 ;;
esac
"$inc8_exec" -m pip install "$inc8_source[tests]"
"$inc8_exec" -m pip check
"$inc8_exec" -m pip freeze > "$inc8_env/requirements-frozen.txt"
git -C "$inc8_source" rev-parse HEAD > "$inc8_env/alf-source-revision.txt"
