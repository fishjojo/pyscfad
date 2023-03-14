#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install wheel
pip install numpy scipy h5py
pip install typing_extensions
pip install jaxlib jax jaxopt
pip install pytest pytest-cov

#pyscf
git clone https://github.com/fishjojo/pyscf.git
cd pyscf; git checkout ad; cd ..

if [ "$RUNNER_OS" == "Linux" ]; then
    os='linux'
elif [ "$RUNNER_OS" == "macOS" ]; then
    os='macos'
else
    echo "$RUNNER_OS not supported"
    exit 1
fi

./.github/workflows/build_pyscf_"$os".sh
