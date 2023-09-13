#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install wheel
pip install numpy
pip install 'scipy<1.11'
pip install h5py
pip install typing_extensions
pip install jaxlib jax jaxopt
pip install pytest pytest-cov

#pyscf
git clone https://github.com/fishjojo/pyscf.git
cd pyscf; git checkout v2.1.1-ad; cd ..

if [ "$RUNNER_OS" == "Linux" ]; then
    os='linux'
elif [ "$RUNNER_OS" == "macOS" ]; then
    os='macos'
else
    echo "$RUNNER_OS not supported"
    exit 1
fi

./.github/workflows/build_pyscf_"$os".sh
