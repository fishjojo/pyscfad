#!/usr/bin/env bash
python -m pip install --upgrade pip
python -m pip cache purge
pip install wheel
pip install numpy
pip install 'scipy<1.11'
pip install h5py
pip install 'jaxlib==0.4.14'
pip install 'jax==0.4.14'
pip install pytest
pip install pytest-cov

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
