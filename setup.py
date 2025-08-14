from setuptools import setup, find_packages

_dct = {}
with open('pyscfad/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

setup(
    name='pyscfad',
    version=__version__,
    description='PySCF with autodiff',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Xing Zhang',
    author_email='zhangxing.nju@gmail.com',
    include_package_data=True,
    packages=find_packages(exclude=["examples","*test*"]),
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.17',
        'scipy',
        'h5py',
        'jax==0.7.0',
        'pyscf>=2.3',
        f'pyscfadlib>=0.1.11,<={__version__}',
        'pyscf-properties',
    ],
    extras_require={
        'cuda12': [
            'jax[cuda12]==0.7.0',
            f'pyscfad-cuda12-plugin[with_cuda]>=0.1.11,<={__version__}',
        ],
    },
    url='https://github.com/fishjojo/pyscfad',
    license='Apache-2.0',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    zip_safe=False,
)
