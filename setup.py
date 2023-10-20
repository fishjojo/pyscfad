import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext

_dct = {}
with open('pyscfad/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

def get_platform():
    from distutils.util import get_platform
    platform = get_platform()
    if sys.platform == 'darwin':
        arch = os.getenv('CMAKE_OSX_ARCHITECTURES')
        if arch:
            osname = platform.rsplit('-', 1)[0]
            if ';' in arch:
                platform = f'{osname}-universal2'
            else:
                platform = f'{osname}-{arch}'
        elif os.getenv('_PYTHON_HOST_PLATFORM'):
            # the cibuildwheel environment
            platform = os.getenv('_PYTHON_HOST_PLATFORM')
            if platform.endswith('arm64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64')
            elif platform.endswith('x86_64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'x86_64')
            else:
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64;x86_64')
    return platform

class CMakeBuildPy(build_py):
    def run(self):
        self.plat_name = get_platform()
        self.build_base = 'build'
        self.build_lib = os.path.join(self.build_base, 'lib')
        self.build_temp = os.path.join(self.build_base, f'temp.{self.plat_name}')

        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'pyscfadlib'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        cmd = ['cmake', '--build', self.build_temp, '-j']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

        super().run()

from wheel.bdist_wheel import bdist_wheel
initialize_options = bdist_wheel.initialize_options
def initialize_with_default_plat_name(self):
    initialize_options(self)
    self.plat_name = get_platform()
bdist_wheel.initialize_options = initialize_with_default_plat_name

setup(
    name='pyscfad',
    version=__version__,
    description='PySCF with autodiff',
    author='Xing Zhang',
    author_email='xzhang8@caltech.edu',
    include_package_data=True,
    packages=find_packages(exclude=["examples","*test*"]),
    python_requires='>=3.8',
    cmdclass={'build_py': CMakeBuildPy,},
    install_requires=[
        'numpy>=1.17',
        'scipy',
        'jax>=0.3.25',
        'jaxlib>=0.3.25',
        #'pyscf @ git+https://github.com/fishjojo/pyscf.git@ad#egg=pyscf',
        #'pyscf-properties @ git+https://github.com/fishjojo/properties.git@ad',
    ],
    url='https://github.com/fishjojo/pyscfad',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
