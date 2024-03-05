import os
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from distutils.util import get_platform

_dct = {}
with open('pyscfadlib/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

class CMakeBuild(build_py):
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
        cmd = ['cmake', '--build', self.build_temp, '-j2']
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
    name='pyscfadlib',
    version=__version__,
    description='Support library for PySCFAD',
    author='Xing Zhang',
    author_email='xzhang8@caltech.edu',
    include_package_data=True,
    packages=find_packages(),
    python_requires='>=3.8',
    cmdclass={'build_py': CMakeBuild},
    install_requires=[
        'numpy>=1.17',
    ],
    url='https://github.com/fishjojo/pyscfad',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
