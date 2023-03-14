import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext

_dct = {}
with open('pyscfad/version.py') as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == 'pyscfad_lib_placeholder'
        self.build_cmake(extension)

    def build_cmake(self, extension):
        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'pyscfad', 'lib'))
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

    # To remove the infix string like cpython-37m-x86_64-linux-gnu.so
    # Python ABI updates since 3.5
    # https://www.python.org/dev/peps/pep-3149/
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix

#from distutils.command.build import build
#build.sub_commands = ([c for c in build.sub_commands if c[0] == 'build_ext'] +
#                      [c for c in build.sub_commands if c[0] != 'build_ext'])

class BuildExtFirst(build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

setup(
    name='pyscfad',
    version=__version__,
    description='PySCF with autodiff',
    author='Xing Zhang',
    author_email='xzhang8@caltech.edu',
    include_package_data=True,
    packages=find_packages(exclude=["examples","*test*"]),
    python_requires='>=3.7',
    #ext_modules=[Extension('pyscfad_lib_placeholder', [])],
    #cmdclass={'build_py': BuildExtFirst,
    #          'build_ext': CMakeBuildExt},
    install_requires=[
        'numpy>=1.17',
        'scipy',
        'jax>=0.3.25',
        'jaxlib>=0.3.25',
        'typing_extensions',
        'jaxopt>=0.2',
        'pyscf @ git+https://github.com/fishjojo/pyscf.git@ad#egg=pyscf',
        'pyscf-properties @ git+https://github.com/fishjojo/properties.git@ad',
    ],
    url='https://github.com/fishjojo/pyscfad',
    license='MIT',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
