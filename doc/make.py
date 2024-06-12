#!/usr/bin/env python3
"""
Python script for building documentation.

To build the docs you must have all optional dependencies for pyscfad
installed. See the installation instructions for a list of these.

Usage
-----
    $ python make.py clean
    $ python make.py html
    $ python make.py latex
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys

import docutils
import docutils.parsers.rst

DOC_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(DOC_PATH, "source")
BUILD_PATH = os.path.join(DOC_PATH, "build")


class DocBuilder:
    """
    Class to wrap the different commands of this script.

    All public methods of this class can be called as parameters of the
    script.
    """

    def __init__(
        self,
        num_jobs="auto",
        verbosity=0,
        warnings_are_errors=False,
    ) -> None:
        self.num_jobs = num_jobs
        self.verbosity = verbosity
        self.warnings_are_errors = warnings_are_errors

    @staticmethod
    def _run_os(*args) -> None:
        """
        Execute a command as a OS terminal.

        Parameters
        ----------
        *args : list of str
            Command and parameters to be executed

        Examples
        --------
        >>> DocBuilder()._run_os("python", "--version")
        """
        subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

    def _sphinx_build(self, kind: str):
        """
        Call sphinx to build documentation.

        Attribute `num_jobs` from the class is used.

        Parameters
        ----------
        kind : {'html', 'latex', 'linkcheck'}

        Examples
        --------
        >>> DocBuilder(num_jobs=4)._sphinx_build("html")
        """
        if kind not in ("html", "latex", "linkcheck"):
            raise ValueError(f"kind must be html, latex or linkcheck, not {kind}")

        cmd = ["sphinx-build", "-b", kind]
        if self.num_jobs:
            cmd += ["-j", self.num_jobs]
        if self.warnings_are_errors:
            cmd += ["-W", "--keep-going"]
        if self.verbosity:
            cmd.append(f"-{'v' * self.verbosity}")
        cmd += [
            "-d",
            os.path.join(BUILD_PATH, "doctrees"),
            SOURCE_PATH,
            os.path.join(BUILD_PATH, kind),
        ]
        return subprocess.call(cmd)

    def _open_browser(self, single_doc_html) -> None:
        """
        Open a browser tab showing single
        """
        url = os.path.join("file://", DOC_PATH, "build", "html", single_doc_html)
        webbrowser.open(url, new=2)

    def _get_page_title(self, page):
        """
        Open the rst file `page` and extract its title.
        """
        fname = os.path.join(SOURCE_PATH, f"{page}.rst")
        doc = docutils.utils.new_document(
            "<doc>",
            docutils.frontend.get_default_settings(docutils.parsers.rst.Parser),
        )
        with open(fname, encoding="utf-8") as f:
            data = f.read()

        parser = docutils.parsers.rst.Parser()
        # do not generate any warning when parsing the rst
        with open(os.devnull, "a", encoding="utf-8") as f:
            doc.reporter.stream = f
            parser.parse(data, doc)

        section = next(
            node for node in doc.children if isinstance(node, docutils.nodes.section)
        )
        title = next(
            node for node in section.children if isinstance(node, docutils.nodes.title)
        )

        return title.astext()

    def html(self):
        """
        Build HTML documentation.
        """
        ret_code = self._sphinx_build("html")
        zip_fname = os.path.join(BUILD_PATH, "html", "pyscfad.zip")
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        return ret_code

    def latex(self, force=False):
        """
        Build PDF documentation.
        """
        if sys.platform == "win32":
            sys.stderr.write("latex build has not been tested on windows\n")
        else:
            ret_code = self._sphinx_build("latex")
            os.chdir(os.path.join(BUILD_PATH, "latex"))
            if force:
                for i in range(3):
                    self._run_os("pdflatex", "-interaction=nonstopmode", "pyscfad.tex")
                raise SystemExit(
                    "You should check the file "
                    '"build/latex/pyscfad.pdf" for problems.'
                )
            self._run_os("make")
            return ret_code

    def latex_forced(self):
        """
        Build PDF documentation with retries to find missing references.
        """
        return self.latex(force=True)

    @staticmethod
    def clean() -> None:
        """
        Clean documentation generated files.
        """
        shutil.rmtree(BUILD_PATH, ignore_errors=True)
        shutil.rmtree(os.path.join(SOURCE_PATH, "reference", "api"), ignore_errors=True)

    def zip_html(self) -> None:
        """
        Compress HTML documentation into a zip file.
        """
        zip_fname = os.path.join(BUILD_PATH, "html", "pyscfad.zip")
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        dirname = os.path.join(BUILD_PATH, "html")
        fnames = os.listdir(dirname)
        os.chdir(dirname)
        self._run_os("zip", zip_fname, "-r", "-q", *fnames)

    def linkcheck(self):
        """
        Check for broken links in the documentation.
        """
        return self._sphinx_build("linkcheck")


def main():
    cmds = [method for method in dir(DocBuilder) if not method.startswith("_")]

    joined = ",".join(cmds)
    argparser = argparse.ArgumentParser(
        description="pyscfad documentation builder", epilog=f"Commands: {joined}"
    )

    joined = ", ".join(cmds)
    argparser.add_argument(
        "command", nargs="?", default="html", help=f"command to run: {joined}"
    )
    argparser.add_argument(
        "--num-jobs", default="auto", help="number of jobs used by sphinx-build"
    )
    argparser.add_argument(
        "--python-path", type=str, default=os.path.dirname(DOC_PATH), help="path"
    )
    argparser.add_argument(
        "-v",
        action="count",
        dest="verbosity",
        default=0,
        help=(
            "increase verbosity (can be repeated), "
            "passed to the sphinx build command"
        ),
    )
    argparser.add_argument(
        "--warnings-are-errors",
        "-W",
        action="store_true",
        help="fail if warnings are raised",
    )
    args = argparser.parse_args()

    if args.command not in cmds:
        joined = ", ".join(cmds)
        raise ValueError(f"Unknown command {args.command}. Available options: {joined}")

    # Below we update both os.environ and sys.path. The former is used by
    # external libraries (namely Sphinx) to compile this module and resolve
    # the import of `python_path` correctly. The latter is used to resolve
    # the import within the module, injecting it into the global namespace
    os.environ["PYTHONPATH"] = args.python_path
    sys.path.insert(0, args.python_path)
    globals()["pyscfad"] = importlib.import_module("pyscfad")

    # Set the matplotlib backend to the non-interactive Agg backend for all
    # child processes.
    #os.environ["MPLBACKEND"] = "module://matplotlib.backends.backend_agg"

    builder = DocBuilder(
        args.num_jobs,
        args.verbosity,
        args.warnings_are_errors,
    )
    return getattr(builder, args.command)()


if __name__ == "__main__":
    sys.exit(main())
