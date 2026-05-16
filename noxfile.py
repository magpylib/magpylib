#!/usr/bin/env -S uv run --script

# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

"""Nox runner."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PROJECT = nox.project.load_toml()

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("prek")
    session.run(
        "prek", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run Pylint.
    """
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install("--group", "dev", "-e", ".", "pylint>=3.2")
    session.run("pylint", "magpylib", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps)
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. First positional argument is the target directory.
    """

    doc_deps = nox.project.dependency_groups(PROJECT, "docs")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.", *doc_deps, "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    # --- regenerate API stubs ---
    autogen_dir = DIR / "docs" / "_autogen"
    shutil.rmtree(autogen_dir, ignore_errors=True)
    autogen_dir.mkdir(parents=True, exist_ok=True)

    # Build sphinx-apidoc args (use your template dir if it exists)
    apidoc_args = [
        "-f",
        "-T",
        "-e",
        "-E",
        "-M",
        "-o",
        str(autogen_dir),
        str(DIR / "src" / "magpylib"),
    ]

    tmpl_dir = DIR / "docs" / "_templates" / "apidoc"
    if tmpl_dir.exists():
        apidoc_args = ["-t", str(tmpl_dir), *apidoc_args]

    session.run("sphinx-apidoc", *apidoc_args)
    # --- end apidoc ---

    if serve:
        session.run(
            "sphinx-autobuild",
            "--open-browser",
            "--watch",
            str(DIR / "src" / "magpylib"),
            *shared_args,
        )
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session(default=False)
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install("sphinx")
    session.run(
        "sphinx-apidoc",
        "-o",
        "docs/api/",
        "--module-first",
        "--no-toc",
        "--force",
        "src/magpylib",
    )


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """
    Build an SDist and wheel.
    """

    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    session.install("build")
    session.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
