import nox
from pathlib import Path

nox.options.sessions = ["typecheck", "test"]
nox.options.default_venv_backend = "uv|virtualenv"

python_versions = ["3.12"]
package_dir = Path("src/gso")
tests_dir = Path("src/tests")

@nox.session(python=python_versions)
def lint(session):
    """Run code style checks"""
    session.install("flake8", "black", "isort")
    session.run("black", str(package_dir))
    session.run("isort", str(package_dir))
    session.run("flake8", "--max-line-length=88", "--ignore=E203,E266,W503", str(package_dir))

@nox.session(python=python_versions)
def typecheck(session):
    """Run static type checking"""
    session.install("mypy", "numpy")
    session.run("mypy", "--strict", "--exclude", "tests/", str(package_dir))

@nox.session(python=python_versions)
def test(session):
    """Run unit tests"""
    session.install("-e", ".[test]")
    session.run("pytest", "-v", str(tests_dir))

@nox.session
def precommits(session):
    """Run pre-commits"""
    session.install("pre-commit")
    session.run("pre-commit", "install")
    args = session.posargs or ["--all-files"]
    session.run("pre-commit", "run", *args)

@nox.session(python=python_versions)
def docs(session):
    """Build documentation"""
    session.install("sphinx", "sphinx-rtd-theme")
    session.chdir("docs")
    session.run("sphinx-build", "-b", "html", ".", "_build/html")
