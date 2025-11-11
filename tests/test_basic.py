import os


def test_readme_exists():
    assert os.path.exists("README.md")


def test_requirements_exists():
    assert os.path.exists("requirements.txt")
