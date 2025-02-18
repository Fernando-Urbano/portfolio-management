# Portfolio Manager Package

## Create Virtual Environment
The virtual environment is created using conda. It is used to manage the dependencies of the package.

The following steps are used to create the virtual environment.

Inside your terminal, write:
```bash
conda create -n finm python=3.12.6
```

Activate virtual environment:
```bash
conda activate finm
```

Install packages:
```bash
pip install -r requirements-dev.txt
```

Install pre-commit hooks:
```bash
pre-commit install
```

## How to Update version in Pipy?

Update the `version/new_version.txt` file with the new version number.

Install the following packages:

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install --upgrade twine
```

Remove existing `build/` and `dist/` folders to ensure a clean build:
```bash
rm -rf build/ dist/ *.egg-info
```

From the project root directory, run:

```bash
python setup.py sdist bdist_wheel
```

Upload the package to pip wit the new version:

```bash
twine upload dist/* -u __token__ -p $(cat build-package/token.txt)
```

Check that the version of the package is updated in the [PyPi](https://pypi.org/project/portfolio-management/).

# Collaboration Guidelines

## Github Issues
- We will manage tasks and features through GitHub Issues.
- Browse the open issues and assign yourself to any that are not yet taken.
- Some issues are labeled good first issue. These are a great way to familiarize yourself with the current codebase (e.g., by writing unit tests).

## Branching Strategy
- main: Contains the most recent version published to PyPI. No direct commits, merges, or pushes allowed.
- release/: The branch intended for our next release. No direct commits or pushes allowed here either. When we decide to publish a new version to PyPI, we will create a pull request from release to main.

## Working on Issues
1. Create a branch named feat/{issue_number}, for example: feat/5.
2. Include the issue number in commit messages (e.g., “Add new utility function #5”).
3. Frequently update your branch by pulling from release so you stay up to date with colleagues’ changes.
4. Open pull requests from your feature branch into release.
5. Once assigned, keep the issue updated with your progress and set an approximate deadline in a comment. This helps everyone stay on the same page.

## Upcoming Release
- Our next release branch is release/0.2.0. Let’s aim to merge completed features and fixes into this branch.

## Support
- If you have any ideas on how to improve the project structure or want to propose new issues, let me know!

# Build Sphix Documentation
```
cd doc
```

On Mac:
```
make html
```

On Windows:
```
make.bat html
```

View documentation:
```
open build/html/index.html
```
