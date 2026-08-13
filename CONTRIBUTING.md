# Contributing to ResearchPath

Thanks for your interest in ResearchPath.

## Development setup

~~~bash
git clone https://github.com/your-user/researchpath.git
cd researchpath
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
~~~

Run the tests and linter before opening a pull request:

~~~bash
pytest -q
ruff check .
~~~

## Contribution guidelines

- Keep retrieval and path-generation decisions explainable.
- Add or update tests for behavior changes.
- Do not commit API keys or downloaded local datasets.
- Document data sources and licenses.
- Keep pull requests focused on one improvement.
- Include benchmark results when changing ranking behavior.

For larger changes, open an issue first and describe the problem, proposed
design, and how the change will be evaluated.
