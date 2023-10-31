# Contributing to `poli`

![Linting: black](https://img.shields.io/badge/Linting-black-black)
![Testing: pytest](https://img.shields.io/badge/Testing-pytest-blue)
![Testing: tox](https://img.shields.io/badge/Testing-tox-blue)
![Main branch: black](https://img.shields.io/badge/Pull_request_to-dev-blue)

This note details how to contribute to `poli`.

## Forking and making pull requests

The main development branch is called `dev`. To contribute, we recommend creating a fork of this repository and making changes on your version. Once you are ready to contribute, we expect you to lint and test.

## Linting your changes

We expect you to lint the code you write or modify using `black`.

```bash
pip install black
black ./path/to/files
```

## Testing your changes

Since we are testing multiple conda environments, we settled for using a combination of `tox` and `pytest`.

```bash
pip install tox

# To test linting (from the root of the project)
tox -e lint

# To test in the base environment for poli
tox -e poli-base-py39
```

If you want to run tests in all environments, remove `-e poli-base-py39` and just run `tox`.

## Create a pull request to dev

Once all tests pass and you are ready to share your changes, create a pull request to the `dev` branch.