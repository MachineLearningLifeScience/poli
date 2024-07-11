# Contributing to `poli`

![Linting: black](https://img.shields.io/badge/Linting-black-black)
![Testing: pytest](https://img.shields.io/badge/Testing-pytest-blue)
![Testing: tox](https://img.shields.io/badge/Testing-tox-blue)
![Main branch: dev](https://img.shields.io/badge/Pull_request_to-dev-blue)

This note details how to contribute to `poli`.

## Forking and making pull requests

The main development branch is called `dev`. To contribute, we recommend creating a fork of this repository and making changes on your version. Once you are ready to contribute, we expect you to document, lint and test.

## Installing dev dependencies and pre-commit hooks

We recommend you create a `poli-dev` environment in conda

```bash
conda create -n poli-dev python=3.10
conda activate poli-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The dev requirements include `pre-commit`. Install the hooks in our config by running

```bash
pre-commit install
```

Now every commit will run linting and isorting for you. You can also run it manually by saying

```bash
pre-commit run --all-files
```

## Documentation standards

We follow [numpy's documentation standards](https://numpydoc.readthedocs.io/en/latest/format.html).

## Creating a new environment? Mark your tests

If you're contributing a black box in a new environment, remember to

1. Mark all your tests with `@pyest.mark.poli__your_env`.
2. Add a description of your marker to the `[tool.pytest.ini_options]`.

## Testing your changes for `dev`

Since we are testing multiple conda environments, we settled for using a combination of `tox` and `pytest`.

```bash
pip install tox

# To test linting (from the root of the project)
tox -c tox.ini -e lint

# To test in the base environment for poli
tox -c tox.ini -e poli-base-py39
```

There are several different environments (depending on the black boxes we test). Check `tox.ini` for more details.

If you want to run tests in all environments, remove `-e poli-base-py39` and just run `tox`. This might take a while, and several conda envs will be created.

## Create a pull request to dev

Once all tests pass and you are ready to share your changes, create a pull request to the `dev` branch.