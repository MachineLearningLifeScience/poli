"""Utilities for downloading files from GitHub repositories.

This module requires the PyGithub package, install it with:

        pip install PyGithub

Most of this code was taken and adapted from:
https://gist.github.com/pdashford/2e4bcd4fc2343e2fd03efe4da17f577d?permalink_comment_id=4274705#gistcomment-4274705
"""

import base64
import os
from pathlib import Path

from github import BadCredentialsException, Github, GithubException
from github.ContentFile import ContentFile
from github.Repository import Repository


def get_sha_for_tag(repository: Repository, tag: str) -> str:
    """
    Returns a commit PyGithub object for the specified repository and tag.

    Parameters
    ----------
    repository : Repository
        The repository.
    tag : str
        The tag.

    Returns
    -------
    commit_sha: str
        The commit SHA for the specified repository and tag.

    Raises
    ------
    ValueError
        If no tag or branch exists with the specified name.

    Examples
    --------
    >>> from github import Github
    >>> from github.Repository import Repository
    >>> github = Github()
    >>> repository = github.get_repo("rdkit/rdkit")
    >>> get_sha_for_tag(repository, "Release_2023_09")
    '068441957858f786c227825d90eb2c43f4f2b000'
    """
    branches = repository.get_branches()
    matched_branches = [match for match in branches if match.name == tag]
    if matched_branches:
        return matched_branches[0].commit.sha

    tags = repository.get_tags()
    matched_tags = [match for match in tags if match.name == tag]
    if not matched_tags:
        raise ValueError("No Tag or Branch exists with that name")

    return matched_tags[0].commit.sha


def download_file_from_github_repository(
    repository_name: str,
    file_path_in_repository: str,
    download_path_for_file: str,
    tag: str = "master",
    commit_sha: str = None,
    exist_ok: bool = False,
    parent_folders_exist_ok: bool = True,
    verbose: bool = False,
    strict: bool = True,
) -> None:
    """
    Download a file from a Github repository.

    Parameters
    ----------
    repository_name: str, required
        The name of the repository (i.e. "user/repo")
    file_path_in_repository: str, required
        path to file in repo
    download_path: str, required
        path to download to
    tag: str, optional
        tag or branch to download, defaults to master
    sha: str, optional
        sha of commit to download, overwrites tag if specified
    exists_ok: bool, optional
        whether to overwrite existing files
    parent_folders_exist_ok: bool, optional
        whether to create parent folders if they do not exist
    verbose: bool, optional
        whether to print progress
    strict: bool, optional
        whether to raise exceptions on errors

    Warnings
    --------

    This function will use an environment variable called
    GITHUB_TOKEN_FOR_POLI if it exists. If it does not exist,
    it will try to download without it. Note that the rate limit
    is 60 requests per hour for anonymous requests.

    To create a GitHub token like this, follow the instructions here:
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token
    """
    github = Github(login_or_token=os.environ.get("GITHUB_TOKEN_FOR_POLI"))

    try:
        repository = github.get_repo(repository_name)
    except BadCredentialsException as e:
        raise ValueError(
            "Your token has likely expired. Please set a new token following the instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
        ) from e

    if commit_sha is None:
        commit_sha = get_sha_for_tag(repository, tag)

    _download_file_from_github_repo(
        repository=repository,
        commit_sha=commit_sha,
        file_path_in_repository=file_path_in_repository,
        download_path_for_file=download_path_for_file,
        exist_ok=exist_ok,
        parent_folders_exist_ok=parent_folders_exist_ok,
        verbose=verbose,
        strict=strict,
    )


def _download_file_from_github_repo(
    repository: Repository,
    commit_sha: str,
    file_path_in_repository: str,
    download_path_for_file: str,
    exist_ok: bool = False,
    parent_folders_exist_ok: bool = True,
    verbose: bool = False,
    strict: bool = True,
) -> None:
    """
    Download a file from a GitHub repository.

    Parameters
    ----------
    repository : Repository
        The GitHub repository object.
    commit_sha : str
        The commit SHA of the file in the repository.
    file_path_in_repository : str
        The path of the file in the repository.
    download_path_for_file : str
        The path where the file will be downloaded.
    exist_ok : bool, optional
        If False and the download path already exists, a
        FileExistsError will be raised.
        If True, the download will proceed even if the path
        already exists. (default is False)
    parent_folders : bool, optional
        If True, create the parent folders for the download path
        if they do not exist. (default is True)
    verbose : bool, optional
        If True, print the progress of the download. (default is False)
    strict : bool, optional
        If True, raise an exception if there is an error during the download.
        If False, print an error message and continue. (default is True)
    """
    if os.path.exists(download_path_for_file):
        if not exist_ok:
            raise FileExistsError("Path already exists: %s")

    if not isinstance(download_path_for_file, Path):
        download_path_for_file = Path(download_path_for_file)

    download_path_for_file.parent.mkdir(parents=True, exist_ok=parent_folders_exist_ok)

    file_content = repository.get_contents(file_path_in_repository, ref=commit_sha)

    if isinstance(file_content, ContentFile):
        _save_file_content_from_github(
            file_content, download_path_for_file, strict=strict, verbose=verbose
        )
    elif isinstance(file_content, list):
        for file_ in file_content:
            _download_file_from_github_repo(
                repository=repository,
                commit_sha=commit_sha,
                file_path_in_repository=file_.path,
                download_path_for_file=download_path_for_file / file_.name,
                exist_ok=exist_ok,
                parent_folders_exist_ok=parent_folders_exist_ok,
                verbose=verbose,
                strict=strict,
            )
    else:
        raise ValueError("Expected ContentFile or list of ContentFile")


def _save_file_content_from_github(
    file_content: ContentFile,
    download_path_for_file: Path,
    strict: bool = True,
    verbose: bool = False,
) -> None:
    if verbose:
        print(f"poli 🧪: Downloading {file_content.path}")
    try:
        if not isinstance(file_content, ContentFile):
            raise ValueError("Expected ContentFile")

        with open(download_path_for_file, "wb") as file_out:
            if file_content.content:
                file_data = base64.b64decode(file_content.content)
                file_out.write(file_data)
            elif file_content.download_url:
                import requests

                file_out.write(requests.get(file_content.download_url).content)

    except (GithubException, IOError, ValueError) as exc:
        if strict:
            raise exc
        else:
            print("Error processing %s: %s", file_content.path, exc)
