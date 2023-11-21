"""
Taken and adapted from:
https://gist.github.com/pdashford/2e4bcd4fc2343e2fd03efe4da17f577d?permalink_comment_id=4274705#gistcomment-4274705
"""
import base64
import getopt
import os
import shutil
import sys
from typing import Optional
from pathlib import Path

from github import Github, GithubException
from github.ContentFile import ContentFile
from github.Repository import Repository


def get_sha_for_tag(repository: Repository, tag: str) -> str:
    """
    Returns a commit PyGithub object for the specified repository and tag.
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
    verbose: bool = False,
    strict: bool = True,
) -> None:
    """
    repository_name: str (expected to be "user/repo")
    file_path_in_repository: str (path to file in repo)
    download_path: str (path to download to)
    tag: str (tag or branch to download)
    sha: str (sha of commit to download, overwrites tag if specified)
    exists_ok: bool (whether to overwrite existing files)

    This function will use an environment variable called GITHUB_TOKEN_FOR_POLI
    if it exists. If it does not exist, it will try to download without it.
    Note that, for anonymous requests, the rate limit is 60 requests per hour.
    """
    github = Github(login_or_token=os.environ.get("GITHUB_TOKEN_FOR_POLI"))
    repository = github.get_repo(repository_name)

    if commit_sha is None:
        commit_sha = get_sha_for_tag(repository, tag)

    _download_file_from_github_repo(
        repository=repository,
        commit_sha=commit_sha,
        file_path_in_repository=file_path_in_repository,
        download_path_for_file=download_path_for_file,
        exist_ok=exist_ok,
        verbose=verbose,
        strict=strict,
    )


def _download_file_from_github_repo(
    repository: Repository,
    commit_sha: str,
    file_path_in_repository: str,
    download_path_for_file: str,
    exist_ok: bool = False,
    verbose: bool = False,
    strict: bool = True,
) -> None:
    """
    Download all contents at server_path with commit tag sha in
    the repository.
    """
    if os.path.exists(download_path_for_file):
        if not exist_ok:
            raise FileExistsError("Path already exists: %s")

    if not isinstance(download_path_for_file, Path):
        download_path_for_file = Path(download_path_for_file)

    download_path_for_file.parent.mkdir(parents=True, exist_ok=exist_ok)

    file_content = repository.get_contents(file_path_in_repository, ref=commit_sha)
    assert isinstance(file_content, ContentFile)

    if verbose:
        print(f"Downloading {file_content.path} from {repository.name}")

    try:
        if not isinstance(file_content, ContentFile):
            raise ValueError("Expected ContentFile")

        with open(download_path_for_file, "wb") as file_out:
            if file_content.content:
                file_data = base64.b64decode(file_content.content)
                file_out.write(file_data)

    except (GithubException, IOError, ValueError) as exc:
        if strict:
            raise exc
        else:
            print("Error processing %s: %s", file_content.path, exc)
