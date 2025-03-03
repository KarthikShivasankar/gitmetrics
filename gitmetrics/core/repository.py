"""
Repository module for GitMetrics.

This module provides the GitRepository class, which is responsible for
interacting with Git repositories and extracting basic information.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

from git import Repo, Commit
import requests

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


class GitRepository:
    """
    Class for interacting with Git repositories.

    This class provides methods for extracting basic information from
    Git repositories, such as commits, authors, and file changes.
    """

    def __init__(self, repo_path: str, is_remote: bool = False):
        """
        Initialize a GitRepository instance.

        Args:
            repo_path: Path to the Git repository or URL for remote repositories
            is_remote: Whether the repository is remote
        """
        self.repo_path = repo_path
        self.is_remote = is_remote

        if is_remote:
            # Clone the repository to a temporary directory
            import tempfile

            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Cloning remote repository {repo_path} to {self.temp_dir}")
            self.repo = Repo.clone_from(repo_path, self.temp_dir)
            self.local_path = self.temp_dir
        else:
            logger.info(f"Opening local repository at {repo_path}")
            self.repo = Repo(repo_path)
            self.local_path = repo_path

    def __del__(self):
        """Clean up temporary directory if this is a remote repository."""
        if hasattr(self, "is_remote") and self.is_remote and hasattr(self, "temp_dir"):
            import shutil

            logger.info(f"Cleaning up temporary directory {self.temp_dir}")
            shutil.rmtree(self.temp_dir)

    def get_commits(
        self, branch: str = "master", max_count: Optional[int] = None
    ) -> List[Commit]:
        """
        Get commits from the repository.

        Args:
            branch: Branch to get commits from
            max_count: Maximum number of commits to get

        Returns:
            List of commits
        """
        try:
            # Try to use the specified branch
            commits = list(self.repo.iter_commits(branch, max_count=max_count))
        except Exception as e:
            logger.warning(f"Error getting commits from branch {branch}: {e}")
            # Try to use the default branch
            try:
                default_branch = self.repo.active_branch.name
                logger.info(f"Using default branch: {default_branch}")
                commits = list(
                    self.repo.iter_commits(default_branch, max_count=max_count)
                )
            except Exception as e2:
                logger.error(f"Error getting commits from default branch: {e2}")
                # Last resort: try to get commits from HEAD
                logger.info("Trying to get commits from HEAD")
                commits = list(self.repo.iter_commits("HEAD", max_count=max_count))

        return commits

    def get_commit_count(self, branch: str = "master") -> int:
        """
        Get the number of commits in the repository.

        Args:
            branch: Branch to count commits from

        Returns:
            Number of commits
        """
        commits = self.get_commits(branch)
        return len(commits)

    def get_authors(self) -> List[Dict[str, Any]]:
        """
        Get the authors who have contributed to the repository.

        Returns:
            List of authors with their name, email, and commit count
        """
        commits = self.get_commits()

        authors = {}
        for commit in commits:
            name = commit.author.name
            email = commit.author.email

            if name not in authors:
                authors[name] = {"name": name, "email": email, "commit_count": 0}

            authors[name]["commit_count"] += 1

        return list(authors.values())

    def get_file_changes(self, branch: str = "master") -> Dict[str, int]:
        """
        Get the number of times each file has been changed.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary mapping file paths to the number of times they've been changed
        """
        commits = self.get_commits(branch)

        file_changes = {}
        for commit in commits:
            if len(commit.parents) == 0:
                # Skip the initial commit
                continue

            parent = commit.parents[0]
            diffs = parent.diff(commit)

            for diff in diffs:
                if diff.a_path:
                    path = diff.a_path
                    if path not in file_changes:
                        file_changes[path] = 0
                    file_changes[path] += 1

                if diff.b_path and diff.b_path != diff.a_path:
                    path = diff.b_path
                    if path not in file_changes:
                        file_changes[path] = 0
                    file_changes[path] += 1

        return file_changes

    def get_commit_activity(self, branch: str = "master") -> Dict[str, int]:
        """
        Get commit activity over time.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary mapping dates to the number of commits on that date
        """
        commits = self.get_commits(branch)

        activity = {}
        for commit in commits:
            date = commit.committed_datetime.date().isoformat()

            if date not in activity:
                activity[date] = 0

            activity[date] += 1

        return activity

    def get_file_types(self, branch: str = "master") -> Dict[str, int]:
        """
        Get the distribution of file types in the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary mapping file extensions to the number of files with that extension
        """
        # Get the tree of the latest commit on the branch
        try:
            tree = self.repo.heads[branch].commit.tree
        except Exception as e:
            logger.warning(f"Error getting tree for branch {branch}: {e}")
            # Try to use the default branch
            try:
                default_branch = self.repo.active_branch.name
                logger.info(f"Using default branch: {default_branch}")
                tree = self.repo.heads[default_branch].commit.tree
            except Exception as e2:
                logger.error(f"Error getting tree for default branch: {e2}")
                # Last resort: try to get tree from HEAD
                logger.info("Trying to get tree from HEAD")
                tree = self.repo.head.commit.tree

        file_types = {}

        # Traverse the tree and count file extensions
        for blob in tree.traverse():
            if blob.type == "blob":
                path = blob.path
                _, ext = os.path.splitext(path)

                if ext:
                    # Remove the dot from the extension
                    ext = ext[1:]
                else:
                    ext = "no_extension"

                if ext not in file_types:
                    file_types[ext] = 0

                file_types[ext] += 1

        return file_types

    def get_lines_of_code(self, branch: str = "master") -> Dict[str, int]:
        """
        Get the lines of code in the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with total lines, code lines, comment lines, and blank lines
        """
        # Get the tree of the latest commit on the branch
        try:
            tree = self.repo.heads[branch].commit.tree
        except Exception as e:
            logger.warning(f"Error getting tree for branch {branch}: {e}")
            # Try to use the default branch
            try:
                default_branch = self.repo.active_branch.name
                logger.info(f"Using default branch: {default_branch}")
                tree = self.repo.heads[default_branch].commit.tree
            except Exception as e2:
                logger.error(f"Error getting tree for default branch: {e2}")
                # Last resort: try to get tree from HEAD
                logger.info("Trying to get tree from HEAD")
                tree = self.repo.head.commit.tree

        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0

        # Define patterns for comments in different languages
        comment_patterns = {
            "py": [r"^\s*#", r'^\s*""".*?"""', r"^\s*'''.*?'''"],
            "js": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "java": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "c": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "cpp": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "rb": [r"^\s*#", r"^\s*=begin.*?=end"],
            "php": [r"^\s*//", r"^\s*#", r"^\s*/\*.*?\*/"],
            "go": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "rs": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "ts": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "swift": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "kt": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "scala": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "html": [r"^\s*<!--.*?-->"],
            "xml": [r"^\s*<!--.*?-->"],
            "css": [r"^\s*/\*.*?\*/"],
            "scss": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "less": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "sh": [r"^\s*#"],
            "bash": [r"^\s*#"],
            "zsh": [r"^\s*#"],
            "ps1": [r"^\s*#"],
            "bat": [r"^\s*REM", r"^\s*::"],
            "cmd": [r"^\s*REM", r"^\s*::"],
            "sql": [r"^\s*--", r"^\s*/\*.*?\*/"],
            "r": [r"^\s*#"],
            "matlab": [r"^\s*%"],
            "pl": [r"^\s*#"],
            "pm": [r"^\s*#"],
            "hs": [r"^\s*--", r"^\s*{-.*?-}"],
            "lhs": [r"^\s*--", r"^\s*{-.*?-}"],
            "fs": [r"^\s*//", r"^\s*\(\*.*?\*\)"],
            "fsx": [r"^\s*//", r"^\s*\(\*.*?\*\)"],
            "clj": [r"^\s*;"],
            "cljs": [r"^\s*;"],
            "erl": [r"^\s*%"],
            "ex": [r"^\s*#"],
            "exs": [r"^\s*#"],
            "lua": [r"^\s*--", r"^\s*--\[\[.*?\]\]"],
            "jl": [r"^\s*#", r"^\s*#=.*?=#"],
            "dart": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "groovy": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "cs": [r"^\s*//", r"^\s*/\*.*?\*/"],
            "vb": [r"^\s*\'", r"^\s*REM"],
        }

        # Traverse the tree and count lines
        for blob in tree.traverse():
            if blob.type == "blob":
                path = blob.path
                _, ext = os.path.splitext(path)

                if ext:
                    # Remove the dot from the extension
                    ext = ext[1:].lower()

                # Skip binary files
                if self._is_binary(blob.data_stream.read(1024)):
                    continue

                # Reset the data stream
                blob.data_stream.close()
                blob.data_stream = blob.data_stream.reopen()

                # Read the file content
                try:
                    content = blob.data_stream.read().decode("utf-8")
                except UnicodeDecodeError:
                    # Skip files that can't be decoded as UTF-8
                    continue

                lines = content.splitlines()
                total_lines += len(lines)

                for line in lines:
                    if not line.strip():
                        blank_lines += 1
                    elif ext in comment_patterns:
                        is_comment = False
                        for pattern in comment_patterns[ext]:
                            if re.match(pattern, line):
                                is_comment = True
                                break

                        if is_comment:
                            comment_lines += 1
                        else:
                            code_lines += 1
                    else:
                        code_lines += 1

        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines,
        }

    def _is_binary(self, data: bytes) -> bool:
        """
        Check if data is binary.

        Args:
            data: Data to check

        Returns:
            True if data is binary, False otherwise
        """
        # Check for null bytes
        if b"\x00" in data:
            return True

        # Check for non-printable characters
        non_printable = sum(1 for byte in data if byte < 32 and byte not in (9, 10, 13))
        return non_printable / len(data) > 0.3 if data else False

    def get_general_stats(self) -> Dict[str, Any]:
        """
        Get general statistics about the repository.

        Returns:
            Dictionary with general statistics
        """
        stats = {}

        # Get repository name
        if self.is_remote:
            # Extract repository name from URL
            repo_name = os.path.basename(self.repo_path)
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            stats["name"] = repo_name
        else:
            # Extract repository name from local path
            stats["name"] = os.path.basename(os.path.abspath(self.repo_path))

        # Get repository URL
        if self.is_remote:
            stats["url"] = self.repo_path
        else:
            try:
                remote_url = self.repo.remotes.origin.url
                stats["url"] = remote_url
            except Exception:
                stats["url"] = None

        # Get commit count
        stats["commit_count"] = self.get_commit_count()

        # Get author count
        authors = self.get_authors()
        stats["author_count"] = len(authors)

        # Get file count
        file_types = self.get_file_types()
        stats["file_count"] = sum(file_types.values())

        # Get file type distribution
        stats["file_types"] = file_types

        # Get lines of code
        stats["lines_of_code"] = self.get_lines_of_code()

        # Get first and last commit dates
        commits = self.get_commits()
        if commits:
            stats["first_commit_date"] = commits[-1].committed_datetime.isoformat()
            stats["last_commit_date"] = commits[0].committed_datetime.isoformat()
        else:
            stats["first_commit_date"] = None
            stats["last_commit_date"] = None

        # Get repository age in days
        if stats["first_commit_date"] and stats["last_commit_date"]:
            first_date = datetime.fromisoformat(stats["first_commit_date"])
            last_date = datetime.fromisoformat(stats["last_commit_date"])
            stats["age_days"] = (last_date - first_date).days
        else:
            stats["age_days"] = None

        return stats

    @staticmethod
    def from_github(repo_url: str) -> "GitRepository":
        """
        Create a GitRepository instance from a GitHub repository URL.

        Args:
            repo_url: GitHub repository URL

        Returns:
            GitRepository instance
        """
        # Ensure the URL is a valid GitHub repository URL
        if not (
            repo_url.startswith("https://github.com/")
            or repo_url.startswith("git@github.com:")
        ):
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        # Convert SSH URL to HTTPS URL if necessary
        if repo_url.startswith("git@github.com:"):
            repo_url = repo_url.replace("git@github.com:", "https://github.com/")

        # Ensure the URL ends with .git
        if not repo_url.endswith(".git"):
            repo_url += ".git"

        return GitRepository(repo_url, is_remote=True)
