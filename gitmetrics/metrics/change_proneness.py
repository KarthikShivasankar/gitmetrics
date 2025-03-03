"""
Change proneness metrics module for GitMetrics.

This module provides functions for calculating change proneness metrics,
which measure how frequently files change over time and identify error-prone files.
"""

import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd
from git import Repo, Commit

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_change_frequency(repo: Repo, branch: str = "master") -> pd.DataFrame:
    """
    Calculate the change frequency for each file in the repository.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        DataFrame with change frequency metrics
    """
    logger.info("Calculating change frequency...")

    try:
        # Try to use the specified branch
        commits = list(repo.iter_commits(branch))
    except:
        # If the branch doesn't exist, try 'main' instead
        try:
            commits = list(repo.iter_commits("main"))
        except:
            # If 'main' doesn't exist either, use the default branch
            commits = list(repo.iter_commits())

    # Dictionary to store file change data
    file_changes = defaultdict(
        lambda: {
            "commit_count": 0,
            "first_commit": None,
            "last_commit": None,
            "authors": set(),
            "lines_added": 0,
            "lines_removed": 0,
        }
    )

    # Process each commit
    for commit in commits:
        commit_date = datetime.fromtimestamp(commit.committed_date)

        # Skip merge commits
        if len(commit.parents) > 1:
            continue

        # For the first commit, get all files
        if not commit.parents:
            for item in commit.tree.traverse():
                if item.type == "blob":
                    file_path = item.path
                    file_changes[file_path]["commit_count"] += 1
                    file_changes[file_path]["first_commit"] = commit_date
                    file_changes[file_path]["last_commit"] = commit_date
                    file_changes[file_path]["authors"].add(commit.author.name)
            continue

        # For other commits, get the diff with the parent
        parent = commit.parents[0]
        diffs = parent.diff(commit, create_patch=True)

        for diff in diffs:
            # Skip binary files
            if diff.diff:
                try:
                    diff_text = diff.diff.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                # Count added and removed lines
                lines_added = 0
                lines_removed = 0

                for line in diff_text.split("\n"):
                    if line.startswith("+") and not line.startswith("+++"):
                        lines_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        lines_removed += 1

                # Update file change data
                file_path = diff.b_path or diff.a_path

                if file_path:
                    file_changes[file_path]["commit_count"] += 1
                    file_changes[file_path]["authors"].add(commit.author.name)
                    file_changes[file_path]["lines_added"] += lines_added
                    file_changes[file_path]["lines_removed"] += lines_removed

                    if (
                        file_changes[file_path]["first_commit"] is None
                        or commit_date < file_changes[file_path]["first_commit"]
                    ):
                        file_changes[file_path]["first_commit"] = commit_date

                    if (
                        file_changes[file_path]["last_commit"] is None
                        or commit_date > file_changes[file_path]["last_commit"]
                    ):
                        file_changes[file_path]["last_commit"] = commit_date

    # Convert to DataFrame
    data = []

    for file_path, stats in file_changes.items():
        # Calculate age in days
        if stats["first_commit"] and stats["last_commit"]:
            age_days = (stats["last_commit"] - stats["first_commit"]).days
        else:
            age_days = 0

        # Calculate change frequency (changes per day)
        if age_days > 0:
            change_frequency = stats["commit_count"] / age_days
        else:
            change_frequency = 0

        # Calculate total lines changed
        total_lines_changed = stats["lines_added"] + stats["lines_removed"]

        data.append(
            {
                "file_path": file_path,
                "commit_count": stats["commit_count"],
                "author_count": len(stats["authors"]),
                "age_days": age_days,
                "lines_added": stats["lines_added"],
                "lines_removed": stats["lines_removed"],
                "total_lines_changed": total_lines_changed,
                "change_frequency": change_frequency,
            }
        )

    df = pd.DataFrame(data)

    # Sort by change frequency in descending order
    df = df.sort_values("change_frequency", ascending=False)

    logger.info(f"Change frequency calculated for {len(df)} files")

    return df


def calculate_change_proneness(repo: Repo, branch: str = "master") -> Dict[str, Any]:
    """
    Calculate change proneness metrics for files in a Git repository.

    Change proneness measures how frequently files change over time.
    Files that change frequently may be more prone to errors and may
    indicate design issues.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        Dictionary with change proneness metrics
    """
    logger.info(f"Calculating change proneness metrics for branch: {branch}")

    # Get all commits
    try:
        commits = list(repo.iter_commits(branch))
    except Exception as e:
        logger.warning(f"Error getting commits from branch {branch}: {e}")
        # Try to use the default branch
        try:
            default_branch = repo.active_branch.name
            logger.info(f"Using default branch: {default_branch}")
            commits = list(repo.iter_commits(default_branch))
        except Exception as e2:
            logger.error(f"Error getting commits from default branch: {e2}")
            # Last resort: try to get commits from HEAD
            logger.info("Trying to get commits from HEAD")
            commits = list(repo.iter_commits("HEAD"))

    # Count changes per file
    file_changes = defaultdict(int)
    file_last_modified = {}
    file_first_modified = {}
    file_authors = defaultdict(set)

    # Group commits by month
    monthly_changes = defaultdict(lambda: defaultdict(int))

    for commit in commits:
        if len(commit.parents) == 0:
            # Skip the initial commit
            continue

        parent = commit.parents[0]
        diffs = parent.diff(commit)

        commit_date = commit.committed_datetime
        month_key = f"{commit_date.year}-{commit_date.month:02d}"

        for diff in diffs:
            if diff.a_path:
                path = diff.a_path
                file_changes[path] += 1
                file_authors[path].add(commit.author.name)

                if path not in file_first_modified:
                    file_first_modified[path] = commit_date

                file_last_modified[path] = commit_date
                monthly_changes[month_key][path] += 1

            if diff.b_path and diff.b_path != diff.a_path:
                path = diff.b_path
                file_changes[path] += 1
                file_authors[path].add(commit.author.name)

                if path not in file_first_modified:
                    file_first_modified[path] = commit_date

                file_last_modified[path] = commit_date
                monthly_changes[month_key][path] += 1

    # Calculate change frequency (changes per month)
    file_change_frequency = {}
    for path, changes in file_changes.items():
        if path in file_first_modified and path in file_last_modified:
            first_date = file_first_modified[path]
            last_date = file_last_modified[path]

            # Calculate the number of months between first and last modification
            months = (
                (last_date.year - first_date.year) * 12
                + (last_date.month - first_date.month)
                + 1
            )

            # Avoid division by zero
            if months > 0:
                file_change_frequency[path] = changes / months
            else:
                file_change_frequency[path] = changes

    # Calculate change density (changes per line)
    file_change_density = {}
    for path, changes in file_changes.items():
        try:
            # Try to get the file from the repository
            file_content = repo.git.show(f"{branch}:{path}")
            lines = file_content.count("\n") + 1

            # Avoid division by zero
            if lines > 0:
                file_change_density[path] = changes / lines
            else:
                file_change_density[path] = 0
        except Exception as e:
            # Skip files that can't be retrieved
            logger.warning(f"Error getting file content for {path}: {e}")
            file_change_density[path] = 0

    # Calculate author ownership (percentage of changes by the main author)
    file_ownership = {}
    for path, authors in file_authors.items():
        if not authors:
            file_ownership[path] = 0
            continue

        # Count changes per author for this file
        author_changes = defaultdict(int)
        for commit in commits:
            if len(commit.parents) == 0:
                continue

            parent = commit.parents[0]
            diffs = parent.diff(commit)

            for diff in diffs:
                if (diff.a_path == path) or (
                    diff.b_path == path and diff.b_path != diff.a_path
                ):
                    author_changes[commit.author.name] += 1

        # Find the main author
        main_author = max(author_changes.items(), key=lambda x: x[1], default=(None, 0))

        if main_author[0] is not None and file_changes[path] > 0:
            file_ownership[path] = main_author[1] / file_changes[path]
        else:
            file_ownership[path] = 0

    # Prepare the results
    results = {
        "file_changes": dict(file_changes),
        "file_change_frequency": file_change_frequency,
        "file_change_density": file_change_density,
        "file_ownership": file_ownership,
        "file_authors": {path: list(authors) for path, authors in file_authors.items()},
        "monthly_changes": {
            month: dict(changes) for month, changes in monthly_changes.items()
        },
    }

    # Add summary statistics
    if file_changes:
        results["summary"] = {
            "total_changes": sum(file_changes.values()),
            "avg_changes_per_file": sum(file_changes.values()) / len(file_changes),
            "max_changes": max(file_changes.values()),
            "min_changes": min(file_changes.values()),
            "median_changes": sorted(file_changes.values())[len(file_changes) // 2],
            "most_changed_file": max(file_changes.items(), key=lambda x: x[1])[0],
            "least_changed_file": min(file_changes.items(), key=lambda x: x[1])[0],
        }
    else:
        results["summary"] = {
            "total_changes": 0,
            "avg_changes_per_file": 0,
            "max_changes": 0,
            "min_changes": 0,
            "median_changes": 0,
            "most_changed_file": None,
            "least_changed_file": None,
        }

    logger.info("Change proneness metrics calculated")

    return results


def calculate_error_proneness(repo: Repo, branch: str = "master") -> Dict[str, Any]:
    """
    Calculate error proneness metrics for files in a Git repository.

    Error proneness identifies files that are likely to contain bugs
    based on commit messages that indicate bug fixes.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        Dictionary with error proneness metrics
    """
    logger.info(f"Calculating error proneness metrics for branch: {branch}")

    # Get all commits
    try:
        commits = list(repo.iter_commits(branch))
    except Exception as e:
        logger.warning(f"Error getting commits from branch {branch}: {e}")
        # Try to use the default branch
        try:
            default_branch = repo.active_branch.name
            logger.info(f"Using default branch: {default_branch}")
            commits = list(repo.iter_commits(default_branch))
        except Exception as e2:
            logger.error(f"Error getting commits from default branch: {e2}")
            # Last resort: try to get commits from HEAD
            logger.info("Trying to get commits from HEAD")
            commits = list(repo.iter_commits("HEAD"))

    # Define patterns for bug fix commits
    bug_patterns = [
        r"\bfix(es|ed)?\b",
        r"\bbug(s)?\b",
        r"\bissue(s)?\b",
        r"\berror(s)?\b",
        r"\bdefect(s)?\b",
        r"\bpatch(es|ed)?\b",
        r"\bsolve(s|d)?\b",
        r"\bresolve(s|d)?\b",
        r"\bcorrect(s|ed)?\b",
        r"\baddress(es|ed)?\b",
        r"\bproblem(s)?\b",
        r"\bfault(s)?\b",
        r"\bcrash(es|ed)?\b",
        r"\bexception(s)?\b",
        r"\bfailure(s)?\b",
        r"\bmalfunction(s)?\b",
        r"\bglitch(es)?\b",
        r"\bhotfix(es)?\b",
    ]

    # Compile the patterns
    bug_regex = re.compile("|".join(bug_patterns), re.IGNORECASE)

    # Count bug fixes per file
    file_bugs = defaultdict(int)
    bug_commits = []

    for commit in commits:
        # Check if the commit message indicates a bug fix
        is_bug_fix = bool(bug_regex.search(commit.message))

        if is_bug_fix:
            bug_commits.append(
                {
                    "hash": commit.hexsha,
                    "message": commit.message,
                    "author": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                }
            )

            if len(commit.parents) == 0:
                # Skip the initial commit
                continue

            parent = commit.parents[0]
            diffs = parent.diff(commit)

            for diff in diffs:
                if diff.a_path:
                    file_bugs[diff.a_path] += 1

                if diff.b_path and diff.b_path != diff.a_path:
                    file_bugs[diff.b_path] += 1

    # Calculate bug density (bugs per line)
    file_bug_density = {}
    for path, bugs in file_bugs.items():
        try:
            # Try to get the file from the repository
            file_content = repo.git.show(f"{branch}:{path}")
            lines = file_content.count("\n") + 1

            # Avoid division by zero
            if lines > 0:
                file_bug_density[path] = bugs / lines
            else:
                file_bug_density[path] = 0
        except Exception as e:
            # Skip files that can't be retrieved
            logger.warning(f"Error getting file content for {path}: {e}")
            file_bug_density[path] = 0

    # Calculate bug frequency (bugs per month)
    file_bug_frequency = {}
    if bug_commits:
        # Get the date range of bug fixes
        first_date = datetime.fromisoformat(bug_commits[-1]["date"])
        last_date = datetime.fromisoformat(bug_commits[0]["date"])

        # Calculate the number of months between first and last bug fix
        months = (
            (last_date.year - first_date.year) * 12
            + (last_date.month - first_date.month)
            + 1
        )

        for path, bugs in file_bugs.items():
            # Avoid division by zero
            if months > 0:
                file_bug_frequency[path] = bugs / months
            else:
                file_bug_frequency[path] = bugs

    # Prepare the results
    results = {
        "file_bugs": dict(file_bugs),
        "file_bug_density": file_bug_density,
        "file_bug_frequency": file_bug_frequency,
        "bug_commits": bug_commits,
    }

    # Add summary statistics
    if file_bugs:
        results["summary"] = {
            "total_bugs": sum(file_bugs.values()),
            "avg_bugs_per_file": sum(file_bugs.values()) / len(file_bugs),
            "max_bugs": max(file_bugs.values()),
            "min_bugs": min(file_bugs.values()),
            "median_bugs": sorted(file_bugs.values())[len(file_bugs) // 2],
            "most_buggy_file": max(file_bugs.items(), key=lambda x: x[1])[0],
            "least_buggy_file": min(file_bugs.items(), key=lambda x: x[1])[0],
            "bug_fix_commits": len(bug_commits),
        }
    else:
        results["summary"] = {
            "total_bugs": 0,
            "avg_bugs_per_file": 0,
            "max_bugs": 0,
            "min_bugs": 0,
            "median_bugs": 0,
            "most_buggy_file": None,
            "least_buggy_file": None,
            "bug_fix_commits": 0,
        }

    logger.info("Error proneness metrics calculated")

    return results
