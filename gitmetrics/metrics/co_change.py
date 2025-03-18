"""
Co-change metrics for GitMetrics.

This module provides functions for calculating co-change metrics,
which identify files that frequently change together.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import pandas as pd
from git import Repo, Commit

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


def get_changed_files(repo: Repo, commit: Commit) -> Set[str]:
    """
    Get the set of files changed in a commit.

    Args:
        repo: Git repository
        commit: Commit to analyze

    Returns:
        Set of file paths that were changed in the commit
    """
    if not commit.parents:
        # For the first commit, get all files
        return {item.path for item in commit.tree.traverse() if item.type == "blob"}

    # For other commits, get the diff with the parent
    parent = commit.parents[0]
    diffs = parent.diff(commit)

    changed_files = set()
    for diff in diffs:
        if diff.a_path:
            changed_files.add(diff.a_path)
        if diff.b_path and diff.b_path != diff.a_path:
            changed_files.add(diff.b_path)

    return changed_files


def calculate_co_change_matrix(
    repo: Repo, branch: str = "main", min_count: int = 2
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate the co-change matrix for a repository.

    The co-change matrix shows how often files change together in commits.

    Args:
        repo: Git repository
        branch: Branch to analyze
        min_count: Minimum number of co-changes to include in the result

    Returns:
        Tuple of (co-change matrix DataFrame, metadata)
    """
    logger.info("Calculating co-change matrix...")

    try:
        # Try to use the specified branch
        commits = list(repo.iter_commits(branch))
    except Exception:
        # If the branch doesn't exist, try 'main' instead
        try:
            commits = list(repo.iter_commits("main"))
        except Exception:
            # If 'main' doesn't exist either, use HEAD
            commits = list(repo.iter_commits("HEAD"))

    # Dictionary to store co-change counts
    co_change_counts = defaultdict(int)

    # Dictionary to store individual file change counts
    file_change_counts = defaultdict(int)

    # Process each commit
    for commit in commits:
        changed_files = get_changed_files(repo, commit)

        # Update individual file change counts
        for file_path in changed_files:
            file_change_counts[file_path] += 1

        # Update co-change counts for each pair of files
        changed_files_list = sorted(list(changed_files))
        for i in range(len(changed_files_list)):
            for j in range(i + 1, len(changed_files_list)):
                file1 = changed_files_list[i]
                file2 = changed_files_list[j]

                # Store the pair in a consistent order
                if file1 > file2:
                    file1, file2 = file2, file1

                co_change_counts[(file1, file2)] += 1

    # Filter out pairs with fewer than min_count co-changes
    filtered_co_changes = {
        pair: count for pair, count in co_change_counts.items() if count >= min_count
    }

    # Get the set of all files involved in co-changes
    all_files = set()
    for file1, file2 in filtered_co_changes.keys():
        all_files.add(file1)
        all_files.add(file2)

    # Sort the files for consistent ordering
    all_files = sorted(list(all_files))

    # Create a DataFrame for the co-change matrix
    matrix = pd.DataFrame(0, index=all_files, columns=all_files)

    # Fill in the co-change counts
    for (file1, file2), count in filtered_co_changes.items():
        matrix.loc[file1, file2] = count
        matrix.loc[file2, file1] = count

    # Set the diagonal to the individual file change counts
    for file_path in all_files:
        matrix.loc[file_path, file_path] = file_change_counts[file_path]

    # Calculate metadata
    metadata = {
        "total_commits": len(commits),
        "total_files": len(file_change_counts),
        "files_with_co_changes": len(all_files),
        "total_co_change_pairs": len(filtered_co_changes),
        "min_count": min_count,
    }

    logger.info(
        f"Co-change matrix calculated: {metadata['total_co_change_pairs']} pairs found"
    )

    return matrix, metadata


def calculate_co_change_metrics(
    repo: Repo, branch: str = "main", min_count: int = 2
) -> Dict[str, Any]:
    """
    Calculate various co-change metrics for a repository.

    Args:
        repo: Git repository
        branch: Branch to analyze
        min_count: Minimum number of co-changes to include in the result

    Returns:
        Dictionary with co-change metrics
    """
    matrix, metadata = calculate_co_change_matrix(repo, branch, min_count)

    # Calculate additional metrics

    # Coupling strength: ratio of co-changes to total changes
    coupling_strength = pd.DataFrame(0.0, index=matrix.index, columns=matrix.columns)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            if i != j:
                file1 = matrix.index[i]
                file2 = matrix.columns[j]

                # Coupling strength = co-changes / (changes_file1 + changes_file2 - co-changes)
                co_changes = matrix.iloc[i, j]
                changes_file1 = matrix.iloc[i, i]
                changes_file2 = matrix.iloc[j, j]

                if co_changes > 0:
                    denominator = changes_file1 + changes_file2 - co_changes
                    if denominator > 0:
                        coupling_strength.iloc[i, j] = co_changes / denominator

    # Find the top coupled file pairs
    top_pairs = []

    for i in range(len(coupling_strength.index)):
        for j in range(i + 1, len(coupling_strength.columns)):
            file1 = coupling_strength.index[i]
            file2 = coupling_strength.columns[j]

            strength = coupling_strength.iloc[i, j]
            co_changes = matrix.iloc[i, j]

            if strength > 0:
                top_pairs.append(
                    {
                        "file1": file1,
                        "file2": file2,
                        "co_changes": int(co_changes),
                        "coupling_strength": float(strength),
                    }
                )

    # Sort by coupling strength in descending order
    top_pairs.sort(key=lambda x: x["coupling_strength"], reverse=True)

    # Calculate the average coupling strength for each file
    avg_coupling = {}

    for file_path in matrix.index:
        strengths = []

        for other_file in matrix.columns:
            if file_path != other_file:
                i = matrix.index.get_loc(file_path)
                j = matrix.columns.get_loc(other_file)

                strength = coupling_strength.iloc[i, j]
                if strength > 0:
                    strengths.append(strength)

        if strengths:
            avg_coupling[file_path] = sum(strengths) / len(strengths)
        else:
            avg_coupling[file_path] = 0.0

    # Sort files by average coupling strength
    sorted_files = sorted(avg_coupling.items(), key=lambda x: x[1], reverse=True)

    # Prepare the result
    result = {
        "metadata": metadata,
        "top_coupled_pairs": top_pairs[:20],  # Top 20 pairs
        "top_coupled_files": [
            {"file": file_path, "avg_coupling_strength": float(strength)}
            for file_path, strength in sorted_files[:20]  # Top 20 files
        ],
        "matrix": matrix.to_dict(),
        "coupling_strength": coupling_strength.to_dict(),
    }

    return result
