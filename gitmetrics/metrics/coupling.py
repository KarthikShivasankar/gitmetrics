"""
Coupling and cohesion metrics for GitMetrics.

This module provides functions for calculating coupling and cohesion metrics,
which measure the relationships between components in a repository.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

import numpy as np
import pandas as pd
import networkx as nx
from git import Repo, Commit

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


def extract_modules(file_paths: List[str]) -> Dict[str, List[str]]:
    """
    Extract modules from file paths based on directory structure.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping module names to lists of file paths
    """
    modules = defaultdict(list)

    for file_path in file_paths:
        # Skip non-source files
        if not any(
            file_path.endswith(ext)
            for ext in [".py", ".java", ".js", ".cpp", ".c", ".h", ".cs", ".php", ".rb"]
        ):
            continue

        # Extract module name from directory structure
        parts = file_path.split(os.sep)

        if len(parts) > 1:
            # Use the first directory as the module name
            module_name = parts[0]
        else:
            # For files in the root directory, use the file name without extension
            module_name = os.path.splitext(file_path)[0]

        modules[module_name].append(file_path)

    return modules


def calculate_structural_coupling(repo: Repo, branch: str = "master") -> Dict[str, Any]:
    """
    Calculate structural coupling metrics for a repository.

    Structural coupling measures the relationships between modules based on
    how often they change together.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        Dictionary with structural coupling metrics
    """
    logger.info("Calculating structural coupling...")

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

    # Get all files in the repository
    all_files = set()

    for commit in commits:
        if not commit.parents:
            # For the first commit, get all files
            for item in commit.tree.traverse():
                if item.type == "blob":
                    all_files.add(item.path)

    # Extract modules
    modules = extract_modules(list(all_files))

    # Dictionary to store module change data
    module_changes = defaultdict(set)

    # Process each commit
    for commit in commits:
        # Skip merge commits
        if len(commit.parents) > 1:
            continue

        # For the first commit, all files are considered changed
        if not commit.parents:
            changed_files = {
                item.path for item in commit.tree.traverse() if item.type == "blob"
            }
        else:
            # For other commits, get the diff with the parent
            parent = commit.parents[0]
            diffs = parent.diff(commit)

            changed_files = set()
            for diff in diffs:
                if diff.a_path:
                    changed_files.add(diff.a_path)
                if diff.b_path and diff.b_path != diff.a_path:
                    changed_files.add(diff.b_path)

        # Determine which modules were changed in this commit
        changed_modules = set()

        for module_name, module_files in modules.items():
            if any(file_path in changed_files for file_path in module_files):
                changed_modules.add(module_name)
                module_changes[module_name].add(commit.hexsha)

        # If only one module was changed, skip this commit for coupling calculation
        if len(changed_modules) <= 1:
            continue

    # Calculate coupling between modules
    coupling_data = []

    module_names = sorted(modules.keys())

    for i in range(len(module_names)):
        for j in range(i + 1, len(module_names)):
            module1 = module_names[i]
            module2 = module_names[j]

            # Calculate the number of commits where both modules changed
            common_commits = module_changes[module1].intersection(
                module_changes[module2]
            )

            if not common_commits:
                continue

            # Calculate coupling metrics
            coupling_strength = len(common_commits) / (
                len(module_changes[module1])
                + len(module_changes[module2])
                - len(common_commits)
            )

            coupling_data.append(
                {
                    "module1": module1,
                    "module2": module2,
                    "common_commits": len(common_commits),
                    "module1_commits": len(module_changes[module1]),
                    "module2_commits": len(module_changes[module2]),
                    "coupling_strength": coupling_strength,
                }
            )

    # Sort by coupling strength in descending order
    coupling_data.sort(key=lambda x: x["coupling_strength"], reverse=True)

    # Calculate module-level metrics
    module_metrics = []

    for module_name in module_names:
        # Calculate the average coupling strength with other modules
        avg_coupling = 0.0
        count = 0

        for data in coupling_data:
            if data["module1"] == module_name:
                avg_coupling += data["coupling_strength"]
                count += 1
            elif data["module2"] == module_name:
                avg_coupling += data["coupling_strength"]
                count += 1

        if count > 0:
            avg_coupling /= count

        module_metrics.append(
            {
                "module_name": module_name,
                "file_count": len(modules[module_name]),
                "commit_count": len(module_changes[module_name]),
                "avg_coupling_strength": avg_coupling,
            }
        )

    # Sort by average coupling strength in descending order
    module_metrics.sort(key=lambda x: x["avg_coupling_strength"], reverse=True)

    # Create a graph for visualization
    G = nx.Graph()

    for module_name in module_names:
        G.add_node(module_name, size=len(modules[module_name]))

    for data in coupling_data:
        G.add_edge(data["module1"], data["module2"], weight=data["coupling_strength"])

    # Calculate graph metrics
    graph_metrics = {
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_shortest_path_length": (
            nx.average_shortest_path_length(G) if nx.is_connected(G) else None
        ),
    }

    # Prepare the result
    result = {
        "module_count": len(modules),
        "file_count": sum(len(files) for files in modules.values()),
        "coupling_pairs": coupling_data,
        "module_metrics": module_metrics,
        "graph_metrics": graph_metrics,
    }

    logger.info(f"Structural coupling calculated for {len(modules)} modules")

    return result


def calculate_semantic_coupling(repo: Repo, branch: str = "master") -> Dict[str, Any]:
    """
    Calculate semantic coupling metrics for a repository.

    Semantic coupling measures the relationships between modules based on
    shared identifiers and dependencies.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        Dictionary with semantic coupling metrics
    """
    logger.info("Calculating semantic coupling...")

    # Get all files in the repository
    all_files = []

    for root, _, filenames in os.walk(repo.working_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, repo.working_dir)

            # Skip .git directory
            if ".git" in rel_path:
                continue

            all_files.append(rel_path)

    # Extract modules
    modules = extract_modules(all_files)

    # Dictionary to store module dependencies
    module_deps = defaultdict(set)

    # Dictionary to store module identifiers
    module_identifiers = defaultdict(set)

    # Process each file
    for module_name, module_files in modules.items():
        for file_path in module_files:
            try:
                with open(
                    os.path.join(repo.working_dir, file_path), "r", encoding="utf-8"
                ) as f:
                    content = f.read()
            except (UnicodeDecodeError, IsADirectoryError):
                # Skip binary files and directories
                continue

            # Extract identifiers (variable names, function names, class names)
            identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content))
            module_identifiers[module_name].update(identifiers)

            # Extract imports and dependencies
            if file_path.endswith(".py"):
                # Python imports
                imports = re.findall(
                    r"^\s*(?:from|import)\s+([a-zA-Z0-9_.]+)", content, re.MULTILINE
                )

                for imp in imports:
                    # Extract the top-level module
                    top_module = imp.split(".")[0]

                    if top_module in modules and top_module != module_name:
                        module_deps[module_name].add(top_module)

            elif file_path.endswith(".java"):
                # Java imports
                imports = re.findall(
                    r"^\s*import\s+([a-zA-Z0-9_.]+)", content, re.MULTILINE
                )

                for imp in imports:
                    # Extract the top-level package
                    top_package = imp.split(".")[0]

                    if top_package in modules and top_package != module_name:
                        module_deps[module_name].add(top_package)

            elif file_path.endswith(".js"):
                # JavaScript imports
                imports = re.findall(
                    r'(?:import|require)\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)', content
                )
                imports.extend(
                    re.findall(r'import\s+.*\s+from\s+[\'"]([^\'"]*)[\'"]\s*', content)
                )

                for imp in imports:
                    # Extract the module name
                    if imp.startswith("./") or imp.startswith("../"):
                        # Relative import
                        imp_path = os.path.normpath(
                            os.path.join(os.path.dirname(file_path), imp)
                        )
                        imp_dir = imp_path.split(os.sep)[0]

                        if imp_dir in modules and imp_dir != module_name:
                            module_deps[module_name].add(imp_dir)
                    else:
                        # External or top-level import
                        imp_parts = imp.split("/")
                        if imp_parts[0] in modules and imp_parts[0] != module_name:
                            module_deps[module_name].add(imp_parts[0])

    # Calculate semantic coupling based on shared identifiers
    coupling_data = []

    module_names = sorted(modules.keys())

    for i in range(len(module_names)):
        for j in range(i + 1, len(module_names)):
            module1 = module_names[i]
            module2 = module_names[j]

            # Calculate the number of shared identifiers
            shared_identifiers = module_identifiers[module1].intersection(
                module_identifiers[module2]
            )

            if not shared_identifiers:
                continue

            # Calculate coupling metrics
            jaccard_similarity = len(shared_identifiers) / len(
                module_identifiers[module1].union(module_identifiers[module2])
            )

            # Check for direct dependencies
            has_dependency = (
                module1 in module_deps[module2] or module2 in module_deps[module1]
            )

            coupling_data.append(
                {
                    "module1": module1,
                    "module2": module2,
                    "shared_identifiers": len(shared_identifiers),
                    "module1_identifiers": len(module_identifiers[module1]),
                    "module2_identifiers": len(module_identifiers[module2]),
                    "jaccard_similarity": jaccard_similarity,
                    "has_dependency": has_dependency,
                }
            )

    # Sort by Jaccard similarity in descending order
    coupling_data.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

    # Calculate module-level metrics
    module_metrics = []

    for module_name in module_names:
        # Calculate the average Jaccard similarity with other modules
        avg_similarity = 0.0
        count = 0

        for data in coupling_data:
            if data["module1"] == module_name:
                avg_similarity += data["jaccard_similarity"]
                count += 1
            elif data["module2"] == module_name:
                avg_similarity += data["jaccard_similarity"]
                count += 1

        if count > 0:
            avg_similarity /= count

        # Calculate the number of dependencies
        dependency_count = len(module_deps[module_name])
        dependent_count = sum(
            1
            for other_module in module_deps
            if module_name in module_deps[other_module]
        )

        module_metrics.append(
            {
                "module_name": module_name,
                "file_count": len(modules[module_name]),
                "identifier_count": len(module_identifiers[module_name]),
                "dependency_count": dependency_count,
                "dependent_count": dependent_count,
                "avg_similarity": avg_similarity,
            }
        )

    # Sort by average similarity in descending order
    module_metrics.sort(key=lambda x: x["avg_similarity"], reverse=True)

    # Create a graph for visualization
    G = nx.Graph()

    for module_name in module_names:
        G.add_node(module_name, size=len(modules[module_name]))

    for data in coupling_data:
        G.add_edge(data["module1"], data["module2"], weight=data["jaccard_similarity"])

    # Calculate graph metrics
    graph_metrics = {
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "avg_shortest_path_length": (
            nx.average_shortest_path_length(G) if nx.is_connected(G) else None
        ),
    }

    # Prepare the result
    result = {
        "module_count": len(modules),
        "file_count": sum(len(files) for files in modules.values()),
        "coupling_pairs": coupling_data,
        "module_metrics": module_metrics,
        "graph_metrics": graph_metrics,
    }

    logger.info(f"Semantic coupling calculated for {len(modules)} modules")

    return result


def calculate_cohesion(repo: Repo, branch: str = "master") -> Dict[str, Any]:
    """
    Calculate cohesion metrics for a repository.

    Cohesion measures how closely the elements within a module are related.

    Args:
        repo: Git repository
        branch: Branch to analyze

    Returns:
        Dictionary with cohesion metrics
    """
    logger.info("Calculating cohesion...")

    # Get all files in the repository
    all_files = []

    for root, _, filenames in os.walk(repo.working_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, repo.working_dir)

            # Skip .git directory
            if ".git" in rel_path:
                continue

            all_files.append(rel_path)

    # Extract modules
    modules = extract_modules(all_files)

    # Dictionary to store module cohesion data
    module_cohesion = {}

    # Process each module
    for module_name, module_files in modules.items():
        # Skip modules with only one file
        if len(module_files) <= 1:
            continue

        # Extract identifiers from each file
        file_identifiers = {}

        for file_path in module_files:
            try:
                with open(
                    os.path.join(repo.working_dir, file_path), "r", encoding="utf-8"
                ) as f:
                    content = f.read()
            except (UnicodeDecodeError, IsADirectoryError):
                # Skip binary files and directories
                continue

            # Extract identifiers (variable names, function names, class names)
            identifiers = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content))
            file_identifiers[file_path] = identifiers

        # Calculate cohesion metrics

        # LCOM (Lack of Cohesion of Methods) - simplified version
        # Count the number of pairs of files that don't share any identifiers
        non_cohesive_pairs = 0
        total_pairs = 0

        file_paths = sorted(file_identifiers.keys())

        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                file1 = file_paths[i]
                file2 = file_paths[j]

                total_pairs += 1

                # Check if the files share any identifiers
                if not file_identifiers[file1].intersection(file_identifiers[file2]):
                    non_cohesive_pairs += 1

        # Calculate LCOM
        lcom = non_cohesive_pairs / total_pairs if total_pairs > 0 else 0

        # Calculate TCC (Tight Class Cohesion)
        # TCC = number of directly connected method pairs / total number of method pairs
        tcc = 1 - lcom

        # Store cohesion metrics
        module_cohesion[module_name] = {
            "file_count": len(module_files),
            "lcom": lcom,
            "tcc": tcc,
        }

    # Sort modules by TCC in descending order
    sorted_modules = sorted(
        module_cohesion.items(), key=lambda x: x[1]["tcc"], reverse=True
    )

    # Prepare the result
    result = {
        "module_count": len(module_cohesion),
        "module_cohesion": [
            {"module_name": module_name, **metrics}
            for module_name, metrics in sorted_modules
        ],
    }

    logger.info(f"Cohesion calculated for {len(module_cohesion)} modules")

    return result
