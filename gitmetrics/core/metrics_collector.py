"""
Metrics collector module for GitMetrics.

This module provides the MetricsCollector class, which is responsible for
collecting and aggregating various metrics from a Git repository.
"""

import os
from typing import Dict, List, Any, Optional

from git import Repo

from gitmetrics.core.repository import GitRepository
from gitmetrics.metrics.co_change import calculate_co_change_metrics
from gitmetrics.metrics.change_proneness import (
    calculate_change_proneness,
    calculate_error_proneness,
)
from gitmetrics.metrics.coupling import (
    calculate_structural_coupling,
    calculate_semantic_coupling,
    calculate_cohesion,
)
from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collector for Git repository metrics.

    This class is responsible for collecting and aggregating various metrics
    from a Git repository.
    """

    def __init__(self, repo: GitRepository):
        """
        Initialize a MetricsCollector instance.

        Args:
            repo: GitRepository instance
        """
        self.repo = repo
        self.metrics = {}

    def collect_all_metrics(self, branch: str = "master") -> Dict[str, Any]:
        """
        Collect all available metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with all collected metrics
        """
        logger.info(f"Collecting all metrics for branch: {branch}")

        # Collect general statistics
        self.collect_general_stats()

        # Collect co-change metrics
        self.collect_co_change_metrics(branch)

        # Collect change proneness metrics
        self.collect_change_proneness_metrics(branch)

        # Collect error proneness metrics
        self.collect_error_proneness_metrics(branch)

        # Collect structural coupling metrics
        self.collect_structural_coupling_metrics(branch)

        # Collect semantic coupling metrics
        self.collect_semantic_coupling_metrics(branch)

        # Collect cohesion metrics
        self.collect_cohesion_metrics(branch)

        logger.info("All metrics collected")

        return self.metrics

    def collect_general_stats(self) -> Dict[str, Any]:
        """
        Collect general statistics about the repository.

        Returns:
            Dictionary with general statistics
        """
        logger.info("Collecting general statistics")

        stats = self.repo.get_general_stats()
        self.metrics["general_stats"] = stats

        # Extract commit data for other visualizations
        commits = self.repo.get_commits()
        commit_data = []

        for commit in commits:
            commit_data.append(
                {
                    "hash": commit.hexsha,
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "date": commit.committed_datetime.isoformat(),
                    "message": commit.message,
                }
            )

        self.metrics["commits"] = commit_data

        return stats

    def collect_co_change_metrics(self, branch: str = "master") -> Dict[str, Any]:
        """
        Collect co-change metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with co-change metrics
        """
        logger.info(f"Collecting co-change metrics for branch: {branch}")

        co_change = calculate_co_change_metrics(self.repo.repo, branch)
        self.metrics["co_change"] = co_change

        return co_change

    def collect_change_proneness_metrics(
        self, branch: str = "master"
    ) -> Dict[str, Any]:
        """
        Collect change proneness metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with change proneness metrics
        """
        logger.info(f"Collecting change proneness metrics for branch: {branch}")

        change_proneness = calculate_change_proneness(self.repo.repo, branch)
        self.metrics["change_proneness"] = change_proneness

        return change_proneness

    def collect_error_proneness_metrics(self, branch: str = "master") -> Dict[str, Any]:
        """
        Collect error proneness metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with error proneness metrics
        """
        logger.info(f"Collecting error proneness metrics for branch: {branch}")

        error_proneness = calculate_error_proneness(self.repo.repo, branch)
        self.metrics["error_proneness"] = error_proneness

        return error_proneness

    def collect_structural_coupling_metrics(
        self, branch: str = "master"
    ) -> Dict[str, Any]:
        """
        Collect structural coupling metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with structural coupling metrics
        """
        logger.info(f"Collecting structural coupling metrics for branch: {branch}")

        structural_coupling = calculate_structural_coupling(self.repo.repo, branch)

        if "coupling" not in self.metrics:
            self.metrics["coupling"] = {}

        self.metrics["coupling"]["structural"] = structural_coupling

        return structural_coupling

    def collect_semantic_coupling_metrics(
        self, branch: str = "master"
    ) -> Dict[str, Any]:
        """
        Collect semantic coupling metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with semantic coupling metrics
        """
        logger.info(f"Collecting semantic coupling metrics for branch: {branch}")

        semantic_coupling = calculate_semantic_coupling(self.repo.repo, branch)

        if "coupling" not in self.metrics:
            self.metrics["coupling"] = {}

        self.metrics["coupling"]["semantic"] = semantic_coupling

        return semantic_coupling

    def collect_cohesion_metrics(self, branch: str = "master") -> Dict[str, Any]:
        """
        Collect cohesion metrics from the repository.

        Args:
            branch: Branch to analyze

        Returns:
            Dictionary with cohesion metrics
        """
        logger.info(f"Collecting cohesion metrics for branch: {branch}")

        cohesion = calculate_cohesion(self.repo.repo, branch)

        if "coupling" not in self.metrics:
            self.metrics["coupling"] = {}

        self.metrics["coupling"]["cohesion"] = cohesion

        return cohesion

    def export_metrics_to_json(self, output_path: str) -> None:
        """
        Export all collected metrics to a JSON file.

        Args:
            output_path: Path to the output JSON file
        """
        import json

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics exported to {output_path}")
