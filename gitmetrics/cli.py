"""
Command-line interface for GitMetrics.

This module provides a command-line interface for the GitMetrics package,
allowing users to analyze Git repositories and generate metrics.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional

from gitmetrics.core.repository import GitRepository
from gitmetrics.core.metrics_collector import MetricsCollector
from gitmetrics.visualization.plots import (
    plot_commit_activity,
    plot_author_activity,
    plot_file_changes,
    plot_co_change_network,
    plot_module_coupling,
    plot_error_proneness,
    create_dashboard,
)
from gitmetrics.visualization.dashboard import GitMetricsDashboard
from gitmetrics.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="GitMetrics - A tool for analyzing Git repositories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "repo_path",
        help="Path to the Git repository or GitHub URL",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="gitmetrics_output",
        help="Directory to store the output files",
    )

    parser.add_argument(
        "--branch",
        "-b",
        default="master",
        help="Branch to analyze",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "html", "dashboard"],
        default="json",
        help="Output format",
    )

    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        choices=[
            "all",
            "general",
            "co_change",
            "change_proneness",
            "error_proneness",
            "coupling",
            "cohesion",
        ],
        default=["all"],
        help="Metrics to calculate",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8050,
        help="Port to run the dashboard on (only used with --format=dashboard)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the GitMetrics CLI.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    logger.info(f"Starting GitMetrics analysis for {args.repo_path}")

    try:
        # Determine if the repository is remote
        is_remote = args.repo_path.startswith(
            ("http://", "https://", "git://", "ssh://", "git@")
        )

        # Create the repository object
        if is_remote and args.repo_path.startswith(
            ("https://github.com/", "git@github.com:")
        ):
            logger.info(f"Creating repository object from GitHub URL: {args.repo_path}")
            repo = GitRepository.from_github(args.repo_path)
        else:
            logger.info(f"Creating repository object from path: {args.repo_path}")
            repo = GitRepository(args.repo_path, is_remote=is_remote)

        # Create the metrics collector
        metrics_collector = MetricsCollector(repo)

        # Determine which metrics to calculate
        metrics_to_calculate = args.metrics
        if "all" in metrics_to_calculate:
            logger.info("Calculating all metrics")
            metrics = metrics_collector.collect_all_metrics(args.branch)
        else:
            metrics = {}

            if "general" in metrics_to_calculate:
                logger.info("Calculating general statistics")
                metrics["general_stats"] = metrics_collector.collect_general_stats()
                metrics["commits"] = metrics_collector.metrics.get("commits", [])

            if "co_change" in metrics_to_calculate:
                logger.info("Calculating co-change metrics")
                metrics["co_change"] = metrics_collector.collect_co_change_metrics(
                    args.branch
                )

            if "change_proneness" in metrics_to_calculate:
                logger.info("Calculating change proneness metrics")
                metrics["change_proneness"] = (
                    metrics_collector.collect_change_proneness_metrics(args.branch)
                )

            if "error_proneness" in metrics_to_calculate:
                logger.info("Calculating error proneness metrics")
                metrics["error_proneness"] = (
                    metrics_collector.collect_error_proneness_metrics(args.branch)
                )

            if "coupling" in metrics_to_calculate:
                logger.info("Calculating coupling metrics")
                metrics["coupling"] = {}
                metrics["coupling"]["structural"] = (
                    metrics_collector.collect_structural_coupling_metrics(args.branch)
                )
                metrics["coupling"]["semantic"] = (
                    metrics_collector.collect_semantic_coupling_metrics(args.branch)
                )

            if "cohesion" in metrics_to_calculate:
                logger.info("Calculating cohesion metrics")
                metrics["coupling"] = metrics.get("coupling", {})
                metrics["coupling"]["cohesion"] = (
                    metrics_collector.collect_cohesion_metrics(args.branch)
                )

        # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Export the metrics
        if args.format == "json":
            # Export to JSON
            output_path = os.path.join(args.output_dir, "metrics.json")
            metrics_collector.export_metrics_to_json(output_path)
            logger.info(f"Metrics exported to {output_path}")

        elif args.format == "html":
            # Export to HTML with visualizations
            logger.info("Generating visualizations")

            # Create plots directory
            plots_dir = os.path.join(args.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Generate plots
            plot_paths = {}

            if "commits" in metrics:
                # Commit activity plot
                commit_activity_path = os.path.join(plots_dir, "commit_activity.html")
                plot_commit_activity(metrics["commits"], commit_activity_path)
                plot_paths["commit_activity"] = commit_activity_path

                # Author activity plot
                author_activity_path = os.path.join(plots_dir, "author_activity.html")
                plot_author_activity(metrics["commits"], author_activity_path)
                plot_paths["author_activity"] = author_activity_path

            if "change_proneness" in metrics:
                # File changes plot
                file_changes_path = os.path.join(plots_dir, "file_changes.html")
                file_changes_data = [
                    {"file": file, "changes": changes}
                    for file, changes in metrics["change_proneness"][
                        "file_changes"
                    ].items()
                ]
                plot_file_changes(file_changes_data, file_changes_path)
                plot_paths["file_changes"] = file_changes_path

            if "co_change" in metrics:
                # Co-change network plot
                co_change_path = os.path.join(plots_dir, "co_change_network.html")
                plot_co_change_network(metrics["co_change"], co_change_path)
                plot_paths["co_change_network"] = co_change_path

            if "coupling" in metrics and "structural" in metrics["coupling"]:
                # Module coupling plot
                module_coupling_path = os.path.join(plots_dir, "module_coupling.html")
                plot_module_coupling(
                    metrics["coupling"]["structural"], module_coupling_path
                )
                plot_paths["module_coupling"] = module_coupling_path

            if "error_proneness" in metrics:
                # Error proneness plot
                error_proneness_path = os.path.join(plots_dir, "error_proneness.html")
                error_data = [
                    {"file": file, "errors": bugs}
                    for file, bugs in metrics["error_proneness"]["file_bugs"].items()
                ]
                plot_error_proneness(error_data, error_proneness_path)
                plot_paths["error_proneness"] = error_proneness_path

            # Create dashboard
            dashboard_path = os.path.join(args.output_dir, "dashboard.html")

            # Extract paths from plot_paths dictionary
            commit_activity_path = plot_paths.get("commit_activity", "")
            author_activity_path = plot_paths.get("author_activity", "")
            file_changes_path = plot_paths.get("file_changes", "")
            co_change_path = plot_paths.get("co_change_network", "")
            error_proneness_path = plot_paths.get("error_proneness", "")

            # Get repository stats
            repo_stats = metrics.get("general_stats", {})

            create_dashboard(
                repo_stats,
                commit_activity_path,
                author_activity_path,
                file_changes_path,
                co_change_path,
                error_proneness_path,
                dashboard_path,
            )

            logger.info(f"HTML dashboard created at {dashboard_path}")

        elif args.format == "dashboard":
            # Create an interactive dashboard
            logger.info(f"Starting interactive dashboard on port {args.port}")

            # Get repository name
            repo_name = (
                metrics["general_stats"]["name"]
                if "general_stats" in metrics
                else "Git Repository"
            )

            # Create and run the dashboard
            dashboard = GitMetricsDashboard(repo_name, metrics)
            dashboard.app.run_server(debug=False, port=args.port)

        logger.info("GitMetrics analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during GitMetrics analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
