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
from gitmetrics.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="GitMetrics - Git repository metrics and analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "repo_path",
        help="Path to the Git repository or GitHub repository URL",
    )

    parser.add_argument(
        "--branch",
        "-b",
        default="main",
        help="Branch to analyze (default: main)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Directory to save output files (default: ./output)",
    )

    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="Specific metrics to collect",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the GitMetrics CLI.
    """
    # Parse command-line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")

    try:
        # Initialize repository
        if args.repo_path.startswith(("http://", "https://", "git@")):
            repo = GitRepository.from_github(args.repo_path)
        else:
            repo = GitRepository(args.repo_path)

        # Create metrics collector
        collector = MetricsCollector(repo)

        # Collect metrics
        if args.metrics:
            metrics = {}
            for metric in args.metrics:
                if metric == "general":
                    metrics["general_stats"] = collector.collect_general_stats()
                elif metric == "co_change":
                    metrics["co_change"] = collector.collect_co_change_metrics(args.branch)
                elif metric == "change_proneness":
                    metrics["change_proneness"] = collector.collect_change_proneness_metrics(args.branch)
                elif metric == "error_proneness":
                    metrics["error_proneness"] = collector.collect_error_proneness_metrics(args.branch)
                elif metric == "structural_coupling":
                    metrics["structural_coupling"] = collector.collect_structural_coupling_metrics(args.branch)
                elif metric == "semantic_coupling":
                    metrics["semantic_coupling"] = collector.collect_semantic_coupling_metrics(args.branch)
                elif metric == "cohesion":
                    metrics["cohesion"] = collector.collect_cohesion_metrics(args.branch)
        else:
            metrics = collector.collect_all_metrics(args.branch)

        # Export metrics to JSON
        output_file = os.path.join(args.output_dir, "metrics.json")
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics exported to {output_file}")

    except Exception as e:
        logger.error(f"Error during GitMetrics analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
