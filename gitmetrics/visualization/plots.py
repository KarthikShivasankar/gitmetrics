"""
Plotting functions for GitMetrics.

This module provides functions for creating visualizations of Git repository metrics.
"""

import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime, timedelta

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


def plot_commit_activity(
    commits: List[Dict[str, Any]], output_path: str, title: str = "Commit Activity"
) -> None:
    """
    Plot commit activity over time.

    Args:
        commits: List of commit data dictionaries
        output_path: Path to save the plot
        title: Plot title
    """
    logger.info("Plotting commit activity...")

    # Extract dates and count commits per day
    dates = [datetime.fromisoformat(commit["date"]) for commit in commits]

    # Create a DataFrame with dates
    df = pd.DataFrame({"date": dates})
    df["date"] = pd.to_datetime(df["date"])

    # Count commits per day
    daily_counts = df.groupby(df["date"].dt.date).size().reset_index(name="count")
    daily_counts["date"] = pd.to_datetime(daily_counts["date"])

    # Sort by date
    daily_counts = daily_counts.sort_values("date")

    # Create the plot
    fig = px.line(
        daily_counts,
        x="date",
        y="count",
        title=title,
        labels={"date": "Date", "count": "Number of Commits"},
    )

    # Add markers for each data point
    fig.update_traces(mode="lines+markers")

    # Improve layout
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Number of Commits", hovermode="x unified"
    )

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"Commit activity plot saved to {output_path}")


def plot_author_activity(
    commits: List[Dict[str, Any]], output_path: str, top_n: int = 10
) -> None:
    """
    Plot commit activity by author.

    Args:
        commits: List of commit data dictionaries
        output_path: Path to save the plot
        top_n: Number of top authors to include
    """
    logger.info("Plotting author activity...")

    # Count commits by author
    author_counts = {}
    for commit in commits:
        author = commit["author"]
        author_counts[author] = author_counts.get(author, 0) + 1

    # Sort authors by commit count
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)

    # Take the top N authors
    top_authors = sorted_authors[:top_n]

    # Create a DataFrame
    df = pd.DataFrame(top_authors, columns=["author", "commits"])

    # Create the plot
    fig = px.bar(
        df,
        x="author",
        y="commits",
        title=f"Top {top_n} Contributors",
        labels={"author": "Author", "commits": "Number of Commits"},
        color="commits",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Author",
        yaxis_title="Number of Commits",
        xaxis={"categoryorder": "total descending"},
    )

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"Author activity plot saved to {output_path}")


def plot_file_changes(
    files: List[Dict[str, Any]], output_path: str, top_n: int = 20
) -> None:
    """
    Plot the most frequently changed files.

    Args:
        files: List of file data dictionaries
        output_path: Path to save the plot
        top_n: Number of top files to include
    """
    logger.info("Plotting file changes...")

    # Sort files by change count
    sorted_files = sorted(files, key=lambda x: x["commit_count"], reverse=True)

    # Take the top N files
    top_files = sorted_files[:top_n]

    # Create a DataFrame
    df = pd.DataFrame(top_files)

    # Shorten file paths for display
    df["short_path"] = df["file_path"].apply(
        lambda x: os.path.basename(x) if len(x) > 30 else x
    )

    # Create the plot
    fig = px.bar(
        df,
        x="commit_count",
        y="short_path",
        title=f"Top {top_n} Most Changed Files",
        labels={"commit_count": "Number of Changes", "short_path": "File"},
        color="commit_count",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_data=["file_path", "lines_added", "lines_removed"],
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Number of Changes",
        yaxis_title="File",
        yaxis={"categoryorder": "total ascending"},
    )

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"File changes plot saved to {output_path}")


def plot_co_change_network(
    co_change_data: Dict[str, Any], output_path: str, min_strength: float = 0.1
) -> None:
    """
    Plot a network of co-changing files.

    Args:
        co_change_data: Co-change metrics data
        output_path: Path to save the plot
        min_strength: Minimum coupling strength to include in the plot
    """
    logger.info("Plotting co-change network...")

    # Create a graph
    G = nx.Graph()

    # Add nodes (files)
    for file_data in co_change_data["top_coupled_files"]:
        file_path = file_data["file"]
        G.add_node(file_path, size=file_data["avg_coupling_strength"] * 10)

    # Add edges (co-change relationships)
    for pair_data in co_change_data["top_coupled_pairs"]:
        if pair_data["coupling_strength"] >= min_strength:
            G.add_edge(
                pair_data["file1"],
                pair_data["file2"],
                weight=pair_data["coupling_strength"],
                co_changes=pair_data["co_changes"],
            )

    # Calculate layout
    pos = nx.spring_layout(G, seed=42)

    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G.edges[edge]["weight"]
        co_changes = G.edges[edge]["co_changes"]
        edge_text.append(f"Strength: {weight:.2f}, Co-changes: {co_changes}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        text=edge_text,
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]["size"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=node_size,
            colorbar=dict(
                thickness=15,
                title="Coupling Strength",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="File Co-change Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"Co-change network plot saved to {output_path}")


def plot_module_coupling(coupling_data: Dict[str, Any], output_path: str) -> None:
    """
    Plot a heatmap of module coupling.

    Args:
        coupling_data: Module coupling data
        output_path: Path to save the plot
    """
    logger.info("Plotting module coupling...")

    # Extract module names and create a matrix
    module_names = []
    for metric in coupling_data["module_metrics"]:
        module_names.append(metric["module_name"])

    # Create an empty matrix
    n = len(module_names)
    matrix = np.zeros((n, n))

    # Fill the matrix with coupling strengths
    for pair in coupling_data["coupling_pairs"]:
        i = module_names.index(pair["module1"])
        j = module_names.index(pair["module2"])
        matrix[i, j] = pair["coupling_strength"]
        matrix[j, i] = pair["coupling_strength"]  # Symmetric

    # Create the heatmap
    fig = px.imshow(
        matrix,
        x=module_names,
        y=module_names,
        color_continuous_scale="Viridis",
        title="Module Coupling Heatmap",
    )

    # Improve layout
    fig.update_layout(xaxis_title="Module", yaxis_title="Module")

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"Module coupling plot saved to {output_path}")


def plot_error_proneness(
    error_data: Dict[str, Any], output_path: str, top_n: int = 15
) -> None:
    """
    Plot error proneness of files.

    Args:
        error_data: Error proneness data
        output_path: Path to save the plot
        top_n: Number of top files to include
    """
    logger.info("Plotting error proneness...")

    # Get the top error-prone files
    top_files = error_data["top_error_prone_files"][:top_n]

    # Create a DataFrame
    df = pd.DataFrame(top_files)

    # Shorten file paths for display
    df["short_path"] = df["file_path"].apply(
        lambda x: os.path.basename(x) if len(x) > 30 else x
    )

    # Create the plot
    fig = px.bar(
        df,
        x="error_frequency",
        y="short_path",
        title=f"Top {top_n} Error-Prone Files",
        labels={"error_frequency": "Error Frequency", "short_path": "File"},
        color="error_frequency",
        color_continuous_scale="Reds",
        hover_data=["file_path", "error_commit_count", "total_commit_count"],
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="Error Frequency",
        yaxis_title="File",
        yaxis={"categoryorder": "total ascending"},
    )

    # Save the plot
    fig.write_html(output_path)
    logger.info(f"Error proneness plot saved to {output_path}")


def create_dashboard(
    repo_stats: Dict[str, Any],
    commit_activity_path: str,
    author_activity_path: str,
    file_changes_path: str,
    co_change_path: str,
    error_proneness_path: str,
    output_path: str,
) -> None:
    """
    Create a dashboard HTML page that includes all plots.

    Args:
        repo_stats: Repository statistics
        commit_activity_path: Path to commit activity plot
        author_activity_path: Path to author activity plot
        file_changes_path: Path to file changes plot
        co_change_path: Path to co-change network plot
        error_proneness_path: Path to error proneness plot
        output_path: Path to save the dashboard
    """
    logger.info("Creating dashboard...")

    # Create the HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Git Repository Analysis Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .stats-container {{ 
                display: flex; 
                flex-wrap: wrap; 
                justify-content: space-between;
                margin-bottom: 20px;
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
            }}
            .stat-box {{ 
                background-color: white; 
                padding: 15px; 
                margin: 10px; 
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 200px;
                text-align: center;
            }}
            .stat-value {{ 
                font-size: 24px; 
                font-weight: bold; 
                margin: 10px 0; 
                color: #2c3e50;
            }}
            .stat-label {{ color: #7f8c8d; }}
            .plot-container {{ 
                margin: 20px 0; 
                padding: 15px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            iframe {{ 
                width: 100%; 
                height: 500px; 
                border: none;
            }}
        </style>
    </head>
    <body>
        <h1>Git Repository Analysis Dashboard</h1>
        
        <div class="stats-container">
            <div class="stat-box">
                <div class="stat-label">Commits</div>
                <div class="stat-value">{repo_stats['num_commits']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Files</div>
                <div class="stat-value">{repo_stats['num_files']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Authors</div>
                <div class="stat-value">{repo_stats['num_authors']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Age (days)</div>
                <div class="stat-value">{repo_stats['age_days']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Lines of Code</div>
                <div class="stat-value">{repo_stats['total_lines']}</div>
            </div>
        </div>
        
        <h2>Commit Activity</h2>
        <div class="plot-container">
            <iframe src="{os.path.basename(commit_activity_path)}"></iframe>
        </div>
        
        <h2>Author Activity</h2>
        <div class="plot-container">
            <iframe src="{os.path.basename(author_activity_path)}"></iframe>
        </div>
        
        <h2>Most Changed Files</h2>
        <div class="plot-container">
            <iframe src="{os.path.basename(file_changes_path)}"></iframe>
        </div>
        
        <h2>Co-change Network</h2>
        <div class="plot-container">
            <iframe src="{os.path.basename(co_change_path)}"></iframe>
        </div>
        
        <h2>Error-Prone Files</h2>
        <div class="plot-container">
            <iframe src="{os.path.basename(error_proneness_path)}"></iframe>
        </div>
        
        <footer>
            <p>Generated with GitMetrics on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """

    # Write the HTML file
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard saved to {output_path}")
