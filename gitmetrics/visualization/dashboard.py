"""
Dashboard module for GitMetrics.

This module provides functions for creating interactive dashboards to visualize
Git repository metrics.
"""

import os
import json
from typing import Dict, List, Any, Optional

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np

from gitmetrics.utils.logger import get_logger

logger = get_logger(__name__)


class GitMetricsDashboard:
    """
    Dashboard for visualizing Git repository metrics.

    This class provides methods for creating and running an interactive Dash
    application to visualize Git repository metrics.
    """

    def __init__(self, repo_name: str, metrics_data: Dict[str, Any]):
        """
        Initialize a GitMetricsDashboard instance.

        Args:
            repo_name: Name of the repository
            metrics_data: Dictionary containing all metrics data
        """
        self.repo_name = repo_name
        self.metrics_data = metrics_data
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ],
        )
        self.app.title = f"GitMetrics - {repo_name}"

        # Set up the layout
        self.app.layout = self._create_layout()

        # Set up callbacks
        self._setup_callbacks()

    def _create_layout(self) -> html.Div:
        """
        Create the layout for the dashboard.

        Returns:
            Dash layout
        """
        # Extract general stats
        general_stats = self.metrics_data.get("general_stats", {})

        # Create stat cards
        stat_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Commits", className="card-title"),
                                    html.H2(
                                        f"{general_stats.get('num_commits', 0):,}",
                                        className="card-value",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=12,
                    sm=6,
                    md=4,
                    lg=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Files", className="card-title"),
                                    html.H2(
                                        f"{general_stats.get('num_files', 0):,}",
                                        className="card-value",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=12,
                    sm=6,
                    md=4,
                    lg=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Authors", className="card-title"),
                                    html.H2(
                                        f"{general_stats.get('num_authors', 0):,}",
                                        className="card-value",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=12,
                    sm=6,
                    md=4,
                    lg=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Age (days)", className="card-title"),
                                    html.H2(
                                        f"{general_stats.get('age_days', 0):,}",
                                        className="card-value",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=12,
                    sm=6,
                    md=4,
                    lg=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Lines of Code", className="card-title"),
                                    html.H2(
                                        f"{general_stats.get('total_lines', 0):,}",
                                        className="card-value",
                                    ),
                                ]
                            )
                        ]
                    ),
                    width=12,
                    sm=6,
                    md=4,
                    lg=2,
                ),
            ],
            className="mb-4",
        )

        # Create tabs for different visualizations
        tabs = dbc.Tabs(
            [
                dbc.Tab(label="Commit Activity", tab_id="tab-commit-activity"),
                dbc.Tab(label="Author Activity", tab_id="tab-author-activity"),
                dbc.Tab(label="File Changes", tab_id="tab-file-changes"),
                dbc.Tab(label="Co-change Network", tab_id="tab-co-change"),
                dbc.Tab(label="Error Proneness", tab_id="tab-error-proneness"),
                dbc.Tab(label="Module Coupling", tab_id="tab-module-coupling"),
            ],
            id="tabs",
            active_tab="tab-commit-activity",
        )

        # Create the tab content container
        tab_content = html.Div(id="tab-content", className="p-4")

        # Create the layout
        layout = html.Div(
            [
                dbc.Container(
                    [
                        html.H1(
                            f"Git Repository Analysis: {self.repo_name}",
                            className="my-4",
                        ),
                        stat_cards,
                        tabs,
                        tab_content,
                        html.Hr(),
                        html.Footer(
                            [
                                html.P(
                                    "Generated with GitMetrics",
                                    className="text-center text-muted",
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                )
            ]
        )

        return layout

    def _setup_callbacks(self) -> None:
        """Set up the callbacks for the dashboard."""

        @self.app.callback(
            Output("tab-content", "children"), Input("tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            """Render the content for the active tab."""
            if active_tab == "tab-commit-activity":
                return self._create_commit_activity_content()
            elif active_tab == "tab-author-activity":
                return self._create_author_activity_content()
            elif active_tab == "tab-file-changes":
                return self._create_file_changes_content()
            elif active_tab == "tab-co-change":
                return self._create_co_change_content()
            elif active_tab == "tab-error-proneness":
                return self._create_error_proneness_content()
            elif active_tab == "tab-module-coupling":
                return self._create_module_coupling_content()
            return html.P("This tab is not yet implemented")

    def _create_commit_activity_content(self) -> html.Div:
        """
        Create content for the commit activity tab.

        Returns:
            Dash layout for commit activity visualization
        """
        # Extract commit data
        commits = self.metrics_data.get("commits", [])

        if not commits:
            return html.Div(
                [
                    html.H3("Commit Activity"),
                    html.P("No commit data available"),
                ]
            )

        # Extract dates and count commits per day
        dates = [
            pd.to_datetime(commit.get("date")) for commit in commits if "date" in commit
        ]

        # Create a DataFrame with dates
        df = pd.DataFrame({"date": dates})

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
            title="Commit Activity Over Time",
            labels={"date": "Date", "count": "Number of Commits"},
        )

        # Add markers for each data point
        fig.update_traces(mode="lines+markers")

        # Improve layout
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Number of Commits", hovermode="x unified"
        )

        return html.Div(
            [
                html.H3("Commit Activity"),
                dcc.Graph(figure=fig),
                html.P(f"Total commits: {len(commits)}"),
            ]
        )

    def _create_author_activity_content(self) -> html.Div:
        """
        Create content for the author activity tab.

        Returns:
            Dash layout for author activity visualization
        """
        # Extract commit data
        commits = self.metrics_data.get("commits", [])

        if not commits:
            return html.Div(
                [
                    html.H3("Author Activity"),
                    html.P("No commit data available"),
                ]
            )

        # Count commits by author
        author_counts = {}
        for commit in commits:
            author = commit.get("author", "Unknown")
            author_counts[author] = author_counts.get(author, 0) + 1

        # Sort authors by commit count
        sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)

        # Take the top 10 authors
        top_authors = sorted_authors[:10]

        # Create a DataFrame
        df = pd.DataFrame(top_authors, columns=["author", "commits"])

        # Create the plot
        fig = px.bar(
            df,
            x="author",
            y="commits",
            title="Top 10 Contributors",
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

        return html.Div(
            [
                html.H3("Author Activity"),
                dcc.Graph(figure=fig),
                html.P(f"Total authors: {len(author_counts)}"),
            ]
        )

    def _create_file_changes_content(self) -> html.Div:
        """
        Create content for the file changes tab.

        Returns:
            Dash layout for file changes visualization
        """
        # Extract file change data
        change_proneness = self.metrics_data.get("change_proneness", {})
        files = change_proneness.get("all_files", [])

        if not files:
            return html.Div(
                [
                    html.H3("File Changes"),
                    html.P("No file change data available"),
                ]
            )

        # Sort files by change count
        sorted_files = sorted(
            files, key=lambda x: x.get("commit_count", 0), reverse=True
        )

        # Take the top 20 files
        top_files = sorted_files[:20]

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
            title="Top 20 Most Changed Files",
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

        return html.Div(
            [
                html.H3("File Changes"),
                dcc.Graph(figure=fig),
                html.P(f"Total files: {len(files)}"),
            ]
        )

    def _create_co_change_content(self) -> html.Div:
        """
        Create content for the co-change network tab.

        Returns:
            Dash layout for co-change network visualization
        """
        # Extract co-change data
        co_change = self.metrics_data.get("co_change", {})

        if not co_change:
            return html.Div(
                [
                    html.H3("Co-change Network"),
                    html.P("No co-change data available"),
                ]
            )

        # Create a graph
        G = nx.Graph()

        # Add nodes (files)
        for file_data in co_change.get("top_coupled_files", []):
            file_path = file_data.get("file", "")
            G.add_node(file_path, size=file_data.get("avg_coupling_strength", 0) * 10)

        # Add edges (co-change relationships)
        for pair_data in co_change.get("top_coupled_pairs", []):
            if pair_data.get("coupling_strength", 0) >= 0.1:
                G.add_edge(
                    pair_data.get("file1", ""),
                    pair_data.get("file2", ""),
                    weight=pair_data.get("coupling_strength", 0),
                    co_changes=pair_data.get("co_changes", 0),
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

        return html.Div(
            [
                html.H3("Co-change Network"),
                dcc.Graph(figure=fig),
                html.P(
                    f"Number of file pairs: {len(co_change.get('top_coupled_pairs', []))}"
                ),
            ]
        )

    def _create_error_proneness_content(self) -> html.Div:
        """
        Create content for the error proneness tab.

        Returns:
            Dash layout for error proneness visualization
        """
        # Extract error proneness data
        error_proneness = self.metrics_data.get("error_proneness", {})

        if not error_proneness:
            return html.Div(
                [
                    html.H3("Error Proneness"),
                    html.P("No error proneness data available"),
                ]
            )

        # Get the top error-prone files
        top_files = error_proneness.get("top_error_prone_files", [])[:15]

        if not top_files:
            return html.Div(
                [
                    html.H3("Error Proneness"),
                    html.P("No error-prone files found"),
                ]
            )

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
            title="Top 15 Error-Prone Files",
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

        return html.Div(
            [
                html.H3("Error Proneness"),
                dcc.Graph(figure=fig),
                html.P(
                    f"Total error-prone files: {len(error_proneness.get('all_files', []))}"
                ),
            ]
        )

    def _create_module_coupling_content(self) -> html.Div:
        """
        Create content for the module coupling tab.

        Returns:
            Dash layout for module coupling visualization
        """
        # Extract module coupling data
        coupling = self.metrics_data.get("coupling", {})

        if not coupling:
            return html.Div(
                [
                    html.H3("Module Coupling"),
                    html.P("No module coupling data available"),
                ]
            )

        # Extract module names and create a matrix
        module_metrics = coupling.get("module_metrics", [])
        module_names = [metric.get("module_name", "") for metric in module_metrics]

        if not module_names:
            return html.Div(
                [
                    html.H3("Module Coupling"),
                    html.P("No modules found"),
                ]
            )

        # Create an empty matrix
        n = len(module_names)
        matrix = np.zeros((n, n))

        # Fill the matrix with coupling strengths
        for pair in coupling.get("coupling_pairs", []):
            if (
                pair.get("module1", "") in module_names
                and pair.get("module2", "") in module_names
            ):
                i = module_names.index(pair.get("module1", ""))
                j = module_names.index(pair.get("module2", ""))
                matrix[i, j] = pair.get("coupling_strength", 0)
                matrix[j, i] = pair.get("coupling_strength", 0)  # Symmetric

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

        return html.Div(
            [
                html.H3("Module Coupling"),
                dcc.Graph(figure=fig),
                html.P(f"Number of modules: {len(module_names)}"),
            ]
        )

    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """
        Run the dashboard server.

        Args:
            debug: Whether to run the server in debug mode
            port: Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)


def create_dashboard(
    repo_name: str, metrics_data: Dict[str, Any], output_dir: str, port: int = 8050
) -> None:
    """
    Create and run a dashboard for visualizing Git repository metrics.

    Args:
        repo_name: Name of the repository
        metrics_data: Dictionary containing all metrics data
        output_dir: Directory to save dashboard assets
        port: Port to run the dashboard on
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the metrics data to a JSON file
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Create and run the dashboard
    dashboard = GitMetricsDashboard(repo_name, metrics_data)
    dashboard.run_server(debug=True, port=port)
