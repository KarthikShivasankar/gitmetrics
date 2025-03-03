# GitMetrics

A Python package for generating and visualizing various Git repository statistics and metrics to help development teams gain insights into their codebase and development practices.

## Features

GitMetrics provides tools for extracting and analyzing the following metrics from Git repositories:

### Repository Metrics

- **Co-change metrics**: Identify files that frequently change together, helping to detect hidden dependencies
- **Change proneness**: Identify files that are frequently modified, which may indicate design issues
- **Error proneness**: Identify files that are frequently involved in bug fixes, helping to focus testing efforts
- **Structural coupling**: Measure how components are structurally related based on dependencies
- **Semantic coupling**: Measure how components are semantically related based on content
- **Module cohesion**: Measure how well modules are designed in terms of internal relationships

### Repository Statistics

- **General statistics**: Files, lines of code, commits, authors, repository age
- **Activity statistics**: Commits by time periods, development trends
- **Author statistics**: Contributions by author, team collaboration patterns
- **File statistics**: Changes by file, hotspots in the codebase
- **Code churn**: Line additions and deletions over time
- **Bug patterns**: Frequency and distribution of bug fixes

## Installation

### From PyPI

```bash
pip install gitmetrics
```

### Local Development Installation

To install GitMetrics for local development:

1. Clone the repository:

   ```bash
   git clone https://github.com/example/gitmetrics.git
   cd gitmetrics
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   # Using venv
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install in development mode:

   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from gitmetrics.core.repository import GitRepository
from gitmetrics.core.metrics_collector import MetricsCollector

# Analyze a local repository
repo = GitRepository('/path/to/local/repo')

# Or analyze a remote GitHub repository
repo = GitRepository.from_github('https://github.com/username/repo.git')

# Create a metrics collector
collector = MetricsCollector(repo)

# Collect all metrics
metrics = collector.collect_all_metrics()

# Or collect specific metrics
general_stats = collector.collect_general_stats()
co_change = collector.collect_co_change_metrics()
error_proneness = collector.collect_error_proneness_metrics()

# Export metrics to JSON
collector.export_metrics_to_json('metrics.json')
```

### Command Line Interface

GitMetrics provides a powerful command-line interface:

```bash
# Analyze a repository with all metrics and output JSON
gitmetrics /path/to/repo --output-dir ./output --format json

# Generate HTML visualizations
gitmetrics /path/to/repo --output-dir ./output --format html

# Launch an interactive dashboard
gitmetrics /path/to/repo --format dashboard --port 8050

# Analyze specific metrics only
gitmetrics /path/to/repo --metrics general co_change error_proneness

# Analyze a specific branch
gitmetrics /path/to/repo --branch develop
```

## Visualizations

GitMetrics generates various visualizations to help teams understand their codebase:

1. **Commit Activity**: Track development activity over time
2. **Author Activity**: Visualize team contributions
3. **File Changes**: Identify the most frequently changed files
4. **Co-change Network**: Discover hidden dependencies between files
5. **Error Proneness**: Identify bug-prone files that need attention
6. **Module Coupling**: Understand relationships between modules

## Team Benefits

GitMetrics helps development teams in several ways:

- **Identify Hotspots**: Find areas of the codebase that require frequent changes and may need refactoring
- **Improve Code Quality**: Target testing and code reviews on error-prone files
- **Enhance Team Collaboration**: Understand how team members interact with the codebase
- **Optimize Architecture**: Detect hidden dependencies and improve module design
- **Track Progress**: Monitor development trends and the impact of improvement initiatives
- **Onboard New Developers**: Help new team members understand the codebase structure and history

## Example Dashboard

GitMetrics can generate an interactive dashboard with all metrics and visualizations:

```bash
gitmetrics /path/to/repo --format dashboard
```

Then open your browser at `http://localhost:8050` to explore the metrics.

## License

MIT License
