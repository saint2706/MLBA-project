# modern_plot.py
# Modern plotting utilities for TV Script Generation
# Updated for robust handling of multiple log formats
# and compatibility with long-context training setup

import re
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import List, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_metrics(
    log_data: Union[str, List[Dict]],
    metrics: List[str] = ("loss", "lr"), # type: ignore
    title: str = "Training Metrics",
    save_path: str = None, # type: ignore
):
    """
    Plot training metrics over epochs/batches.

    Args:
        log_data: Path to .json/.csv/.pkl log file, or list[dict] with required keys.
        metrics: List of metric names to plot (in order).
        title: Figure title.
        save_path: Optional path to save PNG/SVG image.
    """
    # Load from file or use directly
    if isinstance(log_data, str):
        log_data = log_data.strip()
        if log_data.endswith(".json"):
            data = pd.read_json(log_data)
        elif log_data.endswith(".csv"):
            data = pd.read_csv(log_data)
        elif log_data.endswith(".pkl") or log_data.endswith(".pickle"):
            data = pd.DataFrame(pickle.load(open(log_data, "rb")))
        else:
            raise ValueError(f"Unsupported log file format: {log_data}")
    else:
        data = pd.DataFrame(log_data)

    # Required columns
    required_cols = {"epoch", "batch"} | set(metrics)
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in log data: {missing}")

    # Create subplots: one per metric
    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[m.capitalize() for m in metrics],
    )

    # Plot each metric
    for i, metric in enumerate(metrics, start=1):
        x_vals = data["epoch"] + (data["batch"] / data["batch"].max())
        fig.add_trace(
            go.Scatter(x=x_vals, y=data[metric], mode="lines+markers", name=metric),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=300 * len(metrics),
        title_text=title,
        xaxis_title="Epoch + batch fraction",
        template="plotly_white",
    )

    if save_path:
        fig.write_image(save_path)
        logger.info(f"Saved training metrics plot to {save_path}")

    fig.show()


def parse_log_file(log_path: str) -> List[Dict]:
    """
    Parse a plain-text training log with lines like:
    "Epoch: 1/10 Loss: 2.345 LR: 0.000300"

    Returns:
        list of dicts: Each dict contains epoch, batch, loss, lr.
    """
    pattern = re.compile(
        r"Epoch:\s*(\d+)/\d+\s+Loss:\s*([0-9.]+)\s+LR:\s*([0-9.eE+-]+)"
    )
    records: List[Dict] = []
    batch_num = 0

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                lr = float(match.group(3))
                records.append(
                    {"epoch": epoch, "batch": batch_num, "loss": loss, "lr": lr}
                )
                batch_num += 1

    if not records:
        raise ValueError(f"No valid log entries found in {log_path}")

    logger.info(f"Parsed {len(records)} records from {log_path}")
    return records


# Example manual test
if __name__ == "__main__":
    # Example: parse a text log and plot
    try:
        recs = parse_log_file("train.log")
        plot_training_metrics(
            recs,
            metrics=["loss", "lr"],
            title="Training Progress",
            save_path="metrics.png",
        )
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
