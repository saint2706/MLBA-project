# modern_plot.py
# 📊 Modern plotting utilities for TV Script Generation
# This file creates beautiful visualizations to help understand model training
# 
# FOR NON-PROGRAMMERS:
# - This file creates graphs and charts to visualize how well our AI is learning
# - Think of it like creating report cards for our AI model
# - The graphs show things like "how much is the AI improving over time?"
# 
# Updated for robust handling of multiple log formats
# and compatibility with long-context training setup

import re
import pickle
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from pathlib import Path

# Set up logging (this helps us track what the program is doing)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_training_metrics(
    log_data: Union[str, List[Dict]],
    metrics: List[str] = ["loss", "lr"],  # Default metrics to plot
    title: str = "Training Metrics", 
    save_path: Optional[str] = None,
    show_validation: bool = True
):
    """
    📈 Plot training metrics over epochs/batches
    
    FOR NON-PROGRAMMERS:
    This function creates graphs showing how our AI model is learning over time.
    It's like creating a progress report with charts.
    
    What it shows:
    - Loss: How many "mistakes" the AI is making (lower = better)
    - Learning Rate: How fast the AI is trying to learn
    - Validation: How well the AI performs on new, unseen data

    Args:
        log_data: Either a file path to training logs, or data directly
        metrics: List of things to measure (like 'loss', 'lr', 'accuracy')
        title: Title for the graph
        save_path: Where to save the graph image (optional)
        show_validation: Whether to show validation data (recommended: True)
    """
    
    # Step 1: Load the training data from various sources
    logger.info(f"📊 Creating training metrics visualization: {title}")
    
    if isinstance(log_data, str):
        # If it's a file path, load the data
        log_data = log_data.strip()
        logger.info(f"📂 Loading data from file: {log_data}")
        
        if log_data.endswith(".json"):
            data = pd.read_json(log_data)
        elif log_data.endswith(".csv"):
            data = pd.read_csv(log_data)
        elif log_data.endswith(".pkl") or log_data.endswith(".pickle"):
            with open(log_data, "rb") as f:
                data = pd.DataFrame(pickle.load(f))
        elif log_data.endswith(".txt") or log_data.endswith(".log"):
            # Parse training_output.txt format
            parsed_data = parse_training_log(log_data)
            data = pd.DataFrame(parsed_data)
        else:
            raise ValueError(f"❌ Unsupported file format: {log_data}")
    else:
        # If it's already data, convert to DataFrame
        data = pd.DataFrame(log_data)

    # Step 2: Check if we have all the data we need
    available_cols = set(data.columns)
    logger.info(f"📋 Available data columns: {available_cols}")
    
    # Create epoch column if it doesn't exist
    if "epoch" not in data.columns and "batch" in data.columns:
        # Estimate epochs from batch numbers (assuming batches restart each epoch)
        data["epoch"] = data["batch"] // data["batch"].nunique() + 1
    
    # Step 3: Create beautiful subplots (one graph per metric)
    num_metrics = len(metrics)
    fig = make_subplots(
        rows=num_metrics,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"📈 {metric.replace('_', ' ').title()}" for metric in metrics],
    )

    # Step 4: Plot each metric with enhanced styling
    colors = px.colors.qualitative.Set3  # Beautiful color palette
    
    for i, metric in enumerate(metrics, start=1):
        if metric not in data.columns:
            logger.warning(f"⚠️ Metric '{metric}' not found in data, skipping...")
            continue
            
        # Create x-axis (time progression)
        if "epoch" in data.columns and "batch" in data.columns:
            # More precise x-axis: epoch + batch fraction
            max_batch = data.groupby("epoch")["batch"].max().max() or 1
            x_vals = data["epoch"] + (data["batch"] % max_batch) / max_batch
            x_title = "Training Progress (Epochs)"
        elif "epoch" in data.columns:
            x_vals = data["epoch"]
            x_title = "Epoch"
        else:
            x_vals = data.index
            x_title = "Training Step"
        
        # Add training metric line
        fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=data[metric], 
                mode="lines+markers",
                name=f"Training {metric}",
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f"<b>{metric}</b><br>Value: %{{y:.6f}}<br>Step: %{{x:.2f}}<extra></extra>"
            ),
            row=i,
            col=1,
        )
        
        # Add validation metric if available and requested
        val_metric = f"val_{metric}"
        if show_validation and val_metric in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=data[val_metric],
                    mode="lines+markers",
                    name=f"Validation {metric}",
                    line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
                    marker=dict(size=4, symbol="diamond"),
                    hovertemplate=f"<b>Validation {metric}</b><br>Value: %{{y:.6f}}<br>Step: %{{x:.2f}}<extra></extra>"
                ),
                row=i,
                col=1,
            )

    # Step 5: Make the plot beautiful and informative
    fig.update_layout(
        height=300 * num_metrics,  # Height scales with number of metrics
        title={
            "text": f"🎯 {title}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20}
        },
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text=x_title, row=num_metrics, col=1)
    
    # Add helpful annotations
    if len(data) > 0:
        final_values = {metric: data[metric].iloc[-1] for metric in metrics if metric in data.columns}
        annotation_text = " | ".join([f"{k}: {v:.4f}" for k, v in final_values.items()])
        
        fig.add_annotation(
            text=f"📊 Final Values: {annotation_text}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

    # Step 6: Save the plot if requested
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path.replace('.png', '.html'))  # Interactive version
        fig.write_image(save_path)  # Static image
        logger.info(f"✅ Saved training metrics plot to {save_path}")

    # Step 7: Display the plot
    fig.show()
    return fig


def parse_training_log(log_path: str) -> List[Dict]:
    """
    🔍 Parse training_output.txt format generated by our enhanced trainer
    
    FOR NON-PROGRAMMERS:
    This function reads the training log file and extracts important numbers
    like loss values, learning rates, etc. so we can make graphs from them.
    
    It's like reading a report card and extracting the grades to make a chart.
    
    Args:
        log_path: Path to the training_output.txt file
        
    Returns:
        List of dictionaries containing extracted training metrics
    """
    logger.info(f"📖 Parsing enhanced training log: {log_path}")
    
    records = []
    current_epoch = 0
    batch_in_epoch = 0
    
    # Patterns to match different log formats
    patterns = {
        'epoch_start': re.compile(r"EPOCH (\d+)"),
        'batch_info': re.compile(r"Batch\s+(\d+)/(\d+).*Loss:\s*([0-9.]+).*LR:\s*([0-9.eE+-]+)"),
        'epoch_summary': re.compile(r"Average Loss:\s*([0-9.]+)"),
        'validation': re.compile(r"Validation Loss:\s*([0-9.]+)")
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Match epoch start
                epoch_match = patterns['epoch_start'].search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    batch_in_epoch = 0
                    continue
                
                # Match batch information
                batch_match = patterns['batch_info'].search(line)
                if batch_match:
                    batch_idx = int(batch_match.group(1))
                    total_batches = int(batch_match.group(2))
                    loss = float(batch_match.group(3))
                    lr = float(batch_match.group(4))
                    
                    records.append({
                        'epoch': current_epoch,
                        'batch': batch_idx,
                        'total_batches': total_batches,
                        'loss': loss,
                        'lr': lr,
                        'progress': (batch_idx / total_batches) * 100
                    })
                    
                # Match validation loss
                val_match = patterns['validation'].search(line)
                if val_match and records:
                    records[-1]['val_loss'] = float(val_match.group(1))
                    
    except FileNotFoundError:
        logger.error(f"❌ Training log file not found: {log_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error parsing training log: {e}")
        raise
    
    logger.info(f"✅ Successfully parsed {len(records)} training records")
    return records


def plot_loss_comparison(train_losses: List[float], val_losses: Optional[List[float]] = None, 
                        save_path: str = "loss_comparison.png") -> go.Figure:
    """
    📉 Create a detailed loss comparison plot
    
    FOR NON-PROGRAMMERS:
    This creates a graph showing how the AI's "mistake rate" (loss) changes over time.
    Lower loss = fewer mistakes = better AI performance.
    
    Args:
        train_losses: List of training loss values
        val_losses: List of validation loss values (optional)
        save_path: Where to save the plot
    """
    logger.info("📉 Creating detailed loss comparison plot")
    
    fig = go.Figure()
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Add training loss
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate="<b>Training Loss</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>"
    ))
    
    # Add validation loss if provided
    if val_losses and len(val_losses) == len(train_losses):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate="<b>Validation Loss</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>"
        ))
        
        # Add gap between training and validation (overfitting indicator)
        gap = [val - train for val, train in zip(val_losses, train_losses)]
        fig.add_trace(go.Scatter(
            x=epochs,
            y=gap,
            mode='lines',
            name='Overfitting Gap',
            line=dict(color='red', width=2, dash='dot'),
            yaxis='y2',
            hovertemplate="<b>Overfitting Gap</b><br>Epoch: %{x}<br>Gap: %{y:.6f}<extra></extra>"
        ))
    
    # Styling
    fig.update_layout(
        title={
            'text': '🎯 Training vs Validation Loss',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Epoch',
        yaxis_title='Loss (Lower = Better)',
        yaxis2=dict(
            title='Overfitting Gap',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    # Add annotations for best performance
    if train_losses:
        best_epoch = train_losses.index(min(train_losses)) + 1
        best_loss = min(train_losses)
        
        fig.add_annotation(
            x=best_epoch,
            y=best_loss,
            text=f"🏆 Best: {best_loss:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            bgcolor="lightgreen"
        )
    
    # Save plot
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path)
        logger.info(f"✅ Saved loss comparison plot to {save_path}")
    
    fig.show()
    return fig


def plot_character_analysis(preprocessed_data: Dict, save_path: str = "character_analysis.png") -> Optional[go.Figure]:
    """
    👥 Create character dialogue distribution analysis
    
    FOR NON-PROGRAMMERS:
    This creates charts showing which Game of Thrones characters appear most often
    in our training data. This helps us understand what our AI will be good at generating.
    
    Args:
        preprocessed_data: The processed training data
        save_path: Where to save the plot
    """
    logger.info("👥 Creating character analysis visualization")
    
    # Extract character information
    if 'metadata' in preprocessed_data and 'character_vocab' in preprocessed_data['metadata']:
        characters = list(preprocessed_data['metadata']['character_vocab'].keys())
    else:
        logger.warning("⚠️ Character information not found in preprocessed data")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Character Frequency', 'Dialogue Length Distribution', 
                       'Character Interaction Network', 'Training Data Coverage'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Character frequency bar chart
    char_counts = {char: characters.count(char) for char in set(characters)}
    chars_sorted = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    
    fig.add_trace(go.Bar(
        x=[char for char, _ in chars_sorted[:15]],  # Top 15 characters
        y=[count for _, count in chars_sorted[:15]],
        name="Character Frequency",
        marker_color=px.colors.qualitative.Set3
    ), row=1, col=1)
    
    # Add more analysis traces...
    # (Additional plots would be added here based on available data)
    
    fig.update_layout(
        height=800,
        title_text="📊 Character Analysis Dashboard",
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path)
        logger.info(f"✅ Saved character analysis to {save_path}")
    
    fig.show()
    return fig


def create_training_dashboard(log_file: str = "training_output.txt", 
                            preprocessed_file: str = "preprocess_modern.pkl") -> None:
    """
    🎛️ Create a comprehensive training dashboard
    
    FOR NON-PROGRAMMERS:
    This creates a complete "dashboard" with multiple charts showing:
    - How well the AI is learning (loss over time)
    - Which characters it knows about
    - Training progress and statistics
    
    Think of it like a comprehensive report card with multiple graphs.
    
    Args:
        log_file: Path to training log file
        preprocessed_file: Path to preprocessed data file
    """
    logger.info("🎛️ Creating comprehensive training dashboard")
    
    try:
        # Plot 1: Training metrics
        if Path(log_file).exists():
            plot_training_metrics(
                log_file, 
                metrics=['loss', 'lr'], 
                title="Training Progress Over Time",
                save_path="dashboard_training_metrics.png"
            )
        
        # Plot 2: Character analysis
        if Path(preprocessed_file).exists():
            with open(preprocessed_file, 'rb') as f:
                preprocessed_data = pickle.load(f)
            plot_character_analysis(
                preprocessed_data, 
                save_path="dashboard_character_analysis.png"
            )
        
        # Plot 3: Model architecture visualization (if possible)
        # This would show the structure of our neural network
        
        logger.info("✅ Training dashboard created successfully!")
        logger.info("📁 Check the generated .html files for interactive versions")
        
    except Exception as e:
        logger.error(f"❌ Error creating training dashboard: {e}")
        raise


def parse_log_file(log_path: str) -> List[Dict]:
    """
    🔍 Parse a plain-text training log with basic format
    
    FOR NON-PROGRAMMERS:
    This is a simpler version that reads older log formats.
    It looks for lines like "Epoch: 1/10 Loss: 2.345 LR: 0.000300"
    
    Args:
        log_path: Path to the log file
        
    Returns:
        List of training records
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


def plot_vocabulary_analysis(vocab_to_int: Dict[str, int], save_path: str = "vocabulary_analysis.png") -> go.Figure:
    """
    📚 Analyze and visualize the vocabulary used in training
    
    FOR NON-PROGRAMMERS:
    This shows what words our AI knows and how often they appear.
    It's like looking at the AI's vocabulary book.
    
    Args:
        vocab_to_int: Dictionary mapping words to numbers
        save_path: Where to save the plot
    """
    logger.info("📚 Creating vocabulary analysis")
    
    # Get word frequencies (if available) or just show top words
    words = list(vocab_to_int.keys())
    word_indices = list(vocab_to_int.values())
    
    # Create frequency analysis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Top 50 Most Common Words', 'Vocabulary Size Distribution'],
        vertical_spacing=0.1
    )
    
    # Top words bar chart
    top_50_words = words[:50] if len(words) >= 50 else words
    top_50_indices = word_indices[:50] if len(word_indices) >= 50 else word_indices
    
    fig.add_trace(go.Bar(
        x=top_50_words,
        y=top_50_indices,
        name="Word Frequency",
        marker_color=px.colors.sequential.Blues_r
    ), row=1, col=1)
    
    # Vocabulary statistics
    fig.add_trace(go.Histogram(
        x=word_indices,
        nbinsx=50,
        name="Index Distribution",
        marker_color=px.colors.sequential.Viridis[0]
    ), row=2, col=1)
    
    fig.update_layout(
        height=700,
        title_text="📚 Vocabulary Analysis Dashboard",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        fig.write_image(save_path)
        logger.info(f"✅ Saved vocabulary analysis to {save_path}")
    
    fig.show()
    return fig


# 🧪 Example usage and testing
if __name__ == "__main__":
    """
    FOR NON-PROGRAMMERS:
    This section runs when you execute this file directly.
    It's like a test to make sure everything works correctly.
    """
    
    logger.info("🧪 Running plotting module tests...")
    
    try:
        # Test 1: Create sample training data
        logger.info("📊 Test 1: Creating sample training metrics plot")
        sample_data = [
            {"epoch": 1, "batch": 0, "loss": 4.5, "lr": 0.001},
            {"epoch": 1, "batch": 1, "loss": 4.2, "lr": 0.001},
            {"epoch": 2, "batch": 0, "loss": 3.8, "lr": 0.0009},
            {"epoch": 2, "batch": 1, "loss": 3.5, "lr": 0.0009},
        ]
        
        plot_training_metrics(
            sample_data,
            metrics=["loss", "lr"],
            title="🧪 Test Training Progress",
            save_path="test_metrics.png"
        )
        logger.info("✅ Test 1 passed!")
        
        # Test 2: Loss comparison
        logger.info("📉 Test 2: Creating loss comparison plot")
        train_losses = [4.5, 3.8, 3.2, 2.9, 2.6, 2.4, 2.2, 2.0]
        val_losses = [4.8, 4.1, 3.5, 3.2, 2.9, 2.8, 2.7, 2.5]
        
        plot_loss_comparison(train_losses, val_losses, "test_loss_comparison.png")
        logger.info("✅ Test 2 passed!")
        
        # Test 3: Try parsing a log file (if it exists)
        if Path("training_output.txt").exists():
            logger.info("📖 Test 3: Parsing real training log")
            create_training_dashboard()
            logger.info("✅ Test 3 passed!")
        else:
            logger.info("⚠️ Test 3 skipped: training_output.txt not found")
        
        logger.info("🎉 All plotting tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.error("💡 This is normal if you haven't run training yet")


# 💡 Helper function for easy dashboard creation
def quick_dashboard():
    """
    🚀 One-click function to create all visualizations
    
    FOR NON-PROGRAMMERS:
    Just call this function to create all the charts and graphs automatically!
    """
    logger.info("🚀 Creating quick training dashboard...")
    create_training_dashboard()
    logger.info("✅ Quick dashboard complete! Check the generated files.")


# 📖 Documentation for non-programmers
USAGE_GUIDE = """
🎯 HOW TO USE THIS PLOTTING MODULE

FOR NON-PROGRAMMERS:

1. BASIC USAGE:
   - Run: python modern_plot.py
   - This creates test graphs to make sure everything works

2. AFTER TRAINING:
   - The training will create 'training_output.txt'
   - Run: quick_dashboard() to create all visualizations

3. UNDERSTANDING THE GRAPHS:
   
   📈 TRAINING METRICS:
   - Loss: How many mistakes the AI makes (lower = better)
   - Learning Rate: How fast the AI learns (changes over time)
   - Validation: How well AI works on new data
   
   📊 CHARACTER ANALYSIS:
   - Shows which characters appear most in training data
   - Helps understand what the AI will be good at generating
   
   📚 VOCABULARY ANALYSIS:
   - Shows what words the AI knows
   - Bigger vocabulary = more varied output

4. INTERPRETING RESULTS:
   - Decreasing loss = AI is learning well ✅
   - Gap between training/validation = possible overfitting ⚠️
   - Flat loss = AI stopped improving (might need more training)

5. FILES GENERATED:
   - .png files: Static images for reports
   - .html files: Interactive graphs you can zoom/explore
"""

# Print usage guide when imported
logger.info(USAGE_GUIDE)
