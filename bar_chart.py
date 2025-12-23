import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your table
data = {
    'Model': ['ResNet', 'ViT', 'AE+ViT'],
    'Accuracy': [0.992, 0.994, 0.996],
    'Precision': [0.9785, 0.9815, 0.9919],
    'Sensitivity': [0.9787, 0.9852, 0.9905],
    'Specificity': [0.9931, 0.995, 0.997],
    'F1-Score': [0.9785, 0.9834, 0.9912],
    'G-Mean': [0.986, 0.9901, 0.9937]
}

df = pd.DataFrame(data)

# Convert decimal values to percentages
metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'G-Mean']
df[metric_cols] = df[metric_cols] * 100

# --- Plotting Setup ---
models = df['Model']
num_models = len(models)
num_metrics = len(metric_cols)
bar_width = 0.12
x = np.arange(num_models)

# Define the colors for the bars
colors = ["#0504AA", '#FFB81C', "#8B8282A6", "#000000", '#fba0e3', '#8b6c5c']

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the bars for each metric
for i, metric in enumerate(metric_cols):
    # Calculate the offset for each group of bars
    offset = (i - num_metrics / 2) * bar_width + bar_width / 2
    ax.bar(x + offset, df[metric], bar_width, label=metric, color=colors[i % len(colors)])

# Add labels and title
ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Models')
ax.set_title('Comparison Analysis')
ax.set_xticks(x)
ax.set_xticklabels(models)

# Set Y-axis limit for better visualization, similar to your reference
# It starts just below the lowest metric score and goes up to 100%
ax.set_ylim(min(df[metric_cols].min().min() - 0.5, 95), 100)

# Add a legend
ax.legend(loc='lower left', bbox_to_anchor=(1, 0))

# Add a grid for readability
ax.grid(axis='y', linestyle='--')

# Adjust layout and save the figure
plt.tight_layout(rect=(0, 0, 0.9, 1))
plt.savefig('comparison_analysis.png', dpi=300)
plt.show()