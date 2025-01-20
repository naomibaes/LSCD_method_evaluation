# Authors: Naomi Baes and Chat GPT

import pandas as pd
import os
import matplotlib.pyplot as plt

# Define file paths
input_dir = os.path.abspath("input/baselines")  # Directory for baseline files
output_folder = os.path.abspath("input/baselines/output")  # Output directory

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Target terms and colors for plotting
targets = ["abuse", "anxiety", "depression", "mental_health", "mental_illness", "trauma"]
colors = {
    'abuse': '#8B0000',
    'anxiety': '#FF6347',
    'depression': '#4B0082',
    'mental_health': '#008080',
    'mental_illness': '#800080',
    'trauma': '#DC143C',
}

# Initialize lists for year and epoch counts
year_count_summary = []
epoch_count_summary = []

# Process each target
for target in targets:
    target_files = [f for f in os.listdir(input_dir) if f.startswith(target) and f.endswith(".csv")]
    for file_name in target_files:
        file_path = os.path.join(input_dir, file_name)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            year_counts = data.groupby("year").size().reset_index(name="num_sentences")
            year_counts['target'] = target
            year_count_summary.extend(year_counts.to_dict(orient="records"))
            data['epoch_start'] = (data['year'] // 5) * 5
            data['epoch_end'] = data['epoch_start'] + 4
            data['epoch'] = data['epoch_start'].astype(str) + "-" + data['epoch_end'].astype(str)
            epoch_counts = data.groupby("epoch").size().reset_index(name="num_sentences")
            epoch_counts['target'] = target
            epoch_count_summary.extend(epoch_counts.to_dict(orient="records"))

# Save year and epoch summaries
year_count_file = os.path.join(output_folder, "year_count_lines.csv")
pd.DataFrame(year_count_summary).to_csv(year_count_file, index=False)
print(f"Year count summary saved to {year_count_file}.")

epoch_count_file = os.path.join(output_folder, "epoch_count_lines.csv")
pd.DataFrame(epoch_count_summary).to_csv(epoch_count_file, index=False)
print(f"Epoch count summary saved to {epoch_count_file}.")

# Generate epoch-based plots
epoch_count_df = pd.DataFrame(epoch_count_summary)
if not epoch_count_df.empty:
    n_targets = len(targets)
    n_cols = 2
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 3), constrained_layout=True)
    axs = axs.flatten()

    for i, target in enumerate(targets):
        ax = axs[i]
        target_data = epoch_count_df[epoch_count_df["target"] == target]
        ax.bar(target_data["epoch"], target_data["num_sentences"], color=colors[target], edgecolor="black")
        ax.set_title(target.capitalize(), fontsize=20)
        ax.set_ylabel("Count", fontsize=18)
        ax.axhline(500, color="black", linestyle="--", linewidth=1, label="Low threshold (500)")
        if target_data["num_sentences"].max() >= 1500:
            ax.axhline(1500, color="darkgreen", linestyle="--", linewidth=1, label="High threshold (1500)")
        # Set x-tick labels only on the last row axes
        if i // n_cols == n_rows - 1:
            ax.set_xticklabels(target_data["epoch"], rotation=45, ha="right", fontsize=18)
        else:
            ax.set_xticklabels([])  # Hide x-tick labels on all but the last row
        ax.legend(loc='upper right', fontsize=14)
        ax.tick_params(axis='y', labelsize=18)  # Adjust '16' to your desired font size

    # Remove unused subplots if any
    for j in range(len(targets), len(axs)):
        fig.delaxes(axs[j])  # Remove unused subplots

    epoch_plot_file = os.path.join(output_folder, "epoch_counts_intensity.png")
    epoch_plot_file = os.path.join("../../", "figures", "plot_appendixB_intensity.png")
    plt.savefig(epoch_plot_file, dpi=300)
    plt.close()
    print(f"Epoch plot saved to {epoch_plot_file}.")
