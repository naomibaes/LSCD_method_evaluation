import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load dataset
annotated_file = "output/annotate/validated_sentences_ANNOTATE.csv"  # Update with the correct file path
df = pd.read_csv(annotated_file)

# Ensure the 'validated' and 'ground_truth' columns are converted to numeric (1 for TRUE, 0 for FALSE)
df['validated'] = df['validated'].map({'TRUE': 1, 'FALSE': 0})  # Convert 'TRUE'/'FALSE' to 1/0
df['ground_truth'] = df['ground_truth'].map({'TRUE': 1, 'FALSE': 0})  # Convert 'TRUE'/'FALSE' to 1/0

# Handle NaN values in 'ground_truth' column by removing rows with NaN
df = df.dropna(subset=['ground_truth'])

# Check if there are any NaN values left in the 'ground_truth' column
print(df['ground_truth'].isna().sum(), "NaN values in 'ground_truth' column")

# Fine-grained threshold selection generator
def generate_fine_grained_thresholds(start, end, step=0.01):
    """Generates fine-grained thresholds between start and end with the given step."""
    return [round(x, 2) for x in np.arange(start, end + step, step)]

# List of model columns with corresponding thresholds
model_columns = {
    'mlm': {
        'columns': ['mlm_ratio_RoBERTa-large', 'mlm_ratio_DeBERTa-v3-large', 'mlm_ratio_BioBERT'],
        'thresholds': generate_fine_grained_thresholds(0.5, 0.7, step=0.01)  # Fine-grained thresholds
    },
    'cosine': {
        'columns': ['MiniLM-L12-v2', 'DistilRoBERTa-v1', 'Sentence-T5'],
        'thresholds': generate_fine_grained_thresholds(0.65, 0.91, step=0.01)  # Fine-grained thresholds
    }
}

# List to store metrics for each model and threshold
model_metrics = []

# Function to calculate Precision, Recall, and F1 scores
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# Generalized model evaluation function
def evaluate_models_with_cross_val(model_type, model_cols, thresholds):
    for model_col in model_cols:
        model_name = model_col
        best_f1 = 0
        best_precision = 0
        best_recall = 0
        best_threshold = None
        
        # Stratified K-Fold Cross Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for threshold in thresholds:
            all_f1 = []
            all_precision = []
            all_recall = []
            
            for train_idx, test_idx in cv.split(df, df['ground_truth']):
                df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]  # Ensure we're using the correct splits
                
                y_true = df_test['ground_truth']  # Ground truth labels
                y_pred = (df_test[model_col] >= threshold).astype(int)  # Predictions based on threshold
                
                precision, recall, f1 = calculate_metrics(y_true, y_pred)
                all_f1.append(f1)
                all_precision.append(precision)
                all_recall.append(recall)
            
            avg_f1 = np.mean(all_f1)
            avg_precision = np.mean(all_precision)
            avg_recall = np.mean(all_recall)
            
            # Update best metrics
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_precision = avg_precision
                best_recall = avg_recall
                best_threshold = threshold
        
        model_metrics.append({
            'model': model_name,
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall
        })

# Step 1: Iterate over all model types (MLM and Cosine)
for model_type, model_info in model_columns.items():
    evaluate_models_with_cross_val(model_type, model_info['columns'], model_info['thresholds'])

# Step 2: Convert metrics to DataFrame for easy visualization
model_metrics_df = pd.DataFrame(model_metrics)

# Sort models by best F1 score
model_metrics_df = model_metrics_df.sort_values(by='best_f1_score', ascending=False)

# Display the results
print("\nBest Model and Threshold based on F1 Score:")
print(model_metrics_df)

# Step 3: Precision-Recall Curves for the best models
plt.figure(figsize=(5, 3))

colors = plt.cm.get_cmap('tab10', len(model_metrics_df))  # Automatically cycle colors
for idx, row in model_metrics_df.iterrows():
    model_name = row['model']
    threshold = row['best_threshold']
    
    # For both MLM and cosine models, use best threshold
    y_pred = (df[model_name] >= threshold).astype(int)
    
    # Calculate Precision and Recall at different thresholds
    precision, recall, _ = precision_recall_curve(df['ground_truth'], df[model_name])
    
    # Plot Precision-Recall curve with corrected label formatting
    plt.plot(recall, precision, lw=2, color=colors(idx), label=f"{model_name}")

plt.xlabel('Recall', fontsize=14)  # Increase the font size for x-axis label
plt.ylabel('Precision', fontsize=14)  # Increase the font size for y-axis label
plt.legend(loc='lower right', fontsize=12)  # Adjust legend font size
plt.tight_layout()

# Save the plot
output_pr_curve_file  = "output/annotate/precision_recall_curve_models.png"
plt.savefig(output_pr_curve_file, dpi=300)
plt.show()

output_pr_curve_file

# Display the plot
plt.show()
