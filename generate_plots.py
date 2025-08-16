import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting plot generation...")

# --- Create Directories ---
# Create a 'static' directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
    print("Created 'static' directory.")

# Create an 'images' subdirectory inside 'static'
if not os.path.exists('static/images'):
    os.makedirs('static/images')
    print("Created 'static/images' directory.")

# --- Load Data ---
try:
    data = pd.read_csv('youtube_channel_real_performance_analytics.csv')
    # Basic cleaning for visualization
    for col in ['Estimated Revenue (USD)', 'Views']:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(subset=['Estimated Revenue (USD)', 'Views'], inplace=True)
    print("Dataset loaded and cleaned successfully.")
except FileNotFoundError:
    print("Error: youtube_channel_real_performance_analytics.csv not found.")
    exit()

# --- Plot 1: Revenue Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(data[data['Estimated Revenue (USD)'] < 100]['Estimated Revenue (USD)'], bins=50, kde=True, color='cyan')
plt.title('Distribution of Estimated Revenue (USD)', fontsize=16)
plt.xlabel('Estimated Revenue (USD)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/revenue_distribution.png')
print("Generated: revenue_distribution.png")

# --- Plot 2: Revenue vs. Views ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Views', y='Estimated Revenue (USD)', data=data, alpha=0.6, color='magenta')
plt.title('Revenue vs. Views', fontsize=16)
plt.xlabel('Views', fontsize=12)
plt.ylabel('Estimated Revenue (USD)', fontsize=12)
plt.xscale('log') # Use a log scale for better visualization of views
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig('static/images/revenue_vs_views.png')
print("Generated: revenue_vs_views.png")

# --- Plot 3: Correlation Heatmap ---
# Select a subset of key features for a more readable heatmap
corr_features = [
    'Estimated Revenue (USD)', 'Views', 'Likes', 'New Comments', 
    'Subscribers', 'Watch Time (hours)', 'Ad Impressions'
]
corr_matrix = data[corr_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
plt.title('Correlation Heatmap of Key Metrics', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png')
print("Generated: correlation_heatmap.png")

print("\nAll plots have been generated and saved in the 'static/images' folder.")
