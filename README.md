📺 Unlocking YouTube Channel Performance Secrets

This project analyzes YouTube channel performance using advanced analytics and Machine Learning. It explores revenue, engagement, audience data, and monetization metrics to uncover hidden insights and optimize channel growth.

📂 Dataset

The dataset provides in-depth YouTube analytics with:

Video Details: Duration, Publish Time, Day of Week, Days Since Publish

Revenue Metrics: Revenue per 1000 views, Estimated Revenue, Ad Impressions, AdSense, YouTube Premium

Engagement Metrics: Views, Likes, Dislikes, Shares, Comments, CTR, Avg. View Duration & %

Audience Data: Subscribers, Unique Viewers, Returning/New Viewers

Monetization Data: Monetized Playbacks, CPM, Premium Revenue, Transactions

📌 Dataset link: Google Drive Dataset

⚙️ Tools & Libraries

Python 🐍

Jupyter Notebook / VS Code

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn (RandomForestRegressor)

Joblib (for model saving)

🛠️ Workflow

Data Loading & Cleaning

Handle null values

Convert video duration to seconds

Format publish time

Exploratory Data Analysis (EDA)

Revenue distribution & scatter plots

Correlation heatmap

Top performing videos by revenue

Feature Engineering

Revenue per View

Engagement Rate (%)

Visualization

Revenue Distribution

Revenue vs Views

Feature Importance

Predictive Modeling

Model: Random Forest Regressor

Features: Views, Subscribers, Likes, Shares, Comments, Engagement Rate

Target: Estimated Revenue (USD)

Metrics: MSE, R²

Deployment

Export trained model as youtube_revenue_predictor.pkl

Insights & recommendations for channel optimization

📊 Results & Insights

Engagement rate and views strongly drive revenue.
🚀 How to Run

Clone the repository:

git clone https://github.com/AjitheswarAkkireddy/Youtube-Channel-Performance.git
cd Youtube-Channel-Performance


Install dependencies:

pip install -r requirements.txt


Run the notebook / script:

jupyter notebook youtube_analysis.ipynb


To load the trained model:

import joblib
model = joblib.load("youtube_revenue_predictor.pkl")

✨ Author

👤 Ajitheswar Akkireddy
📧 [ajitheswar1200@gmail.com]
🔗 GitHub Profile
Top-performing videos contribute disproportionately to revenue.

Random Forest model achieved reasonable prediction accuracy (low RMSE).
