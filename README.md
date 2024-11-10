# Instagram Analysis Project

This project focuses on analyzing Instagram data to uncover trends, predict follower growth, and estimate post impressions. By leveraging machine learning models and embeddings, we aim to provide actionable insights to improve audience engagement and optimize content strategy.

## Project Overview

### 1. Trend Analysis
- **Goal**: Identify content trends and engagement patterns across posts
- **Approach**: Analyzed historical Instagram data to track engagement metrics (likes, comments, shares)
- **Result**: Improved understanding of audience preferences.

### 2. Follower Growth Prediction
- **Goal**: Predict the number of followers an account will gain based on impressions and other key features
- **Approach**: Implemented a Random Forest algorithm to model follower growth using impressions and feature data
- **Result**: Achieved a score of 0.83 for follower gain predictions.

### 3. Impression Prediction
- **Goal**: Accurately predict future impressions for posts based on captions and hashtags
- **Approach**: Used Hugging Face embedding models to tokenize and embed captions and hashtags, feeding these embeddings into a neural network to predict impression counts
- **Result**: Increased impression prediction

# Data

- from [Kaggle](https://www.kaggle.com/datasets/amirmotefaker/instagram-data)

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas, numpy, and other standard data science libraries

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/instagram-analysis.git
   ```
2. Preprocess the data and generate embeddings
3. Train the Random Forest model for follower growth prediction
4. Train the neural network for impression prediction

## Future Work

- **Imporve models**: Improve model's prediction score.
- **Experiment with Models**: Test advanced transformer-based architectures for text analysis
- **Deploy Models**: Develop an interactive dashboard for real-time prediction and trend analysis
