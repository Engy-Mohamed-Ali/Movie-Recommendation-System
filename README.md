# 🎬 Hybrid Movie Recommendation System

## 📌 Project Overview

This project implements a **Hybrid Movie Recommendation System** using the MovieLens dataset.

The system combines:

- **Content-Based Filtering**
- **Collaborative Filtering (SVD)**

to generate personalized movie recommendations based on:

- Movie similarity
- User preferences and rating history

The project also includes:

- Model evaluation
- Performance comparison
- Interactive Streamlit user interface

---

# 📂 Dataset

Dataset Used:

- MovieLens Dataset

Files:

- `movies.csv`
- `ratings.csv`

---

# ⚙️ Technologies & Libraries

## Python Libraries

- pandas
- numpy
- scikit-learn
- scikit-surprise
- streamlit

---

# 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Handling missing values using `dropna()`
- Removing duplicate records
- Merging movie and rating datasets
- Cleaning movie genres
- Title preprocessing using regular expressions

---

# 🎯 Recommendation Techniques

## 1️⃣ Content-Based Filtering

Content-based filtering recommends movies similar to a selected movie based on movie metadata.

### Techniques Used

- TF-IDF Vectorization
- Cosine Similarity

### Features Used

- Movie titles
- Movie genres

### Workflow

1. Combine movie title and genres
2. Convert text into TF-IDF vectors
3. Compute cosine similarity between movies
4. Recommend most similar movies

---

## 2️⃣ Collaborative Filtering

Collaborative filtering recommends movies using user-item interaction patterns.

### Algorithm Used

- Singular Value Decomposition (SVD)

### Workflow

1. Build user-movie rating matrix
2. Train SVD model
3. Predict ratings for unseen movies
4. Recommend highest predicted ratings

---

## 3️⃣ Hybrid Recommendation System

The hybrid system combines both approaches using weighted averaging.

### Hybrid Formula

```python
Hybrid Score =
(Content Weight × Content Score)
+
(Collaborative Weight × Collaborative Score)
```

### Normalization

Collaborative scores were normalized to the range [0,1] before combining with cosine similarity scores.

---

# 📊 Evaluation Metrics

The following evaluation metrics were used:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision
- Recall
- F1-Score

---

# 📈 Model Performance Comparison

| Model | RMSE | MAE | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Content-Based | N/A | N/A | 0.0380 | 0.0031 | 0.0057 |
| Collaborative Filtering | 0.8798 | 0.6764 | 0.7380 | 0.5114 | 0.6042 |
| Hybrid Model | 1.1348 | 0.8906 | 0.0720 | 0.0057 | 0.0105 |

---

# 📌 Evaluation Insights

## Content-Based Filtering

### Advantages

- Works without requiring user ratings
- Recommends movies with similar genres and titles
- Simple and interpretable

### Limitations

- Limited personalization
- Suffers from over-specialization
- Lower Precision and Recall scores

---

## Collaborative Filtering

### Advantages

- Best overall performance
- Strong rating prediction capability
- Captures hidden user preferences using latent features

### Results

- Lowest RMSE and MAE
- Highest Precision, Recall, and F1-score

### Limitations

- Cold-start problem for new users or movies
- Requires sufficient rating data

---

## Hybrid Recommendation System

### Advantages

- Combines strengths of both models
- Improves recommendation diversity
- Reduces dependence on only one recommendation strategy

### Observations

The hybrid model achieved lower performance compared to collaborative filtering in RMSE and MAE because:

- Cosine similarity scores are not true rating predictions
- Combining similarity scores with predicted ratings introduces additional noise
- The selected weights may not be fully optimized

Despite this, the hybrid system still improves recommendation diversity and personalization.

---

# 🎛️ Weight Tuning

Different weight combinations were tested to balance:

- Content-Based contribution
- Collaborative contribution

Example:

```python
Final Score =
0.5 × Content Score
+
0.5 × Collaborative Score
```

Weights can be adjusted interactively in the Streamlit UI.

---

# 🖥️ Streamlit User Interface

The project includes an interactive Streamlit application where users can:

- Enter User ID
- Select a movie
- Adjust hybrid weights
- Choose number of recommendations
- Receive personalized movie recommendations

---

# 🚀 Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```bash
├── Movie_Recommendation_System.ipynb
├── app.py
├── movies.csv
├── ratings.csv
├── requirements.txt
├── README.md
```

---

# ✅ Conclusion

This project successfully implemented:

- Content-Based Filtering
- Collaborative Filtering using SVD
- Hybrid Recommendation System
- Evaluation Metrics
- Streamlit User Interface

Among all models, Collaborative Filtering achieved the best prediction accuracy and recommendation quality.

The Hybrid model demonstrated how combining multiple recommendation strategies can improve recommendation diversity and personalization.

---

# 👨‍💻 Developed Using

- Python
- Streamlit
- Scikit-learn
- Surprise Library
- TF-IDF
- Cosine Similarity
- SVD