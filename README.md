# Movie Rating Final Project

## Project Overview & Research Question

This project predicts whether a movie will achieve a "high rating" (IMDb score $\ge$ 7.0) based on features like budget, genre, runtime, and user votes. The goal is to identify which factors best predict audience reception and understand if financial success or creative choices drive higher ratings.

**Research Question:** What factors are most predictive of a movie achieving a high audience rating (IMDb score $\ge$ 7.0)?

## Dataset & Target Variable

We used a dataset of 7,668 movies released between 1980 and 2020. After handling missing values (primarily in the `budget` column), the working dataset contained 5,421 rows. 

**Features used:** `rating`, `genre`, `year`, `votes`, `director`, `star`, `country`, `budget`, `company`, `runtime`.

**Target Variable (`high_rating`):**
- **1**: IMDb score $\ge$ 7.0 (~30% of data)
- **0**: IMDb score < 7.0 (~70% of data)

## Key Findings from EDA

- **Audience Engagement:** High-rated movies averaged ~236K votes, compared to ~65K for lower-rated movies.
- **Runtime:** High-rated movies were generally longer (118 mins vs. 104 mins).
- **Budget:** Budget showed minimal difference between the two classes ($38M vs $35.2M).
- **Genre:** Genres like Biography, Drama, and Comedy showed distinct rating patterns.

## Machine Learning Models

We built three classification pipelines using Scikit-learn (handling missing values with median/most-frequent imputers and applying one-hot encoding). The dataset was split into training (4,336) and testing (1,085) sets.

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.8129** | **Best overall.** Highest accuracy and best balance of precision/recall, though it still missed some high-rated movies (Recall for class 1: 0.49). |
| **Decision Tree** (max depth 3) | 0.7714 | Easy to interpret, but struggled with recall for high-rated movies. Early splits confirmed votes and runtime as key features. |
| **Random Forest** (100 trees, max depth 10) | 0.7115 | Poor performance on the minority class. Due to class imbalance, it almost exclusively predicted the majority class (0 recall for class 1). |

## Feature Importance

Based on model analysis, audience engagement is a much stronger predictor than budget. 

| Top Features | Importance |
| :--- | :--- |
| `votes` | 0.1521 |
| `runtime` | 0.0945 |
| `genre: Biography` | 0.0358 |
| `genre: Comedy` | 0.0309 |

## Limitations & Future Improvements

**Limitations:**
- **Missing Data:** Dropping rows with missing budgets removed ~30% of the dataset.
- **Post-Release Features:** `votes` is only available after release, making this model less useful for pre-release predictions.
- **Subjectivity:** IMDb scores may not perfectly reflect general audience opinion due to review bombing or niche fanbases.
- **Missing Factors:** Critic reviews, marketing spend, and plot details are not included.

**Future Improvements:**
- Test advanced models like XGBoost or Neural Networks.
- Use techniques like SMOTE or class weighting to address the target class imbalance.
- Build a strictly "pre-release" model excluding post-release features like `votes`.

## Setup & Execution

**Technologies:** Python, Pandas, Matplotlib, Scikit-learn

**How to Run:**
1. Ensure `movies_dataset.csv` and `FinalProject.ipynb` are in the same directory.
2. Open and run `FinalProject.ipynb` sequentially from top to bottom.