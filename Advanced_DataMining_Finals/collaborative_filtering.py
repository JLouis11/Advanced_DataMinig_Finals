# ============================================================
#  MOVIE RECOMMENDATION SYSTEM - Advanced Data Mining Project
#  Topic: Collaborative Filtering for Movie Recommendations
#  Models: SVD (Matrix Factorization) vs KNN Collaborative Filter
#  Evaluation: RMSE, MAE, Precision@K, Recall@K
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# SECTION 1: LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  MOVIE RECOMMENDATION SYSTEM")
print("  Collaborative Filtering - Advanced Data Mining")
print("=" * 60)

print("\n[1] LOADING DATASET...")
df = pd.read_csv("ratings_raw.csv")
print(f"    Rows loaded    : {len(df)}")
print(f"    Columns        : {list(df.columns)}")
print(f"\n--- FIRST 5 ROWS (RAW) ---")
print(df.head())

# ─────────────────────────────────────────────
# SECTION 2: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[2] INITIAL DATA ANALYSIS (Before Cleaning)...")
print(f"\n    Shape          : {df.shape}")
print(f"\n--- MISSING VALUES (RAW) ---")
print(df.isnull().sum())
print(f"\n--- DATA TYPES ---")
print(df.dtypes)
print(f"\n--- RATING VALUE COUNTS (RAW - shows invalid values) ---")
print(df['rating'].value_counts(dropna=False).head(20))

n_before    = len(df)
n_dupes_raw = df.duplicated().sum()
print(f"\n    Duplicate rows : {n_dupes_raw}")
print(f"    Unique users   : {df['user_id'].nunique()}")
print(f"    Unique movies  : {df['movie_id'].nunique()}")

# ─────────────────────────────────────────────
# SECTION 3: DATA PREPROCESSING & CLEANING
# ─────────────────────────────────────────────
print("\n[3] DATA PREPROCESSING & CLEANING...")

# Step 3.1 - Fix rating column
print("    Step 3.1 - Fixing rating column (has NR, empty, 0, 6, 7, nulls)...")
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
invalid_ratings = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
print(f"             Ratings out of [1-5] range : {invalid_ratings}")
df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = np.nan
df.dropna(subset=['rating'], inplace=True)
print(f"             Rows after fixing ratings  : {len(df)}")

# Step 3.2 - Remove duplicate rows
print("    Step 3.2 - Removing duplicate rows...")
before_dedup = len(df)
df.drop_duplicates(inplace=True)
print(f"             Duplicates removed         : {before_dedup - len(df)}")
print(f"             Rows after deduplication   : {len(df)}")

# Step 3.3 - Fix year (valid: 1888-2024)
print("    Step 3.3 - Fixing year column (has -1, 0, 1800, 3000)...")
df['year'] = pd.to_numeric(df['year'], errors='coerce')
bad_year = ((df['year'] < 1888) | (df['year'] > 2024)).sum()
print(f"             Invalid years found        : {bad_year}")
df = df[(df['year'] >= 1888) & (df['year'] <= 2024)]
print(f"             Rows after fixing year     : {len(df)}")

# Step 3.4 - Strip whitespace from user_id
print("    Step 3.4 - Stripping whitespace from user_id...")
df['user_id'] = df['user_id'].str.strip()

# Step 3.5 - Strip whitespace from director
print("    Step 3.5 - Stripping whitespace from director column...")
df['director'] = df['director'].str.strip()

# Step 3.6 - Fix age (valid: 5-100)
print("    Step 3.6 - Fixing age column (has -5, 0, 150, 999)...")
df['age'] = pd.to_numeric(df['age'], errors='coerce')
bad_age = ((df['age'] < 5) | (df['age'] > 100)).sum()
print(f"             Invalid ages found         : {bad_age}")
df.loc[(df['age'] < 5) | (df['age'] > 100), 'age'] = np.nan
df['age'].fillna(df['age'].median(), inplace=True)

# Step 3.7 - Standardize genre casing
print("    Step 3.7 - Standardizing genre casing (ACTION, action -> Action)...")
df['genre'] = df['genre'].str.strip().str.title()

# Step 3.8 - Fix timestamp
print("    Step 3.8 - Fixing timestamp column (has unknown, 99/99/9999)...")
def is_valid_date(val):
    try:
        pd.to_datetime(val, format='%Y-%m-%d')
        return True
    except:
        return False
invalid_ts = (~df['timestamp'].apply(is_valid_date)).sum()
print(f"             Invalid timestamps found   : {invalid_ts}")
df.loc[~df['timestamp'].apply(is_valid_date), 'timestamp'] = np.nan
df['timestamp'].fillna('2024-01-01', inplace=True)

# Step 3.9 - Fill remaining missing values
print("    Step 3.9 - Filling remaining missing values...")
df['genre'].fillna('Unknown',      inplace=True)
df['director'].fillna('Unknown',   inplace=True)
df['language'].fillna('Unknown',   inplace=True)
df['gender'].fillna('Unknown',     inplace=True)
df['occupation'].fillna('Unknown', inplace=True)

print(f"\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())
print(f"\n    Raw dataset rows              : {n_before}")
print(f"    Final clean dataset rows      : {len(df)}")
print(f"    Records removed during clean  : {n_before - len(df)}")

df.to_csv('ratings_cleaned.csv', index=False)
print("    Cleaned dataset saved as      : ratings_cleaned.csv")

# ─────────────────────────────────────────────
# SECTION 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[4] FEATURE ENGINEERING...")

def age_group(age):
    if age < 25:   return 'Youth'
    elif age < 35: return 'Young Adult'
    elif age < 50: return 'Adult'
    else:          return 'Senior'

df['age_group'] = df['age'].apply(age_group)
print("    Created age_group from age (Youth / Young Adult / Adult / Senior)")

df['movie_age'] = 2024 - df['year']
print("    Created movie_age = 2024 - year")

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['rating_month'] = df['timestamp'].dt.month.fillna(1).astype(int)
print("    Created rating_month from timestamp")

user_avg = df.groupby('user_id')['rating'].mean().rename('user_avg_rating')
df = df.merge(user_avg, on='user_id', how='left')
print("    Created user_avg_rating = average rating per user")

movie_avg = df.groupby('movie_id')['rating'].mean().rename('movie_avg_rating')
df = df.merge(movie_avg, on='movie_id', how='left')
print("    Created movie_avg_rating = average rating per movie")

user_count = df.groupby('user_id')['rating'].count().rename('user_rating_count')
df = df.merge(user_count, on='user_id', how='left')
print("    Created user_rating_count = number of ratings per user")

le = LabelEncoder()
for col in ['genre', 'language', 'age_rating', 'director', 'gender', 'age_group', 'occupation']:
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
print("    Encoded: genre, language, age_rating, director, gender, age_group, occupation")

print(f"\n    Final columns: {list(df.columns)}")

df.to_csv('ratings_featured.csv', index=False)
print("    Feature engineered dataset saved as: ratings_featured.csv")
