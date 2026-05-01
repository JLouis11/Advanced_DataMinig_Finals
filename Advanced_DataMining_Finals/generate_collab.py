import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# Target: ~2966 clean rows, so raw needs ~3050
# Use fewer users/movies but more controlled count
n_users  = 300
n_movies = 150

genres     = ['Action','Comedy','Drama','Sci-Fi','Romance','Thriller','Horror','Animation','Documentary','Fantasy']
directors  = ['Christopher Nolan','Greta Gerwig','Steven Spielberg','Bong Joon-ho','Quentin Tarantino',
              'Sofia Coppola','Martin Scorsese','Ava DuVernay','Denis Villeneuve','Jordan Peele']
languages  = ['English','Spanish','French','Korean','Japanese','Hindi','Portuguese','Italian']
age_ratings= ['G','PG','PG-13','R','NC-17']

user_ids  = [f"U{str(i).zfill(4)}" for i in range(1, n_users+1)]
movie_ids = [f"M{str(i).zfill(4)}" for i in range(1, n_movies+1)]

adj  = ['Dark','Lost','Eternal','Wild','Silent','Broken','Last','Hidden','Neon','Silver',
        'Burning','Frozen','Rising','Fallen','Golden','Midnight','Sacred','Shattered','Twisted','Bright']
noun = ['Storm','Heart','Dream','City','Shadow','Path','Journey','Kingdom','Legacy','Horizon',
        'Echo','Flame','Ocean','Mountain','River','Forest','Sky','Star','Moon','Sun']

movie_meta = {}
for i, mid in enumerate(movie_ids):
    movie_meta[mid] = {
        'title'      : f"The {adj[i % len(adj)]} {noun[i % len(noun)]}",
        'genre'      : genres[i % len(genres)],
        'director'   : directors[i % len(directors)],
        'language'   : languages[i % len(languages)],
        'age_rating' : age_ratings[i % len(age_ratings)],
        'year'       : random.randint(1990, 2023),
    }

# User genre preferences
user_pref = {}
for uid in user_ids:
    user_pref[uid] = {g: np.random.uniform(0.5, 1.5) for g in genres}

# Generate ratings — target ~2900 clean rows before injecting mess
records = []
for uid in user_ids:
    n_rated = random.randint(8, 12)   # 300 users × ~10 ratings = ~3000
    rated   = random.sample(movie_ids, n_rated)
    for mid in rated:
        genre  = movie_meta[mid]['genre']
        pref   = user_pref[uid][genre]
        base   = np.random.normal(3.0 * pref, 0.8)
        rating = round(np.clip(base, 1.0, 5.0) * 2) / 2
        records.append({
            'user_id'    : uid,
            'movie_id'   : mid,
            'title'      : movie_meta[mid]['title'],
            'genre'      : movie_meta[mid]['genre'],
            'director'   : movie_meta[mid]['director'],
            'language'   : movie_meta[mid]['language'],
            'age_rating' : movie_meta[mid]['age_rating'],
            'year'       : movie_meta[mid]['year'],
            'rating'     : rating,
            'age'        : random.randint(18, 65),
            'gender'     : random.choice(['M','F','Other']),
            'occupation' : random.choice(['Student','Engineer','Artist','Doctor',
                                          'Teacher','Manager','Retired','Other']),
            'timestamp'  : f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        })

df = pd.DataFrame(records)
print(f"Clean base rows: {len(df)}")

# ── Inject exactly enough mess to get raw=3050, clean≈2966 ──

# 1. Missing values (~8%) in several columns
for col in ['genre','director','language','gender','occupation']:
    mask = np.random.random(len(df)) < 0.08
    df.loc[mask, col] = np.nan

# 2. Duplicate rows — add 50
dup_idx = np.random.choice(len(df), 50, replace=False)
dupes   = df.iloc[dup_idx].copy()
df      = pd.concat([df, dupes], ignore_index=True)

# 3. Invalid ratings (will be dropped) — add ~20 bad rows
df = df.astype(object)
bad_r = np.random.choice(len(df), 20, replace=False)
for idx in bad_r[:10]:
    df.loc[idx, 'rating'] = random.choice([0, 6, 7, 'NR', ''])
for idx in bad_r[10:]:
    df.loc[idx, 'rating'] = np.nan

# 4. Age errors — replace (not drop, just fix later)
bad_a = np.random.choice(len(df), 14, replace=False)
for idx in bad_a:
    df.loc[idx, 'age'] = random.choice([-5, 0, 150, 999])

# 5. Inconsistent genre casing
bad_g = np.random.choice(len(df), 80, replace=False)
for idx in bad_g:
    if pd.notna(df.loc[idx,'genre']):
        df.loc[idx,'genre'] = df.loc[idx,'genre'].upper() if idx%2==0 else df.loc[idx,'genre'].lower()

# 6. Whitespace in user_id
ws = np.random.choice(len(df), 30, replace=False)
for idx in ws:
    df.loc[idx,'user_id'] = '  ' + str(df.loc[idx,'user_id']) + '  '

# 7. Bad timestamps
bad_t = np.random.choice(len(df), 16, replace=False)
for idx in bad_t:
    df.loc[idx,'timestamp'] = random.choice(['99/99/9999','unknown','N/A','2024-13-45'])

# 8. Negative year (entry errors)
bad_y = np.random.choice(len(df), 10, replace=False)
for idx in bad_y:
    df.loc[idx,'year'] = random.choice([-1, 0, 1800, 3000])

# 9. Whitespace in director
wd = np.random.choice(len(df), 25, replace=False)
for idx in wd:
    if pd.notna(df.loc[idx,'director']):
        df.loc[idx,'director'] = '  ' + str(df.loc[idx,'director']) + '  '

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Trim or pad to exactly 3050
if len(df) > 3050:
    df = df.iloc[:3050]
elif len(df) < 3050:
    shortage = 3050 - len(df)
    extra = df.sample(shortage, replace=True, random_state=99)
    df = pd.concat([df, extra], ignore_index=True)

df = df.reset_index(drop=True)
df.to_csv('/home/claude/ratings_raw.csv', index=False)
print(f"Final raw rows : {len(df)}")
print(f"Missing values:\n{df.isnull().sum()}")
