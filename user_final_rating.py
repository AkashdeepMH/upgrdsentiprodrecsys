import pandas as pd
import pickle

# Load dataset
df = pd.read_csv('sample30.csv')

# Optional: clean NaNs
df.dropna(subset=['reviews_username', 'name'], inplace=True)

# Fill NaN ratings with 0 if needed or assign default
if 'reviews_rating' not in df.columns:
    df['reviews_rating'] = 1  # or any logic you prefer

# Create user-product matrix (e.g., using mean rating)
user_product_matrix = df.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating',
    aggfunc='mean'
).fillna(0)

# Save as pickle
with open('pickle_file/user_final_rating.pkl', 'wb') as f:
    pickle.dump(user_product_matrix, f)

print("user_final_rating.pkl saved!")
