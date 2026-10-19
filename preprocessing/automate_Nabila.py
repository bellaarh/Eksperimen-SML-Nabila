import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def run_preprocessing(input_path, output_path):
    print("Starting the data preprocessing process...")
    df = pd.read_excel(input_path)

    df['is_delayed'] = (df['Delivery_Time'] > 60).astype(int)
    df_clean = df.dropna().copy()

    features = ['Agent_Age', 'Weather', 'Traffic', 'Vehicle', 'is_delayed']
    df_selected = df_clean[features].copy()

    le = LabelEncoder()
    categorical_cols = ['Weather', 'Traffic', 'Vehicle']
    for col in categorical_cols:
        df_selected[col] = le.fit_transform(df_selected[col])
       
    scaler = StandardScaler()
    df_selected['Agent_Age'] = scaler.fit_transform(df_selected[['Agent_Age']])

    df_selected.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Data saved to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "../amazon_delivery.xlsx"
    OUTPUT_FILE = "amazon_delivery_preprocessing.csv"
   
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)