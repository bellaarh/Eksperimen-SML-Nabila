import os
import sys
import numpy as np
import pandas as pd

def run_preprocessing(input_path, output_dir):
    print(f" Membaca data dari: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ Error: File {input_path} tidak ditemukan!")
        sys.exit(1)
        
    df = pd.read_csv(input_path, sep=';')
    df.columns = df.columns.str.strip()

    kolom_desimal = ['Agent_Rating', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
    for col in kolom_desimal:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    if 'Delivery_Time' in df.columns:
        df['Late_delivery_risk'] = np.where(df['Delivery_Time'] > 120, 1, 0)

    if 'Order_Date' in df.columns:
        df['Order_Date_dt'] = pd.to_datetime(df['Order_Date'].astype(str).str.strip(), format='mixed', errors='coerce')
        df['Is_Weekend'] = df['Order_Date_dt'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

    if 'Order_Time' in df.columns and 'Pickup_Time' in df.columns:
        order_time_clean = df['Order_Time'].astype(str).str.strip().str.replace('.', ':', regex=False)
        pickup_time_clean = df['Pickup_Time'].astype(str).str.strip().str.replace('.', ':', regex=False)

        df['Order_Time_dt'] = pd.to_datetime('2026-01-01 ' + order_time_clean, format='mixed', errors='coerce')
        df['Pickup_Time_dt'] = pd.to_datetime('2026-01-01 ' + pickup_time_clean, format='mixed', errors='coerce')
        df['Prep_Time_Minutes'] = (df['Pickup_Time_dt'] - df['Order_Time_dt']).dt.total_seconds() / 60

    if all(col in df.columns for col in ['Store_Latitude', 'Drop_Latitude', 'Store_Longitude', 'Drop_Longitude']):
        df['Distance_KM'] = (abs(df['Store_Latitude'] - df['Drop_Latitude'])
                             + abs(df['Store_Longitude'] - df['Drop_Longitude'])) * 111

    kolom_dibuang = ['Order_ID', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 
                     'Order_Date', 'Order_Time', 'Pickup_Time', 'Order_Time_dt', 'Pickup_Time_dt', 'Order_Date_dt', 'Delivery_Time']
    df_cleaned = df.drop(columns=[col for col in kolom_dibuang if col in df.columns], errors='ignore')

    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().sum() > 0:
            mode_series = df_cleaned[col].mode()
            if not mode_series.empty:
                df_cleaned[col] = df_cleaned[col].fillna(mode_series[0])


    kolom_kategori = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    kolom_kategori_ada = [col for col in kolom_kategori if col in df_cleaned.columns]
    df_cleaned = pd.get_dummies(df_cleaned, columns=kolom_kategori_ada, drop_first=True)

    kolom_boolean = df_cleaned.select_dtypes(include='bool').columns
    df_cleaned[kolom_boolean] = df_cleaned[kolom_boolean].astype(int)

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'amazon_clean.csv')
    df_cleaned.to_csv(output_file_path, index=False)
    
    print(f" Sukses! Data hasil otomatisasi disimpan di: {output_file_path}")
    print(f"📊 Dimensi Data Akhir: {df_cleaned.shape}\n")

if __name__ == "__main__":
    import os

    INPUT_DATA = "amazondelivery_raw.csv"
    OUTPUT_FILE = "amazon_delivery.csv"

    if not os.path.exists(INPUT_DATA):
        INPUT_DATA = "../amazondelivery_raw.csv"
        OUTPUT_FILE = "../amazon_delivery.csv"

    run_preprocessing(INPUT_DATA, OUTPUT_FILE)
