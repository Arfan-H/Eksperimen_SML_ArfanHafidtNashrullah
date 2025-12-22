import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(base_dir, "raw", "housing.csv")
output_dir = os.path.join(base_dir, "preprocessing", "housing_preprocessing")
output_file = os.path.join(output_dir, "housing_clean_auto.csv")

os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df = pd.read_csv(raw_path)

print("1. Imputasi & Hapus Duplikat")
imputer = SimpleImputer(strategy='mean')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if 'median_house_value' in num_cols:
    num_cols = num_cols.drop('median_house_value')

print("2. Hapus Outlier")
def remove_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean

df = remove_outliers_iqr(df, num_cols)

print("3. Standardisasi")
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("4. One-Hot Encoding")
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_data = encoder.fit_transform(df[['ocean_proximity']])
encoded_cols = encoder.get_feature_names_out(['ocean_proximity'])
df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
df = pd.concat([df.drop('ocean_proximity', axis=1), df_encoded], axis=1)

print("5. Simpan File")
# Menggunakan path absolut yang sudah dibuat di awal
df.to_csv(output_file, index=False)
print(f"File disimpan di: {output_file}")
print("Seluruh Proses Selesai!")