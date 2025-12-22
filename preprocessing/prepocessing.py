import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


df = pd.read_csv("raw/housing.csv")

print("1. Imputasi & Hapus Duplikat")
imputer = SimpleImputer(strategy='mean')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
if 'median_house_value' in num_cols:
    num_cols = num_cols.drop('median_house_value')

print("2. Hapus Outlier (Sebelum Scaling)")
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
df.to_csv(r'preprocessing\housing_preprocessing\housing_clean_auto.csv', index=False)
print("Seluruh Proses Selesai!")