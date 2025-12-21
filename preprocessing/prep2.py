import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing pipeline (converted from experiment notebook)"""

    # Pisahkan fitur numerik & kategorikal
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # ========================
    # HANDLE MISSING VALUES
    # ========================
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # ========================
    # ONE HOT ENCODING
    # ========================
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # ========================
    # SCALING NUMERIK
    # ========================
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[num_cols])

    scaled_num_df = pd.DataFrame(
        scaled_num,
        columns=num_cols
    )

    # ========================
    # GABUNG DATA
    # ========================
    final_df = pd.concat([scaled_num_df, encoded_df], axis=1)

    return final_df


def save_data(df: pd.DataFrame, output_path: str):
    """Save processed dataset"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    RAW_PATH = r"raw\housing.csv"
    OUTPUT_PATH = "preprocessing/housing_preprocessing/housing_clean.csv"

    df_raw = load_data(RAW_PATH)
    df_clean = preprocess_data(df_raw)
    save_data(df_clean, OUTPUT_PATH)

    print("✅ Preprocessing selesai.")
    print(f"📁 Dataset tersimpan di: {OUTPUT_PATH}")
