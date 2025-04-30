import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
import warnings
from dateutil.parser import parse
import os

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


def convert_excel_to_csv_utf8(input_path):
    """If input is Excel, convert to UTF-8 CSV."""
    if input_path.endswith(('.xlsx', '.xls')):
        try:
            df = pd.read_excel(input_path, engine='openpyxl')
            csv_path = input_path.rsplit('.', 1)[0] + '.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"📄 Converted Excel to CSV: {csv_path}")
            return csv_path
        except Exception as e:
            print(f"❌ Failed to convert Excel file: {e}")
            return None
    return input_path  # Already a CSV


def robust_date_parser(date_series):
    parsed_dates = []
    for date_str in date_series.astype(str):
        try:
            dt = pd.to_datetime(date_str, format='ISO8601', errors='raise')
        except:
            try:
                dt = parse(date_str, dayfirst=True)
                if dt.year < 2000:
                    dt = parse(date_str, yearfirst=True)
            except:
                dt = pd.NaT
        parsed_dates.append(dt)
    return pd.to_datetime(parsed_dates)


def load_and_clean(filepath):
    try:
        df = pd.read_csv(
            filepath,
            na_values=['NA', 'N/A', 'NaN', 'nan', '−99', '-99', '–99', '–', '-', '--', '0.001', '', ' ', 'NULL', 'null', '<NULL>'],
            keep_default_na=True
        )
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if not time_col:
            raise ValueError("No time column found")

        df[time_col] = robust_date_parser(df[time_col])
        df = df[~df[time_col].isna()].copy()
        df = df.set_index(time_col)

        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                cleaned = (
                    df[col].astype(str)
                    .str.replace(r'[()]', '', regex=True)  # Remove ( and )
                    .str.replace(r'[^0-9.\-eE]', '', regex=True)  # Keep only numeric characters and notation
                    .replace('', np.nan)
                )
                df[col] = pd.to_numeric(cleaned, errors='coerce')

        return df

    except Exception as e:
        print(f"❌ Error loading {filepath}: {str(e)}")
        return None


def process_missing_data(df):
    if df is None or df.empty:
        return df

    irrad_cols = [c for c in df.columns if any(x in c.lower() for x in ['irrad', 'solar'])]
    power_cols = [c for c in df.columns if 'power' in c.lower()]
    weather_cols = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'hpa', 'humidity'])]

    print("\n=== Column Summary ===")
    print(f"📊 Irradiance: {irrad_cols}")
    print(f"🌡️ Weather: {weather_cols}")
    print(f"⚡ Power: {power_cols}")

    for col in irrad_cols:
        day_mask = (df.index.time >= time(6, 0)) & (df.index.time < time(20, 0))
        df.loc[day_mask, col] = df.loc[day_mask, col].replace(0, np.nan)

    for col in irrad_cols + weather_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=6)
            df[col] = df[col].fillna(df.groupby([df.index.month, df.index.hour])[col].transform('median'))
            df[col] = df[col].ffill().bfill()

    for col in power_cols:
        if col in df.columns:
            print(f"\n🔧 Power cleaning for {col}:")
            print("Before cleaning:")
            print(df[col].describe())

            df[col] = df[col].clip(lower=0)
            df[col] = df[col].interpolate(method='time', limit=12)
            df[col] = df[col].fillna(df.groupby(df.index.hour)[col].transform('median'))
            df[col] = df[col].ffill().bfill()

            print("\nAfter cleaning:")
            print(df[col].describe())

    for col in weather_cols:
        if col in df.columns:
            if 'humidity' in col.lower():
                df[col] = df[col].clip(0, 100)
            df[col] = df[col].fillna(df.groupby([df.index.month, df.index.hour])[col].transform('median'))

    if irrad_cols:
        night_mask = (df.index.time >= time(20, 0)) | (df.index.time < time(6, 0))
        df.loc[night_mask, irrad_cols] = 0

    return df


def validate_results(df):
    if df is None or df.empty:
        print("No data to validate")
        return

    print("\n=== Data Quality Report ===")
    print(f"📅 Time range: {df.index.min()} to {df.index.max()}")
    print(f"🕒 Time frequency: {pd.infer_freq(df.index)}")

    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string())

    print("\n=== Descriptive Statistics ===")
    print(df.describe().to_string(float_format="%.2f"))

    sample_col = next((c for c in df.columns if 'irrad' in c.lower()), df.columns[0])
    plt.figure(figsize=(14, 5))
    df[sample_col].plot(title=f'{sample_col} - Final Cleaned Data')
    plt.ylabel(sample_col.split('(')[-1].split(')')[0])
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    INPUT_FILE = "solardata/Solar station site 1 (Nominal capacity-50MW).xlsx"
    INPUT_FILE = convert_excel_to_csv_utf8(INPUT_FILE)

    if INPUT_FILE is None:
        raise ValueError("❌ Failed to convert or locate input file.")

    OUTPUT_FILE = "solardata/newcleanedsingle_1solar1.csv"

    try:
        print("\n" + "=" * 50)
        print(f"⚡ Processing {INPUT_FILE}")
        print("=" * 50)

        print("🔄 Loading data...")
        df = load_and_clean(INPUT_FILE)

        if df is None:
            raise ValueError("❌ Failed to load data - check error messages above")

        print("\n🧹 Cleaning data...")
        df_clean = process_missing_data(df)

        print("\n🔍 Validating results...")
        validate_results(df_clean)

        df_clean.to_csv(OUTPUT_FILE, index=True)
        print(f"\n✅ Success! Cleaned data saved to {OUTPUT_FILE}")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                df_clean[numeric_cols].corr(),
                annot=True, fmt=".2f", cmap='coolwarm',
                vmin=-1, vmax=1, center=0
            )
            plt.title("Feature Correlations")
            plt.tight_layout()
            # plt.show()

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")

