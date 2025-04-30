import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
import os
import warnings
from dateutil.parser import parse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
warnings.filterwarnings('ignore')


def convert_xlsx_to_csv(filepath):
    """Convert .xlsx file to UTF-8 CSV and return new path."""
    csv_path = filepath.replace('.xlsx', '.csv')
    try:
        df_xlsx = pd.read_excel(filepath)
        df_xlsx.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"📁 Converted {filepath} to {csv_path}")
        return csv_path
    except Exception as e:
        print(f"❌ Failed to convert Excel file: {e}")
        return None


def robust_date_parser(date_series):
    """Parse dates while preserving original index positions"""
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
        if filepath.endswith('.xlsx'):
            filepath = convert_xlsx_to_csv(filepath)
            if filepath is None:
                return None, None

        # Read CSV while preserving original data as much as possible
        df = pd.read_csv(
            filepath,
            na_values=['NA', 'N/A', 'NaN', 'nan', '−99', '-99', '–99', '–', '-', '--',
                       '0.001', '', ' ', 'NULL', 'null'],
            keep_default_na=True,
            dtype=object  # Read everything as object first to prevent automatic conversion
        )

        print(f"\n📊 Original data shape: {df.shape}")

        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if not time_col:
            raise ValueError("No time column found")

        # Parse dates
        df[time_col] = robust_date_parser(df[time_col])

        # Separate invalid rows
        invalid_rows = df[df[time_col].isna()].copy()
        df = df[~df[time_col].isna()].copy()
        df = df.set_index(time_col)

        # Convert numeric columns while preserving exact values
        for col in df.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Special handling for temperature column
            if 'temp' in col.lower():
                try:
                    # First try direct conversion to float without any modification
                    original_values = df[col].copy()
                    df[col] = pd.to_numeric(df[col], errors='raise')

                    # Verify no values were changed
                    if not np.allclose(df[col].dropna(), pd.to_numeric(original_values.dropna(), errors='coerce')):
                        print(f"⚠️ Temperature values were modified during conversion - reverting")
                        df[col] = original_values
                        df[col] = pd.to_numeric(df[col], errors='coerce')  # Fallback to coerce if needed
                except:
                    # Fallback cleaning only for truly problematic values
                    cleaned = (
                        df[col].astype(str)
                        .str.replace('[°℃CF]', '', regex=True)
                        .str.replace(r'[^0-9.\-]', '', regex=True)
                        .replace('', np.nan)
                    )
                    df[col] = pd.to_numeric(cleaned, errors='coerce')
            else:
                # Standard numeric conversion for other columns
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    cleaned = (
                        df[col].astype(str)
                        .str.replace(r'[^0-9.\-eE]+', '', regex=True)
                        .replace('', np.nan)
                    )
                    df[col] = pd.to_numeric(cleaned, errors='coerce')

        # Temperature validation
        temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
        if temp_col:
            print(f"\n=== Temperature Validation ===")
            print(f"Column '{temp_col}' has {df[temp_col].isna().sum()} missing values")
            print(f"Value range: {df[temp_col].min()} to {df[temp_col].max()}")
            print("Sample values:")
            print(df[temp_col].head())

        return df, invalid_rows

    except Exception as e:
        print(f"❌ Error loading {filepath}: {str(e)}")
        return None, None


def process_missing_data(df):
    if df is None or df.empty:
        return df

    original_shape = df.shape
    print(f"\n🧹 Processing missing data (initial shape: {original_shape})")

    wind_cols = [c for c in df.columns if 'wind' in c.lower()]
    irrad_cols = [c for c in df.columns if 'irrad' in c.lower()]
    power_cols = [c for c in df.columns if 'power' in c.lower()]
    weather_cols = [c for c in df.columns if any(x in c.lower() for x in ['temp', 'hpa', 'humidity'])]

    print("\n=== Column Summary ===")
    print(f"🌬️ Wind: {wind_cols}")
    print(f"🌞 Irradiance: {irrad_cols}")
    print(f"🌡️ Weather: {weather_cols}")
    print(f"⚡ Power: {power_cols}")

    # Step 1: Handle invalid values (negatives) - skip temperature
    for col in wind_cols + irrad_cols + [c for c in weather_cols if 'temp' not in c.lower()]:
        if col in df.columns:
            df[col] = df[col].where(df[col] >= 0, np.nan)

    # Step 2: Replace zeros during daytime for irradiance
    for col in irrad_cols:
        day_mask = (df.index.time >= time(6, 0)) & (df.index.time < time(20, 0))
        df.loc[day_mask, col] = df.loc[day_mask, col].replace(0, np.nan)

    # Step 3: Interpolation (skip temperature if no missing values)
    interpolate_cols = wind_cols + irrad_cols + [c for c in weather_cols if 'temp' not in c.lower()]
    for col in interpolate_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].interpolate(method='time', limit=6)
            df[col] = df[col].fillna(df.groupby([df.index.month, df.index.hour])[col].transform('median'))
            df[col] = df[col].ffill().bfill()

    # Special handling for temperature - only process if missing values exist
    temp_col = next((c for c in weather_cols if 'temp' in c.lower()), None)
    if temp_col and temp_col in df.columns and df[temp_col].isna().any():
        print(f"\nProcessing missing values in temperature column")
        original_temp = df[temp_col].copy()
        df[temp_col] = df[temp_col].interpolate(method='time', limit=6)
        df[temp_col] = df[temp_col].fillna(df.groupby([df.index.month, df.index.hour])[temp_col].transform('median'))
        df[temp_col] = df[temp_col].ffill().bfill()

        # Verify non-missing values weren't changed
        non_missing_mask = original_temp.notna()
        if not np.allclose(df[temp_col][non_missing_mask], original_temp[non_missing_mask]):
            print("⚠️ Warning: Some non-missing temperature values were modified during imputation")

    # Step 4: Clean power
    for col in power_cols:
        if df[col].isna().any():
            df[col] = df[col].clip(lower=0).interpolate(method='time', limit=12)
            df[col] = df[col].fillna(df.groupby(df.index.hour)[col].transform('median'))
            df[col] = df[col].ffill().bfill()

    # Step 5: Humidity constraint
    for col in weather_cols:
        if 'humidity' in col.lower():
            df[col] = df[col].clip(0, 100)

    # Step 6: Nighttime irradiance = 0
    if irrad_cols:
        night_mask = (df.index.time >= time(20, 0)) | (df.index.time < time(6, 0))
        df.loc[night_mask, irrad_cols] = 0

    assert df.shape[0] == original_shape[0], "Row count changed during processing"
    print(f"✅ Missing data processed (final shape: {df.shape})")

    # Final temperature check
    if temp_col:
        print(f"\n=== Final Temperature Check ===")
        print(f"Missing values: {df[temp_col].isna().sum()}")
        print(f"Value range: {df[temp_col].min()} to {df[temp_col].max()}")

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

    temp_col = next((c for c in df.columns if 'temp' in c.lower()), None)
    if temp_col:
        plt.figure(figsize=(14, 5))
        df[temp_col].plot(title=f'{temp_col} - Temperature Values')
        plt.ylabel("Temperature")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()


if __name__ == "__main__":
    INPUT_FILE = "solardata/Wind farm site 5 (Nominal capacity-36MW).xlsx"
    OUTPUT_FILE = "solardata/newcleanedsingle_1wind5.csv"
    INVALID_FILE = "invalid_rows.csv"

    try:
        print("\n" + "=" * 50)
        print(f"⚡ Processing {INPUT_FILE}")
        print("=" * 50)

        print("🔄 Loading data...")
        df, invalid_rows = load_and_clean(INPUT_FILE)

        if df is None:
            raise ValueError("❌ Failed to load data - check error messages above")

        print("\n🧹 Cleaning data...")
        df_clean = process_missing_data(df)

        print("\n🔍 Validating results...")
        validate_results(df_clean)

        df_clean.to_csv(OUTPUT_FILE, index=True)
        print(f"\n📄 Cleaned data saved to {OUTPUT_FILE}")

        if invalid_rows is not None and not invalid_rows.empty:
            invalid_rows.to_csv(INVALID_FILE, index=False)
            print(f"⚠️ Saved {len(invalid_rows)} invalid rows to {INVALID_FILE}")

        # Correlation plot
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df_clean[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Feature Correlations")
            plt.tight_layout()
            #plt.show()

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
