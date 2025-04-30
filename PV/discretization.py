import pandas as pd
import numpy as np
from datetime import time
import warnings

# Suppress performance warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def load_data(filepath):
    """
    Robust data loading with comprehensive datetime parsing and validation
    Returns:
        - DataFrame with datetime index
        - None if loading fails
    """
    try:
        print(f"\n📂 Loading data from: {filepath}")

        # Read data with careful handling
        df = pd.read_csv(filepath, parse_dates=False, low_memory=False)
        print(f"📊 Initial row count: {len(df):,}")

        # Clean column names
        df.columns = df.columns.str.strip()
        print("🔍 Detected columns:", df.columns.tolist())

        # Find time column with multiple fallbacks
        time_col = None
        time_col_candidates = [
            col for col in df.columns
            if any(kw in col.lower() for kw in ['time', 'date', 'timestamp', 'datetime'])
        ]

        if not time_col_candidates:
            raise ValueError("No time column found in dataset")

        time_col = time_col_candidates[0]
        print(f"⏰ Selected time column: '{time_col}'")

        # Try multiple datetime parsing strategies
        parsing_attempts = [
            {'dayfirst': True, 'yearfirst': False},
            {'dayfirst': False, 'yearfirst': True},
            {'dayfirst': False, 'yearfirst': False},
            {'format': '%Y-%m-%d %H:%M:%S'},
            {'format': '%m/%d/%Y %H:%M:%S'},
            {'format': '%d/%m/%Y %H:%M:%S'}
        ]

        best_na_count = float('inf')
        best_parsed = None

        for attempt in parsing_attempts:
            try:
                temp_parsed = pd.to_datetime(
                    df[time_col],
                    errors='coerce',
                    **attempt
                )
                current_na = temp_parsed.isna().sum()

                if current_na < best_na_count:
                    best_na_count = current_na
                    best_parsed = temp_parsed
                    print(f"✅ Improved parsing: {attempt} → {len(df) - best_na_count:,} valid timestamps")

                if best_na_count == 0:
                    break

            except Exception as e:
                continue

        if best_parsed is None:
            raise ValueError("All datetime parsing attempts failed")

        df[time_col] = best_parsed

        # Handle remaining NA values
        if best_na_count > 0:
            print(f"⚠️ Warning: Couldn't parse {best_na_count:,} timestamps ({best_na_count / len(df):.2%})")
            print("Sample problematic values:")
            print(df.loc[df[time_col].isna(), time_col].unique()[:10])

            # Option: Keep rows with NA timestamps with dummy index
            # df.index = df[time_col].fillna(pd.Timestamp('1970-01-01'))
            # Or: Drop them as in original
            df = df.dropna(subset=[time_col])

        # Set index and clean
        df = df.set_index(time_col)
        df.index.name = 'datetime'
        print(f"📊 Final valid rows: {len(df):,}")

        return df

    except Exception as e:
        print(f"❌ Critical error in load_data: {str(e)}")
        return None


def safe_cut(series, bins, labels, colname):
    """
    Robust binning with comprehensive error handling
    Returns:
        - Series with categorical values (as strings)
    """
    try:
        if not isinstance(series, pd.Series):
            raise ValueError("Input must be a pandas Series")

        if len(bins) < 2:
            raise ValueError("Must provide at least 2 bin edges")

        if len(labels) != len(bins) - 1:
            raise ValueError(f"Need {len(bins) - 1} labels for {len(bins)} bin edges")

        # Handle empty/invalid series
        if series.empty or series.isna().all():
            return pd.Series(['Unknown'] * len(series), index=series.index)

        # Handle constant values
        if series.nunique() == 1:
            val = series.iloc[0]
            if pd.isna(val):
                return pd.Series(['Unknown'] * len(series), index=series.index)
            return pd.Series([labels[0]] * len(series), index=series.index)

        # Actual binning
        result = pd.cut(
            series,
            bins=bins,
            labels=labels,
            include_lowest=True,
            duplicates='drop'
        )

        # Fill NA and convert to strings
        return result.astype(str).replace('nan', 'Unknown')

    except Exception as e:
        print(f"⚠️ Binning error for '{colname}': {str(e)}")
        return pd.Series(['Unknown'] * len(series), index=series.index)


def discretize_all(df):
    """
    Comprehensive feature discretization with validation
    Returns:
        - DataFrame with all features discretized (as strings)
        - None if processing fails
    """
    try:
        if df is None or df.empty:
            raise ValueError("Empty or None DataFrame provided")

        print("\n🔢 Beginning discretization...")
        result = pd.DataFrame(index=df.index)

        # 1. Temporal Features ----------------------------
        # Season (with validation)
        month_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        }
        result['Season'] = df.index.month.map(month_map).fillna('Unknown')

        # Day/Night (with edge case handling)
        day_mask = (df.index.time >= time(6, 0)) & (df.index.time < time(18, 0))
        night_mask = ~day_mask
        result['DayNight'] = np.select(
            [day_mask, night_mask],
            ['Day', 'Night'],
            default='Unknown'
        )

        # 2. Environmental Features -----------------------
        # Solar Irradiance
        solar_bins = [-np.inf, 400, 700, 1000, np.inf]
        solar_labels = ['Low', 'Medium', 'High', 'Very High']

        irrad_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['irrad', 'solar', 'horizontal'])]
        for col in irrad_cols:
            clean_name = col.split(' (')[0].strip()
            result[f"{clean_name}"] = safe_cut(
                df[col],
                bins=solar_bins,
                labels=solar_labels,
                colname=col
            )
            # Special handling for nighttime zero values
            #result.loc[(result['DayNight'] == 'Night') & (df[col] <= 0.1), f"{clean_name}"] = 'Night'

        # Temperature
        temp_cols = [c for c in df.columns if 'temp' in c.lower()]
        if temp_cols:
            col = temp_cols[0]
            result['Temp'] = safe_cut(
                df[col],
                bins=[-np.inf, 0, 10, 20, 30, np.inf],
                labels=['Freezing', 'Cold', 'Mild', 'Warm', 'HighT'],
                colname=col
            )

        # Pressure
        press_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['hpa', 'atmos', 'pressure'])]
        if press_cols:
            col = press_cols[0]
            result['Pressure'] = safe_cut(
                df[col],
                bins=[-np.inf, 950, 980, 1000, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High'],
                colname=col
            )

        # Humidity
        hum_cols = [c for c in df.columns if 'humid' in c.lower()]
        if hum_cols:
            col = hum_cols[0]
            result['Humidity'] = safe_cut(
                df[col],
                bins=[-1, 30, 60, 80, 100],
                labels=['Dry', 'Comfortable', 'Humid', 'Very Humid'],
                colname=col
            )

        # 3. Power Features -------------------------------
        power_cols = [c for c in df.columns if 'power' in c.lower()]
        if power_cols:
            col = power_cols[0]

            # Try quantile-based binning first
            try:
                result['Power'] = pd.qcut(
                    df[col],
                    q=4,
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    duplicates='drop'
                ).astype(str).replace('nan', 'Unknown')
            except:
                # Fallback to fixed bins
                result['Power'] = safe_cut(
                    df[col],
                    bins=[-0.1, 5, 15, 25, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Very High'],
                    colname=col
                )

        # Final validation
        print("\n✅ Discretization completed successfully")
        print(f"Final shape: {result.shape}")
        print("Missing values per column:")
        print(result.isna().sum())

        # Convert all to string type (no categorical for now)
        return result.fillna('Unknown')

    except Exception as e:
        print(f"❌ Discretization failed: {str(e)}")
        return None


def main():
    """Main execution pipeline with comprehensive logging"""
    INPUT_FILE = "solardata/newcleanedsingle_1solar1.csv"
    OUTPUT_FILE = "solardata/newsinglediscretizedsolar8.csv"

    print("\n" + "=" * 50)
    print("🌞 Solar Data Discretization Pipeline")
    print("=" * 50)

    # 1. Data Loading
    print("\n🔧 STEP 1: Loading and preparing data...")
    df = load_data(INPUT_FILE)

    if df is None:
        print("❌ Pipeline aborted due to loading failure")
        return

    # 2. Data Discretization
    print("\n🔧 STEP 2: Discretizing features...")
    discretized = discretize_all(df)

    if discretized is None:
        print("❌ Pipeline aborted due to discretization failure")
        return

    # 3. Result Validation
    print("\n🔧 STEP 3: Validating output...")
    print(f"\n📋 Final columns: {discretized.columns.tolist()}")
    print(f"📊 Final row count: {len(discretized):,}")

    # 4. Save Results
    try:
        discretized.to_csv(OUTPUT_FILE, index=False)
        print(f"\n💾 Success! Saved to {OUTPUT_FILE}")

        # Print sample
        print("\n🔍 Sample output (5 random rows):")
        print(discretized.sample(5))

    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")


if __name__ == "__main__":
    main()
