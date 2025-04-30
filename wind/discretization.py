import pandas as pd
import numpy as np
from datetime import time
import warnings
import re
#df.columns = df.columns.str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()

warnings.filterwarnings('ignore')

def load_data(filepath):
    try:
        print(f"\n📂 Loading data from: {filepath}")

        df = pd.read_csv(filepath, low_memory=False)
        print(f"📊 Initial row count: {len(df):,}")

        df.columns = df.columns.str.strip().str.replace('\u00b0', 'deg')

        time_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['time', 'date', 'timestamp'])), None)
        if not time_col:
            raise ValueError("No time column found")

        print(f"⏰ Time column identified: '{time_col}'")

        for dayfirst in [True, False]:
            df['_datetime_temp'] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=dayfirst, infer_datetime_format=True)
            if df['_datetime_temp'].isna().sum() == 0:
                break

        na_count = df['_datetime_temp'].isna().sum()
        if na_count > 0:
            print(f"⚠️ Couldn't parse {na_count:,} timestamps ({na_count/len(df):.1%})")
            print(df.loc[df['_datetime_temp'].isna(), time_col].head())

        df = df.reset_index(drop=True)
        df['datetime'] = df['_datetime_temp']
        df = df.drop(columns=['_datetime_temp', time_col], errors='ignore')

        print(f"📊 Final row count: {len(df):,}")
        return df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def safe_cut(series, bins, labels, colname):
    try:
        if len(bins) - 1 != len(labels):
            raise ValueError(f"❌ Column '{colname}': bins and labels mismatch")

        result = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
        result = result.astype("category")
        result = result.cat.add_categories('Unknown').fillna('Unknown')
        return result

    except Exception as e:
        print(f"⚠️ Error in safe_cut for {colname}: {e}")
        return pd.Series(['Unknown'] * len(series), index=series.index)

def discretize_all(df):
    try:
        print("\n🔢 Starting discretization...")

        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        df.columns = df.columns.str.replace('˚', 'deg')

        result = pd.DataFrame(index=df.index)

        result['Season'] = df['datetime'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        }).astype('category')

        result['DayNight'] = np.where(
            (df['datetime'].dt.time >= time(6, 0)) & (df['datetime'].dt.time < time(18, 0)),
            'Day', 'Night'
        )

        irr_bins = [0, 100, 400, 700, float('inf')]
        irr_labels = ['VL', 'L', 'M', 'H']

        wind_speed_bins = [-0.1, 2, 5, 8, 12, np.inf]
        wind_speed_labels = ['Calm', 'Breeze', 'Moderate', 'Strong', 'Gale']

        wind_dir_bins = [-0.1, 90, 180, 270, 360]
        wind_dir_labels = ['N-E', 'E-S', 'S-W', 'W-N']

        for col in df.columns:
            if col == 'datetime':
                continue

            col_clean = col.lower()

            if any(x in col_clean for x in ['solar', 'irrad', 'horizontal']):
                new_col = f"{col.split(' (')[0].strip()}_bin"
                result[new_col] = safe_cut(df[col], bins=irr_bins, labels=irr_labels, colname=col)

            elif 'wind speed' in col_clean:
                new_col = f"{col.split(' (')[0].strip()}"
                result[new_col] = safe_cut(df[col], bins=wind_speed_bins, labels=wind_speed_labels, colname=col)

            elif 'wind direction' in col_clean:
                new_col = f"{col.split(' (')[0].strip()}"
                result[new_col] = safe_cut(df[col], bins=wind_dir_bins, labels=wind_dir_labels, colname=col)

            elif 'temp' in col_clean:
                result['Temp'] = safe_cut(df[col], [-np.inf, 0, 10, 20, 30, np.inf], ['Freezing', 'Cold', 'Mild', 'Warm', 'High'], colname=col)

            elif 'hpa' in col_clean or 'atmos' in col_clean:
                result['Pressure'] = safe_cut(df[col], [-0.1, 950, 980, 1000, np.inf], ['Low', 'Medium', 'High', 'Very High'], colname=col)

            elif 'humid' in col_clean:
                result['Humidity'] = safe_cut(df[col], [-0.1, 30, 60, 80, np.inf], ['Dry', 'Comfort', 'Humid', 'Very Humid'], colname=col)

            elif 'power' in col_clean:
                try:
                    result['Power'] = pd.qcut(df[col], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
                    result['Power'] = result['Power'].astype("category").cat.add_categories('Unknown').fillna('Unknown')
                except:
                    result['Power'] = safe_cut(df[col], [-0.1, 5, 15, 25, np.inf], ['Low', 'Medium', 'High', 'Very High'], colname=col)

        #result['datetime'] = df['datetime']
        print(f"\n✅ Discretization complete. Final row count: {len(result):,}")
        return result

    except Exception as e:
        print(f"❌ Discretization error: {e}")
        return None

if __name__ == "__main__":
    INPUT_FILE = "solardata/newcleanedsingle_1wind5.csv"
    OUTPUT_FILE = "solardata/newcleandsinglediscretized.csv"

    print("="*50)
    print("🌪️ Wind Data Discretization Pipeline")
    print("="*50)

    df = load_data(INPUT_FILE)
    if df is not None:
        print("\n🔍 Columns detected:", df.columns.tolist())
        final = discretize_all(df)
        if final is not None:
            try:
                final.to_csv(OUTPUT_FILE, index=False)
                print(f"\n💾 Saved {len(final):,} rows to {OUTPUT_FILE}")
                print("\n🔍 Sample output:")
                print(final.head())
            except Exception as e:
                print(f"❌ Failed to save: {e}")
    else:
        print("❌ Processing failed")
