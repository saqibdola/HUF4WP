# solar_pipeline.py
import os
import pandas as pd
from datetime import datetime

# Import your two scripts as modules
from missingdatahandling import convert_excel_to_csv_utf8, load_and_clean, process_missing_data
from discretization import load_data as load_for_discretization, discretize_all

def run_solar_pipeline(input_file, cleaned_output=None, discretized_output=None):
    print("\n☀️ Starting Solar Data Pipeline")

    try:
        # Step 1: Convert Excel if needed
        input_file = convert_excel_to_csv_utf8(input_file)
        if input_file is None:
            raise ValueError("Failed to convert input file")

        # Step 2: Load and clean data
        print("\n🔄 Cleaning data...")
        df = load_and_clean(input_file)
        if df is None:
            raise ValueError("Cleaning failed")
        df_clean = process_missing_data(df)

        if cleaned_output:
            df_clean.to_csv(cleaned_output, index=True)
            print(f"✅ Cleaned data saved: {cleaned_output}")

        # Step 3: Discretization
        print("\n🔄 Discretizing data...")
        df_clean['datetime'] = df_clean.index  # Required by discretizer
        df_for_disc = df_clean.reset_index()  # Match expected input format
        df_for_disc.to_csv("temp_clean.csv", index=False)

        df_loaded = load_for_discretization("temp_clean.csv")
        if df_loaded is None:
            raise ValueError("Discretization loading failed")

        df_disc = discretize_all(df_loaded)
        os.remove("temp_clean.csv")

        if df_disc is None:
            raise ValueError("Discretization failed")

        if discretized_output:
            df_disc.to_csv(discretized_output, index=False)
            print(f"✅ Discretized data saved: {discretized_output}")

        print("\n🎉 Solar pipeline completed successfully!")
        return df_clean, df_disc

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return None, None

if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "" # Input solar data file(s)
    CLEANED_FILE = "" # Output cleaned file(s)
    DISCRETIZED_FILE = "" # Output discretized file(s)

    run_solar_pipeline(INPUT_FILE, CLEANED_FILE, DISCRETIZED_FILE)
