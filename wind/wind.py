"""
main_pipeline.py
1. Cleaning raw input data using clean_wind_data.py
2. Discretizing cleaned data using discretize_wind_data.py
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Import your existing scripts as modules
sys.path.append(str(Path(__file__).parent))
from missingdatahandlingwind import load_and_clean, process_missing_data, validate_results
from missinghandlingdataplusdiscretizationwindnew import load_data, discretize_all


def run_pipeline(input_file, cleaned_output=None, discretized_output=None):
    """
    Run complete pipeline from raw data to discretized output

    Args:
        input_file (str): Path to raw input file (.xlsx or .csv)
        cleaned_output (str, optional): Path to save cleaned data
        discretized_output (str, optional): Path to save discretized data
    """
    print(f"\n🚀 Starting pipeline for {input_file}")

    # --- Cleaning Phase ---
    print("\n=== CLEANING PHASE ===")
    raw_df, invalid_rows = load_and_clean(input_file)

    if raw_df is None:
        print("❌ Cleaning failed - aborting pipeline")
        return None

    cleaned_df = process_missing_data(raw_df)
    validate_results(cleaned_df)

    # Save cleaned data if requested
    if cleaned_output:
        cleaned_df.to_csv(cleaned_output, index=True)
        print(f"\n💾 Saved cleaned data to {cleaned_output}")

    # --- Discretization Phase ---
    print("\n=== DISCRETIZATION PHASE ===")
    # Need to reset index and save to temp file for discretization script
    temp_clean_path = "temp_cleaned.csv"
    cleaned_df.reset_index().to_csv(temp_clean_path, index=False)

    discretize_df = load_data(temp_clean_path)
    if discretize_df is None:
        print("❌ Discretization failed - aborting pipeline")
        return None

    discretized_df = discretize_all(discretize_df)

    # Save final output if requested
    if discretized_output and discretized_df is not None:
        discretized_df.to_csv(discretized_output, index=False)
        print(f"\n💾 Saved discretized data to {discretized_output}")

    # Clean up temp file
    if os.path.exists(temp_clean_path):
        os.remove(temp_clean_path)

    return discretized_df


if __name__ == "__main__":
    # Example usage
    INPUT_FILE = "solardata/Wind farm site 5 (Nominal capacity-36MW).xlsx"
    CLEANED_OUTPUT = "solardata/newcleaned_wind5.csv"
    DISCRETIZED_OUTPUT = "solardata/newdiscretized_wind5.csv"

    final_result = run_pipeline(
        input_file=INPUT_FILE,
        cleaned_output=CLEANED_OUTPUT,
        discretized_output=DISCRETIZED_OUTPUT
    )

    if final_result is not None:
        print("\n✅ Pipeline completed successfully!")
        print("\nFinal discretized data sample:")
        print(final_result.head())
