"""
main_pipeline.py
Unified pipeline for processing both wind and solar datasets with:
1. Data cleaning
2. Discretization
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import time
import warnings

# Configure environment
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Import your existing scripts as modules
sys.path.append(str(Path(__file__).parent))
from missingdatahandlingwind import load_and_clean as load_wind, process_missing_data as process_wind
from missingdataimputationsolar import load_and_clean as load_solar, process_missing_data as process_solar
from missinghandlingdataplusdiscretizationwindnew import discretize_all as discretize_wind
from missinghandlingdataplusdiscretizationsolarnew import discretize_all as discretize_solar

class DataProcessor:
    """Handles both wind and solar data processing pipelines"""

    @staticmethod
    def detect_data_type(filepath):
        """Determine if file contains wind or solar data"""
        filename = os.path.basename(filepath).lower()
        if 'wind' in filename:
            return 'wind'
        elif 'solar' in filename:
            return 'solar'
        else:
            # Try to detect from columns
            try:
                df_sample = pd.read_csv(filepath, nrows=1)
                if any('wind' in col.lower() for col in df_sample.columns):
                    return 'wind'
                elif any('irrad' in col.lower() or 'solar' in col.lower() for col in df_sample.columns):
                    return 'solar'
            except:
                pass
        raise ValueError("Cannot determine data type (wind/solar) - check filename or columns")

    @staticmethod
    def run_cleaning(data_type, input_file):
        """Run appropriate cleaning pipeline"""
        if data_type == 'wind':
            df, invalid = load_wind(input_file)
            return process_wind(df), invalid
        elif data_type == 'solar':
            df = load_solar(input_file)
            return process_solar(df), None
        return None, None

    @staticmethod
    def run_discretization(data_type, clean_df):
        """Run appropriate discretization pipeline"""
        # Save to temp file for compatibility
        temp_path = "temp_clean.csv"
        clean_df.reset_index().to_csv(temp_path, index=False)

        if data_type == 'wind':
            df = pd.read_csv(temp_path)
            result = discretize_wind(df)
        else:
            df = pd.read_csv(temp_path)
            result = discretize_solar(df)

        os.remove(temp_path)
        return result

def unified_pipeline(input_file, cleaned_output=None, discretized_output=None):
    """
    Complete processing pipeline for both wind and solar data

    Args:
        input_file: Path to input data file
        cleaned_output: Path to save cleaned data (optional)
        discretized_output: Path to save discretized data (optional)
    """
    print(f"\n🚀 Starting unified pipeline for {input_file}")

    try:
        # Step 1: Detect data type
        data_type = DataProcessor.detect_data_type(input_file)
        print(f"🔍 Detected data type: {data_type.upper()}")

        # Step 2: Data cleaning
        print("\n=== CLEANING PHASE ===")
        clean_df, invalid = DataProcessor.run_cleaning(data_type, input_file)

        if clean_df is None:
            raise ValueError("Data cleaning failed")

        if cleaned_output:
            clean_df.to_csv(cleaned_output, index=True)
            print(f"💾 Saved cleaned data to {cleaned_output}")

        # Step 3: Discretization
        print("\n=== DISCRETIZATION PHASE ===")
        discretized = DataProcessor.run_discretization(data_type, clean_df)

        if discretized is None:
            raise ValueError("Discretization failed")

        if discretized_output:
            discretized.to_csv(discretized_output, index=False)
            print(f"💾 Saved discretized data to {discretized_output}")

            print("\n🔍 Sample output:")
            print(discretized.head())

        return discretized

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage - will auto-detect data type
    INPUT_FILES = [
        #"solardata/Wind farm site 5 (Nominal capacity-36MW).xlsx",
        "solardata/Solar station site 1 (Nominal capacity-50MW).xlsx"
    ]

    for input_file in INPUT_FILES:
        # Generate output paths
        base = os.path.splitext(os.path.basename(input_file))[0]
        clean_out = f"cleaned_{base}.csv"
        discrete_out = f"discretized_{base}.csv"

        # Run pipeline
        unified_pipeline(
            input_file=input_file,
            cleaned_output=clean_out,
            discretized_output=discrete_out
        )
