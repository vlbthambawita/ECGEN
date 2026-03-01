#!/usr/bin/env python3
"""
Test script to validate data loading before training.
This helps debug issues with the dataset.
"""

import os
import sys
import argparse
from pathlib import Path

def test_data_path(data_dir: str):
    """Test if data directory structure is correct."""
    print("=" * 80)
    print("Testing MIMIC-IV-ECG Data Path")
    print("=" * 80)
    print()
    
    # Check if directory exists
    print(f"1. Checking data directory: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"   ❌ ERROR: Directory does not exist!")
        return False
    print(f"   ✓ Directory exists")
    print()
    
    # Check for machine_measurements.csv
    print(f"2. Checking for machine_measurements.csv")
    measurements_file = os.path.join(data_dir, "machine_measurements.csv")
    if not os.path.exists(measurements_file):
        print(f"   ❌ ERROR: machine_measurements.csv not found!")
        print(f"   Expected at: {measurements_file}")
        return False
    print(f"   ✓ machine_measurements.csv found")
    
    # Check file size
    file_size = os.path.getsize(measurements_file) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.2f} MB")
    print()
    
    # Try to load and check structure
    print(f"3. Loading machine_measurements.csv")
    try:
        import pandas as pd
        df = pd.read_csv(measurements_file)
        print(f"   ✓ Successfully loaded")
        print(f"   Number of rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print()
        
        # Check required columns
        print(f"4. Checking required columns")
        required_cols = ["subject_id", "study_id", "rr_interval", "p_onset", "p_end", 
                        "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"   ❌ ERROR: Missing columns: {missing_cols}")
            return False
        print(f"   ✓ All required columns present")
        print()
        
        # Check for ECG files
        print(f"5. Checking for ECG waveform files (sampling first 10)")
        files_dir = os.path.join(data_dir, "files")
        if not os.path.exists(files_dir):
            print(f"   ❌ ERROR: 'files' directory not found at: {files_dir}")
            return False
        print(f"   ✓ 'files' directory exists")
        
        found_count = 0
        checked_count = 0
        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            subject_id = int(row["subject_id"])
            study_id = int(row["study_id"])
            
            # MIMIC-IV-ECG uses first 4 digits for directory structure
            subject_str = str(subject_id)
            if len(subject_str) >= 4:
                p_dir = f"p{subject_str[:4]}"
            else:
                p_dir = f"p{subject_str[:2]}"
            p_subdir = f"p{subject_id}"
            s_dir = f"s{study_id}"
            
            hea_path = os.path.join(files_dir, p_dir, p_subdir, s_dir, f"{study_id}.hea")
            dat_path = os.path.join(files_dir, p_dir, p_subdir, s_dir, f"{study_id}.dat")
            
            checked_count += 1
            if os.path.exists(hea_path) and os.path.exists(dat_path):
                found_count += 1
        
        print(f"   Found {found_count}/{checked_count} ECG file pairs")
        
        if found_count == 0:
            print(f"   ❌ ERROR: No ECG files found!")
            print(f"   Expected structure: {files_dir}/pXX/pXXXXXXX/sXXXXXXXX/XXXXXXXX.hea/.dat")
            return False
        elif found_count < checked_count:
            print(f"   ⚠ WARNING: Some ECG files are missing")
            print(f"   Consider using --skip-missing-check flag")
        else:
            print(f"   ✓ All sampled ECG files found")
        print()
        
        # Test loading one ECG
        print(f"6. Testing ECG loading with wfdb")
        try:
            import wfdb
            import numpy as np
            
            # Find first valid file
            for idx in range(min(100, len(df))):
                row = df.iloc[idx]
                subject_id = int(row["subject_id"])
                study_id = int(row["study_id"])
                
                # MIMIC-IV-ECG uses first 4 digits for directory structure
                subject_str = str(subject_id)
                if len(subject_str) >= 4:
                    p_dir = f"p{subject_str[:4]}"
                else:
                    p_dir = f"p{subject_str[:2]}"
                p_subdir = f"p{subject_id}"
                s_dir = f"s{study_id}"
                
                record_path = os.path.join(files_dir, p_dir, p_subdir, s_dir, str(study_id))
                
                if os.path.exists(record_path + ".hea"):
                    try:
                        record = wfdb.rdrecord(record_path)
                        ecg = record.p_signal
                        print(f"   ✓ Successfully loaded ECG")
                        print(f"   Shape: {ecg.shape}")
                        print(f"   Number of leads: {ecg.shape[1]}")
                        print(f"   Sequence length: {ecg.shape[0]}")
                        print(f"   Sample frequency: {record.fs} Hz")
                        break
                    except Exception as e:
                        continue
            else:
                print(f"   ⚠ WARNING: Could not load any ECG file")
        except ImportError:
            print(f"   ⚠ WARNING: wfdb not installed (pip install wfdb)")
        print()
        
        print("=" * 80)
        print("✓ Data validation PASSED")
        print("=" * 80)
        print()
        print("You can now run training with:")
        print(f"  DATA_DIR={data_dir} ./run_train_vqvae.sh both")
        print()
        print("Or for quick testing:")
        print(f"  MAX_SAMPLES=100 DATA_DIR={data_dir} ./run_train_vqvae.sh both")
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MIMIC-IV-ECG data loading")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DATA_DIR", "/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"),
        help="Path to MIMIC-IV-ECG dataset"
    )
    
    args = parser.parse_args()
    
    success = test_data_path(args.data_dir)
    
    if not success:
        print()
        print("=" * 80)
        print("❌ Data validation FAILED")
        print("=" * 80)
        print()
        print("Please check:")
        print("1. Data directory path is correct")
        print("2. You have downloaded the complete MIMIC-IV-ECG dataset")
        print("3. The directory structure matches MIMIC-IV-ECG format")
        print()
        print("Expected structure:")
        print("  <data_dir>/")
        print("    ├── machine_measurements.csv")
        print("    └── files/")
        print("        └── pXX/")
        print("            └── pXXXXXXX/")
        print("                └── sXXXXXXXX/")
        print("                    ├── XXXXXXXX.hea")
        print("                    └── XXXXXXXX.dat")
        sys.exit(1)


if __name__ == "__main__":
    main()
