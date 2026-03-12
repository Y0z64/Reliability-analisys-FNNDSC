#!/usr/bin/env python3
"""
Batch process all subjects from subject.csv through the Reliability notebook.
Each subject gets their own notebook execution with 2 pages added to the PDF.
"""

import pandas as pd
import papermill as pm
from pathlib import Path

# Configuration
NOTEBOOK_PATH = "Reliability.ipynb"
CSV_PATH = "../data/subject.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Read subjects
subjects_df = pd.read_csv(CSV_PATH)
print(f"Found {len(subjects_df)} subjects to process\n")

# Process each subject
for idx, row in subjects_df.iterrows():
    subject_id = str(row['subject_id'])
    session_id = str(row['session_id'])
    
    print(f"\n{'='*80}")
    print(f"[{idx+1}/{len(subjects_df)}] Processing: {subject_id}")
    print(f"{'='*80}")
    
    # Output notebook for this subject (for debugging if needed)
    output_notebook = OUTPUT_DIR / f"{subject_id}_{session_id}.ipynb"
    
    try:
        # Run the notebook with parameters
        pm.execute_notebook(
            NOTEBOOK_PATH,
            str(output_notebook),
            parameters={
                'subject_id': subject_id,
                'session_id': session_id,
                'pdf_output_path': f"../reports/{subject_id}_{session_id}_reliability_report.pdf" #Overwrite pdf path with new parameters
            },
            kernel_name='python3'
        )
        print(f"✓ Completed: {subject_id}")
        
    except Exception as e:
        print(f"✗ ERROR: {subject_id}")
        print(f"   {str(e)}")
        continue

print(f"\n{'='*80}")
print("BATCH PROCESSING COMPLETE")
print("PDF report: reliability_report.pdf")
print("CSV metrics: cross_split_metrics.csv")
print(f"Individual notebooks saved to: {OUTPUT_DIR}/")
print(f"{'='*80}\n")
