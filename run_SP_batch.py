#!. /neuro/users/mri.team/packages/env_MRI_team

#=============================================================
# Title: Subplate Surface extraction over a batch of subjects
# Author: Andrea Gondova 
# Contact: andrea.gondova@childrens.harvard.edu
#=============================================================

import pandas as pd
import os
import argparse
from multiprocessing import Pool, cpu_count
import subprocess

subjects_base_path = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"

def process_split(split_data):
  """Process a single split - to be run in parallel"""
  subject_id, session_id, split, args = split_data
  
  print(f"▶ Starting: {subject_id}_{session_id}_{split}", flush=True)
  
  subject_path = os.path.join(subjects_base_path, subject_id, session_id)
  segm_dir = os.path.join(subject_path, split, 'anat/segmentations')
  path2segm = os.path.join(segm_dir, f'{subject_id}_{session_id}_nuc_deep_subplate_dilate_mc.nii')
  
  if not os.path.exists(path2segm):
    return {
      'subject_id': subject_id,
      'session_id': session_id,
      'split': split,
      'reason': 'Data not found'
    }
  
  outdir = os.path.join(subject_path, split, "default_surfaces")
  os.makedirs(outdir, exist_ok=True)
  
  if args.force_rerun:
    os.system(f"rm -rf {outdir}/*")
  
  # Log file for this specific split
  split_dir = os.path.join(subject_path, split)
  error_log = os.path.join(split_dir, f"{subject_id}_{session_id}_{split}_error.log")
  
  # Run extraction pipeline, redirect all output to error log
  cmd = [
    'python3', '/neuro/users/yair.beltran/Reliability/extract_SP_surface.py',
    '--subject_id', subject_id,
    '--session_id', session_id,
    '--path2segm', path2segm,
    '--outdir', outdir,
    '--estimate_cp', 'yes',
    '--path2cp_d', outdir,
    '--convert', args.convert,
    '--clean_up', args.clean_up,
    '--log', 'no',  # Disable SP_log.txt
    '--smooth_WM', args.smooth_WM
  ]
  
  with open(error_log, 'w') as log_file:
    result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
  
  if result.returncode != 0:
    # Read last 500 chars of error log to show what failed
    with open(error_log, 'r') as f:
      error_content = f.read()
      error_snippet = error_content[-500:] if len(error_content) > 500 else error_content
    
    return {
      'subject_id': subject_id,
      'session_id': session_id,
      'split': split,
      'reason': f'See log: {error_log}',
      'error_snippet': error_snippet
    }
  
  return None


if __name__ == '__main__':
  parser = argparse.ArgumentParser('   ==========   Subplate surface extraction batch   ==========   ')
  parser.add_argument('--subjects', required=True, help='Path to .csv file with subjects')
  parser.add_argument('--convert', default='yes', help='convert for visualisation')
  parser.add_argument('--clean_up', default='yes', help='clean up temp files')
  parser.add_argument('--smooth_WM', default='yes', help='apply Taubin smoothing')
  parser.add_argument('--max_workers', type=int, help='Max parallel workers (default: all CPUs)')
  parser.add_argument('--force_rerun', action='store_true', help='Force rerun')
  
  args = parser.parse_args()
  
  # Determine number of workers
  max_workers = args.max_workers if args.max_workers else cpu_count()
  print(f"Using {max_workers} parallel workers (available CPUs: {cpu_count()})")
  
  subjects = pd.read_csv(args.subjects)
  
  # Build flat list of all jobs
  all_jobs = []
  for _, row in subjects.iterrows():
    subject_id, session_id = str(row.subject_id), str(row.session_id)
    for split in ["S1", "S2", "S3", "S4"]:
      all_jobs.append((subject_id, session_id, split, args))
  
  print(f"Total jobs: {len(all_jobs)}")
  print(f"{'='*80}\n")
  
  # Process all jobs in parallel with progress tracking
  failed_subjects = []
  completed = 0
  
  with Pool(processes=max_workers) as pool:
    for result in pool.imap_unordered(process_split, all_jobs):
      completed += 1
      if result is not None:
        failed_subjects.append(result)
        print(f"[{completed}/{len(all_jobs)}] FAILED: {result['subject_id']}_{result['session_id']}_{result['split']}")
        print(f"  Error: {result['reason']}")
        if 'error_snippet' in result:
          print(f"  Last output:\n{result['error_snippet']}\n")
      else:
        print(f"[{completed}/{len(all_jobs)}] ✓ Success")
  
  # Save failures
  if failed_subjects:
    pd.DataFrame(failed_subjects).to_csv('failed.csv', index=False)
    print(f"\n{len(failed_subjects)} jobs failed. Details in failed.csv")
  else:
    print("\nAll jobs completed successfully!")

