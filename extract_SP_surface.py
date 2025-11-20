#===========================================
# Title: Extracting SP surfaces
# Author: Andrea Gondova (based on Fetal Brain Surface Extraction v1.1.0 by Jennings Zhang)
# Date: 16 August 2024
# Comment: ...
#===========================================
#. /neuro/users/mri.team/packages/env_MRI_team
### to go from .mnc to .nii for visualisation mnc2nii lh.innersp.mnc lh.innersp.nii (freesurfer activated)
### to go from .obj to .gii for f in *.obj; do /neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/processing/convert_obj2gii.sh ${f}; done;
### also had to install apt install parallel as a sudo user

import pandas as pd
import os
import argparse
import glob
import sys
import tempfile
import shutil
import subprocess

# SNAKEFILE=/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/SP_surface_extraction/Snakefile
SINGULARITIES_FOLDER = '/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/SP_surface_extraction/SP_singularities'


def run_command(cmd, step_name, log_file=None):
  """Run a command and check its exit code - EXIT IMMEDIATELY ON ANY ERROR
  Now captures stdout/stderr to log file"""
  print(f"\n{'='*60}")
  print(f"Running: {step_name}")
  print(f"Command: {cmd[:200]}..." if len(cmd) > 200 else f"Command: {cmd}")
  print(f"{'='*60}")
  
  if log_file:
    with open(log_file, 'a') as f:
      f.write(f"\n{'='*60}\n")
      f.write(f"Running: {step_name}\n")
      f.write(f"Command: {cmd}\n")
      f.write(f"{'='*60}\n")
  
  # Use subprocess to capture output
  try:
    result = subprocess.run(
      cmd,
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      executable='/bin/bash'
    )
    exit_code = result.returncode
    output = result.stdout
    
    # Print output to console
    if output:
      print(output)
    
    # Write output to log file
    if log_file and output:
      with open(log_file, 'a') as f:
        f.write(output)
        if not output.endswith('\n'):
          f.write('\n')
    
  except Exception as e:
    exit_code = 1
    output = f"Exception running command: {str(e)}"
    print(output)
    if log_file:
      with open(log_file, 'a') as f:
        f.write(output + '\n')
  
  if log_file:
    with open(log_file, 'a') as f:
      if exit_code != 0:
        f.write(f"FATAL ERROR: {step_name} failed with exit code {exit_code}\n")
        f.write(f"Pipeline terminated.\n")
      else:
        f.write(f"SUCCESS: {step_name} completed\n")
  
  if exit_code != 0:
    error_msg = f"FATAL ERROR: {step_name} failed with exit code {exit_code}"
    print(f"\n{'!'*60}")
    print(error_msg)
    print(f"{'!'*60}\n")
    # Ensure we exit with a non-zero code
    if exit_code == 0:
      exit_code = 1
    sys.exit(exit_code)
  
  print(f"SUCCESS: {step_name} completed")
  return exit_code


def main(args):
  subject_id = args.subject_id
  session_id = args.session_id
  path2segm = args.path2segm
  outdir = args.outdir
  estimate_cp = args.estimate_cp
  path2cp_d = args.path2cp_d
  convert = args.convert
  clean_up = args.clean_up
  log = args.log
  smooth_WM = args.smooth_WM

  # Create log file in output directory
  log_file = f"{outdir}/{subject_id}_{session_id}_pipeline.log"
  os.makedirs(outdir, exist_ok=True)
  
  # Create temporary directory in /tmp or outdir
  tmp_base = tempfile.mkdtemp(prefix=f'sp_pipeline_{subject_id}_{session_id}_', dir='/tmp')
  
  with open(log_file, 'w') as f:
    f.write(f"Pipeline log for {subject_id} {session_id}\n")
    f.write(f"Started at: {pd.Timestamp.now()}\n")
    f.write(f"Working directory: {os.getcwd()}\n")
    f.write(f"Output directory: {outdir}\n")
    f.write(f"Temp directory: {tmp_base}\n")
    f.write(f"="*60 + "\n")

  # Change to temp directory for processing
  original_dir = os.getcwd()
  os.chdir(tmp_base)
  
  try:
    # ========== PREPROCESSING ==========
    print(f"\nProcessing subject: {subject_id}, session: {session_id}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Output directory: {outdir}")
    print(f"Temp directory: {tmp_base}")
    print(f"Log file: {log_file}")
    
    run_command('mkdir -p tmp', "Step 1a: Create tmp directory", log_file)
    run_command('mkdir -p tmp/segmentations', "Step 1b: Create tmp/segmentations", log_file)
    run_command(f'cp {path2segm} tmp/segmentations', "Step 1c: Copy segmentation file", log_file)

    print(os.getcwd())
    # ========== Step2: Convert nifti to minc format ==========
    run_command(f'singularity run {SINGULARITIES_FOLDER}/pl-nii2mnc_1.1.0.sif niis2mncs --unsigned --byte tmp/segmentations tmp/minc', "Step 2: Convert nifti to minc", log_file)

    # ========== MASK EXTRACTION ==========
    # ========== Step3: WM masks ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-nums2mask_2.0.0.sif nums2mask --mask 'lh.wm.mnc:160,4 rh.wm.mnc:161,5' tmp/minc tmp/wm_mask", "Step 3: Extract WM masks", log_file)

    # ========== Step4: Inner SP masks ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-nums2mask_2.0.0.sif nums2mask --mask 'lh.innersp.mnc:160 rh.innersp.mnc:161' tmp/minc tmp/innersp_mask", "Step 4: Extract Inner SP masks", log_file)

    # ========== INNER SP SURFACE EXTRACTION ==========
    # ========== Step 5: wm surface, i.e. outer SP, i.e. CP surface ==========
    # creates within folder .../outersp_surface/{subject_id}_{session_id}_nuc_deep_subplate_dilate: lh.wm._{}.asc, rh.wm._{}.asc, lh.wm._{}.obj, rh.wm._{}.obj files
    # preferably should use the previously extracted CP masks if available,
    # the following steps use .obj files for surfaces and .mnc files for volumes
    # also creates smtherr.txt files // but do not need these, I compute error post-hoc

    if estimate_cp == 'no':
      message = 'WM from CP pipeline'
      # check cp surface files available // this is very stupid as it expects specific surface names!
      # should probs input surfaces / or look for those with different names and rename for next steps
      run_command('mkdir -p tmp/outersp_surface', "Step 5a: Create outersp_surface directory", log_file)
      for hemi in ['lh', 'rh']:
        if os.path.isfile(os.path.join(path2cp_d, f'{hemi}.wm.obj')):
          print(f'copying {hemi} wm mesh')
          run_command(f'cp {path2cp_d}/{hemi}.wm.obj tmp/outersp_surface/', f"Step 5b: Copy {hemi} WM mesh", log_file)
        else:
          print(f'ERROR: Tried to use previous WM mesh but {hemi}.wm.obj NOT FOUND')
          sys.exit(1)

    else:
      print('Computing new CP surface')
      message = 'WM from SP pipeline'
      run_command(f'singularity run {SINGULARITIES_FOLDER}/pl-fetal-surface-extract_2.1.1.sif extract_cp -J $(nproc) --target-smoothness 0.13 tmp/wm_mask tmp/outersp_surface', "Step 5: Extract CP surface", log_file)

    # ========== Step 6: Deform inner SP surface ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-bichamfer_1.0.1.sif bichamfer tmp/innersp_mask tmp/_innersp_chamfer", "Step 6: Compute chamfer distance", log_file)

    # ========== Step 7: Join chamfer and outer SP (== wm surface) ==========
    run_command(f"python {SINGULARITIES_FOLDER}/join_innersp_chamfer_and_outersp_surface.py tmp/_innersp_chamfer tmp/outersp_surface tmp/_inputs_for_innersp_surface", "Step 7: Join chamfer and outer SP", log_file)

    # ========== Step 8: Fit inner SP ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-gifit_0.3.0.sif gifit --threads $(nproc) tmp/_inputs_for_innersp_surface tmp/innersp_surface", "Step 8: Fit inner SP surface", log_file)

    # ========== Step 9: Medial cut surface mask registration to outer SP surface ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-bestsurfreg-surface-resample_1.0.0.sif bsrr -o '{{{{}}}}[email protected]' tmp/outersp_surface tmp/medial_cut_mask", "Step 9: Medial cut mask registration", log_file)

    # ========== Step 10: Get inner SP smoothness error  ==========
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-smoothness-error_2.0.2.sif smtherr tmp/innersp_surface tmp/innersp_smtherr", "Step 10: Compute smoothness error", log_file)

    # ========== Step 11: Join inner SP smoothness error with inner sp surfaces   ==========
    run_command(f"python {SINGULARITIES_FOLDER}/join_innersp_smtherr_and_surfaces.py tmp/innersp_surface tmp/innersp_smtherr tmp/_innersp_surface_and_smtherr", "Step 11: Join smoothness error with surfaces", log_file)

    # ========== Step 12: Invert medial cut mask   ==========
    print("\nStep 12: Inverting medial cut mask (complex command with parallel)")
    run_command(f"singularity run {SINGULARITIES_FOLDER}/pl-surfigures_1.2.0.sif sh -c 'yes 1 | head -n 40962 > /tmp/one.txt'", "Step 12a: Create ones file", log_file)
    run_command(f"mkdir -p tmp/_medial_cut_mask_neg", "Step 12b: Create output directory", log_file)
    # Run parallel command separately
    parallel_cmd = f"cd tmp/medial_cut_mask && find . -type f -name '*.medial_cut_mask.txt' | parallel --bar 'vertstats_math -old_style_file /tmp/one.txt -sub {{{{}}}} ../tmp/_medial_cut_mask_neg/{{{{.}}}}.keep.txt'"
    run_command(parallel_cmd, "Step 12c: Invert masks with parallel", log_file)

    # ========== Step 13: Mask inner SP data   ==========
    run_command(f"python {SINGULARITIES_FOLDER}/mask_innersp_data.py tmp/_innersp_surface_and_smtherr tmp/_medial_cut_mask_neg tmp/innersp_data_masked", "Step 13: Mask inner SP data", log_file)

    # ========== CONVERT, COPY, and CLEAN ==========
    # convert to .gii for easy visualisation - if convert yes
    print("\n" + "="*60)
    print("Final step: Converting and copying output files")
    print("="*60)
    
    for hemi in ['lh', 'rh']:
      print(f"\nProcessing {hemi} hemisphere...")
      
      innersp_obj = f'tmp/innersp_data_masked/{hemi}.innersp.obj'
      if not os.path.exists(innersp_obj):
        print(f"ERROR: Expected file not found: {innersp_obj}")
        sys.exit(1)
      
      run_command(f'/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/surface_processing/convert_obj2gii.sh {innersp_obj}', f"Convert {hemi} innersp to gii", log_file)
      run_command(f'cp tmp/innersp_data_masked/{hemi}.innersp.* {outdir}/', f"Copy {hemi} innersp files", log_file)

      if estimate_cp == 'yes':
        # this does not work because wrong filename!
        wm_obj = f'tmp/outersp_surface/{hemi}.wm._81920.obj'
        if os.path.exists(wm_obj):
          run_command(f'/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/surface_processing/convert_obj2gii.sh {wm_obj}', f"Convert {hemi} WM to gii", log_file)
        for file_name in glob.glob(f'tmp/outersp_surface/{hemi}.wm.*'):
          new_name = file_name.split('/')[-1].replace('_81920.', '.')
          run_command(f'cp {file_name} {outdir}/{new_name}', f"Copy {hemi} WM file: {new_name}", log_file)

    # ========= ADDITIONAL SMOOTHING for CP ==========
    # to approximate the CP extraction pipeline
    if smooth_WM == 'yes':
      print("\nApplying additional Taubin smoothing to WM surfaces...")
      for hemi in ['lh', 'rh']:
        iFile = f'{outdir}/{hemi}.wm.obj'
        oFile = f'{outdir}/{hemi}.wm.taubin100.obj'
        if os.path.exists(iFile):
          run_command(f'adapt_object_mesh {iFile} {oFile} 0 100 0 0', f"Smooth {hemi} WM surface", log_file)
          run_command(f'/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/surface_processing/convert_obj2gii.sh {oFile}', f"Convert {hemi} smoothed WM to gii", log_file)
        else:
          print(f"WARNING: {iFile} not found, skipping smoothing")    # clean up tmp subdirectory (inside temp directory)
    if clean_up == 'yes':
      print("\nCleaning up tmp directory...")
      run_command('rm -rvf tmp', "Clean up tmp directory", log_file)

    # Verify output files
    if log == 'yes':
      required_files = [f'{outdir}/lh.innersp.obj', f'{outdir}/rh.innersp.obj', 
                       f'{outdir}/lh.wm.obj', f'{outdir}/rh.wm.obj']
      missing = [f for f in required_files if not os.path.exists(f)]
      
      with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Pipeline completed at: {pd.Timestamp.now()}\n")
        if missing:
          f.write(f"FAILED: Missing files: {missing}\n")
        else:
          f.write(f"SUCCESS: All files created\n")
        f.write(f"{'='*60}\n")
      
      if missing:
        print(f"\n✗ FAILED: Missing {missing}")
        sys.exit(1)
    
  finally:
    # Clean up temp directory
    os.chdir(original_dir)
    if clean_up == 'yes':
      print(f"\nCleaning up temp directory: {tmp_base}")
      # TODO: ADD AGAIN, NOT CLEANING FOR TESTING POURPOSES
      # shutil.rmtree(tmp_base, ignore_errors=True)
    else:
      print(f"\nTemp directory preserved: {tmp_base}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser('   ==========   Subplate surface extraction   ==========   ')
  parser.add_argument('--subject_id', action='store', dest='subject_id', type=str, required=True, help='subject ID')
  parser.add_argument('--session_id', action='store', dest='session_id', type=str, required=True, help='session ID')
  parser.add_argument('--path2segm', action='store', dest='path2segm', type=str, required=True, help='path to SP segmentation file')
  parser.add_argument('--outdir', action='store', dest='outdir', type=str, required=True, help='output directory')
  parser.add_argument('--estimate_cp', action='store', dest='estimate_cp', type=str, default='no', help='whether to estimate new CP')
  parser.add_argument('--path2cp_d', action='store', dest='path2cp_d', type=str, help='path to folder with CP surfaces')
  parser.add_argument('--convert', action='store', dest='convert', type=str, default='yes', help='convert for visualisation with freeview')
  parser.add_argument('--clean_up', action='store', dest='clean_up', type=str, default='yes', help='clean up the temp files')
  parser.add_argument('--log', action='store', dest='log', type=str, default='yes', help='Keep track of subjects that were finished')
  parser.add_argument('--smooth_WM', action='store', dest='smooth_WM', type=str, default='yes', help='Whether to apply additional taubin smoothing (100 iter)')

  args = parser.parse_args()
  if args.estimate_cp == 'no' and args.path2cp_d is None:
    parser.error("if opting to use previous CP, --path2cp_d required")

  print(args)
  main(args)

