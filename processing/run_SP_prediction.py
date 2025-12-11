import argparse
import pandas as pd 
import os
from pathlib import Path


#### PLEASE ACTIVATE THE APPROPRIATE PYTHON ENVIRONMENT BEFORE PREDICTING SUBPLATE
# source ~/tools/miniconda3/bin/activate /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/
# does not work on el-jobo!

if __name__ == '__main__':
    parser = argparse.ArgumentParser('   ==========   Surface to Volume texture projection   ==========   ')
    parser.add_argument('--subjects', action='store', dest='subjects', type=str, required=True, help='Path to .csv file with subjects')
    args = parser.parse_args()
    subjects = pd.read_csv(args.subjects)

    # Track failed subjects
    failed_subjects = []

    ### Copying placental reconstructions
    '''
  iDir='/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/Data/Placenta_protocol'

  for i, row in subjects.iterrows():	
  print(row.Path)
  iRECON=os.path.join(iDir, row.Path, 'recon_segmentation/recon_to31_nuc.nii')
  #print(iRECON)
  oDir = os.path.join('/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/DerivedData/subjects', row.Path)

  if os.path.isdir(oDir):
  os.system(f'rm -r {oDir}')
  os.system(f'mkdir -p {oDir}')
  #print(oDir)
  oVol = os.path.join(oDir, f"{row.Path.split('/')[0]}_{row.Path.split('/')[1]}_recon_to31_nuc.nii")
  os.system(f'cp {iRECON} {oVol}')
  if not os.path.isfile(oVol):
  print(oVol, 'PROBLEM')
  #	print('\n')
  '''	

    # iDIR='/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/DerivedData/subjects'
    iDIR='/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST'
    script='/neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/predict_sp.py'
    weights='/neuro/labs/grantlab/research/MRI_processing/milton.candela/fetal_subplate/models/C120' 

    ### PREDICTING SP
    for i, row in subjects.iterrows():
        try:
            subject_path = os.path.join(str(row.subject_id), str(row.session_id))
            for split_name in ["S1", "S2", "S3", "S4"]:
                split_path = os.path.join(subject_path, split_name)
                iRECON = os.path.join(
                    iDIR, split_path, "recon_segmentation", "recon_to31_nuc.nii"
                )

                if not os.path.exists(iRECON):
                    raise FileNotFoundError(f"Input not found: {iRECON}")

                oDIR = os.path.join(iDIR, split_path, "segmentations")
                os.system(f'mkdir -p {oDIR}')

                tmpRECON = os.path.join(oDIR, "recon_to31_nuc_deep_agg.nii.gz")
                oRECON = os.path.join(oDIR, f"{row.subject_id}_{row.session_id}_nuc_deep_subplate_dilate_mc.nii")

                cmd = f'python3 {script} -input {iRECON} -output {oDIR} -weights {weights} -gpu 0'
                if os.system(cmd) != 0:
                    raise RuntimeError("Prediction failed")

                os.system(f'gunzip -c {tmpRECON} > {oRECON} && rm {tmpRECON}')
                print(f"SUCCESS: {oRECON}")
        except Exception as e:
            failed_subjects.append({'subject_id': row.subject_id, 'session_id': row.session_id, 'error': str(e)})
            print(f"FAILED: {row.subject_id}/{row.session_id} - {e}")

    if failed_subjects:
        pd.DataFrame(failed_subjects).to_csv('failed.csv', index=False)


## python3 /neuro/labs/grantlab/research/MRI_processing/milton.candela/highres_subplate/predict_sp.py -input /neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2024/NewModel_test/Alignment/allData/FCB094/2016.10.04-026Y-MR_EI_Fetal_Neuro-30615/recon_segmentation/recon_to31_nuc.nii -output . --weights_loc /neuro/labs/grantlab/research/MRI_processing/milton.candela/fetal_subplate/models/C120 -gpu 0
