## Environment setup
```bash
source ~/miniconda3/bin/activate /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/
```

### Alternative environment
```bash
micromamba activate -p /neuro/labs/grantlab/research/MRI_processing/environment
```

## Base path
```bash
/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/
```

## Prediction
```bash
run_SP_predicton.py --subjects [subject.csv]
```

# How to run

**Single subject:**
- Simply change the parameters in cell 2 of the notebook and run all cells.

**Multiple subjects:**
- Make sure to be in `/neuro/users/yair,beltran/Reliability`
- Select correct subject.csv file in `batch_process.py`
- Run the following command:
```bash
python batch_process.py
```
**Note:** Apart from the pdf reports and the .csv cache this will generate a copy of the notebook for each subject in the `outputs` folder.



