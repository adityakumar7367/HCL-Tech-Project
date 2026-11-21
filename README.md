# Wine Dataset Exploratory Data Analysis (EDA)

**Student:** Aditya Kumar  
**Admission Number:** 2026B01011203  
**Roll Number:** 2026B01011203

## Objective
Perform basic descriptive EDA on the UCI Wine Recognition Dataset and answer six assignment questions. This project does not use machine learning.

## Dataset
- Official source: https://archive.ics.uci.edu/dataset/109/wine
- Raw files: `data/raw/wine.data` and `data/raw/wine.names`
- The data contains 178 records, 1 class label, and 13 measured attributes.
- The original class labels 1, 2, and 3 are preserved.

## Tools
Python, NumPy, Pandas, and Matplotlib. Jupyter is included for opening the notebook.

## Project Structure
```text
Wine_Dataset_EDA_Aditya_Kumar/
  data/raw/
  data/processed/
  images/
  notebooks/
  scripts/
  reports/
  README.md
  requirements.txt
  .gitignore
```

## Questions Answered
1. Record counts for each class.
2. Mean, median, minimum, and maximum Alcohol values.
3. Classes with the highest average Alcohol and Magnesium.
4. Ten records with the highest Color intensity.
5. Missing values and duplicate rows.
6. Average Alcohol, Malic acid, and Color intensity across classes.

## Setup
From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Command Prompt setup:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

General terminal setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the Analysis
Windows PowerShell, Command Prompt, or another terminal:

```text
python scripts/analyze_wine.py
```

The script can also be launched from outside the project root because it uses paths relative to the project file.

## Open the Notebook
```text
jupyter notebook notebooks/wine_dataset_eda.ipynb
```

## Input Handling
`wine.data` and `wine.names` are copied unchanged into `data/raw/`. If `Wine_EDA_Input_Only.zip` is supplied, extract only those original source files into `data/raw/`. Generated notebooks, scripts, charts, reports, and processed CSV files remain in this project and are never added to the input-only archive.
