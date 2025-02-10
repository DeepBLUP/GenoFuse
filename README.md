1. Data Preparation
   The 012 file(bed, fam, bim) with quality control and gene imputation has been completed. Execute the plink command to generate a new file(raw):
     plink --bfile testdate --recodeA --out testdate
   Run the raw_to_csv.py to convert the raw file into a CSV file, and fill the phenotype values into the sixth column "PHENOTYPE" based on individual IDs. The data preparation is complete, and the new file example is as follows:
   ![image](https://github.com/user-attachments/assets/b4702895-3b6a-4aca-927a-aecfdebfe2ed)
2. Modify the input file path
   Modify the two file paths in GenoFuse.py.
   ![image](https://github.com/user-attachments/assets/e7f36ad9-8b8e-4378-81e8-f511c5a9d9c8)
3. Run GenoFuse.py
     python GenoFuse.py

