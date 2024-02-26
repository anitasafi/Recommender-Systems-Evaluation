import pandas as pd
import glob
import os

# Step 1: Collect TSV files
directory = "results/cat_dbpedia_movielens_1m/performance/"
pattern = os.path.join(directory, "rec_cutoff_10_relthreshold*.tsv")
file_names = glob.glob(pattern)

# Initialize an empty DataFrame to collect all data
all_data = pd.DataFrame()

for file_name in file_names:
    # Load the TSV file into a DataFrame
    df = pd.read_csv(file_name, sep='\t')
    
    # Step 2: Extract the model name
    df['model'] = df['model'].apply(lambda x: x.split('_')[0])
    
    # Append to the all_data DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)

# Step 3: Modify the column names
new_column_names = {
    'nDCG': 'nDCG↑', 'Recall': 'Recall↑', 'Precision': 'Precision↑', 
    'Gini': 'Gini↑', 'ItemCoverage': 'ItemCoverage↑', 'EPC': 'EPC↑', 
    'EFD': 'EFD↑', 'APLT': 'APLT↑', 'ARP': 'ARP↓'
}
all_data.rename(columns=new_column_names, inplace=True)

# Step 4 & 5: Save the modified DataFrame as a CSV file
output_file = "modified_performance_data.csv"
all_data.to_csv(output_file, index=False)

print(f"Data saved to {output_file}.")
