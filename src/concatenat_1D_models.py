import os
import pandas as pd

def concatenate_csv_files(folder_path, output_file):
    # Get a list of all CSV files in the specified folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Check if there are any CSV files in the folder
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Create an empty DataFrame to store concatenated data
    concatenated_data = pd.DataFrame()

    # Iterate through each CSV file and concatenate them vertically
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

    target = concatenated_data.drop(concatenated_data.columns[0], axis=1)
    # Save the concatenated data to a new CSV file
    target.to_csv(output_file, index=False)
    print(f"Concatenated data saved to {output_file}")
    return 

# Example usage:
folder_path = './'
output_file = 'target_training.csv'
concatenate_csv_files(folder_path, output_file)
