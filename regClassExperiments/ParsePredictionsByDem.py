import sys
import os

# Add the parent directory (dlatk/) to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)

from dlatk.featureGetter import FeatureGetter
from dlatk.database.dataEngine import DataEngine
from dlatk.regressionPredictor import RegressionPredictor, ClassifyPredictor
from dlatk.outcomeGetter import OutcomeGetter
import dlatk.dlaConstants as dlac
import DLATKTests
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import csv
import subprocess
import multiprocessing
from abc import ABC, abstractmethod

import pandas as pd





NUM_OF_CONTROLS=3




def main():

    

    # Define the path to your CSV file
    csv_file = 'REG_PostStratFactorAdaptTests3.csv'

    # Read the CSV file in chunks to handle multiple headers
    dfs = []
    with open(csv_file, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    current_header = None
    current_test = None
    data = []

    for line in lines:
        # Check if the line contains 'Regression Test' indicating a new header
        
        if not line[0].isdigit():
            if(not data and not current_test):
                print("*" * 80)
                current_test = line.split(",")[0]
            #print("LINE: ", line.split(",")[0])
            if current_header and data:
                #print("TEST: ", current_header)
                # Create a DataFrame for the previous header and data
                df = pd.DataFrame(data, columns=current_header)
                dfs.append((current_test, df))
                data = []  # Reset data
                #print("\n")
                #print(current_test, "=============================")
                #print("===========================================")
                #split_df_by_first_column(df)
                current_test = line.split(",")[0]

            # Parse the new header
            parts = line.strip().split(',')
            current_header = parts[0:]  # Ignore 'Regression Test' part

            # Clean up header elements (removing extra quotes and brackets)
            current_header = [
                eval(col) if col.startswith('[') and col.endswith(']') else col.strip('"') 
                for col in current_header
            ]
            current_header = [item for sublist in current_header for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Check if the line contains data (assuming data starts with an integer ID)
        elif line.strip() and line[0].isdigit():
            data.append(line.strip().split(','))

    # Append the last DataFrame
    if current_header and data:
        df = pd.DataFrame(data, columns=current_header)
        dfs.append((current_test, df))

    # Combine all DataFrames into a single DataFrame
    #final_df = pd.concat(dfs, ignore_index=True)

    # Convert appropriate columns to numeric
    #final_df = final_df.apply(pd.to_numeric, errors='ignore')

    # Print the resulting DataFrame
    results_df = None
    for df in dfs:
        results_df = split_df_by_first_column(df[1], df[0], results_df)
    print("RESULTS: ", results_df.columns)
    custom_order = ['Regression Test', 'Residualized Controls Regression Test', 'Factor Adaptation Regression Test', 'Residualized Factor Adaptation Regression Test']
    results_df['Test'] = pd.Categorical(results_df['Test'], categories=custom_order, ordered=True)

    results_df = results_df.sort_values(by=['AllControls', 'Control', 'Test'])
    
    results_df.to_csv(csv_file.replace(".csv", "") + 'disparity_cleaned.csv', index=False)


mask_functions = {
    #'age': lambda df: (df == "1", df == "0"),
    'control_logincomeHC01_VC85ACS3yr': lambda df: (df > df.median(), df <= df.median()),
    'hsgrad': lambda df: (df > df.median(), df <= df.median()),
    'forgnborn': lambda df: (df > df.median(), df <= df.median())
}


def get_mask_function(control_name):
    for key in mask_functions:
        if key in control_name:  # Match based on control name
            return mask_functions[key]



def split_df_by_first_column(df, name, results_df):
    # Assuming the first column is the one to check
    # Create boolean masks for values equal to 1 and 0
    #print("NAME: ", name)
    for control in df.columns[1:NUM_OF_CONTROLS + 1]:
        #print("CONT: ", control)
        #mask_1 = df[control] == "1"
        #mask_0 = df[control] == "0"
        try:
            mask_func = get_mask_function(control)
            mask_1, mask_0 = mask_func(pd.to_numeric(df[control], errors='coerce'))
            #print("DATAFRAME: ", df.iloc[:, 1] == 1)
            
            

            # Split the DataFrame based on the masks
            df_1 = df[mask_1]  # DataFrame where the first column equals 1
            df_0 = df[mask_0]  # DataFrame where the first column equals 0
            #print("DATAFRAME: ", df_1)
            #print("DATAFRAME: ", df_0)
            section_df_1 = processSection(df_1, "ONE", name, control, str(df.columns[1:-4]))
            
            section_df_0 = processSection(df_0, "ZERO", name, control, str(df.columns[1:-4]))
            section_df_1_modified = section_df_1.drop(columns=['Name'])
            section_df_1_modified.columns = [col + '_low' if col not in ['Test', 'Control', 'AllControls'] 
                                            else col for col in section_df_1_modified.columns]

            # Select relevant columns from section_df_0 and rename them
            section_df_0_modified = section_df_0.drop(columns=['Name'])
            section_df_0_modified.columns = [col + '_high' if col not in ['Test', 'Control', 'AllControls'] 
                                            else col for col in section_df_0_modified.columns]
    
            combined_df = pd.concat([section_df_1_modified.reset_index(drop=True), 
                          section_df_0_modified.reset_index(drop=True)], axis=1)
            #print("COLS: ", combined_df.columns)
        except:
            print("WARNING: out of controls")

        # If results_df does not exist, initialize it
        if results_df is None:
            results_df = pd.DataFrame(columns=combined_df.columns)  # Initialize with the same columns as combined_df

        results_df = pd.concat([results_df, combined_df], ignore_index=True)
        
    return results_df



def processSection(df, name, test, control, allControls):

    #print(df.columns[1], ": ", name)
    #print("LENGTH: ", len(df))
    
    yTrueCol = [col for col in df.columns if col.endswith('trues')]
    df_clean = df.dropna(subset=[yTrueCol[0], df.columns[-1]])
    yTrue = pd.to_numeric(df_clean[yTrueCol[0]], errors='coerce')  # Assuming you want the first matching column
    yPred = pd.to_numeric(df_clean.iloc[:, -1], errors='coerce')  # Last column in the DataFrame

    #print("DF: ", df_clean)
    #print("YTREU: ", len(yTrue))
    #print("YFalse: ", len(yPred))

    # Calculate Mean Squared Error
    mse = mean_squared_error(yTrue, yPred)
    #print("Mean Squared Error (MSE): %f" % mse)

    # Calculate R-squared value
    r_squared = r2_score(yTrue, yPred)
    #print("R-squared (R²): %f" % r_squared)

    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(yTrue, yPred)
    #print("Pearson correlation coefficient (r): %f" % pearson_corr)


    # Create a new row with the results
    new_row = {
        'Test': test,
        'Name': name,
        'Control': control,
        'AllControls': allControls,
        'Length': len(df_clean),
        'MSE': mse,
        'R-squared': r_squared,
        'Pearson_r': pearson_corr
    }

    

    # Append the new row to the existing DataFrame
    section_df = pd.DataFrame([new_row])
    return section_df


if __name__ == "__main__":

    main()

