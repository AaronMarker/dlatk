import sys
import os
import numpy as np
import math
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score
import pandas as pd



#ToDo
#Make classification easy
#Make regression easy
#Make getting p_values easy
#Get scatterplots easy
#make tacking on gini coefficients and inverse parity ratio easy
#automate grabbing # of controls
#automate grabbing # of outcomes
#get spearman



BOOTSTRAP_COUNT = 1000
P_VALUE = False



NUM_OF_CONTROLS=1
NUM_OF_OUTCOMES=4
IS_REGRESSION = True
task = "Regression"#"Classification"
correlationsDict = {}
errorCorrelationsDict = {}
p_yPredsDict = defaultdict(dict)
p_yTruesDict = defaultdict(dict)
p_terciles = defaultdict(dict)
pValueDict = []
fig, axes = plt.subplots(NUM_OF_CONTROLS * NUM_OF_OUTCOMES, 6, figsize=(40, 60))
CSV_FILE = 'REG_CTLB_1grams_SingleControls.csv'#'REG_CTLB_1grams_predictionsFixed.csv'
#CSV_FILE = 'REG_CTLB_1grams_predictionsFixed.csv'
#CSV_FILE = 'REG_CTLB_1grams_SingleControls.csv'#'REG_CTLB_1grams_SingleControls.csv'#
#CSV_FILE = 'CLASS_DS4UD_Outcomes_SingleControlClass.csv'#'REG_DS4UD_5_Outcomes_SingleControlAgeFemale.csv'#REG_CTLB_1grams_predictionsFixed.csv'#REG_DS4UD_5_Outcomes_1Control_black.csv'
#CSV_FILE = 'REG_DS4UD_5_Outcomes_SingleControlAgeFemale.csv'

p_result_df = None



def merge_dicts(original_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in original_dict:
            # Merge recursively if both values are dictionaries
            merge_dicts(original_dict[key], value)
        else:
            # Otherwise, set or overwrite the value
            original_dict[key] = value



def main():

    if(IS_REGRESSION):
        task = "Regression"
    else:
        task = "Classification"
    
    dfAllRuns = readDataFromCSV(CSV_FILE)
    
    #correlationsDict["control"]=[]
    outcomes = set()
    
    fig.subplots_adjust(hspace=0.7, wspace=0.4)
    


    results_df = None
    if(P_VALUE):
        p_vals = calculatePValues()
    else:
        for dfApproach in dfAllRuns:
            approachName = dfApproach[0]
            approachData = dfApproach[1]
            
            for ctrlIdx, control in enumerate(approachData.columns[1:NUM_OF_CONTROLS+1]):
                splitDf, tercileLabeledDf = splitDataIntoTerciles(approachData, control)
                calculateSectionPValues(tercileLabeledDf, control, approachName)
                for lvlIdx, splitData in enumerate(splitDf):
                    sectionDf = processSection(splitData, lvlIdx, approachName, control, NUM_OF_CONTROLS)
                    
                    if results_df is None:
                        results_df = pd.DataFrame(columns=sectionDf.columns)
                    results_df = pd.concat([results_df, sectionDf], ignore_index=True)
                    
                    #SCATTERPLOT CODE
                    calculateErrorCorrelations(approachData, control, approachName, outcomes, ctrlIdx)
                    calculateCorrelations(approachData, control, approachName, outcomes)
                    #correlationsDict, outcomes, axes = runCorrelations(approachData, control, approachName, correlationsDict, outcomes, ctrlIdx, fig, axes)
                    #results_df, correlationsDict, axes = split_df_by_first_column(df[1], df[0], results_df, correlationsDict, fig, axes)
                    #correlationsDict = split_df_by_first_column(df[1], df[0], results_df, correlationsDict, fig, axes)
                #correlationsDict["control"].extend(outcomes)
                #correlationsDict["control"].extend([control])
    

    
    #Correlation with error code
    generateSubScatterPlot()
    global errorCorrelationsDict
    errorCorrelationsDf = pd.DataFrame(errorCorrelationsDict).T
    errorCorrelationsDf['index'] = errorCorrelationsDf.index
    errorCorrelationsDf[['Correlation', 'Test', 'Control', 'Outcome']] = errorCorrelationsDf['index'].apply(lambda x: pd.Series(x.split(',')))
    errorCorrelationsDf.drop(['index', 1, 2], axis=1, inplace=True)
    errorCorrelationsDf.reset_index(drop=True, inplace=True)
    errorCorrelationsDf = errorCorrelationsDf.pivot_table(
        index=['Correlation', 'Outcome', 'Control'],
        columns=['Test'],
        values=[0],
        aggfunc='first'
    )
    errorCorrelationsDf.to_csv('CorrelationsByMethodWithError.csv')



    #Correlations code
    generateSubScatterPlot()
    global correlationsDict
    correlationsDf = pd.DataFrame(correlationsDict).T
    correlationsDf['index'] = correlationsDf.index
    correlationsDf[['Correlation', 'Test', 'Control', 'Outcome']] = correlationsDf['index'].apply(lambda x: pd.Series(x.split(',')))
    correlationsDf.drop(['index', 1, 2], axis=1, inplace=True)
    correlationsDf.reset_index(drop=True, inplace=True)
    correlationsDf = correlationsDf.pivot_table(
        index=['Correlation', 'Outcome', 'Control'],
        columns=['Test'],
        values=[0],
        aggfunc='first'
    )
    correlationsDf.to_csv('CorrelationsByMethod.csv')



    if(IS_REGRESSION):
        custom_order = [task + " Lang Only Test", task +' Control Only Test', task + ' Test', 'Residualized Controls ' + task + ' Test', 'Factor Adaptation ' + task + ' Test', 'Residualized Factor Adaptation ' + task + ' Test']
    else:
        custom_order = [task + " Lang Only Test", task +' Control Only Test', task + ' Test', 'Factor Adaptation ' + task + ' Test']

    results_df['Test'] = pd.Categorical(results_df['Test'], categories=custom_order, ordered=True)
    
    plt.tight_layout()
    plt.savefig('scatterplot_grid.png')
    plt.close()
    #print("PfeIVOT2: ", results_df)
    #custom_order = ["LOW", 'MEDIUM', 'HIGH']
    #results_df['Name'] = pd.Categorical(results_df['Name'], categories=custom_order, ordered=True)
    print(results_df.columns)
    results_df = results_df.sort_values(by=['Control', 'Outcome', 'Test'])
    #print("PfeIVOT1: ", results_df)
    # Step 1: Pivot the dataframe
    if(IS_REGRESSION):
        pivoted_df = results_df.pivot_table(
            index=['Control', 'Outcome', 'Name'],
            columns=['Test'],
            values=['Length', 'MSE', 'R-squared', 'Pearson_r', 'Spearman r', 'Spearman p val'],
            aggfunc='first'
        )
    else:
        pivoted_df = results_df.pivot_table(
            index=['Control', 'Outcome', 'Name'],
            columns=['Test'],
            values=['Length', 'AUC', 'F1'],
            aggfunc='first'
        )

    #print(pivoted_df)
    pivoted_df = calculateDisparity(pivoted_df)
    pivoted_df = calculateGini(pivoted_df)

    pivoted_df.to_csv(CSV_FILE.replace(".csv", "") + 'disparity_cleaned.csv', index=True)
    #print("PfeIVOT: ", pivoted_df)



    #calculate p values for disparity
    global pValueDict
    global p_yPredsDict
    global p_yTruesDict
    global p_terciles
    for approach in p_yPredsDict.keys():
        for control in p_yPredsDict[approach].keys():
            for outcome in p_yPredsDict[approach][control].keys():
                old = p_yPredsDict['Regression Test lang'][control][outcome]
                
                p = andysBootstrap(p_yPredsDict[approach][control][outcome], old, p_yTruesDict[approach][control][outcome], p_terciles[approach][control][outcome])
                print('Regression Test lang' + 'VS ' + approach + ' ' + control + ' ' + outcome + ' :')
                print("P: ", p)
                pValueDict.append({
                    'p_value': p,
                    'approachName': approach,
                    'controlName': control,
                    'outcome': outcome
                })

    pValueDf = pd.DataFrame(pValueDict)
    pValueDf = pValueDf.pivot_table(
            index=['controlName', 'outcome'],
            columns=['approachName'],
            values=['p_value'],
            aggfunc='first'
        )
    pValueDf.to_csv('pValues.csv')
    print("DF: ", pValueDf)



def readDataFromCSV(csv_file):
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

    return dfs



def generateSubScatterPlot():
    # Assuming NUM_OF_CONTROLS * NUM_OF_OUTCOMES rows in the first column.
    fig, new_axes = plt.subplots(3, 4, figsize=(24, 12))

    # Loop through the first column of your original `axes` array to replot data in the new 3x4 grid
    for i, ax in enumerate(new_axes.flatten()):
        if i < NUM_OF_CONTROLS * NUM_OF_OUTCOMES:
            # Here, replicate the plot that was made in `axes[i, 0]`
            original_ax = axes[i, 0]

            # Copy line plots
            for line in original_ax.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                        color=line.get_color(), linestyle=line.get_linestyle(), linewidth=line.get_linewidth())

            # Copy scatter plots
            for path_collection in original_ax.collections:
                offsets = path_collection.get_offsets()
                sizes = path_collection.get_sizes()
                colors = path_collection.get_facecolors()
                
                # Plot the scatter points with retrieved properties
                ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, color=colors, label=path_collection.get_label())
                
            # Copy title, labels, legend, etc.
            ax.set_title(original_ax.get_title())
            ax.set_xlabel(original_ax.get_xlabel())
            ax.set_ylabel(original_ax.get_ylabel())
            ax.legend()

        else:
            ax.axis('off')  # Turn off any unused subplots

    plt.tight_layout()
    plt.savefig('subscatterplot_grid.png')
    plt.close()



def calculateSectionPValues(df, controlName, approachName):
    
    yPredDict, yTrueDict, tercile = andyBootstrapProcessSection(df, approachName)
    global p_yPredsDict
    global p_yTruesDict
    global p_terciles
    
    #Handle Regression's lang only and controls only
    for k, v in yPredDict.items():
        approachNameSplit = k.split(" ")
        if len(approachNameSplit) == 1:
            approachNameSplit.append("")
        else:
            approachNameSplit[1] = " " + approachNameSplit[1]
        p_yPredsDict[approachName + approachNameSplit[1]].setdefault(controlName, {})
        p_yTruesDict[approachName + approachNameSplit[1]].setdefault(controlName, {})
        p_terciles[approachName + approachNameSplit[1]].setdefault(controlName, {})
        p_yPredsDict[approachName + approachNameSplit[1]][controlName][approachNameSplit[0]] = yPredDict[k]
        p_yTruesDict[approachName + approachNameSplit[1]][controlName][approachNameSplit[0]] = yTrueDict[k]
        p_terciles[approachName + approachNameSplit[1]][controlName][approachNameSplit[0]] = tercile[k]
    
    pass



def calculatePValues(dfs):
    #correlationsDict["control"] = []
    ypreds = {}
    ytrues = {}
    terciles = {}
    for df in dfs:
        #results_df, correlationsDict, axes = split_df_by_first_column(df[1], df[0], results_df, correlationsDict, fig, axes)
        ypred, ytrue, tercile = split_df_by_first_column(df[1], df[0], results_df, correlationsDict, fig, axes)
        if df[0] in ypreds:
            ypreds[df[0]][list(ypred.keys())[0]] = ypred[list(ypred.keys())[0]]
        else:
            ypreds[df[0]] = ypred

        if df[0] in ytrues:
            ytrues[df[0]][list(ytrue.keys())[0]] = ytrue[list(ytrue.keys())[0]]
        else:
            ytrues[df[0]] = ytrue

        if df[0] in terciles:
            terciles[df[0]][list(tercile.keys())[0]] = tercile[list(tercile.keys())[0]]
        else:
            terciles[df[0]] = tercile


    #print("Tea: ", terciles['Residualized Factor Adaptation Regression Test'].keys())

    for approach in ypreds.keys():
        for control in ypreds[approach].keys():
            for outcome in ypreds[approach][control].keys():
                #print("Reg: ", ypreds['Regression Test'][control].keys())
                #print("TEST: ", ypreds[approach][control].keys())
                print("EX: ", approach, " : ", control, " : ", outcome)
                
                found_keys = [key for key in ypreds['Regression Test'][control].keys() if key.startswith(outcome[0:7]) and key.endswith('lang')]
                #print("KEY:! ", found_keys)
                old = ypreds['Regression Test'][control][found_keys[0]]
                print("EX: ", 'Regression Test', " : ", control, " : ", found_keys[0])
                
                #print("SO CLOSE: ", ypreds[approach].keys())
                print("P: ", andysBootstrap(ypreds[approach][control][outcome], old, ytrues[approach][control][outcome], terciles[approach][control][outcome]))

    return {}



def discrete_gini_coefficient(pearson_r, length):
    # Mean of the weighted values
    mean_value = np.average(pearson_r, weights=length)
    
    # Gini coefficient calculation
    n = len(pearson_r)
    total_weight = np.sum(length)
    
    gini_sum = 0
    for i in range(n):
        for j in range(n):
            gini_sum += length[i] * length[j] * abs(pearson_r[i] - pearson_r[j])
    
    gini = gini_sum / (2 * total_weight * np.sum(length) * mean_value)
    
    return gini



def calculateGini(df):
    # Grouping and calculating Gini coefficient
    result = []
    group_size = 3
    
    # Iterate over DataFrame in groups of 3 rows
    for start in range(0, len(df), group_size):

        end = start + group_size
        group = df.iloc[start:end]
        
        # Calculate the Gini coefficient for each subcolumn
        for col in group.columns.levels[1]:  # Loop over 'pearson_r' and 'length'

            if(IS_REGRESSION):
                pearson_values = group[('Pearson_r', col)].values
            else:
                pearson_values = group[('AUC', col)].values
                
            length_values = group[('Length', col)].values
            
            gini = discrete_gini_coefficient(pearson_values, length_values)
            new_col = ('Gini Coefficient', col)

            # Ensure that the new column is float and not a categorical index
            if new_col not in df.columns:
                df[new_col] = pd.Series(dtype='float')
            #print("INDEX: ", group)

            # Assign the Gini coefficient to the specified row (index 2)
            # Use loc since new_col is a MultiIndex (a tuple)
            #df[new_col] = pd.Series(dtype='float')
            first_row = group.iloc[0]
            first_index = first_row.name
            df.loc[(first_index[0], first_index[1], first_index[2]), new_col] = gini
            
    return df



def calculateDisparitySignificance(new_dist, old_dist):
    #determines the p-value of whether new has less disparity than old
    total = len(new_dist[0])
    score = 0
    for i in range(total):
        
        new_vals = [new_dist[0][i], new_dist[1][i], new_dist[2][i]]
        old_vals = [old_dist[0][i], old_dist[1][i], old_dist[2][i]]
        #print("THIS: ", new_dist)
        new_disp = 1 - (min(new_vals) / max(new_vals))
        old_disp = 1 - (min(old_vals) / max(old_vals))
        print("THIS and LANG: ", new_disp, " : ", old_disp)
        if new_disp < old_disp:
            score += 1



    p_val = score / total
    return p_val
    

def calculateDisparity(df):
    group_size = 3
    
    # Iterate over DataFrame in groups of 3 rows
    for start in range(0, len(df), group_size):

        end = start + group_size
        group = df.iloc[start:end]
        try:
            # Calculate the Gini coefficient for each subcolumn
            for col in group.columns.levels[1]:  # Loop over 'pearson_r' and 'length'

                if(IS_REGRESSION):
                    pearson_values = group[('Pearson_r', col)].values
                    this_dist = group[('dist', col)].values
                    lang_dist = group[('dist', task + " Lang Only Test")].values
                else:
                    pearson_values = group[('AUC', col)].values
                length_values = group[('Length', col)].values
                
                signif = calculateDisparitySignificance(this_dist, lang_dist)
                disp = 1 - (min(pearson_values) / max(pearson_values))
                new_col = ('Disparity', col)
                new_col2 = ('Disparity_Sig', col)

                # Ensure that the new column is float and not a categorical index
                if new_col not in df.columns:
                    df[new_col] = pd.Series(dtype='float')
                    df[new_col2] = pd.Series(dtype='float')

                first_row = group.iloc[0]
                first_index = first_row.name
                df.loc[(first_index[0], first_index[1], first_index[2]), new_col] = disp
                df.loc[(first_index[0], first_index[1], first_index[2]), new_col2] = signif
        except:
            pass
    '''if 'dist' in df.columns:
        df = df.drop('dist', axis=1)
    else:
        print("'dist' column not found.")'''

    return df



#["age_binarized", "is_female", "is_black"]
mask_functions = {
    'control_logincomeHC01_VC85ACS3yr': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'hsgrad': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'forgnborn': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'age': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
    'is_female': lambda df: df.map({1: 'Medium', 0: 'Low', None: 'High'}),
    'is_black': lambda df: df.map({1: 'Medium', 0: 'Low', None: 'High'}),
    'individual_income': lambda df: pd.cut(df, bins=[df.min()-1, df.quantile(1/3), df.quantile(2/3), df.max()], labels=['Low', 'Medium', 'High']),
}



def get_mask_function(control_name):
    for key in mask_functions:
        if key in control_name:  # Match based on control name
            return mask_functions[key]



names = [task + " Lang Only Test", task +' Control Only Test', task + ' Test', 'Residualized Controls ' + task + ' Test', 'Factor Adaptation ' + task + ' Test', 'Residualized Factor Adaptation ' + task + ' Test']
controlNames = {
    "control_hsgradHC03_VC93ACS3yr$10":"percent high school grad", 
    "control_forgnbornHC03_VC134ACS3yr$10":"percent foreign born", 
    "control_logincomeHC01_VC85ACS3yr$10":"log average income",
    "control_age": "age",
    "control_is_female":"is female", 
    "control_is_black": "is black"
    }


def _saveOutliers(df_clean, yPred, yTrue, name, control, yTrueCol):
    error_percentile_df = pd.DataFrame({
        'id': df_clean['Id'],
        'error': yPred - yTrue
    })
    threshold = 3 * error_percentile_df['error'].std()
    outliers = error_percentile_df.loc[abs(error_percentile_df['error'] - error_percentile_df['error'].mean()) > threshold, 'id']

    data_to_append = pd.DataFrame({
        'name': [name],
        'yTrueCol': [yTrueCol[:-6]],
        'control': [control],
        'outliers': [' '.join(outliers.astype(str))]  # Convert list to string for CSV
    })

    data_to_append.to_csv('ErrorVals.csv', mode='a' if os.path.exists('ErrorVals.csv') else 'w', header=not os.path.exists('ErrorVals.csv'), index=False)



def _approachErrorCorrelation(df_clean, yPred, yTrue, yTrueCol, ctrlIdx, name, control, outcomeIdx):
    controlVal = pd.to_numeric(df_clean[control], errors='coerce')

    error = abs(yTrue - yPred)
    correlation, p_value = pearsonr(error, controlVal)
    spearman_corr, spearman_p = spearmanr(error, controlVal)

    valueInformation = "," + name + "," + control + "," + yTrueCol[:-6]

    correlation_results = {
        "pearson" + valueInformation: str(correlation),
        "pearson_p" + valueInformation: p_value,
        "spearman" + valueInformation: spearman_corr,
        "spearman_p" + valueInformation: spearman_p
    }

    #Outlier Calculation
    _saveOutliers(df_clean, yPred, yTrue, name, control, yTrueCol)

    data = pd.DataFrame({
        'Error': error,
        'Control Value': controlVal
    })
    #print("AARONTEST: ", control, " : ", len(yPred))
    col = outcomeIdx + ctrlIdx * NUM_OF_OUTCOMES
    global axes
    sns.scatterplot(x='Control Value', y='Error', edgecolor='none', data=data, alpha=0.3, s=6, ax=axes[col, names.index(name)])
    sns.regplot(x='Control Value', y='Error', data=data, lowess=True, 
        scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ax=axes[col, names.index(name)])

    if name == "Regression Test":
        titleName = "Regression (Lang and Controls) Test"
    else:
        titleName = name
    
    axes[col, names.index(name)].set_title('Error predicting ' + yTrueCol[:-6].replace('_', ' ') + ' using \n' + titleName[:-5] + ' vs ' + controlNames[str(control)], 
                                    fontsize=14, fontweight='bold')
    axes[col, names.index(name)].set_ylabel('Error in predicting ' + yTrueCol[:-6].replace('_', ' '), fontsize=12, fontweight='bold')
    axes[col, names.index(name)].set_xlabel(controlNames[str(control)], fontsize=12, fontweight='bold')
    return correlation_results



def calculateErrorCorrelations(fullDf, control, name, outcomes, ctrlIdx):
    global errorCorrelationsDict
    for outcomeIdx, i in enumerate(range(NUM_OF_CONTROLS+1, len(fullDf.columns)-1, 4)):
        #df with controls and outcomes
        df = pd.concat([fullDf.iloc[:, 0:NUM_OF_CONTROLS+1], fullDf.iloc[:, i:i+4]], axis=1)

        yTrueCol = next(col for col in df.columns if col.endswith('trues'))
        df['yTrue'] = pd.to_numeric(df[yTrueCol], errors='coerce')
        df_clean = df.dropna(subset=['yTrue'])
        
        outcomes.add(yTrueCol)
        yTrue = df_clean['yTrue']
        #print("AARONTEST: ", len(df_clean), " : ", len(df))
        yPred = pd.to_numeric(df_clean.iloc[:, -2], errors='coerce')  # Last column in the DataFrame
        correlation_results = [_approachErrorCorrelation(df_clean, yPred, yTrue, yTrueCol, ctrlIdx, name, control, outcomeIdx)]

        if name == task + " Test":
            yPred = pd.to_numeric(df_clean.iloc[:, -5], errors='coerce')
            correlation_results.append(_approachErrorCorrelation(df_clean, yPred, yTrue, yTrueCol, ctrlIdx, "Regression Lang Only Test", control, outcomeIdx))

        for correlation_result in correlation_results:
            for key in correlation_result:
                #print("Value: ", correlation_result[key])
                #print("LEY: ", key)
                errorCorrelationsDict.setdefault(key, []).extend([correlation_result[key]])
    return outcomes



def _approachCorrelation(yPred, yTrue, yTrueCol, name, control):

    correlation, p_value = pearsonr(yPred, yTrue)
    spearman_corr, spearman_p = spearmanr(yPred, yTrue)
    mse = mean_squared_error(yTrue, yPred)
    mae = mean_absolute_error(yTrue, yPred)

    valueInformation = "," + name + "," + control + "," + yTrueCol[:-6]

    correlation_results = {
        "pearson" + valueInformation: str(correlation),
        "pearson_p" + valueInformation: p_value,
        "spearman" + valueInformation: spearman_corr,
        "spearman_p" + valueInformation: spearman_p,
        "MSE" + valueInformation: mse,
        "MAE" + valueInformation: mae
    }

    return correlation_results



def calculateCorrelations(fullDf, control, name, outcomes):
    global correlationsDict
    for outcomeIdx, i in enumerate(range(NUM_OF_CONTROLS+1, len(fullDf.columns)-1, 4)):
        #df with controls and outcomes
        df = pd.concat([fullDf.iloc[:, 0:NUM_OF_CONTROLS+1], fullDf.iloc[:, i:i+4]], axis=1)

        yTrueCol = next(col for col in df.columns if col.endswith('trues'))
        df['yTrue'] = pd.to_numeric(df[yTrueCol], errors='coerce')
        df_clean = df.dropna(subset=['yTrue'])
        
        outcomes.add(yTrueCol)
        yTrue = df_clean['yTrue']
        #print("AARONTEST: ", len(df_clean), " : ", len(df))
        yPred = pd.to_numeric(df_clean.iloc[:, -2], errors='coerce')  # Last column in the DataFrame
        correlation_results = [_approachCorrelation(yPred, yTrue, yTrueCol, name, control)]

        if name == task + " Test":
            yPred = pd.to_numeric(df_clean.iloc[:, -5], errors='coerce')
            correlation_results.append(_approachCorrelation(yPred, yTrue, yTrueCol, "Regression Lang Only Test", control))

        for correlation_result in correlation_results:
            for key in correlation_result:
                #print("Value: ", correlation_result[key])
                #print("LEY: ", key)
                correlationsDict.setdefault(key, []).extend([correlation_result[key]])



def runCorrelations(fullDf, control, name, correlationsDict, outcomes, idx, fig, axes):
    #print("NAME: ", name, ", CONTROL: ", control)
    correlations = []
    pValues = []
    correlations_lang = []
    pValues_lang = []
    spearmans = []
    spearmanP = []
    spearmans_lang = []
    spearmanP_lang = []

    outcome = 0
    for i in range(NUM_OF_CONTROLS+1, len(fullDf.columns)-1, 4):
        
        df = pd.concat([fullDf.iloc[:, 0:NUM_OF_CONTROLS+1], fullDf.iloc[:, i:i+4]], axis=1)
        
        yTrueCol = [col for col in df.columns if col.endswith('trues')]
        df['yTrue'] = pd.to_numeric(df[yTrueCol[0]], errors='coerce')

        # Drop rows where yTrue is null and keep the original DataFrame structure
        df_clean = df.dropna(subset=['yTrue'])
        yTrue = df_clean['yTrue']
        outcomes.add(yTrueCol[0])
        yPred = pd.to_numeric(df_clean.iloc[:, -2], errors='coerce')  # Last column in the DataFrame
        controlVal = pd.to_numeric(df_clean[control], errors='coerce')

        error = abs(yTrue - yPred)
        #print("TEST: ", df.columns)
        correlation, p_value = pearsonr(error, controlVal)
        spearman_corr, spearman_p = spearmanr(error, controlVal)
        
        #print("Outcome: ", yTrueCol[0])
        correlations.append(correlation)#"" + str(correlation) + yTrueCol[0] + control)
        pValues.append(p_value)
        spearmans.append(spearman_corr)
        spearmanP.append(spearman_p)

        error_percentile_df = pd.DataFrame({
            'id': df_clean['Id'],
            'error': yPred - yTrue
        })
        mean_err = error_percentile_df['error'].mean()
        std_err = error_percentile_df['error'].std()
        threshold = 3 * std_err
        outliers_df = error_percentile_df[abs(error_percentile_df['error'] - mean_err) > threshold]
        #print(name, ": ", yTrueCol[0][:-6]," : ", control, " : ", outliers_df['id'].tolist())
        data_to_append = pd.DataFrame({
            'name': [name],
            'yTrueCol': [yTrueCol[0][:-6]],
            'control': [control],
            'outliers': [str(' '.join(outliers_df['id'].tolist()))]  # Convert list to string for CSV
        })

        # Append the data to the existing CSV or create the file if it doesn't exist
        csv_file = 'ErrorVals.csv'  # Specify your CSV file name
        if os.path.exists(csv_file):
            # Append to the file if it exists
            data_to_append.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            # Create the file and write the data with the header if it doesn't exist
            data_to_append.to_csv(csv_file, mode='w', header=True, index=False)

        if name == task + " Test":
            yPred = pd.to_numeric(df_clean.iloc[:, -5], errors='coerce')  # Last column in the DataFrame
            #print("TESTS: ", df_clean.columns[-5])
            errorLang = abs(yTrue - yPred)
            
            #print("TEST: ", df.columns)
            #print("Error: ", df_clean.columns)
            correlation_lang, p_value_lang = pearsonr(errorLang, controlVal)
            spearman_corr_lang, spearman_p_lang = spearmanr(errorLang, controlVal)
            #print("ERRORS: ", correlation)
            #print("Outcome: ", yTrueCol[0])
            correlations_lang.append(correlation_lang)#"" + str(correlation) + yTrueCol[0] + control)
            #print("Pearson correlation: %f" % correlation)
            pValues_lang.append(p_value_lang)
            spearmans_lang.append(spearman_corr_lang)
            spearmanP_lang.append(spearman_p_lang)



            error_percentile_df = pd.DataFrame({
                'id': df_clean['Id'],
                'error': yPred - yTrue
            })
            mean_err = error_percentile_df['error'].mean()
            std_err = error_percentile_df['error'].std()
            threshold = 3 * std_err
            outliers_df = error_percentile_df[abs(error_percentile_df['error'] - mean_err) > threshold]
            #print(name, ": ", yTrueCol[0][:-6]," : ", control, " : ", outliers_df['id'].tolist())
            data_to_append = pd.DataFrame({
                'name': [task + " Lang Only Test"],
                'yTrueCol': [yTrueCol[0][:-6]],
                'control': [control],
                'outliers': [str(' '.join(outliers_df['id'].tolist()))]  # Convert list to string for CSV
            })

            # Append the data to the existing CSV or create the file if it doesn't exist
            csv_file = 'ErrorVals.csv'  # Specify your CSV file name
            if os.path.exists(csv_file):
                # Append to the file if it exists
                data_to_append.to_csv(csv_file, mode='a', header=False, index=False)
            else:
                # Create the file and write the data with the header if it doesn't exist
                data_to_append.to_csv(csv_file, mode='w', header=True, index=False)



            data = pd.DataFrame({
                'Error': errorLang,
                'Control Value': controlVal
            })
            col = outcome + idx * NUM_OF_OUTCOMES
            sns.scatterplot(x='Control Value', y='Error', edgecolor='none', data=data, alpha=0.3, s=6, ax=axes[col, names.index(task + " Lang Only Test")])
            sns.regplot(x='Control Value', y='Error', data=data, lowess=True, 
                scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ax=axes[col, names.index(task + " Lang Only Test")])

            axes[col, names.index(task + " Lang Only Test")].set_title('(r = %.2f) ' % correlation_lang + 'Error predicting ' + yTrueCol[0][:-6].replace('_', ' ') + ' using \n' + "Regression (Lang Only)" + ' vs ' + controlNames[str(control)], 
                                        fontsize=14, fontweight='bold')
            axes[col, names.index(task + " Lang Only Test")].set_ylabel('Error in predicting ' + yTrueCol[0][:-6].replace('_', ' '), fontsize=12, fontweight='bold')
            axes[col, names.index(task + " Lang Only Test")].set_xlabel(controlNames[str(control)], fontsize=12, fontweight='bold')

        
        # Create the scatterplot using Seaborn
        #plt.figure(figsize=(8, 6))
        data = pd.DataFrame({
            'Error': error,
            'Control Value': controlVal
        })
        col = outcome + idx * NUM_OF_OUTCOMES
        sns.scatterplot(x='Control Value', y='Error', edgecolor='none', data=data, alpha=0.3, s=6, ax=axes[col, names.index(name)])
        sns.regplot(x='Control Value', y='Error', data=data, lowess=True, 
            scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ax=axes[col, names.index(name)])

        if name == "Regression Test":
            titleName = "Regression (Lang and Controls) Test"
        else:
            titleName = name
        axes[col, names.index(name)].set_title('(r = %.2f) ' % correlation + 'Error predicting ' + yTrueCol[0][:-6].replace('_', ' ') + ' using \n' + titleName[:-5] + ' vs ' + controlNames[str(control)], 
                                       fontsize=14, fontweight='bold')
        axes[col, names.index(name)].set_ylabel('Error in predicting ' + yTrueCol[0][:-6].replace('_', ' '), fontsize=12, fontweight='bold')
        axes[col, names.index(name)].set_xlabel(controlNames[str(control)], fontsize=12, fontweight='bold')
        outcome += 1
        # Save the plot to a file
        #plt.savefig('scatterplot_' + name + '_' + control + '_' + yTrueCol[0][:-6] + '.png')

    if ("cor_" + name) not in correlationsDict.keys():
        correlationsDict["cor_" + name] = correlations
        correlationsDict["p_" + name] = pValues
        correlationsDict["spe_" + name] = spearmans
        correlationsDict["spep_" + name] = spearmanP
    else:            
        correlationsDict["cor_" + name].extend(correlations)
        correlationsDict["p_" + name].extend(pValues)
        correlationsDict["spe_" + name].extend(spearmans)
        correlationsDict["spep_" + name].extend(spearmanP)

    if len(correlations_lang) > 0:
        if ("cor_" + "Regression Lang Only Test") not in correlationsDict.keys():
            correlationsDict["cor_" + "Regression Lang Only Test"] = correlations_lang
            correlationsDict["p_" + "Regression Lang Only Test"] = pValues_lang
            correlationsDict["spe_" + "Regression Lang Only Test"] = spearmans_lang
            correlationsDict["spep_" + "Regression Lang Only Test"] = spearmanP_lang
        else:           
            correlationsDict["cor_" + "Regression Lang Only Test"].extend(correlations_lang)
            correlationsDict["p_" + "Regression Lang Only Test"].extend(pValues_lang)
            correlationsDict["spe_" + "Regression Lang Only Test"].extend(spearmans_lang)
            correlationsDict["spep_" + "Regression Lang Only Test"].extend(spearmanP_lang)
    
    #print("DF", correlationsDict)
    return correlationsDict, outcomes, axes



def splitDataIntoTerciles(df, control):
    mask_func = get_mask_function(control)
    masks = mask_func(pd.to_numeric(df[control], errors='coerce'))

    splitDfs = [df[masks == level] for level in ['Low', 'Medium']]

    if 'High' in masks.unique():
        splitDfs.append(df[masks == 'High'])

    # Create labeledDf by adding a new 'category' column to the original df
    labeledDf = df.copy()
    labeledDf['Category'] = masks  # Assign the masks as the category column
    
    return splitDfs, labeledDf


def split_df_by_first_column(df, name, results_df, correlationsDict, fig, axes):
    # Assuming the first column is the one to check
    # Create boolean masks for values equal to 1 and 0
    #print("NAME: ", name)
    yPredsDict = {}
    yTruesDict = {}
    terciles = {}
    outcomes = set()
    for idx, control in enumerate(df.columns[1:NUM_OF_CONTROLS+1]):
        #mask_1 = df[control] == "1"
        #mask_0 = df[control] == "0"
        #try:
        mask_func = get_mask_function(control)
        
        masks = mask_func(pd.to_numeric(df[control], errors='coerce'))

        #SCATTERPLOT CODE
        #correlationsDict, outcomes, axes = runCorrelations(df, control, name, correlationsDict, outcomes, idx, fig, axes)



        # Split the DataFrame based on the masks
        df_0 = df[masks == 'Low']  # DataFrame where the first column equals 1
        df_1 = df[masks == 'Medium']  # DataFrame where the first column equals 0
        try:
            df_2 = df[masks == 'High']
            if(not P_VALUE):
                section_df_2 = processSection(df_2, "HIGH", name, control, NUM_OF_CONTROLS)
        except:
            pass
        if(not P_VALUE):
            section_df_1 = processSection(df_1, "MEDIUM", name, control, NUM_OF_CONTROLS)
            section_df_0 = processSection(df_0, "LOW", name, control, NUM_OF_CONTROLS)
        if(P_VALUE):
            df['Category'] = masks
            yPredDict, yTrueDict, tercile = andyBootstrapProcessSection(df, name)
            yPredsDict[control] = yPredDict
            yTruesDict[control] = yTrueDict
            terciles[control] = tercile

        if(not P_VALUE):
            # If results_df does not exist, initialize it
            if results_df is None:
                results_df = pd.DataFrame(columns=section_df_1.columns)  # Initialize with the same columns as combined_df
            try:
                results_df = pd.concat([results_df, section_df_2], ignore_index=True)
            except:
                pass
            results_df = pd.concat([results_df, section_df_1], ignore_index=True)
            results_df = pd.concat([results_df, section_df_0], ignore_index=True)
        
        #correlationsDict["control"].extend(outcomes)
        #correlationsDict["control"].extend([control])
        #print("TEST: ", len(correlationsDict["control"]))
    if(not P_VALUE):
        return results_df, correlationsDict, axes
    return yPredsDict, yTruesDict, terciles


def bootstrapSample(yTrue, yPred):
    # Ensure ypred and ytrue have the same length
    assert len(yPred) == len(yTrue), "ypred and ytrue must have the same length"
    n = len(yPred)  # Number of observations
    # Generate random indices with replacement
    indices = np.random.choice(n, size=n, replace=True)
    # Resample ypred and ytrue using the random indices
    ypred_resampled = yPred.iloc[indices]
    ytrue_resampled = yTrue.iloc[indices]
    #print("YPRED: ", ypred_resampled)
    #print("YTRUE: ", ytrue_resampled)
    return ytrue_resampled, ypred_resampled



def andyBootstrapProcessSection(fullDf, test):

    yPreds = {}
    yTrues = {}
    terciles = {}

    for i in range(NUM_OF_CONTROLS+1, len(fullDf.columns)-1, 4):
        df = fullDf.iloc[:, i:i+4]
        yTrueCol = [col for col in df.columns if col.endswith('trues')]
        df_clean = df.dropna(subset=[yTrueCol[0], df.columns[2], df.columns[0], df.columns[-1]])

        yTrue = pd.to_numeric(df_clean[yTrueCol[0]], errors='coerce').dropna()  # Assuming you want the first matching column
        yPred = pd.to_numeric(df_clean.iloc[:, -1], errors='coerce').dropna()  # Last column in the DataFrame

        print("GRABBING: ", df_clean.columns[-1])
        colName = df_clean.columns[0].split("__")[0]
        yTrues[colName] = yTrue
        yPreds[colName] = yPred
        terciles[colName] = fullDf['Category']

        

        if test == task + " Test":
            #Lang
            df_clean = df.dropna(subset=[yTrueCol[0], df.columns[0]])
            yPred = pd.to_numeric(df_clean.iloc[:, 0], errors='coerce').dropna()
            print("GRABBING Lang: ", df_clean.columns[0])
            yTrues[colName + " lang"] = yTrue
            yPreds[colName + " lang"] = yPred
            terciles[colName + " lang"] = fullDf['Category']

            #Controls
            df_clean = df.dropna(subset=[yTrueCol[0], df.columns[0]])
            yPred = pd.to_numeric(df_clean.iloc[:, -2], errors='coerce').dropna()
            print("GRABBING Cont: ", df_clean.columns[-2])
            yTrues[colName + " control"] = yTrue
            yPreds[colName + " control"] = yPred
            terciles[colName + " control"] = fullDf['Category']

    
    return yPreds, yTrues, terciles
    



def processSection(fullDf, name, test, control, allControls):

    section_df = pd.DataFrame(columns=['Test', 'Name', 'Outcome', 'Control', 'AllControls', 'Length', 'MSE', 'R-squared', 'Pearson_r', 'Spearman r', 'Spearman p val'])

    for i in range(NUM_OF_CONTROLS+1, len(fullDf.columns)-1, 4):
        
        df = fullDf.iloc[:, i:i+4]
        yTrueCol = [col for col in df.columns if col.endswith('trues')]
        
        df_clean = df.dropna(subset=[yTrueCol[0], df.columns[2], df.columns[0], df.columns[-1]])
        yTrue = pd.to_numeric(df_clean[yTrueCol[0]], errors='coerce').dropna()  # Assuming you want the first matching column
        yPred = pd.to_numeric(df_clean.iloc[:, -1], errors='coerce').dropna()  # Last column in the DataFrame

        if "Residual" in test:
            controlPred = pd.to_numeric(df_clean.iloc[:, 2], errors='coerce').dropna()
            #yPred = pd.to_numeric(df_clean.iloc[:, 0], errors='coerce').dropna()
            yTrue = yTrue + controlPred
            yPred = yPred + controlPred
        
        if(IS_REGRESSION):
            #print("P" * 800)
            mse = mean_squared_error(yTrue, yPred)
            r_squared = r2_score(yTrue, yPred)
            pearson_corr, _ = pearsonr(yTrue, yPred)
            spearman_corr, spearman_p = spearmanr(yTrue, yPred)
            # dist = []
            # print("TEST TYPE: ", test)
            # for i in range(BOOTSTRAP_COUNT):
            #     ytrue_resampled, ypred_resampled = bootstrapSample(yTrue, yPred)
            #     cor, _ = pearsonr(ytrue_resampled, ypred_resampled)
            #     dist.append(cor)
            # Create a new row with the results
            new_row = {
                'Test': test,
                'Name': name,
                'Outcome': yTrueCol[0][:-6],
                'Control': control,
                'AllControls': allControls,
                'Length': len(df_clean),
                'MSE': mse,
                'R-squared': r_squared,
                'Pearson_r': pearson_corr,
                'Spearman r': spearman_corr,
                'Spearman p val': spearman_p
                # 'dist': dist
            }
        else:
            auc = roc_auc_score(yTrue, yPred)
            yPred_labels = [1 if prob >= 0.5 else 0 for prob in yPred]
            f1_val = f1_score(yTrue, yPred_labels)
            # Create a new row with the results
            new_row = {
                'Test': test,
                'Name': name,
                'Outcome': yTrueCol[0][:-6],
                'Control': control,
                'AllControls': allControls,
                'Length': len(df_clean),
                'AUC': auc,
                'F1': f1_val
            }

        # Append the new row to the existing DataFrame
        section_df = section_df.append(new_row, ignore_index=True)

        if test == task + " Test":
            df_clean = df.dropna(subset=[yTrueCol[0], df.columns[2], df.columns[0], df.columns[-1]])
            yTrue = pd.to_numeric(df_clean[yTrueCol[0]], errors='coerce').dropna()  # Assuming you want the first matching column
            yPred = pd.to_numeric(df_clean.iloc[:, -1], errors='coerce').dropna() 
            # df_clean = df.dropna(subset=[yTrueCol[0], df.columns[0]])
            # yPred = pd.to_numeric(df_clean.iloc[:, 0], errors='coerce').dropna()  # Last column in the DataFrame
            # dist = []
            # for i in range(BOOTSTRAP_COUNT):
            #     ytrue_resampled, ypred_resampled = bootstrapSample(yTrue, yPred)
            #     cor, _ = pearsonr(ytrue_resampled, ypred_resampled)
            #     dist.append(cor)
            if(IS_REGRESSION):
                mse = mean_squared_error(yTrue, yPred)
                r_squared = r2_score(yTrue, yPred)
                pearson_corr, _ = pearsonr(yTrue, yPred)
                spearman_corr, spearman_p = spearmanr(yTrue, yPred)
                # Create a new row with the results
                new_row = {
                    'Test': task + " Lang Only Test",
                    'Name': name,
                    'Outcome': yTrueCol[0][:-6],
                    'Control': control,
                    'AllControls': allControls,
                    'Length': len(df_clean),
                    'MSE': mse,
                    'R-squared': r_squared,
                    'Pearson_r': pearson_corr,
                    'Spearman r': spearman_corr,
                    'Spearman p val': spearman_p
                    # 'dist': dist
                }
            else:
                auc = roc_auc_score(yTrue, yPred)
                yPred_labels = [1 if prob >= 0.5 else 0 for prob in yPred]
                f1_val = f1_score(yTrue, yPred_labels)
                # Create a new row with the results
                new_row = {
                    'Test': task + " Lang Only Test",
                    'Name': name,
                    'Outcome': yTrueCol[0][:-6],
                    'Control': control,
                    'AllControls': allControls,
                    'Length': len(df_clean),
                    'AUC': auc,
                    'F1': f1_val
                }

            # Append the new row to the existing DataFrame
            section_df = section_df.append(new_row, ignore_index=True)

            yPred = pd.to_numeric(df_clean.iloc[:, -2], errors='coerce').dropna()  # Last column in the DataFrame
            
            if(IS_REGRESSION):
                mse = mean_squared_error(yTrue, yPred)
                r_squared = r2_score(yTrue, yPred)
                pearson_corr, _ = pearsonr(yTrue, yPred)
                spearman_corr, spearman_p = spearmanr(yTrue, yPred)
                # dist = []
                # for i in range(BOOTSTRAP_COUNT):
                #     ytrue_resampled, ypred_resampled = bootstrapSample(yTrue, yPred)
                #     cor, _ = pearsonr(ytrue_resampled, ypred_resampled)
                #     dist.append(cor)
                # Create a new row with the results
                new_row = {
                    'Test': task +' Control Only Test',
                    'Name': name,
                    'Outcome': yTrueCol[0][:-6],
                    'Control': control,
                    'AllControls': allControls,
                    'Length': len(df_clean),
                    'MSE': mse,
                    'R-squared': r_squared,
                    'Pearson_r': pearson_corr,
                    'Spearman r': spearman_corr,
                    'Spearman p val': spearman_p
                    # 'dist': dist
                }
            else:
                auc = roc_auc_score(yTrue, yPred)
                yPred_labels = [1 if prob >= 0.5 else 0 for prob in yPred]
                f1_val = f1_score(yTrue, yPred_labels)
                # Create a new row with the results
                new_row = {
                    'Test': task +' Control Only Test',
                    'Name': name,
                    'Outcome': yTrueCol[0][:-6],
                    'Control': control,
                    'AllControls': allControls,
                    'Length': len(df_clean),
                    'AUC': auc,
                    'F1': f1_val
                }

            # Append the new row to the existing DataFrame
            section_df = section_df.append(new_row, ignore_index=True)
            #print("COLS: ", section_df)



    #print("TEST2: ", section_df)
    return section_df


def andysBootstrap(new_ypreds, old_ypreds, ytrues, tercile_ids, num_resamples = BOOTSTRAP_COUNT):
    #new_ypreds -- the predictions model being tested for less disparity than the old
    #tercile_id --
    #print("TESTS: ", new_ypreds[:5])
    #print("TESTS: ", old_ypreds[:5])

    
    n = len(new_ypreds)
    count_null_trials = 0
    disp_new = andysCalcMinMaxDisparity(new_ypreds.reset_index(drop=True).tolist(), ytrues.reset_index(drop=True).tolist(), tercile_ids.reset_index(drop=True).tolist())
    for k in range(num_resamples):
        indices = np.random.choice(n, size=n, replace=True)
        bs_new_ypreds = new_ypreds.reset_index(drop=True).loc[indices].tolist() 

        bs_old_ypreds = old_ypreds.reset_index(drop=True).loc[indices].tolist() 
        bs_ytrues = ytrues.reset_index(drop=True).loc[indices].tolist() 
        bs_tercile_ids = tercile_ids.reset_index(drop=True).loc[indices].tolist() 
        #print("TESTP: ", any(math.isnan(x) for x in bs_new_ypreds))
        #print("TESTT: ", bs_ytrues[:5])
        #print("TESTING")
        
        disp_old = andysCalcMinMaxDisparity(bs_old_ypreds, bs_ytrues, bs_tercile_ids)
        if disp_old <= disp_new:
            count_null_trials += 1

    return count_null_trials / num_resamples












def andysCalcMinMaxDisparity(ypreds, ytrues, tercile_ids, internal_metric = pearsonr, disp_metric=lambda x: 1-minMaxRatio(x)):
    terc_dict = {
        "High": 2,
        "Medium": 1,
        "Low": 0
    }
    #print("TESTSETE: ", tercile_ids[tercile_ids.keys()[0]])
    #first max three terciles:
    terc_yp = [[], [], []]
    terc_ytrues = [[], [], []]
    for i in range(len(ypreds)):
        terc = terc_dict[tercile_ids[i]]
        terc_yp[terc].append(ypreds[i])
        terc_ytrues[terc].append(ytrues[i])

    rs = []
    for t in range(3):#calculate teh score per tercile:
        '''
        print("PREDS: ", len(terc_yp[2]))
        print("PREDS: ", len(terc_ytrues[1]))
        
        print("trues: ", type(terc_ytrues[t]))
        print("PREDS: ", len(terc_yp[t][6]))
        print("trues: ", len(terc_ytrues[t]))
        print("PREDS: ", type(terc_yp[t][0][0]))
        #print("trues: ", terc_ytrues[t][0])'''
        cor, _ = internal_metric(terc_yp[t], terc_ytrues[t])
        rs.append(cor)
    #print("Rs: ", disp_metric(rs))
    #print("Rs: ", terc_yp[t])

    return disp_metric(rs)
        





def minMaxRatio(rs):
    return np.min(rs) / np.max(rs)
    
    
    
    


if __name__ == "__main__":

    main()


