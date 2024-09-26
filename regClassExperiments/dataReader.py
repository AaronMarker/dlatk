import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for headless environments
import matplotlib.pyplot as plt
import re


FILTER_OUT_CONTROLS_ONLY=True
FILTER_OUT_LANGUAGE_ONLY=True
INCLUDE_VALUES = True
AVERAGE_METHODS = False
COL = 'r'



def parse_csv(file_path):
    """
    Parse the CSV file with repeated sections. Each section has:
    - A title row
    - Column header row
    - A series of data rows
    """
    sections = []
    
    # Use pandas to read the entire file without manually splitting lines
    # Read the CSV file with no specific constraints on columns
    
    # Assume 47 is the maximum number of columns found in the file
    column_names = ['col ' + str(i) for i in range(48)]
    
    # Read the CSV with predefined column names
    data = pd.read_csv(file_path, header=None, skip_blank_lines=True, names=column_names, engine='python')

    #data = pd.read_csv(file_path, header=None, error_bad_lines=False, warn_bad_lines=True, skip_blank_lines=True, engine='python')
    #print("DATA: ", data.iloc[3].name)
    i = 0
    while i < len(data):
        #print("ROW: ", data.iloc[i])
        #if pd.isna(data.iloc[i, 0]):  # Skip empty rows
        #    i += 1
        #    continue
        title = data.iloc[i][0].strip()  # First row of section is the title
        outcomes = data.iloc[i][1].strip()  # First row of section is the title
        column_headers = list(data.iloc[i+1])  # Second row is the column names
        data_rows = []
        i += 2
        #print("COLUMNNAMES: ", column_headers)
        #print("TITLE: ", title)
        #print("COLS: ", column_headers)
        # Collect data rows until we hit a new section or end of file
        #print("TEST", data.iloc[i][0])
        while i < len(data) and not data.iloc[i][0].startswith('row') and data.iloc[i][0].isdigit():  # Assuming empty row separates sections
            #print("TEST: ", data.iloc[i].name[0])
            data_rows.append(list(data.iloc[i]))
            #print("COLUMNNAMES: ", list(data.iloc[i]))
            i += 1

        # Convert the data into a DataFrame
        df = pd.DataFrame(data_rows, columns=column_headers)
        #print("DATAFRAME: ", df)
        df['Title'] = title  # Add the title as a column for easy reference
        df['allOutcomes'] = outcomes
        sections.append(df)
    
    return sections

def plot_all_sections(sections):
    """
    For each section in the CSV file, plot the 'r' column from all sections on a single graph if it exists.
    """
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to combine all data
    
    for idx, df in enumerate(sections):

        if FILTER_OUT_CONTROLS_ONLY:
            df = df[df['model_controls'].str[-1] == '1']

        if FILTER_OUT_LANGUAGE_ONLY:
            df = df[~(df['model_controls'].str[-3] == '(')]

        #print("\nRows where the last character is '0':")
        #print(df[df['model_controls'].str[-1] != '0'])

        title = df['Title'].iloc[0]  # Get the section title
        #print("TESTING", title)
        outcomes = df['allOutcomes'].iloc[0]
        chars_to_remove = "[]'"
        outcomes = ''.join(c for c in outcomes if c not in chars_to_remove)

        #outcomes.replace(']', '')
        #outcomes.replace('\'', '')

        

        # Check if the column 'r' exists in the DataFrame
        if COL in df.columns:
            df_r = df[[COL]]  # Select only the 'r' column
            # Convert the 'r' column to numeric, ignore errors
            df_r[COL] = pd.to_numeric(df_r[COL], errors='coerce')
            
            # Drop rows where 'r' is NaN
            df_r = df_r.dropna()
            
            
            if not df_r.empty:
                # Add a column for the title to identify each section
                df_r['TestType'] = ''.join(word[:3] for word in title.split()[:-1] if word) 
                df_r['Title'] = '_'.join(max((sub.strip() for sub in word.split("_") if sub), key=len, default='')[:3] for word in outcomes.split(",") if word)
                
                df_r['outcome'] = df['outcome']
                df_r['controls'] = df['model_controls']
                df_r['N'] = df['test_size'].astype(int) + df['train_size'].astype(int)
                df_r['num_features'] = df['num_features']
                df_r['label'] = df_r['Title'] + "_" + df_r['controls'].astype(str).apply(lambda x: '_'.join(word[:5] for word in re.sub(r'["\'()_]', '', x).split(",") if word))
                combined_df = pd.concat([combined_df, df_r], ignore_index=True)
                print(df_r)
                
            else:
                print("No valid data in column for section: {}".format(title))

        else:
            print("Column not found in section: {}".format(title))



    combined_df.to_csv('clean_' + file_path, index=False)
    if AVERAGE_METHODS:
        combined_df = combined_df.groupby('TestType', as_index=False).mean()
        combined_df['Title'] = ""



    if not combined_df.empty:

        manual_colors = {
            'Reg': '#8D99AE',         # Example color for 'Reg'
            'ResConReg': '#F5B841',   # Example color for 'ResConReg'
            'FacAdaReg': '#931621',   # Example color for 'FacAdaReg'
            'ResFacAdaReg': '#2E294E' # Example color for 'ResFacAdaReg'
        }

        custom_order = ['Reg', 'ResConReg', 'FacAdaReg', 'ResFacAdaReg']
        combined_df['TestType'] = pd.Categorical(combined_df['TestType'], categories=custom_order, ordered=True)
        try:
            combined_df = combined_df.sort_values(by=['outcome', 'TestType'])
        except:
             combined_df = combined_df.sort_values(by=['TestType'], ascending=True)
        num_columns = len(combined_df['TestType'].unique())
        figure_width = max(10, num_columns * 0.5)  # Adjust the scaling factor as needed
        #''.join(word[:3] for word in combined_df['Title'].split()[:-1] if word)

        colors_for_plot = combined_df['TestType'].map(manual_colors)
        unique_titles = combined_df['TestType'].unique()
        color_map = plt.get_cmap('tab20')  # You can choose a different colormap
        colors = {title: color_map(i / len(unique_titles)) for i, title in enumerate(unique_titles)}
       


        ax = combined_df.plot(kind='bar', x='outcome', y=COL, stacked=False, color=colors_for_plot)#[colors[title] for title in combined_df['TestType']])
        
        # Create a custom legend with labels for each color
        handles = [plt.Line2D([0], [0], color=manual_colors[title], lw=4) for title in unique_titles]
        ax.legend(handles, unique_titles, title='Sections', bbox_to_anchor=(1, 1), loc='upper left')
        if INCLUDE_VALUES:
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, '{:.3f}'.format(height), ha='center', va='bottom')
        
        plt.gcf().set_size_inches(figure_width, 6)
        plt.ylim(.1, .9)
        plt.title('')
        plt.ylabel('Value')
        plt.xlabel('Section')
        plt.xticks(rotation=90)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        # Save the combined plot to a file
        plt.savefig('combined_plot.png')

        plt.close()  # Close the plot to free memory

        # Save the combined DataFrame to a CSV file
        
    else:
        print("No valid data found to plot.")



def unique_values_ordered(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



# File path to your CSV
file_path = 'original_printDS4UD_Tests9Reg_NO_extra.csv'#original_printPostStratFactorAdaptTests3.csv'

# Parse the CSV file into sections
sections = parse_csv(file_path)

# Plot all sections
plot_all_sections(sections)














