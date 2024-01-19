import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():

        df_norm_par10 = pd.read_csv("normalized_par10_results_over_folds.csv")


        data = {
            'scenario_name': [],
            'approach': [],
            'mean': [],
            'std': []
        }
        df = pd.DataFrame(data)

        table_data = {
            'scenario_name': [],
            'ours': []
        }
        table = pd.DataFrame(table_data)

        # get the min value of last column
        min_val = df_norm_par10.iloc[:,-1].min()
        print("min_val ", min_val)
        print(df_norm_par10)

        mean_over_folds = df_norm_par10.groupby(['scenario_name', 'approach'])['normalized_par_10'].mean().reset_index()
        std_over_folds = df_norm_par10.groupby(['scenario_name', 'approach'])['normalized_par_10'].std().reset_index()
        print("jo")

        mean_over_folds = mean_over_folds.pivot_table(index='scenario_name', columns='approach', values='normalized_par_10')
        mean_over_folds = mean_over_folds[['ours']]
        mean_over_folds = mean_over_folds.dropna()
        mean_over_folds = mean_over_folds.round(2)


        std_over_folds = std_over_folds.pivot_table(index='scenario_name', columns='approach', values='normalized_par_10')
        std_over_folds = std_over_folds[['ours']]
        std_over_folds = std_over_folds.dropna()
        std_over_folds = std_over_folds.round(2)

        print("mean_over_folds ", mean_over_folds)
        print("std_over_folds ", std_over_folds)


        ct = 0
        # iterate through rows in mean
        for index, row in mean_over_folds.iterrows():
            if ct <3:
                corresponding_std_row = std_over_folds.loc[index]

                ours_mean = row['ours']
                ours_std = corresponding_std_row['ours']

                scenario = index

                # append to df
                table = table.append({'scenario_name': scenario,  'ours': str(ours_mean) +"+-"+str(ours_std)}, ignore_index=True)

            ct+=1




        # Display the formatted DataFrame




        # Convert DataFrame to LaTeX table format
        table = table[['scenario_name', 'ours']]
        latex_table = table.to_latex(index=False)


        final_table = latex_table.replace('+-', '$\pm$')

        print(final_table)




if __name__ == "__main__":
    main()