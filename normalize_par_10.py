import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mysql.connector

def main():


        data = {
            'scenario_name': [],
            'fold': [],
            'approach': [],
            'normalized_par_10': []
        }
        df_norm_par10 = pd.DataFrame(data)


        par10_df = pd.read_csv("results/r2ss_par_10.csv")

        ct = 0



        vbs_sbs_df = pd.read_csv('results/sbs_vbs_results.csv')

        for index, row in par10_df.iterrows():
            scenario = row['scenario_name']
            fold = row['fold']
            approach = "ours"
            par10 = row['result']

            # Filter vbs_sbs_df based on conditions
            sbs_row = vbs_sbs_df[(vbs_sbs_df['fold'] == fold) &
                                 (vbs_sbs_df['scenario_name'] == scenario) &
                                 (vbs_sbs_df['metric'] == 'par10') &
                                 (vbs_sbs_df['approach'] == 'sbs')]

            vbs_row = vbs_sbs_df[(vbs_sbs_df['fold'] == fold) &
                                 (vbs_sbs_df['scenario_name'] == scenario) &
                                 (vbs_sbs_df['metric'] == 'par10') &
                                 (vbs_sbs_df['approach'] == 'oracle')]

            if not sbs_row.empty and not vbs_row.empty:
                sbs = sbs_row.iloc[0]['result']
                vbs = vbs_row.iloc[0]['result']


                normalized = (par10 - vbs) / (sbs - vbs)
                print("par10 ", par10, "sbs ", sbs, "vbs ", vbs, "normalized ", normalized)
                if par10 == float(-1):
                    print("jo")
                    normalized = 1
                    ct+=1
                print("normalized ", normalized)
                # Append to the DataFrame
                df_norm_par10 = df_norm_par10.append(
                    {'scenario_name': scenario, 'fold': fold, 'approach': approach, 'normalized_par_10': normalized},
                    ignore_index=True)
            else:
                print("emptyyyy")
                print("scenario ", scenario, "fold ", fold, "approach ", approach)


        aslibstrings = ["GRAPHS-2015", "ASP-POTASSCO", "BNSL-2016",
                        "CPMP-2015", "CSP-2010", "CSP-Minizinc-Time-2016", "CSP-MZN-2013",
                        "MAXSAT-PMS-2016", "MAXSAT-WPMS-2016", "MAXSAT12-PMS", "MAXSAT19-UCMS", "QBF-2011", "QBF-2014",
                        "QBF-2016", "SAT03-16_INDU",
                        "SAT11-HAND", "SAT11-INDU", "SAT11-RAND", "SAT12-ALL", "SAT12-HAND", "SAT12-INDU", "SAT12-RAND",
                        "SAT15-INDU",
                        "SAT16-MAIN", "SAT18-EXP", "TSP-LION2015", "PROTEUS-2014", "MAXSAT15-PMS-INDU",
                        "MIP-2016", "GLUHACK-2018", "SAT20-MAIN", "TTP-2016", "OPENML-WEKA-2017"]

        print("LEN ", len(aslibstrings))
        # fill empty entries with 0

        for string in aslibstrings:
            for fold in range(1, 11):
                filtered_df = df_norm_par10[(df_norm_par10['scenario_name'] == string) &
                                            (df_norm_par10['fold'] == float(fold)) &
                                            (df_norm_par10['approach'] == "ours")].iloc[:,:].values
                print("approach" , approach, "filtered_df ", float(fold), "scen name ", string)


                #print("filtered_df ", filtered_df)
                if len(filtered_df) == 0:
                    df_norm_par10 = df_norm_par10.append(
                        {'scenario_name': string, 'fold': fold, 'approach': approach, 'normalized_par_10': np.inf},
                        ignore_index=True)


        print("NORM PAR 10")
        print(df_norm_par10)

        df_norm_par10.to_csv('normalized_par10_results_over_folds.csv', index=False)


if __name__ == "__main__":
    main()