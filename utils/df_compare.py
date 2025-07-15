import pandas as pd

default_file = pd.read_csv("data/GTSRB/predictions.csv", sep=";")

def df_compare(df_pred):
    df_final = pd.read_csv("data/GTSRB/GT-Final_test.csv", sep=";")

    df_final = df_final[["Filename", "ClassId"]]


    merged_df = pd.merge(df_pred, df_final, on='Filename', suffixes=('_pred', '_final'))
    merged_df['is_match'] = merged_df['ClassId_pred'] == merged_df['ClassId_final']

    total = len(merged_df)
    matches = merged_df['is_match'].sum()
    accuracy = matches / total

    print(f"Matches: {matches}/{total} ({accuracy:.2%})")



# mismatches = merged_df[~merged_df['is_match']].sort_values(by="ClassId_pred")
# mismatches.to_csv("mismatch_list.csv", index=False)

