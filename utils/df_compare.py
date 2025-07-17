import pandas as pd


def df_compare(df_preds : pd.DataFrame, df_finals : pd.DataFrame):
    df_finals = pd.read_csv(df_finals, sep=";")
    df_finals = df_finals[["Filename", "ClassId"]]


    merged_df = pd.merge(df_preds, df_finals, on='Filename', suffixes=('_pred', '_final'))
    merged_df['is_match'] = merged_df['ClassId_pred'] == merged_df['ClassId_final']

    total = len(merged_df)
    matches = merged_df['is_match'].sum()
    accuracy = matches / total

    print(f"Matches: {matches}/{total} ({accuracy:.2%})")



# mismatches = merged_df[~merged_df['is_match']].sort_values(by="ClassId_pred")
# mismatches.to_csv("mismatch_list.csv", index=False)

