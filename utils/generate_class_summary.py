import pandas as pd

def generate_class_summary(df_preds: pd.DataFrame, ground_truth_csv_path: str, output_csv_path="class_summary.csv"):
    # Load and prepare ground truth
    df_gt = pd.read_csv(ground_truth_csv_path, sep=";")[["Filename", "ClassId"]]
    df_gt = df_gt.rename(columns={"ClassId": "ClassId_final"})

    # Merge predictions with ground truth
    df_merged = pd.merge(df_preds, df_gt, on="Filename")
    df_merged["is_match"] = df_merged["ClassId"] == df_merged["ClassId_final"]

    # Total existed: count of ground truth per class
    total_existed = df_merged["ClassId_final"].value_counts().sort_index()

    # Total failed guess: ground truth != predicted
    failed_guesses = df_merged[~df_merged["is_match"]]["ClassId_final"].value_counts().sort_index()

    # Total guessed: how many times model predicted each class
    total_guessed = df_merged["ClassId"].value_counts().sort_index()

    # Merge all into one DataFrame
    summary_df = pd.DataFrame({
        "ClassId": range(43),  # 0 to 42
        "Total_Existed": total_existed.reindex(range(43), fill_value=0),
        "Total_Failed_Guess": failed_guesses.reindex(range(43), fill_value=0),
        "Total_Guessed": total_guessed.reindex(range(43), fill_value=0)
    })

    # Optional: sort by ClassId
    summary_df = summary_df.sort_values("ClassId")

    # Save to CSV
    summary_df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved class-wise summary to: {output_csv_path}")
