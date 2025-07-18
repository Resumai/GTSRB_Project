
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_class_summary(csv_path="class_summary.csv"):
    df = pd.read_csv(csv_path)

    x = np.arange(len(df))  # the label locations
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, df["Total_Existed"], width, label="Total Existed (GT)")
    ax.bar(x, df["Total_Failed_Guess"], width, label="Failed Predictions")
    ax.bar(x + width, df["Total_Guessed"], width, label="Total Guessed by Model")

    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    ax.set_title("Class-wise Model Prediction Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(df["ClassId"], rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()


plot_class_summary()