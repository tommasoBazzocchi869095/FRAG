import pandas as pd

df = pd.read_csv("grid_search_results.csv")
global_metrics = df.groupby(["Model", "Alpha"])[["Accuracy", "Precision", "Recall", "F1_Score"]].mean()
global_metrics = global_metrics.sort_values(by="Accuracy", ascending=False)

print("=== CLASSIFICA GLOBALE (Media dei 3 Dataset) ===\n")
print(global_metrics)
