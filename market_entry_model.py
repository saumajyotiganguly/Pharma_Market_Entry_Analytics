
import pandas as pd
import matplotlib.pyplot as plt

# 1. Base Target Profile Dataset
raw_data = {
    "Doctor_ID": ["D001", "D002", "D003", "D004", "D005", "D006", "D007", "D008", "D009", "D010"],
    "Monthly_Volume": [320, 150, 480, 90, 275, 410, 60, 530, 195, 360],
    "Diabetes_Count": [85, 40, 120, 22, 70, 105, 15, 140, 55, 95],
    "Avg_Annual_Visits": [6.2, 3.8, 8.5, 2.1, 5.4, 7.9, 1.8, 9.2, 4.6, 7.1]
}

df = pd.DataFrame(raw_data)

print("--- Raw Input Database ---")
print(df.to_string(index=False))

# 2. Feature Scaling Pipeline (Min-Max Normalization)
# Scale attributes to a 0-1 range to handle unit variances across fields
target_metrics = ["Monthly_Volume", "Diabetes_Count", "Avg_Annual_Visits"]

for metric in target_metrics:
    min_val = df[metric].min()
    max_val = df[metric].max()
    df[f"{metric}_Scaled"] = (df[metric] - min_val) / (max_val - min_val)

print("\n--- Normalized Analytical Metrics ---")
normalized_views = ["Doctor_ID", "Monthly_Volume_Scaled", "Diabetes_Count_Scaled", "Avg_Annual_Visits_Scaled"]
print(df[normalized_views].round(4).to_string(index=False))

# 3. Apply Strategic Weights Matrix (Base Total: 100)
# Resource Allocation Mix: Target Demographics (38%), Visibility (32%), Retention (30%)
df["Priority_Score"] = (
    (df["Monthly_Volume_Scaled"] * 0.32) +
    (df["Diabetes_Count_Scaled"] * 0.38) +
    (df["Avg_Annual_Visits_Scaled"] * 0.30)
) * 100

df["Priority_Score"] = df["Priority_Score"].round(2)

# 4. Generate Commercial Rankings
df_ranked = df.sort_values("Priority_Score", ascending=False).reset_index(drop=True)
df_ranked.index += 1 
df_ranked.index.name = "Rank"

# 5. Pipeline Reporting and Console Logs
reporting_fields = ["Doctor_ID", "Monthly_Volume", "Diabetes_Count", "Avg_Annual_Visits", "Priority_Score"]
print("\n--- Final Market-Entry Priority Standings ---")
print(df_ranked[reporting_fields].to_string())

# 6. Corporate Presentation Asset Generation
plt.figure(figsize=(10, 5))
plt.barh(df_ranked["Doctor_ID"], df_ranked["Priority_Score"], color="#117A65")
plt.xlabel("Strategic Priority Index (0-100)")
plt.ylabel("Target Provider ID")
plt.title("Pharma Sales Distribution Architecture: Top Medical Targets", pad=15)
plt.gca().invert_yaxis() 
plt.tight_layout() 
plt.savefig("doctor_rankings.png", dpi=300) 
