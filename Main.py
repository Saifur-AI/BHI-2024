import pandas as pd
import numpy as np
import seaborn as sns
from tslearn.shapelets import LearningShapelets

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
# Example ground truth (y_true) and predicted (y_pred) values
df=pd.read_csv("updated_glucosetimeseris_groundtruth.csv")
# print(df.iloc[0,1:-5].values[-1]+np.mean(np.diff(df.iloc[0,1:-5].values)))
# print(df.iloc[0,1:-5])
# print(df['Prediction'][0])
# print(df['groundtruth'][0])
# exit(0)
y_true = df['groundtruth'].values#[:5000]
y_pred = df['Prediction'].values#[:5000]
# Reshape data for clustering
X = df[['Glucose_Level_1','Glucose_Level_2','Glucose_Level_3','Glucose_Level_4','Glucose_Level_5','Glucose_Level_6']]
# for idx in range(100):
#  plt.plot(X.iloc[idx])
# plt.show()
# exit()
# X=df[['HbA1C','Gender']]
# Calculate WCSS for different numbers of clusters
wcss = []
max_clusters = 10
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(5,3))
plt.plot(range(1, max_clusters + 1), wcss, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.xticks(range(1, max_clusters + 1))
plt.grid(True)
plt.axvline(x=3, linestyle='--', color='red')  # Example line for elbow point, adjust accordingly
plt.tight_layout()
for spine in plt.gca().spines.values():
    spine.set_linestyle('dotted')
plt.savefig(f"figure/elbow.pdf", bbox_inches='tight')
# exit()
# Fit KMeans
kmeans = KMeans(n_clusters=3)  # Choose the number of clusters
clusters = kmeans.fit_predict(X)
# Fit DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=1)  # eps is the maximum distance between points in the same cluster
# clusters = dbscan.fit_predict(X)

# Add cluster labels to the DataFrame
df['cluster'] = clusters

# Print clustered data
# print(df)

# # Visualization of the clusters
# plt.figure(figsize=(5,3))
# plt.scatter(df['Prediction'], df['Prediction'], c=df['cluster'], cmap='viridis', marker='o')
# plt.title('Clusters of Predicted vs True Values')
# plt.xlabel('Ground Truth Values')
# plt.ylabel('Predicted Values')
# plt.colorbar(label='Cluster')
# plt.grid(True)
# plt.axline((0, 0), slope=1, color='red', linestyle='--')  # Line y=x for reference
# plt.show()

# sns.kdeplot(data=df, y="Glucose_Level_6", hue="cluster")
# plt.show()
# exit()
# Group by clusters and calculate correlation for each group
grouped = df.groupby('cluster')
plt.figure(figsize=(6,6))
correlations = {}
cnt=1
# ['HbA1C','Gender','SubjectID']
for tk in ['Glucose_Level_1','Glucose_Level_2','Glucose_Level_3','Glucose_Level_4','Glucose_Level_5',
    'Glucose_Level_6','HbA1C','Gender']:
    plt.subplot(3,3,cnt)
    # plt.title(tk)
    print(tk)
    sns.kdeplot(data=df, x=tk, y="Prediction",  fill=True,palette="tab10",hue="cluster")
    # Adjust legend for each subplot
    # if cnt == 1:  # Only show the legend on the first plot
    #     plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    # else:
    plt.legend().set_visible(False)
    for spine in plt.gca().spines.values():
        spine.set_linestyle('dotted')
    # for cluster, group in grouped:
    #     corr, p_value = pearsonr(group['groundtruth'], group['Prediction'])
    #     print(f"Cluster {cluster}",group['Gender'].unique())
    #     mae = mean_absolute_error(group['groundtruth'], group['Prediction'])
    #     print(f"Mean Absolute Error (MAE): {mae}")
    #     mse = mean_squared_error(group['groundtruth'], group['Prediction'])
    #     print(f"Mean Squared Error (MSE): {mse}")
    #     # plt.scatter(np.arange(len(group[tk].unique())),group[tk].unique(),label=f"Cluster {cluster}")
    #     # sns.kdeplot(data=group, x="Prediction",y=tk, hue="cluster", multiple="stack")
    #     correlations[cluster] = {'correlation': corr, 'p_value': p_value}
    #     print(f"Cluster {cluster} - Correlation: {corr}, p-value: {p_value}")
    # plt.legend(ncols=3)

    cnt += 1
plt.tight_layout()
plt.savefig(f"figure/features.jpg", bbox_inches='tight')
plt.show()
# Show grouped data with clusters
# print(data)

# # Optional: Visualization of clustered data
# plt.figure(figsize=(8, 6))
# for cluster, group in grouped:
#     plt.scatter(group['groundtruth'], group['Prediction'], label=f'Cluster {cluster}')
#
# plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
# plt.title('Clusters of Predicted vs True Values')
# plt.xlabel('Ground Truth Values')
# plt.ylabel('Predicted Values')
# plt.legend()
# plt.grid(True)
# plt.show()
# # Calculate WCSS for different numbers of clusters
# wcss = []
# max_clusters = 10
# for i in range(1, max_clusters + 1):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# # Plotting the Elbow Method
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, max_clusters + 1), wcss, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS (Inertia)')
# plt.xticks(range(1, max_clusters + 1))
# plt.grid(True)
# plt.axvline(x=3, linestyle='--', color='red')  # Example line for elbow point, adjust accordingly
# plt.show()
exit()
######################################################
# 1. Residuals
residuals = y_true - y_pred

# 2. Mean Absolute Error (MAE)
mae = mean_absolute_error(y_true, y_pred)
# 3. Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
# 4. Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
# 5. R-squared (R²)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# 6. Test for normality of residuals (Shapiro-Wilk test)
shapiro_stat, shapiro_pvalue = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test for Normality of Residuals: {shapiro_stat}, p-value = {shapiro_pvalue}")

# 7. Homoscedasticity test (Breusch-Pagan equivalent using residuals)
_, bp_pvalue = stats.bartlett(y_pred, residuals)  # Bartlett's test for constant variance
print(f"Bartlett Test for Homoscedasticity p-value: {bp_pvalue}")

# 8. Correlation between Ground Truth and Predicted Values (with p-value)
correlation, corr_pvalue = stats.pearsonr(y_true, y_pred)
print(f"Correlation between Ground Truth and Predicted: {correlation}, p-value = {corr_pvalue}")
# Interpretation:
if corr_pvalue < 0.05:
    print("The correlation between ground truth and predicted values is statistically significant.")
else:
    print("The correlation between ground truth and predicted values is not statistically significant.")
# Visualization

# Scatter plot of Ground Truth vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, color='blue')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
plt.title('Ground Truth vs Predicted Values')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# Residuals vs Predicted values plot
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Distribution of residuals (should ideally resemble normal distribution)
plt.figure(figsize=(6,4))
plt.hist(residuals, color='purple', bins=10, edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Q-Q plot for normality of residuals
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Calculate the mean and difference
mean_values = np.mean(np.array([y_true, y_pred]), axis=0)
differences = y_true - y_pred

# Calculate mean difference and limits of agreement
mean_diff = np.mean(differences)
std_diff = np.std(differences)
lower_limit = mean_diff - 1.96 * std_diff
upper_limit = mean_diff + 1.96 * std_diff

# Create the Bland-Altman plot
plt.figure(figsize=(8, 6))
plt.scatter(mean_values, differences, color='blue', label='Differences')
plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference')
plt.axhline(lower_limit, color='green', linestyle='--', label='Lower Limit of Agreement')
plt.axhline(upper_limit, color='green', linestyle='--', label='Upper Limit of Agreement')

plt.title('Bland-Altman Plot')
plt.xlabel('Mean of Two Measurements')
plt.ylabel('Difference between Two Measurements')
plt.legend()
plt.grid(True)
plt.show()