import pandas as pd
from shapesimilarity import shape_similarity
from scipy.spatial import procrustes
import numpy as np
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tslearn.clustering import KernelKMeans
from tslearn.datasets import CachedDatasets

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
# # Index(['Index', 'Glucose_Level_1', 'Glucose_Level_2', 'Glucose_Level_3',
# #        'Glucose_Level_4', 'Glucose_Level_5', 'Glucose_Level_6', 'HbA1C',
# #        'Prediction', 'Gender', 'groundtruth', 'SubjectID'],
# #       dtype='object')
# df=pd.read_csv("updated_glucose_predictions_gt_with_SubjectID(in).csv")
# # for t in range(df.shape[0]-1):
# subjectlst=np.unique(df['SubjectID'])
# for sub in subjectlst:
#     temp=df[df['SubjectID']==sub]
#     for t in range(temp.shape[0] - 1):
#         temp.iloc[t,10]=temp.iloc[t+1,6]
#     if sub==1:
#         all_df=temp
#     else:
#         all_df=pd.concat([all_df,temp])
# all_df.to_csv("updated_glucose_predictions_groundtruth.csv",index=False)
# # print(np.unique(df['SubjectID']))
# # exit(0)


# df=pd.read_csv("updated_glucose_predictions_groundtruth.csv")
# df=df.iloc[:10000,:]
# # Reshape data for clustering
# X = df[['Glucose_Level_1','Glucose_Level_2','Glucose_Level_3','Glucose_Level_4','Glucose_Level_5','Glucose_Level_6']]
#
# X_train = TimeSeriesScalerMeanVariance().fit_transform(X)
# sz = X_train.shape[1]
#
# gak_km = KernelKMeans(n_clusters=3,
#                       kernel="gak",
#                       kernel_params={"sigma": "auto"},
#                       n_init=20,
#                       verbose=True,
#                       random_state=23)
# y_pred = gak_km.fit_predict(X_train)
# df['cluster'] = y_pred
# df.to_csv("updated_glucosetimeseris_groundtruth.csv",index=False)
#
# clr=["b-","g-","k-"]
# plt.figure()
# for yi in range(3):
#     plt.subplot(3, 1, 1 + yi)
#     for xx in X_train[y_pred == yi]:
#         plt.plot(xx.ravel(), clr[yi], alpha=.2)
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.title("Cluster %d" % (yi + 1))
#
# plt.tight_layout()
# plt.show()

df=pd.read_csv("updated_glucosetimeseris_groundtruth.csv")


# corr, p_value = pearsonr(df['groundtruth'], df['Prediction'])
# # print(f"Cluster {cluster}", group['Gender'].unique())
# mae = mean_absolute_error(df['groundtruth'], df['Prediction'])
#
# print(f"Cluster",f"Mean Absolute Error (MAE): {mae}")
# mse = mean_squared_error(df['groundtruth'], df['Prediction'])
# print(f"Cluster ",f"Mean Squared Error (MSE): {mse}")
# # similarity = shape_similarity(group['groundtruth'].values.flatten(), group['Prediction'].values.flatten())
# # Perform Procrustes Analysis
# print(f"Cluster", f"corr: {corr}")
#
clr=sns.color_palette("Paired")[1:6]

clr=["tab:blue","tab:green","tab:brown"]
grouped = df.groupby('cluster')
cnt=1
all_result=[]
se=[]
abe=[]
corr2=[]
plt.figure(figsize=(4,4))
for cluster, group in grouped:
    corr, p_value = pearsonr(group['groundtruth'], group['Prediction'])

    # print(f"Cluster {cluster}", group['Gender'].unique())
    # mae = mean_absolute_error(group['groundtruth'], group['Prediction'])

    X_train = TimeSeriesScalerMeanVariance().fit_transform(group[['Glucose_Level_1','Glucose_Level_2','Glucose_Level_3','Glucose_Level_4','Glucose_Level_5',
                    'Glucose_Level_6']])
    #

    # clr=["b-","g-","m-"]
    # bxclr = ["b", "g", "m"]
    plt.subplot(3,1, cnt)
    # for yi in range(3):
    #     plt.subplot(3, 1, 1 + yi)
    # for xx in X_train:
    #     plt.plot(xx.ravel(),color=clr[cluster], alpha=.2)
    #

    # Plot each series with transparency and cluster-specific color
    mean_line=[]
    for xx in X_train:
        plt.plot(np.arange(6),xx.ravel(), alpha=0.1)
        mean_line.append(xx.ravel())


    mean_line = np.mean(mean_line, axis=0)
    plt.plot(np.arange(6),mean_line, 'k--', linewidth=1.5, label=f"Cluster {cluster} Mean")
    plt.xlabel(f"Cluster {cluster+1}")
    # plt.legend()

    for spine in plt.gca().spines.values():
        spine.set_linestyle('dotted')
    cnt+=1
        # spine.set_linewidth(2)
plt.tight_layout()
# plt.show()
# exit()
plt.savefig(f"figure/timeseriescluster.pdf", bbox_inches='tight')
exit()


    # abe.append(abs(group['groundtruth'].values-group['Prediction'].values))
    #
    # print(f"Cluster {cluster}",f"Mean Absolute Error (MAE): {mae}")
    # mse = mean_squared_error(group['groundtruth'], group['Prediction'])
    # print(f"Cluster {cluster}",f"Mean Squared Error (MSE): {mse}")
    # print(group['groundtruth'].values.shape, group['Prediction'].values.shape)
    # # similarity = shape_similarity(group['groundtruth'].values.flatten(), group['Prediction'].values.flatten())
    # # Perform Procrustes Analysis
    # print(f"Cluster {cluster}", f"corr: {corr}")
    # all_result.append([cluster, mae, mse, corr])

    # # Combine data into a DataFrame for easier plotting
    # data = pd.DataFrame({
    #     'Absolute_error': abe,
    # })
    #
    # # Create a boxplot
    # plt.figure(figsize=(4,3))
    # sns.boxplot(y='Absolute_error', data=data ,color=clr[cluster])
    # for spine in plt.gca().spines.values():
    #     spine.set_linestyle('dotted')
    #     # spine.set_linewidth(2)
    # plt.ylabel('Absolute Error')
    #
    # # Optional: Show the mean as a line (if desired)
    # plt.tight_layout()
    # plt.savefig(f"figure/comparison_{cluster}_abe.pdf", bbox_inches='tight')
    # # Combine data into a DataFrame for easier plotting
    # se.append((group['groundtruth'].values-group['Prediction'].values)**2)
    # data = pd.DataFrame({
    #         'Squared_error': se,
    # })
    # # Create a boxplot
    # plt.figure(figsize=(4, 3))
    # sns.boxplot(y='Squared_error', data=data,color=clr[cluster])
    # # Set dotted borders for the axes
    # for spine in plt.gca().spines.values():
    #     spine.set_linestyle('dotted')
    #     # spine.set_linewidth(2)
    # plt.ylabel('Squared Error')
    #
    # # Optional: Show the mean as a line (if desired)
    # plt.tight_layout()
    # plt.savefig(f"figure/comparison_{cluster}_se.pdf", bbox_inches='tight')
# pvalues = [mannwhitneyu(se[0], se[1],alternative="two-sided").pvalue,mannwhitneyu(se[1], se[2], alternative="two-sided").pvalue]
# formatted_pvalues = [f"P={np.round(p,3)}" for p in pvalues]
# Create DataFrame in wide format
print(se[0].shape,se[1].shape,se[2].shape)
data = pd.DataFrame({
    'Cluster 1': abe[0][0:abe[2].shape[0]],
    'Cluster 2': abe[1][0:abe[2].shape[0]],
    'Cluster 3': abe[2][0:abe[2].shape[0]]
})
u_statistic, p_value = stats.mannwhitneyu(data['Cluster 1'], data['Cluster 2'], alternative='two-sided')
# Add stars based on p-value
print(p_value)
if p_value < 0.001:
    stars = '***'
elif p_value < 0.01:
    stars = '**'
elif p_value < 0.05:
    stars = '*'
else:
    stars = 'ns'  # Not significant
# Melt DataFrame to long format for seaborn
data_melted = data.melt(var_name='Cluster', value_name='Absolute Error')
colors = ["#FF9999", "#66B2FF", "#99FF99"]  # Colors for each cluster
# Plot the boxplot
plt.text((0+1) * 0.5, 45 + 0.5, stars, ha='center', va='bottom', color='k')
plt.plot([0,0,1,1], [45, 45 + 1, 45 + 1, 45], lw=1.5, c='k')

u_statistic, p_value = stats.mannwhitneyu(data['Cluster 2'], data['Cluster 3'], alternative='two-sided')
# Add stars based on p-value
print(p_value)
if p_value < 0.001:
    stars = '***'
elif p_value < 0.01:
    stars = '**'
elif p_value < 0.05:
    stars = '*'
else:
    stars = 'ns'  # Not significant
# Melt DataFrame to long format for seaborn
# data_melted = data.melt(var_name='Cluster', value_name='Absolute Error')
colors = ["#FF9999", "#66B2FF", "#99FF99"]  # Colors for each cluster
# Plot the boxplot
plt.text((1+2) * 0.5,50 + 0.5, stars, ha='center', va='bottom', color='m')
plt.plot([1,1,2,2], [50, 50 + 1, 50 + 1, 50], lw=1.5, c='m')

u_statistic, p_value = stats.mannwhitneyu(data['Cluster 1'], data['Cluster 3'], alternative='two-sided')
# Add stars based on p-value
print(p_value)
if p_value < 0.001:
    stars = '***'
elif p_value < 0.01:
    stars = '**'
elif p_value < 0.05:
    stars = '*'
else:
    stars = 'ns'  # Not significant
# Melt DataFrame to long format for seaborn
# data_melted = data.melt(var_name='Cluster', value_name='Absolute Error')
colors = ["#FF9999", "#66B2FF", "#99FF99"]  # Colors for each cluster
# Plot the boxplot
plt.text((0+2) * 0.5, 55 + 0.5, stars, ha='center', va='bottom', color='b')
plt.plot([0,0,2,2], [55, 55+1, 55+ 1 ,55], lw=1.5, c='b')


sns.boxplot(x='Cluster', y='Absolute Error', data=data_melted,showfliers=False,palette=sns.color_palette("tab10"))
plt.tight_layout()
plt.ylim([0, 60])
# Display plot
# plt.show()
plt.savefig(f"figure/abe.pdf", bbox_inches='tight')
# all_result=pd.DataFrame(all_result,columns=["cluster", "mae", "mse", "corr"])
# all_result.to_csv("all_results.csv",index=False)