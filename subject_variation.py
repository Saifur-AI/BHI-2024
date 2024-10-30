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
# exit(0)
df=pd.read_csv("updated_glucose_predictions_groundtruth.csv")
print(df["SubjectID"].unique())
df['Absolute Error']=abs(df['groundtruth']-df['Prediction'])
plt.figure(figsize=(5,3))
sns.barplot(df,x='SubjectID',y='Absolute Error',palette="husl",hue='SubjectID',legend=None)
for spine in plt.gca().spines.values():
    spine.set_linestyle('dotted')
plt.tight_layout()
# plt.show()
# exit()
plt.savefig(f"figure/subject_variation.pdf", bbox_inches='tight')
