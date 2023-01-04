from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

"""
Description.：
Requires users to enter their own data paths
This code uses relative paths, but users can of course modify them to suit their needs
"""


np.set_printoptions(suppress= True)
#Path to the original data
path1 = ' '
# Path to synthetic data
path2 = ' '
data1 = pd.read_csv(path1,header=None)
x1 = data1.drop(0, axis='columns')

data2 = pd.read_csv(path2,header=None)
x2= data2.drop(0, axis='columns')


#The first step is to normalize the data
X1_std = StandardScaler().fit_transform(x1)
X2_std = StandardScaler().fit_transform(x2)
#Instantiating PCA
pca = PCA(n_components = 1)
#training data
pca.fit(X1_std)
pca.fit(X2_std)

#Transformation of the original dataset
new_data1 = x1.dot(pca.components_.T)
new_data2 = x2.dot(pca.components_.T)

minx = min(new_data1.shape[0], new_data2.shape[0])
new_data1 = new_data1.sample(n=minx,replace=False)
new_data2 = new_data2.sample(n=minx,replace=False)


#The sample feature retention rate and offset rate (SFROR)
def mtx_similar(new_data1, new_data2) ->float:
    '''
    First calculate the angle between the two vectors after processing, because the smaller the angle, the smaller the offset of the two vectors；
    Next, calculate the distance between the two vectors, because the closer the distance is, the more features are preserved between the two vectors
    :param :Matrix 1
    :param :Matrix 2
    :return:SFROR
    '''

    # offset rate
    numer = np.sum(new_data1.values.ravel() * new_data2.values.ravel())
    denom = np.sqrt(np.sum(new_data1.values**2) * np.sum(new_data2.values**2))
    similar1 = numer / denom
    similar1 = (similar1+1) / 2

    # feature retention rate
    differ = new_data1.values - new_data2.values
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(new_data1.values)
    len2 = np.linalg.norm(new_data2.values)
    denom = (len1 + len2)
    similar2 = 1 - (dist / denom)
    similar = (similar1 * 0.2 + similar2 * 0.8)  # Pareto's Law
    return similar1, similar2,similar

test_angle,test_F,test_comprehensive =  mtx_similar(new_data1, new_data2)
print('offset rate：',test_angle)
print('feature retention rate：',test_F)
print('SFROR：',test_comprehensive)


