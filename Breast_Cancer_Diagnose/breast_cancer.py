# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:43:50 2020

@author: HP
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description


cancer.keys()

def answer_one():
    
      
    # Your code here
    print(len(cancer['feature_names']))
    columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension','radius error', 'texture error', 'perimeter error', 'area error','smoothness error', 'compactness error', 'concavity error','concave points error', 'symmetry error', 'fractal dimension error','worst radius', 'worst texture', 'worst perimeter', 'worst area','worst smoothness', 'worst compactness', 'worst concavity','worst concave points', 'worst symmetry', 'worst fractal dimension','target']
    
    
    index = range(0, 569, 1)
#     print(cancer['data'].shape)
    df = pd.DataFrame(data=cancer['data'], index=index, columns = columns[:30])
#     print(cancer['target'])
    df['target'] = cancer['target']
    
#     print(df.head())
    ans = df
    
    return ans


answer_one()


def answer_two():
    cancerdf = answer_one()
    
#     cancerdf = pd.DataFrame(cancerdf)
    
    # Your code here
    malignant_count = len(cancerdf[cancerdf['target'] == 0])
    benign_count = len(cancerdf[cancerdf['target'] == 1])
    
    index = ['malignant', 'benign']
    
    target = pd.Series(data=[malignant_count, benign_count], index=index)
    
#     print(target)
    ans = target
    
    
    return ans


answer_two()

def answer_three():
    cancerdf = answer_one()
#     from sklearn.model_selection import train_test_split
    
#     # Your code here
#     X = fruits[['height', 'width', 'mass', 'color_score']]
#     y = fruits['fruit_label']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
#     return X, y
    X = cancerdf.iloc[:,:30]
    y = cancerdf.iloc[:,30:32]
#     y = cancerdf['target']
    y = cancerdf.target
#     print(y)
    return X, y

answer_three()





from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    
    # Your code here
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#     print(y_test.shape)
    
#     return 'ans'
    return X_train, X_test, y_train, y_test
answer_four()



from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    
    # Your code here
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    
    
    return knn
answer_five()

def answer_six():
    cancerdf = answer_one()
    knn = answer_five()
    means = (cancerdf.mean()[:-1].values.reshape(1, -1))
#     print(means.shape)
    # Your code here
    prediction = knn.predict(means)
#     print(prediction)
    ans = np.array(prediction)
    return ans
answer_six()






def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    # Your code here
    
    prediction = knn.predict(X_test)
#     print(prediction)
    ans = np.array(prediction)
    return ans
    
#     return # Return your answer
answer_seven()



def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    
    
    # Your code here
    ans = knn.score(X_test, y_test)
    
    return ans
answer_eight()





def accuracy_plot():
    


    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    
    
    
    
accuracy_plot()