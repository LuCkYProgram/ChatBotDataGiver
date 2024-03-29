import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

### Regression
def train_regression(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X_train, y_train)
    print (model)
    return(model)

def evaluation_of_regression(model, X_test, y_test):
    predictions = model.predict(X_test)
    np.set_printoptions(suppress=True)
    print('Predicted labels: ', np.round(predictions)[:10])
    print('Actual labels   : ' ,y_test[:10])
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions')
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test,p(y_test), color='magenta')
    plt.show()
    
def train_binary_classification(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    reg = 0.01
    model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
    print (model)
    return(model)

### Classification
def evaluation_of_binary_classification(model, X_test, y_test):
    predictions = model.predict(X_test)
    print('Predicted labels: ', predictions)
    print('Actual labels:    ' ,y_test)
    from sklearn.metrics import accuracy_score
    print('Accuracy: ', accuracy_score(y_test, predictions))
    
def train_multi_classification(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    reg = 0.1
    multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(X_train, y_train)
    print (multi_model)
    return(multi_model)

def evaluation_of_multi_classification(multi_model, X_test, y_test):
    multi_model_predictions = multi_model.predict(X_test)
    print('Predicted labels: ', multi_model[:15])
    print('Actual labels   : ' ,y_test[:15])
    return multi_model_predictions
    
def classifier_report(predictions, y_test, average_targets='binary'):
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions))
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print("Overall Accuracy:",accuracy_score(y_test, predictions))
    print("Overall Precision:",precision_score(y_test, predictions, average=average_targets))
    print("Overall Recall:",recall_score(y_test, predictions, average=average_targets))
    
def confusion_report(df, predictions, labels, y_test):
    from sklearn.metrics import confusion_matrix
    mcm = confusion_matrix(y_test, predictions)
    print(mcm)
    plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    classes = df.labels.unique()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted " + labels[0])
    plt.ylabel("Actual Species " + labels[0])
    plt.show()
    return classes
    
def ROC_curve(multi_model, classes, X_test, y_test):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    probability_scores = multi_model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    thresh ={}
    for i in range(len(classes)):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, probability_scores[:,i], pos_label=i)
        plt.plot(fpr[i], tpr[i], linestyle='--', label=classes[i] + ' vs Rest')
    ### 
    # plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=penguin_classes[0] + ' vs Rest')
    # plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label=penguin_classes[1] + ' vs Rest')
    # plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=penguin_classes[2] + ' vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    auc = roc_auc_score(y_test, probability_scores, multi_class='ovr')
    print('Average AUC:', auc)

### Clustering
def clustering_prep(df):
    #add custom feature addin
    features = df[df.columns[0:6]]
    features.sample(10)
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    scaled_features = MinMaxScaler().fit_transform(features[df.columns[0:6]])
    pca = PCA(n_components=2).fit(scaled_features)
    features_2d = pca.transform(scaled_features)
    features_2d[0:10]
    return (features, features_2d)

def sum_of_squares_clustering(features):
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(features.values)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('WCSS by Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def k_means_clustering(features):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
    km_clusters = model.fit_predict(features.values)
    km_clusters
    return km_clusters
    
def agglomerative_clustering(features):
    from sklearn.cluster import AgglomerativeClustering
    agg_model = AgglomerativeClustering(n_clusters=3)
    agg_clusters = agg_model.fit_predict(features.values)
    agg_clusters
    return agg_clusters
    
def plot_clusters(samples, clusters):
    col_dic = {0:'blue',1:'green',2:'orange'}
    mrk_dic = {0:'*',1:'x',2:'+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color = colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()
    
def deep_learning_models():
    print("To edit")

