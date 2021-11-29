from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
KNN = KNeighborsClassifier
KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)
accuracy_score(y_test, predictions)
SKFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
params = {
    'n_neighbors' : [5]
}
GSCV = GridSearchCV(estimator=KNeighborsClassifier(), 
             param_grid=params, 
             cv=SKFold, 
             verbose=1, 
             scoring='accuracy', 
             return_train_score=True)
GSCV.fit(X, y)
GSCV.cv_results_['mean_test_score']

# KMEANS
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
sns.scatterplot(x='horas', y='taxa_de_cliques', data=X, hue=kmeans.labels_, palette='viridis')

# OPTICS
from sklearn.cluster import OPTICS
from sklearn import metrics
optics = OPTICS(min_samples=8).fit(X)
sns.scatterplot(x='horas', y='taxa_de_cliques', data=X, hue=optics.labels_, palette='viridis')

knwon_class = engajamento['Classe']
estimated_class1 = optics.labels_
estimated_class2 = kmeans.labels_
metrics.adjusted_rand_score(known_class, estimated_class1) #optics
metrics.adjusted_rand_score(known_class, estimated_class2) #kmeans

#silhouette pode ser aplicada quando não temos uma classe previamente conhecida
metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
metrics.silhouette_score(X, optics.labels_, metric='euclidean')
