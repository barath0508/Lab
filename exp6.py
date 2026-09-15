import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
data = pd.read_csv("csv_expt6.txt")
data.head()
feature_cols = ['Age', 'EstimatedSalary']
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(
    np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
    np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01)
)
plt.contourf(
    x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(("red", "green"))
)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        x_set[y_set == j, 0], x_set[y_set == j, 1], 
        c=np.array(ListedColormap(("red", "green"))(i)).reshape(1, -1),
        label=j
    )
plt.title("Decision Tree (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
plt.figure(figsize=(20, 10))
plot_tree(classifier, 
          filled=True, 
          rounded=True,
          feature_names=feature_cols, 
          class_names=['0', '1'],
          fontsize=10,
          max_depth=4)  
plt.title("Decision Tree (Initial)")
plt.savefig('decisiontree.png', dpi=300, bbox_inches='tight')
plt.show()
classifier_opt = DecisionTreeClassifier(criterion="gini", max_depth=3)
classifier_opt = classifier_opt.fit(x_train, y_train)
y_pred_opt = classifier_opt.predict(x_test)
print("Optimized Accuracy:", metrics.accuracy_score(y_test, y_pred_opt))
plt.figure(figsize=(15, 8))
plot_tree(classifier_opt, 
          filled=True, 
          rounded=True,
          feature_names=feature_cols, 
          class_names=['0', '1'],
          fontsize=10)
plt.title("Decision Tree (Optimized with max_depth=3)")
plt.savefig('opt_decisiontree_gini.png', dpi=300, bbox_inches='tight')
plt.show()
print("Trees saved as 'decisiontree.png' and 'opt_decisiontree_gini.png'")
