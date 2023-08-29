# required pakages
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

warnings.filterwarnings("ignore")
data = pd.read_csv("cars_class.csv")

# Data cleaning part

data.drop(columns = 'ID', inplace=True)
data.info()

# data preparation for classification

X = data.iloc[:, :-1]
y = data[['Class']]


# RandomForestClassifier on whole dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)
# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)
# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# LogisticRegression on whole dataset

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# LDA(LinearDiscriminantAnalysis) for feature selection & improve better accuracy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=3)
from sklearn.model_selection import train_test_split
X_transformed = lda.fit_transform(X, y)
X_transformed = pd.DataFrame(X_transformed)
x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size= 0.3, random_state = 100)

### applying RandomForestClassifier on  X_transformed data from LDA


# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=50)
rf.fit(x_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(x_test)

# Calculate and print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate and print F1-score (Macro)
f1_macro = f1_score(y_test, y_pred, average='macro')
print("F1-score (Macro):", f1_macro)

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate and display class-wise accuracy
class_accuracy = (np.diag(cm) / cm.sum(axis=1)) * 100
class_accuracy_dict = dict(enumerate(class_accuracy))
print("Class-wise Accuracy:")
print(class_accuracy_dict)

# The overall accuracy of my model is approximately 83.8%. This means that around 83.8% of the instances in  test dataset were correctly classified.

# The F1-score is a measure that combines both precision and recall, giving you a balanced measure of the model's performance. The macro-average F1-score  calculated is around 0.834, indicating a good balance between precision and recall across all classes.

#Class-wise Accuracy:
# This section provides the accuracy for each individual class. For instance:

#Class 0 (index 0) has an accuracy of 96.7%. This means that 96.7% of instances belonging to class 0 were correctly classified.
#Class 1 (index 1) has an accuracy of 71.4%. This means that 71.4% of instances belonging to class 1 were correctly classified.
#Class 2 (index 2) has an accuracy of 72.2%. This means that 72.2% of instances belonging to class 2 were correctly classified.
#Class 3 (index 3) has an accuracy of 92.3%. This means that 92.3% of instances belonging to class 3 were correctly classified.



#overall, with a high accuracy and a good F1-score

 
 

#need to on improving the prediction accuracy for classes with lower accuracies and consider potential strategies to address class imbalance if applicable



