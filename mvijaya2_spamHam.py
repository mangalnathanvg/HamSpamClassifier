# Submitted by:
# Mangalnathan Vijayagopal - mvijaya2

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score


# Method to process every text in the dataset
def process_text(text_msg):
    # Converting messages to lowercase
    text_msg = text_msg.lower()
    return text_msg

# Import the data set.
file = open("SMSSpamCollection")
data = []

# Prepare the data into a dataframe
for line in file:
    line = line.replace('\"', '')
    values = line.split('\t')
    values[1] = values[1].rstrip('\n')
    data.append(values)

# Create Data Frame and populate it with data from the data set.
data = pd.DataFrame(data, columns = ['Label', 'Text'])

# Processing text messages
data['Text'] = data['Text'].apply(process_text)

# Train test split 
X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size = 0.3, random_state = 1)

# Transforming train and test data into numerical vectors using TFID Vectorizer. See Report for details
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Support Vector Machines

# Training the SVM classifier using training data 
svm = svm.SVC(C=100)
svm.fit(X_train, y_train)

# Testing the classifier using testing data to get predictions.
y_pred = svm.predict(X_test)

# Comparing predictions of test data with actual labels of test data to determine accuracy.
print("\nConfusion Matrix for Support Vector Machines")
print(confusion_matrix(y_test, y_pred))
print("SVM Accuracy: ", round(100 * accuracy_score(y_test, y_pred), 2), "%")

# Logistic Regression

# Training the Model
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(X_train, y_train)

# Testing the Model
pred = Spam_model.predict(X_test)

# Determining the Accuracy
print("\nConfusion Matrix for Logistic Regression Classifier")
print(confusion_matrix(y_test, pred))
print("Logistic Regression Accuracy: ", round(100 * accuracy_score(y_test,pred), 2), "%")


# Multinomial Naive Bayes

# Training the Model
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# Testing the Model
all_predictions = spam_detect_model.predict(X_test)

# Determining the Accuracy
print("\nConfusion Matrix for Multinomial Naive Bayes Classifier")
print(confusion_matrix(y_test, all_predictions))
print("Multinomial Naive Bayes Classifier Accuracy: ", round(100*accuracy_score(y_test, all_predictions), 2), "%")

# Complement Naive Bayes

# Training the Model
spam_detect_model = ComplementNB().fit(X_train, y_train)

# Testing the Model
all_predictions = spam_detect_model.predict(X_test)

# Determining the Accuracy
print("\nConfusion Matrix for Complement Naive Bayes Classifier")
print(confusion_matrix(y_test, all_predictions))
print("Complement Naive Bayes Classifier Accuracy: ", round(100*accuracy_score(y_test, all_predictions), 2), "%")


# Bernoulli Naive Bayes

# Training the Model
spam_detect_model = BernoulliNB().fit(X_train, y_train)

# Testing the Model
all_predictions = spam_detect_model.predict(X_test)

# Determining Accuracy
print("\nConfusion Matrix for Bernoulli Naive Bayes Classifier")
print(confusion_matrix(y_test, all_predictions))
print("Bernoulli Naive Bayes Classifier Accuracy: ", round(100*accuracy_score(y_test, all_predictions), 2), "%")