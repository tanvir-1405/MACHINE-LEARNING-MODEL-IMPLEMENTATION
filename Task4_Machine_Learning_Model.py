# 📦 Importing necessary libraries
import pandas as pd  # For handling data tables
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For making pretty plots
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.feature_extraction.text import CountVectorizer  # For converting text to numbers
from sklearn.naive_bayes import MultinomialNB  # For building the spam classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For checking performance

# 📥 Loading the dataset from an online source
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])

# 👀 Showing the first few rows
df.head()

# 🔁 Converting text labels into numbers: 'ham' = 0, 'spam' = 1
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 📊 Checking how many messages are ham vs. spam
df['label'].value_counts() 

# ✂️ Splitting the messages into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# 🧾 Printing the number of training and testing messages
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")

# ✨ Converting text messages into numerical vectors (Bag of Words model)
vectorizer = CountVectorizer()

# 🧠 Learning vocabulary from training messages and transforming them
X_train_vec = vectorizer.fit_transform(X_train)

# 🔄 Transforming test messages using the same vocabulary
X_test_vec = vectorizer.transform(X_test)

# 🧪 Creating the Naive Bayes model
model = MultinomialNB()

# 🎯 Training the model using the training data
model.fit(X_train_vec, y_train)

# 📬 Making predictions on the test messages
y_pred = model.predict(X_test_vec)

# ✅ Printing the accuracy of predictions
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🧾 Showing the confusion matrix (actual vs. predicted)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 🖼️ Plotting the confusion matrix as a heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 📋 Printing a detailed report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))
