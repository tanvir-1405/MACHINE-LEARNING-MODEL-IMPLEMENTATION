# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: TANVIR KAUR KHOKHAR

*INTERN ID*: CT04DM1447

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

The fourth task in my project series was the implementation of a Machine Learning model to automatically classify SMS messages as spam or ham (non-spam). This task focused on the practical use of supervised learning techniques to detect unwanted or malicious text messages, a common real-world problem in digital communication systems.

The main objective of this project was to build a lightweight and efficient spam classifier using Natural Language Processing (NLP) and Machine Learning techniques, implemented in a Jupyter Notebook for easy code execution and visualization. This classifier provides real-time predictions based on input text, showcasing how data science can be used to enhance communication security.

Tools and Technologies Used:

Jupyter Notebook – for interactive Python coding, step-by-step execution, and easy debugging.

Python – the primary programming language due to its extensive libraries and readability.

scikit-learn – for model building and evaluation, especially the MultinomialNB classifier.

CountVectorizer – for text vectorization and tokenization (converting text into numbers).

Matplotlib & Seaborn – for plotting confusion matrices and visualizing results.

Data Used:

The dataset used in this project was a collection of SMS messages, labeled as either ham or spam. The dataset was imported from file hosted online, and included: 

The label column (indicating "spam" or "ham")

The message column (containing the actual SMS text)

These messages simulate real-world text communication and are representative of typical content received by users.

How the Model Works:

1. Data Preprocessing:

The labels were converted from text ("ham" / "spam") to numerical values (0 and 1).

The dataset was then split into training (80%) and testing (20%) sets to ensure unbiased evaluation.

2. Text Vectorization:

SMS messages, which are textual data, were converted into numerical form using CountVectorizer. This creates a matrix of token counts for each message, allowing machine learning algorithms to process them.

3. Model Training:

A Multinomial Naive Bayes classifier (MultinomialNB) was used, which is especially suitable for text classification tasks involving word frequencies.

The model was trained on the training set, learning patterns and word distributions associated with spam versus non-spam.

4. Evaluation:

The model was tested on the unseen testing data, and predictions were generated.

Key metrics such as accuracy, confusion matrix, and a classification report (including precision, recall, and F1-score) were computed.

The confusion matrix was visualized using Seaborn's heatmap to easily understand the number of true/false positives and negatives.

Real-World Applications:

This type of spam classifier is widely used in:

Email spam filters (e.g., Gmail or Outlook’s spam detection)

SMS-based fraud prevention in mobile carriers

WhatsApp/Telegram content moderation

Customer support chatbots to filter abusive/spam content

Marketing tools to segment and filter messages before delivery

This project successfully demonstrated how machine learning and NLP can work together to solve a practical communication problem. With simple tools and a well-prepared dataset, we were able to create a smart system that can distinguish spam from genuine messages with a high degree of accuracy. The Jupyter Notebook interface made development and testing smooth and user-friendly, and the model itself can be deployed or improved further with deep learning or real-time APIs. This hands-on experience highlighted the true power of applied AI in day-to-day technology.

#OUTPUT

![Image](https://github.com/user-attachments/assets/888d46f5-91b6-4aeb-bdf4-4affccb3bee3)

![Image](https://github.com/user-attachments/assets/b659df14-c2b5-4749-aa80-b30bdf5c62a3)

![Image](https://github.com/user-attachments/assets/32148154-c2d9-4201-aa68-775e937d36b8)

![Image](https://github.com/user-attachments/assets/ce2aba62-9113-4d17-b5dd-96fdd8104daf)

![Image](https://github.com/user-attachments/assets/0964e166-6570-40c5-9cfc-08d054913232)

![Image](https://github.com/user-attachments/assets/2ed92a6b-858d-4189-a937-09b5323a5aae)

![Image](https://github.com/user-attachments/assets/9b7e551f-754a-43ba-b98b-1bde5cfeb98d)

![Image](https://github.com/user-attachments/assets/3ed11340-053f-4838-b8e5-0231f87b2f9e)

![Image](https://github.com/user-attachments/assets/be7b34a8-a99b-40ae-8bec-13237060695b)
