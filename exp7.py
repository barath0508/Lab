import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn import svm
from matplotlib import pyplot

# Download required NLTK data for tokenization and stopwords
#nltk.download('stopwords')

#nltk.download('punkt')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # <-- Add this new line

class data_read_write(object):
    def __init__(self, file_link):
        self.data_frame = pd.read_csv(file_link)

    def read_csv_file(self, file_link):
        self.data_frame = pd.read_csv(file_link)
        return self.data_frame

    def write_to_csvfile(self, file_link):
        self.data_frame.to_csv(file_link, encoding='utf-8', index=False, header=True)
        return


class generate_word_cloud(data_read_write):
    def __init__(self):
        pass

    def variance_column(self, data):
        return np.variance(data)

    def word_cloud(self, data_frame_column, output_image_file):
        text = " ".join(review for review in data_frame_column)
        stop_words = set(STOPWORDS)
        stop_words.update(["subject"])
        wordcloud = WordCloud(width=1200, height=800, stopwords=stop_words,
                              max_font_size=50, margin=0,
                              background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig("Distribution.png")
        plt.show()
        wordcloud.to_file(output_image_file)
        return


class data_cleaning(data_read_write):
    def __init__(self):
        pass

    def message_cleaning(self, message):
        Test_punc_removed = [char for char in message if char not in string.punctuation]
        Test_punc_removed_join = ''.join(Test_punc_removed)
        Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split()
                                        if word.lower() not in stopwords.words('english')]
        final_join = ' '.join(Test_punc_removed_join_clean)
        return final_join

    def apply_to_column(self, data_column_text):
        data_processed = data_column_text.apply(self.message_cleaning)
        return data_processed


class apply_embeddding_and_model(data_read_write):
    def __init__(self):
        pass

    def apply_count_vector(self, v_data_column):
        vectorizer = CountVectorizer(min_df=2, analyzer="word", tokenizer=None,
                                     preprocessor=None, stop_words=None)
        return vectorizer.fit_transform(v_data_column)
    def apply_svm(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        params = {'kernel': 'linear', 'C': 2, 'gamma': 1}
        
        # --- NEW CODE TO FIX WARNING ---
        base_svm = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
        svm_cv = CalibratedClassifierCV(base_svm, ensemble=False)
        # -------------------------------
        
        svm_cv.fit(X_train, y_train)
        y_predict_test = svm_cv.predict(X_test)
        cm = confusion_matrix(y_test, y_predict_test)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Heatmap')
        plt.show()
        
        print(classification_report(y_test, y_predict_test))
        print("test set")

        print("\nAccuracy Score: " + str(accuracy_score(y_test, y_predict_test)))
        print("F1 Score: " + str(f1_score(y_test, y_predict_test)))
        print("Recall: " + str(recall_score(y_test, y_predict_test)))
        print("Precision: " + str(precision_score(y_test, y_predict_test)))

        class_names = ['ham', 'spam']
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                svm_cv, X_test, y_test,
                display_labels=class_names,
                cmap=plt.cm.Blues,
                normalize=normalize
            )
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            
        plt.savefig("SVM.png")
        plt.show()

        ns_probs = [0 for _ in range(len(y_test))]
        lr_probs = svm_cv.predict_proba(X_test)
        lr_probs = lr_probs[:, 1]
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('SVM: ROC AUC=%.3f' % (lr_auc))
        
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC Curve')
        pyplot.legend()
        pyplot.savefig("SVMMat.png")
        pyplot.show()
        return

   
    
    

# ==========================================
#              MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    # 1. Read Data (Ensure 'emails.csv' exists in the same directory)
    data_obj = data_read_write("emails_exp7.csv")
    data_frame = data_obj.data_frame

    # 2. Exploratory Data Analysis (EDA)
    print("--- Data Head ---")
    print(data_frame.head())
    print("\n--- Data Tail ---")
    print(data_frame.tail())
    print("\n--- Data Description ---")
    print(data_frame.describe())
    print("\n--- Data Info ---")
    print(data_frame.info())

    print("\n--- Groupby Spam Describe ---")
    print(data_frame.groupby('spam').describe())

    # Add length column
    data_frame['length'] = data_frame['text'].apply(len)
    print("\nMax email length:", data_frame['length'].max())

    # Plot length distribution
    sns.set_theme(rc={'figure.figsize': (11.7, 8.27)})
    ham_messages = data_frame[data_frame['spam'] == 0]
    spam_messages = data_frame[data_frame['spam'] == 1]

    ham_messages['length'].plot(bins=100, kind='hist', label='Ham', alpha=0.6)
    spam_messages['length'].plot(bins=100, kind='hist', label='Spam', alpha=0.6)
    plt.title('Distribution of Length of Email Text')
    plt.xlabel('Length of Email Text')
    plt.legend()
    plt.show()

    # Number of words analysis
    ham_words_length = [len(word_tokenize(title)) for title in ham_messages.text.values]
    spam_words_length = [len(word_tokenize(title)) for title in spam_messages.text.values]
    print("Max words in Ham:", max(ham_words_length))
    print("Max words in Spam:", max(spam_words_length))

    sns.histplot(ham_words_length, stat="density", bins=30, label='Ham', color='blue', alpha=0.5, kde=True)
    sns.histplot(spam_words_length, stat="density", bins=30, label='Spam', color='red', alpha=0.5, kde=True)
    plt.title('Distribution of Number of Words')
    plt.xlabel('Number of Words')
    plt.legend()
    plt.savefig("SVMGraph.png")
    plt.show()

    # Mean word length analysis
    def mean_word_length(x):
        words = word_tokenize(x)
        if len(words) == 0:
            return 0
        return np.mean([len(word) for word in words])

    ham_meanword_length = ham_messages.text.apply(mean_word_length)
    spam_meanword_length = spam_messages.text.apply(mean_word_length)

    sns.histplot(ham_meanword_length, stat="density", bins=30, label='Ham', color='blue', alpha=0.5, kde=True)
    sns.histplot(spam_meanword_length, stat="density", bins=30, label='Spam', color='red', alpha=0.5, kde=True)
    plt.title('Distribution of Mean Word Length')
    plt.xlabel('Mean Word Length')
    plt.legend()
    plt.savefig("Graph.png")
    plt.show()

    # Stopwords ratio analysis
    stop_words_set = set(stopwords.words('english'))

    def stop_words_ratio(x):
        words = word_tokenize(x)
        num_total_words = len(words)
        if num_total_words == 0:
            return 0
        num_stop_words = sum(1 for word in words if word.lower() in stop_words_set)
        return num_stop_words / num_total_words

    ham_stopwords = ham_messages.text.apply(stop_words_ratio)
    spam_stopwords = spam_messages.text.apply(stop_words_ratio)

    sns.histplot(ham_stopwords, stat="density", label='Ham', color='blue', alpha=0.5, kde=True)
    sns.histplot(spam_stopwords, stat="density", label='Spam', color='red', alpha=0.5, kde=True)

    print('Ham Mean Stopword Ratio: {:.3f}'.format(ham_stopwords.values.mean()))
    print('Spam Mean Stopword Ratio: {:.3f}'.format(spam_stopwords.values.mean()))
    plt.title('Distribution of Stop-word Ratio')
    plt.xlabel('Stop Word Ratio')
    plt.legend()
    plt.show()

    # Class balance analysis
    data_frame['Ham(0) and Spam(1)'] = data_frame['spam']
    print('Spam percentage =', (len(spam_messages) / len(data_frame)) * 100, "%")
    print('Ham percentage =', (len(ham_messages) / len(data_frame)) * 100, "%")

    plt.figure(figsize=(8, 6))
    sns.countplot(data=data_frame, x='Ham(0) and Spam(1)', hue='Ham(0) and Spam(1)', legend=False)
    plt.title('Count of Ham vs Spam')
    plt.show()

    # 3. Data Cleaning
    print("Cleaning text data... this may take a moment.")
    data_clean_obj = data_cleaning()
    data_frame['clean_text'] = data_clean_obj.apply_to_column(data_frame['text'])
    print(data_frame[['text', 'clean_text']].head())

    # Save processed Data
    data_obj.data_frame = data_frame
    data_obj.write_to_csvfile("processed_file.csv")
    print("Processed file saved as 'processed_file.csv'")

    # 4. Feature Extraction and Model Training
    cv_object = apply_embeddding_and_model()
    spamham_countvectorizer = cv_object.apply_count_vector(data_frame['clean_text'])
    
    X = spamham_countvectorizer
    y = data_frame['spam'].values

    print("Training SVM and evaluating...")
    cv_object.apply_svm(X, y)
