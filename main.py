import json
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

lemmatizer = WordNetLemmatizer()


# Prepare and clean raw text
def preprocess(sentence):
    # sentence = sentence.replace('.', '')
    # sentence = sentence.replace(',', '')
    sentence = re.sub(r'[.,|]', '', sentence)
    sentence = sentence.lower()
    tokens = sentence.split(' ')
    filtered_words = [w for w in tokens if len(w) > 2 if w not in stopwords.words('english')]
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


# Gets reviews file name as an input and return Dataframe
# with 2 columns:  'overall' , 'reviewText' and their values from file
def get_dataframe_from_reviews_file(reviews_file):
    with open(reviews_file, 'r') as f:
        file_data = f.read()

    raw_reviews = file_data.split('\n')
    reviews = []
    for review in raw_reviews:
        if review:
            review_dict = json.loads(review)
            # Extract the values associated with the desired keys
            if "reviewText" in review_dict.keys() and "summary" in review_dict.keys() and "overall" in review_dict.keys():
                # We care only about overall, reviewText and summary values from each review.
                filtered_dict = {
                    "overall": review_dict["overall"],
                    # 'reviewText' and 'summery'  values will be both under reviewText.
                    "reviewText": review_dict["reviewText"] + ' ' + review_dict["summary"]
                }
                # Add the filtered dictionary to the list of dictionaries
                reviews.append(filtered_dict)

    # Create Dataframe where each review represent single row
    # Dataframe columns are:  1.'overall' , 2.'reviewText'
    df = pd.DataFrame.from_dict(reviews)
    return df


def print_best_k_features(text_clf, k):
    # Get the SelectKBest object from the pipeline

    # selector = text_clf.named_steps['select_k_best']
    selector = SelectKBest(k=15)

    # Use the get_support method to get a boolean mask of the selected features
    mask = selector.get_support()

    # Get the feature names from the count Vectorizer
    feature_names = text_clf.named_steps['count_vect'].get_feature_names_out()

    # Print the names of the selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    print('Selected features: \n', selected_features)


def calc_f1_avg(res):
    sum = 0.0
    for element in res:
        sum += element
    avg = sum / 5
    return avg


def classify(train_file, test_file):
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    # Create train and test dataframes to manage the data
    df_train = get_dataframe_from_reviews_file(train_file)
    df_test = get_dataframe_from_reviews_file(test_file)
    df_train['reviewText'] = df_train['reviewText'].apply(lambda s: preprocess(s))
    df_test['reviewText'] = df_test['reviewText'].apply(lambda s: preprocess(s))

    clf = Pipeline([('count_vect', CountVectorizer(ngram_range=(1, 2), max_features=1000)),
                    # ('select_k_best', SelectKBest(k=15)),
                    ('tfidf_transformer', TfidfTransformer()),
                    ('clf', LogisticRegression(max_iter=1000, random_state=0))
                    ])

    x_train = df_train['reviewText']
    x_test = df_test['reviewText']
    y_train = df_train['overall']
    y_test = df_test['overall']

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    f1_scores_list = f1_score(y_test, y_predict, average=None)
    print('Confusion matrix: \n', confusion_matrix(y_test, y_predict))

    test_results = {'class_1_F1': f1_scores_list[0],
                    'class_2_F1': f1_scores_list[1],
                    'class_3_F1': f1_scores_list[2],
                    'class_4_F1': f1_scores_list[3],
                    'class_5_F1': f1_scores_list[4],
                    'accuracy': calc_f1_avg(f1_scores_list)}

    # print_best_k_features(clf, 15)
    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    results = classify(config['train_data'], config['test_data'])
    for k, v in results.items():
        print(k, v)
