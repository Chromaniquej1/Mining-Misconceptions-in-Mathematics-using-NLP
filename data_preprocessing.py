import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(data):
    def clean_text(text):
        text = str(text).lower()
        text = text.replace('\\(', '').replace('\\)', '')
        text = text.replace('[^a-zA-Z0-9]', ' ')
        return text

    for col in ['QuestionText', 'AnswerAText', 'AnswerBText', 'AnswerCText', 'AnswerDText']:
        data[col] = data[col].apply(clean_text)

    data[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']] = \
        data[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].fillna(0).astype(int)

    return data

def vectorize_data(data):
    combined_texts = data['QuestionText'] + ' ' + data['AnswerAText'] + ' ' + \
                      data['AnswerBText'] + ' ' + data['AnswerCText'] + ' ' + \
                      data['AnswerDText']

    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(combined_texts).toarray()

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(data[['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']].values.tolist())

    return X, y, vectorizer, mlb

def preprocess_data(file_path):
    data = load_data(file_path)
    data = clean_data(data)
    return vectorize_data(data)
