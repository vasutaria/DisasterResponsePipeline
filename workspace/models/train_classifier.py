import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, func, select, MetaData
from sqlalchemy import *
import nltk
nltk.download(['punkt', 'wordnet'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """Load the filepath and return the data"""
    engine = create_engine('sqlite:///' + database_filepath)
    conn = sqlite3.connect('DisasterResponse.db')    
    df= pd.read_sql('SELECT * FROM messages;', conn)
    df.head()
    X = df['message'].values
    y = df.loc[:, 'related':'direct_report'].values
   category_names = (df.columns[4:]).tolist()
    return X, y, category_names
    #pass


def tokenize(text):
    """tokenize and transform input text. Return cleaned text"""
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    #pass


def build_model():
    """Return Grid Search model with pipeline and Classifier
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    """
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', moc)
    ])
    
    """pipeline.get_params()"""
    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    return cv
    #pass


def evaluate_model(model, X_test, y_test, category_names):
    """Print model results

    INPUT
    model -- required, estimator-object
    X_test -- required
    y_test -- required
    category_names = required, list of category strings

    OUTPUT
    Does not return any data
    """
    

    y_pred = model.predict(X_test)
    print("----Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(y_test[:, i], y_pred[:, i]))

    #pass


def save_model(model, model_filepath):
    """
    Saves the model to disk
    INPUT 
        model: The model we need to save
        model_filepath: filepath where the model needs to be saved
    OUTPUT
        none.  save model pickle file
    """   
    pickle.dump(model, open(model_filepath, 'wb'))
     

    #pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()