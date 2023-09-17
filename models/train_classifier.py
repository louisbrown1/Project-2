import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('nltk.download')
import re
import numpy as np
import pandas as pd
import pickle
import sys
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sqlalchemy import create_engine


# load data from database
def load_data(database_filepath):
    """
    Load features and labels from SQLite database.
    
    Parameters:
    ----------
    database_filepath : str
        Path to SQLite database.
        
    Returns:
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        Features and labels DataFrames.
        
    Requires SQLAlchemy and pandas.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    with engine.begin() as conn:
        df = pd.read_sql_table('messages_categories',conn)
    X = df.iloc[:,:2]
    y = df.iloc[:,2:]
    X.drop('id',  axis=1, inplace=True)

    return X, y



#tokenizer for text.

def tokenize(text):
    """
    Tokenize and preprocess a text string.
    
    Parameters:
    -----------
    text : str
        The input text to tokenize.
        
    Returns:
    --------
    list
        List of preprocessed and tokenized words.
    
    Dependencies:
    -------------
    Requires nltk's word_tokenize, stopwords, and WordNetLemmatizer.
    """
        
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    STOPWORDS = set(stopwords.words('english'))
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def train_model():
    """
    Create a pipeline and parameter grid for text classification using Grid Search.
    
    Returns:
    -------
    GridSearchCV
        Configured GridSearchCV object ready for fitting.
    
    Requires scikit-learn and custom 'tokenize' function.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_features=None, max_df=0.5)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    parameters = {
    'clf__estimator__n_estimators': [150,200],
    'clf__estimator__min_samples_split': [3, 4, 8, 10],
    #'clf__estimator__max_depth': [100, 110, 150,None],
    #'vect__ngram_range': ((1, 1), (1, 2)),
    #'#vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 2500, 5000),
    'tfidf__use_idf': (True, False)
    }

    # create grid search object
    cv =  GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2)

    return cv

def evaluate_model(predicted, y_test):
    """
    Evaluate a multi-output classification model by printing classification reports.
    
    Parameters:
    ----------
    predicted : np.ndarray
        Array of predicted labels.
    y_test : pd.DataFrame
        DataFrame of true labels.
        
    Outputs classification reports to stdout.
    
    Requires scikit-learn's classification_report.
    """
        
    for index, column in enumerate(y_test):
        print(column, classification_report(y_test[column], predicted[:, index]))


def export_model(model, model_filepath):
    """ Exports the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))

def run_pipeline():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

    X,y = load_data(database_filepath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    X_train = X_train.squeeze()
    X_test = X_test.squeeze()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    print('............model build............')
    model = train_model()

    print('............train..................')
    trained_mod = model.fit(X_train, y_train)

    print('..........test model...............')
    predicted = trained_mod.best_estimator_.predict(X_test)

    print('.......... model outcomes..........')
    evaluate_model(predicted, y_test)

    print('Export model...\n    MODEL: {}'.format(model_filepath))
    export_model(trained_mod.best_estimator_,model_filepath)


if __name__ == "__main__":
    run_pipeline()