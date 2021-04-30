import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=1978)

nltk.download("punkt")
nltk.download("wordnet")


def load_data(database_filepath):
    """
       Function:
       load data from database
       Args:
       database_filepath: the path of the database
       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
       """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_responses", engine)
    X = df["message"] 
    y = df.drop(["id", "message", "original", "genre", "child_alone"], axis=1)
    return X, y


def tokenize(text):
    """
    Function:
      split text into words and return the root form of the words

      Args:
      text (str) : content of the messages

      Return: a list of the root form of the message words
    """
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # clean through each token
    return [lemmatizer.lemmatize(tok.lower().strip(), pos="v") for tok in tokens]


def build_model():
    """
    Function: build a model for classifing the disaster messages

    Return: cv(list of str): classification model
    """
    # Create pipeline
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier())),
        ]
    )

    # Create Grid search parameters
    parameters = {
        "clf__estimator__n_jobs": [2, 4],
        "clf__estimator__verbose": [1, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
    Args:
    model: the classification model
    X_test: test messages
    Y_test: test target
    """

    y_pred = model.predict(X_test)
    i = 0
    for col in y_test:
        print("Feature {}: {}".format(i + 1, col))
        print(classification_report(y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == y_test.values).mean()
    print("The model accuracy is {:.3f}".format(accuracy))


def save_model(model, model_filepath):
    """
    Function: Save pickle file of the model

    Args:
    model: the classfication model
    model_filepath: the path of the pickle file
    """

    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, y_test)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
