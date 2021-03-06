import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
      Function:
      Load data from two csv file and then merge them

      Args:
      messages_filepath (str): the file path of messages csv file
      categories_filepath (str): the file path of categories csv file

      Return:
      df (DataFrame): A dataframe of messages and categories
      """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
      Function:
      clean the Dataframe df

      Args:
      df (DataFrame): A dataframe of messages and categories need to be cleaned

      Return:
      df (DataFrame): A cleaned dataframe of messages and categories
      """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    categories.columns = row.apply(lambda x: x[:-2])

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column].str[-1])
        
    # based on the figure 8 documentation original mapping is the following: 1 - yes, 2 - no, so I will convert all the 2's to 0's
    categories['related'] = categories['related'].replace(2, 0)

    # replace the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], sort=False, axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
       Function:
       Save the Dataframe df in a database

       Args:
       df (DataFrame): A dataframe of messages and categories
       database_filename (str): The file name of the database
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("disaster_responses", engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
