import pandas as pd
import numpy as np
from sqlalchemy import create_engine 
import sys


def get_data(messages_filepath,categories_filepath):
    """
    Load and process data from 'messages.csv' and 'categories.csv'.
    
    Steps:
    1. Load messages and categories datasets.
    2. Merge datasets based on 'id'.
    3. Expand 'categories' column into separate category columns.
    4. Split each category into name and binary value (0 or 1).
    5. Merge the processed categories back into the main dataframe.
    
    Returns:
        DataFrame: Processed data containing messages and categorized labels.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages, categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories_expanded = categories['categories'].str.split(";", expand=True)

    # Split each of the new columns into category and value columns
    for column in categories_expanded.columns:
        # Split each column based on the hyphen
        split_columns = categories_expanded[column].str.split("-", expand=True)
    
        # Assign the split values to new columns in the original dataframe
        categories[split_columns[0].iloc[0]] = split_columns[1]

    # Drop the original 'categories' column
    categories.drop('categories', axis=1, inplace=True)
    for column in categories.columns:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    categories.drop('id', axis=1, inplace=True)
    merged_df.drop('categories', axis=1, inplace=True)
    merged_df.drop('original', axis=1, inplace=True)
    merged_df.drop('genre', axis=1, inplace=True)
    data = pd.concat([merged_df, categories], axis=1)
    #pd.merge(merged_df,categories, on= 'id')
    #Filter out rows with "2" values in category columns
    data = data[(data != 2).all(axis=1)]
    return data

def drop_dups(df):
    """
    Remove duplicate rows from a DataFrame and display related statistics.
    
    Args:
        df (DataFrame): The input dataframe to process.
    
    Prints:
        - Number of duplicate rows.
        - Initial length of the dataframe.
        - Length of dataframe after removing duplicates.
    
    Returns:
        DataFrame: The dataframe after duplicates are removed.
    """
    duplicates = df[df.duplicated()]
    print('number of duplicates:', len(duplicates))
    print('dataframe length',len(df))
    print('dataframe size duplicates removed', len(df.drop_duplicates()))
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        dat = get_data(messages_filepath,categories_filepath)
        de_dup = drop_dups(dat)
        engine = create_engine('sqlite:///{database_filepath}')
        de_dup.to_sql('messages_categories',  engine, index=False, if_exists='replace')

if __name__ == "__main__":
    main()
