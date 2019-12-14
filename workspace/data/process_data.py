import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''Load dataframe from filepaths

    INPUT
    messages_filepath -- str, link to file
    categories_filepath -- str, link to file

    OUTPUT
    df - pandas DataFrame
    '''
    #messages = pd.read_csv(messages_filepath) 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df
    #pass


def clean_data(df):
    '''
    Create a clean, combined dataframe of messages and category dummy variables.

    Args:
        messages: DataFrame. Contains 'id' column for joining.
        categories: DataFrame. Contains 'id' column for joining and 
        'categories' column with strings of categories separated by ;.
    '''
    # merge datasets (1)
    #df = messages.merge(categories, on='id')
    #df = pd.merge(messages, categories, on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    #Row = 0 and column =1
    row = categories.loc[0]

    colnames = []
    for entry in row:
        colnames.append(entry[:-2])
    category_colnames = colnames

    categories.columns = category_colnames
    
    # convert category values to numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    categories.head()
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    #df["original"].fillna("no data", inplace = True)
    #df = df.drop(['original'], axis=1)
    return df
    #pass(2)


def save_data(df, database_filename):
    '''
    Save dataframe to database in 'messages' table. Replace any existing data.
    '''

    conn = sqlite3.connect('DisasterResponse.db')
    df.to_sql('messages', con=conn, if_exists='replace', index=False)
    engine = 'data/DisasterResponse.db.backends.sqlite3'
    name = 'data/DisasterResponse/data.sqlite3'
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()