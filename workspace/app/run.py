import json
import plotly
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine, func, select, MetaData
from sqlalchemy import *

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from plotly.graph_objs import Bar, Heatmap
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot




app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#engine = 'data/DisasterResponse.db.backends.sqlite3'
#name = 'data/DisasterResponse/data.sqlite3'
engine = create_engine('sqlite:///../DisasterResponse.db')
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
#engine = create_engine('sqlite:///./data/DisasterResponse.db')
#df = pd.read_sql_table('messages', engine)
#engine = create_engine('sqlite:///' + database_filepath)
#conn = sqlite3.connect('DisasterResponse.db')    
#df= pd.read_sql('SELECT * FROM messages;', engine)
df= pd.read_sql('SELECT * FROM messages;', engine)
#cursor = conn.cursor()
#df = cursor.execute('SELECT * FROM messages')


# load model
#model = joblib.load("../models/your_model_name.pkl")
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Show distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column_name in category:
        category_counts.append(np.sum(df[column_name]))

    
    # extract data Top 10
    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = list(categories.sum().sort_values(ascending=False)[1:11].index)
    
    cats = df[df.columns[5:]]
    cats_counts = cats.mean()*cats.shape[0]

    nlarge_counts = cats_counts.nlargest(5)
    nlarge_names = list(nlarge_counts.index)
    
    dfnew = df.drop(['id', 'message', 'original', 'genre','child_alone'], axis = 1)
    corr_list = []
    correl = dfnew.corr().values
    for row in correl:
        corr_list.append(list(row))
    
    col_names = [col.replace('_', ' ').title() for col in dfnew.columns]
    
    # Calculate proportion of each category with label = 1
    cat_props = df.drop(['id', 'message', 'original', 'genre','child_alone'], axis = 1).sum()/len(df)
    cat_props = cat_props.sort_values(ascending = False)
    cat_names = list(cat_props.index)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Top Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=nlarge_names,
                    y=nlarge_counts
                )
            ],

            'layout': {
                'title': 'Top message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr_list, 
                    x=col_names,
                    y=col_names,
                    colorscale='Viridis',
                )
            ],

            'layout': {
                'title': 'What Types of Messages Occur Together?',
                'height': 750,
                'margin': dict(
                    l = 150,
                    r = 30, 
                    b = 160,
                    t = 30,
                    pad = 4
                    ),
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_props
                )
            ],

            'layout': {
                'title': 'Proportion of Messages by Category',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        }
   ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()