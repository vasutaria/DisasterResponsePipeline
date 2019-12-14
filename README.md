
In this project,I analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, there are two datefiles containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that when someone send the messages it gets categorized and sent to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will my software skills, including my ability to create basic data pipelines and write clean, organized code!


# DisasterResponsePipeline files in the workspace and what they are.

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
