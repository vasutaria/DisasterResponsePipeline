# DisasterResponsePipeline
In this Project Workspace, you'll find a data set containing real messages that were sent during disaster events. In this project I have created a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

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
