## Disaster Response Pipelines
### Table of Contents

[Installation](#Installation)

[Project Motivation](#Motivation)

[Project Descriptions](#Descriptions)

[Files Descriptions](#Description)

[Instructions](#Instructions)

### Installation<a name="Installation"></a>

All libraries are available in Anaconda distribution of Python. The used libraries are:

- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

### Project Motivation<a name="Motivation"></a>

The goal of the project is to analyze the disaster data from [Figure Eight](https://appen.com/) in order to build a model that classifies the disaster messages into categories. Through a web app, the user can input a new message and get classification results in several categories. This project will help people and organization quickly categorize messages into categories in the event of disaster

### Project Descriptions<a name="Description"></a>

The project has three componants which are:

#### ETL Pipeline: [process_data.py](https://github.com/Suveesh/Disaster-Response-Pipeline/blob/main/data/process_data.py) file contain the script to create ETL pipline which:

- [x] Loads the messages and categories datasets.
- [x] Merges the two datasets
- [x] Cleans the data
- [x] Stores the data in a SQLite database

#### ML Pipeline: [train_classifier.py](https://github.com/Suveesh/Disaster-Response-Pipeline/blob/main/model/train_classifier.py) file contain the script to create ML pipline which:

- [x] Loads data from the SQLite database
- [x] Splits the dataset into training and test sets
- [x] Builds a text processing and machine learning pipeline
- [x] Trains and tunes a model using GridSearchCV
- [x] Outputs results on the test set
- [x] Exports the final model as a pickle file

#### Flask Web App: the web app enables the user to enter a disaster message, and then view the categories of the message.

- [x] The web app also contains some visualizations that describe the data.

### Files Descriptions

#### The files structure is arranged as below:

- README.md: read me file
- Project Components.txt: Project requirements to complete
- workspace
	- \app
		- run.py: flask file to run the app
	- \templates
		- master.html: main page of the web application 
		- go.html: result web page
	- \data
		- disaster_categories.csv: categories dataset
		- disaster_responses.csv: messages dataset
		- DisasterResponse.db: disaster response database
		- process_data.py: ETL process
	- \models
		- train_classifier.py: classification code
		- classifier.pkl: model pickle file

### Instructions<a name="Instruction"></a>

To execute the app follow the instructions:

   Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database python 'data/process_data.py data/disaster_messages.csv' 'data/disaster_categories.csv' 'data/DisasterResponse.db'
        To run ML pipeline that trains classifier and saves python 'models/train_classifier.py' 'data/DisasterResponse.db' 'models/classifier.pkl'

   Run the following command in the app's directory to run your web app. python run.py

   Go to http://0.0.0.0:3001/
 
  
