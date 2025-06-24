import pandas as pd
import os
import json
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import GridSearchCV

#Function to stip json file of excess garbage string values
def parse_json_garbage(s):
    start_idx = next(idx for idx, c in enumerate(s) if c in "{[")
    s = s[start_idx:]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        return json.loads(s[:e.pos])
    

#Function to automate model training and metrics generation
def runner(json_file, model_file):
    # Load parameters from JSON
    directory = './JSONs/'
    with open(directory+json_file, 'r') as file:
        parameters = json.load(file)
        
    with open(directory+model_file, 'r') as file:
        model_param = json.load(file)

    model_name = parameters['model_name']
    df = pd.read_csv("./datasets/" + str(parameters['filename']))
    flag = parameters['flag']

    #Check for target_variable is present or not
    target_variable = parameters.get("target_variable", None)
    if target_variable is None:
        raise ValueError("Target variable not specified in the parameters.")

    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters['split'], random_state=42)

    def_param = {
      "decision_tree" : {"param_dict" : "default_decision_tree_parameters", "lib_name" : "DecisionTreeClassifier"},
      "svm" : {"param_dict" : "default_svm_parameters", "lib_name" : "SVC"},
      "logistic_regression" : {"param_dict" : "default_lr_parameters", "lib_name" : "LogisticRegression"}
    }

    params= def_param[model_name]["param_dict"]
    param = model_param[params]
    lib_name = def_param[model_name]["lib_name"]
    
    #NO hyperparameter tuning
    if flag == 0:
      # Merge default and user-provided parameters
        merged_parameters = {**eval(str(param)), **parameters.get("param", {})}
      # print(merged_parameters)

      # Initialize machine learning model with the merged parameters
        model = eval(lib_name)(**merged_parameters)

      # Train the model
        model.fit(X_train, y_train)
        para = parameters.get("param", {})

      # Make predictions on the test set
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return_dict = {"acc" : acc, "cr" : cr, "cm" : cm ,"paramters" : str(para)}
        return return_dict
    
    #Hyperparameter tuning
    else:
        param_grid = parameters.get("param", {})
        grid_search = GridSearchCV(eval(lib_name)(), param_grid, cv=2, scoring='accuracy')

        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        # Get the best parameters 
        best_params = grid_search.best_params_

        # Use the best parameters to train the final model
        final_model = grid_search.best_estimator_
        
        y_pred = final_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test,y_pred)
        cm = confusion_matrix(y_test, y_pred)

        return_dict = {"acc" : acc, "cr" : cr, "cm" : cm ,"paramters" : str(best_params)}
        return return_dict