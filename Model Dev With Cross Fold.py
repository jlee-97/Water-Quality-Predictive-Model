# Final Model Script
# Model Development with Cross-Fold Validation
# ------------------------------------------------------------------------
# Loading libraries
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
# For imputing
from sklearn.impute import KNNImputer
# Models
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
from xgboost                 import XGBClassifier
from sklearn.ensemble        import RandomForestClassifier
# For modeling and evaluating
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics         import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics         import roc_auc_score
from sklearn.metrics         import roc_curve

# ------------------------------------------------------------------------
# Loading in the data
PATH = "C:\\Users\\jleer\\Documents\\BCIT\\COMP 4254\\assignments\\assign 2\\data\\"
FILE = "water_potability.csv"
dataset = pd.read_csv(PATH+FILE)

# ------------------------------------------------------------------------
# Imputing columns using the KNN imputing method
imputer = KNNImputer(n_neighbors=5) # Using the 5 closest neighbours
dataset = pd.DataFrame(imputer.fit_transform(dataset),
                         columns = dataset.columns)

# Printing the dataset
# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("Data Overview (After Imputing)\n1st 5 rows of data:")
print(dataset.head(5)) # Printing the 1st 5 records
print("\nDescriptive stats:")
print(dataset.describe()) # Printing descriptive stats for each col

# ------------------------------------------------------------------------
# Setting up the data for modeling
y = dataset['Potability'] # Target variable
X = dataset[['Solids', 'Sulfate', 'Hardness',
             'Organic_carbon', 'Chloramines']] # Predictor variables

# Printing the model data set
print("\nModelling Dataset\ny:")
print(y.head(5))
print("\nValue counts for target variable, y:")
print(dataset['Potability'].value_counts())
print("\nX:")
print(X.head(5))

# ------------------------------------------------------------------------
# Setting up the functions to:
#   - retrieve a list of models to fit
#   - evaluate models
#   - fit base models
#   - fit stacked models
#   - function that combines fitting the base and stacked models by
#     calling the functions that do those things separately
#   - function to evaluate the models with validation data

# Function to retrieve a list of models to fit:
def getUnfitModels():
    # Initializing an empty list
    models = list()

    # Appending logistic regr/classifier models to the list
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(GradientBoostingClassifier())
    models.append(XGBClassifier())
    models.append(RandomForestClassifier(n_estimators=10))

    # Returning the list to the calling function
    return models

# Function to evaluate models:
def evaluateModel(y_test, predictions, model):
    # Calculating model performance metrics
    precision = round(precision_score(y_test, predictions, zero_division=0),2)
    recall    = round(recall_score(y_test, predictions), 2)
    f1        = round(f1_score(y_test, predictions), 2)
    accuracy  = round(accuracy_score(y_test, predictions), 2)

    # Printing the metrics
    print("Precision:" + str(precision) + " Recall:" + str(recall) +\
          " F1:" + str(f1) + " Accuracy:" + str(accuracy) +\
          "   " + model.__class__.__name__)

# Function to fit the base models:
def fitBaseModels(X_train, y_train, X_test, y_test, models):
    # Initializing an empty dataframe
    dfPredictions = pd.DataFrame()

    print("Base:") # Indicator for fold evaluations

    # Fitting base model and store its predictions in dataframe
    # by looping through the list of models
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        evaluateModel(y_test, predictions, models[i])
        colName = str(i)

        # Adding base model predictions to column of data frame
        dfPredictions[colName] = predictions

    # Returning base model predictions and the base models
    return dfPredictions, models

# Function to fit the stacked model:
def fitStackedModel(X, y):
    # Using Ada Boost Classifier to fit the stacked model
    model = AdaBoostClassifier()
    model.fit(X, y)

    # Evaluating stacked model
    print("Stacked:")
    evaluateModel(y, model.predict(X), model)

    # Returning the model
    return model

# Function that handles fitting the base and stacked models
def fitAllModels(X,y):
    # Getting the list of unfit models
    unfitModels = getUnfitModels()
    # Initializing empty objects for the base and stacked models
    models = None
    stackedModel = None

    # Setting up cross-fold 3 folds
    kfold = KFold(n_splits=3, shuffle=True)
    count = 0
    # Converting y to dataframe
    y = y.to_frame()
    # Looping for each fold
    for train_index, test_index in kfold.split(X):
        # Splitting the data for the fold
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_train = np.array(y_train).reshape(-1, 1) # Reshaping the data
        y_train = np.ravel(y_train)
        y_test = y.loc[y.index.intersection(test_index), :]
        y_test = np.array(y_test).reshape(-1, 1)  # Reshaping the data
        y_test =np.ravel(y_test)

        # Fitting base and stacked models
        dfPredictions, models = fitBaseModels(X_train, y_train, X_test, y_test, unfitModels)
        stackedModel = fitStackedModel(dfPredictions, y_test)

        # Printing the current fold
        count += 1
        print("K-fold: " + str(count) + "\n")

    # Returning the base and stacked models
    return models, stackedModel

# Function to validate the base and stacked models with the validation set
def evaluateBaseAndStackModelsWithUnseenData(X, y, models, stackedModel):
    # Evaluating base models with validation data
    print("\nEvaluating base models:")
    # Initializing an empty dataframe
    dfValidationPredictions = pd.DataFrame()
    # Looping for the length of the models list
    for i in range(0, len(models)):
        predictions = models[i].predict(X) # Using the validation set (X) to make predictions
        colName = str(i)
        dfValidationPredictions[colName] = predictions # Putting the predictions into the dataframe
        evaluateModel(y, predictions, models[i]) # Evaluating the predictions with the validation set (y)

    # Evaluating stacked model with validation data
    # Using the predictions from the base models to make predictions using the stacked model
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\nEvaluating stacked model:")
    evaluateModel(y, stackedPredictions, stackedModel) # Evaluating the predictions
    # Using the predictions from the base models to get probability estimates using the stacked model
    stackedProb = stackedModel.predict_proba(dfValidationPredictions)
    auc = roc_auc_score(y, stackedProb[:, 1], )
    print('Logistic: ROC AUC=%.3f' % (auc))
    # Calculating roc curves
    lr_fpr, lr_tpr, _ = roc_curve(y, stackedProb[:, 1])
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    plt.plot([0, 1], [0, 1], '--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# ------------------------------------------------------------------------
# Building the model

# Splitting data into train, test, and validation sets
# Splitting X and y into testing and validation sets on a 30% testing/
# 70% training split, where the training set will be split when going
# through cross-fold to build the stacked model
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

# Calling the function to fit the base and stacked models
print("\nFitting Models")
models, stackedModel = fitAllModels(X_train, y_train)

# Calling the function to evaluate the models
print("\nEvaluating Models With Unseen Data (Validation Set)")
evaluateBaseAndStackModelsWithUnseenData(X_test, y_test, models, stackedModel)