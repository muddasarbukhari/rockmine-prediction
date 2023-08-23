import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)

sonar_data.head()

sonar_data.shape

sonar_data.describe()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

# seprating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)

# Training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,  test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape) # checking the shape of data after spliting

# Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = model.predict(X_train) # checking training prediction
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print(training_data_accuracy)

# accouracy on test data
X_test_prediction = model.predict(X_test)
test_data_accouracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accouracy)

# Making a Predictive System
input_data = (0.0039,0.0063,0.0152,0.0336,0.0310,0.0284,0.0396,0.0272,0.0323,0.0452,0.0492,0.0996,0.1424,0.1194,0.0628,0.0907,0.1177,0.1429,0.1223,0.1104,0.1847,0.3715,0.4382,0.5707,0.6654,0.7476,0.7654,0.8555,0.9720,0.9221,0.7502,0.7209,0.7757,0.6055,0.5021,0.4499,0.3947,0.4281,0.4427,0.3749,0.1972,0.0511,0.0793,0.1269,0.1533,0.0690,0.0402,0.0534,0.0228,0.0073,0.0062,0.0062,0.0120,0.0052,0.0056,0.0093,0.0042,0.0003,0.0053,0.0036,)

# coverting data type to numpy array
input_data_as_numpy_array = np.asarray (input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
  print("The object is a Rock")
else:
  print("The object is a Mine")