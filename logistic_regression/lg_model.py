# Logistic regresion model for a fake advertising data set.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')


# Using seaborn will explore the data to select important data for the model.
# Uncomment each plot and show function to see results.

# Count of people by age.
# sns.histplot(data=ad_data, x='Age')

# Kde distributions of daily time spent on site vs age.
# sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site', kind='kde', color='red')

# Pairplot with the hue defined by the 'clicked on ad' column.
# sns.pairplot(data=ad_data, hue='Clicked on Ad', height=2)

# plt.show()

# Model fit and training.

# We want to predict if the person will click on the ad, so the Y variable will be 'Clicked on Ad'
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = ad_data['Clicked on Ad']

# Test Training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ad_regression = LogisticRegression()
ad_regression.fit(X=X_train, y=y_train)

# Create a prediction with the test data for X. The result should be close to Y test.
prediction = ad_regression.predict(X_test)

print(classification_report(y_true=y_test, y_pred=prediction))
