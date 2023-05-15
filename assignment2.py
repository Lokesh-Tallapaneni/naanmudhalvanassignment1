
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

df = pd.read_csv("/content/Housing.csv")

# List of columns to encode
columns_to_encode = ['mainroad', 'basement', 'guestroom', 'hotwaterheating', 'airconditioning', 'furnishingstatus']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode each column in the DataFrame
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

df.head(10)

df=df.drop(df.loc[(df['area']<6800) & (df['price']>8000000)].index)
df=df.drop(df.loc[(df['area']<6800) & (df['price']>7000000)].index)
df=df.drop(df.loc[(df['bedrooms']>5) & (df['price']<9000000)].index)
df=df.drop(df.loc[(df['bathrooms']==3) & (df['price']<8000000)].index)
df=df.drop(df.loc[(df['parking']==0) & (df['price']<7000000)& (df['stories']<=1)].index)
df=df.drop(df.loc[(df['airconditioning']==0) & (df['price']>6000000)].index)
df=df.drop(df.loc[(df['furnishingstatus']==2) & (df['price']>7000000)].index)
df=df.drop(df.loc[(df['area']<8000) & (df['guestroom']==0)&(df['basement']==0)&(df['hotwaterheating']==0)&(df['price']>8000000)].index)
df.shape

area = df['stories']
price = df['area']
plt.scatter(area, price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Scatter Plot: Area vs Price')
plt.show()

# Univariate visualization using histogram
df['area'].hist(bins=50)
plt.xlabel('area')
plt.ylabel('acres')
plt.show()

# Bivariate analysis using scatter plot
sns.scatterplot(x='bedrooms', y='price', data = df)

# Multivariate visualization using scatter plot matrix
sns.pairplot(df[["bedrooms", "bathrooms", "area", "price"]])

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
print(X_train)


# =============================================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Predict
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)

# ==========================================================
from sklearn.tree import DecisionTreeClassifier

# Create the model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)
# ===========================================================

from sklearn.ensemble import RandomForestClassifier

# Create the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)
# ======================================================

from sklearn.svm import SVC

# Create the model
model = SVC()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)
# ========================================================

from sklearn.naive_bayes import GaussianNB

# Create the model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)
# ======================================================

from sklearn.neighbors import KNeighborsClassifier

# Create the model
model = KNeighborsClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy*100)
# ===========================