import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score , mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn import utils






data = pd.read_csv('fruits.csv')
data = data.replace('\.+', np.NaN, regex=True)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


df=pd.DataFrame(data) 


lab = preprocessing.LabelEncoder()
df['mass'] = lab.fit_transform(df['mass'])
df['color_score'] = lab.fit_transform(df['color_score'])


# df.info()

# df.notnull().sum()


# df.describe()

# df.isnull().sum()

# df.isnull()




# plt.figure(0)
# df['fruit_name'].value_counts().plot.pie(shadow=True)
# plt.figure(1)
# df['fruit_subtype'].value_counts().plot.pie(shadow=True)

# plt.scatter(df['fruit_label'], df['color_score'], c ="black")
# plt.xlabel("fruit_label")
# plt.ylabel('color_score')


# plt.scatter(df['fruit_label'], df['mass'], c ="brown")
# plt.xlabel("fruit_label")
# plt.ylabel('mass')


# plt.scatter(df['fruit_label'], df['width'], c ="blue")

# plt.xlabel("fruit_label")
# plt.ylabel('width')

# plt.scatter(df['fruit_label'], df['height'], c ="green")
# plt.xlabel("fruit_label")
# plt.ylabel('height')


# lines = df.plot.line()

# plt.xlabel("indices")
# plt.ylabel('Values')
# plt.title("Data of Fruits")

# splitting train and test data in 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(df['mass'], df['color_score'], test_size=0.20, random_state=0)

# x_train, x_test, y_train, y_test = train_test_split(df['Field of Study'], df['Discount on Fees'], test_size=0.30, random_state=0)


x_train = x_train.values.reshape(len(x_train), 1)
x_train.shape

x_test = x_test.values.reshape(len(x_test), 1)
x_test.shape

# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# model.score(x_test,y_test)
# RMSE = np.sqrt( mean_squared_error(y_test, y_pred))

# r2 = r2_score(y_test, y_pred)

svc = SVC()
svc.fit(x_train, y_train)
Y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc

# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(x_train, y_train)
# Y_pred = knn.predict(x_test)
# acc_knn = round(knn.score(x_train, y_train) * 100, 2)
# acc_knn

# gaussian = GaussianNB()
# gaussian.fit(x_train, y_train)
# Y_pred = gaussian.predict(x_test)
# acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
# acc_gaussian


plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, svc.predict(x_train), color='r')
plt.xlabel("color_score")
plt.ylabel('mass')
plt.title("Fruit Data with GaussianNB regression model ")
plt.show()