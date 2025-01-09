import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("obesity_dataset.csv")
data["NObeyesdad"].replace(
    ["Insufficient_Weight",
     "Normal_Weight",
     "Overweight_Level_I",
     "Overweight_Level_II",
     "Obesity_Type_I",
     "Obesity_Type_II",
     "Obesity_Type_III"],
     [0,1,2,3,4,5,6], inplace=True)

data["Gender"].replace(["Male", "Female"], [0, 1], inplace=True)
data["family_history_with_overweight"].replace(["no", "yes"], [0, 1], inplace=True)
data["FAVC"].replace(["no", "yes"], [0, 1], inplace=True)
data["CAEC"].replace(["no", "Sometimes", "Frequently", "Always"], [0, 1, 2, 3], inplace=True)
data["SMOKE"].replace(["no", "yes"], [0, 1], inplace=True)
data["SCC"].replace(["no", "yes"], [0, 1], inplace=True)
data["CALC"].replace(["no", "Sometimes", "Frequently", "Always"], [0, 1, 2, 3], inplace=True)

x = data[["Gender", "Age", "family_history_with_overweight", "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC"]].values
y = data[["NObeyesdad"]].values

scaler = StandardScaler().fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = linear_model.LogisticRegression().fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test))

# test example
# example = [[0, 45, 1, 0, 1, 3]]
# scaler.fit(example)
# example = scaler.transform(example)

# print(model.predict(example))

# fig, graph = plt.subplots(4)

# graph[0].scatter(data["family_history_with_overweight"], y)
# graph[0].set_xlabel("Family History With Overweight")
# graph[0].set_ylabel("Obesity Level")

# graph[1].scatter(data["FAF"], y)
# graph[1].set_xlabel("Regularity of Physical Activity")
# graph[1].set_ylabel("Obesity Level")

# graph[2].scatter(data["SCC"], y)
# graph[2].set_xlabel("Monitoring Caloric Intake")
# graph[2].set_ylabel("Obesity Level")

# graph[3].scatter(data["CAEC"], y)
# graph[3].set_xlabel("Eating Between Meals")
# graph[3].set_ylabel("Obesity Level")

# plt.tight_layout()
# plt.show()