import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

weight_classes = [
    "Insufficient_Weight",
     "Normal_Weight",
     "Overweight_Level_I",
     "Overweight_Level_II",
     "Obesity_Type_I",
     "Obesity_Type_II",
     "Obesity_Type_III"
]

data = pd.read_csv("obesity_dataset.csv")
data["NObeyesdad"].replace(
    weight_classes,
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

predict = model.predict(x_test)

avg_diff = 0

def mapnames(x):
    return weight_classes[x]

for index in range(len(x_test)):
    actual = y_test[index][0]
    predicted_y = predict[index]
    x_coord = x_test[index]

    actual_name = mapnames(actual)
    predicted_y_name = mapnames(predicted_y)

    print(f"Actual: {actual_name} Predicted: {predicted_y_name}")
    if actual == predicted_y:
        print("Correct!")
    else:
        avg_diff += abs(predicted_y - actual)
        print("Incorrect. Difference: " + str(abs(predicted_y - actual)))

print("average difference: " + str(avg_diff/len(x_test)))