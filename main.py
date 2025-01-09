import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("obesity_dataset.csv")
data["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"].replace(['Male','Female'],[0,1],inplace=True)


print(data)
x = data[["family_history_with_overweight", "FAF", "SCC", "CAEC"]].values
y = data[["NObeyesdad"]].values

