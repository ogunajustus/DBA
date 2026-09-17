# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/tourism.csv")

# Encode categorical columns
label_encoder = LabelEncoder()
for col in ["TypeofContact", "CityTier", "Occupation", "Gender", "MaritalStatus", "ProductPitched", "Designation"]:
    df[col] = label_encoder.fit_transform(df[col])

# Target column
target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
