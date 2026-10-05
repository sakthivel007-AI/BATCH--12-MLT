import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv("parking_data.csv")

# Remove duplicate records
data.drop_duplicates(inplace=True)



day_map = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

traffic_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

occupancy_map = {
    "AVAILABLE": 0,
    "OCCUPIED": 1
}


data["Day"] = data["Day"].map(day_map)
data["Traffic"] = data["Traffic"].map(traffic_map)
data["PreviousOccupancy"] = data["PreviousOccupancy"].map(occupancy_map)
data["Occupancy"] = data["Occupancy"].map(occupancy_map)

# Remove missing values
data.dropna(inplace=True)


# Input features
X = data[[
    "SlotID",
    "Hour",
    "Day",
    "PreviousOccupancy",
    "Traffic"
]]

# Target
y = data["Occupancy"]




XTrain, XTemp, yTrain, yTemp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)




XValidation, XTest, yValidation, yTest = train_test_split(
    XTemp,
    yTemp,
    test_size=0.50,
    random_state=42,
    stratify=yTemp
)




model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(XTrain, yTrain)





validation_predictions = model.predict(XValidation)

validation_accuracy = accuracy_score(
    yValidation,
    validation_predictions
)




predictions = model.predict(XTest)

accuracy = accuracy_score(
    yTest,
    predictions
)




print("SMART PARKING AVAILABILITY SYSTEM")
print("---------------------------------")

print("Total records:", len(data))

print("Training records:", len(XTrain))
print("Validation records:", len(XValidation))
print("Testing records:", len(XTest))

print("\nData Split:")
print("Training   : 70%")
print("Validation : 15%")
print("Testing    : 15%")

print("\nValidation Accuracy:",
      round(validation_accuracy * 100, 2), "%")

print("Testing Accuracy:",
      round(accuracy * 100, 2), "%")


print("\nClassification Report:")

print(classification_report(
    yTest,
    predictions,
    target_names=["AVAILABLE", "OCCUPIED"],
    zero_division=0
))


print("Confusion Matrix:")

print(confusion_matrix(
    yTest,
    predictions
))




sample = XTest.iloc[[0]]

result = model.predict(sample)[0]

if result == 0:
    print("\nPredicted Parking Status: AVAILABLE")
else:
    print("\nPredicted Parking Status: OCCUPIED")