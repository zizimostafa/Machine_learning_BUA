from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Decision Tree — full vs pruned (max_depth=3)
#Full Tree
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, full_tree.predict(X_train))
test_acc_full = accuracy_score(y_test, full_tree.predict(X_test))

print("Full Decision Tree:")
print("Train Accuracy:", train_acc_full)
print("Test Accuracy:", test_acc_full)
print()

#pruned tree
pruned_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
pruned_tree.fit(X_train, y_train)

train_acc_pruned = accuracy_score(y_train, pruned_tree.predict(X_train))
test_acc_pruned = accuracy_score(y_test, pruned_tree.predict(X_test))

print("Pruned Decision Tree (max_depth=3):")
print("Train Accuracy:", train_acc_pruned)
print("Test Accuracy:", test_acc_pruned)
print()


# Train a Random Forest (100 trees)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, rf.predict(X_test))

print("Random Forest (100 trees):")
print("Train Accuracy:", train_acc_rf)
print("Test Accuracy:", test_acc_rf)
print()


#Train a Gradient Boosting model
learning_rates = [0.01, 0.1]
estimators = [50, 100, 200]

print("Gradient Boosting Results:\n")

for lr in learning_rates:
    for n in estimators:
        gb = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42)
        gb.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, gb.predict(X_train))
        test_acc = accuracy_score(y_test, gb.predict(X_test))

        print(f"learning_rate={lr}, n_estimators={n}  -->  Train: {train_acc:.4f}, Test: {test_acc:.4f}")


#learning_rates = [0.01, 0.1]
estimators = [50, 100, 200]

print("Gradient Boosting Results:\n")

for lr in learning_rates:
    for n in estimators:
        gb = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42)
        gb.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, gb.predict(X_train))
        test_acc = accuracy_score(y_test, gb.predict(X_test))

        print(f"learning_rate={lr}, n_estimators={n}  -->  Train: {train_acc:.4f}, Test: {test_acc:.4f}")



#learning_rates = [0.01, 0.1]
estimators = [50, 100, 200]

print("Gradient Boosting Results:\n")

for lr in learning_rates:
    for n in estimators:
        gb = GradientBoostingClassifier(learning_rate=lr, n_estimators=n, random_state=42)
        gb.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, gb.predict(X_train))
        test_acc = accuracy_score(y_test, gb.predict(X_test))

        print(f"learning_rate={lr}, n_estimators={n}  -->  Train: {train_acc:.4f}, Test: {test_acc:.4f}")


#For Random Forest and Gradient Boosting, print the top 5 feature importances
#Random Forest
rf_importances = rf.feature_importances_
rf_top_idx = np.argsort(rf_importances)[::-1][:5]

print("\nTop 5 Random Forest features:")
for i in rf_top_idx:
    print(data.feature_names[i], ":", rf_importances[i])

#Gradiant Boosting
gb_final = GradientBoostingClassifier(random_state=42)
gb_final.fit(X_train, y_train)
gb_importances = gb_final.feature_importances_
gb_top_idx = np.argsort(gb_importances)[::-1][:5]

print("\nTop 5 Gradient Boosting features:")
for i in gb_top_idx:
    print(data.feature_names[i], ":", gb_importances[i])

