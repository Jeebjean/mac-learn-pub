import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler


def sensitivity_specificity_from_cm(cm):
	"""Compute sensitivity (TPR) and specificity (TNR) from confusion matrix.
	cm is expected as [[TN, FP],[FN, TP]]
	"""
	tn, fp, fn, tp = cm.ravel()
	sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
	return sensitivity, specificity


def main():
	# 1) Load dataset (run from the folder that contains `Titanic-Dataset.csv`)
	df = pd.read_csv("Titanic-Dataset.csv")
	print("Original shape:", df.shape)

	# 2) Delete rows with textual "na" / "NA"
	df_clean = df.replace(["na", "NA"], np.nan)
	relevant_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
	df_clean = df_clean.dropna(subset=relevant_cols)
	print("Cleaned shape:", df_clean.shape)

	# Convert numeric columns to numeric types (coerce errors)
	num_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
	for c in num_cols:
		df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
	# If conversion introduced NaNs, drop those rows
	df_clean = df_clean.dropna(subset=num_cols + ["Survived"]) 

	# 7) Modify `Sex` (binary) and `Embarked` (3 categories) so they can be used
	df_clean["Sex"] = df_clean["Sex"].str.lower().map({"male": 0, "female": 1})
	embarked_dummies = pd.get_dummies(df_clean["Embarked"], prefix="Embarked")
	df_clean = pd.concat([df_clean, embarked_dummies], axis=1)

	# Candidate predictors include numeric base predictors + Sex + all Embarked dummies
	candidate_predictors = num_cols + ["Sex"] + list(embarked_dummies.columns)

	# 4) Correlation-based feature selection: keep predictors with |corr| >= 0.10
	corr_df = df_clean[candidate_predictors + ["Survived"]].corr()
	corr_with_target = corr_df["Survived"].drop("Survived").round(3)

	print("\nCorrelations with Survived:")
	print(corr_with_target.to_string())

	selected_predictors = [p for p in candidate_predictors if abs(corr_with_target.get(p, 0)) >= 0.10]
	if not selected_predictors:
		print("\nNo predictors meet |corr| >= 0.10; using all candidate predictors.")
		selected_predictors = candidate_predictors
	else:
		print(f"\nSelected predictors (|corr| >= 0.10): {selected_predictors}")

	# 3) Prepare X and y and split (use random_state=1 as requested)
	X = df_clean[selected_predictors]
	y = df_clean["Survived"].astype(int)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	# Train logistic regression (set large max_iter to avoid convergence warnings)
	model = LogisticRegression(max_iter=100000)
	model.fit(X_train, y_train)

	# 5) Predictions (should be 0s and 1s)
	y_pred = model.predict(X_test)
	print("\nPredictions sample:", y_pred[:10])

	# Evaluate
	acc = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)
	sens, spec = sensitivity_specificity_from_cm(cm)

	print(f"\nLogistic Regression (before balancing) accuracy: {acc:.4f}")
	print("Confusion matrix:")
	print(cm)
	print(f"Sensitivity (TPR): {sens:.4f}")
	print(f"Specificity (TNR): {spec:.4f}")

	# 6) Baseline: all ones prediction
	ones_pred = np.ones_like(y_test)
	acc_ones = accuracy_score(y_test, ones_pred)
	cm_ones = confusion_matrix(y_test, ones_pred)
	sens_ones, spec_ones = sensitivity_specificity_from_cm(cm_ones)
	print(f"\nAll-ones baseline accuracy: {acc_ones:.4f}")
	print("All-ones confusion matrix:")
	print(cm_ones)

	# 10) Balance the training set using RandomOverSampler (random_state=1)
	ros = RandomOverSampler(random_state=1)
	X_res, y_res = ros.fit_resample(X_train, y_train)
	print(f"\nAfter oversampling, training set class distribution: {np.bincount(y_res)}")

	# Retrain model on balanced data
	model_bal = LogisticRegression(max_iter=100000)
	model_bal.fit(X_res, y_res)
	y_pred_bal = model_bal.predict(X_test)

	acc_bal = accuracy_score(y_test, y_pred_bal)
	cm_bal = confusion_matrix(y_test, y_pred_bal)
	sens_bal, spec_bal = sensitivity_specificity_from_cm(cm_bal)

	print(f"\nLogistic Regression (after balancing) accuracy: {acc_bal:.4f}")
	print("Confusion matrix (after balancing):")
	print(cm_bal)
	print(f"Sensitivity (TPR) after balancing: {sens_bal:.4f}")
	print(f"Specificity (TNR) after balancing: {spec_bal:.4f}")

	# 12) Write sensitivity/specificity before and after balancing to a text file
	out_lines = []
	out_lines.append("Sensitivity and Specificity for Titanic logistic regression")
	out_lines.append("")
	out_lines.append("Before balancing:")
	out_lines.append(f"Accuracy: {acc:.4f}")
	out_lines.append(f"Sensitivity (TPR): {sens:.4f}")
	out_lines.append(f"Specificity (TNR): {spec:.4f}")
	out_lines.append("")
	out_lines.append("After balancing:")
	out_lines.append(f"Accuracy: {acc_bal:.4f}")
	out_lines.append(f"Sensitivity (TPR): {sens_bal:.4f}")
	out_lines.append(f"Specificity (TNR): {spec_bal:.4f}")

	with open("sensitivity_specificity.txt", "w") as f:
		f.write("\n".join(out_lines))

	print('\nWrote sensitivity/specificity values to `sensitivity_specificity.txt`.')


if __name__ == "__main__":
	main()