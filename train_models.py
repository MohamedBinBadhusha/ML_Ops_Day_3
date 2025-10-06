from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_models():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, "logistic_regression_model.pkl")

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, "random_forest_model.pkl")

    print("✅ Models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
