import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_model(path="data/datasets/sample.csv", target="target"):
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    f1 = f1_score(y_test,preds,average="macro")

    return model, X_train.columns.tolist(), acc, f1
