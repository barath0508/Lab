import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("train_data_exp8.csv")

target = df["target"]
train = df.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    train,
    target,
    test_size=0.20
)

train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(
    train,
    target,
    test_size=1 - train_ratio
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test,
    y_test,
    test_size=test_ratio / (test_ratio + validation_ratio)
)

model_1 = LinearRegression()
model_2 = xgb.XGBRegressor()
model_3 = RandomForestRegressor()

model_1.fit(x_train, y_train)

val_pred_1 = model_1.predict(x_val)
test_pred_1 = model_1.predict(x_test)

val_pred_1 = pd.DataFrame(val_pred_1)
test_pred_1 = pd.DataFrame(test_pred_1)

model_2.fit(x_train, y_train)

val_pred_2 = model_2.predict(x_val)
test_pred_2 = model_2.predict(x_test)

val_pred_2 = pd.DataFrame(val_pred_2)
test_pred_2 = pd.DataFrame(test_pred_2)

model_3.fit(x_train, y_train)

val_pred_3 = model_1.predict(x_val)
test_pred_3 = model_1.predict(x_test)

val_pred_3 = pd.DataFrame(val_pred_3)
test_pred_3 = pd.DataFrame(test_pred_3)

df_val = pd.concat(
    [x_val, val_pred_1, val_pred_2, val_pred_3],
    axis=1
)

df_test = pd.concat(
    [x_test, test_pred_1, test_pred_2, test_pred_3],
    axis=1
)

final_model = LinearRegression()

final_model.fit(df_val, y_val)

final_pred = final_model.predict(df_test)

print(mean_squared_error(y_test, final_pred))
