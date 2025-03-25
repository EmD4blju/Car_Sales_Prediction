import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 1. Wczytanie i przygotowanie danych
dataset = pd.read_csv('train_flagged.csv', index_col='ID')

# Sprawdzamy, czy kolumna Cena_log istnieje, jeśli nie, to ją tworzymy
if 'Cena_log' not in dataset.columns:
    dataset['Cena_log'] = np.log1p(dataset['Cena'])

# 2. Inżynieria cech
# Funkcja dla kodowania średniego
def add_mean_encoding(df, categorical_cols, target_col):
    df = df.copy()
    # Dołączamy target_col do df
    df[target_col.name] = target_col
    for col in categorical_cols:
        # Obliczamy średnią dla każdej kategorii
        means = df.groupby(col)[target_col.name].mean()
        df[col + "_mean_enc"] = df[col].map(means)
    # Usuwamy tymczasową kolumnę target_col
    df.drop(columns=[target_col.name], inplace=True)
    return df

# Funkcja dla kodowania częstości
def add_frequency_encoding(df, categorical_cols):
    df = df.copy()
    for col in categorical_cols:
        frequency = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(frequency)
    return df

# 3. Podział danych
X = dataset.drop(['Cena', 'Cena_log'], axis=1)
y = dataset['Cena_log']  # Używamy zlogarytmizowanej ceny
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Zastosowanie RobustScaler do danych numerycznych
scaler = RobustScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Dodajemy encodowanie średniej i częstości do zbioru treningowego
X_train_encoded = add_mean_encoding(X_train, categorical_cols, target_col=y_train)
X_train_encoded = add_frequency_encoding(X_train_encoded, categorical_cols)

# Dodajemy encodowanie średniej i częstości do zbioru testowego (używamy wartości z treningu)
X_test_encoded = X_test.copy()
for col in categorical_cols:
    # Obliczamy średnią dla każdej kategorii w zbiorze treningowym, używając y_train, który zawiera Cena_log
    means = y_train.groupby(X_train[col]).mean()  # Zmieniamy tutaj na y_train, ponieważ Cena_log to y_train
    X_test_encoded[f'{col}_mean_target'] = X_test[col].map(means)

    # Obliczamy częstość dla każdej kategorii w zbiorze treningowym
    frequency = X_train[col].value_counts(normalize=True)
    X_test_encoded[f'{col}_freq'] = X_test[col].map(frequency)

# 5. Label Encoding i obsługa brakujących wartości
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()

    # Fit the LabelEncoder on training data
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col])

    # Store the LabelEncoder for future use on test data
    label_encoders[col] = le

    # Apply transformation to the test data
    try:
        X_test_encoded[col] = le.transform(X_test_encoded[col])
    except ValueError:
        # If the test set has unseen categories, we handle by assigning the most frequent class
        most_frequent_value = le.classes_[0]
        X_test_encoded[col] = X_test_encoded[col].apply(
            lambda x: most_frequent_value if x not in le.classes_ else x
        )

    # Handling NaN values after transformation
    X_test_encoded[col] = X_test_encoded[col].fillna(X_train_encoded[col].mode()[0])

# 6. Optymalizacja za pomocą Optuna dla maksymalnej redukcji RMSE
def objective(trial):
    # XGBoost parameters
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'gamma': trial.suggest_float('xgb_gamma', 0, 5),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 5),
    }

    # LightGBM parameters
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 15),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 5),
    }

    # RandomForest parameters
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('rf_max_features', 0.5, 1.0),
    }

    # ExtraTrees parameters
    et_params = {
        'n_estimators': trial.suggest_int('et_n_estimators', 50, 500),
        'max_depth': trial.suggest_int('et_max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('et_min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('et_min_samples_leaf', 1, 10),
    }

    # Tworzenie modeli z optymalizowanymi parametrami
    xgb = XGBRegressor(**xgb_params, n_jobs=-1, random_state=42)
    lgb = LGBMRegressor(**lgb_params, n_jobs=-1, random_state=42)
    rf = RandomForestRegressor(**rf_params, n_jobs=-1, random_state=42)
    et = ExtraTreesRegressor(**et_params, n_jobs=-1, random_state=42)

    # Wagi dla ensemble
    weight_xgb = trial.suggest_float('weight_xgb', 0.1, 1.0)
    weight_lgb = trial.suggest_float('weight_lgb', 0.1, 1.0)
    weight_rf = trial.suggest_float('weight_rf', 0.1, 1.0)
    weight_et = trial.suggest_float('weight_et', 0.1, 1.0)

    # Tworzenie ensemble modelu z odpowiednimi wagami
    ensemble = VotingRegressor([('xgb', xgb), ('lgb', lgb), ('rf', rf), ('et', et)],
                               weights=[weight_xgb, weight_lgb, weight_rf, weight_et])

    # Trenowanie ensemble modelu
    ensemble.fit(X_train_encoded, y_train)

    # Predykcje
    preds = ensemble.predict(X_test_encoded)

    # Ocena - konwertujemy z powrotem logarytmiczną cenę
    preds_original = np.expm1(preds)
    y_test_original = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_original, preds_original))
    return rmse

# Optuna - wykonaj optymalizację
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Większa liczba prób dla lepszych wyników

print('Najlepsze parametry:')
print(study.best_params)
print(f'Najlepsze RMSE: {study.best_value}')

# 7. Finalny model
final_xgb = XGBRegressor(**{k.replace('xgb_', ''): v for k, v in study.best_params.items() if k.startswith('xgb_')},
                         n_jobs=-1, random_state=42)
final_lgb = LGBMRegressor(**{k.replace('lgb_', ''): v for k, v in study.best_params.items() if k.startswith('lgb_')},
                          n_jobs=-1, random_state=42)
final_rf = RandomForestRegressor(
    **{k.replace('rf_', ''): v for k, v in study.best_params.items() if k.startswith('rf_')}, n_jobs=-1,
    random_state=42)
final_et = ExtraTreesRegressor(**{k.replace('et_', ''): v for k, v in study.best_params.items() if k.startswith('et_')},
                               n_jobs=-1, random_state=42)

# Finalny ensemble
weights = [study.best_params['weight_xgb'], study.best_params['weight_lgb'],
           study.best_params['weight_rf'], study.best_params['weight_et']]

final_ensemble = VotingRegressor([('xgb', final_xgb), ('lgb', final_lgb), ('rf', final_rf), ('et', final_et)],
                                 weights=weights)

# Trenuj model
final_ensemble.fit(X_train_encoded, y_train)

# Ewaluacja modelu
train_preds = final_ensemble.predict(X_train_encoded)
test_preds = final_ensemble.predict(X_test_encoded)

train_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(train_preds)))
test_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(test_preds)))

print(f'RMSE dla zbioru treningowego: {train_rmse}')
print(f'RMSE dla zbioru testowego: {test_rmse}')

# Wizualizacja wyników
sns.scatterplot(x=np.expm1(y_test), y=np.expm1(test_preds))
plt.xlabel('Prawdziwe wartości')
plt.ylabel('Predykcje')
plt.title('Porównanie prawdziwych wartości i predykcji')
plt.show()
