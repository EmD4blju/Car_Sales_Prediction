import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings('ignore')

# 1. Wczytanie i przygotowanie danych
dataset = pd.read_csv('datasets/train_flagged.csv', index_col='ID')

# Analizujemy typy danych i podstawowe statystyki
print("Informacje o danych:")
print(dataset.info())
print("\nPodstawowe statystyki:")
print(dataset.describe())

# 2. Inżynieria cech

# Sprawdzamy czy mamy wartości odstające w cenie
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Cena'], kde=True)
plt.title('Rozkład cen')
plt.show()

# Logarytmujemy cenę, żeby uniknąć wpływu wartości odstających
dataset['Cena_log'] = np.log1p(dataset['Cena'])

# Sprawdzamy korelację numerycznych cech z ceną
numeric_cols = dataset.select_dtypes(include=['int64', 'float64']).columns
correlation = dataset[numeric_cols].corr()['Cena'].sort_values(ascending=False)
print("Korelacja z ceną:")
print(correlation)

# Dodajemy nowe cechy na podstawie kategorycznych
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()

# Tworzymy średnie encodowanie dla kategorycznych zmiennych
def add_mean_encoding(df, cat_cols, target_col):
    df_copy = df.copy()
    for col in cat_cols:
        means = df.groupby(col)[target_col].mean()
        df_copy[f'{col}_mean_target'] = df[col].map(means)
    return df_copy

# Miary częstości występowania kategorii
def add_frequency_encoding(df, cat_cols):
    df_copy = df.copy()
    for col in cat_cols:
        frequency = df[col].value_counts(normalize=True)
        df_copy[f'{col}_freq'] = df[col].map(frequency)
    return df_copy

# 3. Podział danych
X = dataset.drop(['Cena', 'Cena_log'], axis=1)
y = dataset['Cena_log']  # Używamy zlogarytmizowanej ceny
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Dodajemy encodowanie średniej i częstości do zbioru treningowego
X_train_encoded = add_mean_encoding(X_train, categorical_cols, y_train)
X_train_encoded = add_frequency_encoding(X_train_encoded, categorical_cols)

# To samo dla zbioru testowego (używamy wartości z treningu)
X_test_encoded = X_test.copy()
for col in categorical_cols:
    means = X_train.groupby(col)[y_train.name].mean()
    X_test_encoded[f'{col}_mean_target'] = X_test[col].map(means)
    
    frequency = X_train[col].value_counts(normalize=True)
    X_test_encoded[f'{col}_freq'] = X_test[col].map(frequency)

# Konwersja kategorycznych cech do numerycznych
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col])
    X_test_encoded[col] = le.transform(X_test_encoded[col].fillna(X_train_encoded[col].mode()[0]))
    label_encoders[col] = le

# 5. Optymalizacja za pomocą Optuna dla maksymalnej redukcji RMSE

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
    ensemble = VotingRegressor([
        ('xgb', xgb),
        ('lgb', lgb),
        ('rf', rf),
        ('et', et)
    ], weights=[weight_xgb, weight_lgb, weight_rf, weight_et])
    
    # Trenowanie ensemble modelu
    ensemble.fit(X_train_encoded, y_train)
    
    # Predykcje
    preds = ensemble.predict(X_test_encoded)
    
    # Ocena - konwertujemy z powrotem logarytmiczną cenę
    preds_original = np.expm1(preds)
    y_test_original = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_original, preds_original))
    return rmse

# Utworzenie badania Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Możesz zwiększyć liczbę prób dla lepszych wyników

print('Najlepsze parametry:')
print(study.best_params)
print(f'Najlepsze RMSE: {study.best_value}')

# 6. Tworzenie finalnego modelu z najlepszymi parametrami

# Wyodrębnij parametry dla każdego modelu
xgb_best_params = {k.replace('xgb_', ''): v for k, v in study.best_params.items() if k.startswith('xgb_')}
lgb_best_params = {k.replace('lgb_', ''): v for k, v in study.best_params.items() if k.startswith('lgb_')}
rf_best_params = {k.replace('rf_', ''): v for k, v in study.best_params.items() if k.startswith('rf_')}
et_best_params = {k.replace('et_', ''): v for k, v in study.best_params.items() if k.startswith('et_')}

# Utwórz finalne modele
final_xgb = XGBRegressor(**xgb_best_params, n_jobs=-1, random_state=42)
final_lgb = LGBMRegressor(**lgb_best_params, n_jobs=-1, random_state=42)
final_rf = RandomForestRegressor(**rf_best_params, n_jobs=-1, random_state=42)
final_et = ExtraTreesRegressor(**et_best_params, n_jobs=-1, random_state=42)

# Utwórz finalny ensemble
weights = [
    study.best_params['weight_xgb'], 
    study.best_params['weight_lgb'], 
    study.best_params['weight_rf'], 
    study.best_params['weight_et']
]

final_ensemble = VotingRegressor([
    ('xgb', final_xgb),
    ('lgb', final_lgb),
    ('rf', final_rf),
    ('et', final_et)
], weights=weights)

# Trenuj model
final_ensemble.fit(X_train_encoded, y_train)

# Ewaluacja modelu
train_preds = final_ensemble.predict(X_train_encoded)
test_preds = final_ensemble.predict(X_test_encoded)

# Konwersja z powrotem do oryginalnej skali
train_preds_original = np.expm1(train_preds)
test_preds_original = np.expm1(test_preds)
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)

# Obliczenie metryk
train_rmse = np.sqrt(mean_squared_error(y_train_original, train_preds_original))
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_preds_original))
train_r2 = r2_score(y_train_original, train_preds_original)
test_r2 = r2_score(y_test_original, test_preds_original)

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'Train R²: {train_r2}')
print(f'Test R²: {test_r2}')

# 7. Predykcje na zbiorze testowym
test_flagged = pd.read_csv('/content/drive/MyDrive/ML/datasets/test_flagged.csv', index_col='ID')
test_ids = test_flagged.index.tolist()

# Przygotowanie danych testowych
test_encoded = test_flagged.copy()

# Dodanie nowych cech
for col in categorical_cols:
    # Mean encoding
    means = X_train.groupby(col)[y_train.name].mean()
    test_encoded[f'{col}_mean_target'] = test_flagged[col].map(means)
    
    # Frequency encoding
    frequency = X_train[col].value_counts(normalize=True)
    test_encoded[f'{col}_freq'] = test_flagged[col].map(frequency)
    
    # Wypełnienie brakujących wartości
    if col in test_encoded.columns:
        test_encoded[f'{col}_mean_target'] = test_encoded[f'{col}_mean_target'].fillna(0)
        test_encoded[f'{col}_freq'] = test_encoded[f'{col}_freq'].fillna(0)

# Label encoding kategorycznych cech
for col in categorical_cols:
    le = label_encoders.get(col)
    if le is not None:
        # Obsługa nowych kategorii w danych testowych
        test_encoded[col] = test_encoded[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
        test_encoded[col] = le.transform(test_encoded[col])

# Predykcje
test_preds_log = final_ensemble.predict(test_encoded)
test_preds = np.expm1(test_preds_log)

# Zapisz predykcje
pd.DataFrame({
    'ID': test_ids,
    'Cena': test_preds
}).to_csv('/content/drive/MyDrive/ML/datasets/predictions_improved.csv', index=False)

print("Zapisano predykcje do pliku predictions_improved.csv")

# 8. Analiza ważności cech
def plot_feature_importance(ensemble_model, feature_names):
    # Zbieramy ważności cech z każdego modelu
    importances = {}
    
    for name, model in ensemble_model.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            importances[name] = model.feature_importances_
    
    # Tworzymy wykres dla każdego modelu
    plt.figure(figsize=(15, 10))
    
    for i, (name, imp) in enumerate(importances.items()):
        plt.subplot(2, 2, i+1)
        feat_imp = pd.Series(imp, index=feature_names)
        feat_imp = feat_imp.sort_values(ascending=False)[:20]  # Top 20 cech
        
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title(f'Ważność cech - {name}')
        plt.tight_layout()
    
    plt.show()

# Wyświetl ważność cech
plot_feature_importance(final_ensemble, X_train_encoded.columns)
