import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(name=__name__)

def create_testing_set(df:pd.DataFrame, target_column:str) -> pd.DataFrame:
    feature_columns = [column for column in df.columns if column != target_column and column != 'ID']
    filtered_df = df[df[target_column].isna() & df[feature_columns].notna().all(axis=1)]
    return filtered_df.drop(columns=[target_column])


def auto_test(training_set:pd.DataFrame, testing_set:pd.DataFrame) -> np.ndarray:
    training_columns = training_set.columns
    testing_columns = testing_set.columns
    
    target_column = training_columns.difference(testing_columns)[0]
    target_type = training_set[target_column].dtype
    
    log.debug(msg=f'Target column: {target_column}, Target type: {target_type}')
    
    match(target_type):
        case np.float64:
            log.debug(msg='Regression')
            model = RandomForestRegressor(criterion='squared_error', random_state=42)
        case np.object_:
            log.debug(msg='Classification')
            model = RandomForestClassifier(criterion='entropy', random_state=42)
        case _:
            log.debug(msg='Not matched')
            return
    
    X_train = training_set.drop(columns=[target_column])
    y_train = training_set[target_column]
    X_test = testing_set.copy()
    
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])
    
    model.fit(
        X=X_train_encoded,
        y=y_train
    )
    
    predictions = model.predict(X=X_test_encoded)
    
    results_df = pd.DataFrame(data={
        'ID': testing_set.index,
        target_column: predictions
    }).set_index(keys='ID')
    
    return results_df