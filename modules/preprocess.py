import modules.config as cfg
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



def add_tail_numbers_and_reformat(flights, tail_numbers):
    flights_with_models = flights.merge(tail_numbers, left_on='TAIL_NUMBER', right_on = 'Tail Number', how='left')
    flights_with_models = flights_with_models[flights_with_models['Aircraft Model'].notna()]
    flights_with_models = flights_with_models.drop(columns=['Tail Number'])
    flights_with_models = flights_with_models.rename(columns={'Aircraft Model': 'AIRCRAFT_MODEL'})

    return flights_with_models

def sample_flights_by_airline(flights_with_models, frac=cfg.FRACTION_TO_SAMPLE, random_state=42):
    sampled_flights = flights_with_models.groupby('AIRLINE', 
                                                  group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))
    return sampled_flights

'''

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the flight data for modeling.
    
    Args:
        df (pd.DataFrame): Raw flight data
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    print("Starting data preprocessing...")
    df = df.copy()
    
    for col in cfg.TIME_COLUMNS_TO_BE_PROCESSED:
        if col in df.columns:
            time_series = pd.to_datetime(df[col], format='%H:%M', errors='coerce')
            df[f'{col}_HOUR'] = time_series.dt.hour.fillna(0).astype(int)
            df[f'{col}_MINUTE'] = time_series.dt.minute.fillna(0).astype(int)

    df['DEPARTURE_DELAYED'] = (df['DEPARTURE_DELAY'] > 15).astype(int)
    df['ARRIVAL_DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)


    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    df['QUARTER'] = df['DATE'].dt.quarter
    df['WEEK_OF_YEAR'] = df['DATE'].dt.isocalendar().week
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'].isin([5, 6, 7])).astype(int)


    df['IS_BUSINESS_FLIGHT'] = ((df['SCHEDULED_DEPARTURE_HOUR'] >= 6) & 
                               (df['SCHEDULED_DEPARTURE_HOUR'] <= 9) & 
                               df['DAY_OF_WEEK'] == 1).astype(int)
    
    df['IS_HOLIDAY_SEASON'] = ((df['MONTH'].isin([11, 12, 1, 6, 7, 8])) | 
                              ((df['MONTH'] == 7) & (df['DAY'] == 4))).astype(int)
    
    df['IS_LEVEL_3_AIRPORT'] = ((df['IATA_CODE_dep'].isin(['JFK', 'LGA', 'DCA'])) |
                               (df['IATA_CODE_arr'].isin(['JFK', 'LGA', 'DCA']))).astype(int)


    for col in cfg.CATEGORICAL_COLUMNS_TO_BE_PROCESSED:
        if col in df.columns:
            top_categories = df[col].value_counts().head(50).index
            df[f'{col}_top'] = df[col].where(df[col].isin(top_categories), 'Other')

    df_model = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    
    print(f"Data shape after preprocessing: {df_model.shape}")
    return df_model

def select_features(df: pd.DataFrame) -> list:
    """
    Select features available in the dataset for modeling.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        
    Returns:
        list: Available features for modeling
    """
    all_features = cfg.NON_CATEGORICAL_FEATURES + cfg.CATEGORICAL_FEATURES
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"Using {len(available_features)} features for prediction")
    return available_features

def prepare_model_data(df: pd.DataFrame, target: str) -> tuple[np.ndarray, pd.Series]:
    """
    Prepare data for machine learning models.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        target (str): Target variable name
        
    Returns:
        tuple: (X_encoded, y) where X_encoded is the feature matrix and y is the target
    """
    features = select_features(df)
    
    X = df[features].copy()
    y = df[target].copy()
    
    categorical_cols = cfg.CATEGORICAL_FEATURES
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_encoded = preprocessor.fit_transform(X)
    

    feature_names = (numeric_cols + 
                    list(preprocessor.named_transformers_['cat']
                         .get_feature_names_out(categorical_cols)))

    print(f"Final feature matrix shape: {X_encoded.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X_encoded, y, feature_names

'''