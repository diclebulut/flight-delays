import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FlightDelayPredictor:
    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.scaler = StandardScaler(with_mean=False)
        self.label_encoders = {}
        
    def preprocess_data(self):
        """Clean and prepare the data for modeling"""
        print("Starting data preprocessing...")
        
        
        
        time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'WHEELS_OFF', 
                       'WHEELS_ON', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
        
        for col in time_columns:
            if col in self.df.columns:

                time_series = pd.to_datetime(self.df[col], format='%H:%M', errors='coerce')
                
                self.df[f'{col}_HOUR'] = time_series.dt.hour.fillna(0).astype(int)
                self.df[f'{col}_MINUTE'] = time_series.dt.minute.fillna(0).astype(int)
        

        self.df['DEPARTURE_DELAYED'] = (self.df['DEPARTURE_DELAY'] > 15).astype(int)
        
        self.df['ARRIVAL_DELAYED'] = (self.df['ARRIVAL_DELAY'] > 15).astype(int)
        

        self.df['DATE'] = pd.to_datetime(self.df[['YEAR', 'MONTH', 'DAY']])
        self.df['QUARTER'] = self.df['DATE'].dt.quarter
        self.df['WEEK_OF_YEAR'] = self.df['DATE'].dt.isocalendar().week
        self.df['IS_WEEKEND'] = (self.df['DAY_OF_WEEK'].isin([5, 6, 7])).astype(int)

        self.df['IS_BUSINESS_FLIGHT'] = ((self.df['SCHEDULED_DEPARTURE_HOUR'] >= 6) & 
                                     (self.df['SCHEDULED_DEPARTURE_HOUR'] <= 9) & self.df['DAY_OF_WEEK'] == 1).astype(int)
        
        self.df['IS_HOLIDAY_SEASON'] = ((self.df['MONTH'].isin([11, 12, 1, 6, 7, 8])) | 
                                       ((self.df['MONTH'] == 7) & (self.df['DAY'] == 4))).astype(int)
        
        self.df['IS_LEVEL_3_AIRPORT'] = ((self.df['IATA_CODE_dep'].isin(['JFK', 'LGA', 'DCA'])) |
                                        (self.df['IATA_CODE_arr'].isin(['JFK', 'LGA', 'DCA'])) ).astype(int)
        
        
        categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'STATE_dep', 'STATE_arr', 'AIRCRAFT_MODEL']
        
        for col in categorical_cols:
            if col in self.df.columns:
                top_categories = self.df[col].value_counts().head(50).index
                self.df[f'{col}_top'] = self.df[col].where(self.df[col].isin(top_categories), 'Other')

        self.df_model = self.df[(self.df['CANCELLED'] == 0) & (self.df['DIVERTED'] == 0)].copy()
        
        print(f"Data shape after preprocessing: {self.df_model.shape}")
        return self.df_model
    
    def select_features(self):
        
        predictive_features = [
            'DAY',  'WEEK_OF_YEAR',
            'IS_WEEKEND', 'IS_HOLIDAY_SEASON',
            'SCHEDULED_DEPARTURE_HOUR', 
            'SCHEDULED_ARRIVAL_HOUR', 
            'IS_MORNING_RUSH', 'IS_EVENING_RUSH', 'DISTANCE'
        ]
        
        categorical_features = [
            'MONTH','DAY_OF_WEEK', 'QUARTER','AIRLINE_top', 'ORIGIN_AIRPORT_top', 'DESTINATION_AIRPORT_top',
            'STATE_dep_top', 'STATE_arr_top', 'AIRCRAFT_MODEL_top'
        ]
        
        all_features = predictive_features + categorical_features
        
        available_features = [f for f in all_features if f in self.df_model.columns]
        
        print(f"Using {len(available_features)} features for prediction")
        return available_features
    
    def prepare_model_data(self, target):
        """Prepare data for machine learning models"""
        
        features = self.select_features()
        
        X = self.df_model[features].copy()
        y = self.df_model[target].copy()
        
        categorical_cols = [
            'MONTH','DAY_OF_WEEK', 'QUARTER','AIRLINE_top', 'ORIGIN_AIRPORT_top', 'DESTINATION_AIRPORT_top',
            'STATE_dep_top', 'STATE_arr_top', 'AIRCRAFT_MODEL_top'
        ]
        
        numeric_cols = [col for col in X.columns if col not in categorical_cols]


        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
        )

        X_encoded = preprocessor.fit_transform(X)

        
        num_features = numeric_cols
        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        self.feature_names = list(num_features) + list(cat_features)

        print(f"Final feature matrix shape: {X_encoded.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X_encoded, y
    
    def train_models(self, target , test_size=0.2):
        """Train multiple models and compare performance"""
        
        X, y = self.prepare_model_data(target)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_test = X_test_scaled
        self.y_test = y_test
       
        

        models_to_try = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in models_to_try.items():
            print(f"\nTraining {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            auc_score = roc_auc_score(y_test, y_pred_proba)
            
  
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'predictions_proba': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            

            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                categorical_cols = [
                    'MONTH','DAY_OF_WEEK', 'QUARTER','AIRLINE_top', 'ORIGIN_AIRPORT_top', 'DESTINATION_AIRPORT_top',
                    'STATE_dep_top', 'STATE_arr_top', 'AIRCRAFT_MODEL_top'
                ]
                feature_importance['original_column'] = feature_importance['feature'].apply(
                lambda x: x.split('_')[0] if any(x.startswith(col + '_') for col in categorical_cols) else x
                )
                aggregated_importance = feature_importance.groupby('original_column')['importance'].sum().sort_values(ascending=False)

                print("\nAggregated Feature Importances:")
                print(aggregated_importance.head(20))

            elif hasattr(model, 'coef_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
                sum_of_importance  = feature_importance['importance'].sum()
                print("sum of importance:", sum_of_importance)
                print("\nTop 20 Most Important Features:")
                print(feature_importance.head(20))
        
        self.models = results
        return results
    
    def evaluate_best_model(self):
        """Evaluate the best performing model in detail"""
        
        best_model_name = max(self.models.keys(), 
                             key=lambda x: self.models[x]['cv_mean'])
        
        print(f"\nBest Model: {best_model_name}")
        print("="*50)
        
        best_model = self.models[best_model_name]
        y_pred = best_model['predictions']
        y_pred_proba = best_model['predictions_proba']
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        if hasattr(best_model['model'], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 20 Most Important Features:")
            print(feature_importance.head(20))
        
        return best_model_name, best_model
    
    def hyperparameter_tuning(self, model_name='Random Forest'):
        """Perform hyperparameter tuning for the specified model"""
        
        X, y = self.prepare_model_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

