import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score,
                           mean_squared_error, r2_score, mean_absolute_error)
import seaborn as sns

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class ModelPerformanceEvaluator:
    """Comprehensive model performance evaluation system"""
    
    def __init__(self):
        self.metrics_history = []
        self.model_comparisons = {}
    
    def evaluate_classification_model(self, model, X_test, y_test, y_pred, model_name, condition_name):
        """Evaluate classification model with comprehensive metrics"""
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate AUC if binary classification
        try:
            if len(np.unique(y_test)) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = None
        except:
            auc = None
        
        # Create detailed classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'model_name': model_name,
            'condition': condition_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'test_samples': len(y_test),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return metrics
    
    def evaluate_regression_model(self, model, X_test, y_test, y_pred, model_name, target_name):
        """Evaluate regression model with comprehensive metrics"""
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        
        metrics = {
            'model_name': model_name,
            'target': target_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'test_samples': len(y_test),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv_folds=5, scoring='accuracy'):
        """Perform cross-validation and return detailed results"""
        
        if scoring == 'accuracy' and len(np.unique(y)) <= 1:
            return {'cv_scores': [0], 'cv_mean': 0, 'cv_std': 0}
        
        try:
            if scoring in ['accuracy', 'precision', 'recall', 'f1']:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv = cv_folds
            
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            return {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except Exception as e:
            print(f"Cross-validation error: {e}")
            return {'cv_scores': [0], 'cv_mean': 0, 'cv_std': 0}
    
    def generate_performance_report(self, metrics_list):
        """Generate a comprehensive performance report"""
        
        report = []
        report.append("="*80)
        report.append("RANDOM FOREST MODEL PERFORMANCE REPORT")
        report.append("="*80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Classification metrics
        classification_metrics = [m for m in metrics_list if 'accuracy' in m]
        if classification_metrics:
            report.append("CLASSIFICATION PERFORMANCE:")
            report.append("-" * 40)
            
            for metrics in classification_metrics:
                report.append(f"\nCondition: {metrics['condition']}")
                report.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
                report.append(f"  Precision: {metrics['precision']:.4f}")
                report.append(f"  Recall:    {metrics['recall']:.4f}")
                report.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
                if metrics['auc_score']:
                    report.append(f"  AUC Score: {metrics['auc_score']:.4f}")
                report.append(f"  Test Samples: {metrics['test_samples']}")
                
                # Add class-wise metrics
                if 'classification_report' in metrics:
                    class_report = metrics['classification_report']
                    if '0' in class_report and '1' in class_report:
                        report.append(f"  Class 0 (Not Suitable) - Precision: {class_report['0']['precision']:.3f}, Recall: {class_report['0']['recall']:.3f}")
                        report.append(f"  Class 1 (Suitable) - Precision: {class_report['1']['precision']:.3f}, Recall: {class_report['1']['recall']:.3f}")
        
        # Regression metrics
        regression_metrics = [m for m in metrics_list if 'mse' in m]
        if regression_metrics:
            report.append("\n\nREGRESSION PERFORMANCE:")
            report.append("-" * 40)
            
            for metrics in regression_metrics:
                report.append(f"\nTarget: {metrics['target']}")
                report.append(f"  R² Score:  {metrics['r2_score']:.4f}")
                report.append(f"  RMSE:      {metrics['rmse']:.4f}")
                report.append(f"  MAE:       {metrics['mae']:.4f}")
                report.append(f"  MAPE:      {metrics['mape']:.2f}%")
                report.append(f"  Test Samples: {metrics['test_samples']}")
        
        # Overall summary
        if classification_metrics:
            avg_accuracy = np.mean([m['accuracy'] for m in classification_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in classification_metrics])
            avg_precision = np.mean([m['precision'] for m in classification_metrics])
            avg_recall = np.mean([m['recall'] for m in classification_metrics])
            
            report.append("\n\nOVERALL CLASSIFICATION SUMMARY:")
            report.append("-" * 40)
            report.append(f"Average Accuracy:  {avg_accuracy:.4f}")
            report.append(f"Average Precision: {avg_precision:.4f}")
            report.append(f"Average Recall:    {avg_recall:.4f}")
            report.append(f"Average F1-Score:  {avg_f1:.4f}")
        
        if regression_metrics:
            avg_r2 = np.mean([m['r2_score'] for m in regression_metrics])
            avg_rmse = np.mean([m['rmse'] for m in regression_metrics])
            
            report.append("\n\nOVERALL REGRESSION SUMMARY:")
            report.append("-" * 40)
            report.append(f"Average R² Score: {avg_r2:.4f}")
            report.append(f"Average RMSE:     {avg_rmse:.4f}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


class HealthAwareRandomForestRecommender:
    """Enhanced Random Forest-based food recommender with comprehensive performance evaluation"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.evaluator = ModelPerformanceEvaluator()
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'model_performance': {},
            'detailed_metrics': []
        }
        
        # Nutritional features for the model
        self.nutritional_features = [
            'Energy(kcal) by calculation', 
            'Protein(g)', 
            'CHOCDF (g) Carbohydrate',
            'SUGAR(g)', 
            'FIBTG (g) Dietary fibre', 
            'Fat(g)',
            'Na(mg)',
            'K(mg)',
            'Ca(mg)',
            'CHOLE(mg) Cholesterol'
        ]
        
        # Load and prepare data
        start_time = time.time()
        self.load_data()
        self.prepare_features()
        self.train_models_with_evaluation()
        self.stats['loading_time'] = time.time() - start_time
    
    def update_status(self, message):
        """Update loading status if callback is provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self):
        """Load and combine all food datasets from CSV files"""
        try:
            dataset_folder = './datasets'
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                self.update_status("No CSV files found in the datasets folder!")
                self.food_data = pd.DataFrame()
                return
            
            dataframes = []
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    category = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                    
                    self.update_status(f"Loading {category} data...")
                    
                    df = pd.read_csv(file_path)
                    
                    if 'Category' not in df.columns:
                        df['Category'] = category
                    
                    # Clean and standardize data
                    df = self._clean_nutritional_data(df)
                    
                    dataframes.append(df)
                    self.update_status(f"Loaded {len(df)} items from {filename}")
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.update_status("Combining all food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                
                if 'Category' in self.food_data.columns:
                    self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
                
                self.update_status(f"Successfully loaded {len(self.food_data)} food items")
                
                # Add health suitability scores
                self._calculate_health_scores()
                
            else:
                self.update_status("No valid data files could be loaded")
                self.food_data = pd.DataFrame()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            self.food_data = pd.DataFrame()
    
    def _clean_nutritional_data(self, df):
        """Clean and standardize nutritional data"""
        # Fill missing values with 0 for nutritional data
        nutritional_columns = self.nutritional_features + ['FASAT (g) Saturated FA']
        
        for col in nutritional_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove rows with all zero nutritional values
        nutrient_cols = [col for col in nutritional_columns if col in df.columns]
        if nutrient_cols:
            df = df[df[nutrient_cols].sum(axis=1) > 0]
        
        return df
    
    def _calculate_health_scores(self):
        """Calculate health suitability scores for each food item"""
        self.update_status("Calculating health suitability scores...")
        
        # Initialize score columns
        self.food_data['Diabetes_Score'] = 0
        self.food_data['Obesity_Score'] = 0
        self.food_data['Hypertension_Score'] = 0
        self.food_data['High_Cholesterol_Score'] = 0
        self.food_data['Overall_Health_Score'] = 0
        
        # Initialize binary suitability columns for classification
        self.food_data['Diabetes_Suitable'] = 0
        self.food_data['Obesity_Suitable'] = 0
        self.food_data['Hypertension_Suitable'] = 0
        self.food_data['High_Cholesterol_Suitable'] = 0
        
        for idx, food in self.food_data.iterrows():
            # Calculate individual condition scores
            diabetes_score = self._calculate_diabetes_score(food)
            obesity_score = self._calculate_obesity_score(food)
            hypertension_score = self._calculate_hypertension_score(food)
            cholesterol_score = self._calculate_cholesterol_score(food)
            
            # Store regression scores
            self.food_data.at[idx, 'Diabetes_Score'] = diabetes_score
            self.food_data.at[idx, 'Obesity_Score'] = obesity_score
            self.food_data.at[idx, 'Hypertension_Score'] = hypertension_score
            self.food_data.at[idx, 'High_Cholesterol_Score'] = cholesterol_score
            
            # Calculate overall health score (lower is better)
            overall_score = (diabetes_score + obesity_score + hypertension_score + cholesterol_score) / 4
            self.food_data.at[idx, 'Overall_Health_Score'] = overall_score
            
            # Create binary classification targets (1 = suitable, 0 = not suitable)
            # Threshold: score <= 1.5 and specific nutrient criteria
            self.food_data.at[idx, 'Diabetes_Suitable'] = int(
                diabetes_score <= 1.5 and float(food.get('SUGAR(g)', 0)) <= 10 and float(food.get('FIBTG (g) Dietary fibre', 0)) >= 2
            )
            
            self.food_data.at[idx, 'Obesity_Suitable'] = int(
                obesity_score <= 1.5 and float(food.get('Energy(kcal) by calculation', 0)) <= 250
            )
            
            self.food_data.at[idx, 'Hypertension_Suitable'] = int(
                hypertension_score <= 1.5 and float(food.get('Na(mg)', 0)) <= 200
            )
            
            self.food_data.at[idx, 'High_Cholesterol_Suitable'] = int(
                cholesterol_score <= 1.5 and float(food.get('FASAT (g) Saturated FA', 0)) <= 3
            )
    
    def _calculate_diabetes_score(self, food):
        """Calculate diabetes suitability score (lower is better)"""
        score = 0
        
        # Sugar penalty (major factor)
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar > 15:
            score += 3
        elif sugar > 8:
            score += 2
        elif sugar > 3:
            score += 1
        
        # Carb quality consideration
        carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        
        if carbs > 0:
            # High carbs with low fiber is problematic
            if carbs > 20 and fiber < 3:
                score += 2
            elif carbs > 30 and fiber < 5:
                score += 3
        
        # Fiber bonus
        if fiber >= 5:
            score -= 1
        elif fiber >= 3:
            score -= 0.5
        
        return max(0, score)
    
    def _calculate_obesity_score(self, food):
        """Calculate obesity suitability score (lower is better)"""
        score = 0
        
        # Calorie density penalty
        calories = float(food.get('Energy(kcal) by calculation', 0))
        if calories > 300:
            score += 3
        elif calories > 200:
            score += 2
        elif calories > 150:
            score += 1
        
        # Fat penalty
        fat = float(food.get('Fat(g)', 0))
        if fat > 15:
            score += 2
        elif fat > 10:
            score += 1
        
        # Sugar penalty
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar > 10:
            score += 2
        elif sugar > 5:
            score += 1
        
        # Protein and fiber bonus (satiety)
        protein = float(food.get('Protein(g)', 0))
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        
        if protein >= 10:
            score -= 1
        if fiber >= 5:
            score -= 1
        
        return max(0, score)
    
    def _calculate_hypertension_score(self, food):
        """Calculate hypertension suitability score (lower is better)"""
        score = 0
        
        # Sodium penalty (major factor)
        sodium = float(food.get('Na(mg)', 0))
        if sodium > 400:
            score += 3
        elif sodium > 200:
            score += 2
        elif sodium > 100:
            score += 1
        
        # Potassium bonus
        potassium = float(food.get('K(mg)', 0))
        if potassium > 300:
            score -= 1
        elif potassium > 200:
            score -= 0.5
        
        # Fiber bonus
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 5:
            score -= 0.5
        
        return max(0, score)
    
    def _calculate_cholesterol_score(self, food):
        """Calculate high cholesterol suitability score (lower is better)"""
        score = 0
        
        # Saturated fat penalty
        sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
        if sat_fat > 5:
            score += 3
        elif sat_fat > 3:
            score += 2
        elif sat_fat > 1:
            score += 1
        
        # Cholesterol penalty
        cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
        if cholesterol > 100:
            score += 2
        elif cholesterol > 50:
            score += 1
        
        # Fiber bonus
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 5:
            score -= 1
        elif fiber >= 3:
            score -= 0.5
        
        return max(0, score)
    
    def prepare_features(self):
        """Prepare features for Random Forest model"""
        if len(self.food_data) == 0:
            self.update_status("Error: No data available for model preparation")
            return
        
        # Available features in the dataset
        available_features = []
        for feature in self.nutritional_features:
            if feature in self.food_data.columns:
                available_features.append(feature)
        
        self.features = available_features
        
        if not self.features:
            self.update_status("Error: No valid features available")
            return
        
        self.update_status(f"Prepared {len(self.features)} features for Random Forest model")
    
    def train_models_with_evaluation(self):
        """Train Random Forest models with comprehensive performance evaluation"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for training")
            return
        
        try:
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Regression targets
            regression_targets = {
                'Overall_Health': self.food_data['Overall_Health_Score'],
                'Diabetes_Score': self.food_data['Diabetes_Score'],
                'Obesity_Score': self.food_data['Obesity_Score'],
                'Hypertension_Score': self.food_data['Hypertension_Score'],
                'Cholesterol_Score': self.food_data['High_Cholesterol_Score']
            }
            
            # Classification targets
            classification_targets = {
                'Diabetes_Suitable': self.food_data['Diabetes_Suitable'],
                'Obesity_Suitable': self.food_data['Obesity_Suitable'],
                'Hypertension_Suitable': self.food_data['Hypertension_Suitable'],
                'High_Cholesterol_Suitable': self.food_data['High_Cholesterol_Suitable']
            }
            
            # Store models
            self.regression_models = {}
            self.classification_models = {}
            
            # Store detailed metrics
            all_metrics = []
            
            self.update_status("Training and evaluating Random Forest models...")
            
            # Train regression models
            for target_name, y in regression_targets.items():
                if y.var() == 0:  # Skip if no variance
                    continue
                
                self.update_status(f"Training regression model for {target_name}...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train Random Forest Regressor
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test)
                
                # Evaluate model
                metrics = self.evaluator.evaluate_regression_model(
                    rf_model, X_test, y_test, y_pred, 'Random Forest', target_name
                )
                
                # Cross-validation
                cv_results = self.evaluator.cross_validate_model(
                    rf_model, X_scaled, y, cv_folds=5, scoring='r2'
                )
                metrics.update(cv_results)
                
                all_metrics.append(metrics)
                self.regression_models[target_name] = rf_model
                
                self.update_status(f"{target_name} - R²: {metrics['r2_score']:.3f}, CV: {metrics['cv_mean']:.3f}")
            
            # Train classification models
            for target_name, y in classification_targets.items():
                if len(np.unique(y)) < 2:  # Skip if not enough classes
                    continue
                
                self.update_status(f"Training classification model for {target_name}...")
                
                # Split data with stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train Random Forest Classifier
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                )
                
                rf_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test)
                
                # Evaluate model
                metrics = self.evaluator.evaluate_classification_model(
                    rf_model, X_test, y_test, y_pred, 'Random Forest', target_name.replace('_Suitable', '')
                )
                
                # Cross-validation for multiple metrics
                for scoring_method in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
                    cv_results = self.evaluator.cross_validate_model(
                        rf_model, X_scaled, y, cv_folds=5, scoring=scoring_method
                    )
                    metrics[f'cv_{scoring_method}'] = cv_results
                
                all_metrics.append(metrics)
                self.classification_models[target_name] = rf_model
                
                self.update_status(f"{target_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
            
            # Store all metrics
            self.stats['detailed_metrics'] = all_metrics
            
            # Generate comprehensive performance report
            performance_report = self.evaluator.generate_performance_report(all_metrics)
            self.stats['performance_report'] = performance_report
            
            # Print performance report
            print("\n" + performance_report)
            
            # Get feature importances from overall health model
            if 'Overall_Health' in self.regression_models:
                feature_importances = self.regression_models['Overall_Health'].feature_importances_
                importance_dict = dict(zip(self.features, feature_importances))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                self.update_status("Top 5 important features:")
                for feature, importance in sorted_importance[:5]:
                    self.update_status(f"  {feature}: {importance:.3f}")
            
        except Exception as e:
            self.update_status(f"Error training models: {e}")
    
    def get_performance_summary(self):
        """Get a summary of model performance metrics"""
        if not self.stats['detailed_metrics']:
            return "No performance metrics available"
        
        metrics = self.stats['detailed_metrics']
        
        # Classification summary
        classification_metrics = [m for m in metrics if 'accuracy' in m]
        regression_metrics = [m for m in metrics if 'mse' in m]
        
        summary = []
        summary.append("MODEL PERFORMANCE SUMMARY")
        summary.append("=" * 40)
        
        if classification_metrics:
            summary.append("\nCLASSIFICATION MODELS:")
            avg_accuracy = np.mean([m['accuracy'] for m in classification_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in classification_metrics])
            avg_precision = np.mean([m['precision'] for m in classification_metrics])
            avg_recall = np.mean([m['recall'] for m in classification_metrics])
            
            summary.append(f"Average Accuracy:  {avg_accuracy:.4f}")
            summary.append(f"Average F1-Score:  {avg_f1:.4f}")
            summary.append(f"Average Precision: {avg_precision:.4f}")
            summary.append(f"Average Recall:    {avg_recall:.4f}")
            
            for m in classification_metrics:
                summary.append(f"\n{m['condition']}:")
                summary.append(f"  Accuracy: {m['accuracy']:.3f}")
                summary.append(f"  F1-Score: {m['f1_score']:.3f}")
                summary.append(f"  Precision: {m['precision']:.3f}")
                summary.append(f"  Recall: {m['recall']:.3f}")
        
        if regression_metrics:
            summary.append(f"\nREGRESSION MODELS:")
            avg_r2 = np.mean([m['r2_score'] for m in regression_metrics])
            avg_rmse = np.mean([m['rmse'] for m in regression_metrics])
            
            summary.append(f"Average R² Score: {avg_r2:.4f}")
            summary.append(f"Average RMSE:     {avg_rmse:.4f}")
            
            for m in regression_metrics:
                summary.append(f"\n{m['target']}:")
                summary.append(f"  R² Score: {m['r2_score']:.3f}")
                summary.append(f"  RMSE: {m['rmse']:.3f}")
        
        return "\n".join(summary)
    
    def get_recommendations(self, user_profile, meal_type='lunch', max_recommendations=10):
        """Get personalized food recommendations based on user profile"""
        if not hasattr(self, 'classification_models') or len(self.food_data) == 0:
            return []
        
        try:
            # Get user's health conditions
            health_conditions = user_profile.get('health_conditions', [])
            
            # Create candidate pool
            candidates = self.food_data.copy()
            
            # Apply category filter if specified
            category_filter = user_profile.get('category_filter', 'All')
            if category_filter != 'All':
                candidates = candidates[candidates['Category'] == category_filter]
            
            if len(candidates) == 0:
                return []
            
            # Extract features for prediction
            X_candidates = candidates[self.features].fillna(0)
            X_candidates_scaled = self.scaler.transform(X_candidates)
            
            # Calculate suitability scores for each candidate
            scores = []
            
            for idx, (_, food) in enumerate(candidates.iterrows()):
                # Use classification models to predict suitability
                suitability_scores = {}
                for condition in health_conditions:
                    model_key = f"{condition}_Suitable"
                    if model_key in self.classification_models:
                        model = self.classification_models[model_key]
                        # Get probability of being suitable
                        prob = model.predict_proba(X_candidates_scaled[idx:idx+1])[0][1]
                        suitability_scores[condition] = prob
                
                # Calculate overall suitability
                if suitability_scores:
                    overall_suitability = np.mean(list(suitability_scores.values()))
                else:
                    overall_suitability = 0.5  # Neutral if no conditions
                
                # Create recommendation object
                recommendation = {
                    'food_id': food.name,
                    'name': food.get('Thai_Name', food.get('English_Name', f"Food {food.name}")),
                    'category': food.get('Category', 'Unknown'),
                    'calories': float(food.get('Energy(kcal) by calculation', 0)),
                    'protein': float(food.get('Protein(g)', 0)),
                    'carbs': float(food.get('CHOCDF (g) Carbohydrate', 0)),
                    'sugar': float(food.get('SUGAR(g)', 0)),
                    'fiber': float(food.get('FIBTG (g) Dietary fibre', 0)),
                    'fat': float(food.get('Fat(g)', 0)),
                    'sodium': float(food.get('Na(mg)', 0)),
                    'potassium': float(food.get('K(mg)', 0)),
                    'cholesterol': float(food.get('CHOLE(mg) Cholesterol', 0)),
                    'saturated_fat': float(food.get('FASAT (g) Saturated FA', 0)),
                    'overall_suitability': overall_suitability,
                    'condition_suitability': suitability_scores,
                    'suitable_for_conditions': [cond for cond, score in suitability_scores.items() if score > 0.5],
                    'diabetes_score': float(food.get('Diabetes_Score', 0)),
                    'obesity_score': float(food.get('Obesity_Score', 0)),
                    'hypertension_score': float(food.get('Hypertension_Score', 0)),
                    'cholesterol_score': float(food.get('High_Cholesterol_Score', 0))
                }
                
                scores.append(recommendation)
            
            # Sort by overall suitability (higher is better)
            scores.sort(key=lambda x: x['overall_suitability'], reverse=True)
            
            # Return top recommendations
            return scores[:max_recommendations]
            
        except Exception as e:
            self.update_status(f"Error getting recommendations: {e}")
            return []
    
    def get_stats(self):
        """Get statistics about the recommendation system"""
        return self.stats
    
    def export_performance_metrics(self, filename=None):
        """Export performance metrics to file"""
        if not filename:
            filename = f"rf_performance_metrics_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(filename, 'w') as f:
                if 'performance_report' in self.stats:
                    f.write(self.stats['performance_report'])
                    f.write("\n\nDETAILED METRICS:\n")
                    f.write("=" * 50 + "\n")
                    
                    for metric in self.stats['detailed_metrics']:
                        f.write(f"\nModel: {metric.get('model_name', 'Unknown')}\n")
                        f.write(f"Target/Condition: {metric.get('condition', metric.get('target', 'Unknown'))}\n")
                        
                        for key, value in metric.items():
                            if key not in ['model_name', 'condition', 'target', 'confusion_matrix', 'classification_report']:
                                f.write(f"{key}: {value}\n")
                        f.write("-" * 30 + "\n")
                
                self.update_status(f"Performance metrics exported to {filename}")
                return filename
        except Exception as e:
            self.update_status(f"Error exporting metrics: {e}")
            return None


# Example usage and testing
def main():
    """Main function to test the enhanced Random Forest system"""
    print("Starting Enhanced Random Forest Food Recommendation System...")
    
    def status_callback(message):
        print(f"Status: {message}")
    
    # Initialize the recommender system
    recommender = HealthAwareRandomForestRecommender(status_callback)
    
    # Print performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(recommender.get_performance_summary())
    
    # Export detailed metrics
    filename = recommender.export_performance_metrics()
    if filename:
        print(f"\nDetailed metrics exported to: {filename}")
    
    # Test recommendations
    test_profile = {
        'health_conditions': ['Diabetes', 'Hypertension'],
        'category_filter': 'All'
    }
    
    recommendations = recommender.get_recommendations(test_profile, max_recommendations=5)
    
    print(f"\nSample recommendations for {test_profile['health_conditions']}:")
    print("-" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']}")
        print(f"   Overall Suitability: {rec['overall_suitability']:.3f}")
        print(f"   Suitable for: {rec['suitable_for_conditions']}")
        print(f"   Calories: {rec['calories']:.0f}, Sugar: {rec['sugar']:.1f}g, Sodium: {rec['sodium']:.0f}mg")
        print()


if __name__ == "__main__":
    main()
