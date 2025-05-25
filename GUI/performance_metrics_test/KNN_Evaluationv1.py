import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
import warnings
import json
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, mean_squared_error, 
                           mean_absolute_error, r2_score)
from scipy.spatial.distance import euclidean
import seaborn as sns

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Set DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class PerformanceEvaluator:
    """Comprehensive performance evaluation for KNN models"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.test_data = {}
        
    def evaluate_classification_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate classification model performance"""
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store results
        self.results[model_name] = {
            'type': 'classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'n_samples': len(y_test),
            'n_features': X_test.shape[1]
        }
        
        self.models[model_name] = model
        self.test_data[model_name] = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
        
        return self.results[model_name]
    
    def evaluate_regression_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate regression model performance"""
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Store results
        self.results[model_name] = {
            'type': 'regression',
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'n_samples': len(y_test),
            'n_features': X_test.shape[1]
        }
        
        self.models[model_name] = model
        self.test_data[model_name] = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
        
        return self.results[model_name]
    
    def generate_comprehensive_report(self, additional_info=None):
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'algorithm': 'K-Nearest Neighbors (KNN)',
                'purpose': 'Health-Aware Food Recommendation System',
                'institution': 'Prince of Songkla University'
            },
            'models_evaluated': len(self.results),
            'model_results': self.results
        }
        
        if additional_info:
            report.update(additional_info)
        
        return report
    
    def save_report(self, filename='knn_performance_report.json'):
        """Save evaluation report to JSON file"""
        report = self.generate_comprehensive_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        return filename


class MedicalNutritionCalculator:
    """Medical-grade nutritional calculation engine"""
    
    def __init__(self):
        self.activity_multipliers = {
            'Sedentary': 1.2,
            'Light': 1.375,
            'Moderate': 1.55,
            'Active': 1.725,
            'Very Active': 1.9
        }
        
        self.medical_guidelines = {
            'Diabetes': {
                'sugar_max_percent': 5,
                'sugar_max_grams': 25,
                'carb_percent': (40, 50),
                'protein_percent': (15, 25),
                'fat_percent': (25, 35),
                'fiber_min': 25,
                'sodium_max': 2300,
            },
            'Obesity': {
                'sugar_max_percent': 5,
                'sugar_max_grams': 25,
                'carb_percent': (45, 60),
                'protein_percent': (20, 35),
                'protein_grams_per_kg': (1.2, 1.6),
                'fat_percent': (20, 30),
                'fiber_min': 25,
                'sodium_max': 2300,
                'calorie_deficit': 500
            },
            'Hypertension': {
                'sugar_max_percent': 10,
                'carb_percent': (45, 65),
                'protein_percent': (15, 20),
                'fat_percent': (25, 30),
                'fiber_min': 30,
                'sodium_max': 1500,
                'potassium_min': 4700
            },
            'High_Cholesterol': {
                'sugar_max_percent': 10,
                'carb_percent': (45, 65),
                'protein_percent': (15, 20),
                'fat_percent': (25, 30),
                'fiber_min': 25,
                'sodium_max': 2300,
                'cholesterol_max': 200
            }
        }
    
    def calculate_bmr(self, weight_kg, height_cm, age_years, gender):
        """Calculate BMR using Mifflin-St Jeor equation"""
        if gender.lower() == 'male':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
        return bmr
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        multiplier = self.activity_multipliers.get(activity_level, 1.55)
        return bmr * multiplier
    
    def calculate_nutritional_targets(self, user_profile):
        """Calculate personalized nutritional targets"""
        weight_kg = user_profile['weight']
        height_cm = user_profile['height']
        age_years = user_profile['age']
        gender = user_profile['gender']
        activity_level = user_profile['activity_level']
        weight_goal = user_profile.get('weight_goal', 'Maintain Weight')
        health_conditions = user_profile.get('health_conditions', [])
        
        bmr = self.calculate_bmr(weight_kg, height_cm, age_years, gender)
        tdee = self.calculate_tdee(bmr, activity_level)
        
        # Adjust calories based on weight goal
        target_calories = tdee
        if weight_goal == 'Lose Weight':
            target_calories = tdee - 300
        elif weight_goal == 'Gain Weight':
            target_calories = tdee + 300
        
        target_calories = max(1200, target_calories)
        
        # Base targets
        targets = {
            'calories': target_calories,
            'carbs_min': (target_calories * 0.45) / 4,
            'carbs_max': (target_calories * 0.60) / 4,
            'protein_min': (target_calories * 0.15) / 4,
            'protein_max': (target_calories * 0.25) / 4,
            'fat_min': (target_calories * 0.25) / 9,
            'fat_max': (target_calories * 0.35) / 9,
            'sugar_max': (target_calories * 0.10) / 4,
            'fiber_min': 25,
            'sodium_max': 2300,
        }
        
        # Apply health condition modifications
        for condition in health_conditions:
            if condition in self.medical_guidelines:
                guidelines = self.medical_guidelines[condition]
                
                if 'sugar_max_percent' in guidelines:
                    sugar_from_percent = (target_calories * guidelines['sugar_max_percent'] / 100) / 4
                    sugar_from_grams = guidelines.get('sugar_max_grams', float('inf'))
                    targets['sugar_max'] = min(sugar_from_percent, sugar_from_grams)
                
                if 'carb_percent' in guidelines:
                    carb_min, carb_max = guidelines['carb_percent']
                    targets['carbs_min'] = (target_calories * carb_min / 100) / 4
                    targets['carbs_max'] = (target_calories * carb_max / 100) / 4
                
                if 'fiber_min' in guidelines:
                    targets['fiber_min'] = max(targets['fiber_min'], guidelines['fiber_min'])
                
                if 'sodium_max' in guidelines:
                    targets['sodium_max'] = min(targets['sodium_max'], guidelines['sodium_max'])
        
        # Meal distribution
        meal_targets = {
            'breakfast': {k: v * 0.25 for k, v in targets.items()},
            'lunch': {k: v * 0.30 for k, v in targets.items()},
            'dinner': {k: v * 0.30 for k, v in targets.items()},
            'snack': {k: v * 0.075 for k, v in targets.items()}
        }
        
        return {
            'daily_targets': targets,
            'meal_targets': meal_targets,
            'bmr': bmr,
            'tdee': tdee
        }


class EnhancedKNNRecommender:
    """Enhanced KNN recommender with comprehensive performance evaluation"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        self.evaluator = PerformanceEvaluator()
        
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
        
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'evaluation_results': {}
        }
        
        start_time = time.time()
        self.load_data()
        self.prepare_features()
        self.train_and_evaluate_models()
        self.stats['loading_time'] = time.time() - start_time
    
    def update_status(self, message):
        if self.status_callback:
            try:
                self.status_callback(message)
            except:
                pass
        print(message)
    
    def load_data(self):
        """Load and combine food datasets"""
        try:
            csv_files = glob.glob('./datasets/*.csv')
            if not csv_files:
                csv_files = glob.glob('*.csv')
            
            if not csv_files:
                self.update_status("No CSV files found!")
                self.food_data = pd.DataFrame()
                return
            
            dataframes = []
            category_map = {
                'drinking': 'Beverages', 'fruit': 'Fruits', 'meat': 'Meat & Poultry',
                'noodle': 'Noodles', 'onedish': 'One Dish Meals', 'vegetables': 'Vegetables',
                'cracker': 'Snacks', 'curry': 'Curries', 'dessert': 'Desserts',
                'esan_food': 'Esan Food', 'processed_food': 'Processed Food'
            }
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0].lower()
                    category = category_map.get(base_name, base_name.title())
                    
                    self.update_status(f"Loading {category}...")
                    
                    df = pd.read_csv(file_path)
                    df['Category'] = category
                    df = self._clean_data(df)
                    
                    if len(df) > 0:
                        dataframes.append(df)
                        
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
                self.update_status(f"Loaded {len(self.food_data)} food items")
                self._calculate_health_scores()
            else:
                self.food_data = pd.DataFrame()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            self.food_data = pd.DataFrame()
    
    def _clean_data(self, df):
        """Clean nutritional data"""
        for col in self.nutritional_features + ['FASAT (g) Saturated FA']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove rows with all zero nutritional values
        nutrient_cols = [col for col in self.nutritional_features if col in df.columns]
        if nutrient_cols:
            df = df[df[nutrient_cols].sum(axis=1) > 0]
        
        return df
    
    def _calculate_health_scores(self):
        """Calculate health suitability scores"""
        self.update_status("Calculating health scores...")
        
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = 0
        
        for idx, food in self.food_data.iterrows():
            # Diabetes score
            score = 0
            sugar = float(food.get('SUGAR(g)', 0))
            if sugar > 15: score += 3
            elif sugar > 8: score += 2
            elif sugar > 3: score += 1
            
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'Diabetes_Score'] = max(0, score)
            
            # Obesity score
            score = 0
            calories = float(food.get('Energy(kcal) by calculation', 0))
            if calories > 300: score += 3
            elif calories > 200: score += 2
            elif calories > 150: score += 1
            
            protein = float(food.get('Protein(g)', 0))
            if protein >= 10: score -= 1
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'Obesity_Score'] = max(0, score)
            
            # Hypertension score
            score = 0
            sodium = float(food.get('Na(mg)', 0))
            if sodium > 400: score += 3
            elif sodium > 200: score += 2
            elif sodium > 100: score += 1
            
            potassium = float(food.get('K(mg)', 0))
            if potassium > 300: score -= 1
            
            self.food_data.at[idx, 'Hypertension_Score'] = max(0, score)
            
            # High cholesterol score
            score = 0
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            if sat_fat > 5: score += 3
            elif sat_fat > 3: score += 2
            elif sat_fat > 1: score += 1
            
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'High_Cholesterol_Score'] = max(0, score)
        
        # Create healthiness categories for classification
        self.food_data['Health_Category'] = pd.cut(
            (self.food_data['Diabetes_Score'] + self.food_data['Obesity_Score'] + 
             self.food_data['Hypertension_Score'] + self.food_data['High_Cholesterol_Score']) / 4,
            bins=[0, 1, 2, float('inf')],
            labels=['Healthy', 'Moderate', 'Unhealthy']
        )
        
        # Create suitability score for regression
        self.food_data['Overall_Suitability'] = (
            self.food_data['Diabetes_Score'] + self.food_data['Obesity_Score'] + 
            self.food_data['Hypertension_Score'] + self.food_data['High_Cholesterol_Score']
        ) / 4
    
    def prepare_features(self):
        """Prepare features for KNN"""
        if len(self.food_data) == 0:
            return
        
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features = available_features + health_features
    
    def train_and_evaluate_models(self):
        """Train and evaluate KNN models with comprehensive metrics"""
        if len(self.food_data) == 0 or not hasattr(self, 'features'):
            return
        
        try:
            X = self.food_data[self.features].fillna(0)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 1. Classification Task: Predict Health Category
            self.update_status("Evaluating KNN Classification...")
            y_classification = self.food_data['Health_Category'].dropna()
            X_classification = X_scaled[:len(y_classification)]
            
            if len(y_classification) > 0:
                le = LabelEncoder()
                y_classification_encoded = le.fit_transform(y_classification)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_classification, y_classification_encoded, 
                    test_size=0.2, random_state=42, stratify=y_classification_encoded
                )
                
                # Test different k values
                k_values = [3, 5, 7, 9, 11]
                best_k = 5
                best_score = 0
                
                for k in k_values:
                    knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
                    scores = cross_val_score(knn_clf, X_train, y_train, cv=3, scoring='accuracy')
                    if scores.mean() > best_score:
                        best_score = scores.mean()
                        best_k = k
                
                # Train best model
                self.knn_classifier = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
                clf_results = self.evaluator.evaluate_classification_model(
                    self.knn_classifier, X_train, X_test, y_train, y_test, 
                    f'KNN_Classification_k{best_k}'
                )
                
                # Add class labels
                clf_results['class_labels'] = le.classes_.tolist()
                clf_results['best_k'] = best_k
            
            # 2. Regression Task: Predict Overall Suitability Score
            self.update_status("Evaluating KNN Regression...")
            y_regression = self.food_data['Overall_Suitability']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_regression, test_size=0.2, random_state=42
            )
            
            # Test different k values for regression
            best_k_reg = 5
            best_score_reg = -float('inf')
            
            for k in k_values:
                knn_reg = KNeighborsRegressor(n_neighbors=k, weights='distance')
                scores = cross_val_score(knn_reg, X_train, y_train, cv=3, scoring='r2')
                if scores.mean() > best_score_reg:
                    best_score_reg = scores.mean()
                    best_k_reg = k
            
            # Train best regression model
            self.knn_regressor = KNeighborsRegressor(n_neighbors=best_k_reg, weights='distance')
            reg_results = self.evaluator.evaluate_regression_model(
                self.knn_regressor, X_train, X_test, y_train, y_test, 
                f'KNN_Regression_k{best_k_reg}'
            )
            reg_results['best_k'] = best_k_reg
            
            # 3. Condition-specific evaluations
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                self.update_status(f"Evaluating {condition} specific model...")
                y_condition = self.food_data[f'{condition}_Score']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_condition, test_size=0.2, random_state=42
                )
                
                knn_condition = KNeighborsRegressor(n_neighbors=7, weights='distance')
                condition_results = self.evaluator.evaluate_regression_model(
                    knn_condition, X_train, X_test, y_train, y_test, 
                    f'KNN_{condition}_Score'
                )
            
            # Store results in stats
            self.stats['evaluation_results'] = self.evaluator.results
            
            # Generate and save comprehensive report
            additional_info = {
                'dataset_info': {
                    'total_samples': len(self.food_data),
                    'total_features': len(self.features),
                    'features_used': self.features,
                    'categories': list(self.stats['categories'].keys()),
                    'category_distribution': self.stats['categories']
                },
                'model_hyperparameters': {
                    'best_k_classification': best_k if 'best_k' in locals() else None,
                    'best_k_regression': best_k_reg,
                    'weight_function': 'distance',
                    'distance_metric': 'euclidean'
                }
            }
            
            report_filename = self.evaluator.save_report('knn_performance_report.json')
            self.update_status(f"Performance report saved as {report_filename}")
            
        except Exception as e:
            self.update_status(f"Error in evaluation: {e}")
    
    def get_recommendations(self, user_profile, meal_type='lunch', max_recommendations=10):
        """Get food recommendations using trained KNN"""
        if not hasattr(self, 'knn_regressor') or len(self.food_data) == 0:
            return []
        
        try:
            # Calculate targets
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            meal_targets = nutritional_data['meal_targets'].get(meal_type, nutritional_data['meal_targets']['lunch'])
            health_conditions = user_profile.get('health_conditions', [])
            
            # Filter by category if specified
            candidates = self.food_data.copy()
            category_filter = user_profile.get('category_filter', 'All')
            if category_filter != 'All':
                candidates = candidates[candidates['Category'] == category_filter]
                if len(candidates) == 0:
                    candidates = self.food_data.copy()
            
            if len(candidates) == 0:
                return []
            
            # Create user vector and get recommendations
            user_vector = self._create_user_vector(meal_targets)
            candidate_features = candidates[self.features].fillna(0)
            candidate_features_scaled = self.scaler.transform(candidate_features)
            scaled_user_vector = self.scaler.transform(user_vector)
            
            # Predict suitability scores
            predicted_scores = self.knn_regressor.predict(candidate_features_scaled)
            
            # Calculate distances and combined scores
            recommendations = []
            for i, (idx, candidate) in enumerate(candidates.iterrows()):
                distance = euclidean(scaled_user_vector[0], candidate_features_scaled[i])
                predicted_score = predicted_scores[i]
                combined_score = 0.6 * distance + 0.4 * predicted_score
                
                rec = {
                    'food_id': idx,
                    'name': candidate.get('Thai_Name', candidate.get('English_Name', f"Food {idx}")),
                    'category': candidate.get('Category', 'Unknown'),
                    'calories': float(candidate.get('Energy(kcal) by calculation', 0)),
                    'protein': float(candidate.get('Protein(g)', 0)),
                    'carbs': float(candidate.get('CHOCDF (g) Carbohydrate', 0)),
                    'sugar': float(candidate.get('SUGAR(g)', 0)),
                    'fiber': float(candidate.get('FIBTG (g) Dietary fibre', 0)),
                    'fat': float(candidate.get('Fat(g)', 0)),
                    'sodium': float(candidate.get('Na(mg)', 0)),
                    'distance': distance,
                    'predicted_suitability': predicted_score,
                    'combined_score': combined_score,
                    'suitable_for_conditions': self._check_suitability(candidate)
                }
                
                recommendations.append(rec)
            
            # Sort by combined score and return top recommendations
            recommendations.sort(key=lambda x: x['combined_score'])
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.update_status(f"Error getting recommendations: {e}")
            return []
    
    def _create_user_vector(self, targets):
        """Create user profile vector"""
        user_vector = np.zeros(len(self.features))
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
        if 'Energy(kcal) by calculation' in feature_map:
            user_vector[feature_map['Energy(kcal) by calculation']] = targets.get('calories', 500)
        if 'Protein(g)' in feature_map:
            user_vector[feature_map['Protein(g)']] = (targets.get('protein_min', 0) + targets.get('protein_max', 50)) / 2
        if 'CHOCDF (g) Carbohydrate' in feature_map:
            user_vector[feature_map['CHOCDF (g) Carbohydrate']] = (targets.get('carbs_min', 0) + targets.get('carbs_max', 50)) / 2
        if 'SUGAR(g)' in feature_map:
            user_vector[feature_map['SUGAR(g)']] = targets.get('sugar_max', 10) * 0.5
        if 'FIBTG (g) Dietary fibre' in feature_map:
            user_vector[feature_map['FIBTG (g) Dietary fibre']] = targets.get('fiber_min', 5)
        if 'Fat(g)' in feature_map:
            user_vector[feature_map['Fat(g)']] = (targets.get('fat_min', 0) + targets.get('fat_max', 20)) / 2
        if 'Na(mg)' in feature_map:
            user_vector[feature_map['Na(mg)']] = targets.get('sodium_max', 1000) * 0.3
        
        # Set ideal health scores
        for health_feature in ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']:
            if health_feature in feature_map:
                user_vector[feature_map[health_feature]] = 0
        
        return user_vector.reshape(1, -1)
    
    def _check_suitability(self, food):
        """Check condition suitability"""
        suitable = []
        if float(food.get('Diabetes_Score', 0)) <= 1.5:
            suitable.append('Diabetes')
        if float(food.get('Obesity_Score', 0)) <= 1.5:
            suitable.append('Obesity')
        if float(food.get('Hypertension_Score', 0)) <= 1.5:
            suitable.append('Hypertension')
        if float(food.get('High_Cholesterol_Score', 0)) <= 1.5:
            suitable.append('High_Cholesterol')
        return suitable
    
    def get_stats(self):
        return self.stats
    
    def get_evaluation_results(self):
        """Get comprehensive evaluation results"""
        return self.evaluator.results


class EnhancedKNNFoodRecommenderUI:
    """Enhanced UI with performance evaluation display"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Enhanced KNN Food Recommender with Performance Evaluation")
        self.master.geometry("1600x1000")
        self.master.configure(bg="#f8f9fa")
        
        # Colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'white': '#ffffff',
            'background': '#f8f9fa',
            'text': '#2c3e50',
            'text_light': '#7f8c8d'
        }
        
        self.setup_styles()
        self.recommender = None
        self.last_recommendations = []
        
        self.create_interface()
        self.initialize_system()
    
    def setup_styles(self):
        """Setup UI styles"""
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        self.fonts = {
            'heading': ('Segoe UI', 16, 'bold'),
            'subheading': ('Segoe UI', 12, 'bold'),
            'body': ('Segoe UI', 10),
            'caption': ('Segoe UI', 9)
        }
    
    def create_interface(self):
        """Create the enhanced interface"""
        # Main container
        main_container = ttk.Frame(self.master, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area with tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Create tabs
        self.create_recommendation_tab()
        self.create_performance_tab()
        self.create_visualization_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self, parent):
        """Create header"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(title_frame, text="Enhanced KNN Food Recommender", 
                font=self.fonts['heading'], fg=self.colors['primary'], 
                bg=self.colors['background']).pack(anchor=tk.W)
        tk.Label(title_frame, text="With Comprehensive Performance Evaluation", 
                font=self.fonts['subheading'], fg=self.colors['text_light'], 
                bg=self.colors['background']).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self.get_recommendations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Show Performance", 
                  command=self.show_performance).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Export Report", 
                  command=self.export_report).pack(side=tk.RIGHT, padx=5)
    
    def create_recommendation_tab(self):
        """Create recommendation tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="Food Recommendations")
        
        # Left panel for inputs
        input_frame = ttk.LabelFrame(rec_frame, text="User Profile", padding="15")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Personal info
        self.weight_var = tk.DoubleVar(value=70.0)
        self.height_var = tk.DoubleVar(value=170.0)
        self.age_var = tk.IntVar(value=30)
        self.gender_var = tk.StringVar(value="Male")
        self.activity_var = tk.StringVar(value="Moderate")
        self.weight_goal_var = tk.StringVar(value="Maintain Weight")
        
        # Health conditions
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        # Settings
        self.meal_type_var = tk.StringVar(value="Lunch")
        self.category_var = tk.StringVar(value="All")
        self.max_results_var = tk.StringVar(value="10")
        
        # Create input widgets (simplified for brevity)
        row = 0
        for label, var in [("Weight (kg)", self.weight_var), ("Height (cm)", self.height_var), 
                          ("Age", self.age_var)]:
            tk.Label(input_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            tk.Entry(input_frame, textvariable=var, width=15).grid(row=row, column=1, pady=2)
            row += 1
        
        # Dropdowns
        for label, var, values in [("Gender", self.gender_var, ['Male', 'Female']),
                                   ("Activity", self.activity_var, ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active']),
                                   ("Weight Goal", self.weight_goal_var, ['Lose Weight', 'Maintain Weight', 'Gain Weight'])]:
            tk.Label(input_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Combobox(input_frame, textvariable=var, values=values, state="readonly", width=12).grid(row=row, column=1, pady=2)
            row += 1
        
        # Health conditions
        tk.Label(input_frame, text="Health Conditions:", font=self.fonts['subheading']).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10,5))
        row += 1
        
        for label, var in [("Diabetes", self.diabetes_var), ("Obesity", self.obesity_var),
                          ("Hypertension", self.hypertension_var), ("High Cholesterol", self.cholesterol_var)]:
            tk.Checkbutton(input_frame, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky=tk.W)
            row += 1
        
        # Right panel for results
        results_frame = ttk.LabelFrame(rec_frame, text="Recommendations", padding="15")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results treeview
        columns = ('Name', 'Category', 'Calories', 'Distance', 'Predicted Score', 'Combined Score')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_performance_tab(self):
        """Create performance evaluation tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="Performance Metrics")
        
        # Create text widget for performance results
        self.performance_text = tk.Text(perf_frame, wrap=tk.WORD, font=self.fonts['body'])
        perf_scrollbar = ttk.Scrollbar(perf_frame, orient="vertical", command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scrollbar.set)
        
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
    
    def create_visualization_tab(self):
        """Create visualization tab"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Performance Visualization")
        
        # Create matplotlib figure
        self.perf_fig = Figure(figsize=(14, 10), dpi=100)
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, viz_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding="5")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Initializing enhanced KNN system...")
        ttk.Label(self.status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
    
    def initialize_system(self):
        """Initialize the enhanced system"""
        def init_system():
            try:
                self.status_var.set("Loading and evaluating KNN models...")
                self.master.update()
                
                self.recommender = EnhancedKNNRecommender(self.update_status)
                
                self.status_var.set("System ready! Performance evaluation completed.")
                self.show_performance()
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to initialize system: {str(e)}")
        
        self.master.after(100, init_system)
    
    def update_status(self, message):
        """Update status"""
        self.status_var.set(message)
        self.master.update()
    
    def get_user_profile(self):
        """Get user profile"""
        health_conditions = []
        if self.diabetes_var.get():
            health_conditions.append('Diabetes')
        if self.obesity_var.get():
            health_conditions.append('Obesity')
        if self.hypertension_var.get():
            health_conditions.append('Hypertension')
        if self.cholesterol_var.get():
            health_conditions.append('High_Cholesterol')
        
        return {
            'weight': self.weight_var.get(),
            'height': self.height_var.get(),
            'age': self.age_var.get(),
            'gender': self.gender_var.get(),
            'activity_level': self.activity_var.get(),
            'weight_goal': self.weight_goal_var.get(),
            'health_conditions': health_conditions,
            'category_filter': self.category_var.get()
        }
    
    def get_recommendations(self):
        """Get food recommendations"""
        if not self.recommender:
            messagebox.showwarning("Warning", "System not ready!")
            return
        
        try:
            user_profile = self.get_user_profile()
            recommendations = self.recommender.get_recommendations(user_profile, 
                                                                 self.meal_type_var.get().lower(), 
                                                                 int(self.max_results_var.get()))
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Display recommendations
            for rec in recommendations:
                self.results_tree.insert('', 'end', values=(
                    rec['name'][:30],
                    rec['category'],
                    f"{rec['calories']:.0f}",
                    f"{rec['distance']:.3f}",
                    f"{rec['predicted_suitability']:.3f}",
                    f"{rec['combined_score']:.3f}"
                ))
            
            self.last_recommendations = recommendations
            self.status_var.set(f"Found {len(recommendations)} recommendations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {str(e)}")
    
    def show_performance(self):
        """Display performance metrics"""
        if not self.recommender:
            return
        
        try:
            results = self.recommender.get_evaluation_results()
            
            # Clear and update performance text
            self.performance_text.delete(1.0, tk.END)
            
            self.performance_text.insert(tk.END, "KNN PERFORMANCE EVALUATION RESULTS\n", "header")
            self.performance_text.insert(tk.END, "="*50 + "\n\n", "separator")
            
            for model_name, metrics in results.items():
                self.performance_text.insert(tk.END, f"📊 {model_name}\n", "subheader")
                self.performance_text.insert(tk.END, "-" * 30 + "\n")
                
                if metrics['type'] == 'classification':
                    self.performance_text.insert(tk.END, f"• Accuracy: {metrics['accuracy']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• Precision: {metrics['precision']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• Recall: {metrics['recall']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• F1-Score: {metrics['f1_score']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• Cross-Validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}\n")
                    if 'best_k' in metrics:
                        self.performance_text.insert(tk.END, f"• Best K: {metrics['best_k']}\n")
                    
                elif metrics['type'] == 'regression':
                    self.performance_text.insert(tk.END, f"• R² Score: {metrics['r2_score']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• MSE: {metrics['mse']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• MAE: {metrics['mae']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• RMSE: {metrics['rmse']:.4f}\n")
                    self.performance_text.insert(tk.END, f"• Cross-Validation: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}\n")
                    if 'best_k' in metrics:
                        self.performance_text.insert(tk.END, f"• Best K: {metrics['best_k']}\n")
                
                self.performance_text.insert(tk.END, f"• Samples: {metrics['n_samples']}\n")
                self.performance_text.insert(tk.END, f"• Features: {metrics['n_features']}\n\n")
            
            # Configure text tags
            self.performance_text.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.performance_text.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.performance_text.tag_configure("separator", foreground=self.colors['text_light'])
            
            # Update visualizations
            self.update_performance_visualizations(results)
            
            # Switch to performance tab
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying performance: {str(e)}")
    
    def update_performance_visualizations(self, results):
        """Update performance visualization charts"""
        self.perf_fig.clear()
        
        # Create subplots
        axes = []
        for i in range(2):
            for j in range(3):
                axes.append(self.perf_fig.add_subplot(2, 3, i*3 + j + 1))
        
        plot_idx = 0
        
        # 1. Accuracy/R² comparison
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            models = []
            scores = []
            colors_list = []
            
            for model_name, metrics in results.items():
                models.append(model_name.replace('_', '\n'))
                if metrics['type'] == 'classification':
                    scores.append(metrics['accuracy'])
                    colors_list.append('#3498db')
                else:
                    scores.append(metrics['r2_score'])
                    colors_list.append('#e74c3c')
            
            bars = ax.bar(models, scores, color=colors_list, alpha=0.7)
            ax.set_title('Model Performance Scores', fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plot_idx += 1
        
        # 2. Cross-validation scores
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            cv_means = []
            cv_stds = []
            model_names = []
            
            for model_name, metrics in results.items():
                model_names.append(model_name.replace('_', '\n'))
                cv_means.append(metrics['cv_mean'])
                cv_stds.append(metrics['cv_std'])
            
            bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                         color='#27ae60', alpha=0.7, error_kw={'color': 'black'})
            ax.set_title('Cross-Validation Scores', fontweight='bold')
            ax.set_ylabel('CV Score ± Std')
            ax.tick_params(axis='x', rotation=45)
            
            plot_idx += 1
        
        # 3. Confusion Matrix (for classification models)
        classification_results = {k: v for k, v in results.items() if v['type'] == 'classification'}
        if classification_results and plot_idx < len(axes):
            ax = axes[plot_idx]
            # Show confusion matrix for the first classification model
            model_name = list(classification_results.keys())[0]
            cm = np.array(classification_results[model_name]['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix\n{model_name}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            if 'class_labels' in classification_results[model_name]:
                labels = classification_results[model_name]['class_labels']
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
            
            plot_idx += 1
        
        # 4. Feature count comparison
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            feature_counts = [metrics['n_features'] for metrics in results.values()]
            sample_counts = [metrics['n_samples'] for metrics in results.values()]
            model_names = [name.replace('_', '\n') for name in results.keys()]
            
            ax.bar(model_names, feature_counts, alpha=0.7, color='#f39c12', label='Features')
            ax.set_title('Dataset Statistics', fontweight='bold')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            
            plot_idx += 1
        
        # 5. Error metrics for regression models
        regression_results = {k: v for k, v in results.items() if v['type'] == 'regression'}
        if regression_results and plot_idx < len(axes):
            ax = axes[plot_idx]
            model_names = []
            rmse_values = []
            mae_values = []
            
            for model_name, metrics in regression_results.items():
                model_names.append(model_name.replace('_', '\n'))
                rmse_values.append(metrics['rmse'])
                mae_values.append(metrics['mae'])
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.7, color='#e74c3c')
            ax.bar(x + width/2, mae_values, width, label='MAE', alpha=0.7, color='#9b59b6')
            
            ax.set_title('Regression Error Metrics', fontweight='bold')
            ax.set_ylabel('Error')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            
            plot_idx += 1
        
        # 6. Sample distribution
        if plot_idx < len(axes) and hasattr(self.recommender, 'food_data'):
            ax = axes[plot_idx]
            
            # Show health category distribution
            if 'Health_Category' in self.recommender.food_data.columns:
                category_counts = self.recommender.food_data['Health_Category'].value_counts()
                
                ax.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                      colors=['#27ae60', '#f39c12', '#e74c3c'])
                ax.set_title('Health Category Distribution', fontweight='bold')
        
        self.perf_fig.suptitle('KNN Model Performance Analysis', fontsize=16, fontweight='bold')
        self.perf_fig.tight_layout()
        self.perf_canvas.draw()
    
    def export_report(self):
        """Export performance report"""
        if not self.recommender:
            messagebox.showwarning("Warning", "System not ready!")
            return
        
        try:
            # Generate additional report with recommendations sample
            additional_info = {
                'sample_recommendations': self.last_recommendations[:5] if self.last_recommendations else [],
                'ui_session_info': {
                    'timestamp': datetime.now().isoformat(),
                    'user_profile_used': self.get_user_profile() if hasattr(self, 'get_user_profile') else {},
                    'recommendations_generated': len(self.last_recommendations)
                }
            }
            
            report = self.recommender.evaluator.generate_comprehensive_report(additional_info)
            
            # Save report
            filename = f"knn_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            messagebox.showinfo("Export Successful", f"Performance report exported as:\n{filename}")
            self.status_var.set(f"Report exported: {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report: {str(e)}")


def main():
    """Main application entry point"""
    print("Starting Enhanced KNN Food Recommendation System with Performance Evaluation...")
    
    root = tk.Tk()
    app = EnhancedKNNFoodRecommenderUI(root)
    
    # Center window
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - root.winfo_width()) // 2
    y = (screen_height - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()