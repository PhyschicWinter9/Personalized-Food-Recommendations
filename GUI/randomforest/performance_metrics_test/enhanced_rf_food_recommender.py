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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
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
    """Comprehensive model performance evaluation for food recommendation system"""
    
    def __init__(self):
        self.metrics_history = []
        self.model_comparisons = {}
    
    def evaluate_regression_model(self, model, X_test, y_test, X_train, y_train, model_name):
        """Evaluate regression model performance"""
        # Predictions
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Regression metrics
        metrics = {
            'model_name': model_name,
            'model_type': 'regression',
            'timestamp': datetime.now().isoformat(),
            
            # Basic regression metrics
            'r2_score_test': r2_score(y_test, y_pred_test),
            'r2_score_train': r2_score(y_train, y_pred_train),
            'mse_test': mean_squared_error(y_test, y_pred_test),
            'mse_train': mean_squared_error(y_train, y_pred_train),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            
            # Model complexity metrics
            'overfitting_indicator': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
            
            # Prediction quality metrics
            'prediction_std': np.std(y_pred_test),
            'prediction_range': np.max(y_pred_test) - np.min(y_pred_test),
            'residuals_mean': np.mean(y_test - y_pred_test),
            'residuals_std': np.std(y_test - y_pred_test),
            
            # Cross-validation metrics
            'cv_scores': None,  # Will be filled by cross-validation
            'cv_mean': None,
            'cv_std': None
        }
        
        return metrics
    
    def evaluate_classification_model(self, model, X_test, y_test, X_train, y_train, model_name, average='weighted'):
        """Evaluate classification model performance"""
        # Predictions
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Get prediction probabilities if available
        try:
            y_proba_test = model.predict_proba(X_test)
            has_proba = True
        except:
            y_proba_test = None
            has_proba = False
        
        # Classification metrics
        metrics = {
            'model_name': model_name,
            'model_type': 'classification',
            'timestamp': datetime.now().isoformat(),
            
            # Basic classification metrics
            'accuracy_test': accuracy_score(y_test, y_pred_test),
            'accuracy_train': accuracy_score(y_train, y_pred_train),
            'precision_test': precision_score(y_test, y_pred_test, average=average, zero_division=0),
            'precision_train': precision_score(y_train, y_pred_train, average=average, zero_division=0),
            'recall_test': recall_score(y_test, y_pred_test, average=average, zero_division=0),
            'recall_train': recall_score(y_train, y_pred_train, average=average, zero_division=0),
            'f1_test': f1_score(y_test, y_pred_test, average=average, zero_division=0),
            'f1_train': f1_score(y_train, y_pred_train, average=average, zero_division=0),
            
            # Model complexity metrics
            'overfitting_indicator': accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test),
            
            # Class distribution metrics
            'class_distribution_test': dict(zip(*np.unique(y_test, return_counts=True))),
            'class_distribution_pred': dict(zip(*np.unique(y_pred_test, return_counts=True))),
            
            # Additional metrics
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        }
        
        # Add AUC metrics if probabilities are available
        if has_proba and len(np.unique(y_test)) == 2:  # Binary classification
            try:
                metrics['auc_roc'] = roc_auc_score(y_test, y_proba_test[:, 1])
                precision, recall, _ = precision_recall_curve(y_test, y_proba_test[:, 1])
                metrics['auc_pr'] = auc(recall, precision)
            except:
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
        elif has_proba and len(np.unique(y_test)) > 2:  # Multiclass
            try:
                metrics['auc_roc_ovr'] = roc_auc_score(y_test, y_proba_test, multi_class='ovr', average=average)
                metrics['auc_roc_ovo'] = roc_auc_score(y_test, y_proba_test, multi_class='ovo', average=average)
            except:
                metrics['auc_roc_ovr'] = None
                metrics['auc_roc_ovo'] = None
        
        return metrics
    
    def evaluate_recommendation_quality(self, recommendations, user_profile, ground_truth=None):
        """Evaluate the quality of food recommendations"""
        if not recommendations:
            return {
                'recommendation_count': 0,
                'avg_score': 0,
                'score_variance': 0,
                'health_coverage': 0,
                'category_diversity': 0
            }
        
        # Basic recommendation metrics
        scores = [rec['combined_score'] for rec in recommendations]
        
        # Health condition coverage
        health_conditions = user_profile.get('health_conditions', [])
        suitable_foods = 0
        for rec in recommendations:
            if any(condition in rec['suitable_for_conditions'] for condition in health_conditions):
                suitable_foods += 1
        
        health_coverage = suitable_foods / len(recommendations) if recommendations else 0
        
        # Category diversity
        categories = set(rec['category'] for rec in recommendations)
        category_diversity = len(categories) / len(recommendations) if recommendations else 0
        
        # Nutritional balance
        avg_calories = np.mean([rec['calories'] for rec in recommendations])
        avg_protein = np.mean([rec['protein'] for rec in recommendations])
        avg_fiber = np.mean([rec['fiber'] for rec in recommendations])
        avg_sugar = np.mean([rec['sugar'] for rec in recommendations])
        
        metrics = {
            'recommendation_count': len(recommendations),
            'avg_combined_score': np.mean(scores),
            'score_variance': np.var(scores),
            'score_std': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'health_coverage': health_coverage,
            'category_diversity': category_diversity,
            'unique_categories': len(categories),
            'avg_calories': avg_calories,
            'avg_protein': avg_protein,
            'avg_fiber': avg_fiber,
            'avg_sugar': avg_sugar,
            'warnings_count': sum(len(rec['health_warnings']) for rec in recommendations),
            'targets_met_avg': np.mean([len(rec['targets_met']) for rec in recommendations])
        }
        
        return metrics
    
    def perform_cross_validation(self, model, X, y, cv_folds=5, model_type='regression'):
        """Perform cross-validation and return detailed metrics"""
        if model_type == 'regression':
            scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        else:
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            # Use stratified k-fold for classification
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for metric in scoring_metrics:
            if model_type == 'classification':
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            else:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
            
            cv_results[f'{metric}_scores'] = scores.tolist()
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        
        return cv_results
    
    def save_metrics_to_file(self, metrics, filename=None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_metrics_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            return filename
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return None
    
    def compare_models(self, metrics_list):
        """Compare multiple models and return comparison summary"""
        if not metrics_list:
            return {}
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': len(metrics_list),
            'comparison_summary': {}
        }
        
        # Separate regression and classification models
        regression_models = [m for m in metrics_list if m.get('model_type') == 'regression']
        classification_models = [m for m in metrics_list if m.get('model_type') == 'classification']
        
        # Compare regression models
        if regression_models:
            comparison['regression_comparison'] = self._compare_regression_models(regression_models)
        
        # Compare classification models
        if classification_models:
            comparison['classification_comparison'] = self._compare_classification_models(classification_models)
        
        return comparison
    
    def _compare_regression_models(self, models):
        """Compare regression models"""
        comparison = {}
        
        # Find best model for each metric
        best_r2 = max(models, key=lambda x: x.get('r2_score_test', 0))
        best_rmse = min(models, key=lambda x: x.get('rmse_test', float('inf')))
        best_mae = min(models, key=lambda x: x.get('mae_test', float('inf')))
        
        comparison['best_r2'] = {
            'model': best_r2['model_name'],
            'score': best_r2.get('r2_score_test', 0)
        }
        comparison['best_rmse'] = {
            'model': best_rmse['model_name'],
            'score': best_rmse.get('rmse_test', 0)
        }
        comparison['best_mae'] = {
            'model': best_mae['model_name'],
            'score': best_mae.get('mae_test', 0)
        }
        
        # Overall ranking
        model_scores = []
        for model in models:
            # Normalized scoring (higher is better)
            r2_norm = model.get('r2_score_test', 0)
            rmse_norm = 1 / (1 + model.get('rmse_test', 1))
            mae_norm = 1 / (1 + model.get('mae_test', 1))
            
            overall_score = (r2_norm + rmse_norm + mae_norm) / 3
            model_scores.append({
                'model': model['model_name'],
                'overall_score': overall_score,
                'r2': r2_norm,
                'rmse_inv': rmse_norm,
                'mae_inv': mae_norm
            })
        
        comparison['ranking'] = sorted(model_scores, key=lambda x: x['overall_score'], reverse=True)
        
        return comparison
    
    def _compare_classification_models(self, models):
        """Compare classification models"""
        comparison = {}
        
        # Find best model for each metric
        best_accuracy = max(models, key=lambda x: x.get('accuracy_test', 0))
        best_f1 = max(models, key=lambda x: x.get('f1_test', 0))
        best_precision = max(models, key=lambda x: x.get('precision_test', 0))
        best_recall = max(models, key=lambda x: x.get('recall_test', 0))
        
        comparison['best_accuracy'] = {
            'model': best_accuracy['model_name'],
            'score': best_accuracy.get('accuracy_test', 0)
        }
        comparison['best_f1'] = {
            'model': best_f1['model_name'],
            'score': best_f1.get('f1_test', 0)
        }
        comparison['best_precision'] = {
            'model': best_precision['model_name'],
            'score': best_precision.get('precision_test', 0)
        }
        comparison['best_recall'] = {
            'model': best_recall['model_name'],
            'score': best_recall.get('recall_test', 0)
        }
        
        # Overall ranking
        model_scores = []
        for model in models:
            # Weighted average of key metrics
            accuracy = model.get('accuracy_test', 0)
            f1 = model.get('f1_test', 0)
            precision = model.get('precision_test', 0)
            recall = model.get('recall_test', 0)
            
            overall_score = (accuracy * 0.3 + f1 * 0.4 + precision * 0.15 + recall * 0.15)
            model_scores.append({
                'model': model['model_name'],
                'overall_score': overall_score,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
        
        comparison['ranking'] = sorted(model_scores, key=lambda x: x['overall_score'], reverse=True)
        
        return comparison


class MedicalNutritionCalculator:
    """Medical-grade nutritional calculation engine based on established formulas and guidelines"""
    
    def __init__(self):
        # Activity level multipliers for TDEE calculation
        self.activity_multipliers = {
            'Sedentary': 1.2,        # Little to no exercise
            'Light': 1.375,          # Light exercise 1-3 days/week
            'Moderate': 1.55,        # Moderate exercise 3-5 days/week
            'Active': 1.725,         # Heavy exercise 6-7 days/week
            'Very Active': 1.9       # Very heavy physical work or 2x/day training
        }
        
        # Medical guidelines for different health conditions (per day)
        self.medical_guidelines = {
            'Diabetes': {
                'sugar_max_percent': 5,          # <5% of total calories (WHO recommendation)
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (40, 50),        # 40-50% of calories (lower than general)
                'protein_percent': (15, 25),     # 15-25% of calories
                'fat_percent': (25, 35),         # 25-35% of calories
                'saturated_fat_percent': 7,      # <7% of calories (ADA)
                'fiber_min': 25,                 # Minimum 25g/day
                'fiber_per_1000kcal': 14,        # 14g per 1000 kcal
                'sodium_max': 2300,              # <2300mg/day
                'cholesterol_max': 200           # <200mg/day
            },
            'Obesity': {
                'sugar_max_percent': 5,          # <5% of total calories
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (45, 60),        # 45-60% of calories
                'protein_percent': (20, 35),     # Higher protein for satiety
                'protein_grams_per_kg': (1.2, 1.6),  # 1.2-1.6g per kg for weight loss
                'fat_percent': (20, 30),         # 20-30% of calories
                'saturated_fat_percent': 7,      # <7% of calories
                'fiber_min': 25,                 # Minimum 25g/day
                'sodium_max': 2300,              # <2300mg/day
                'calorie_deficit': 500           # 500 kcal deficit for 1lb/week loss
            },
            'Hypertension': {
                'sugar_max_percent': 10,         # <10% of total calories
                'sugar_max_grams': 50,           # Maximum 50g/day
                'carb_percent': (45, 65),        # 45-65% of calories
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories (DASH)
                'fiber_min': 30,                 # Minimum 30g/day (DASH)
                'sodium_max': 1500,              # <1500mg/day (ideal for BP)
                'potassium_min': 4700,           # Minimum 4700mg/day
                'calcium_min': 1200,             # Minimum 1200mg/day
                'magnesium_min': 400             # Minimum 400mg/day
            },
            'High_Cholesterol': {
                'sugar_max_percent': 10,         # <10% of total calories
                'sugar_max_grams': 50,           # Maximum 50g/day
                'carb_percent': (45, 65),        # 45-65% of calories
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories (AHA)
                'fiber_min': 25,                 # Minimum 25g/day
                'soluble_fiber_min': 10,         # Minimum 10g soluble fiber
                'sodium_max': 2300,              # <2300mg/day
                'cholesterol_max': 200           # <200mg/day
            }
        }
    
    def calculate_bmr(self, weight_kg, height_cm, age_years, gender):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if gender.lower() == 'male':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
        else:  # female
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
        return bmr
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        multiplier = self.activity_multipliers.get(activity_level, 1.55)
        return bmr * multiplier
    
    def calculate_target_calories(self, tdee, weight_goal, health_conditions):
        """Calculate target calories based on weight goals and health conditions"""
        target_calories = tdee
        
        if weight_goal == 'Lose Weight':
            # Create moderate deficit for sustainable weight loss
            if 'Obesity' in health_conditions:
                target_calories = tdee - self.medical_guidelines['Obesity']['calorie_deficit']
            else:
                target_calories = tdee - 300  # Moderate deficit
        elif weight_goal == 'Gain Weight':
            target_calories = tdee + 300  # Moderate surplus
        # 'Maintain Weight' keeps target_calories = tdee
        
        return max(1200, target_calories)  # Ensure minimum safe calories
    
    def calculate_nutritional_targets(self, user_profile):
        """Calculate personalized nutritional targets based on medical guidelines"""
        # Extract user profile data
        weight_kg = user_profile['weight']
        height_cm = user_profile['height']
        age_years = user_profile['age']
        gender = user_profile['gender']
        activity_level = user_profile['activity_level']
        weight_goal = user_profile.get('weight_goal', 'Maintain Weight')
        health_conditions = user_profile.get('health_conditions', [])
        
        # Calculate BMR and TDEE
        bmr = self.calculate_bmr(weight_kg, height_cm, age_years, gender)
        tdee = self.calculate_tdee(bmr, activity_level)
        target_calories = self.calculate_target_calories(tdee, weight_goal, health_conditions)
        
        # Initialize targets with general healthy adult guidelines
        targets = {
            'calories': target_calories,
            'carbs_min': (target_calories * 0.45) / 4,      # 45% of calories
            'carbs_max': (target_calories * 0.60) / 4,      # 60% of calories
            'protein_min': (target_calories * 0.15) / 4,    # 15% of calories
            'protein_max': (target_calories * 0.25) / 4,    # 25% of calories
            'fat_min': (target_calories * 0.25) / 9,        # 25% of calories
            'fat_max': (target_calories * 0.35) / 9,        # 35% of calories
            'sugar_max': (target_calories * 0.10) / 4,      # 10% of calories
            'fiber_min': 25,                                # General recommendation
            'sodium_max': 2300,                             # General recommendation
            'saturated_fat_max': (target_calories * 0.10) / 9,  # 10% of calories
            'cholesterol_max': 300,                         # General recommendation
            'potassium_min': 3500,                          # General recommendation
            'calcium_min': 1000,                            # General recommendation
        }
        
        # Apply health condition modifications
        if health_conditions:
            targets = self._apply_health_condition_modifications(targets, health_conditions, target_calories, weight_kg)
        
        # Add meal-specific targets (assuming 3 main meals + 2 snacks)
        meal_targets = self._calculate_meal_targets(targets)
        
        return {
            'daily_targets': targets,
            'meal_targets': meal_targets,
            'bmr': bmr,
            'tdee': tdee,
            'calculations': {
                'bmr_formula': f"BMR = 10×{weight_kg} + 6.25×{height_cm} - 5×{age_years} {'+ 5' if gender.lower() == 'male' else '- 161'} = {bmr:.0f} kcal",
                'tdee_formula': f"TDEE = BMR × {self.activity_multipliers.get(activity_level, 1.55)} = {tdee:.0f} kcal",
                'target_formula': f"Target = TDEE {'+' if weight_goal == 'Gain Weight' else '-' if weight_goal == 'Lose Weight' else '='} {abs(target_calories - tdee):.0f} = {target_calories:.0f} kcal"
            }
        }
    
    def _apply_health_condition_modifications(self, targets, health_conditions, target_calories, weight_kg):
        """Apply medical guidelines for specific health conditions"""
        
        for condition in health_conditions:
            if condition in self.medical_guidelines:
                guidelines = self.medical_guidelines[condition]
                
                # Sugar modifications
                if 'sugar_max_percent' in guidelines:
                    sugar_from_percent = (target_calories * guidelines['sugar_max_percent'] / 100) / 4
                    sugar_from_grams = guidelines.get('sugar_max_grams', float('inf'))
                    targets['sugar_max'] = min(sugar_from_percent, sugar_from_grams)
                
                # Carbohydrate modifications
                if 'carb_percent' in guidelines:
                    carb_min, carb_max = guidelines['carb_percent']
                    targets['carbs_min'] = (target_calories * carb_min / 100) / 4
                    targets['carbs_max'] = (target_calories * carb_max / 100) / 4
                
                # Protein modifications
                if 'protein_percent' in guidelines:
                    protein_min, protein_max = guidelines['protein_percent']
                    targets['protein_min'] = (target_calories * protein_min / 100) / 4
                    targets['protein_max'] = (target_calories * protein_max / 100) / 4
                
                # Alternative protein calculation based on body weight
                if 'protein_grams_per_kg' in guidelines:
                    protein_min_kg, protein_max_kg = guidelines['protein_grams_per_kg']
                    protein_from_weight_min = weight_kg * protein_min_kg
                    protein_from_weight_max = weight_kg * protein_max_kg
                    # Use the higher value for protein targets
                    targets['protein_min'] = max(targets['protein_min'], protein_from_weight_min)
                    targets['protein_max'] = max(targets['protein_max'], protein_from_weight_max)
                
                # Fat modifications
                if 'fat_percent' in guidelines:
                    fat_min, fat_max = guidelines['fat_percent']
                    targets['fat_min'] = (target_calories * fat_min / 100) / 9
                    targets['fat_max'] = (target_calories * fat_max / 100) / 9
                
                # Saturated fat modifications
                if 'saturated_fat_percent' in guidelines:
                    targets['saturated_fat_max'] = (target_calories * guidelines['saturated_fat_percent'] / 100) / 9
                
                # Fiber modifications
                if 'fiber_min' in guidelines:
                    targets['fiber_min'] = max(targets['fiber_min'], guidelines['fiber_min'])
                
                # Sodium modifications
                if 'sodium_max' in guidelines:
                    targets['sodium_max'] = min(targets['sodium_max'], guidelines['sodium_max'])
                
                # Cholesterol modifications
                if 'cholesterol_max' in guidelines:
                    targets['cholesterol_max'] = min(targets['cholesterol_max'], guidelines['cholesterol_max'])
                
                # Other mineral modifications
                if 'potassium_min' in guidelines:
                    targets['potassium_min'] = max(targets.get('potassium_min', 0), guidelines['potassium_min'])
                
                if 'calcium_min' in guidelines:
                    targets['calcium_min'] = max(targets.get('calcium_min', 0), guidelines['calcium_min'])
        
        return targets
    
    def _calculate_meal_targets(self, daily_targets):
        """Calculate targets for individual meals"""
        # Typical meal distribution: Breakfast 25%, Lunch 30%, Dinner 30%, Snacks 15%
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.30,
            'dinner': 0.30,
            'snack': 0.075  # Per snack (2 snacks = 15%)
        }
        
        meal_targets = {}
        for meal_type, proportion in meal_distribution.items():
            meal_targets[meal_type] = {}
            for nutrient, value in daily_targets.items():
                if nutrient in ['calories', 'carbs_min', 'carbs_max', 'protein_min', 'protein_max', 
                               'fat_min', 'fat_max', 'sugar_max']:
                    meal_targets[meal_type][nutrient] = value * proportion
                elif nutrient in ['fiber_min', 'sodium_max', 'saturated_fat_max', 'cholesterol_max']:
                    meal_targets[meal_type][nutrient] = value * proportion
                else:
                    meal_targets[meal_type][nutrient] = value * proportion
        
        return meal_targets


class HealthAwareRandomForestRecommender:
    """Random Forest-based food recommender with comprehensive performance evaluation"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        self.evaluator = ModelPerformanceEvaluator()
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'model_performance': {},
            'comprehensive_metrics': {}
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
        self.train_models()
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
        
        # Classification labels (for classification models)
        self.food_data['Diabetes_Class'] = 0  # 0: suitable, 1: moderate, 2: not suitable
        self.food_data['Obesity_Class'] = 0
        self.food_data['Hypertension_Class'] = 0
        self.food_data['High_Cholesterol_Class'] = 0
        self.food_data['Overall_Health_Class'] = 0
        
        for idx, food in self.food_data.iterrows():
            # Calculate individual condition scores
            diabetes_score = self._calculate_diabetes_score(food)
            obesity_score = self._calculate_obesity_score(food)
            hypertension_score = self._calculate_hypertension_score(food)
            cholesterol_score = self._calculate_cholesterol_score(food)
            
            # Store scores
            self.food_data.at[idx, 'Diabetes_Score'] = diabetes_score
            self.food_data.at[idx, 'Obesity_Score'] = obesity_score
            self.food_data.at[idx, 'Hypertension_Score'] = hypertension_score
            self.food_data.at[idx, 'High_Cholesterol_Score'] = cholesterol_score
            
            # Calculate overall health score (lower is better)
            overall_score = (diabetes_score + obesity_score + hypertension_score + cholesterol_score) / 4
            self.food_data.at[idx, 'Overall_Health_Score'] = overall_score
            
            # Create classification labels based on scores
            self.food_data.at[idx, 'Diabetes_Class'] = min(2, int(diabetes_score))
            self.food_data.at[idx, 'Obesity_Class'] = min(2, int(obesity_score))
            self.food_data.at[idx, 'Hypertension_Class'] = min(2, int(hypertension_score))
            self.food_data.at[idx, 'High_Cholesterol_Class'] = min(2, int(cholesterol_score))
            self.food_data.at[idx, 'Overall_Health_Class'] = min(2, int(overall_score))
    
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
        
        # Add health score features
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features.extend(health_features)
        
        self.update_status(f"Prepared {len(self.features)} features for Random Forest model")
    
    def train_models(self):
        """Train Random Forest models with comprehensive evaluation"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for training")
            return
        
        try:
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Multiple target variables for different aspects
            regression_targets = {
                'Overall_Health': self.food_data['Overall_Health_Score'],
                'Diabetes_Suitability': self.food_data['Diabetes_Score'],
                'Obesity_Suitability': self.food_data['Obesity_Score'],
                'Hypertension_Suitability': self.food_data['Hypertension_Score'],
                'Cholesterol_Suitability': self.food_data['High_Cholesterol_Score']
            }
            
            classification_targets = {
                'Overall_Health_Class': self.food_data['Overall_Health_Class'],
                'Diabetes_Class': self.food_data['Diabetes_Class'],
                'Obesity_Class': self.food_data['Obesity_Class'],
                'Hypertension_Class': self.food_data['Hypertension_Class'],
                'Cholesterol_Class': self.food_data['High_Cholesterol_Class']
            }
            
            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for each target
            self.regression_models = {}
            self.classification_models = {}
            self.update_status("Training Random Forest models with comprehensive evaluation...")
            
            all_metrics = []
            
            # Train regression models
            for target_name, y in regression_targets.items():
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=None
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
                self.regression_models[target_name] = rf_model
                
                # Comprehensive evaluation
                metrics = self.evaluator.evaluate_regression_model(
                    rf_model, X_test, y_test, X_train, y_train, f"RF_Regression_{target_name}"
                )
                
                # Add cross-validation results
                cv_results = self.evaluator.perform_cross_validation(
                    rf_model, X_scaled, y, cv_folds=5, model_type='regression'
                )
                metrics.update(cv_results)
                
                # Add feature importance
                feature_importance = dict(zip(self.features, rf_model.feature_importances_))
                metrics['feature_importance'] = feature_importance
                
                all_metrics.append(metrics)
                
                self.update_status(f"Regression {target_name} - R²: {metrics['r2_score_test']:.3f}, RMSE: {metrics['rmse_test']:.3f}")
            
            # Train classification models
            for target_name, y in classification_targets.items():
                # Skip if not enough classes
                if len(np.unique(y)) < 2:
                    continue
                
                # Split data with stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train Random Forest Classifier
                rf_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                rf_classifier.fit(X_train, y_train)
                self.classification_models[target_name] = rf_classifier
                
                # Comprehensive evaluation
                metrics = self.evaluator.evaluate_classification_model(
                    rf_classifier, X_test, y_test, X_train, y_train, f"RF_Classification_{target_name}"
                )
                
                # Add cross-validation results
                cv_results = self.evaluator.perform_cross_validation(
                    rf_classifier, X_scaled, y, cv_folds=5, model_type='classification'
                )
                metrics.update(cv_results)
                
                # Add feature importance
                feature_importance = dict(zip(self.features, rf_classifier.feature_importances_))
                metrics['feature_importance'] = feature_importance
                
                all_metrics.append(metrics)
                
                self.update_status(f"Classification {target_name} - Accuracy: {metrics['accuracy_test']:.3f}, F1: {metrics['f1_test']:.3f}")
            
            # Store comprehensive metrics
            self.stats['comprehensive_metrics'] = all_metrics
            
            # Generate model comparison
            comparison = self.evaluator.compare_models(all_metrics)
            self.stats['model_comparison'] = comparison
            
            # Save metrics to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.evaluator.save_metrics_to_file({
                'all_metrics': all_metrics,
                'comparison': comparison,
                'dataset_info': {
                    'total_foods': len(self.food_data),
                    'features_used': self.features,
                    'categories': list(self.stats['categories'].keys())
                }
            }, f"rf_food_recommender_metrics_{timestamp}.json")
            
            if metrics_file:
                self.update_status(f"Metrics saved to: {metrics_file}")
            
            # Display summary
            self.display_performance_summary()
            
        except Exception as e:
            self.update_status(f"Error training models: {e}")
    
    def display_performance_summary(self):
        """Display a summary of model performance"""
        if 'comprehensive_metrics' not in self.stats or not self.stats['comprehensive_metrics']:
            return
        
        self.update_status("\n" + "="*60)
        self.update_status("COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
        self.update_status("="*60)
        
        # Regression models summary
        regression_metrics = [m for m in self.stats['comprehensive_metrics'] if m['model_type'] == 'regression']
        if regression_metrics:
            self.update_status("\nREGRESSION MODELS PERFORMANCE:")
            self.update_status("-" * 40)
            for metrics in regression_metrics:
                name = metrics['model_name'].replace('RF_Regression_', '')
                self.update_status(f"{name}:")
                self.update_status(f"  R² Score: {metrics['r2_score_test']:.4f}")
                self.update_status(f"  RMSE: {metrics['rmse_test']:.4f}")
                self.update_status(f"  MAE: {metrics['mae_test']:.4f}")
                if 'r2_mean' in metrics:
                    self.update_status(f"  CV R² Mean: {metrics['r2_mean']:.4f} (±{metrics['r2_std']:.4f})")
                self.update_status(f"  Overfitting: {metrics['overfitting_indicator']:.4f}")
        
        # Classification models summary
        classification_metrics = [m for m in self.stats['comprehensive_metrics'] if m['model_type'] == 'classification']
        if classification_metrics:
            self.update_status("\nCLASSIFICATION MODELS PERFORMANCE:")
            self.update_status("-" * 40)
            for metrics in classification_metrics:
                name = metrics['model_name'].replace('RF_Classification_', '')
                self.update_status(f"{name}:")
                self.update_status(f"  Accuracy: {metrics['accuracy_test']:.4f}")
                self.update_status(f"  Precision: {metrics['precision_test']:.4f}")
                self.update_status(f"  Recall: {metrics['recall_test']:.4f}")
                self.update_status(f"  F1 Score: {metrics['f1_test']:.4f}")
                if 'accuracy_mean' in metrics:
                    self.update_status(f"  CV Accuracy: {metrics['accuracy_mean']:.4f} (±{metrics['accuracy_std']:.4f})")
                self.update_status(f"  Overfitting: {metrics['overfitting_indicator']:.4f}")
        
        # Model comparison summary
        if 'model_comparison' in self.stats:
            comparison = self.stats['model_comparison']
            self.update_status("\nMODEL COMPARISON SUMMARY:")
            self.update_status("-" * 30)
            
            if 'regression_comparison' in comparison:
                reg_comp = comparison['regression_comparison']
                if 'ranking' in reg_comp and reg_comp['ranking']:
                    self.update_status("Best Regression Models:")
                    for i, model in enumerate(reg_comp['ranking'][:3], 1):
                        self.update_status(f"  {i}. {model['model']}: {model['overall_score']:.4f}")
            
            if 'classification_comparison' in comparison:
                class_comp = comparison['classification_comparison']
                if 'ranking' in class_comp and class_comp['ranking']:
                    self.update_status("Best Classification Models:")
                    for i, model in enumerate(class_comp['ranking'][:3], 1):
                        self.update_status(f"  {i}. {model['model']}: {model['overall_score']:.4f}")
        
        self.update_status("="*60)
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics for external analysis"""
        return {
            'comprehensive_metrics': self.stats.get('comprehensive_metrics', []),
            'model_comparison': self.stats.get('model_comparison', {}),
            'dataset_stats': {
                'total_items': self.stats['total_items'],
                'categories': self.stats['categories'],
                'loading_time': self.stats['loading_time']
            }
        }
    
    def evaluate_recommendations(self, recommendations, user_profile):
        """Evaluate the quality of generated recommendations"""
        if not recommendations:
            return {}
        
        recommendation_metrics = self.evaluator.evaluate_recommendation_quality(
            recommendations, user_profile
        )
        
        return recommendation_metrics
    
    def get_recommendations(self, user_profile, meal_type='meal', max_recommendations=10):
        """Get personalized food recommendations based on user profile"""
        if not hasattr(self, 'regression_models') or len(self.food_data) == 0:
            return []
        
        try:
            # Calculate nutritional targets using medical calculator
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            daily_targets = nutritional_data['daily_targets']
            meal_targets = nutritional_data['meal_targets'].get(meal_type.lower(), 
                                                              nutritional_data['meal_targets']['lunch'])
            
            # Get user's health conditions
            health_conditions = user_profile.get('health_conditions', [])
            
            # Create candidate pool with health condition filtering
            candidates = self.food_data.copy()
            
            # Apply category filter if specified
            category_filter = user_profile.get('category_filter', 'All')
            if category_filter != 'All':
                candidates = candidates[candidates['Category'] == category_filter]
            
            if len(candidates) == 0:
                return []
            
            # Calculate suitability scores for each candidate
            scores = []
            
            for idx, food in candidates.iterrows():
                # Calculate nutritional match score
                nutrition_score = self._calculate_nutrition_match_score(food, meal_targets)
                
                # Calculate health condition penalty
                health_penalty = self._calculate_health_penalty(food, health_conditions)
                
                # Combined score (lower is better)
                combined_score = nutrition_score + health_penalty
                
                # Create recommendation object
                recommendation = {
                    'food_id': idx,
                    'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
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
                    'nutrition_score': nutrition_score,
                    'health_penalty': health_penalty,
                    'combined_score': combined_score,
                    'suitable_for_conditions': self._check_condition_suitability(food, health_conditions),
                    'targets_met': self._check_targets_met(food, meal_targets),
                    'health_warnings': self._generate_health_warnings(food, health_conditions)
                }
                
                scores.append(recommendation)
            
            # Sort by combined score (lower is better)
            scores.sort(key=lambda x: x['combined_score'])
            
            # Return top recommendations
            recommendations = scores[:max_recommendations]
            
            # Add explanation data
            for rec in recommendations:
                rec['explanation'] = self._generate_explanation(rec, meal_targets, health_conditions)
                rec['nutritional_data'] = nutritional_data
            
            # Evaluate recommendation quality
            rec_quality = self.evaluate_recommendations(recommendations, user_profile)
            
            # Add quality metrics to the first recommendation for reference
            if recommendations:
                recommendations[0]['recommendation_quality'] = rec_quality
            
            return recommendations
            
        except Exception as e:
            self.update_status(f"Error getting recommendations: {e}")
            return []
    
    def _calculate_nutrition_match_score(self, food, targets):
        """Calculate how well a food matches nutritional targets"""
        score = 0
        
        # Calorie match (within reasonable range)
        calories = float(food.get('Energy(kcal) by calculation', 0))
        target_calories = targets.get('calories', 500)
        calorie_diff = abs(calories - target_calories) / target_calories
        score += calorie_diff * 2  # Weight calorie matching
        
        # Protein match
        protein = float(food.get('Protein(g)', 0))
        protein_target = (targets.get('protein_min', 0) + targets.get('protein_max', 50)) / 2
        if protein_target > 0:
            protein_diff = abs(protein - protein_target) / protein_target
            score += protein_diff
        
        # Carb match
        carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
        carbs_target = (targets.get('carbs_min', 0) + targets.get('carbs_max', 50)) / 2
        if carbs_target > 0:
            carbs_diff = abs(carbs - carbs_target) / carbs_target
            score += carbs_diff
        
        # Sugar penalty (should be low)
        sugar = float(food.get('SUGAR(g)', 0))
        sugar_max = targets.get('sugar_max', 10)
        if sugar > sugar_max:
            score += (sugar - sugar_max) / sugar_max
        
        # Fiber bonus (should be high)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        fiber_min = targets.get('fiber_min', 5)
        if fiber < fiber_min:
            score += (fiber_min - fiber) / fiber_min
        
        return score
    
    def _calculate_health_penalty(self, food, health_conditions):
        """Calculate penalty based on health conditions"""
        penalty = 0
        
        for condition in health_conditions:
            if condition == 'Diabetes':
                penalty += float(food.get('Diabetes_Score', 0))
            elif condition == 'Obesity':
                penalty += float(food.get('Obesity_Score', 0))
            elif condition == 'Hypertension':
                penalty += float(food.get('Hypertension_Score', 0))
            elif condition == 'High_Cholesterol':
                penalty += float(food.get('High_Cholesterol_Score', 0))
        
        return penalty
    
    def _check_condition_suitability(self, food, health_conditions):
        """Check which health conditions this food is suitable for"""
        suitable = []
        
        # Diabetes suitability
        if float(food.get('Diabetes_Score', 0)) <= 1.5 and float(food.get('SUGAR(g)', 0)) <= 8:
            suitable.append('Diabetes')
        
        # Obesity suitability
        if (float(food.get('Obesity_Score', 0)) <= 1.5 and 
            float(food.get('Energy(kcal) by calculation', 0)) <= 250):
            suitable.append('Obesity')
        
        # Hypertension suitability
        if (float(food.get('Hypertension_Score', 0)) <= 1.5 and 
            float(food.get('Na(mg)', 0)) <= 200):
            suitable.append('Hypertension')
        
        # High cholesterol suitability
        if (float(food.get('High_Cholesterol_Score', 0)) <= 1.5 and 
            float(food.get('FASAT (g) Saturated FA', 0)) <= 3):
            suitable.append('High_Cholesterol')
        
        return suitable
    
    def _check_targets_met(self, food, targets):
        """Check which nutritional targets are met by this food"""
        met = []
        
        # Check calorie range
        calories = float(food.get('Energy(kcal) by calculation', 0))
        target_calories = targets.get('calories', 500)
        if 0.7 * target_calories <= calories <= 1.3 * target_calories:
            met.append('Calories')
        
        # Check protein
        protein = float(food.get('Protein(g)', 0))
        if protein >= targets.get('protein_min', 0):
            met.append('Protein')
        
        # Check fiber
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= targets.get('fiber_min', 0):
            met.append('Fiber')
        
        # Check sugar limit
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar <= targets.get('sugar_max', 10):
            met.append('Sugar')
        
        return met
    
    def _generate_health_warnings(self, food, health_conditions):
        """Generate health warnings for specific conditions"""
        warnings = []
        
        for condition in health_conditions:
            if condition == 'Diabetes':
                if float(food.get('SUGAR(g)', 0)) > 10:
                    warnings.append("High sugar content - monitor blood glucose")
                if (float(food.get('CHOCDF (g) Carbohydrate', 0)) > 25 and 
                    float(food.get('FIBTG (g) Dietary fibre', 0)) < 3):
                    warnings.append("High carbs with low fiber - may spike blood sugar")
            
            elif condition == 'Hypertension':
                if float(food.get('Na(mg)', 0)) > 300:
                    warnings.append("High sodium content - may increase blood pressure")
            
            elif condition == 'High_Cholesterol':
                if float(food.get('FASAT (g) Saturated FA', 0)) > 4:
                    warnings.append("High saturated fat - may raise cholesterol")
                if float(food.get('CHOLE(mg) Cholesterol', 0)) > 100:
                    warnings.append("High cholesterol content")
        
        return warnings
    
    def _generate_explanation(self, recommendation, targets, health_conditions):
        """Generate explanation for why this food was recommended"""
        explanations = []
        
        # Nutritional match explanations
        if 'Calories' in recommendation['targets_met']:
            explanations.append(f"Good calorie match ({recommendation['calories']:.0f} kcal)")
        
        if 'Protein' in recommendation['targets_met']:
            explanations.append(f"Adequate protein ({recommendation['protein']:.1f}g)")
        
        if 'Fiber' in recommendation['targets_met']:
            explanations.append(f"Good fiber content ({recommendation['fiber']:.1f}g)")
        
        # Health condition explanations
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            conditions_text = ", ".join(suitable_conditions)
            explanations.append(f"Suitable for {conditions_text}")
        
        # Score explanation
        if recommendation['combined_score'] < 2:
            explanations.append("Excellent nutritional match")
        elif recommendation['combined_score'] < 4:
            explanations.append("Good nutritional match")
        else:
            explanations.append("Fair nutritional match")
        
        return " | ".join(explanations)
    
    def get_stats(self):
        """Get statistics about the recommendation system"""
        return self.stats
    
    def get_food_details(self, food_name):
        """Get detailed information about a specific food item"""
        if not hasattr(self, 'food_data') or self.food_data.empty:
            return {}
        
        food_items = self.food_data[
            (self.food_data['Thai_Name'] == food_name) | 
            (self.food_data['English_Name'] == food_name)
        ]
        
        if food_items.empty:
            return {}
        
        return food_items.iloc[0].to_dict()


# The rest of the UI code remains the same as in the previous version
# but with enhanced metrics display capabilities

class HealthDrivenFoodRecommenderUI:
    """Modern GUI for the health-driven food recommendation system with performance metrics display"""
    
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Health-Driven Food Recommendation System with Performance Metrics")
        self.master.geometry("1500x900")
        self.master.configure(bg="#f8f9fa")
        
        # Screen dimensions for responsive design
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        # Modern color scheme
        self.colors = {
            'primary': '#2c3e50',      # Dark blue-gray
            'secondary': '#3498db',     # Blue
            'success': '#27ae60',       # Green
            'warning': '#f39c12',       # Orange
            'danger': '#e74c3c',        # Red
            'light': '#ecf0f1',         # Light gray
            'white': '#ffffff',
            'background': '#f8f9fa',    # Very light gray
            'text': '#2c3e50',
            'text_light': '#7f8c8d'
        }
        
        # Configure styles
        self.setup_styles()
        
        # Initialize recommender
        self.recommender = recommender or HealthAwareRandomForestRecommender()
        
        # Create UI
        self.create_main_interface()
        
        # Initialize variables
        self.last_recommendations = []
        self.current_nutritional_data = None
        
        # Update stats
        self.update_stats_display()
    
    def setup_styles(self):
        """Setup modern UI styles"""
        self.style = ttk.Style()
        
        # Configure theme
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        # Font sizes
        self.fonts = {
            'heading': ('Segoe UI', 16, 'bold'),
            'subheading': ('Segoe UI', 12, 'bold'),
            'body': ('Segoe UI', 10),
            'caption': ('Segoe UI', 9)
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=self.fonts['heading'], 
                           foreground=self.colors['primary'])
        self.style.configure('Subtitle.TLabel', font=self.fonts['subheading'], 
                           foreground=self.colors['text'])
        self.style.configure('Primary.TButton', font=self.fonts['body'])
        self.style.configure('Success.TButton', font=self.fonts['body'])
        
        # Treeview styling
        self.style.configure('Health.Treeview', font=self.fonts['body'], rowheight=25)
        self.style.configure('Health.Treeview.Heading', font=self.fonts['subheading'])
    
    def create_main_interface(self):
        """Create the main interface with modern design"""
        # Main container with padding
        main_container = ttk.Frame(self.master, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        self.create_header(main_container)
        
        # Content area with panels
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Create panels layout
        self.create_input_panel(content_frame)
        self.create_results_panel(content_frame)
        self.create_charts_panel(main_container)
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self, parent):
        """Create header with title and main controls"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title and subtitle
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(title_frame, text="Health-Driven Food Recommendation System", 
                 style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Random Forest Model with Comprehensive Performance Metrics", 
                 style='Subtitle.TLabel', foreground=self.colors['text_light']).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self.get_recommendations, style='Primary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="View Metrics", 
                  command=self.show_performance_metrics, style='Success.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Calculate Nutrition", 
                  command=self.show_nutrition_calculation).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_form).pack(side=tk.RIGHT, padx=5)
    
    def show_performance_metrics(self):
        """Show comprehensive performance metrics in a new window"""
        metrics_window = tk.Toplevel(self.master)
        metrics_window.title("Model Performance Metrics")
        metrics_window.geometry("800x600")
        
        # Create notebook for different metric views
        notebook = ttk.Notebook(metrics_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get performance metrics
        performance_data = self.recommender.get_performance_metrics()
        
        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        overview_text = tk.Text(overview_frame, wrap=tk.WORD, font=self.fonts['body'])
        overview_scroll = ttk.Scrollbar(overview_frame, orient="vertical", command=overview_text.yview)
        overview_text.configure(yscrollcommand=overview_scroll.set)
        
        overview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        overview_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate overview
        self.populate_overview_metrics(overview_text, performance_data)
        
        # Detailed metrics tab
        detailed_frame = ttk.Frame(notebook)
        notebook.add(detailed_frame, text="Detailed Metrics")
        
        detailed_text = tk.Text(detailed_frame, wrap=tk.WORD, font=('Consolas', 9))
        detailed_scroll = ttk.Scrollbar(detailed_frame, orient="vertical", command=detailed_text.yview)
        detailed_text.configure(yscrollcommand=detailed_scroll.set)
        
        detailed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detailed_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate detailed metrics
        self.populate_detailed_metrics(detailed_text, performance_data)
        
        # Model comparison tab
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="Model Comparison")
        
        comparison_text = tk.Text(comparison_frame, wrap=tk.WORD, font=self.fonts['body'])
        comparison_scroll = ttk.Scrollbar(comparison_frame, orient="vertical", command=comparison_text.yview)
        comparison_text.configure(yscrollcommand=comparison_scroll.set)
        
        comparison_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        comparison_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate comparison
        self.populate_comparison_metrics(comparison_text, performance_data)
    
    def populate_overview_metrics(self, text_widget, performance_data):
        """Populate overview metrics"""
        text_widget.delete(1.0, tk.END)
        
        text_widget.insert(tk.END, "RANDOM FOREST FOOD RECOMMENDER PERFORMANCE OVERVIEW\n", "header")
        text_widget.insert(tk.END, "="*60 + "\n\n")
        
        # Dataset statistics
        dataset_stats = performance_data.get('dataset_stats', {})
        text_widget.insert(tk.END, "Dataset Information:\n", "subheader")
        text_widget.insert(tk.END, f"• Total Food Items: {dataset_stats.get('total_items', 0)}\n")
        text_widget.insert(tk.END, f"• Food Categories: {len(dataset_stats.get('categories', {}))}\n")
        text_widget.insert(tk.END, f"• Loading Time: {dataset_stats.get('loading_time', 0):.2f} seconds\n\n")
        
        # Model performance summary
        metrics = performance_data.get('comprehensive_metrics', [])
        
        if metrics:
            # Regression models summary
            regression_metrics = [m for m in metrics if m['model_type'] == 'regression']
            if regression_metrics:
                text_widget.insert(tk.END, "Regression Models Performance:\n", "subheader")
                for metric in regression_metrics:
                    name = metric['model_name'].replace('RF_Regression_', '')
                    text_widget.insert(tk.END, f"\n{name}:\n")
                    text_widget.insert(tk.END, f"  • R² Score: {metric['r2_score_test']:.4f}\n")
                    text_widget.insert(tk.END, f"  • RMSE: {metric['rmse_test']:.4f}\n")
                    text_widget.insert(tk.END, f"  • MAE: {metric['mae_test']:.4f}\n")
                    if 'r2_mean' in metric:
                        text_widget.insert(tk.END, f"  • Cross-Validation R²: {metric['r2_mean']:.4f} (±{metric['r2_std']:.4f})\n")
                
                text_widget.insert(tk.END, "\n")
            
            # Classification models summary
            classification_metrics = [m for m in metrics if m['model_type'] == 'classification']
            if classification_metrics:
                text_widget.insert(tk.END, "Classification Models Performance:\n", "subheader")
                for metric in classification_metrics:
                    name = metric['model_name'].replace('RF_Classification_', '')
                    text_widget.insert(tk.END, f"\n{name}:\n")
                    text_widget.insert(tk.END, f"  • Accuracy: {metric['accuracy_test']:.4f}\n")
                    text_widget.insert(tk.END, f"  • Precision: {metric['precision_test']:.4f}\n")
                    text_widget.insert(tk.END, f"  • Recall: {metric['recall_test']:.4f}\n")
                    text_widget.insert(tk.END, f"  • F1 Score: {metric['f1_test']:.4f}\n")
                    if 'accuracy_mean' in metric:
                        text_widget.insert(tk.END, f"  • Cross-Validation Accuracy: {metric['accuracy_mean']:.4f} (±{metric['accuracy_std']:.4f})\n")
        
        # Configure text tags
        text_widget.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
        text_widget.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
        
        text_widget.config(state=tk.DISABLED)
    
    def populate_detailed_metrics(self, text_widget, performance_data):
        """Populate detailed metrics in JSON format"""
        text_widget.delete(1.0, tk.END)
        
        # Convert metrics to formatted JSON
        metrics_json = json.dumps(performance_data, indent=2, default=str)
        text_widget.insert(tk.END, metrics_json)
        
        text_widget.config(state=tk.DISABLED)
    
    def populate_comparison_metrics(self, text_widget, performance_data):
        """Populate model comparison metrics"""
        text_widget.delete(1.0, tk.END)
        
        text_widget.insert(tk.END, "MODEL COMPARISON ANALYSIS\n", "header")
        text_widget.insert(tk.END, "="*40 + "\n\n")
        
        comparison = performance_data.get('model_comparison', {})
        
        if 'regression_comparison' in comparison:
            reg_comp = comparison['regression_comparison']
            text_widget.insert(tk.END, "Regression Models Ranking:\n", "subheader")
            
            if 'ranking' in reg_comp:
                for i, model in enumerate(reg_comp['ranking'], 1):
                    text_widget.insert(tk.END, f"{i}. {model['model']}\n")
                    text_widget.insert(tk.END, f"   Overall Score: {model['overall_score']:.4f}\n")
                    text_widget.insert(tk.END, f"   R² Component: {model['r2']:.4f}\n")
                    text_widget.insert(tk.END, f"   RMSE Component: {model['rmse_inv']:.4f}\n")
                    text_widget.insert(tk.END, f"   MAE Component: {model['mae_inv']:.4f}\n\n")
            
            text_widget.insert(tk.END, "Best Models by Metric:\n", "subheader")
            if 'best_r2' in reg_comp:
                text_widget.insert(tk.END, f"• Best R²: {reg_comp['best_r2']['model']} ({reg_comp['best_r2']['score']:.4f})\n")
            if 'best_rmse' in reg_comp:
                text_widget.insert(tk.END, f"• Best RMSE: {reg_comp['best_rmse']['model']} ({reg_comp['best_rmse']['score']:.4f})\n")
            if 'best_mae' in reg_comp:
                text_widget.insert(tk.END, f"• Best MAE: {reg_comp['best_mae']['model']} ({reg_comp['best_mae']['score']:.4f})\n\n")
        
        if 'classification_comparison' in comparison:
            class_comp = comparison['classification_comparison']
            text_widget.insert(tk.END, "Classification Models Ranking:\n", "subheader")
            
            if 'ranking' in class_comp:
                for i, model in enumerate(class_comp['ranking'], 1):
                    text_widget.insert(tk.END, f"{i}. {model['model']}\n")
                    text_widget.insert(tk.END, f"   Overall Score: {model['overall_score']:.4f}\n")
                    text_widget.insert(tk.END, f"   Accuracy: {model['accuracy']:.4f}\n")
                    text_widget.insert(tk.END, f"   F1 Score: {model['f1']:.4f}\n")
                    text_widget.insert(tk.END, f"   Precision: {model['precision']:.4f}\n")
                    text_widget.insert(tk.END, f"   Recall: {model['recall']:.4f}\n\n")
            
            text_widget.insert(tk.END, "Best Models by Metric:\n", "subheader")
            if 'best_accuracy' in class_comp:
                text_widget.insert(tk.END, f"• Best Accuracy: {class_comp['best_accuracy']['model']} ({class_comp['best_accuracy']['score']:.4f})\n")
            if 'best_f1' in class_comp:
                text_widget.insert(tk.END, f"• Best F1: {class_comp['best_f1']['model']} ({class_comp['best_f1']['score']:.4f})\n")
            if 'best_precision' in class_comp:
                text_widget.insert(tk.END, f"• Best Precision: {class_comp['best_precision']['model']} ({class_comp['best_precision']['score']:.4f})\n")
            if 'best_recall' in class_comp:
                text_widget.insert(tk.END, f"• Best Recall: {class_comp['best_recall']['model']} ({class_comp['best_recall']['score']:.4f})\n")
        
        # Configure text tags
        text_widget.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
        text_widget.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
        
        text_widget.config(state=tk.DISABLED)
    
    # ... (rest of the UI methods remain the same as in the previous version)
    
    def update_stats_display(self):
        """Update statistics display with performance metrics"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Create stats labels
            total_items = stats['total_items']
            num_categories = len(stats['categories'])
            loading_time = stats['loading_time']
            
            stats_text = f"Dataset: {total_items} foods | Categories: {num_categories} | Load time: {loading_time:.1f}s"
            
            # Add model performance summary
            if 'comprehensive_metrics' in stats and stats['comprehensive_metrics']:
                # Get best performing models
                regression_metrics = [m for m in stats['comprehensive_metrics'] if m['model_type'] == 'regression']
                classification_metrics = [m for m in stats['comprehensive_metrics'] if m['model_type'] == 'classification']
                
                if regression_metrics:
                    best_r2 = max(regression_metrics, key=lambda x: x.get('r2_score_test', 0))
                    stats_text += f" | Best R²: {best_r2['r2_score_test']:.3f}"
                
                if classification_metrics:
                    best_acc = max(classification_metrics, key=lambda x: x.get('accuracy_test', 0))
                    stats_text += f" | Best Accuracy: {best_acc['accuracy_test']:.3f}"
            
            ttk.Label(self.stats_frame, text=stats_text, font=self.fonts['caption'],
                     foreground=self.colors['text_light']).pack()
            
        except Exception as e:
            print(f"Error updating stats: {e}")


def main():
    """Main function to run the application"""
    print("Starting Health-Driven Food Recommendation System with Performance Metrics...")
    
    # Create main window
    root = tk.Tk()
    
    # Create splash screen
    splash = tk.Toplevel(root)
    splash.title("Loading...")
    splash.geometry("600x350")
    splash.resizable(False, False)
    
    # Center splash screen
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() // 2) - (splash.winfo_width() // 2)
    y = (splash.winfo_screenheight() // 2) - (splash.winfo_height() // 2)
    splash.geometry(f"+{x}+{y}")
    
    # Splash content
    splash_frame = ttk.Frame(splash, padding="50")
    splash_frame.pack(fill=tk.BOTH, expand=True)
    
    ttk.Label(splash_frame, text="Health-Driven Food Recommendation System", 
             font=('Segoe UI', 16, 'bold')).pack(pady=20)
    ttk.Label(splash_frame, text="Random Forest with Comprehensive Performance Metrics", 
             font=('Segoe UI', 12)).pack()
    ttk.Label(splash_frame, text="Accuracy • F1 Score • Recall • Precision • R² • RMSE • MAE", 
             font=('Segoe UI', 10)).pack(pady=5)
    ttk.Label(splash_frame, text="Prince of Songkla University", 
             font=('Segoe UI', 10)).pack(pady=10)
    
    # Progress bar
    progress = ttk.Progressbar(splash_frame, mode='indeterminate')
    progress.pack(fill=tk.X, pady=20)
    progress.start()
    
    status_label = ttk.Label(splash_frame, text="Initializing system...")
    status_label.pack()
    
    # Hide main window initially
    root.withdraw()
    
    def update_splash_status(message):
        status_label.config(text=message)
        splash.update()
    
    def initialize_system():
        try:
            # Initialize recommender system
            update_splash_status("Loading Thai food database...")
            recommender = HealthAwareRandomForestRecommender(update_splash_status)
            
            update_splash_status("Training Random Forest models...")
            time.sleep(0.5)  # Brief pause for visual effect
            
            update_splash_status("Calculating performance metrics...")
            time.sleep(0.5)
            
            update_splash_status("Building user interface...")
            
            # Create main UI
            app = HealthDrivenFoodRecommenderUI(root, recommender)
            
            # Show main window
            root.deiconify()
            root.lift()
            root.focus_force()
            
            # Close splash
            splash.destroy()
            
            print("System initialized successfully with comprehensive performance metrics!")
            
        except Exception as e:
            splash.destroy()
            messagebox.showerror("Initialization Error", f"Failed to initialize system: {str(e)}")
            root.quit()
    
    # Schedule initialization
    root.after(100, initialize_system)
    
    # Configure main window
    root.title("Health-Driven Food Recommendation System with Performance Metrics")
    root.geometry("1500x900")
    
    # Center main window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start application
    root.mainloop()


if __name__ == "__main__":
    main()