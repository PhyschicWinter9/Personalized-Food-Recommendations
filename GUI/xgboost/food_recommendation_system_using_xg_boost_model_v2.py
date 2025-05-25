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
import seaborn as sns

# XGBoost and ML imports
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_curve, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

# Hyperparameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Using default hyperparameters.")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Model interpretability features disabled.")

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class MedicalNutritionCalculator:
    """Enhanced medical-grade nutritional calculation engine"""
    
    def __init__(self):
        # Activity level multipliers for TDEE calculation
        self.activity_multipliers = {
            'Sedentary': 1.2,
            'Light': 1.375,
            'Moderate': 1.55,
            'Active': 1.725,
            'Very Active': 1.9
        }
        
        # Comprehensive medical guidelines based on research analysis
        self.medical_guidelines = {
            'Diabetes': {
                'sugar_max_percent': 5,          # <5% of total calories
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (40, 50),        # 40-50% of calories
                'protein_percent': (15, 25),     # 15-25% of calories
                'fat_percent': (25, 35),         # 25-35% of calories
                'saturated_fat_percent': 7,      # <7% of calories
                'fiber_min': 25,                 # Minimum 25g/day
                'sodium_max': 2300,              # <2300mg/day
                'cholesterol_max': 200,          # <200mg/day
                'glycemic_index_preference': 'low',  # Prefer low GI foods
                'meal_frequency': 'regular'      # Regular meal timing important
            },
            'Obesity': {
                'sugar_max_percent': 5,          # <5% of total calories
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (40, 50),        # Lower carb for weight loss
                'protein_percent': (20, 35),     # Higher protein for satiety
                'protein_grams_per_kg': (1.2, 1.6),  # 1.2-1.6g per kg
                'fat_percent': (20, 30),         # 20-30% of calories
                'fiber_min': 30,                 # Higher fiber for satiety
                'sodium_max': 2300,              # <2300mg/day
                'calorie_deficit': 500,          # 500 kcal deficit
                'energy_density_max': 1.5       # kcal/g maximum
            },
            'Hypertension': {
                'sugar_max_percent': 10,         # <10% of total calories
                'carb_percent': (45, 65),        # DASH diet pattern
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories
                'fiber_min': 30,                 # DASH recommendation
                'sodium_max': 1500,              # <1500mg/day ideal
                'potassium_min': 4700,           # High potassium
                'calcium_min': 1200,             # Adequate calcium
                'magnesium_min': 400             # Adequate magnesium
            },
            'High_Cholesterol': {
                'sugar_max_percent': 10,         # <10% of total calories
                'carb_percent': (45, 65),        # Complex carbs
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories
                'trans_fat_max': 0,              # Eliminate trans fats
                'fiber_min': 25,                 # Total fiber
                'soluble_fiber_min': 10,         # Soluble fiber important
                'sodium_max': 2300,              # <2300mg/day
                'cholesterol_max': 200,          # <200mg/day
                'plant_sterol_min': 2            # 2g plant sterols/day
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
    
    def calculate_target_calories(self, tdee, weight_goal, health_conditions):
        """Calculate target calories based on goals and conditions"""
        target_calories = tdee
        
        if weight_goal == 'Lose Weight':
            if 'Obesity' in health_conditions:
                target_calories = tdee - self.medical_guidelines['Obesity']['calorie_deficit']
            else:
                target_calories = tdee - 300
        elif weight_goal == 'Gain Weight':
            target_calories = tdee + 300
        
        return max(1200, target_calories)
    
    def calculate_nutritional_targets(self, user_profile):
        """Calculate comprehensive nutritional targets"""
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
            'saturated_fat_max': (target_calories * 0.10) / 9,
            'cholesterol_max': 300,
            'potassium_min': 3500
        }
        
        # Apply health condition modifications
        if health_conditions:
            targets = self._apply_health_condition_modifications(
                targets, health_conditions, target_calories, weight_kg
            )
        
        # Calculate meal targets
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
        """Apply medical guidelines for specific conditions"""
        for condition in health_conditions:
            if condition in self.medical_guidelines:
                guidelines = self.medical_guidelines[condition]
                
                # Apply all guideline modifications
                for key, value in guidelines.items():
                    if key == 'sugar_max_percent':
                        sugar_from_percent = (target_calories * value / 100) / 4
                        sugar_from_grams = guidelines.get('sugar_max_grams', float('inf'))
                        targets['sugar_max'] = min(sugar_from_percent, sugar_from_grams)
                    
                    elif key == 'carb_percent':
                        carb_min, carb_max = value
                        targets['carbs_min'] = (target_calories * carb_min / 100) / 4
                        targets['carbs_max'] = (target_calories * carb_max / 100) / 4
                    
                    elif key == 'protein_percent':
                        protein_min, protein_max = value
                        targets['protein_min'] = (target_calories * protein_min / 100) / 4
                        targets['protein_max'] = (target_calories * protein_max / 100) / 4
                    
                    elif key == 'protein_grams_per_kg':
                        protein_min_kg, protein_max_kg = value
                        targets['protein_min'] = max(targets['protein_min'], weight_kg * protein_min_kg)
                        targets['protein_max'] = max(targets['protein_max'], weight_kg * protein_max_kg)
                    
                    elif key == 'fat_percent':
                        fat_min, fat_max = value
                        targets['fat_min'] = (target_calories * fat_min / 100) / 9
                        targets['fat_max'] = (target_calories * fat_max / 100) / 9
                    
                    elif key == 'fiber_min':
                        targets['fiber_min'] = max(targets['fiber_min'], value)
                    
                    elif key == 'sodium_max':
                        targets['sodium_max'] = min(targets['sodium_max'], value)
                    
                    elif key == 'cholesterol_max':
                        targets['cholesterol_max'] = min(targets['cholesterol_max'], value)
        
        return targets
    
    def _calculate_meal_targets(self, daily_targets):
        """Calculate meal-specific targets"""
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.30,
            'dinner': 0.30,
            'snack': 0.075
        }
        
        meal_targets = {}
        for meal_type, proportion in meal_distribution.items():
            meal_targets[meal_type] = {
                nutrient: value * proportion 
                for nutrient, value in daily_targets.items()
            }
        
        return meal_targets


class AdvancedFeatureEngineer:
    """Advanced feature engineering specifically for XGBoost"""
    
    def __init__(self):
        self.feature_interactions = [
            ('Protein(g)', 'FIBTG (g) Dietary fibre'),
            ('SUGAR(g)', 'Energy(kcal) by calculation'),
            ('Fat(g)', 'CHOCDF (g) Carbohydrate'),
            ('Na(mg)', 'K(mg)'),
            ('CHOLE(mg) Cholesterol', 'FASAT (g) Saturated FA')
        ]
        
        self.health_ratios = {
            'protein_to_calorie_ratio': ('Protein(g)', 'Energy(kcal) by calculation'),
            'fiber_to_carb_ratio': ('FIBTG (g) Dietary fibre', 'CHOCDF (g) Carbohydrate'),
            'sodium_potassium_ratio': ('Na(mg)', 'K(mg)'),
            'sat_fat_to_total_fat_ratio': ('FASAT (g) Saturated FA', 'Fat(g)'),
            'sugar_to_carb_ratio': ('SUGAR(g)', 'CHOCDF (g) Carbohydrate')
        }
    
    def create_interaction_features(self, df):
        """Create interaction features for XGBoost"""
        df_enhanced = df.copy()
        
        # Multiplicative interactions
        for feat1, feat2 in self.feature_interactions:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}".replace('(', '').replace(')', '').replace(' ', '_')
                df_enhanced[interaction_name] = df[feat1] * df[feat2]
        
        # Health ratios
        for ratio_name, (numerator, denominator) in self.health_ratios.items():
            if numerator in df.columns and denominator in df.columns:
                # Add small epsilon to avoid division by zero
                df_enhanced[ratio_name] = df[numerator] / (df[denominator] + 1e-8)
        
        return df_enhanced
    
    def create_composite_health_scores(self, df):
        """Create composite health suitability scores"""
        df_enhanced = df.copy()
        
        # Diabetes composite score
        diabetes_features = ['SUGAR(g)', 'CHOCDF (g) Carbohydrate', 'FIBTG (g) Dietary fibre']
        if all(feat in df.columns for feat in diabetes_features):
            # Lower sugar and carbs, higher fiber = better for diabetes
            sugar_norm = df['SUGAR(g)'] / (df['SUGAR(g)'].max() + 1e-8)
            carb_norm = df['CHOCDF (g) Carbohydrate'] / (df['CHOCDF (g) Carbohydrate'].max() + 1e-8)
            fiber_norm = df['FIBTG (g) Dietary fibre'] / (df['FIBTG (g) Dietary fibre'].max() + 1e-8)
            
            df_enhanced['diabetes_composite_score'] = (
                (1 - sugar_norm) * 0.4 + 
                (1 - carb_norm) * 0.3 + 
                fiber_norm * 0.3
            )
        
        # Cardiovascular composite score
        cardio_features = ['Na(mg)', 'FASAT (g) Saturated FA', 'FIBTG (g) Dietary fibre', 'K(mg)']
        if all(feat in df.columns for feat in cardio_features):
            sodium_norm = df['Na(mg)'] / (df['Na(mg)'].max() + 1e-8)
            sat_fat_norm = df['FASAT (g) Saturated FA'] / (df['FASAT (g) Saturated FA'].max() + 1e-8)
            fiber_norm = df['FIBTG (g) Dietary fibre'] / (df['FIBTG (g) Dietary fibre'].max() + 1e-8)
            potassium_norm = df['K(mg)'] / (df['K(mg)'].max() + 1e-8)
            
            df_enhanced['cardiovascular_composite_score'] = (
                (1 - sodium_norm) * 0.3 + 
                (1 - sat_fat_norm) * 0.3 + 
                fiber_norm * 0.2 + 
                potassium_norm * 0.2
            )
        
        # Weight management composite score
        weight_features = ['Energy(kcal) by calculation', 'Fat(g)', 'Protein(g)', 'FIBTG (g) Dietary fibre']
        if all(feat in df.columns for feat in weight_features):
            calorie_norm = df['Energy(kcal) by calculation'] / (df['Energy(kcal) by calculation'].max() + 1e-8)
            fat_norm = df['Fat(g)'] / (df['Fat(g)'].max() + 1e-8)
            protein_norm = df['Protein(g)'] / (df['Protein(g)'].max() + 1e-8)
            fiber_norm = df['FIBTG (g) Dietary fibre'] / (df['FIBTG (g) Dietary fibre'].max() + 1e-8)
            
            df_enhanced['weight_management_score'] = (
                (1 - calorie_norm) * 0.3 + 
                (1 - fat_norm) * 0.2 + 
                protein_norm * 0.25 + 
                fiber_norm * 0.25
            )
        
        return df_enhanced
    
    def create_nutritional_density_features(self, df):
        """Create nutritional density features"""
        df_enhanced = df.copy()
        
        if 'Energy(kcal) by calculation' in df.columns:
            calorie_base = df['Energy(kcal) by calculation'] + 1e-8
            
            # Nutrient density per calorie
            density_nutrients = ['Protein(g)', 'FIBTG (g) Dietary fibre', 'K(mg)', 'Ca(mg)']
            for nutrient in density_nutrients:
                if nutrient in df.columns:
                    density_name = f"{nutrient}_per_calorie".replace('(', '').replace(')', '').replace(' ', '_')
                    df_enhanced[density_name] = df[nutrient] / calorie_base
        
        return df_enhanced


class HealthAwareXGBoostRecommender:
    """XGBoost-based food recommender with advanced features"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.shap_explainers = {}
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'model_performance': {},
            'hyperparameter_tuning': {},
            'feature_importance': {}
        }
        
        # Base nutritional features
        self.base_nutritional_features = [
            'Energy(kcal) by calculation', 
            'Protein(g)', 
            'CHOCDF (g) Carbohydrate',
            'SUGAR(g)', 
            'FIBTG (g) Dietary fibre', 
            'Fat(g)',
            'Na(mg)',
            'K(mg)',
            'Ca(mg)',
            'CHOLE(mg) Cholesterol',
            'FASAT (g) Saturated FA'
        ]
        
        # Initialize system
        start_time = time.time()
        self.load_data()
        self.prepare_features()
        self.train_models()
        self.stats['loading_time'] = time.time() - start_time
    
    def update_status(self, message):
        """Update loading status"""
        try:
            if self.status_callback:
                self.status_callback(message)
            print(message)
        except Exception as e:
            print(f"Status: {message}")
    
    def load_data(self):
        """Load and combine food datasets"""
        try:
            dataset_folder = './datasets'
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                self.update_status("No CSV files found in datasets folder!")
                self.food_data = pd.DataFrame()
                return
            
            dataframes = []
            category_map = {
                'drinking': 'Beverages',
                'esan_food': 'Esan Food',
                'fruit': 'Fruits',
                'meat': 'Meat & Poultry',
                'noodle': 'Noodles',
                'onedish': 'One Dish Meals',
                'processed_food': 'Processed Food',
                'vegetables': 'Vegetables',
                'cracker': 'Snacks & Crackers',
                'curry': 'Curries',
                'dessert': 'Desserts'
            }
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    base_name = os.path.splitext(filename)[0].lower()
                    category = category_map.get(base_name, 
                        os.path.splitext(filename)[0].replace('_', ' ').title())
                    
                    self.update_status(f"Loading {category} data...")
                    
                    df = pd.read_csv(file_path)
                    df['Category'] = category
                    
                    # Clean data
                    df = self._clean_nutritional_data(df)
                    
                    if len(df) > 0:
                        dataframes.append(df)
                        self.update_status(f"Loaded {len(df)} items from {filename}")
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.update_status("Combining all food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
                
                self.update_status(f"Successfully loaded {len(self.food_data)} food items")
                self._calculate_health_scores()
                
            else:
                self.food_data = pd.DataFrame()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            self.food_data = pd.DataFrame()
    
    def _clean_nutritional_data(self, df):
        """Clean and standardize nutritional data"""
        nutritional_columns = self.base_nutritional_features
        
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
        self.update_status("Calculating comprehensive health scores...")
        
        # Initialize score columns
        health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        for condition in health_conditions:
            self.food_data[f'{condition}_Score'] = 0
        
        self.food_data['Overall_Health_Score'] = 0
        
        for idx, food in self.food_data.iterrows():
            scores = {}
            
            # Calculate individual condition scores
            scores['Diabetes'] = self._calculate_diabetes_score(food)
            scores['Obesity'] = self._calculate_obesity_score(food)
            scores['Hypertension'] = self._calculate_hypertension_score(food)
            scores['High_Cholesterol'] = self._calculate_cholesterol_score(food)
            
            # Store scores
            for condition, score in scores.items():
                self.food_data.at[idx, f'{condition}_Score'] = score
            
            # Overall health score
            overall_score = sum(scores.values()) / len(scores)
            self.food_data.at[idx, 'Overall_Health_Score'] = overall_score
    
    def _calculate_diabetes_score(self, food):
        """Calculate diabetes suitability score (lower is better)"""
        score = 0
        
        # Sugar penalty (major factor)
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar > 15:
            score += 4
        elif sugar > 8:
            score += 2.5
        elif sugar > 3:
            score += 1
        
        # Glycemic load estimation (carbs with low fiber)
        carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        
        if carbs > 0:
            glycemic_load = carbs * max(0, 1 - fiber/carbs)
            if glycemic_load > 20:
                score += 3
            elif glycemic_load > 10:
                score += 1.5
        
        # Fiber bonus (significant for diabetes)
        if fiber >= 5:
            score -= 1.5
        elif fiber >= 3:
            score -= 0.8
        
        return max(0, score)
    
    def _calculate_obesity_score(self, food):
        """Calculate obesity suitability score (lower is better)"""
        score = 0
        
        # Energy density penalty
        calories = float(food.get('Energy(kcal) by calculation', 0))
        if calories > 400:
            score += 4
        elif calories > 250:
            score += 2.5
        elif calories > 150:
            score += 1
        
        # Fat content consideration
        fat = float(food.get('Fat(g)', 0))
        if fat > 20:
            score += 3
        elif fat > 10:
            score += 1.5
        
        # Sugar penalty for weight management
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar > 15:
            score += 2.5
        elif sugar > 8:
            score += 1.2
        
        # Protein bonus (satiety)
        protein = float(food.get('Protein(g)', 0))
        if protein >= 15:
            score -= 1.5
        elif protein >= 8:
            score -= 0.8
        
        # Fiber bonus (satiety and calorie displacement)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 8:
            score -= 1.5
        elif fiber >= 4:
            score -= 0.8
        
        return max(0, score)
    
    def _calculate_hypertension_score(self, food):
        """Calculate hypertension suitability score (lower is better)"""
        score = 0
        
        # Sodium penalty (critical for hypertension)
        sodium = float(food.get('Na(mg)', 0))
        if sodium > 600:
            score += 4
        elif sodium > 300:
            score += 2.5
        elif sodium > 140:
            score += 1
        
        # Potassium bonus (important for BP control)
        potassium = float(food.get('K(mg)', 0))
        if potassium > 400:
            score -= 1.5
        elif potassium > 200:
            score -= 0.8
        
        # Fiber bonus (DASH diet)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 5:
            score -= 1
        
        return max(0, score)
    
    def _calculate_cholesterol_score(self, food):
        """Calculate high cholesterol suitability score (lower is better)"""
        score = 0
        
        # Saturated fat penalty (major factor)
        sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
        if sat_fat > 6:
            score += 4
        elif sat_fat > 3:
            score += 2
        elif sat_fat > 1:
            score += 0.5
        
        # Cholesterol penalty
        cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
        if cholesterol > 150:
            score += 3
        elif cholesterol > 50:
            score += 1.5
        
        # Soluble fiber bonus (cholesterol lowering)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 6:
            score -= 1.5
        elif fiber >= 3:
            score -= 0.8
        
        return max(0, score)
    
    def prepare_features(self):
        """Prepare enhanced features for XGBoost"""
        if len(self.food_data) == 0:
            self.update_status("Error: No data available for feature preparation")
            return
        
        self.update_status("Engineering advanced features for XGBoost...")
        
        # Start with base features
        available_features = []
        for feature in self.base_nutritional_features:
            if feature in self.food_data.columns:
                available_features.append(feature)
        
        # Create interaction features
        self.food_data = self.feature_engineer.create_interaction_features(self.food_data)
        
        # Create composite health scores
        self.food_data = self.feature_engineer.create_composite_health_scores(self.food_data)
        
        # Create nutritional density features
        self.food_data = self.feature_engineer.create_nutritional_density_features(self.food_data)
        
        # Add health score features
        health_features = [
            'Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 
            'High_Cholesterol_Score', 'Overall_Health_Score'
        ]
        
        # Composite features
        composite_features = [
            'diabetes_composite_score', 'cardiovascular_composite_score', 
            'weight_management_score'
        ]
        
        # Interaction features (dynamically generated)
        interaction_features = [col for col in self.food_data.columns if '_x_' in col]
        
        # Ratio features
        ratio_features = [col for col in self.food_data.columns if '_ratio' in col]
        
        # Density features
        density_features = [col for col in self.food_data.columns if '_per_calorie' in col]
        
        # Combine all features
        self.features = (available_features + health_features + composite_features + 
                        interaction_features + ratio_features + density_features)
        
        # Remove any features with NaN or infinite values
        self.features = [feat for feat in self.features if feat in self.food_data.columns]
        feature_data = self.food_data[self.features]
        
        # Replace infinite values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.fillna(0)
        self.food_data[self.features] = feature_data
        
        self.update_status(f"Prepared {len(self.features)} enhanced features for XGBoost")
        self.update_status(f"Feature types: Base({len(available_features)}), Health({len(health_features)}), "
                          f"Composite({len(composite_features)}), Interactions({len(interaction_features)}), "
                          f"Ratios({len(ratio_features)}), Density({len(density_features)})")
    
    def train_models(self):
        """Train XGBoost models with hyperparameter optimization"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for training")
            return
        
        try:
            self.update_status("Preparing data for XGBoost training...")
            
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Multiple target variables
            targets = {
                'Overall_Health': self.food_data['Overall_Health_Score'],
                'Diabetes_Suitability': self.food_data['Diabetes_Score'],
                'Obesity_Suitability': self.food_data['Obesity_Score'],
                'Hypertension_Suitability': self.food_data['Hypertension_Score'],
                'Cholesterol_Suitability': self.food_data['High_Cholesterol_Score']
            }
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.features)
            
            # Train models for each target
            for target_name, y in targets.items():
                self.update_status(f"Training XGBoost model for {target_name}...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=None
                )
                
                # Hyperparameter optimization
                best_params = self._optimize_hyperparameters(X_train, y_train, target_name)
                
                # Train final model with best parameters
                xgb_model = xgb.XGBRegressor(
                    **best_params,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Use early stopping
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Evaluate model
                y_pred = xgb_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=5, scoring='r2')
                
                self.models[target_name] = xgb_model
                
                # Store performance metrics
                self.stats['model_performance'][target_name] = {
                    'r2_score': r2,
                    'mse': mse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'best_params': best_params
                }
                
                # Feature importance
                feature_importance = dict(zip(self.features, xgb_model.feature_importances_))
                self.feature_importances[target_name] = feature_importance
                
                # SHAP explainer
                if SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(xgb_model)
                        # Use subset for SHAP to avoid memory issues
                        shap_sample = X_train.sample(min(100, len(X_train)))
                        shap_values = explainer.shap_values(shap_sample)
                        self.shap_explainers[target_name] = {
                            'explainer': explainer,
                            'sample_data': shap_sample,
                            'shap_values': shap_values
                        }
                    except Exception as e:
                        self.update_status(f"SHAP initialization failed for {target_name}: {e}")
                
                self.update_status(f"{target_name} model - R²: {r2:.3f}, CV: {cv_scores.mean():.3f}")
            
            # Overall feature importance analysis
            self._analyze_feature_importance()
            
        except Exception as e:
            self.update_status(f"Error training XGBoost models: {e}")
    
    def _optimize_hyperparameters(self, X_train, y_train, target_name):
        """Optimize XGBoost hyperparameters"""
        
        if HYPEROPT_AVAILABLE and len(X_train) > 100:
            self.update_status(f"Optimizing hyperparameters for {target_name}...")
            
            # Define search space
            dimensions = [
                Real(0.01, 0.3, name='learning_rate'),
                Integer(3, 10, name='max_depth'),
                Integer(50, 300, name='n_estimators'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0, 10, name='reg_alpha'),
                Real(0, 10, name='reg_lambda')
            ]
            
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                model = xgb.XGBRegressor(
                    **params,
                    random_state=42,
                    n_jobs=1  # Use single job for optimization
                )
                
                # Cross-validation score
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
                return -scores.mean()
            
            # Optimize
            try:
                result = gp_minimize(objective, dimensions, n_calls=20, random_state=42)
                
                best_params = {
                    'learning_rate': result.x[0],
                    'max_depth': result.x[1],
                    'n_estimators': result.x[2],
                    'subsample': result.x[3],
                    'colsample_bytree': result.x[4],
                    'reg_alpha': result.x[5],
                    'reg_lambda': result.x[6]
                }
                
                self.stats['hyperparameter_tuning'][target_name] = {
                    'best_score': -result.fun,
                    'best_params': best_params,
                    'n_calls': 20
                }
                
                return best_params
                
            except Exception as e:
                self.update_status(f"Hyperparameter optimization failed for {target_name}: {e}")
        
        # Default parameters if optimization fails or not available
        return {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1
        }
    
    def _analyze_feature_importance(self):
        """Analyze feature importance across all models"""
        if not self.feature_importances:
            return
        
        # Aggregate feature importance across models
        aggregated_importance = {}
        for feature in self.features:
            importance_values = []
            for model_name, importance_dict in self.feature_importances.items():
                importance_values.append(importance_dict.get(feature, 0))
            aggregated_importance[feature] = np.mean(importance_values)
        
        # Sort features by importance
        sorted_importance = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.stats['feature_importance']['aggregated'] = dict(sorted_importance)
        
        # Log top features
        self.update_status("Top 10 most important features:")
        for feature, importance in sorted_importance[:10]:
            self.update_status(f"  {feature}: {importance:.4f}")
    
    def get_recommendations(self, user_profile, meal_type='meal', max_recommendations=10):
        """Get XGBoost-based food recommendations with comprehensive debugging"""
        if not self.models or len(self.food_data) == 0:
            self.update_status("❌ No models trained or no food data available")
            return []
        
        try:
            self.update_status("🔍 Starting XGBoost recommendation process...")
            
            # Calculate nutritional targets
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            daily_targets = nutritional_data['daily_targets']
            meal_targets = nutritional_data['meal_targets'].get(meal_type.lower(), 
                                                              nutritional_data['meal_targets']['lunch'])
            
            # Get user's health conditions
            health_conditions = user_profile.get('health_conditions', [])
            self.update_status(f"🏥 Health conditions: {health_conditions if health_conditions else 'None'}")
            
            # Filter candidates
            candidates = self.food_data.copy()
            self.update_status(f"📊 Starting with {len(candidates)} total food items")
            
            category_filter = user_profile.get('category_filter', 'All')
            if category_filter != 'All':
                before_filter = len(candidates)
                candidates = candidates[candidates['Category'] == category_filter]
                self.update_status(f"🏷️ Category filter '{category_filter}': {len(candidates)} items (was {before_filter})")
            
            if len(candidates) == 0:
                self.update_status("❌ No candidates after category filtering")
                return []
            
            # Check feature availability
            missing_features = [f for f in self.features if f not in candidates.columns]
            if missing_features:
                self.update_status(f"⚠️ Missing features: {missing_features[:5]}...")
            
            # Prepare candidate features with better error handling
            try:
                candidate_features = candidates[self.features].fillna(0)
                
                # Check for infinite or very large values
                inf_mask = np.isinf(candidate_features.values)
                if inf_mask.any():
                    self.update_status("⚠️ Found infinite values in features, replacing with 0")
                    candidate_features = candidate_features.replace([np.inf, -np.inf], 0)
                
                # Check for very large values that might cause issues
                large_mask = np.abs(candidate_features.values) > 1e10
                if large_mask.any():
                    self.update_status("⚠️ Found very large values in features, clipping")
                    candidate_features = candidate_features.clip(-1e10, 1e10)
                
                candidate_features_scaled = self.scaler.transform(candidate_features)
                candidate_features_scaled = pd.DataFrame(candidate_features_scaled, columns=self.features)
                
                self.update_status(f"✅ Features prepared: {candidate_features_scaled.shape}")
                
            except Exception as e:
                self.update_status(f"❌ Error preparing features: {e}")
                return []
            
            # Get predictions from relevant models
            predictions = {}
            model_weights = {}
            
            self.update_status(f"🤖 Available models: {list(self.models.keys())}")
            
            if health_conditions:
                # Use condition-specific models
                condition_model_map = {
                    'Diabetes': 'Diabetes_Suitability',
                    'Obesity': 'Obesity_Suitability',
                    'Hypertension': 'Hypertension_Suitability',
                    'High_Cholesterol': 'Cholesterol_Suitability'
                }
                
                for condition in health_conditions:
                    if condition in condition_model_map:
                        model_name = condition_model_map[condition]
                        if model_name in self.models:
                            try:
                                pred = self.models[model_name].predict(candidate_features_scaled)
                                predictions[model_name] = pred
                                model_weights[model_name] = 1.0
                                self.update_status(f"✅ {model_name}: predictions range {pred.min():.3f} to {pred.max():.3f}")
                            except Exception as e:
                                self.update_status(f"❌ Error with {model_name}: {e}")
                        else:
                            self.update_status(f"⚠️ Model {model_name} not found for {condition}")
            
            # Always include overall health model as backup
            if 'Overall_Health' in self.models:
                try:
                    pred = self.models['Overall_Health'].predict(candidate_features_scaled)
                    if not predictions:  # Use as primary if no condition-specific models
                        predictions['Overall_Health'] = pred
                        model_weights['Overall_Health'] = 1.0
                    else:  # Use as secondary with lower weight
                        predictions['Overall_Health'] = pred
                        model_weights['Overall_Health'] = 0.3
                    self.update_status(f"✅ Overall_Health: predictions range {pred.min():.3f} to {pred.max():.3f}")
                except Exception as e:
                    self.update_status(f"❌ Error with Overall_Health model: {e}")
            
            if not predictions:
                self.update_status("❌ No successful model predictions")
                # Return simple nutritional matching as fallback
                return self._get_nutritional_fallback_recommendations(candidates, meal_targets, max_recommendations)
            
            # Combine predictions (lower scores are better for health suitability)
            combined_scores = np.zeros(len(candidates))
            total_weight = sum(model_weights.values())
            
            for model_name, pred in predictions.items():
                weight = model_weights[model_name] / total_weight
                combined_scores += pred * weight
            
            self.update_status(f"📊 Combined XGBoost scores: {combined_scores.min():.3f} to {combined_scores.max():.3f}")
            
            # Calculate nutritional match scores
            nutrition_scores = self._calculate_nutrition_match_scores(candidates, meal_targets)
            self.update_status(f"📊 Nutrition scores: {nutrition_scores.min():.3f} to {nutrition_scores.max():.3f}")
            
            # Combine XGBoost scores with nutritional matching
            model_weight = user_profile.get('model_weight', 0.6)
            final_scores = combined_scores * model_weight + nutrition_scores * (1 - model_weight)
            
            self.update_status(f"📊 Final scores: {final_scores.min():.3f} to {final_scores.max():.3f}")
            
            # Create recommendations
            recommendations = []
            candidate_indices = candidates.index.tolist()
            
            for i, (idx, score) in enumerate(zip(candidate_indices, final_scores)):
                food = candidates.loc[idx]
                
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
                    'xgboost_score': combined_scores[i],
                    'nutrition_score': nutrition_scores[i],
                    'final_score': score,
                    'model_predictions': {model: predictions[model][i] for model in predictions},
                    'suitable_for_conditions': self._check_condition_suitability(food, health_conditions),
                    'targets_met': self._check_targets_met(food, meal_targets),
                    'health_warnings': self._generate_health_warnings(food, health_conditions)
                }
                
                recommendations.append(recommendation)
            
            # Sort by final score (lower is better)
            recommendations.sort(key=lambda x: x['final_score'])
            
            # Add explanations
            for rec in recommendations:
                rec['explanation'] = self._generate_explanation(rec, meal_targets, health_conditions)
                rec['nutritional_data'] = nutritional_data
                
                # Add SHAP explanation if available
                if SHAP_AVAILABLE and health_conditions:
                    rec['shap_explanation'] = self._get_shap_explanation(rec, health_conditions)
            
            result_count = min(max_recommendations, len(recommendations))
            self.update_status(f"✅ Successfully generated {result_count} recommendations")
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.update_status(f"❌ Error in recommendation process: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return nutritional fallback
            try:
                return self._get_nutritional_fallback_recommendations(
                    self.food_data.copy(), {}, max_recommendations
                )
            except:
                return []
    
    def _get_nutritional_fallback_recommendations(self, candidates, meal_targets, max_recommendations):
        """Fallback recommendation method using simple nutritional matching"""
        self.update_status("🔄 Using nutritional fallback recommendations...")
        
        try:
            recommendations = []
            
            for idx, food in candidates.iterrows():
                # Simple scoring based on nutritional content
                score = 0
                
                # Prefer foods with balanced nutrition
                calories = float(food.get('Energy(kcal) by calculation', 0))
                protein = float(food.get('Protein(g)', 0))
                fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
                sugar = float(food.get('SUGAR(g)', 0))
                sodium = float(food.get('Na(mg)', 0))
                
                # Scoring criteria (lower score is better)
                if calories > 500:
                    score += 2
                elif calories > 300:
                    score += 1
                
                if protein >= 10:
                    score -= 1
                elif protein >= 5:
                    score -= 0.5
                
                if fiber >= 5:
                    score -= 1
                elif fiber >= 3:
                    score -= 0.5
                
                if sugar > 15:
                    score += 2
                elif sugar > 8:
                    score += 1
                
                if sodium > 400:
                    score += 2
                elif sodium > 200:
                    score += 1
                
                recommendation = {
                    'food_id': idx,
                    'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                    'category': food.get('Category', 'Unknown'),
                    'calories': calories,
                    'protein': protein,
                    'carbs': float(food.get('CHOCDF (g) Carbohydrate', 0)),
                    'sugar': sugar,
                    'fiber': fiber,
                    'fat': float(food.get('Fat(g)', 0)),
                    'sodium': sodium,
                    'potassium': float(food.get('K(mg)', 0)),
                    'cholesterol': float(food.get('CHOLE(mg) Cholesterol', 0)),
                    'saturated_fat': float(food.get('FASAT (g) Saturated FA', 0)),
                    'xgboost_score': score,  # Using simple score as proxy
                    'nutrition_score': score,
                    'final_score': score,
                    'model_predictions': {'Fallback': score},
                    'suitable_for_conditions': [],
                    'targets_met': [],
                    'health_warnings': [],
                    'explanation': "Simple nutritional matching (XGBoost models unavailable)"
                }
                
                recommendations.append(recommendation)
            
            # Sort by score (lower is better)
            recommendations.sort(key=lambda x: x['final_score'])
            
            self.update_status(f"✅ Fallback recommendations: {len(recommendations[:max_recommendations])}")
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.update_status(f"❌ Fallback recommendations failed: {e}")
            return []
    
    def _calculate_nutrition_match_scores(self, candidates, targets):
        """Calculate nutritional matching scores"""
        scores = []
        
        for _, food in candidates.iterrows():
            score = 0
            
            # Calorie match
            calories = float(food.get('Energy(kcal) by calculation', 0))
            target_calories = targets.get('calories', 500)
            if target_calories > 0:
                calorie_diff = abs(calories - target_calories) / target_calories
                score += calorie_diff * 2
            
            # Protein match
            protein = float(food.get('Protein(g)', 0))
            protein_target = (targets.get('protein_min', 0) + targets.get('protein_max', 50)) / 2
            if protein_target > 0:
                protein_diff = abs(protein - protein_target) / protein_target
                score += protein_diff
            
            # Sugar penalty
            sugar = float(food.get('SUGAR(g)', 0))
            sugar_max = targets.get('sugar_max', 10)
            if sugar > sugar_max and sugar_max > 0:
                score += (sugar - sugar_max) / sugar_max
            
            # Fiber bonus
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            fiber_min = targets.get('fiber_min', 5)
            if fiber < fiber_min and fiber_min > 0:
                score += (fiber_min - fiber) / fiber_min
            
            scores.append(score)
        
        return np.array(scores)
    
    def _check_condition_suitability(self, food, health_conditions):
        """Check health condition suitability"""
        suitable = []
        
        score_thresholds = {
            'Diabetes': 2.0,
            'Obesity': 2.0,
            'Hypertension': 2.0,
            'High_Cholesterol': 2.0
        }
        
        for condition in health_conditions:
            score_col = f'{condition}_Score'
            if score_col in food and float(food.get(score_col, 0)) <= score_thresholds.get(condition, 2.0):
                suitable.append(condition)
        
        return suitable
    
    def _check_targets_met(self, food, targets):
        """Check which nutritional targets are met"""
        met = []
        
        # Check various targets
        calories = float(food.get('Energy(kcal) by calculation', 0))
        target_calories = targets.get('calories', 500)
        if 0.7 * target_calories <= calories <= 1.3 * target_calories:
            met.append('Calories')
        
        protein = float(food.get('Protein(g)', 0))
        if protein >= targets.get('protein_min', 0):
            met.append('Protein')
        
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= targets.get('fiber_min', 0):
            met.append('Fiber')
        
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar <= targets.get('sugar_max', 10):
            met.append('Sugar')
        
        return met
    
    def _generate_health_warnings(self, food, health_conditions):
        """Generate health warnings"""
        warnings = []
        
        for condition in health_conditions:
            if condition == 'Diabetes':
                if float(food.get('SUGAR(g)', 0)) > 15:
                    warnings.append("High sugar content - monitor blood glucose carefully")
                if (float(food.get('CHOCDF (g) Carbohydrate', 0)) > 30 and 
                    float(food.get('FIBTG (g) Dietary fibre', 0)) < 3):
                    warnings.append("High carbs with low fiber - may cause blood sugar spike")
            
            elif condition == 'Hypertension':
                if float(food.get('Na(mg)', 0)) > 400:
                    warnings.append("High sodium content - may increase blood pressure")
            
            elif condition == 'High_Cholesterol':
                if float(food.get('FASAT (g) Saturated FA', 0)) > 5:
                    warnings.append("High saturated fat - may raise cholesterol levels")
        
        return warnings
    
    def _get_shap_explanation(self, recommendation, health_conditions):
        """Get SHAP explanation for recommendation"""
        if not SHAP_AVAILABLE or not health_conditions:
            return "SHAP explanations not available"
        
        # Use the most relevant model for explanation
        condition_model_map = {
            'Diabetes': 'Diabetes_Suitability',
            'Obesity': 'Obesity_Suitability',
            'Hypertension': 'Hypertension_Suitability',
            'High_Cholesterol': 'Cholesterol_Suitability'
        }
        
        for condition in health_conditions:
            model_name = condition_model_map.get(condition)
            if model_name in self.shap_explainers:
                try:
                    explainer_data = self.shap_explainers[model_name]
                    # Get feature importance for this prediction
                    food_id = recommendation['food_id']
                    food_data = self.food_data.loc[food_id, self.features].fillna(0)
                    food_scaled = self.scaler.transform([food_data])
                    
                    shap_values = explainer_data['explainer'].shap_values(food_scaled)
                    
                    # Get top contributing features
                    feature_contributions = list(zip(self.features, shap_values[0]))
                    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    explanation = f"Top factors for {condition}:\n"
                    for feature, contribution in feature_contributions[:5]:
                        direction = "increases" if contribution > 0 else "decreases"
                        explanation += f"• {feature}: {direction} suitability ({contribution:.3f})\n"
                    
                    return explanation
                    
                except Exception as e:
                    return f"SHAP explanation error: {e}"
        
        return "No SHAP explanation available for selected conditions"
    
    def _generate_explanation(self, recommendation, targets, health_conditions):
        """Generate comprehensive explanation"""
        explanations = []
        
        # XGBoost score interpretation
        xgb_score = recommendation['xgboost_score']
        if xgb_score < 1.0:
            explanations.append("Excellent XGBoost health match")
        elif xgb_score < 2.0:
            explanations.append("Good XGBoost health match")
        else:
            explanations.append("Fair XGBoost health match")
        
        # Nutritional matching
        nutrition_score = recommendation['nutrition_score']
        if nutrition_score < 1.0:
            explanations.append("Excellent nutritional match")
        elif nutrition_score < 2.0:
            explanations.append("Good nutritional match")
        
        # Health condition suitability
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            conditions_text = ", ".join(suitable_conditions)
            explanations.append(f"XGBoost verified suitable for {conditions_text}")
        
        # Specific benefits
        if recommendation['fiber'] >= 5:
            explanations.append("High fiber content")
        if recommendation['protein'] >= 10:
            explanations.append("Good protein source")
        if recommendation['sugar'] <= 5:
            explanations.append("Low sugar content")
        
        return " | ".join(explanations)
    
    def get_stats(self):
        """Get comprehensive statistics"""
        return self.stats
    
    def get_model_insights(self):
        """Get detailed model insights"""
        insights = {
            'feature_importance': self.stats.get('feature_importance', {}),
            'model_performance': self.stats.get('model_performance', {}),
            'hyperparameter_tuning': self.stats.get('hyperparameter_tuning', {}),
            'shap_available': SHAP_AVAILABLE,
            'total_features': len(self.features),
            'models_trained': len(self.models)
        }
        return insights


class HealthDrivenXGBoostFoodRecommenderUI:
    """Advanced GUI for XGBoost food recommendation system"""
    
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Health-Driven XGBoost Food Recommendation System")
        self.master.geometry("1500x1000")
        self.master.configure(bg="#f8f9fa")
        
        # Modern color scheme
        self.colors = {
            'primary': '#1a365d',
            'secondary': '#2b6cb0',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'light': '#f7fafc',
            'white': '#ffffff',
            'background': '#f8f9fa',
            'text': '#2d3748',
            'text_light': '#718096',
            'accent': '#805ad5'
        }
        
        # Setup styles
        self.setup_styles()
        
        # Initialize recommender
        self.recommender = recommender or HealthAwareXGBoostRecommender()
        
        # Create UI
        self.create_main_interface()
        
        # Initialize variables
        self.last_recommendations = []
        self.current_nutritional_data = None
        self.model_insights = None
        
        # Update components
        self.update_category_list()
        self.update_stats_display()
        self.update_model_insights()
    
    def setup_styles(self):
        """Setup modern UI styles"""
        self.style = ttk.Style()
        
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        # Fonts
        self.fonts = {
            'heading': ('Segoe UI', 18, 'bold'),
            'subheading': ('Segoe UI', 14, 'bold'),
            'body': ('Segoe UI', 11),
            'caption': ('Segoe UI', 9),
            'code': ('Consolas', 10)
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=self.fonts['heading'], 
                           foreground=self.colors['primary'])
        self.style.configure('Subtitle.TLabel', font=self.fonts['subheading'], 
                           foreground=self.colors['text'])
        self.style.configure('XGBoost.Treeview', font=self.fonts['body'], rowheight=28)
        self.style.configure('XGBoost.Treeview.Heading', font=self.fonts['subheading'])
    
    def create_main_interface(self):
        """Create the main interface"""
        # Main container
        main_container = ttk.Frame(self.master, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area with three columns
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (input)
        self.create_input_panel(content_frame)
        
        # Middle panel (results)
        self.create_results_panel(content_frame)
        
        # Right panel (insights)
        self.create_insights_panel(content_frame)
        
        # Bottom panels
        self.create_charts_panel(main_container)
        self.create_status_bar()
    
    def create_header(self, parent):
        """Create header with title and controls"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title section
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(title_frame, text="Health-Driven XGBoost Food Recommendation System", 
                 style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Advanced Gradient Boosting with Medical Guidelines & SHAP Interpretability", 
                 style='Subtitle.TLabel', foreground=self.colors['text_light']).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Get XGBoost Recommendations", 
                  command=self.get_recommendations, width=20).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_form, width=10).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Calculate Nutrition", 
                  command=self.show_nutrition_calculation, width=15).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Model Insights", 
                  command=self.show_model_insights, width=12).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="🔍 Diagnostics", 
                  command=self.show_diagnostics, width=12).pack(side=tk.RIGHT, padx=5)
    
    def create_input_panel(self, parent):
        """Create input panel"""
        left_panel = ttk.LabelFrame(parent, text="Health Profile & Settings", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Personal Information
        personal_frame = ttk.LabelFrame(left_panel, text="Personal Information", padding="10")
        personal_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Form fields
        self.create_form_field(personal_frame, "Weight (kg):", "weight_var", 70.0, 0, 0)
        self.create_form_field(personal_frame, "Height (cm):", "height_var", 170.0, 1, 0)
        self.create_form_field(personal_frame, "Age (years):", "age_var", 30, 2, 0)
        
        # Dropdowns
        fields = [
            ("Gender:", "gender_var", "Male", ['Male', 'Female'], 3),
            ("Activity Level:", "activity_var", "Moderate", 
             ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'], 4),
            ("Weight Goal:", "weight_goal_var", "Maintain Weight", 
             ['Lose Weight', 'Maintain Weight', 'Gain Weight'], 5)
        ]
        
        for label, var_name, default, values, row in fields:
            ttk.Label(personal_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            combo = ttk.Combobox(personal_frame, textvariable=var, values=values, 
                               state="readonly", width=15)
            combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        
        # BMI Display
        ttk.Label(personal_frame, text="BMI:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.bmi_var = tk.StringVar(value="Calculating...")
        ttk.Label(personal_frame, textvariable=self.bmi_var, 
                 foreground=self.colors['secondary']).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Health Conditions
        conditions_frame = ttk.LabelFrame(left_panel, text="Health Conditions", padding="10")
        conditions_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        conditions = [
            ("Diabetes", self.diabetes_var),
            ("Obesity", self.obesity_var),
            ("Hypertension", self.hypertension_var),
            ("High Cholesterol", self.cholesterol_var)
        ]
        
        for condition, var in conditions:
            ttk.Checkbutton(conditions_frame, text=condition, variable=var).pack(anchor=tk.W, pady=2)
        
        # XGBoost Settings
        xgb_frame = ttk.LabelFrame(left_panel, text="XGBoost Settings", padding="10")
        xgb_frame.pack(fill=tk.X, pady=(0, 15))
        
        settings = [
            ("Meal Type:", "meal_type_var", "Lunch", 
             ['Breakfast', 'Lunch', 'Dinner', 'Snack'], 0),
            ("Category Filter:", "category_var", "All", ['All'], 1),
            ("Max Results:", "max_results_var", "10", 
             ['5', '10', '15', '20', '25'], 2)
        ]
        
        for label, var_name, default, values, row in settings:
            ttk.Label(xgb_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            combo = ttk.Combobox(xgb_frame, textvariable=var, values=values, 
                               state="readonly", width=15)
            combo.grid(row=row, column=1, sticky=tk.W, pady=5)
            
            if var_name == "category_var":
                self.category_combo = combo
        
        # Model Confidence
        ttk.Label(xgb_frame, text="Model Weight:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.model_weight_var = tk.DoubleVar(value=0.6)
        weight_scale = ttk.Scale(xgb_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, 
                               variable=self.model_weight_var, length=100)
        weight_scale.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(xgb_frame, text="XGBoost ← → Nutrition", 
                 font=self.fonts['caption']).grid(row=4, column=0, columnspan=2, pady=2)
        
        # Bind BMI calculation
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()
    
    def create_form_field(self, parent, label_text, var_name, default_value, row, col):
        """Create form field"""
        ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky=tk.W, pady=5)
        
        if isinstance(default_value, float):
            var = tk.DoubleVar(value=default_value)
        else:
            var = tk.IntVar(value=default_value)
        
        setattr(self, var_name, var)
        
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=col+1, sticky=tk.W, pady=5)
        
        return var, entry
    
    def calculate_bmi(self, *args):
        """Calculate and display BMI"""
        try:
            weight = self.weight_var.get()
            height = self.height_var.get() / 100
            
            if weight > 0 and height > 0:
                bmi = weight / (height * height)
                
                if bmi < 18.5:
                    category = "Underweight"
                    color = self.colors['warning']
                elif bmi < 25:
                    category = "Normal"
                    color = self.colors['success']
                elif bmi < 30:
                    category = "Overweight"
                    color = self.colors['warning']
                else:
                    category = "Obese"
                    color = self.colors['danger']
                
                self.bmi_var.set(f"{bmi:.1f} ({category})")
            else:
                self.bmi_var.set("Invalid input")
        except:
            self.bmi_var.set("Calculating...")
    
    def update_category_list(self):
        """Update category dropdown"""
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
            if 'Category' in self.recommender.food_data.columns:
                unique_categories = sorted(self.recommender.food_data['Category'].dropna().unique())
                categories.extend(unique_categories)
        
        if hasattr(self, 'category_combo'):
            self.category_combo['values'] = categories
            if self.category_var.get() not in categories:
                self.category_var.set('All')
    
    def create_results_panel(self, parent):
        """Create results panel"""
        middle_panel = ttk.LabelFrame(parent, text="XGBoost Recommendations", padding="15")
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Notebook for different views
        self.notebook = ttk.Notebook(middle_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Recommendations tab
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="🎯 Recommendations")
        
        # Enhanced treeview
        columns = ('Name', 'Category', 'Calories', 'Protein', 'Sugar', 'Fiber', 
                  'XGB Score', 'Final Score', 'Suitable For')
        self.tree = ttk.Treeview(rec_frame, columns=columns, show='headings', 
                               style='XGBoost.Treeview', height=16)
        
        # Configure columns with better widths
        column_widths = {
            'Name': 140, 'Category': 90, 'Calories': 70, 'Protein': 60, 
            'Sugar': 50, 'Fiber': 50, 'XGB Score': 70, 'Final Score': 70, 'Suitable For': 100
        }
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Scrollbars
        tree_scroll_frame = ttk.Frame(rec_frame)
        tree_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar = ttk.Scrollbar(tree_scroll_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_scroll_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Nutritional targets tab
        targets_frame = ttk.Frame(self.notebook)
        self.notebook.add(targets_frame, text="📊 Calculated Targets")
        
        self.targets_text = tk.Text(targets_frame, wrap=tk.WORD, font=self.fonts['body'])
        targets_scrollbar = ttk.Scrollbar(targets_frame, orient="vertical", command=self.targets_text.yview)
        self.targets_text.configure(yscrollcommand=targets_scrollbar.set)
        
        self.targets_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        targets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Food details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="🔍 Food Details")
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, font=self.fonts['body'])
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
        # Initial text
        self.targets_text.insert(tk.END, "Click 'Calculate Nutrition' to see personalized targets based on medical guidelines and XGBoost feature engineering.")
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information, XGBoost predictions, and SHAP interpretability analysis.")
    
    def create_insights_panel(self, parent):
        """Create model insights panel"""
        right_panel = ttk.LabelFrame(parent, text="Model Insights & Performance", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Model performance section
        perf_frame = ttk.LabelFrame(right_panel, text="XGBoost Performance", padding="10")
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.performance_text = tk.Text(perf_frame, wrap=tk.WORD, font=self.fonts['caption'], 
                                       height=8, width=40)
        perf_scroll = ttk.Scrollbar(perf_frame, orient="vertical", command=self.performance_text.yview)
        self.performance_text.configure(yscrollcommand=perf_scroll.set)
        
        self.performance_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature importance section
        feat_frame = ttk.LabelFrame(right_panel, text="Top Features", padding="10")
        feat_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.features_text = tk.Text(feat_frame, wrap=tk.WORD, font=self.fonts['caption'], 
                                    height=10, width=40)
        feat_scroll = ttk.Scrollbar(feat_frame, orient="vertical", command=self.features_text.yview)
        self.features_text.configure(yscrollcommand=feat_scroll.set)
        
        self.features_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # SHAP section
        shap_frame = ttk.LabelFrame(right_panel, text="SHAP Status", padding="10")
        shap_frame.pack(fill=tk.X)
        
        shap_status = "✅ Available" if SHAP_AVAILABLE else "❌ Not Available"
        ttk.Label(shap_frame, text=f"SHAP Interpretability: {shap_status}").pack()
        
        if SHAP_AVAILABLE:
            ttk.Label(shap_frame, text="Model explanations enabled", 
                     foreground=self.colors['success']).pack()
        else:
            ttk.Label(shap_frame, text="Install SHAP for explanations", 
                     foreground=self.colors['warning']).pack()
    
    def create_charts_panel(self, parent):
        """Create comprehensive charts panel"""
        charts_frame = ttk.LabelFrame(parent, text="Advanced Analytics & Visualizations", padding="10")
        charts_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Create matplotlib figure with more subplots
        self.fig = Figure(figsize=(16, 6), dpi=100)
        self.fig.patch.set_facecolor(self.colors['white'])
        
        # Create 5 subplots
        self.ax1 = self.fig.add_subplot(151)
        self.ax2 = self.fig.add_subplot(152)
        self.ax3 = self.fig.add_subplot(153)
        self.ax4 = self.fig.add_subplot(154)
        self.ax5 = self.fig.add_subplot(155)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_charts([])
    
    def create_status_bar(self):
        """Create enhanced status bar"""
        self.status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding="8")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready - XGBoost system initialized. Enter health profile and click 'Calculate Nutrition'")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, 
                                     font=self.fonts['body'])
        self.status_label.pack(side=tk.LEFT)
        
        # Stats display
        self.stats_frame = ttk.Frame(self.status_frame)
        self.stats_frame.pack(side=tk.RIGHT)
    
    def get_user_profile(self):
        """Get user profile from form"""
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
            'category_filter': self.category_var.get(),
            'model_weight': self.model_weight_var.get()
        }
    
    def show_nutrition_calculation(self):
        """Show calculated nutritional targets"""
        try:
            user_profile = self.get_user_profile()
            
            # Calculate targets
            nutritional_data = self.recommender.nutrition_calculator.calculate_nutritional_targets(user_profile)
            self.current_nutritional_data = nutritional_data
            
            # Display results
            self.targets_text.delete(1.0, tk.END)
            
            # Header
            self.targets_text.insert(tk.END, "🎯 PERSONALIZED NUTRITIONAL TARGETS (XGBoost Enhanced)\n", "header")
            self.targets_text.insert(tk.END, "="*70 + "\n\n", "separator")
            
            # Medical calculations
            calc = nutritional_data['calculations']
            self.targets_text.insert(tk.END, "📋 Medical Calculations:\n", "subheader")
            self.targets_text.insert(tk.END, f"• {calc['bmr_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['tdee_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['target_formula']}\n\n")
            
            # Daily targets
            daily = nutritional_data['daily_targets']
            self.targets_text.insert(tk.END, "🍽️ Daily Nutritional Targets:\n", "subheader")
            self.targets_text.insert(tk.END, f"• Calories: {daily['calories']:.0f} kcal\n")
            self.targets_text.insert(tk.END, f"• Protein: {daily['protein_min']:.0f}-{daily['protein_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Carbohydrates: {daily['carbs_min']:.0f}-{daily['carbs_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Sugar (max): {daily['sugar_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Fiber (min): {daily['fiber_min']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Fat: {daily['fat_min']:.0f}-{daily['fat_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Sodium (max): {daily['sodium_max']:.0f} mg\n\n")
            
            # Meal targets
            meal_type = self.meal_type_var.get().lower()
            if meal_type in nutritional_data['meal_targets']:
                meal = nutritional_data['meal_targets'][meal_type]
                self.targets_text.insert(tk.END, f"🍴 {self.meal_type_var.get()} Targets (XGBoost Input):\n", "subheader")
                self.targets_text.insert(tk.END, f"• Calories: {meal['calories']:.0f} kcal\n")
                self.targets_text.insert(tk.END, f"• Protein: {meal['protein_min']:.0f}-{meal['protein_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Carbohydrates: {meal['carbs_min']:.0f}-{meal['carbs_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Sugar (max): {meal['sugar_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fiber (min): {meal['fiber_min']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fat: {meal['fat_min']:.0f}-{meal['fat_max']:.0f} g\n\n")
            
            # Health conditions
            health_conditions = user_profile['health_conditions']
            if health_conditions:
                self.targets_text.insert(tk.END, "🏥 Health Condition Models:\n", "subheader")
                for condition in health_conditions:
                    self.targets_text.insert(tk.END, f"• {condition}: Specialized XGBoost model active\n")
                self.targets_text.insert(tk.END, "\n")
            
            # XGBoost process
            self.targets_text.insert(tk.END, "🤖 XGBoost Process:\n", "subheader")
            self.targets_text.insert(tk.END, "1. Advanced feature engineering (interactions, ratios, composites)\n")
            self.targets_text.insert(tk.END, "2. Health condition-specific model selection\n")
            self.targets_text.insert(tk.END, "3. Gradient boosting prediction with regularization\n")
            self.targets_text.insert(tk.END, "4. SHAP-based interpretability analysis\n")
            self.targets_text.insert(tk.END, "5. Combined scoring with nutritional matching\n")
            
            # Configure text tags
            self.targets_text.tag_configure("header", font=self.fonts['heading'], 
                                          foreground=self.colors['primary'])
            self.targets_text.tag_configure("subheader", font=self.fonts['subheading'], 
                                          foreground=self.colors['secondary'])
            self.targets_text.tag_configure("separator", foreground=self.colors['text_light'])
            
            # Switch to targets tab
            self.notebook.select(1)
            
            self.status_var.set("Nutritional targets calculated - XGBoost models ready for food matching")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating nutrition targets: {str(e)}")
    
    def get_recommendations(self):
        """Get XGBoost recommendations with better error handling"""
        try:
            if not self.current_nutritional_data:
                messagebox.showwarning("Warning", "Please calculate nutrition targets first!")
                return
            
            self.update_category_list()
            
            user_profile = self.get_user_profile()
            meal_type = self.meal_type_var.get().lower()
            max_results = int(self.max_results_var.get())
            
            # Show diagnostic information
            self.status_var.set("🔍 Running XGBoost diagnostics...")
            
            # Check if recommender has data
            if not hasattr(self.recommender, 'food_data') or self.recommender.food_data.empty:
                error_msg = ("❌ XGBoost Error: No food data loaded\n\n"
                           "Possible solutions:\n"
                           "1. Check that CSV files are in 'datasets/' folder\n"
                           "2. Verify CSV file format and encoding\n"
                           "3. Check console output for detailed error messages\n\n"
                           "Expected files: Drinking.csv, Fruit.csv, meat.csv, etc.")
                messagebox.showerror("Data Error", error_msg)
                return
            
            # Check if models are trained
            if not hasattr(self.recommender, 'models') or not self.recommender.models:
                error_msg = ("❌ XGBoost Error: No models trained\n\n"
                           "Possible solutions:\n"
                           "1. Check console output for model training errors\n"
                           "2. Verify that nutritional data is valid\n"
                           "3. Try restarting the application\n"
                           "4. Check that XGBoost is properly installed")
                messagebox.showerror("Model Error", error_msg)
                return
            
            self.status_var.set("🤖 Getting XGBoost predictions...")
            
            # Get recommendations
            recommendations = self.recommender.get_recommendations(
                user_profile, meal_type, max_results
            )
            
            self.last_recommendations = recommendations
            
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Display recommendations
            if recommendations:
                self.status_var.set("✅ Processing XGBoost results...")
                
                for rec in recommendations:
                    suitable_text = ", ".join(rec['suitable_for_conditions']) if rec['suitable_for_conditions'] else "General"
                    
                    # Color coding based on scores
                    tags = []
                    if rec['final_score'] < 1.0:
                        tags.append('excellent')
                    elif rec['final_score'] < 2.0:
                        tags.append('good')
                    else:
                        tags.append('fair')
                    
                    self.tree.insert('', 'end', values=(
                        rec['name'][:18],
                        rec['category'],
                        f"{rec['calories']:.0f}",
                        f"{rec['protein']:.1f}",
                        f"{rec['sugar']:.1f}",
                        f"{rec['fiber']:.1f}",
                        f"{rec['xgboost_score']:.2f}",
                        f"{rec['final_score']:.2f}",
                        suitable_text[:12]
                    ), tags=tags)
                
                # Configure row colors
                self.tree.tag_configure('excellent', background='#f0fff0')
                self.tree.tag_configure('good', background='#f8f8ff')
                self.tree.tag_configure('fair', background='#fff8f0')
                
                # Select first item
                if self.tree.get_children():
                    first_item = self.tree.get_children()[0]
                    self.tree.selection_set(first_item)
                    self.tree.focus(first_item)
                    self.show_food_details(None)
                
                # Update charts
                self.update_charts(recommendations)
                
                # Switch to recommendations
                self.notebook.select(0)
                
                health_conditions = user_profile['health_conditions']
                condition_text = ", ".join(health_conditions) if health_conditions else "general health"
                self.status_var.set(f"✅ XGBoost found {len(recommendations)} optimal foods for {condition_text}")
                
            else:
                # More helpful error message based on the actual problem
                category_filter = user_profile.get('category_filter', 'All')
                health_conditions = user_profile.get('health_conditions', [])
                
                # Check what might be the issue
                total_foods = len(self.recommender.food_data) if hasattr(self.recommender, 'food_data') else 0
                
                if category_filter != 'All':
                    filtered_foods = len(self.recommender.food_data[
                        self.recommender.food_data['Category'] == category_filter
                    ]) if total_foods > 0 else 0
                    
                    error_msg = (f"❌ No XGBoost matches in '{category_filter}' category\n\n"
                               f"📊 Data Summary:\n"
                               f"• Total foods in database: {total_foods}\n"
                               f"• Foods in '{category_filter}': {filtered_foods}\n\n"
                               f"🔧 Suggestions:\n"
                               f"1. Try 'All' categories\n"
                               f"2. Adjust model weight slider (currently {self.model_weight_var.get():.1f})\n"
                               f"3. Modify health conditions\n"
                               f"4. Check different meal types\n\n"
                               f"🏥 Current health conditions: {', '.join(health_conditions) if health_conditions else 'None'}")
                else:
                    error_msg = (f"❌ No XGBoost matches found\n\n"
                               f"📊 Data Summary:\n"
                               f"• Total foods in database: {total_foods}\n"
                               f"• Models trained: {len(self.recommender.models) if hasattr(self.recommender, 'models') else 0}\n\n"
                               f"🔧 Possible issues:\n"
                               f"1. Very restrictive health conditions\n"
                               f"2. XGBoost models too strict (adjust model weight)\n"
                               f"3. Limited food data for your criteria\n"
                               f"4. Model training issues\n\n"
                               f"💡 Try:\n"
                               f"• Reducing model weight to 0.3-0.4\n"
                               f"• Removing some health conditions temporarily\n"
                               f"• Checking console output for detailed logs")
                
                messagebox.showinfo("XGBoost Results", error_msg)
                self.status_var.set("❌ No XGBoost matches - see diagnostic information")
                
        except Exception as e:
            error_msg = (f"❌ XGBoost Error: {str(e)}\n\n"
                       f"🔧 Troubleshooting:\n"
                       f"1. Check console output for detailed error messages\n"
                       f"2. Verify all CSV files are properly formatted\n"
                       f"3. Ensure XGBoost and dependencies are installed\n"
                       f"4. Try restarting the application\n\n"
                       f"📋 Required libraries:\n"
                       f"• xgboost\n"
                       f"• scikit-learn\n"
                       f"• pandas\n"
                       f"• numpy")
            
            messagebox.showerror("XGBoost Error", error_msg)
            self.status_var.set("❌ XGBoost error occurred - check details")
    
    def show_food_details(self, event):
        """Show detailed food information with XGBoost insights"""
        selected_items = self.tree.selection()
        if not selected_items or not self.last_recommendations:
            return
        
        item = selected_items[0]
        item_index = self.tree.index(item)
        
        if item_index < len(self.last_recommendations):
            rec = self.last_recommendations[item_index]
            
            self.details_text.delete(1.0, tk.END)
            
            # Food header
            self.details_text.insert(tk.END, f"🍽️ {rec['name']}\n", "title")
            self.details_text.insert(tk.END, f"Category: {rec['category']}\n\n", "subtitle")
            
            # XGBoost Analysis
            self.details_text.insert(tk.END, "🤖 XGBoost Analysis:\n", "header")
            self.details_text.insert(tk.END, f"• XGBoost Score: {rec['xgboost_score']:.3f}\n")
            self.details_text.insert(tk.END, f"• Nutrition Score: {rec['nutrition_score']:.3f}\n")
            self.details_text.insert(tk.END, f"• Final Combined Score: {rec['final_score']:.3f}\n")
            
            # Model predictions breakdown
            if 'model_predictions' in rec and rec['model_predictions']:
                self.details_text.insert(tk.END, "\n📊 Model Predictions:\n", "header")
                for model_name, prediction in rec['model_predictions'].items():
                    condition = model_name.replace('_Suitability', '')
                    self.details_text.insert(tk.END, f"• {condition}: {prediction:.3f}\n")
            
            # Score interpretation
            final_score = rec['final_score']
            if final_score < 1.0:
                interpretation = "Excellent match ⭐⭐⭐"
                color = "excellent"
            elif final_score < 2.0:
                interpretation = "Good match ⭐⭐"
                color = "good"
            else:
                interpretation = "Fair match ⭐"
                color = "fair"
            
            self.details_text.insert(tk.END, f"• Overall Rating: {interpretation}\n\n", color)
            
            # Nutritional Information
            self.details_text.insert(tk.END, "📋 Nutritional Information (per 100g):\n", "header")
            self.details_text.insert(tk.END, f"• Energy: {rec['calories']:.0f} kcal\n")
            self.details_text.insert(tk.END, f"• Protein: {rec['protein']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Carbohydrates: {rec['carbs']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sugar: {rec['sugar']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Dietary Fiber: {rec['fiber']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Fat: {rec['fat']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sodium: {rec['sodium']:.0f} mg\n")
            self.details_text.insert(tk.END, f"• Potassium: {rec['potassium']:.0f} mg\n")
            self.details_text.insert(tk.END, f"• Cholesterol: {rec['cholesterol']:.0f} mg\n")
            self.details_text.insert(tk.END, f"• Saturated Fat: {rec['saturated_fat']:.1f} g\n\n")
            
            # Health suitability
            if rec['suitable_for_conditions']:
                self.details_text.insert(tk.END, "✅ Suitable for Health Conditions:\n", "good")
                for condition in rec['suitable_for_conditions']:
                    self.details_text.insert(tk.END, f"• {condition}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Health warnings
            if rec['health_warnings']:
                self.details_text.insert(tk.END, "⚠️ Health Considerations:\n", "warning")
                for warning in rec['health_warnings']:
                    self.details_text.insert(tk.END, f"• {warning}\n", "warning")
                self.details_text.insert(tk.END, "\n")
            
            # Targets met
            if rec['targets_met']:
                self.details_text.insert(tk.END, "🎯 Nutritional Targets Met:\n", "good")
                for target in rec['targets_met']:
                    self.details_text.insert(tk.END, f"• {target}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # XGBoost explanation
            self.details_text.insert(tk.END, "🔍 Why XGBoost Recommended This Food:\n", "header")
            self.details_text.insert(tk.END, f"{rec['explanation']}\n\n")
            
            # SHAP explanation if available
            if 'shap_explanation' in rec:
                self.details_text.insert(tk.END, "🧠 SHAP Interpretability Analysis:\n", "header")
                self.details_text.insert(tk.END, f"{rec['shap_explanation']}\n")
            
            # Configure text tags
            self.details_text.tag_configure("title", font=self.fonts['heading'], 
                                          foreground=self.colors['primary'])
            self.details_text.tag_configure("subtitle", font=self.fonts['subheading'], 
                                          foreground=self.colors['text'])
            self.details_text.tag_configure("header", font=self.fonts['subheading'], 
                                          foreground=self.colors['secondary'])
            self.details_text.tag_configure("good", foreground=self.colors['success'])
            self.details_text.tag_configure("warning", foreground=self.colors['warning'])
            self.details_text.tag_configure("excellent", foreground=self.colors['success'], 
                                          font=self.fonts['subheading'])
            self.details_text.tag_configure("fair", foreground=self.colors['warning'])
    
    def show_model_insights(self):
        """Show detailed model insights window"""
        insights_window = tk.Toplevel(self.master)
        insights_window.title("XGBoost Model Insights & Performance")
        insights_window.geometry("800x600")
        insights_window.configure(bg=self.colors['background'])
        
        # Create notebook for different insight tabs
        insights_notebook = ttk.Notebook(insights_window)
        insights_notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Model Performance Tab
        perf_frame = ttk.Frame(insights_notebook)
        insights_notebook.add(perf_frame, text="📈 Model Performance")
        
        perf_text = tk.Text(perf_frame, wrap=tk.WORD, font=self.fonts['body'])
        perf_scroll = ttk.Scrollbar(perf_frame, orient="vertical", command=perf_text.yview)
        perf_text.configure(yscrollcommand=perf_scroll.set)
        
        perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        perf_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature Importance Tab
        feat_frame = ttk.Frame(insights_notebook)
        insights_notebook.add(feat_frame, text="🎯 Feature Importance")
        
        feat_text = tk.Text(feat_frame, wrap=tk.WORD, font=self.fonts['body'])
        feat_scroll = ttk.Scrollbar(feat_frame, orient="vertical", command=feat_text.yview)
        feat_text.configure(yscrollcommand=feat_scroll.set)
        
        feat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Hyperparameters Tab
        hyper_frame = ttk.Frame(insights_notebook)
        insights_notebook.add(hyper_frame, text="⚙️ Hyperparameters")
        
        hyper_text = tk.Text(hyper_frame, wrap=tk.WORD, font=self.fonts['body'])
        hyper_scroll = ttk.Scrollbar(hyper_frame, orient="vertical", command=hyper_text.yview)
        hyper_text.configure(yscrollcommand=hyper_scroll.set)
        
        hyper_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hyper_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate insights
        insights = self.recommender.get_model_insights()
        
        # Model Performance
        perf_text.insert(tk.END, "🚀 XGBoost MODEL PERFORMANCE ANALYSIS\n", "header")
        perf_text.insert(tk.END, "="*60 + "\n\n")
        
        if 'model_performance' in insights:
            for model_name, metrics in insights['model_performance'].items():
                perf_text.insert(tk.END, f"📊 {model_name} Model:\n", "subheader")
                perf_text.insert(tk.END, f"• R² Score: {metrics.get('r2_score', 0):.4f}\n")
                perf_text.insert(tk.END, f"• Mean Squared Error: {metrics.get('mse', 0):.4f}\n")
                perf_text.insert(tk.END, f"• Cross-Validation Mean: {metrics.get('cv_mean', 0):.4f}\n")
                perf_text.insert(tk.END, f"• Cross-Validation Std: {metrics.get('cv_std', 0):.4f}\n\n")
        
        perf_text.insert(tk.END, f"🔧 Total Models Trained: {insights.get('models_trained', 0)}\n")
        perf_text.insert(tk.END, f"🎛️ Total Features Used: {insights.get('total_features', 0)}\n")
        perf_text.insert(tk.END, f"🧠 SHAP Available: {'Yes' if insights.get('shap_available', False) else 'No'}\n")
        
        # Feature Importance
        feat_text.insert(tk.END, "🎯 FEATURE IMPORTANCE ANALYSIS\n", "header")
        feat_text.insert(tk.END, "="*50 + "\n\n")
        
        if 'feature_importance' in insights and 'aggregated' in insights['feature_importance']:
            feat_text.insert(tk.END, "Top 20 Most Important Features:\n", "subheader")
            feat_text.insert(tk.END, "(Aggregated across all XGBoost models)\n\n")
            
            importance_items = list(insights['feature_importance']['aggregated'].items())[:20]
            for i, (feature, importance) in enumerate(importance_items, 1):
                feat_text.insert(tk.END, f"{i:2d}. {feature:<30} {importance:.4f}\n", "mono")
        
        feat_text.insert(tk.END, "\n\n🔍 Feature Categories:\n", "subheader")
        feat_text.insert(tk.END, "• Base Nutritional: Original nutrient values\n")
        feat_text.insert(tk.END, "• Interaction: Feature combinations (e.g., protein × fiber)\n")
        feat_text.insert(tk.END, "• Ratio: Derived ratios (e.g., sodium/potassium)\n")
        feat_text.insert(tk.END, "• Composite: Health-specific composite scores\n")
        feat_text.insert(tk.END, "• Density: Nutrient per calorie calculations\n")
        
        # Hyperparameters
        hyper_text.insert(tk.END, "⚙️ HYPERPARAMETER OPTIMIZATION RESULTS\n", "header")
        hyper_text.insert(tk.END, "="*55 + "\n\n")
        
        if 'hyperparameter_tuning' in insights:
            if insights['hyperparameter_tuning']:
                for model_name, tuning_results in insights['hyperparameter_tuning'].items():
                    hyper_text.insert(tk.END, f"🎛️ {model_name} Optimization:\n", "subheader")
                    hyper_text.insert(tk.END, f"• Best Score: {tuning_results.get('best_score', 0):.4f}\n")
                    hyper_text.insert(tk.END, f"• Optimization Calls: {tuning_results.get('n_calls', 0)}\n")
                    
                    if 'best_params' in tuning_results:
                        hyper_text.insert(tk.END, "• Best Parameters:\n")
                        for param, value in tuning_results['best_params'].items():
                            hyper_text.insert(tk.END, f"  - {param}: {value}\n", "mono")
                    hyper_text.insert(tk.END, "\n")
            else:
                hyper_text.insert(tk.END, "Using default hyperparameters.\n")
                hyper_text.insert(tk.END, "Install scikit-optimize for automatic tuning.\n\n")
        
        hyper_text.insert(tk.END, "🔬 XGBoost Algorithm Details:\n", "subheader")
        hyper_text.insert(tk.END, "• Gradient Boosting Framework\n")
        hyper_text.insert(tk.END, "• Tree-based learners with regularization\n")
        hyper_text.insert(tk.END, "• L1 and L2 regularization (alpha & lambda)\n")
        hyper_text.insert(tk.END, "• Early stopping to prevent overfitting\n")
        hyper_text.insert(tk.END, "• Feature subsampling for robustness\n")
        hyper_text.insert(tk.END, "• Cross-validation for model selection\n")
        
        # Configure text tags for all tabs
        for text_widget in [perf_text, feat_text, hyper_text]:
            text_widget.tag_configure("header", font=self.fonts['heading'], 
                                    foreground=self.colors['primary'])
            text_widget.tag_configure("subheader", font=self.fonts['subheading'], 
                                    foreground=self.colors['secondary'])
            text_widget.tag_configure("mono", font=self.fonts['code'])
    
    def update_charts(self, recommendations):
        """Update comprehensive visualization charts"""
        # Clear previous charts
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.clear()
        
        if not recommendations:
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
            self.canvas.draw()
            return
        
        try:
            # Chart 1: XGBoost Score Distribution
            xgb_scores = [r['xgboost_score'] for r in recommendations]
            self.ax1.hist(xgb_scores, bins=6, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            self.ax1.set_title('XGBoost Score Distribution', fontsize=11, fontweight='bold')
            self.ax1.set_xlabel('XGBoost Score')
            self.ax1.set_ylabel('Count')
            
            # Chart 2: Final Score vs XGBoost Score
            final_scores = [r['final_score'] for r in recommendations]
            self.ax2.scatter(xgb_scores, final_scores, color=self.colors['accent'], alpha=0.6, s=50)
            self.ax2.set_title('Final vs XGBoost Scores', fontsize=11, fontweight='bold')
            self.ax2.set_xlabel('XGBoost Score')
            self.ax2.set_ylabel('Final Score')
            
            # Add trend line
            if len(xgb_scores) > 1:
                z = np.polyfit(xgb_scores, final_scores, 1)
                p = np.poly1d(z)
                self.ax2.plot(sorted(xgb_scores), p(sorted(xgb_scores)), 
                             color=self.colors['danger'], linestyle='--', alpha=0.8)
            
            # Chart 3: Macronutrient Distribution
            avg_protein = np.mean([r['protein'] for r in recommendations])
            avg_carbs = np.mean([r['carbs'] for r in recommendations])
            avg_fat = np.mean([r['fat'] for r in recommendations])
            
            labels = ['Protein', 'Carbs', 'Fat']
            sizes = [avg_protein * 4, avg_carbs * 4, avg_fat * 9]
            colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]
            
            if sum(sizes) > 0:
                wedges, texts, autotexts = self.ax3.pie(sizes, labels=labels, colors=colors, 
                                                       autopct='%1.1f%%', startangle=90)
                self.ax3.set_title('Avg Macronutrient Dist.', fontsize=11, fontweight='bold')
                for autotext in autotexts:
                    autotext.set_fontsize(9)
            
            # Chart 4: Category Performance
            categories = {}
            category_scores = {}
            
            for r in recommendations:
                cat = r['category']
                if cat not in categories:
                    categories[cat] = 0
                    category_scores[cat] = []
                categories[cat] += 1
                category_scores[cat].append(r['final_score'])
            
            if categories:
                cats = list(categories.keys())
                avg_scores = [np.mean(category_scores[cat]) for cat in cats]
                
                bars = self.ax4.bar(range(len(cats)), avg_scores, 
                                   color=self.colors['secondary'], alpha=0.7)
                self.ax4.set_title('Category Performance', fontsize=11, fontweight='bold')
                self.ax4.set_ylabel('Avg Final Score')
                self.ax4.set_xticks(range(len(cats)))
                self.ax4.set_xticklabels([cat[:8] for cat in cats], rotation=45, fontsize=9)
                
                # Add score labels on bars
                for bar, score in zip(bars, avg_scores):
                    height = bar.get_height()
                    self.ax4.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                                 f'{score:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Chart 5: Health Condition Suitability Matrix
            health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
            condition_matrix = np.zeros((len(health_conditions), len(recommendations)))
            
            for j, rec in enumerate(recommendations):
                for i, condition in enumerate(health_conditions):
                    if condition in rec['suitable_for_conditions']:
                        condition_matrix[i, j] = 1
            
            if len(recommendations) > 0:
                im = self.ax5.imshow(condition_matrix, cmap='RdYlGn', aspect='auto', alpha=0.8)
                self.ax5.set_title('Health Suitability Matrix', fontsize=11, fontweight='bold')
                self.ax5.set_ylabel('Health Conditions')
                self.ax5.set_xlabel('Recommended Foods')
                self.ax5.set_yticks(range(len(health_conditions)))
                self.ax5.set_yticklabels([cond.replace('_', ' ') for cond in health_conditions], 
                                        fontsize=9)
                
                # Show only subset of x-labels to avoid crowding
                if len(recommendations) <= 10:
                    self.ax5.set_xticks(range(len(recommendations)))
                    self.ax5.set_xticklabels(range(1, len(recommendations) + 1), fontsize=8)
                else:
                    self.ax5.set_xticks(range(0, len(recommendations), max(1, len(recommendations)//5)))
                    self.ax5.set_xticklabels(range(1, len(recommendations) + 1, 
                                                  max(1, len(recommendations)//5)), fontsize=8)
            
        except Exception as e:
            print(f"Error updating charts: {e}")
        
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_stats_display(self):
        """Update statistics display"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
            
            # Create stats display
            total_items = stats.get('total_items', 0)
            num_categories = len(stats.get('categories', {}))
            loading_time = stats.get('loading_time', 0)
            
            stats_text = f"📊 Loaded: {total_items} foods | Categories: {num_categories} | Time: {loading_time:.1f}s"
            
            # Add XGBoost specific stats
            if 'model_performance' in stats and stats['model_performance']:
                num_models = len(stats['model_performance'])
                stats_text += f" | XGBoost Models: {num_models}"
                
                # Show best performing model
                best_r2 = 0
                best_model = None
                for model_name, metrics in stats['model_performance'].items():
                    r2 = metrics.get('r2_score', 0)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
                
                if best_model:
                    stats_text += f" | Best R²: {best_r2:.3f} ({best_model})"
            
            # SHAP status
            if SHAP_AVAILABLE:
                stats_text += " | SHAP: ✅"
            else:
                stats_text += " | SHAP: ❌"
            
            ttk.Label(self.stats_frame, text=stats_text, font=self.fonts['caption'],
                     foreground=self.colors['text_light']).pack()
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def update_model_insights(self):
        """Update model insights display"""
        try:
            insights = self.recommender.get_model_insights()
            
            # Update performance text
            self.performance_text.delete(1.0, tk.END)
            self.performance_text.insert(tk.END, "🚀 XGBoost Performance Summary\n", "header")
            self.performance_text.insert(tk.END, "="*35 + "\n\n")
            
            if 'model_performance' in insights:
                for model_name, metrics in insights['model_performance'].items():
                    short_name = model_name.replace('_Suitability', '').replace('_', ' ')
                    self.performance_text.insert(tk.END, f"📊 {short_name}:\n", "subheader")
                    self.performance_text.insert(tk.END, f"R²: {metrics.get('r2_score', 0):.3f}\n")
                    self.performance_text.insert(tk.END, f"CV: {metrics.get('cv_mean', 0):.3f}\n\n")
            
            # Update features text
            self.features_text.delete(1.0, tk.END)
            self.features_text.insert(tk.END, "🎯 Top Features\n", "header")
            self.features_text.insert(tk.END, "="*20 + "\n\n")
            
            if 'feature_importance' in insights and 'aggregated' in insights['feature_importance']:
                importance_items = list(insights['feature_importance']['aggregated'].items())[:15]
                for i, (feature, importance) in enumerate(importance_items, 1):
                    # Shorten feature names for display
                    short_feature = feature[:20] + "..." if len(feature) > 20 else feature
                    self.features_text.insert(tk.END, f"{i:2d}. {short_feature}\n")
                    self.features_text.insert(tk.END, f"    {importance:.4f}\n\n")
            
            # Configure text tags
            for text_widget in [self.performance_text, self.features_text]:
                text_widget.tag_configure("header", font=self.fonts['subheading'], 
                                        foreground=self.colors['primary'])
                text_widget.tag_configure("subheader", font=self.fonts['body'], 
                                        foreground=self.colors['secondary'])
            
        except Exception as e:
            print(f"Error updating model insights: {e}")
    
    def reset_form(self):
        """Reset all form inputs to defaults"""
        try:
            # Reset form variables
            self.weight_var.set(70.0)
            self.height_var.set(170.0)
            self.age_var.set(30)
            self.gender_var.set("Male")
            self.activity_var.set("Moderate")
            self.weight_goal_var.set("Maintain Weight")
            self.meal_type_var.set("Lunch")
            self.category_var.set("All")
            self.max_results_var.set("10")
            self.model_weight_var.set(0.6)
            
            # Reset health conditions
            self.diabetes_var.set(False)
            self.obesity_var.set(False)
            self.hypertension_var.set(False)
            self.cholesterol_var.set(False)
            
            # Clear results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Clear text widgets
            self.targets_text.delete(1.0, tk.END)
            self.targets_text.insert(tk.END, "Click 'Calculate Nutrition' to see personalized targets based on medical guidelines and XGBoost feature engineering.")
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information, XGBoost predictions, and SHAP interpretability analysis.")
            
            # Update charts
            self.update_charts([])
            
            # Reset data
            self.current_nutritional_data = None
            self.last_recommendations = []
            
            self.status_var.set("Form reset to defaults - XGBoost models ready")
            
        except Exception as e:
            print(f"Error during reset: {e}")
            self.status_var.set("Reset completed with minor issues")
        
    def show_diagnostics(self):
        try:
            """Show comprehensive system diagnostics"""
            diag_window = tk.Toplevel(self.master)
            diag_window.title("XGBoost System Diagnostics")
            diag_window.geometry("900x700")
            diag_window.configure(bg=self.colors['background'])
            
            # Create scrollable text
            text_frame = ttk.Frame(diag_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            diag_text = tk.Text(text_frame, wrap=tk.WORD, font=self.fonts['code'])
            diag_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=diag_text.yview)
            diag_text.configure(yscrollcommand=diag_scroll.set)
            
            diag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            diag_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Run diagnostics
            diag_text.insert(tk.END, "🔍 XGBOOST SYSTEM DIAGNOSTICS\n")
            diag_text.insert(tk.END, "="*60 + "\n\n")
            
            # 1. Library checks
            diag_text.insert(tk.END, "📚 LIBRARY STATUS:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            libraries = {
                'xgboost': 'XGBoost',
                'sklearn': 'Scikit-learn',
                'pandas': 'Pandas',
                'numpy': 'NumPy',
                'matplotlib': 'Matplotlib',
                'seaborn': 'Seaborn',
                'shap': 'SHAP (Optional)',
                'skopt': 'Scikit-optimize (Optional)'
            }
            
            for lib, name in libraries.items():
                try:
                    if lib == 'sklearn':
                        import sklearn
                        version = sklearn.__version__
                    elif lib == 'skopt':
                        import skopt
                        version = skopt.__version__
                    else:
                        module = __import__(lib)
                        version = module.__version__
                    diag_text.insert(tk.END, f"✅ {name}: v{version}\n")
                except ImportError:
                    optional = "(Optional)" in name
                    status = "⚠️" if optional else "❌"
                    diag_text.insert(tk.END, f"{status} {name}: Not installed\n")
            
            # 2. Data status
            diag_text.insert(tk.END, f"\n📊 DATA STATUS:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            if hasattr(self.recommender, 'food_data'):
                data_shape = self.recommender.food_data.shape
                diag_text.insert(tk.END, f"✅ Food data loaded: {data_shape[0]} items, {data_shape[1]} columns\n")
                
                # Check categories
                if 'Category' in self.recommender.food_data.columns:
                    categories = self.recommender.food_data['Category'].value_counts()
                    diag_text.insert(tk.END, f"📂 Categories ({len(categories)}):\n")
                    for cat, count in categories.items():
                        diag_text.insert(tk.END, f"   • {cat}: {count} items\n")
                
                # Check key nutritional columns
                key_cols = ['Energy(kcal) by calculation', 'Protein(g)', 'CHOCDF (g) Carbohydrate', 'SUGAR(g)']
                missing_cols = [col for col in key_cols if col not in self.recommender.food_data.columns]
                if missing_cols:
                    diag_text.insert(tk.END, f"❌ Missing key columns: {missing_cols}\n")
                else:
                    diag_text.insert(tk.END, f"✅ All key nutritional columns present\n")
            else:
                diag_text.insert(tk.END, "❌ No food data loaded\n")
            
            # 3. File system check
            diag_text.insert(tk.END, f"\n📁 FILE SYSTEM:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            datasets_path = os.path.abspath('./datasets')
            diag_text.insert(tk.END, f"📍 Looking for datasets in: {datasets_path}\n")
            
            if os.path.exists(datasets_path):
                csv_files = glob.glob(os.path.join(datasets_path, '*.csv'))
                diag_text.insert(tk.END, f"✅ Datasets folder exists\n")
                diag_text.insert(tk.END, f"📄 CSV files found ({len(csv_files)}):\n")
                for file_path in csv_files:
                    filename = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path)
                    diag_text.insert(tk.END, f"   • {filename}: {file_size:,} bytes\n")
            else:
                diag_text.insert(tk.END, f"❌ Datasets folder not found\n")
                diag_text.insert(tk.END, f"💡 Create folder: {datasets_path}\n")
            
            # 4. Model status
            diag_text.insert(tk.END, f"\n🤖 MODEL STATUS:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            if hasattr(self.recommender, 'models') and self.recommender.models:
                diag_text.insert(tk.END, f"✅ Models trained: {len(self.recommender.models)}\n")
                for model_name in self.recommender.models.keys():
                    diag_text.insert(tk.END, f"   • {model_name}\n")
                
                if hasattr(self.recommender, 'stats') and 'model_performance' in self.recommender.stats:
                    diag_text.insert(tk.END, f"\n📈 Model Performance:\n")
                    for model_name, metrics in self.recommender.stats['model_performance'].items():
                        r2 = metrics.get('r2_score', 0)
                        samples = metrics.get('training_samples', 0)
                        diag_text.insert(tk.END, f"   • {model_name}: R²={r2:.3f}, samples={samples}\n")
            else:
                diag_text.insert(tk.END, "❌ No models trained\n")
            
            # 5. Features status
            diag_text.insert(tk.END, f"\n🎯 FEATURE ENGINEERING:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            if hasattr(self.recommender, 'features'):
                diag_text.insert(tk.END, f"✅ Features prepared: {len(self.recommender.features)}\n")
                
                # Count feature types
                base_features = [f for f in self.recommender.features if not any(x in f for x in ['_x_', '_ratio', '_Score', '_score', '_per_calorie'])]
                interaction_features = [f for f in self.recommender.features if '_x_' in f]
                ratio_features = [f for f in self.recommender.features if '_ratio' in f]
                score_features = [f for f in self.recommender.features if '_Score' in f or '_score' in f]
                density_features = [f for f in self.recommender.features if '_per_calorie' in f]
                
                diag_text.insert(tk.END, f"   • Base nutritional: {len(base_features)}\n")
                diag_text.insert(tk.END, f"   • Interaction features: {len(interaction_features)}\n")
                diag_text.insert(tk.END, f"   • Ratio features: {len(ratio_features)}\n")
                diag_text.insert(tk.END, f"   • Health scores: {len(score_features)}\n")
                diag_text.insert(tk.END, f"   • Density features: {len(density_features)}\n")
            else:
                diag_text.insert(tk.END, "❌ No features prepared\n")
            
            # 6. Configuration
            diag_text.insert(tk.END, f"\n⚙️ CONFIGURATION:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            diag_text.insert(tk.END, f"🧠 SHAP interpretability: {'✅ Available' if SHAP_AVAILABLE else '❌ Not available'}\n")
            diag_text.insert(tk.END, f"🔧 Hyperparameter optimization: {'✅ Available' if HYPEROPT_AVAILABLE else '❌ Not available'}\n")
            
            # 7. Recommendations
            diag_text.insert(tk.END, f"\n💡 RECOMMENDATIONS:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            
            if not SHAP_AVAILABLE:
                diag_text.insert(tk.END, "📦 Install SHAP for model interpretability:\n")
                diag_text.insert(tk.END, "   pip install shap\n")
            
            if not HYPEROPT_AVAILABLE:
                diag_text.insert(tk.END, "📦 Install scikit-optimize for better hyperparameter tuning:\n")
                diag_text.insert(tk.END, "   pip install scikit-optimize\n")
            
            if not hasattr(self.recommender, 'food_data') or self.recommender.food_data.empty:
                diag_text.insert(tk.END, "📁 Ensure CSV files are in ./datasets/ folder\n")
                diag_text.insert(tk.END, "📋 Check CSV format matches expected columns\n")
            
            if hasattr(self.recommender, 'models') and not self.recommender.models:
                diag_text.insert(tk.END, "🔄 Try restarting the application\n")
                diag_text.insert(tk.END, "📊 Check that food data has sufficient variety\n")
            
            # 8. Common CSV format
            diag_text.insert(tk.END, f"\n📄 EXPECTED CSV FORMAT:\n")
            diag_text.insert(tk.END, "-"*30 + "\n")
            expected_cols = [
                "Food_Code", "Thai_Name", "English_Name", 
                "Energy(kcal) by calculation", "Protein(g)", 
                "CHOCDF (g) Carbohydrate", "SUGAR(g)", 
                "FIBTG (g) Dietary fibre", "Fat(g)", 
                "Na(mg)", "K(mg)", "Ca(mg)", 
                "CHOLE(mg) Cholesterol", "FASAT (g) Saturated FA"
            ]
            
            for col in expected_cols:
                diag_text.insert(tk.END, f"   • {col}\n")
            
        except Exception as e:
            print(f"Error during reset: {e}")
            self.status_var.set("Reset completed with minor issues")
    
    def show_diagnostics(self):
        """Show comprehensive system diagnostics"""
        diag_window = tk.Toplevel(self.master)
        diag_window.title("XGBoost System Diagnostics")
        diag_window.geometry("900x700")
        diag_window.configure(bg=self.colors['background'])
        
        # Create scrollable text
        text_frame = ttk.Frame(diag_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        diag_text = tk.Text(text_frame, wrap=tk.WORD, font=self.fonts['code'])
        diag_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=diag_text.yview)
        diag_text.configure(yscrollcommand=diag_scroll.set)
        
        diag_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        diag_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Run diagnostics
        diag_text.insert(tk.END, "🔍 XGBOOST SYSTEM DIAGNOSTICS\n")
        diag_text.insert(tk.END, "="*60 + "\n\n")
        
        # 1. Library checks
        diag_text.insert(tk.END, "📚 LIBRARY STATUS:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        libraries = {
            'xgboost': 'XGBoost',
            'sklearn': 'Scikit-learn',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'matplotlib': 'Matplotlib',
            'seaborn': 'Seaborn',
            'shap': 'SHAP (Optional)',
            'skopt': 'Scikit-optimize (Optional)'
        }
        
        for lib, name in libraries.items():
            try:
                if lib == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif lib == 'skopt':
                    import skopt
                    version = skopt.__version__
                else:
                    module = __import__(lib)
                    version = module.__version__
                diag_text.insert(tk.END, f"✅ {name}: v{version}\n")
            except ImportError:
                optional = "(Optional)" in name
                status = "⚠️" if optional else "❌"
                diag_text.insert(tk.END, f"{status} {name}: Not installed\n")
        
        # 2. Data status
        diag_text.insert(tk.END, f"\n📊 DATA STATUS:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        if hasattr(self.recommender, 'food_data'):
            data_shape = self.recommender.food_data.shape
            diag_text.insert(tk.END, f"✅ Food data loaded: {data_shape[0]} items, {data_shape[1]} columns\n")
            
            # Check categories
            if 'Category' in self.recommender.food_data.columns:
                categories = self.recommender.food_data['Category'].value_counts()
                diag_text.insert(tk.END, f"📂 Categories ({len(categories)}):\n")
                for cat, count in categories.items():
                    diag_text.insert(tk.END, f"   • {cat}: {count} items\n")
            
            # Check key nutritional columns
            key_cols = ['Energy(kcal) by calculation', 'Protein(g)', 'CHOCDF (g) Carbohydrate', 'SUGAR(g)']
            missing_cols = [col for col in key_cols if col not in self.recommender.food_data.columns]
            if missing_cols:
                diag_text.insert(tk.END, f"❌ Missing key columns: {missing_cols}\n")
            else:
                diag_text.insert(tk.END, f"✅ All key nutritional columns present\n")
        else:
            diag_text.insert(tk.END, "❌ No food data loaded\n")
        
        # 3. File system check
        diag_text.insert(tk.END, f"\n📁 FILE SYSTEM:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        datasets_path = os.path.abspath('./datasets')
        diag_text.insert(tk.END, f"📍 Looking for datasets in: {datasets_path}\n")
        
        if os.path.exists(datasets_path):
            csv_files = glob.glob(os.path.join(datasets_path, '*.csv'))
            diag_text.insert(tk.END, f"✅ Datasets folder exists\n")
            diag_text.insert(tk.END, f"📄 CSV files found ({len(csv_files)}):\n")
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                diag_text.insert(tk.END, f"   • {filename}: {file_size:,} bytes\n")
        else:
            diag_text.insert(tk.END, f"❌ Datasets folder not found\n")
            diag_text.insert(tk.END, f"💡 Create folder: {datasets_path}\n")
        
        # 4. Model status
        diag_text.insert(tk.END, f"\n🤖 MODEL STATUS:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        if hasattr(self.recommender, 'models') and self.recommender.models:
            diag_text.insert(tk.END, f"✅ Models trained: {len(self.recommender.models)}\n")
            for model_name in self.recommender.models.keys():
                diag_text.insert(tk.END, f"   • {model_name}\n")
            
            if hasattr(self.recommender, 'stats') and 'model_performance' in self.recommender.stats:
                diag_text.insert(tk.END, f"\n📈 Model Performance:\n")
                for model_name, metrics in self.recommender.stats['model_performance'].items():
                    r2 = metrics.get('r2_score', 0)
                    samples = metrics.get('training_samples', 0)
                    diag_text.insert(tk.END, f"   • {model_name}: R²={r2:.3f}, samples={samples}\n")
        else:
            diag_text.insert(tk.END, "❌ No models trained\n")
        
        # 5. Features status
        diag_text.insert(tk.END, f"\n🎯 FEATURE ENGINEERING:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        if hasattr(self.recommender, 'features'):
            diag_text.insert(tk.END, f"✅ Features prepared: {len(self.recommender.features)}\n")
            
            # Count feature types
            base_features = [f for f in self.recommender.features if not any(x in f for x in ['_x_', '_ratio', '_Score', '_score', '_per_calorie'])]
            interaction_features = [f for f in self.recommender.features if '_x_' in f]
            ratio_features = [f for f in self.recommender.features if '_ratio' in f]
            score_features = [f for f in self.recommender.features if '_Score' in f or '_score' in f]
            density_features = [f for f in self.recommender.features if '_per_calorie' in f]
            
            diag_text.insert(tk.END, f"   • Base nutritional: {len(base_features)}\n")
            diag_text.insert(tk.END, f"   • Interaction features: {len(interaction_features)}\n")
            diag_text.insert(tk.END, f"   • Ratio features: {len(ratio_features)}\n")
            diag_text.insert(tk.END, f"   • Health scores: {len(score_features)}\n")
            diag_text.insert(tk.END, f"   • Density features: {len(density_features)}\n")
        else:
            diag_text.insert(tk.END, "❌ No features prepared\n")
        
        # 6. Configuration
        diag_text.insert(tk.END, f"\n⚙️ CONFIGURATION:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        diag_text.insert(tk.END, f"🧠 SHAP interpretability: {'✅ Available' if SHAP_AVAILABLE else '❌ Not available'}\n")
        diag_text.insert(tk.END, f"🔧 Hyperparameter optimization: {'✅ Available' if HYPEROPT_AVAILABLE else '❌ Not available'}\n")
        
        # 7. Recommendations
        diag_text.insert(tk.END, f"\n💡 RECOMMENDATIONS:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        
        if not SHAP_AVAILABLE:
            diag_text.insert(tk.END, "📦 Install SHAP for model interpretability:\n")
            diag_text.insert(tk.END, "   pip install shap\n")
        
        if not HYPEROPT_AVAILABLE:
            diag_text.insert(tk.END, "📦 Install scikit-optimize for better hyperparameter tuning:\n")
            diag_text.insert(tk.END, "   pip install scikit-optimize\n")
        
        if not hasattr(self.recommender, 'food_data') or self.recommender.food_data.empty:
            diag_text.insert(tk.END, "📁 Ensure CSV files are in ./datasets/ folder\n")
            diag_text.insert(tk.END, "📋 Check CSV format matches expected columns\n")
        
        if hasattr(self.recommender, 'models') and not self.recommender.models:
            diag_text.insert(tk.END, "🔄 Try restarting the application\n")
            diag_text.insert(tk.END, "📊 Check that food data has sufficient variety\n")
        
        # 8. Common CSV format
        diag_text.insert(tk.END, f"\n📄 EXPECTED CSV FORMAT:\n")
        diag_text.insert(tk.END, "-"*30 + "\n")
        expected_cols = [
            "Food_Code", "Thai_Name", "English_Name", 
            "Energy(kcal) by calculation", "Protein(g)", 
            "CHOCDF (g) Carbohydrate", "SUGAR(g)", 
            "FIBTG (g) Dietary fibre", "Fat(g)", 
            "Na(mg)", "K(mg)", "Ca(mg)", 
            "CHOLE(mg) Cholesterol", "FASAT (g) Saturated FA"
        ]
        
        for col in expected_cols:
            diag_text.insert(tk.END, f"   • {col}\n")
        
        # Close button
        close_button = ttk.Button(diag_window, text="Close", 
                                 command=diag_window.destroy)
        close_button.pack(pady=10)


def main():
    """Main function to run the XGBoost application"""
    print("Starting Health-Driven XGBoost Food Recommendation System...")
    print("Loading advanced gradient boosting models with medical guidelines...")
    
    # Create main window
    root = tk.Tk()
    root.title("Health-Driven XGBoost Food Recommendation System")
    root.geometry("1500x1000")
    
    # Center window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - 750
    y = (screen_height // 2) - 500
    root.geometry(f"1500x1000+{x}+{y}")
    
    # Hide main window initially
    root.withdraw()
    
    # Create enhanced splash screen
    splash = tk.Toplevel(root)
    splash.title("Loading XGBoost System...")
    splash.geometry("600x400")
    splash.resizable(False, False)
    splash.configure(bg="#1a365d")
    
    # Center splash
    splash_x = (screen_width // 2) - 300
    splash_y = (screen_height // 2) - 200
    splash.geometry(f"600x400+{splash_x}+{splash_y}")
    
    # Splash content with modern styling
    splash_frame = tk.Frame(splash, bg="#1a365d", padx=50, pady=50)
    splash_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = tk.Label(splash_frame, 
                          text="Health-Driven XGBoost Food Recommendation System", 
                          font=('Segoe UI', 18, 'bold'),
                          fg="white", bg="#1a365d", wraplength=500)
    title_label.pack(pady=(0, 10))
    
    # Subtitle
    subtitle_label = tk.Label(splash_frame, 
                             text="Advanced Gradient Boosting with Medical Guidelines", 
                             font=('Segoe UI', 14),
                             fg="#90cdf4", bg="#1a365d")
    subtitle_label.pack(pady=(0, 20))
    
    # Institution
    inst_label = tk.Label(splash_frame, 
                         text="Prince of Songkla University", 
                         font=('Segoe UI', 12),
                         fg="#a0a0a0", bg="#1a365d")
    inst_label.pack(pady=(0, 30))
    
    # Features list
    features_frame = tk.Frame(splash_frame, bg="#1a365d")
    features_frame.pack(pady=(0, 20))
    
    features = [
        "🚀 XGBoost Gradient Boosting",
        "🧠 SHAP Interpretability",
        "⚙️ Hyperparameter Optimization",
        "🎯 Advanced Feature Engineering",
        "🏥 Medical Guidelines Integration"
    ]
    
    for feature in features:
        feature_label = tk.Label(features_frame, text=feature,
                               font=('Segoe UI', 11), fg="white", bg="#1a365d")
        feature_label.pack(anchor=tk.W, pady=2)
    
    # Progress bar
    progress_frame = tk.Frame(splash_frame, bg="#1a365d")
    progress_frame.pack(fill=tk.X, pady=(20, 10))
    
    progress = ttk.Progressbar(progress_frame, mode='indeterminate', length=400)
    progress.pack()
    progress.start()
    
    # Status label
    status_label = tk.Label(splash_frame, text="Initializing XGBoost models...",
                           font=('Segoe UI', 10), fg="#90cdf4", bg="#1a365d")
    status_label.pack()
    
    def update_splash_status(message):
        if splash.winfo_exists():
            try:
                status_label.config(text=message)
                splash.update()
            except tk.TclError:
                pass
    
    def initialize_system():
        try:
            # Initialize XGBoost recommender system
            update_splash_status("Loading Thai food database...")
            recommender = HealthAwareXGBoostRecommender(update_splash_status)
            
            update_splash_status("Training XGBoost models with hyperparameter optimization...")
            root.update()
            time.sleep(0.5)
            
            update_splash_status("Initializing SHAP explainers...")
            root.update()
            time.sleep(0.3)
            
            update_splash_status("Building advanced user interface...")
            root.update()
            
            # Create main UI
            app = HealthDrivenXGBoostFoodRecommenderUI(root, recommender)
            
            # Close splash
            progress.stop()
            splash.destroy()
            
            # Show main window
            root.deiconify()
            root.lift()
            root.focus_force()
            
            print("XGBoost Food Recommendation System initialized successfully!")
            print(f"Models trained: {len(recommender.models)}")
            print(f"Features engineered: {len(recommender.features)}")
            print(f"SHAP available: {SHAP_AVAILABLE}")
            print(f"Hyperparameter optimization: {HYPEROPT_AVAILABLE}")
            
        except Exception as e:
            progress.stop()
            splash.destroy()
            root.deiconify()
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize XGBoost system: {str(e)}\n\n"
                               "Please check that all required libraries are installed:\n"
                               "- xgboost\n"
                               "- scikit-learn\n"
                               "- pandas\n"
                               "- numpy\n"
                               "- matplotlib\n"
                               "- seaborn\n"
                               "\nOptional for enhanced features:\n"
                               "- shap (for interpretability)\n"
                               "- scikit-optimize (for hyperparameter tuning)")
            print(f"Error: {e}")
    
    # Start initialization
    root.after(100, initialize_system)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        try:
            if splash.winfo_exists():
                splash.destroy()
        except:
            pass


if __name__ == "__main__":
    main()