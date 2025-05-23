import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

# Safe matplotlib import with backend handling
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI integration
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Matplotlib not available: {e}")
    MATPLOTLIB_AVAILABLE = False

import numpy as np
import pandas as pd

# Safe sklearn imports
try:
    from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Error: Scikit-learn not available: {e}")
    print("Please install scikit-learn: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Safe seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# Check critical dependencies
if not SKLEARN_AVAILABLE:
    print("❌ Critical dependency missing: scikit-learn")
    print("Please install: pip install scikit-learn matplotlib pandas numpy")
    exit(1)


class MedicalNutritionalCalculator:
    """Medical-grade nutritional calculation engine based on established formulas and clinical guidelines"""
    
    def __init__(self):
        # Activity level multipliers for TDEE calculation (Mifflin-St Jeor based)
        self.activity_multipliers = {
            'Sedentary': 1.2,        # Little to no exercise, desk job
            'Light': 1.375,          # Light exercise 1-3 days/week
            'Moderate': 1.55,        # Moderate exercise 3-5 days/week
            'Active': 1.725,         # Heavy exercise 6-7 days/week
            'Very Active': 1.9       # Very heavy physical work or 2x/day training
        }
        
        # Medical guidelines for different health conditions (consolidated from all sources)
        self.medical_guidelines = {
            'Diabetes': {
                'sugar_max_percent': 5,          # <5% of total calories (WHO/ADA)
                'sugar_max_grams': 25,           # Maximum 25g/day (WHO optimal)
                'carb_percent': (40, 50),        # 40-50% of calories (ADA 2023)
                'protein_percent': (20, 35),     # 20-35% of calories (higher for metabolic control)
                'protein_grams_per_kg': (1.0, 1.5),  # 1.0-1.5g per kg (ADA/Joslin)
                'fat_percent': (25, 35),         # 25-35% of calories
                'saturated_fat_percent': 7,      # <7% of calories (ADA)
                'fiber_min': 25,                 # Minimum 25g/day (focus on soluble)
                'fiber_per_1000kcal': 14,        # 14g per 1000 kcal (USDA)
                'sodium_max': 2300,              # <2300mg/day (ADA)
                'cholesterol_max': 200,          # <200mg/day (ADA)
                'gi_preference': 'low'           # Low glycemic index foods
            },
            'Obesity': {
                'sugar_max_percent': 5,          # <5% of total calories (WHO)
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (40, 50),        # 40-50% of calories (lower for weight loss)
                'protein_percent': (25, 35),     # Higher protein for satiety and muscle preservation
                'protein_grams_per_kg': (1.2, 1.6),  # 1.2-1.6g per kg for weight loss
                'fat_percent': (20, 30),         # 20-30% of calories
                'saturated_fat_percent': 7,      # <7% of calories
                'fiber_min': 25,                 # Minimum 25g/day (satiety)
                'fiber_per_1000kcal': 14,        # 14g per 1000 kcal
                'sodium_max': 2300,              # <2300mg/day
                'calorie_deficit': 500,          # 500 kcal deficit for 1lb/week loss
                'energy_density': 'low'          # Low energy density foods
            },
            'Hypertension': {
                'sugar_max_percent': 10,         # <10% of total calories
                'sugar_max_grams': 50,           # Maximum 50g/day
                'carb_percent': (45, 65),        # 45-65% of calories (DASH pattern)
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories (DASH/AHA)
                'fiber_min': 30,                 # Minimum 30g/day (DASH)
                'fiber_per_1000kcal': 14,        # 14g per 1000 kcal
                'sodium_max': 1500,              # <1500mg/day (DASH optimal)
                'potassium_min': 4700,           # Minimum 4700mg/day (DASH)
                'calcium_min': 1200,             # Minimum 1200mg/day (DASH)
                'magnesium_min': 400,            # Minimum 400mg/day (DASH)
                'dash_compliance': True
            },
            'High_Cholesterol': {
                'sugar_max_percent': 10,         # <10% of total calories
                'sugar_max_grams': 50,           # Maximum 50g/day
                'carb_percent': (45, 65),        # 45-65% of calories
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories (AHA strict)
                'fiber_min': 25,                 # Minimum 25g/day
                'soluble_fiber_min': 10,         # Minimum 10g soluble fiber (cholesterol lowering)
                'fiber_per_1000kcal': 14,        # 14g per 1000 kcal
                'sodium_max': 2300,              # <2300mg/day
                'cholesterol_max': 200,          # <200mg/day (strict)
                'plant_sterol_focus': True
            }
        }
        
        # Age and gender specific adjustments (Dietary Reference Intakes)
        self.age_gender_adjustments = {
            'protein_rda': {  # g/kg body weight
                ('male', 19, 50): 0.8,
                ('male', 51, 70): 0.8,
                ('male', 71, 120): 1.0,
                ('female', 19, 50): 0.8,
                ('female', 51, 70): 0.8,
                ('female', 71, 120): 1.0
            },
            'fiber_ai': {  # Adequate Intake g/day
                ('male', 19, 50): 38,
                ('male', 51, 120): 30,
                ('female', 19, 50): 25,
                ('female', 51, 120): 21
            },
            'calcium_rda': {  # mg/day
                ('male', 19, 50): 1000,
                ('male', 51, 70): 1000,
                ('male', 71, 120): 1200,
                ('female', 19, 50): 1000,
                ('female', 51, 120): 1200
            }
        }
    
    def calculate_bmr(self, weight_kg, height_cm, age_years, gender):
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation (most accurate)"""
        if gender.lower() == 'male':
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
        else:  # female
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
        return bmr
    
    def calculate_bmi(self, weight_kg, height_cm):
        """Calculate Body Mass Index"""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m * height_m)
        
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        return bmi, category
    
    def calculate_tdee(self, bmr, activity_level):
        """Calculate Total Daily Energy Expenditure"""
        multiplier = self.activity_multipliers.get(activity_level, 1.55)
        return bmr * multiplier
    
    def calculate_target_calories(self, tdee, weight_goal, health_conditions, bmi):
        """Calculate target calories based on weight goals and health conditions"""
        target_calories = tdee
        
        if weight_goal == 'Lose Weight':
            # Create appropriate deficit based on BMI and health conditions
            if 'Obesity' in health_conditions or bmi >= 30:
                target_calories = tdee - self.medical_guidelines['Obesity']['calorie_deficit']
            else:
                target_calories = tdee - 300  # Moderate deficit for overweight
        elif weight_goal == 'Gain Weight':
            target_calories = tdee + 300  # Moderate surplus
        # 'Maintain Weight' keeps target_calories = tdee
        
        # Ensure minimum safe calories (never below 1200 for women, 1500 for men)
        min_calories = 1200  # Will be adjusted based on gender in the full calculation
        return max(min_calories, target_calories)
    
    def get_age_gender_specific_values(self, age, gender, nutrient_type):
        """Get age and gender specific nutritional values"""
        for key, value in self.age_gender_adjustments[nutrient_type].items():
            sex, min_age, max_age = key
            if sex == gender.lower() and min_age <= age <= max_age:
                return value
        
        # Default fallback values
        defaults = {
            'protein_rda': 0.8,
            'fiber_ai': 25 if gender.lower() == 'female' else 38,
            'calcium_rda': 1000
        }
        return defaults.get(nutrient_type, 0)
    
    def calculate_nutritional_targets(self, user_profile):
        """Calculate personalized nutritional targets based on medical guidelines and DRI"""
        # Extract user profile data
        weight_kg = user_profile['weight']
        height_cm = user_profile['height']
        age_years = user_profile['age']
        gender = user_profile['gender']
        activity_level = user_profile['activity_level']
        weight_goal = user_profile.get('weight_goal', 'Maintain Weight')
        health_conditions = user_profile.get('health_conditions', [])
        
        # Calculate basic metabolic parameters
        bmr = self.calculate_bmr(weight_kg, height_cm, age_years, gender)
        bmi, bmi_category = self.calculate_bmi(weight_kg, height_cm)
        tdee = self.calculate_tdee(bmr, activity_level)
        target_calories = self.calculate_target_calories(tdee, weight_goal, health_conditions, bmi)
        
        # Adjust minimum calories based on gender
        min_calories = 1200 if gender.lower() == 'female' else 1500
        target_calories = max(min_calories, target_calories)
        
        # Get age and gender specific requirements
        protein_rda_per_kg = self.get_age_gender_specific_values(age_years, gender, 'protein_rda')
        fiber_ai = self.get_age_gender_specific_values(age_years, gender, 'fiber_ai')
        calcium_rda = self.get_age_gender_specific_values(age_years, gender, 'calcium_rda')
        
        # Initialize targets with healthy adult guidelines (no health conditions)
        targets = {
            'calories': target_calories,
            'carbs_min': (target_calories * 0.45) / 4,      # 45% of calories
            'carbs_max': (target_calories * 0.60) / 4,      # 60% of calories
            'protein_min': max((target_calories * 0.15) / 4, weight_kg * protein_rda_per_kg),  # 15% or RDA
            'protein_max': (target_calories * 0.25) / 4,    # 25% of calories
            'fat_min': (target_calories * 0.25) / 9,        # 25% of calories
            'fat_max': (target_calories * 0.35) / 9,        # 35% of calories
            'sugar_max': (target_calories * 0.10) / 4,      # 10% of calories (general)
            'fiber_min': fiber_ai,                          # Age/gender specific AI
            'sodium_max': 2300,                             # General recommendation (mg)
            'saturated_fat_max': (target_calories * 0.10) / 9,  # 10% of calories
            'cholesterol_max': 300,                         # General recommendation (mg)
            'potassium_min': 3500,                          # General recommendation (mg)
            'calcium_min': calcium_rda,                     # Age/gender specific RDA
        }
        
        # Apply health condition modifications
        if health_conditions:
            targets = self._apply_health_condition_modifications(
                targets, health_conditions, target_calories, weight_kg, age_years, gender
            )
        
        # Calculate meal-specific targets
        meal_targets = self._calculate_meal_targets(targets)
        
        # Create detailed calculations explanation
        calculations = {
            'bmr_formula': f"BMR = 10×{weight_kg} + 6.25×{height_cm} - 5×{age_years} {'+ 5' if gender.lower() == 'male' else '- 161'} = {bmr:.0f} kcal",
            'bmi_calculation': f"BMI = {weight_kg}kg ÷ ({height_cm/100}m)² = {bmi:.1f} ({bmi_category})",
            'tdee_formula': f"TDEE = BMR × {self.activity_multipliers.get(activity_level, 1.55)} = {tdee:.0f} kcal",
            'target_formula': f"Target = TDEE {'+' if weight_goal == 'Gain Weight' else '-' if weight_goal == 'Lose Weight' else '='} {abs(target_calories - tdee):.0f} = {target_calories:.0f} kcal",
            'protein_calculation': f"Protein = max(15% calories, {protein_rda_per_kg}g/kg × {weight_kg}kg) = {targets['protein_min']:.0f}g",
            'fiber_calculation': f"Fiber = {fiber_ai}g/day (AI for {gender.lower()}, age {age_years})"
        }
        
        return {
            'daily_targets': targets,
            'meal_targets': meal_targets,
            'bmr': bmr,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'tdee': tdee,
            'calculations': calculations,
            'health_modifications': self._get_health_modifications_summary(health_conditions),
            'medical_rationale': self._generate_medical_rationale(health_conditions, targets)
        }
    
    def _apply_health_condition_modifications(self, targets, health_conditions, target_calories, weight_kg, age, gender):
        """Apply medical guidelines for specific health conditions"""
        
        modifications_applied = []
        
        for condition in health_conditions:
            if condition in self.medical_guidelines:
                guidelines = self.medical_guidelines[condition]
                
                # Sugar modifications (most restrictive wins)
                if 'sugar_max_percent' in guidelines:
                    sugar_from_percent = (target_calories * guidelines['sugar_max_percent'] / 100) / 4
                    sugar_from_grams = guidelines.get('sugar_max_grams', float('inf'))
                    new_sugar_max = min(sugar_from_percent, sugar_from_grams)
                    if new_sugar_max < targets['sugar_max']:
                        targets['sugar_max'] = new_sugar_max
                        modifications_applied.append(f"{condition}: Sugar limited to {new_sugar_max:.0f}g")
                
                # Carbohydrate modifications
                if 'carb_percent' in guidelines:
                    carb_min, carb_max = guidelines['carb_percent']
                    new_carb_min = (target_calories * carb_min / 100) / 4
                    new_carb_max = (target_calories * carb_max / 100) / 4
                    
                    # Apply more restrictive limits
                    if new_carb_min > targets['carbs_min']:
                        targets['carbs_min'] = new_carb_min
                    if new_carb_max < targets['carbs_max']:
                        targets['carbs_max'] = new_carb_max
                        modifications_applied.append(f"{condition}: Carbs adjusted to {carb_min}-{carb_max}%")
                
                # Protein modifications
                if 'protein_percent' in guidelines:
                    protein_min, protein_max = guidelines['protein_percent']
                    protein_from_percent_min = (target_calories * protein_min / 100) / 4
                    protein_from_percent_max = (target_calories * protein_max / 100) / 4
                    
                    # Use higher of percentage or body weight calculation
                    if 'protein_grams_per_kg' in guidelines:
                        protein_min_kg, protein_max_kg = guidelines['protein_grams_per_kg']
                        protein_from_weight_min = weight_kg * protein_min_kg
                        protein_from_weight_max = weight_kg * protein_max_kg
                        
                        final_protein_min = max(protein_from_percent_min, protein_from_weight_min)
                        final_protein_max = max(protein_from_percent_max, protein_from_weight_max)
                    else:
                        final_protein_min = protein_from_percent_min
                        final_protein_max = protein_from_percent_max
                    
                    if final_protein_min > targets['protein_min']:
                        targets['protein_min'] = final_protein_min
                        modifications_applied.append(f"{condition}: Protein increased to {final_protein_min:.0f}g")
                    if final_protein_max > targets['protein_max']:
                        targets['protein_max'] = final_protein_max
                
                # Fat modifications
                if 'fat_percent' in guidelines:
                    fat_min, fat_max = guidelines['fat_percent']
                    new_fat_min = (target_calories * fat_min / 100) / 9
                    new_fat_max = (target_calories * fat_max / 100) / 9
                    
                    # Apply more restrictive limits
                    if new_fat_min > targets['fat_min']:
                        targets['fat_min'] = new_fat_min
                    if new_fat_max < targets['fat_max']:
                        targets['fat_max'] = new_fat_max
                        modifications_applied.append(f"{condition}: Fat limited to {fat_min}-{fat_max}%")
                
                # Saturated fat modifications (most restrictive wins)
                if 'saturated_fat_percent' in guidelines:
                    new_sat_fat_max = (target_calories * guidelines['saturated_fat_percent'] / 100) / 9
                    if new_sat_fat_max < targets['saturated_fat_max']:
                        targets['saturated_fat_max'] = new_sat_fat_max
                        modifications_applied.append(f"{condition}: Saturated fat limited to {guidelines['saturated_fat_percent']}%")
                
                # Fiber modifications (highest requirement wins)
                if 'fiber_min' in guidelines:
                    if guidelines['fiber_min'] > targets['fiber_min']:
                        targets['fiber_min'] = guidelines['fiber_min']
                        modifications_applied.append(f"{condition}: Fiber increased to {guidelines['fiber_min']}g")
                
                # Sodium modifications (most restrictive wins)
                if 'sodium_max' in guidelines:
                    if guidelines['sodium_max'] < targets['sodium_max']:
                        targets['sodium_max'] = guidelines['sodium_max']
                        modifications_applied.append(f"{condition}: Sodium limited to {guidelines['sodium_max']}mg")
                
                # Cholesterol modifications (most restrictive wins)
                if 'cholesterol_max' in guidelines:
                    if guidelines['cholesterol_max'] < targets['cholesterol_max']:
                        targets['cholesterol_max'] = guidelines['cholesterol_max']
                        modifications_applied.append(f"{condition}: Cholesterol limited to {guidelines['cholesterol_max']}mg")
                
                # Mineral requirements (highest requirement wins)
                if 'potassium_min' in guidelines:
                    if guidelines['potassium_min'] > targets.get('potassium_min', 0):
                        targets['potassium_min'] = guidelines['potassium_min']
                        modifications_applied.append(f"{condition}: Potassium increased to {guidelines['potassium_min']}mg")
                
                if 'calcium_min' in guidelines:
                    if guidelines['calcium_min'] > targets.get('calcium_min', 0):
                        targets['calcium_min'] = guidelines['calcium_min']
        
        targets['modifications_applied'] = modifications_applied
        return targets
    
    def _calculate_meal_targets(self, daily_targets):
        """Calculate targets for individual meals based on typical distribution patterns"""
        # Evidence-based meal distribution for optimal metabolic health
        meal_distribution = {
            'breakfast': 0.25,      # 25% - Important for metabolic kickstart
            'lunch': 0.35,          # 35% - Largest meal for sustained energy
            'dinner': 0.30,         # 30% - Moderate evening meal
            'snack': 0.05           # 5% per snack (2 snacks = 10%)
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
    
    def _get_health_modifications_summary(self, health_conditions):
        """Get summary of health condition modifications"""
        if not health_conditions:
            return "No health conditions - using general healthy adult guidelines"
        
        modifications = []
        for condition in health_conditions:
            if condition == 'Diabetes':
                modifications.append("Diabetes: Reduced sugar (<5% calories), moderate carbs (40-50%), higher protein")
            elif condition == 'Obesity':
                modifications.append("Obesity: Calorie deficit, higher protein (satiety), reduced sugar")
            elif condition == 'Hypertension':
                modifications.append("Hypertension: DASH pattern, reduced sodium (<1500mg), increased potassium")
            elif condition == 'High_Cholesterol':
                modifications.append("High Cholesterol: Reduced saturated fat (<6%), increased soluble fiber")
        
        return " | ".join(modifications)
    
    def _generate_medical_rationale(self, health_conditions, targets):
        """Generate medical rationale for the calculated targets"""
        rationale = []
        
        if not health_conditions:
            rationale.append("Targets based on WHO, USDA Dietary Guidelines, and DRI recommendations for healthy adults.")
        else:
            rationale.append("Targets modified based on clinical guidelines:")
            
            for condition in health_conditions:
                if condition == 'Diabetes':
                    rationale.append("• ADA 2023: <5% sugar for glycemic control, 40-50% carbs from low-GI sources")
                elif condition == 'Obesity':
                    rationale.append("• WHO/USDA: Higher protein (1.2-1.6g/kg) for satiety, 500kcal deficit for 1lb/week loss")
                elif condition == 'Hypertension':
                    rationale.append("• DASH/AHA: <1500mg sodium, increased K/Ca/Mg for blood pressure control")
                elif condition == 'High_Cholesterol':
                    rationale.append("• AHA 2021: <6% saturated fat, increased soluble fiber for cholesterol lowering")
        
        return " ".join(rationale)


class HealthAwareKNNRecommender:
    """Advanced KNN-based food recommender with automatic medical target calculation"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionalCalculator()
        
        # KNN model parameters
        self.knn_models = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'knn_performance': {},
            'feature_importance': {}
        }
        
        # Enhanced nutritional features for KNN
        self.base_nutritional_features = [
            'Energy(kcal) by calculation', 
            'Protein(g)', 
            'CHOCDF (g) Carbohydrate',
            'SUGAR(g)', 
            'FIBTG (g) Dietary fibre', 
            'Fat(g)'
        ]
        
        # Additional features (may not be in all datasets)
        self.extended_features = [
            'Na(mg)',           # Sodium
            'K(mg)',            # Potassium  
            'Ca(mg)',           # Calcium
            'CHOLE(mg) Cholesterol',  # Cholesterol
            'FASAT (g) Saturated FA'  # Saturated fat
        ]
        
        # Health condition feature weights (for medical relevance)
        self.condition_feature_weights = {
            'Diabetes': {
                'SUGAR(g)': 3.0,
                'CHOCDF (g) Carbohydrate': 2.0,
                'FIBTG (g) Dietary fibre': 2.0,
                'Energy(kcal) by calculation': 1.5
            },
            'Obesity': {
                'Energy(kcal) by calculation': 3.0,
                'Fat(g)': 2.0,
                'SUGAR(g)': 2.0,
                'Protein(g)': 1.5,
                'FIBTG (g) Dietary fibre': 1.5
            },
            'Hypertension': {
                'Na(mg)': 3.0,
                'K(mg)': 2.0,
                'FIBTG (g) Dietary fibre': 1.5,
                'Fat(g)': 1.5
            },
            'High_Cholesterol': {
                'FASAT (g) Saturated FA': 3.0,
                'CHOLE(mg) Cholesterol': 2.5,
                'FIBTG (g) Dietary fibre': 2.0,
                'Fat(g)': 1.5
            }
        }
        
        # Load and prepare data
        start_time = time.time()
        self.load_data()
        self.prepare_features()
        self.train_knn_models()
        self.stats['loading_time'] = time.time() - start_time
    
    def update_status(self, message):
        """Update loading status if callback is provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self):
        """Load and combine all Thai food datasets"""
        try:
            # Look for CSV files in current directory or datasets folder
            possible_paths = ['./', './datasets/', '../datasets/']
            csv_files = []
            
            for path in possible_paths:
                if os.path.exists(path):
                    csv_files.extend(glob.glob(os.path.join(path, '*.csv')))
            
            if not csv_files:
                self.update_status("No CSV files found! Please ensure food data files are available.")
                self.food_data = pd.DataFrame()
                return
            
            self.update_status(f"Found {len(csv_files)} CSV files to process...")
            dataframes = []
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    
                    # Skip files that don't look like food data
                    if any(skip_word in filename.lower() for skip_word in ['performance', 'results', 'analysis', 'test']):
                        self.update_status(f"Skipping {filename} (not food data)")
                        continue
                    
                    category = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                    
                    self.update_status(f"Loading {category} data...")
                    
                    df = pd.read_csv(file_path)
                    self.update_status(f"  Raw data: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Skip files that don't contain food data
                    if len(df) < 3:
                        self.update_status(f"  Skipping {filename} - too few rows")
                        continue
                    
                    # Check for nutritional columns
                    has_nutrition = any(col in df.columns for col in self.base_nutritional_features)
                    if not has_nutrition:
                        # Check alternative column names
                        alt_nutrition_cols = ['Energy', 'Protein', 'Carbohydrate', 'Fat', 'Sugar', 'calories', 'protein', 'carbs']
                        has_alt_nutrition = any(any(alt in col for alt in alt_nutrition_cols) for col in df.columns)
                        
                        if not has_alt_nutrition:
                            self.update_status(f"  Skipping {filename} - no nutritional data columns found")
                            print(f"  Available columns: {list(df.columns)}")
                            continue
                    
                    if 'Category' not in df.columns:
                        df['Category'] = category
                    
                    # Clean and standardize data
                    df_clean = self._clean_nutritional_data(df)
                    
                    if len(df_clean) > 0:  # Only add if data remains after cleaning
                        dataframes.append(df_clean)
                        self.update_status(f"✅ Loaded {len(df_clean)} valid items from {filename}")
                    else:
                        self.update_status(f"⚠️ No valid data after cleaning {filename}")
                    
                except Exception as e:
                    self.update_status(f"❌ Error loading {file_path}: {e}")
                    print(f"Detailed error for {file_path}: {e}")
            
            if dataframes:
                self.update_status("Combining all Thai food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                
                # Debug information about combined data
                print(f"\n📊 Combined Dataset Info:")
                print(f"  Total items: {len(self.food_data)}")
                print(f"  Total columns: {len(self.food_data.columns)}")
                print(f"  Available nutritional features:")
                for feature in self.base_nutritional_features:
                    if feature in self.food_data.columns:
                        non_zero = (self.food_data[feature] > 0).sum()
                        print(f"    ✓ {feature}: {non_zero}/{len(self.food_data)} items have data")
                    else:
                        print(f"    ✗ {feature}: Missing")
                
                if 'Category' in self.food_data.columns:
                    self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
                    print(f"  Categories: {list(self.stats['categories'].keys())}")
                
                self.update_status(f"Successfully loaded {len(self.food_data)} Thai food items")
                
                # Calculate health suitability scores
                self._calculate_health_suitability_scores()
                
            else:
                self.update_status("No valid food data files could be loaded")
                self.food_data = pd.DataFrame()
                print("❌ No valid data found. Check CSV file formats and content.")
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            self.food_data = pd.DataFrame()
            print(f"Critical data loading error: {e}")
            import traceback
            traceback.print_exc()
    
    def _clean_nutritional_data(self, df):
        """Clean and standardize nutritional data for KNN"""
        # Standard column mapping for different datasets
        column_mapping = {
            'ENERGY(kcal)': 'Energy(kcal) by calculation',
            'Energy (kcal)': 'Energy(kcal) by calculation',
            'PROCNT(g)': 'Protein(g)',
            'Protein (g)': 'Protein(g)',
            'CHOCDF(g)': 'CHOCDF (g) Carbohydrate',
            'Carbohydrate (g)': 'CHOCDF (g) Carbohydrate',
            'SUGAR(g)': 'SUGAR(g)',
            'Sugar (g)': 'SUGAR(g)',
            'FIBTG(g)': 'FIBTG (g) Dietary fibre',
            'Fiber (g)': 'FIBTG (g) Dietary fibre',
            'FAT(g)': 'Fat(g)',
            'Fat (g)': 'Fat(g)'
        }
        
        # Apply column mapping
        df = df.rename(columns=column_mapping)
        
        # Get all possible nutritional columns
        all_nutritional_columns = self.base_nutritional_features + self.extended_features
        
        # Fill missing values appropriately
        for col in all_nutritional_columns:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Ensure no negative values for nutritional data
                df[col] = df[col].clip(lower=0)
                
                # Replace extremely high values that might be data entry errors
                if col == 'Energy(kcal) by calculation':
                    # Cap calories at 900 kcal per 100g (very high but possible)
                    df[col] = df[col].clip(upper=900)
                elif col in ['Protein(g)', 'CHOCDF (g) Carbohydrate', 'Fat(g)']:
                    # Cap macronutrients at 100g per 100g
                    df[col] = df[col].clip(upper=100)
                elif col == 'SUGAR(g)':
                    # Cap sugar at 100g per 100g
                    df[col] = df[col].clip(upper=100)
                elif col == 'FIBTG (g) Dietary fibre':
                    # Cap fiber at 50g per 100g
                    df[col] = df[col].clip(upper=50)
                elif col == 'Na(mg)':
                    # Cap sodium at 10000mg per 100g
                    df[col] = df[col].clip(upper=10000)
        
        # Remove rows with all zero nutritional values (invalid entries)
        nutrient_cols = [col for col in all_nutritional_columns if col in df.columns]
        if nutrient_cols:
            # Keep rows that have at least some nutritional information
            df = df[df[nutrient_cols].sum(axis=1) > 0]
        
        # Debug information
        if len(df) > 0:
            print(f"  Data quality check: {len(df)} valid rows")
            
            # Check for any remaining problematic values
            for col in self.base_nutritional_features:
                if col in df.columns:
                    col_data = df[col]
                    if col_data.isnull().any():
                        print(f"  Warning: {col} has NaN values")
                    if (col_data == 0).all():
                        print(f"  Warning: {col} is all zeros")
                    if np.isinf(col_data).any():
                        print(f"  Warning: {col} has infinite values")
        
        return df
    
    def _calculate_health_suitability_scores(self):
        """Calculate health suitability scores for each food item using medical guidelines"""
        self.update_status("Calculating health suitability scores using medical guidelines...")
        
        # Initialize score columns
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = 0.0
        
        self.food_data['Overall_Health_Score'] = 0.0
        
        for idx, food in self.food_data.iterrows():
            # Calculate individual condition scores (lower is better)
            diabetes_score = self._calculate_diabetes_suitability(food)
            obesity_score = self._calculate_obesity_suitability(food)
            hypertension_score = self._calculate_hypertension_suitability(food)
            cholesterol_score = self._calculate_cholesterol_suitability(food)
            
            # Store scores
            self.food_data.at[idx, 'Diabetes_Score'] = diabetes_score
            self.food_data.at[idx, 'Obesity_Score'] = obesity_score
            self.food_data.at[idx, 'Hypertension_Score'] = hypertension_score
            self.food_data.at[idx, 'High_Cholesterol_Score'] = cholesterol_score
            
            # Calculate overall health score (weighted average)
            overall_score = (diabetes_score + obesity_score + hypertension_score + cholesterol_score) / 4
            self.food_data.at[idx, 'Overall_Health_Score'] = overall_score
    
    def _calculate_diabetes_suitability(self, food):
        """Calculate diabetes suitability score using ADA 2023 guidelines"""
        score = 0
        
        # Sugar content (major factor - ADA: <5% calories)
        sugar = float(food.get('SUGAR(g)', 0))
        calories = float(food.get('Energy(kcal) by calculation', 0))
        
        if calories > 0:
            sugar_percent = (sugar * 4) / calories * 100
            if sugar_percent > 10:
                score += 3  # Very high sugar
            elif sugar_percent > 5:
                score += 2  # High sugar
            elif sugar_percent > 2:
                score += 1  # Moderate sugar
        
        # Absolute sugar content
        if sugar > 20:
            score += 2
        elif sugar > 10:
            score += 1
        
        # Carbohydrate quality
        carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        
        if carbs > 0:
            fiber_ratio = fiber / carbs
            if fiber_ratio < 0.1:  # Less than 10% fiber
                score += 2
            elif fiber_ratio < 0.2:  # Less than 20% fiber
                score += 1
        
        # High carbs with low fiber (poor glycemic control)
        if carbs > 30 and fiber < 3:
            score += 2
        
        # Fiber bonus (helps with glycemic control)
        if fiber >= 10:
            score -= 1.5
        elif fiber >= 5:
            score -= 1
        elif fiber >= 3:
            score -= 0.5
        
        # Calorie density consideration
        if calories > 400:  # Very calorie dense
            score += 1
        
        return max(0, score)
    
    def _calculate_obesity_suitability(self, food):
        """Calculate obesity suitability score using WHO/USDA guidelines"""
        score = 0
        
        # Calorie density (major factor for weight management)
        calories = float(food.get('Energy(kcal) by calculation', 0))
        if calories > 350:
            score += 3
        elif calories > 250:
            score += 2
        elif calories > 150:
            score += 1
        
        # Fat content
        fat = float(food.get('Fat(g)', 0))
        if fat > 20:
            score += 2
        elif fat > 15:
            score += 1.5
        elif fat > 10:
            score += 1
        
        # Sugar content (contributes to calorie excess)
        sugar = float(food.get('SUGAR(g)', 0))
        if sugar > 15:
            score += 2
        elif sugar > 10:
            score += 1.5
        elif sugar > 5:
            score += 1
        
        # Satiety factors (protein and fiber reduce score)
        protein = float(food.get('Protein(g)', 0))
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        
        # Protein bonus (satiety and muscle preservation)
        if protein >= 20:
            score -= 1.5
        elif protein >= 15:
            score -= 1
        elif protein >= 10:
            score -= 0.5
        
        # Fiber bonus (satiety and calorie displacement)
        if fiber >= 10:
            score -= 1.5
        elif fiber >= 5:
            score -= 1
        elif fiber >= 3:
            score -= 0.5
        
        return max(0, score)
    
    def _calculate_hypertension_suitability(self, food):
        """Calculate hypertension suitability score using DASH guidelines"""
        score = 0
        
        # Sodium content (major factor - DASH: <1500mg/day)
        sodium = float(food.get('Na(mg)', 0))
        if sodium > 500:  # Very high sodium per 100g
            score += 3
        elif sodium > 300:
            score += 2
        elif sodium > 150:
            score += 1
        
        # Potassium content (beneficial for BP - DASH: >4700mg/day)
        potassium = float(food.get('K(mg)', 0))
        if potassium > 400:  # Good potassium source
            score -= 1
        elif potassium > 200:
            score -= 0.5
        
        # Sodium to potassium ratio
        if potassium > 0 and sodium > 0:
            na_k_ratio = sodium / potassium
            if na_k_ratio > 2:  # Poor ratio
                score += 1
            elif na_k_ratio < 0.5:  # Good ratio
                score -= 0.5
        
        # Fiber content (DASH emphasizes fiber)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 5:
            score -= 0.5
        
        # Saturated fat (DASH: <6% calories)
        sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
        calories = float(food.get('Energy(kcal) by calculation', 0))
        
        if calories > 0 and sat_fat > 0:
            sat_fat_percent = (sat_fat * 9) / calories * 100
            if sat_fat_percent > 10:
                score += 1.5
            elif sat_fat_percent > 6:
                score += 1
        
        return max(0, score)
    
    def _calculate_cholesterol_suitability(self, food):
        """Calculate high cholesterol suitability score using AHA 2021 guidelines"""
        score = 0
        
        # Saturated fat content (major factor - AHA: <6% calories)
        sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
        calories = float(food.get('Energy(kcal) by calculation', 0))
        
        if calories > 0 and sat_fat > 0:
            sat_fat_percent = (sat_fat * 9) / calories * 100
            if sat_fat_percent > 10:
                score += 3
            elif sat_fat_percent > 6:
                score += 2
            elif sat_fat_percent > 4:
                score += 1
        
        # Absolute saturated fat content
        if sat_fat > 8:
            score += 2
        elif sat_fat > 5:
            score += 1
        
        # Dietary cholesterol content
        cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
        if cholesterol > 200:
            score += 2
        elif cholesterol > 100:
            score += 1.5
        elif cholesterol > 50:
            score += 1
        
        # Soluble fiber bonus (cholesterol lowering)
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        if fiber >= 10:  # High fiber foods often have soluble fiber
            score -= 1.5
        elif fiber >= 5:
            score -= 1
        elif fiber >= 3:
            score -= 0.5
        
        # Total fat consideration
        total_fat = float(food.get('Fat(g)', 0))
        if total_fat > 25:  # Very high fat
            score += 1
        
        return max(0, score)
    
    def prepare_features(self):
        """Prepare features for KNN models with health condition awareness"""
        if len(self.food_data) == 0:
            self.update_status("Error: No data available for feature preparation")
            return
        
        # Identify available features
        available_features = []
        for feature in self.base_nutritional_features + self.extended_features:
            if feature in self.food_data.columns:
                available_features.append(feature)
        
        self.features = available_features
        
        if not self.features:
            self.update_status("Error: No valid nutritional features available")
            return
        
        # Add health score features for KNN
        health_score_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features.extend(health_score_features)
        
        # Engineer additional features for better KNN performance
        self._engineer_additional_features()
        
        self.update_status(f"Prepared {len(self.features)} features for KNN models")
    
    def _engineer_additional_features(self):
        """Engineer additional features for improved KNN performance"""
        self.update_status("Engineering additional features for KNN...")
        
        try:
            # Calorie density (kcal per 100g)
            self.food_data['Calorie_Density'] = self.food_data.get('Energy(kcal) by calculation', 0).fillna(0)
            
            # Protein density (protein per 100 kcal) - safe division
            calories = self.food_data.get('Energy(kcal) by calculation', 0).fillna(0)
            protein = self.food_data.get('Protein(g)', 0).fillna(0)
            
            # Safe division with proper handling
            self.food_data['Protein_Density'] = 0.0
            mask = calories > 0
            if mask.any():
                self.food_data.loc[mask, 'Protein_Density'] = (protein[mask] / calories[mask]) * 100
            
            # Fiber density (fiber per 100 kcal) - safe division
            fiber = self.food_data.get('FIBTG (g) Dietary fibre', 0).fillna(0)
            self.food_data['Fiber_Density'] = 0.0
            if mask.any():
                self.food_data.loc[mask, 'Fiber_Density'] = (fiber[mask] / calories[mask]) * 100
            
            # Sugar percentage of total carbs - safe division
            sugar = self.food_data.get('SUGAR(g)', 0).fillna(0)
            carbs = self.food_data.get('CHOCDF (g) Carbohydrate', 0).fillna(0)
            
            self.food_data['Sugar_Carb_Ratio'] = 0.0
            carb_mask = carbs > 0
            if carb_mask.any():
                self.food_data.loc[carb_mask, 'Sugar_Carb_Ratio'] = sugar[carb_mask] / carbs[carb_mask]
            
            # Sodium to potassium ratio (important for hypertension) - safe division
            sodium = self.food_data.get('Na(mg)', 0).fillna(0)
            potassium = self.food_data.get('K(mg)', 0).fillna(0)
            
            self.food_data['Sodium_Potassium_Ratio'] = 0.0
            # For foods with no potassium, use a standard ratio based on sodium content
            potassium_mask = potassium > 0
            if potassium_mask.any():
                self.food_data.loc[potassium_mask, 'Sodium_Potassium_Ratio'] = (
                    sodium[potassium_mask] / potassium[potassium_mask]
                )
            
            # For foods with sodium but no potassium, assign a high ratio
            no_potassium_mask = (potassium == 0) & (sodium > 0)
            if no_potassium_mask.any():
                self.food_data.loc[no_potassium_mask, 'Sodium_Potassium_Ratio'] = sodium[no_potassium_mask] / 100
            
            # Saturated fat percentage of total fat - safe division
            sat_fat = self.food_data.get('FASAT (g) Saturated FA', 0).fillna(0)
            total_fat = self.food_data.get('Fat(g)', 0).fillna(0)
            
            self.food_data['SatFat_TotalFat_Ratio'] = 0.0
            fat_mask = total_fat > 0
            if fat_mask.any():
                self.food_data.loc[fat_mask, 'SatFat_TotalFat_Ratio'] = sat_fat[fat_mask] / total_fat[fat_mask]
            
            # Replace any remaining infinite or NaN values
            engineered_features = [
                'Calorie_Density', 'Protein_Density', 'Fiber_Density', 
                'Sugar_Carb_Ratio', 'Sodium_Potassium_Ratio', 'SatFat_TotalFat_Ratio'
            ]
            
            for feature in engineered_features:
                if feature in self.food_data.columns:
                    # Replace infinite values with 0
                    self.food_data[feature] = self.food_data[feature].replace([np.inf, -np.inf], 0)
                    # Fill any remaining NaN with 0
                    self.food_data[feature] = self.food_data[feature].fillna(0)
                    # Ensure all values are finite
                    self.food_data[feature] = np.where(
                        np.isfinite(self.food_data[feature]), 
                        self.food_data[feature], 
                        0
                    )
            
            # Add engineered features to feature list
            self.features.extend(engineered_features)
            
            self.update_status(f"✅ Engineered {len(engineered_features)} additional features")
            
        except Exception as e:
            self.update_status(f"⚠️ Feature engineering error: {e}")
            # Continue without engineered features if there's an error
            self.update_status("Continuing with basic nutritional features only...")
            
            # Make sure basic features are clean
            for feature in self.base_nutritional_features:
                if feature in self.food_data.columns:
                    self.food_data[feature] = pd.to_numeric(self.food_data[feature], errors='coerce').fillna(0)
            
            print(f"Feature engineering error details: {e}")
            print("First few rows of problematic data:")
            print(self.food_data[['Energy(kcal) by calculation', 'Protein(g)', 'CHOCDF (g) Carbohydrate']].head())
    
    def train_knn_models(self):
        """Train condition-specific KNN models with optimized parameters"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for KNN training")
            return
        
        try:
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Train models for different health conditions + general model
            conditions = ['General', 'Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
            
            for condition in conditions:
                self.update_status(f"Training KNN model for {condition}...")
                
                # Get condition-specific feature weights
                feature_weights = self._get_condition_feature_weights(condition)
                
                # Apply feature weighting
                X_weighted = X.copy()
                for feature, weight in feature_weights.items():
                    if feature in X_weighted.columns:
                        X_weighted[feature] *= weight
                
                # Scale features for KNN
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_weighted)
                
                # Optimize KNN parameters using grid search
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
                
                # Use a subset for parameter optimization if dataset is large
                if len(X_scaled) > 1000:
                    sample_size = min(500, len(X_scaled))
                    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                    X_sample = X_scaled[sample_indices]
                    y_sample = np.arange(len(sample_indices))  # Use indices as targets for similarity
                else:
                    X_sample = X_scaled
                    y_sample = np.arange(len(X_scaled))
                
                # Grid search for optimal parameters
                knn = KNeighborsRegressor()
                grid_search = GridSearchCV(
                    knn, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                grid_search.fit(X_sample, y_sample)
                best_params = grid_search.best_params_
                
                # Train final model with best parameters
                final_knn = KNeighborsRegressor(**best_params)
                final_knn.fit(X_scaled, np.arange(len(X_scaled)))
                
                # Store model and scaler
                self.knn_models[condition] = final_knn
                self.scalers[condition] = scaler
                
                # Store performance metrics
                cv_scores = cross_val_score(final_knn, X_scaled, np.arange(len(X_scaled)), cv=5)
                self.stats['knn_performance'][condition] = {
                    'best_params': best_params,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'n_samples': len(X_scaled)
                }
                
                self.update_status(f"{condition} KNN - Best params: {best_params}, CV score: {cv_scores.mean():.3f}")
            
            # Calculate feature importance based on variance and correlation
            self._calculate_feature_importance(X)
            
        except Exception as e:
            self.update_status(f"Error training KNN models: {e}")
    
    def _get_condition_feature_weights(self, condition):
        """Get condition-specific feature weights for KNN"""
        if condition == 'General':
            # Equal weights for general recommendations
            return {feature: 1.0 for feature in self.features}
        elif condition in self.condition_feature_weights:
            # Merge condition-specific weights with defaults
            weights = {feature: 1.0 for feature in self.features}
            weights.update(self.condition_feature_weights[condition])
            return weights
        else:
            return {feature: 1.0 for feature in self.features}
    
    def _calculate_feature_importance(self, X):
        """Calculate feature importance for interpretability"""
        try:
            # Calculate variance-based importance
            feature_variances = X.var()
            
            # Normalize to [0, 1]
            max_var = feature_variances.max()
            if max_var > 0:
                variance_importance = feature_variances / max_var
            else:
                variance_importance = pd.Series(0, index=X.columns)
            
            # Calculate correlation with health scores
            health_scores = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
            correlation_importance = pd.Series(0, index=X.columns)
            
            for score in health_scores:
                if score in X.columns:
                    correlations = X.corrwith(X[score]).abs()
                    correlation_importance += correlations.fillna(0)
            
            # Normalize correlation importance
            max_corr = correlation_importance.max()
            if max_corr > 0:
                correlation_importance /= max_corr
            
            # Combined importance (weighted average)
            combined_importance = 0.6 * variance_importance + 0.4 * correlation_importance
            
            # Store top important features
            top_features = combined_importance.nlargest(10)
            self.stats['feature_importance'] = {
                'top_features': top_features.to_dict(),
                'variance_based': variance_importance.nlargest(5).to_dict(),
                'correlation_based': correlation_importance.nlargest(5).to_dict()
            }
            
        except Exception as e:
            self.update_status(f"Error calculating feature importance: {e}")
    
    def get_recommendations(self, user_profile, meal_type='lunch', max_recommendations=10):
        """Get personalized food recommendations using health-aware KNN"""
        if not self.knn_models or len(self.food_data) == 0:
            return []
        
        try:
            # Calculate nutritional targets using medical calculator
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            daily_targets = nutritional_data['daily_targets']
            meal_targets = nutritional_data['meal_targets'].get(meal_type.lower(), 
                                                              nutritional_data['meal_targets']['lunch'])
            
            # Get user's health conditions
            health_conditions = user_profile.get('health_conditions', [])
            
            # Create user target vector for KNN
            user_vector = self._create_user_target_vector(meal_targets, health_conditions)
            
            # Select appropriate KNN model
            if health_conditions:
                # Use most severe condition's model
                condition_priority = ['Diabetes', 'High_Cholesterol', 'Hypertension', 'Obesity']
                selected_condition = 'General'
                for condition in condition_priority:
                    if condition in health_conditions:
                        selected_condition = condition
                        break
            else:
                selected_condition = 'General'
            
            knn_model = self.knn_models[selected_condition]
            scaler = self.scalers[selected_condition]
            
            # Scale user vector
            user_vector_scaled = scaler.transform([user_vector])
            
            # Find nearest neighbors
            distances, indices = knn_model.kneighbors(user_vector_scaled, n_neighbors=min(max_recommendations * 3, len(self.food_data)))
            
            # Apply category filter if specified
            category_filter = user_profile.get('category_filter', 'All')
            candidates = self.food_data.iloc[indices[0]]
            
            if category_filter != 'All':
                candidates = candidates[candidates['Category'] == category_filter]
            
            # Score and rank candidates
            recommendations = []
            for idx, food in candidates.iterrows():
                # Calculate comprehensive score
                knn_distance = distances[0][list(indices[0]).index(idx)]
                health_penalty = self._calculate_health_penalty(food, health_conditions)
                nutrition_match = self._calculate_nutrition_match_score(food, meal_targets)
                
                # Combined score (lower is better)
                combined_score = (0.4 * knn_distance + 0.3 * health_penalty + 0.3 * nutrition_match)
                
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
                    'knn_distance': knn_distance,
                    'health_penalty': health_penalty,
                    'nutrition_match': nutrition_match,
                    'combined_score': combined_score,
                    'knn_model_used': selected_condition,
                    'suitable_for_conditions': self._check_condition_suitability(food, health_conditions),
                    'targets_met': self._check_targets_met(food, meal_targets),
                    'health_warnings': self._generate_health_warnings(food, health_conditions)
                }
                
                recommendations.append(recommendation)
            
            # Sort by combined score and return top recommendations
            recommendations.sort(key=lambda x: x['combined_score'])
            top_recommendations = recommendations[:max_recommendations]
            
            # Add explanations
            for rec in top_recommendations:
                rec['explanation'] = self._generate_knn_explanation(rec, meal_targets, health_conditions)
                rec['nutritional_data'] = nutritional_data
            
            return top_recommendations
            
        except Exception as e:
            self.update_status(f"Error getting KNN recommendations: {e}")
            return []
    
    def _create_user_target_vector(self, meal_targets, health_conditions):
        """Create user target vector for KNN similarity search"""
        # Initialize vector with zeros
        user_vector = np.zeros(len(self.features))
        
        # Map targets to feature vector
        target_mapping = {
            'Energy(kcal) by calculation': meal_targets.get('calories', 0),
            'Protein(g)': (meal_targets.get('protein_min', 0) + meal_targets.get('protein_max', 0)) / 2,
            'CHOCDF (g) Carbohydrate': (meal_targets.get('carbs_min', 0) + meal_targets.get('carbs_max', 0)) / 2,
            'SUGAR(g)': meal_targets.get('sugar_max', 0) * 0.5,  # Aim for half of maximum
            'FIBTG (g) Dietary fibre': meal_targets.get('fiber_min', 0),
            'Fat(g)': (meal_targets.get('fat_min', 0) + meal_targets.get('fat_max', 0)) / 2,
            'Na(mg)': meal_targets.get('sodium_max', 0) * 0.3,  # Aim for 30% of maximum
            'K(mg)': meal_targets.get('potassium_min', 0) * 0.3,  # 30% of daily minimum per meal
            'Ca(mg)': meal_targets.get('calcium_min', 0) * 0.3,  # 30% of daily minimum per meal
            'CHOLE(mg) Cholesterol': meal_targets.get('cholesterol_max', 0) * 0.2,  # Aim for 20% of maximum
            'FASAT (g) Saturated FA': meal_targets.get('saturated_fat_max', 0) * 0.3,  # Aim for 30% of maximum
        }
        
        # Set ideal health scores (0 = most suitable)
        health_score_targets = {
            'Diabetes_Score': 1.0 if 'Diabetes' in health_conditions else 2.0,
            'Obesity_Score': 1.0 if 'Obesity' in health_conditions else 2.0,
            'Hypertension_Score': 1.0 if 'Hypertension' in health_conditions else 2.0,
            'High_Cholesterol_Score': 1.0 if 'High_Cholesterol' in health_conditions else 2.0
        }
        
        # Fill vector
        for i, feature in enumerate(self.features):
            if feature in target_mapping:
                user_vector[i] = target_mapping[feature]
            elif feature in health_score_targets:
                user_vector[i] = health_score_targets[feature]
            elif feature.startswith('Calorie_Density'):
                user_vector[i] = meal_targets.get('calories', 200)  # Target moderate calorie density
            elif feature.startswith('Protein_Density'):
                user_vector[i] = 15  # Target 15g protein per 100 kcal
            elif feature.startswith('Fiber_Density'):
                user_vector[i] = 5   # Target 5g fiber per 100 kcal
            elif feature.startswith('Sugar_Carb_Ratio'):
                user_vector[i] = 0.2  # Target 20% sugar of total carbs
            elif feature.startswith('Sodium_Potassium_Ratio'):
                user_vector[i] = 0.5  # Target good sodium to potassium ratio
            elif feature.startswith('SatFat_TotalFat_Ratio'):
                user_vector[i] = 0.3  # Target 30% saturated fat of total fat
        
        return user_vector
    
    def _calculate_health_penalty(self, food, health_conditions):
        """Calculate penalty based on health conditions"""
        penalty = 0
        
        for condition in health_conditions:
            score_col = f'{condition}_Score'
            if score_col in food.index:
                penalty += float(food.get(score_col, 0))
        
        return penalty / len(health_conditions) if health_conditions else 0
    
    def _calculate_nutrition_match_score(self, food, targets):
        """Calculate how well a food matches nutritional targets"""
        score = 0
        
        # Calorie match (normalized difference)
        calories = float(food.get('Energy(kcal) by calculation', 0))
        target_calories = targets.get('calories', 200)
        if target_calories > 0:
            calorie_diff = abs(calories - target_calories) / target_calories
            score += calorie_diff
        
        # Protein match
        protein = float(food.get('Protein(g)', 0))
        protein_target = (targets.get('protein_min', 0) + targets.get('protein_max', 50)) / 2
        if protein_target > 0:
            protein_diff = abs(protein - protein_target) / protein_target
            score += protein_diff * 0.8
        
        # Carb match
        carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
        carbs_target = (targets.get('carbs_min', 0) + targets.get('carbs_max', 50)) / 2
        if carbs_target > 0:
            carbs_diff = abs(carbs - carbs_target) / carbs_target
            score += carbs_diff * 0.8
        
        # Sugar penalty
        sugar = float(food.get('SUGAR(g)', 0))
        sugar_max = targets.get('sugar_max', 10)
        if sugar > sugar_max:
            score += (sugar - sugar_max) / sugar_max
        
        # Fiber bonus/penalty
        fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
        fiber_min = targets.get('fiber_min', 5)
        if fiber < fiber_min:
            score += (fiber_min - fiber) / fiber_min
        
        return score
    
    def _check_condition_suitability(self, food, health_conditions):
        """Check which health conditions this food is suitable for"""
        suitable = []
        
        for condition in health_conditions:
            score_col = f'{condition}_Score'
            if score_col in food.index:
                score = float(food.get(score_col, 0))
                if score <= 1.5:  # Low score means suitable
                    suitable.append(condition)
        
        return suitable
    
    def _check_targets_met(self, food, targets):
        """Check which nutritional targets are met by this food"""
        met = []
        
        # Check calorie range
        calories = float(food.get('Energy(kcal) by calculation', 0))
        target_calories = targets.get('calories', 200)
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
                if float(food.get('SUGAR(g)', 0)) > 15:
                    warnings.append("High sugar - monitor blood glucose carefully")
                if (float(food.get('CHOCDF (g) Carbohydrate', 0)) > 30 and 
                    float(food.get('FIBTG (g) Dietary fibre', 0)) < 3):
                    warnings.append("High carbs with low fiber - may cause glucose spike")
            
            elif condition == 'Hypertension':
                if float(food.get('Na(mg)', 0)) > 300:
                    warnings.append("High sodium - may elevate blood pressure")
            
            elif condition == 'High_Cholesterol':
                if float(food.get('FASAT (g) Saturated FA', 0)) > 5:
                    warnings.append("High saturated fat - may increase cholesterol")
                if float(food.get('CHOLE(mg) Cholesterol', 0)) > 100:
                    warnings.append("Contains dietary cholesterol")
            
            elif condition == 'Obesity':
                if float(food.get('Energy(kcal) by calculation', 0)) > 300:
                    warnings.append("High calorie density - portion control recommended")
        
        return warnings
    
    def _generate_knn_explanation(self, recommendation, targets, health_conditions):
        """Generate explanation for KNN recommendation"""
        explanations = []
        
        # KNN model explanation
        model_used = recommendation['knn_model_used']
        explanations.append(f"Matched using {model_used} KNN model")
        
        # Distance explanation
        knn_distance = recommendation['knn_distance']
        if knn_distance < 1.0:
            explanations.append("Excellent nutritional similarity")
        elif knn_distance < 2.0:
            explanations.append("Good nutritional similarity")
        else:
            explanations.append("Fair nutritional similarity")
        
        # Health suitability
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            conditions_text = ", ".join(suitable_conditions)
            explanations.append(f"Suitable for {conditions_text}")
        
        # Targets met
        targets_met = recommendation['targets_met']
        if targets_met:
            targets_text = ", ".join(targets_met)
            explanations.append(f"Meets {targets_text} targets")
        
        return " | ".join(explanations)
    
    def get_stats(self):
        """Get statistics about the KNN recommendation system"""
        return self.stats
    
    def get_model_info(self):
        """Get detailed information about trained KNN models"""
        model_info = {}
        
        for condition, model in self.knn_models.items():
            model_info[condition] = {
                'n_neighbors': model.n_neighbors,
                'weights': model.weights,
                'metric': model.metric,
                'n_features': len(self.features),
                'performance': self.stats['knn_performance'].get(condition, {})
            }
        
        return model_info


class HealthDrivenKNNFoodRecommenderUI:
    """Modern GUI for the health-driven KNN food recommendation system"""
    
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Health-Driven KNN Food Recommendation System")
        self.master.geometry("1600x900")
        self.master.configure(bg="#f8f9fa")
        
        # Modern color scheme
        self.colors = {
            'primary': '#2c3e50',      # Dark blue-gray
            'secondary': '#3498db',     # Blue
            'success': '#27ae60',       # Green
            'warning': '#f39c12',       # Orange
            'danger': '#e74c3c',        # Red
            'info': '#8e44ad',          # Purple
            'light': '#ecf0f1',         # Light gray
            'white': '#ffffff',
            'background': '#f8f9fa',    # Very light gray
            'text': '#2c3e50',
            'text_light': '#7f8c8d'
        }
        
        # Configure styles
        self.setup_styles()
        
        # Initialize recommender
        self.recommender = recommender or HealthAwareKNNRecommender()
        
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
            'heading': ('Segoe UI', 18, 'bold'),
            'subheading': ('Segoe UI', 12, 'bold'),
            'body': ('Segoe UI', 10),
            'caption': ('Segoe UI', 9),
            'small': ('Segoe UI', 8)
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=self.fonts['heading'], 
                           foreground=self.colors['primary'])
        self.style.configure('Subtitle.TLabel', font=self.fonts['subheading'], 
                           foreground=self.colors['text'])
        self.style.configure('Success.TButton', font=self.fonts['body'])
        self.style.configure('Primary.TButton', font=self.fonts['body'])
        
        # Treeview styling
        self.style.configure('KNN.Treeview', font=self.fonts['body'], rowheight=25)
        self.style.configure('KNN.Treeview.Heading', font=self.fonts['subheading'])
    
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
        
        # Create layout: left panel, center panel, right panel
        self.create_input_panel(content_frame)
        self.create_results_panel(content_frame)
        self.create_analysis_panel(content_frame)
        
        # Charts section
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
        
        ttk.Label(title_frame, text="Health-Driven KNN Food Recommendation System", 
                 style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Automatic Medical Target Calculation + K-Nearest Neighbors Matching", 
                 style='Subtitle.TLabel', foreground=self.colors['info']).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="🔍 Get KNN Recommendations", 
                  command=self.get_recommendations, style='Success.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="🔄 Reset", 
                  command=self.reset_form).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="📊 Calculate Targets", 
                  command=self.show_nutrition_calculation).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="🤖 Model Info", 
                  command=self.show_model_info).pack(side=tk.RIGHT, padx=5)
    
    def create_input_panel(self, parent):
        """Create input panel with health profile form"""
        # Left panel for inputs
        left_panel = ttk.LabelFrame(parent, text="Health Profile & KNN Settings", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), ipadx=10)
        
        # Personal Information
        personal_frame = ttk.LabelFrame(left_panel, text="Personal Information", padding="10")
        personal_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Create form fields
        self.create_form_field(personal_frame, "Weight (kg):", "weight_var", 70.0, 0, 0)
        self.create_form_field(personal_frame, "Height (cm):", "height_var", 170.0, 1, 0)
        self.create_form_field(personal_frame, "Age (years):", "age_var", 30, 2, 0)
        
        # Gender selection
        ttk.Label(personal_frame, text="Gender:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar(value="Male")
        gender_combo = ttk.Combobox(personal_frame, textvariable=self.gender_var,
                                   values=['Male', 'Female'], state="readonly", width=15)
        gender_combo.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Activity level
        ttk.Label(personal_frame, text="Activity Level:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.activity_var = tk.StringVar(value="Moderate")
        activity_combo = ttk.Combobox(personal_frame, textvariable=self.activity_var,
                                     values=['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'],
                                     state="readonly", width=15)
        activity_combo.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Weight goal
        ttk.Label(personal_frame, text="Weight Goal:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.weight_goal_var = tk.StringVar(value="Maintain Weight")
        weight_goal_combo = ttk.Combobox(personal_frame, textvariable=self.weight_goal_var,
                                        values=['Lose Weight', 'Maintain Weight', 'Gain Weight'],
                                        state="readonly", width=15)
        weight_goal_combo.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        # BMI Display
        ttk.Label(personal_frame, text="BMI:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.bmi_var = tk.StringVar(value="Calculating...")
        bmi_label = ttk.Label(personal_frame, textvariable=self.bmi_var, font=self.fonts['body'])
        bmi_label.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Health Conditions
        conditions_frame = ttk.LabelFrame(left_panel, text="Health Conditions (NCDs)", padding="10")
        conditions_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        ttk.Checkbutton(conditions_frame, text="Type 2 Diabetes", variable=self.diabetes_var,
                       command=self.on_condition_change).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="Obesity", variable=self.obesity_var,
                       command=self.on_condition_change).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="Hypertension", variable=self.hypertension_var,
                       command=self.on_condition_change).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var,
                       command=self.on_condition_change).pack(anchor=tk.W)
        
        # KNN Settings
        knn_frame = ttk.LabelFrame(left_panel, text="KNN Model Settings", padding="10")
        knn_frame.pack(fill=tk.X)
        
        ttk.Label(knn_frame, text="Meal Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.meal_type_var = tk.StringVar(value="Lunch")
        meal_combo = ttk.Combobox(knn_frame, textvariable=self.meal_type_var,
                                 values=['Breakfast', 'Lunch', 'Dinner', 'Snack'],
                                 state="readonly", width=15)
        meal_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(knn_frame, text="Category Filter:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.category_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(knn_frame, textvariable=self.category_var,
                                          state="readonly", width=15)
        self.category_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.update_category_list()
        
        ttk.Label(knn_frame, text="Max Results:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_results_var = tk.StringVar(value="10")
        results_combo = ttk.Combobox(knn_frame, textvariable=self.max_results_var,
                                    values=['5', '8', '10', '12', '15', '20'],
                                    state="readonly", width=15)
        results_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # KNN Model Selection (auto-selected based on conditions)
        ttk.Label(knn_frame, text="KNN Model:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.selected_model_var = tk.StringVar(value="General")
        model_label = ttk.Label(knn_frame, textvariable=self.selected_model_var, 
                               font=self.fonts['body'], foreground=self.colors['info'])
        model_label.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Bind events for BMI calculation and model selection
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()
        self.on_condition_change()
    
    def create_form_field(self, parent, label_text, var_name, default_value, row, col):
        """Create a form field with label and entry"""
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
            height = self.height_var.get() / 100  # Convert to meters
            
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
                
                # Auto-check obesity if BMI >= 30
                if bmi >= 30:
                    self.obesity_var.set(True)
                    self.on_condition_change()
                
            else:
                self.bmi_var.set("Invalid input")
        except:
            self.bmi_var.set("Calculating...")
    
    def on_condition_change(self):
        """Update selected KNN model based on health conditions"""
        health_conditions = []
        if self.diabetes_var.get():
            health_conditions.append('Diabetes')
        if self.obesity_var.get():
            health_conditions.append('Obesity')
        if self.hypertension_var.get():
            health_conditions.append('Hypertension')
        if self.cholesterol_var.get():
            health_conditions.append('High_Cholesterol')
        
        # Select appropriate KNN model
        if health_conditions:
            # Priority: Diabetes > High_Cholesterol > Hypertension > Obesity
            condition_priority = ['Diabetes', 'High_Cholesterol', 'Hypertension', 'Obesity']
            selected_condition = 'General'
            for condition in condition_priority:
                if condition in health_conditions:
                    selected_condition = condition
                    break
        else:
            selected_condition = 'General'
        
        self.selected_model_var.set(f"{selected_condition} Model")
    
    def update_category_list(self):
        """Update the category dropdown with available categories"""
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
            if 'Category' in self.recommender.food_data.columns:
                categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
    
    def create_results_panel(self, parent):
        """Create results panel with recommendations table"""
        # Center panel for results
        center_panel = ttk.LabelFrame(parent, text="KNN Food Recommendations", padding="15")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(center_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Recommendations tab
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="🎯 KNN Results")
        
        # Create treeview for recommendations
        columns = ('Name', 'Category', 'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'KNN Score', 'Model Used')
        self.tree = ttk.Treeview(rec_frame, columns=columns, show='headings', 
                               style='KNN.Treeview', height=18)
        
        # Configure columns
        column_widths = {'Name': 140, 'Category': 90, 'Calories': 70, 'Protein': 60, 'Carbs': 60, 
                        'Sugar': 50, 'Fiber': 50, 'KNN Score': 80, 'Model Used': 100}
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Add scrollbars
        tree_frame = ttk.Frame(rec_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Nutritional targets tab
        targets_frame = ttk.Frame(self.notebook)
        self.notebook.add(targets_frame, text="📊 Calculated Targets")
        
        # Create text widget for targets
        targets_container = ttk.Frame(targets_frame)
        targets_container.pack(fill=tk.BOTH, expand=True)
        
        self.targets_text = tk.Text(targets_container, wrap=tk.WORD, font=self.fonts['body'])
        targets_scrollbar = ttk.Scrollbar(targets_container, orient="vertical", command=self.targets_text.yview)
        self.targets_text.configure(yscrollcommand=targets_scrollbar.set)
        
        self.targets_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        targets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
        # Initial text
        self.targets_text.insert(tk.END, "Click 'Calculate Targets' to see your personalized nutritional targets calculated using medical formulas (BMR, TDEE) and health condition guidelines (ADA, AHA, WHO, DASH).")
    
    def create_analysis_panel(self, parent):
        """Create analysis panel for detailed food information and KNN analysis"""
        # Right panel for analysis
        right_panel = ttk.LabelFrame(parent, text="Food Analysis & KNN Details", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, ipadx=10)
        
        # Create notebook for analysis views
        analysis_notebook = ttk.Notebook(right_panel)
        analysis_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Food details tab
        details_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(details_frame, text="🍽️ Food Details")
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, font=self.fonts['body'], width=35)
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # KNN analysis tab
        knn_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(knn_frame, text="🤖 KNN Analysis")
        
        self.knn_text = tk.Text(knn_frame, wrap=tk.WORD, font=self.fonts['body'], width=35)
        knn_scrollbar = ttk.Scrollbar(knn_frame, orient="vertical", command=self.knn_text.yview)
        self.knn_text.configure(yscrollcommand=knn_scrollbar.set)
        
        self.knn_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        knn_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature importance tab
        features_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(features_frame, text="📈 Features")
        
        self.features_text = tk.Text(features_frame, wrap=tk.WORD, font=self.fonts['small'], width=35)
        features_scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=self.features_text.yview)
        self.features_text.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        features_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial text for analysis panels
        self.details_text.insert(tk.END, "Select a food item from the KNN recommendations to view detailed nutritional information, health suitability analysis, and medical rationale.")
        self.knn_text.insert(tk.END, "KNN analysis will appear here after getting recommendations, showing distance calculations, feature weights, and model selection rationale.")
        self.features_text.insert(tk.END, "Feature importance analysis will be displayed here after model training completes.")
        
        # Update feature importance display
        self.update_feature_importance_display()
    
    def create_charts_panel(self, parent):
        """Create charts panel for KNN visualizations"""
        charts_frame = ttk.LabelFrame(parent, text="KNN Analysis & Nutritional Visualizations", padding="10")
        charts_frame.pack(fill=tk.X, pady=(20, 0))
        
        if not MATPLOTLIB_AVAILABLE:
            # Create simple text display if matplotlib is not available
            no_charts_label = ttk.Label(charts_frame, 
                                       text="📊 Visualization charts not available\n(matplotlib not installed)",
                                       font=self.fonts['body'], 
                                       foreground=self.colors['text_light'])
            no_charts_label.pack(pady=20)
            return
        
        try:
            # Create matplotlib figure
            self.fig = Figure(figsize=(16, 5), dpi=100)
            self.fig.patch.set_facecolor(self.colors['white'])
            
            # Create subplots for different analyses
            self.ax1 = self.fig.add_subplot(151)  # KNN distance distribution
            self.ax2 = self.fig.add_subplot(152)  # Health score comparison
            self.ax3 = self.fig.add_subplot(153)  # Macronutrient radar
            self.ax4 = self.fig.add_subplot(154)  # Category distribution
            self.ax5 = self.fig.add_subplot(155)  # Feature importance
            
            # Initialize charts
            self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.update_charts([])
            
        except Exception as chart_error:
            # Fallback to simple text display
            error_label = ttk.Label(charts_frame, 
                                   text=f"📊 Charts temporarily unavailable\n({str(chart_error)[:50]}...)",
                                   font=self.fonts['body'], 
                                   foreground=self.colors['warning'])
            error_label.pack(pady=20)
            print(f"Charts creation error: {chart_error}")
            
            # Set dummy chart attributes to prevent errors
            self.fig = None
            self.canvas = None
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding="5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready - Enter your health profile and click 'Calculate Targets'")
        ttk.Label(status_frame, textvariable=self.status_var, font=self.fonts['body']).pack(side=tk.LEFT)
        
        # Stats display
        self.stats_frame = ttk.Frame(status_frame)
        self.stats_frame.pack(side=tk.RIGHT)
    
    def get_user_profile(self):
        """Get user profile from form inputs"""
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
    
    def show_nutrition_calculation(self):
        """Show the calculated nutritional targets with medical rationale"""
        try:
            user_profile = self.get_user_profile()
            
            # Calculate nutritional targets
            nutritional_data = self.recommender.nutrition_calculator.calculate_nutritional_targets(user_profile)
            self.current_nutritional_data = nutritional_data
            
            # Display in targets tab
            self.targets_text.delete(1.0, tk.END)
            
            # Add header
            self.targets_text.insert(tk.END, "🏥 MEDICAL-GRADE NUTRITIONAL TARGET CALCULATION\n", "header")
            self.targets_text.insert(tk.END, "="*70 + "\n\n", "separator")
            
            # Add medical calculations
            calc = nutritional_data['calculations']
            self.targets_text.insert(tk.END, "📊 Medical Formula Calculations:\n", "subheader")
            self.targets_text.insert(tk.END, f"• {calc['bmr_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['bmi_calculation']}\n")
            self.targets_text.insert(tk.END, f"• {calc['tdee_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['target_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['protein_calculation']}\n")
            self.targets_text.insert(tk.END, f"• {calc['fiber_calculation']}\n\n")
            
            # Add health condition modifications
            health_mods = nutritional_data['health_modifications']
            self.targets_text.insert(tk.END, "🏥 Health Condition Modifications:\n", "subheader")
            self.targets_text.insert(tk.END, f"{health_mods}\n\n")
            
            # Add daily targets
            daily = nutritional_data['daily_targets']
            self.targets_text.insert(tk.END, "🎯 Daily Nutritional Targets:\n", "subheader")
            self.targets_text.insert(tk.END, f"• Calories: {daily['calories']:.0f} kcal\n")
            self.targets_text.insert(tk.END, f"• Protein: {daily['protein_min']:.0f}-{daily['protein_max']:.0f} g ({daily['protein_min']*4/daily['calories']*100:.0f}-{daily['protein_max']*4/daily['calories']*100:.0f}%)\n")
            self.targets_text.insert(tk.END, f"• Carbohydrates: {daily['carbs_min']:.0f}-{daily['carbs_max']:.0f} g ({daily['carbs_min']*4/daily['calories']*100:.0f}-{daily['carbs_max']*4/daily['calories']*100:.0f}%)\n")
            self.targets_text.insert(tk.END, f"• Sugar (max): {daily['sugar_max']:.0f} g ({daily['sugar_max']*4/daily['calories']*100:.1f}%)\n")
            self.targets_text.insert(tk.END, f"• Fat: {daily['fat_min']:.0f}-{daily['fat_max']:.0f} g ({daily['fat_min']*9/daily['calories']*100:.0f}-{daily['fat_max']*9/daily['calories']*100:.0f}%)\n")
            self.targets_text.insert(tk.END, f"• Saturated Fat (max): {daily['saturated_fat_max']:.0f} g ({daily['saturated_fat_max']*9/daily['calories']*100:.1f}%)\n")
            self.targets_text.insert(tk.END, f"• Fiber (min): {daily['fiber_min']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Sodium (max): {daily['sodium_max']:.0f} mg\n")
            if 'potassium_min' in daily:
                self.targets_text.insert(tk.END, f"• Potassium (min): {daily['potassium_min']:.0f} mg\n")
            if 'calcium_min' in daily:
                self.targets_text.insert(tk.END, f"• Calcium (min): {daily['calcium_min']:.0f} mg\n")
            self.targets_text.insert(tk.END, "\n")
            
            # Add meal targets
            meal_type = self.meal_type_var.get().lower()
            if meal_type in nutritional_data['meal_targets']:
                meal = nutritional_data['meal_targets'][meal_type]
                self.targets_text.insert(tk.END, f"🍽️ {self.meal_type_var.get()} Targets (for KNN matching):\n", "subheader")
                self.targets_text.insert(tk.END, f"• Calories: {meal['calories']:.0f} kcal\n")
                self.targets_text.insert(tk.END, f"• Protein: {meal['protein_min']:.0f}-{meal['protein_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Carbohydrates: {meal['carbs_min']:.0f}-{meal['carbs_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Sugar (max): {meal['sugar_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fat: {meal['fat_min']:.0f}-{meal['fat_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fiber (min): {meal['fiber_min']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Sodium (max): {meal['sodium_max']:.0f} mg\n\n")
            
            # Add medical rationale
            self.targets_text.insert(tk.END, "📚 Medical Rationale:\n", "subheader")
            self.targets_text.insert(tk.END, f"{nutritional_data['medical_rationale']}\n\n")
            
            # Add KNN preparation info
            health_conditions = user_profile['health_conditions']
            if health_conditions:
                self.targets_text.insert(tk.END, "🤖 KNN Model Selection:\n", "subheader")
                condition_priority = ['Diabetes', 'High_Cholesterol', 'Hypertension', 'Obesity']
                selected_condition = 'General'
                for condition in condition_priority:
                    if condition in health_conditions:
                        selected_condition = condition
                        break
                self.targets_text.insert(tk.END, f"Selected Model: {selected_condition} (based on condition priority)\n")
                self.targets_text.insert(tk.END, f"Feature Weights: Optimized for {selected_condition} management\n")
            else:
                self.targets_text.insert(tk.END, "🤖 KNN Model Selection: General model (no specific health conditions)\n")
            
            # Configure text tags for formatting
            self.targets_text.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.targets_text.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.targets_text.tag_configure("separator", foreground=self.colors['text_light'])
            
            # Switch to targets tab
            self.notebook.select(1)
            
            self.status_var.set("✅ Medical targets calculated successfully - Ready for KNN recommendations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating nutrition targets: {str(e)}")
    
    def get_recommendations(self):
        """Get KNN food recommendations based on calculated targets"""
        try:
            if not self.current_nutritional_data:
                messagebox.showwarning("Warning", "Please calculate nutrition targets first!")
                return
            
            user_profile = self.get_user_profile()
            meal_type = self.meal_type_var.get().lower()
            max_results = int(self.max_results_var.get())
            
            self.status_var.set("🔍 Running KNN algorithm to find optimal food matches...")
            self.master.update()
            
            # Get KNN recommendations
            recommendations = self.recommender.get_recommendations(
                user_profile, meal_type, max_results
            )
            
            self.last_recommendations = recommendations
            
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    model_used = rec.get('knn_model_used', 'General')
                    
                    self.tree.insert('', 'end', values=(
                        rec['name'][:18],  # Truncate long names
                        rec['category'],
                        f"{rec['calories']:.0f}",
                        f"{rec['protein']:.1f}",
                        f"{rec['carbs']:.1f}",
                        f"{rec['sugar']:.1f}",
                        f"{rec['fiber']:.1f}",
                        f"{rec['combined_score']:.2f}",
                        model_used
                    ))
                
                # Select first item
                if self.tree.get_children():
                    first_item = self.tree.get_children()[0]
                    self.tree.selection_set(first_item)
                    self.tree.focus(first_item)
                    self.show_food_details(None)
                
                # Update charts and analysis
                self.update_charts(recommendations)
                self.update_knn_analysis(recommendations)
                
                # Switch to recommendations tab
                self.notebook.select(0)
                
                health_conditions = user_profile['health_conditions']
                condition_text = ", ".join(health_conditions) if health_conditions else "general health"
                model_used = recommendations[0].get('knn_model_used', 'General')
                self.status_var.set(f"✅ Found {len(recommendations)} optimal foods using {model_used} KNN model for {condition_text}")
                
            else:
                messagebox.showinfo("No Results", "No suitable foods found matching your criteria.")
                self.status_var.set("❌ No suitable foods found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error getting KNN recommendations: {str(e)}")
            self.status_var.set("❌ Error occurred during KNN recommendation")
    
    def show_food_details(self, event):
        """Show detailed information for selected food"""
        selected_items = self.tree.selection()
        if not selected_items or not self.last_recommendations:
            return
        
        # Get selected item
        item = selected_items[0]
        item_index = self.tree.index(item)
        
        if item_index < len(self.last_recommendations):
            rec = self.last_recommendations[item_index]
            
            # Clear details text
            self.details_text.delete(1.0, tk.END)
            
            # Add food name and category
            self.details_text.insert(tk.END, f"🍽️ {rec['name']}\n", "title")
            self.details_text.insert(tk.END, f"Category: {rec['category']}\n\n", "subtitle")
            
            # Add KNN matching information
            self.details_text.insert(tk.END, "🤖 KNN Matching Results:\n", "header")
            self.details_text.insert(tk.END, f"• Model Used: {rec.get('knn_model_used', 'General')}\n")
            self.details_text.insert(tk.END, f"• KNN Distance: {rec.get('knn_distance', 0):.3f}\n")
            self.details_text.insert(tk.END, f"• Combined Score: {rec['combined_score']:.3f} (lower = better)\n")
            self.details_text.insert(tk.END, f"• Health Penalty: {rec.get('health_penalty', 0):.3f}\n")
            self.details_text.insert(tk.END, f"• Nutrition Match: {rec.get('nutrition_match', 0):.3f}\n\n")
            
            # Add nutritional information
            self.details_text.insert(tk.END, "📊 Nutritional Information (per 100g):\n", "header")
            self.details_text.insert(tk.END, f"• Energy: {rec['calories']:.0f} kcal\n")
            self.details_text.insert(tk.END, f"• Protein: {rec['protein']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Carbohydrates: {rec['carbs']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sugar: {rec['sugar']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Dietary Fiber: {rec['fiber']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Fat: {rec['fat']:.1f} g\n")
            
            if rec['sodium'] > 0:
                self.details_text.insert(tk.END, f"• Sodium: {rec['sodium']:.0f} mg\n")
            if rec['potassium'] > 0:
                self.details_text.insert(tk.END, f"• Potassium: {rec['potassium']:.0f} mg\n")
            if rec['cholesterol'] > 0:
                self.details_text.insert(tk.END, f"• Cholesterol: {rec['cholesterol']:.0f} mg\n")
            if rec['saturated_fat'] > 0:
                self.details_text.insert(tk.END, f"• Saturated Fat: {rec['saturated_fat']:.1f} g\n")
            
            self.details_text.insert(tk.END, "\n")
            
            # Add health suitability
            if rec['suitable_for_conditions']:
                self.details_text.insert(tk.END, "✅ Suitable for Health Conditions:\n", "good")
                for condition in rec['suitable_for_conditions']:
                    self.details_text.insert(tk.END, f"• {condition}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Add health warnings
            if rec['health_warnings']:
                self.details_text.insert(tk.END, "⚠️ Health Considerations:\n", "warning")
                for warning in rec['health_warnings']:
                    self.details_text.insert(tk.END, f"• {warning}\n", "warning")
                self.details_text.insert(tk.END, "\n")
            
            # Add targets met
            if rec['targets_met']:
                self.details_text.insert(tk.END, "🎯 Nutritional Targets Met:\n", "good")
                for target in rec['targets_met']:
                    self.details_text.insert(tk.END, f"• {target}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Add KNN explanation
            self.details_text.insert(tk.END, "🔍 Why This Food Was Recommended:\n", "header")
            self.details_text.insert(tk.END, f"{rec['explanation']}\n")
            
            # Configure text tags
            self.details_text.tag_configure("title", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.details_text.tag_configure("subtitle", font=self.fonts['subheading'], foreground=self.colors['text'])
            self.details_text.tag_configure("header", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.details_text.tag_configure("good", foreground=self.colors['success'])
            self.details_text.tag_configure("warning", foreground=self.colors['warning'])
    
    def update_knn_analysis(self, recommendations):
        """Update KNN analysis panel"""
        self.knn_text.delete(1.0, tk.END)
        
        if not recommendations:
            self.knn_text.insert(tk.END, "No KNN analysis available.")
            return
        
        # Add KNN analysis header
        self.knn_text.insert(tk.END, "🤖 K-NEAREST NEIGHBORS ANALYSIS\n", "header")
        self.knn_text.insert(tk.END, "="*40 + "\n\n", "separator")
        
        # Model information
        model_used = recommendations[0].get('knn_model_used', 'General')
        self.knn_text.insert(tk.END, f"📊 Model Used: {model_used}\n", "subheader")
        
        # Get model info
        model_info = self.recommender.get_model_info()
        if model_used in model_info:
            info = model_info[model_used]
            self.knn_text.insert(tk.END, f"• K (neighbors): {info['n_neighbors']}\n")
            self.knn_text.insert(tk.END, f"• Distance weighting: {info['weights']}\n")
            self.knn_text.insert(tk.END, f"• Distance metric: {info['metric']}\n")
            self.knn_text.insert(tk.END, f"• Features used: {info['n_features']}\n")
            
            if 'performance' in info and info['performance']:
                perf = info['performance']
                self.knn_text.insert(tk.END, f"• Model CV score: {perf.get('cv_mean', 0):.3f}\n")
        
        self.knn_text.insert(tk.END, "\n")
        
        # Distance analysis
        distances = [rec.get('knn_distance', 0) for rec in recommendations]
        self.knn_text.insert(tk.END, "📏 Distance Analysis:\n", "subheader")
        self.knn_text.insert(tk.END, f"• Best match distance: {min(distances):.3f}\n")
        self.knn_text.insert(tk.END, f"• Average distance: {np.mean(distances):.3f}\n")
        self.knn_text.insert(tk.END, f"• Distance range: {min(distances):.3f} - {max(distances):.3f}\n\n")
        
        # Score breakdown for top recommendation
        top_rec = recommendations[0]
        self.knn_text.insert(tk.END, f"🏆 Top Recommendation: {top_rec['name']}\n", "subheader")
        self.knn_text.insert(tk.END, f"• KNN Distance: {top_rec.get('knn_distance', 0):.3f} (40% weight)\n")
        self.knn_text.insert(tk.END, f"• Health Penalty: {top_rec.get('health_penalty', 0):.3f} (30% weight)\n")
        self.knn_text.insert(tk.END, f"• Nutrition Match: {top_rec.get('nutrition_match', 0):.3f} (30% weight)\n")
        self.knn_text.insert(tk.END, f"• Combined Score: {top_rec['combined_score']:.3f}\n\n")
        
        # Feature importance for selected model
        stats = self.recommender.get_stats()
        if 'feature_importance' in stats and stats['feature_importance']:
            importance = stats['feature_importance']
            if 'top_features' in importance:
                self.knn_text.insert(tk.END, "🔍 Top Important Features:\n", "subheader")
                for feature, score in list(importance['top_features'].items())[:5]:
                    self.knn_text.insert(tk.END, f"• {feature}: {score:.3f}\n")
        
        # Configure text tags
        self.knn_text.tag_configure("header", font=self.fonts['subheading'], foreground=self.colors['primary'])
        self.knn_text.tag_configure("subheader", font=self.fonts['body'], foreground=self.colors['secondary'])
        self.knn_text.tag_configure("separator", foreground=self.colors['text_light'])
    
    def update_feature_importance_display(self):
        """Update feature importance display"""
        try:
            stats = self.recommender.get_stats()
            
            self.features_text.delete(1.0, tk.END)
            
            self.features_text.insert(tk.END, "📈 FEATURE IMPORTANCE ANALYSIS\n", "header")
            self.features_text.insert(tk.END, "="*35 + "\n\n", "separator")
            
            if 'feature_importance' in stats and stats['feature_importance']:
                importance = stats['feature_importance']
                
                # Top features overall
                if 'top_features' in importance:
                    self.features_text.insert(tk.END, "🏆 Top Features (Overall):\n", "subheader")
                    for i, (feature, score) in enumerate(list(importance['top_features'].items())[:10], 1):
                        self.features_text.insert(tk.END, f"{i:2d}. {feature:<25} {score:.3f}\n", "mono")
                    self.features_text.insert(tk.END, "\n")
                
                # Variance-based importance
                if 'variance_based' in importance:
                    self.features_text.insert(tk.END, "📊 Variance-Based Importance:\n", "subheader")
                    for feature, score in list(importance['variance_based'].items())[:5]:
                        self.features_text.insert(tk.END, f"• {feature}: {score:.3f}\n")
                    self.features_text.insert(tk.END, "\n")
                
                # Correlation-based importance
                if 'correlation_based' in importance:
                    self.features_text.insert(tk.END, "🔗 Health Correlation Importance:\n", "subheader")
                    for feature, score in list(importance['correlation_based'].items())[:5]:
                        self.features_text.insert(tk.END, f"• {feature}: {score:.3f}\n")
            
            else:
                self.features_text.insert(tk.END, "Feature importance analysis will be available after model training completes.")
            
            # Model performance summary
            if 'knn_performance' in stats:
                self.features_text.insert(tk.END, "\n🤖 KNN Model Performance:\n", "subheader")
                for model_name, perf in stats['knn_performance'].items():
                    cv_score = perf.get('cv_mean', 0)
                    n_samples = perf.get('n_samples', 0)
                    self.features_text.insert(tk.END, f"• {model_name}: CV={cv_score:.3f} (n={n_samples})\n")
            
            # Configure text tags
            self.features_text.tag_configure("header", font=self.fonts['subheading'], foreground=self.colors['primary'])
            self.features_text.tag_configure("subheader", font=self.fonts['body'], foreground=self.colors['secondary'])
            self.features_text.tag_configure("separator", foreground=self.colors['text_light'])
            self.features_text.tag_configure("mono", font=('Courier New', 8))
            
        except Exception as e:
            print(f"Error updating feature importance: {e}")
    
    def update_charts(self, recommendations):
        """Update visualization charts for KNN analysis"""
        # Skip if matplotlib is not available or charts failed to create
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'fig') or self.fig is None:
            return
        
        try:
            # Clear previous charts
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.clear()
        except:
            return  # Skip if axes are not available
        
        if not recommendations:
            # Show placeholder text
            try:
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                    ax.text(0.5, 0.5, 'No data\nto display', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                self.canvas.draw()
            except:
                pass  # Ignore errors in placeholder display
            return
        
        try:
            # Chart 1: KNN Distance Distribution
            try:
                distances = [rec.get('knn_distance', 0) for rec in recommendations]
                self.ax1.hist(distances, bins=max(3, len(distances)//2), color=self.colors['secondary'], 
                             alpha=0.7, edgecolor='black')
                self.ax1.set_title('KNN Distance Distribution', fontsize=10, fontweight='bold')
                self.ax1.set_xlabel('Distance')
                self.ax1.set_ylabel('Count')
            except Exception as e1:
                self.ax1.text(0.5, 0.5, 'Chart 1\nError', ha='center', va='center', transform=self.ax1.transAxes)
            
            # Chart 2: Health vs Nutrition Scores
            try:
                health_penalties = [rec.get('health_penalty', 0) for rec in recommendations]
                nutrition_scores = [rec.get('nutrition_match', 0) for rec in recommendations]
                
                scatter = self.ax2.scatter(health_penalties, nutrition_scores, 
                                         c=[rec['combined_score'] for rec in recommendations],
                                         cmap='RdYlGn_r', alpha=0.7, s=50)
                self.ax2.set_title('Health vs Nutrition Scores', fontsize=10, fontweight='bold')
                self.ax2.set_xlabel('Health Penalty')
                self.ax2.set_ylabel('Nutrition Match Score')
            except Exception as e2:
                self.ax2.text(0.5, 0.5, 'Chart 2\nError', ha='center', va='center', transform=self.ax2.transAxes)
            
            # Chart 3: Macronutrient Radar Chart (Top recommendation)
            if recommendations:
                top_rec = recommendations[0]
                try:
                    categories = ['Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar']
                    values = [
                        min(1.0, top_rec['protein'] / 30),  # Normalize to 0-1 scale
                        min(1.0, top_rec['carbs'] / 60),
                        min(1.0, top_rec['fat'] / 25),
                        min(1.0, top_rec['fiber'] / 15),
                        min(1.0, (15 - top_rec['sugar']) / 15)  # Invert sugar (lower is better)
                    ]
                    
                    # Ensure values are between 0 and 1
                    values = [max(0, min(1, v)) for v in values]
                    
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    values += values[:1]  # Complete the circle
                    angles += angles[:1]
                    
                    # Create simple polygon plot instead of polar
                    self.ax3.clear()
                    
                    # Convert polar coordinates to cartesian for regular plot
                    x_coords = [np.cos(angle) * value for angle, value in zip(angles, values)]
                    y_coords = [np.sin(angle) * value for angle, value in zip(angles, values)]
                    
                    self.ax3.plot(x_coords, y_coords, 'o-', linewidth=2, color=self.colors['success'])
                    self.ax3.fill(x_coords, y_coords, alpha=0.25, color=self.colors['success'])
                    
                    # Add category labels
                    for i, (angle, category) in enumerate(zip(angles[:-1], categories)):
                        x_label = np.cos(angle) * 1.2
                        y_label = np.sin(angle) * 1.2
                        self.ax3.text(x_label, y_label, category, ha='center', va='center', fontsize=8)
                    
                    self.ax3.set_xlim(-1.5, 1.5)
                    self.ax3.set_ylim(-1.5, 1.5)
                    self.ax3.set_aspect('equal')
                    self.ax3.set_title(f'Top Match:\n{top_rec["name"][:15]}...', fontsize=9, fontweight='bold')
                    self.ax3.grid(True, alpha=0.3)
                    self.ax3.set_xticks([])
                    self.ax3.set_yticks([])
                    
                except Exception as radar_error:
                    # Fallback to simple bar chart
                    try:
                        self.ax3.clear()
                        nutrients = ['Protein', 'Carbs', 'Fat', 'Fiber']
                        values = [top_rec['protein'], top_rec['carbs'], top_rec['fat'], top_rec['fiber']]
                        self.ax3.bar(nutrients, values, color=self.colors['success'], alpha=0.7)
                        self.ax3.set_title(f'Top Match: {top_rec["name"][:15]}', fontsize=9, fontweight='bold')
                        self.ax3.tick_params(axis='x', rotation=45, labelsize=8)
                    except:
                        self.ax3.text(0.5, 0.5, 'Chart 3\nError', ha='center', va='center', transform=self.ax3.transAxes)
            
            # Chart 4: Category Distribution
            try:
                categories = {}
                for rec in recommendations:
                    cat = rec['category']
                    categories[cat] = categories.get(cat, 0) + 1
                
                if categories:
                    cats = list(categories.keys())
                    counts = list(categories.values())
                    colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cats)))
                    
                    wedges, texts, autotexts = self.ax4.pie(counts, labels=cats, colors=colors_cat, 
                                                           autopct='%1.0f%%', startangle=90)
                    self.ax4.set_title('Food Categories', fontsize=10, fontweight='bold')
                    
                    # Adjust text size
                    for text in texts:
                        text.set_fontsize(8)
                    for autotext in autotexts:
                        autotext.set_fontsize(7)
            except Exception as e4:
                self.ax4.text(0.5, 0.5, 'Chart 4\nError', ha='center', va='center', transform=self.ax4.transAxes)
            
            # Chart 5: Model Comparison or Feature Importance
            try:
                model_info = self.recommender.get_model_info()
                if len(model_info) > 1:
                    models = list(model_info.keys())
                    cv_scores = [model_info[model]['performance'].get('cv_mean', 0) for model in models]
                    
                    bars = self.ax5.bar(range(len(models)), cv_scores, color=self.colors['info'], alpha=0.7)
                    self.ax5.set_title('KNN Model Performance', fontsize=10, fontweight='bold')
                    self.ax5.set_xlabel('Model')
                    self.ax5.set_ylabel('CV Score')
                    self.ax5.set_xticks(range(len(models)))
                    self.ax5.set_xticklabels([m[:8] for m in models], rotation=45, fontsize=8)
                    
                    # Highlight selected model
                    for i, model in enumerate(models):
                        if model == recommendations[0].get('knn_model_used', 'General'):
                            bars[i].set_color(self.colors['success'])
                            bars[i].set_alpha(1.0)
                else:
                    # Show feature importance instead
                    stats = self.recommender.get_stats()
                    if 'feature_importance' in stats and stats['feature_importance']:
                        importance = stats['feature_importance']
                        if 'top_features' in importance:
                            features = list(importance['top_features'].keys())[:5]
                            scores = list(importance['top_features'].values())[:5]
                            
                            bars = self.ax5.barh(range(len(features)), scores, color=self.colors['warning'], alpha=0.7)
                            self.ax5.set_title('Top Feature Importance', fontsize=10, fontweight='bold')
                            self.ax5.set_xlabel('Importance Score')
                            self.ax5.set_yticks(range(len(features)))
                            self.ax5.set_yticklabels([f[:12] for f in features], fontsize=8)
            except Exception as e5:
                self.ax5.text(0.5, 0.5, 'Chart 5\nError', ha='center', va='center', transform=self.ax5.transAxes)
            
        except Exception as e:
            print(f"Error updating charts: {e}")
        
        # Adjust layout and refresh
        try:
            self.fig.tight_layout()
            if self.canvas:
                self.canvas.draw()
        except Exception as canvas_error:
            print(f"Canvas drawing error: {canvas_error}")
    
    def show_model_info(self):
        """Show detailed KNN model information"""
        try:
            model_info = self.recommender.get_model_info()
            stats = self.recommender.get_stats()
            
            info_text = "🤖 KNN MODEL INFORMATION\n"
            info_text += "="*50 + "\n\n"
            
            # Dataset information
            info_text += f"📊 Dataset Statistics:\n"
            info_text += f"• Total food items: {stats['total_items']}\n"
            info_text += f"• Food categories: {len(stats['categories'])}\n"
            info_text += f"• Loading time: {stats['loading_time']:.2f} seconds\n\n"
            
            # Model information
            info_text += f"🤖 Trained KNN Models:\n"
            for model_name, info in model_info.items():
                info_text += f"\n{model_name} Model:\n"
                info_text += f"  • K (neighbors): {info['n_neighbors']}\n"
                info_text += f"  • Distance weighting: {info['weights']}\n"
                info_text += f"  • Distance metric: {info['metric']}\n"
                info_text += f"  • Number of features: {info['n_features']}\n"
                
                if 'performance' in info:
                    perf = info['performance']
                    info_text += f"  • Cross-validation score: {perf.get('cv_mean', 0):.4f} ± {perf.get('cv_std', 0):.4f}\n"
                    info_text += f"  • Training samples: {perf.get('n_samples', 0)}\n"
                    if 'best_params' in perf:
                        info_text += f"  • Optimized parameters: {perf['best_params']}\n"
            
            # Feature information
            if 'feature_importance' in stats:
                importance = stats['feature_importance']
                info_text += f"\n📈 Feature Analysis:\n"
                if 'top_features' in importance:
                    info_text += f"• Most important features:\n"
                    for i, (feature, score) in enumerate(list(importance['top_features'].items())[:5], 1):
                        info_text += f"  {i}. {feature}: {score:.3f}\n"
            
            # Show in message box
            messagebox.showinfo("KNN Model Information", info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error retrieving model information: {str(e)}")
    
    def update_stats_display(self):
        """Update statistics display"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Create stats labels
            total_items = stats['total_items']
            num_categories = len(stats['categories'])
            loading_time = stats['loading_time']
            num_models = len(self.recommender.knn_models) if hasattr(self.recommender, 'knn_models') else 0
            
            stats_text = f"Thai Foods: {total_items} | Categories: {num_categories} | KNN Models: {num_models} | Load Time: {loading_time:.1f}s"
            
            ttk.Label(self.stats_frame, text=stats_text, font=self.fonts['caption'],
                     foreground=self.colors['text_light']).pack()
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def reset_form(self):
        """Reset all form inputs to defaults"""
        self.weight_var.set(70.0)
        self.height_var.set(170.0)
        self.age_var.set(30)
        self.gender_var.set("Male")
        self.activity_var.set("Moderate")
        self.weight_goal_var.set("Maintain Weight")
        self.meal_type_var.set("Lunch")
        self.category_var.set("All")
        self.max_results_var.set("10")
        
        self.diabetes_var.set(False)
        self.obesity_var.set(False)
        self.hypertension_var.set(False)
        self.cholesterol_var.set(False)
        
        # Clear results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.targets_text.delete(1.0, tk.END)
        self.targets_text.insert(tk.END, "Click 'Calculate Targets' to see your personalized nutritional targets calculated using medical formulas (BMR, TDEE) and health condition guidelines (ADA, AHA, WHO, DASH).")
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "Select a food item from the KNN recommendations to view detailed nutritional information, health suitability analysis, and medical rationale.")
        
        self.knn_text.delete(1.0, tk.END)
        self.knn_text.insert(tk.END, "KNN analysis will appear here after getting recommendations, showing distance calculations, feature weights, and model selection rationale.")
        
        self.update_charts([])
        self.current_nutritional_data = None
        self.last_recommendations = []
        
        self.calculate_bmi()
        self.on_condition_change()
        
        self.status_var.set("✅ Form reset to defaults - Ready for new calculation")


def main():
    """Main function to run the KNN food recommendation application"""
    print("🚀 Starting Health-Driven KNN Food Recommendation System...")
    
    # Create main window
    root = tk.Tk()
    
    # Create splash screen
    splash = tk.Toplevel(root)
    splash.title("Loading KNN System...")
    splash.geometry("600x350")
    splash.resizable(False, False)
    splash.configure(bg="#f8f9fa")
    
    # Center splash screen
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() // 2) - (splash.winfo_width() // 2)
    y = (splash.winfo_screenheight() // 2) - (splash.winfo_height() // 2)
    splash.geometry(f"+{x}+{y}")
    
    # Splash content with modern styling
    splash_frame = ttk.Frame(splash, padding="50")
    splash_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title and description
    title_label = ttk.Label(splash_frame, text="🏥 Health-Driven KNN Food Recommendation System", 
                           font=('Segoe UI', 18, 'bold'), foreground="#2c3e50")
    title_label.pack(pady=20)
    
    subtitle_label = ttk.Label(splash_frame, text="K-Nearest Neighbors + Medical Target Calculation", 
                              font=('Segoe UI', 12), foreground="#8e44ad")
    subtitle_label.pack()
    
    desc_label = ttk.Label(splash_frame, text="Automatic BMR/TDEE calculation with health condition modifications\nusing WHO, ADA, AHA, and DASH guidelines", 
                          font=('Segoe UI', 10), foreground="#7f8c8d")
    desc_label.pack(pady=10)
    
    # University information
    uni_label = ttk.Label(splash_frame, text="Prince of Songkla University, Surat Thani Campus\nMaster's Thesis Research Project", 
                         font=('Segoe UI', 10), foreground="#34495e")
    uni_label.pack(pady=10)
    
    # Progress bar
    progress = ttk.Progressbar(splash_frame, mode='indeterminate', length=400)
    progress.pack(fill=tk.X, pady=20)
    progress.start()
    
    # Status label
    status_label = ttk.Label(splash_frame, text="Initializing system...", 
                            font=('Segoe UI', 10), foreground="#2c3e50")
    status_label.pack()
    
    # Hide main window initially
    root.withdraw()
    
    def update_splash_status(message):
        status_label.config(text=message)
        splash.update()
    
    def initialize_system():
        try:
            # Initialize KNN recommender system
            update_splash_status("🔄 Loading Thai food database...")
            
            try:
                recommender = HealthAwareKNNRecommender(update_splash_status)
                
                # Check if data was loaded successfully
                if len(recommender.food_data) == 0:
                    raise Exception("No valid food data could be loaded from CSV files")
                
                update_splash_status("🤖 Training KNN models for health conditions...")
                time.sleep(0.5)  # Brief pause for visual effect
                
                # Check if models were trained successfully
                if len(recommender.knn_models) == 0:
                    raise Exception("KNN model training failed - no models were created")
                
            except Exception as recommender_error:
                raise Exception(f"Recommender initialization failed: {str(recommender_error)}")
            
            update_splash_status("🎨 Building modern user interface...")
            time.sleep(0.3)
            
            # Create main UI
            try:
                app = HealthDrivenKNNFoodRecommenderUI(root, recommender)
            except Exception as ui_error:
                raise Exception(f"User interface creation failed: {str(ui_error)}")
            
            update_splash_status("✅ System ready!")
            time.sleep(0.5)
            
            # Show main window
            root.deiconify()
            root.lift()
            root.focus_force()
            
            # Close splash
            splash.destroy()
            
            print("✅ KNN Food Recommendation System initialized successfully!")
            print(f"📊 Loaded {len(recommender.food_data)} Thai food items")
            print(f"🤖 Trained {len(recommender.knn_models)} KNN models")
            
        except Exception as e:
            splash.destroy()
            
            # More specific error messages
            error_title = "KNN System Initialization Error"
            error_message = f"Failed to initialize the system:\n\n{str(e)}"
            
            if "No valid food data" in str(e):
                error_message += "\n\n💡 Troubleshooting:\n"
                error_message += "• Ensure CSV files contain nutritional data columns\n"
                error_message += "• Check that files are not empty or corrupted\n"
                error_message += "• Verify column names match expected format"
            elif "KNN model training failed" in str(e):
                error_message += "\n\n💡 Troubleshooting:\n"
                error_message += "• Data may have insufficient nutritional information\n"
                error_message += "• Try with a larger dataset\n"
                error_message += "• Check for data quality issues"
            elif "User interface creation failed" in str(e):
                error_message += "\n\n💡 Troubleshooting:\n"
                error_message += "• GUI libraries may not be properly installed\n"
                error_message += "• Try running in a different environment\n"
                error_message += "• Check tkinter installation"
            else:
                error_message += "\n\n💡 General troubleshooting:\n"
                error_message += "• Ensure all required libraries are installed\n"
                error_message += "• Check that Thai food CSV files are available\n"
                error_message += "• Try running with administrator privileges"
            
            messagebox.showerror(error_title, error_message)
            print(f"❌ Error: {str(e)}")
            root.quit()
    
    # Schedule initialization
    root.after(100, initialize_system)
    
    # Configure main window
    root.title("Health-Driven KNN Food Recommendation System")
    root.geometry("1600x900")
    root.configure(bg="#f8f9fa")
    
    # Center main window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start application
    root.mainloop()


if __name__ == "__main__":
    main()