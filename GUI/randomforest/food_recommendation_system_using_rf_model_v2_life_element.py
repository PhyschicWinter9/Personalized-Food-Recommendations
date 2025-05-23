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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI integration
warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class ThaiLifeElementCalculator:
    """Thai traditional medicine life element calculator based on birth month and year"""
    
    def __init__(self):
        # Thai life elements and their properties
        self.elements = {
            'Earth': {
                'thai_name': 'ธาตุดิน',
                'characteristics': ['Stability', 'Grounding', 'Nurturing', 'Slow metabolism'],
                'dietary_focus': ['Warm foods', 'Cooked vegetables', 'Grains', 'Root vegetables'],
                'avoid': ['Too much raw food', 'Cold drinks', 'Excessive dairy'],
                'beneficial_nutrients': ['Complex carbs', 'Fiber', 'Iron', 'B vitamins'],
                'meal_timing': 'Regular, consistent meal times',
                'color_preference': ['Yellow', 'Brown', 'Orange']
            },
            'Water': {
                'thai_name': 'ธาตุน้ำ',
                'characteristics': ['Fluidity', 'Cooling', 'Cleansing', 'Sensitive digestion'],
                'dietary_focus': ['Hydrating foods', 'Fresh fruits', 'Soups', 'Mild flavors'],
                'avoid': ['Spicy food', 'Alcohol', 'Excessive salt'],
                'beneficial_nutrients': ['Potassium', 'Magnesium', 'Antioxidants', 'Natural sugars'],
                'meal_timing': 'Small, frequent meals',
                'color_preference': ['Blue', 'White', 'Clear']
            },
            'Wind': {
                'thai_name': 'ธาตุลม',
                'characteristics': ['Movement', 'Quick energy', 'Variable appetite', 'Active metabolism'],
                'dietary_focus': ['Protein-rich foods', 'Nuts', 'Seeds', 'Warming spices'],
                'avoid': ['Irregular eating', 'Too much caffeine', 'Cold foods'],
                'beneficial_nutrients': ['Protein', 'Healthy fats', 'Calcium', 'Vitamin D'],
                'meal_timing': 'Regular meals with healthy snacks',
                'color_preference': ['Green', 'Purple', 'Blue-green']
            },
            'Fire': {
                'thai_name': 'ธาตุไฟ',
                'characteristics': ['Heat', 'Intensity', 'Strong digestion', 'High energy'],
                'dietary_focus': ['Cooling foods', 'Raw vegetables', 'Fresh herbs', 'Lean proteins'],
                'avoid': ['Excessive spice', 'Fried foods', 'Too much heat'],
                'beneficial_nutrients': ['Vitamin C', 'Beta-carotene', 'Lean protein', 'Cooling minerals'],
                'meal_timing': 'Regular meals, avoid overeating',
                'color_preference': ['Red', 'Pink', 'Light colors']
            }
        }
        
        # Element calculation matrix based on month and year modulo
        self.element_matrix = {
            (0, 0): 'Earth', (0, 1): 'Water', (0, 2): 'Wind', (0, 3): 'Fire',
            (1, 0): 'Water', (1, 1): 'Wind', (1, 2): 'Fire', (1, 3): 'Earth',
            (2, 0): 'Wind', (2, 1): 'Fire', (2, 2): 'Earth', (2, 3): 'Water',
            (3, 0): 'Fire', (3, 1): 'Earth', (3, 2): 'Water', (3, 3): 'Wind',
            (4, 0): 'Earth', (4, 1): 'Water', (4, 2): 'Wind', (4, 3): 'Fire',
            (5, 0): 'Water', (5, 1): 'Wind', (5, 2): 'Fire', (5, 3): 'Earth',
            (6, 0): 'Wind', (6, 1): 'Fire', (6, 2): 'Earth', (6, 3): 'Water',
            (7, 0): 'Fire', (7, 1): 'Earth', (7, 2): 'Water', (7, 3): 'Wind',
            (8, 0): 'Earth', (8, 1): 'Water', (8, 2): 'Wind', (8, 3): 'Fire',
            (9, 0): 'Water', (9, 1): 'Wind', (9, 2): 'Fire', (9, 3): 'Earth',
            (10, 0): 'Wind', (10, 1): 'Fire', (10, 2): 'Earth', (10, 3): 'Water',
            (11, 0): 'Fire', (11, 1): 'Earth', (11, 2): 'Water', (11, 3): 'Wind'
        }
    
    def calculate_life_element(self, birth_month, birth_year):
        """Calculate life element based on Thai traditional medicine"""
        try:
            # Convert month to 0-based index
            month_index = (birth_month - 1) % 12
            year_index = birth_year % 4
            
            element = self.element_matrix.get((month_index, year_index), 'Earth')
            return element
        except:
            return 'Earth'  # Default fallback
    
    def get_element_info(self, element):
        """Get detailed information about a life element"""
        return self.elements.get(element, self.elements['Earth'])
    
    def get_dietary_recommendations(self, element):
        """Get dietary recommendations for a specific element"""
        element_info = self.get_element_info(element)
        return {
            'focus_foods': element_info['dietary_focus'],
            'avoid_foods': element_info['avoid'],
            'beneficial_nutrients': element_info['beneficial_nutrients'],
            'meal_timing': element_info['meal_timing']
        }
    
    def calculate_element_food_score(self, food_data, element):
        """Calculate how well a food matches the element's needs"""
        element_info = self.get_element_info(element)
        score = 0
        
        # Element-specific scoring logic
        if element == 'Earth':
            # Prefer warming, grounding foods
            carbs = float(food_data.get('CHOCDF (g) Carbohydrate', 0))
            fiber = float(food_data.get('FIBTG (g) Dietary fibre', 0))
            
            if carbs > 15:  # Good carb content
                score += 1
            if fiber > 3:   # Good fiber for grounding
                score += 1
            
            # Prefer cooked over raw (lower sodium as proxy for processed/cooked)
            sodium = float(food_data.get('Na(mg)', 0))
            if 50 < sodium < 300:  # Moderate sodium indicates cooked but not over-processed
                score += 0.5
                
        elif element == 'Water':
            # Prefer hydrating, cooling foods
            potassium = float(food_data.get('K(mg)', 0))
            sodium = float(food_data.get('Na(mg)', 0))
            sugar = float(food_data.get('SUGAR(g)', 0))
            
            if potassium > 200:  # Good potassium for hydration
                score += 1
            if sodium < 100:     # Low sodium
                score += 1
            if 2 < sugar < 10:   # Natural sugars from fruits
                score += 0.5
                
        elif element == 'Wind':
            # Prefer protein-rich, stabilizing foods
            protein = float(food_data.get('Protein(g)', 0))
            fat = float(food_data.get('Fat(g)', 0))
            
            if protein > 8:      # Good protein content
                score += 1
            if 5 < fat < 15:     # Moderate healthy fats
                score += 1
                
        elif element == 'Fire':
            # Prefer cooling, light foods
            calories = float(food_data.get('Energy(kcal) by calculation', 0))
            fiber = float(food_data.get('FIBTG (g) Dietary fibre', 0))
            fat = float(food_data.get('Fat(g)', 0))
            
            if calories < 200:   # Lower calorie density
                score += 1
            if fiber > 4:        # High fiber
                score += 1
            if fat < 8:          # Lower fat content
                score += 0.5
        
        return max(0, score)


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
    """Random Forest-based food recommender with health condition awareness and Thai life elements"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        self.life_element_calculator = ThaiLifeElementCalculator()
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'model_performance': {}
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
                
                # Add life element scores
                self._calculate_life_element_scores()
                
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
    
    def _calculate_life_element_scores(self):
        """Calculate life element suitability scores for each food item"""
        self.update_status("Calculating Thai life element suitability scores...")
        
        # Initialize element score columns
        elements = ['Earth', 'Water', 'Wind', 'Fire']
        for element in elements:
            self.food_data[f'{element}_Element_Score'] = 0
        
        for idx, food in self.food_data.iterrows():
            # Calculate scores for each element
            for element in elements:
                score = self.life_element_calculator.calculate_element_food_score(food, element)
                self.food_data.at[idx, f'{element}_Element_Score'] = score
    
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
        
        # Add life element features
        element_features = ['Earth_Element_Score', 'Water_Element_Score', 'Wind_Element_Score', 'Fire_Element_Score']
        self.features.extend(element_features)
        
        self.update_status(f"Prepared {len(self.features)} features for Random Forest model")
    
    def train_models(self):
        """Train Random Forest models for different prediction tasks"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for training")
            return
        
        try:
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Multiple target variables for different aspects
            targets = {
                'Overall_Health': self.food_data['Overall_Health_Score'],
                'Diabetes_Suitability': self.food_data['Diabetes_Score'],
                'Obesity_Suitability': self.food_data['Obesity_Score'],
                'Hypertension_Suitability': self.food_data['Hypertension_Score'],
                'Cholesterol_Suitability': self.food_data['High_Cholesterol_Score']
            }
            
            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models for each target
            self.models = {}
            self.update_status("Training Random Forest models...")
            
            for target_name, y in targets.items():
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Train Random Forest
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = rf_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
                
                self.models[target_name] = rf_model
                
                self.stats['model_performance'][target_name] = {
                    'r2_score': r2,
                    'mse': mse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                self.update_status(f"{target_name} model - R²: {r2:.3f}, CV: {cv_scores.mean():.3f}")
            
            # Get feature importances from overall health model
            if 'Overall_Health' in self.models:
                feature_importances = self.models['Overall_Health'].feature_importances_
                importance_dict = dict(zip(self.features, feature_importances))
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                self.update_status("Top 5 important features:")
                for feature, importance in sorted_importance[:5]:
                    self.update_status(f"  {feature}: {importance:.3f}")
            
        except Exception as e:
            self.update_status(f"Error training models: {e}")
    
    def get_recommendations(self, user_profile, meal_type='meal', max_recommendations=10):
        """Get personalized food recommendations based on user profile and life element"""
        if not hasattr(self, 'models') or len(self.food_data) == 0:
            return []
        
        try:
            # Calculate nutritional targets using medical calculator
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            daily_targets = nutritional_data['daily_targets']
            meal_targets = nutritional_data['meal_targets'].get(meal_type.lower(), 
                                                              nutritional_data['meal_targets']['lunch'])
            
            # Calculate life element
            birth_month = user_profile.get('birth_month', 1)
            birth_year = user_profile.get('birth_year', 1990)
            life_element = self.life_element_calculator.calculate_life_element(birth_month, birth_year)
            
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
                
                # Calculate life element bonus
                element_bonus = self._calculate_life_element_bonus(food, life_element)
                
                # Combined score (lower is better, except element bonus)
                combined_score = nutrition_score + health_penalty - element_bonus
                
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
                    'element_bonus': element_bonus,
                    'combined_score': combined_score,
                    'life_element': life_element,
                    'suitable_for_conditions': self._check_condition_suitability(food, health_conditions),
                    'targets_met': self._check_targets_met(food, meal_targets),
                    'health_warnings': self._generate_health_warnings(food, health_conditions),
                    'element_match': self._check_element_suitability(food, life_element)
                }
                
                scores.append(recommendation)
            
            # Sort by combined score (lower is better)
            scores.sort(key=lambda x: x['combined_score'])
            
            # Return top recommendations
            recommendations = scores[:max_recommendations]
            
            # Add explanation data
            for rec in recommendations:
                rec['explanation'] = self._generate_explanation(rec, meal_targets, health_conditions, life_element)
                rec['nutritional_data'] = nutritional_data
                rec['life_element_info'] = self.life_element_calculator.get_element_info(life_element)
            
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
    
    def _calculate_life_element_bonus(self, food, life_element):
        """Calculate bonus based on life element match"""
        element_score = float(food.get(f'{life_element}_Element_Score', 0))
        return element_score  # Higher is better, so this is a bonus
    
    def _check_element_suitability(self, food, life_element):
        """Check how well food matches the life element"""
        element_score = float(food.get(f'{life_element}_Element_Score', 0))
        
        if element_score >= 2:
            return 'Excellent'
        elif element_score >= 1:
            return 'Good'
        elif element_score >= 0.5:
            return 'Fair'
        else:
            return 'Poor'
    
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
    
    def _generate_explanation(self, recommendation, targets, health_conditions, life_element):
        """Generate explanation for why this food was recommended"""
        explanations = []
        
        # Nutritional match explanations
        if 'Calories' in recommendation['targets_met']:
            explanations.append(f"Good calorie match ({recommendation['calories']:.0f} kcal)")
        
        if 'Protein' in recommendation['targets_met']:
            explanations.append(f"Adequate protein ({recommendation['protein']:.1f}g)")
        
        if 'Fiber' in recommendation['targets_met']:
            explanations.append(f"Good fiber content ({recommendation['fiber']:.1f}g)")
        
        # Life element explanation
        element_match = recommendation['element_match']
        if element_match in ['Excellent', 'Good']:
            explanations.append(f"{element_match} match for {life_element} element")
        
        # Health condition explanations
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            conditions_text = ", ".join(suitable_conditions)
            explanations.append(f"Suitable for {conditions_text}")
        
        # Score explanation
        if recommendation['combined_score'] < 2:
            explanations.append("Excellent overall match")
        elif recommendation['combined_score'] < 4:
            explanations.append("Good overall match")
        else:
            explanations.append("Fair overall match")
        
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


class HealthDrivenFoodRecommenderUI:
    """Modern GUI for the health-driven food recommendation system with Thai life elements"""
    
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Health-Driven Food Recommendation System with Thai Life Elements")
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
            'text_light': '#7f8c8d',
            'thai_gold': '#FFD700',     # Thai gold
            'thai_red': '#FF6B6B'       # Thai red
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
        self.current_life_element = None
        
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
            'caption': ('Segoe UI', 9),
            'thai': ('Segoe UI', 11, 'bold')
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=self.fonts['heading'], 
                           foreground=self.colors['primary'])
        self.style.configure('Subtitle.TLabel', font=self.fonts['subheading'], 
                           foreground=self.colors['text'])
        self.style.configure('Thai.TLabel', font=self.fonts['thai'], 
                           foreground=self.colors['thai_gold'])
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
        
        # Create three-panel layout
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
        ttk.Label(title_frame, text="Medical Guidelines + Thai Traditional Medicine Life Elements (ธาตุ)", 
                 style='Thai.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Automatic Nutritional Calculation & Life Element Analysis", 
                 style='Subtitle.TLabel', foreground=self.colors['text_light']).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self.get_recommendations, style='Primary.TButton').pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_form).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Calculate Nutrition + Element", 
                  command=self.show_nutrition_calculation).pack(side=tk.RIGHT, padx=5)
    
    def create_input_panel(self, parent):
        """Create input panel with health profile form and life element"""
        # Left panel for inputs
        left_panel = ttk.LabelFrame(parent, text="Health Profile + Life Element", padding="15")
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
        ttk.Label(personal_frame, textvariable=self.bmi_var).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Thai Life Element Section
        element_frame = ttk.LabelFrame(left_panel, text="Thai Life Element (ธาตุ)", padding="10")
        element_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Birth month
        ttk.Label(element_frame, text="Birth Month:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.birth_month_var = tk.IntVar(value=1)
        month_combo = ttk.Combobox(element_frame, textvariable=self.birth_month_var,
                                  values=list(range(1, 13)), state="readonly", width=15)
        month_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Birth year
        ttk.Label(element_frame, text="Birth Year:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.birth_year_var = tk.IntVar(value=1990)
        year_combo = ttk.Combobox(element_frame, textvariable=self.birth_year_var,
                                 values=list(range(1950, 2010)), state="readonly", width=15)
        year_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Life element display
        ttk.Label(element_frame, text="Life Element:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.life_element_var = tk.StringVar(value="Calculate to see")
        ttk.Label(element_frame, textvariable=self.life_element_var, 
                 style='Thai.TLabel').grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Health Conditions
        conditions_frame = ttk.LabelFrame(left_panel, text="Health Conditions", padding="10")
        conditions_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        ttk.Checkbutton(conditions_frame, text="Diabetes", variable=self.diabetes_var).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="Obesity", variable=self.obesity_var).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="Hypertension", variable=self.hypertension_var).pack(anchor=tk.W)
        ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var).pack(anchor=tk.W)
        
        # Settings
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="Meal Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.meal_type_var = tk.StringVar(value="Lunch")
        meal_combo = ttk.Combobox(settings_frame, textvariable=self.meal_type_var,
                                 values=['Breakfast', 'Lunch', 'Dinner', 'Snack'],
                                 state="readonly", width=15)
        meal_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(settings_frame, text="Category Filter:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.category_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_var,
                                          state="readonly", width=15)
        self.category_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.update_category_list()
        
        ttk.Label(settings_frame, text="Max Results:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_results_var = tk.StringVar(value="10")
        results_combo = ttk.Combobox(settings_frame, textvariable=self.max_results_var,
                                    values=['5', '10', '15', '20', '25'],
                                    state="readonly", width=15)
        results_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Bind events for BMI calculation and life element
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.birth_month_var.trace_add("write", self.calculate_life_element)
        self.birth_year_var.trace_add("write", self.calculate_life_element)
        self.calculate_bmi()
        self.calculate_life_element()
    
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
                elif bmi < 25:
                    category = "Normal"
                elif bmi < 30:
                    category = "Overweight"
                else:
                    category = "Obese"
                
                self.bmi_var.set(f"{bmi:.1f} ({category})")
            else:
                self.bmi_var.set("Invalid input")
        except:
            self.bmi_var.set("Calculating...")
    
    def calculate_life_element(self, *args):
        """Calculate and display life element"""
        try:
            birth_month = self.birth_month_var.get()
            birth_year = self.birth_year_var.get()
            
            if hasattr(self.recommender, 'life_element_calculator'):
                element = self.recommender.life_element_calculator.calculate_life_element(birth_month, birth_year)
                element_info = self.recommender.life_element_calculator.get_element_info(element)
                self.life_element_var.set(f"{element} ({element_info['thai_name']})")
                self.current_life_element = element
            else:
                self.life_element_var.set("Calculator not available")
        except:
            self.life_element_var.set("Calculate to see")
    
    def update_category_list(self):
        """Update the category dropdown with available categories"""
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
            if 'Category' in self.recommender.food_data.columns:
                categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
    
    def create_results_panel(self, parent):
        """Create results panel with recommendations table"""
        # Right panel for results
        right_panel = ttk.LabelFrame(parent, text="Food Recommendations", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Recommendations tab
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="Recommendations")
        
        # Create treeview for recommendations
        columns = ('Name', 'Category', 'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Score', 'Element Match', 'Suitable For')
        self.tree = ttk.Treeview(rec_frame, columns=columns, show='headings', 
                               style='Health.Treeview', height=15)
        
        # Configure columns
        column_widths = {'Name': 130, 'Category': 90, 'Calories': 60, 'Protein': 50, 'Carbs': 50, 
                        'Sugar': 45, 'Fiber': 45, 'Score': 50, 'Element Match': 80, 'Suitable For': 100}
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(rec_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(rec_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Nutritional targets tab
        targets_frame = ttk.Frame(self.notebook)
        self.notebook.add(targets_frame, text="Calculated Targets + Element")
        
        # Create text widget for targets
        self.targets_text = tk.Text(targets_frame, wrap=tk.WORD, font=self.fonts['body'])
        targets_scrollbar = ttk.Scrollbar(targets_frame, orient="vertical", command=self.targets_text.yview)
        self.targets_text.configure(yscrollcommand=targets_scrollbar.set)
        
        self.targets_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        targets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Food Details")
        
        # Create text widget for details
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, font=self.fonts['body'])
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
        # Initial text
        self.targets_text.insert(tk.END, "Click 'Calculate Nutrition + Element' to see your personalized nutritional targets and Thai life element analysis.")
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information, health suitability, and life element compatibility.")
    
    def create_charts_panel(self, parent):
        """Create charts panel for visualizations"""
        charts_frame = ttk.LabelFrame(parent, text="Nutritional Analysis + Life Element Distribution", padding="10")
        charts_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(15, 4), dpi=100)
        self.fig.patch.set_facecolor(self.colors['white'])
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(151)
        self.ax2 = self.fig.add_subplot(152)
        self.ax3 = self.fig.add_subplot(153)
        self.ax4 = self.fig.add_subplot(154)
        self.ax5 = self.fig.add_subplot(155)
        
        # Initialize charts
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_charts([])
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding="5")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready - Enter your health profile and birth info, then click 'Calculate Nutrition + Element'")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)
        
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
            'birth_month': self.birth_month_var.get(),
            'birth_year': self.birth_year_var.get(),
            'category_filter': self.category_var.get()
        }
    
    def show_nutrition_calculation(self):
        """Show the calculated nutritional targets and life element info"""
        try:
            user_profile = self.get_user_profile()
            
            # Calculate nutritional targets
            nutritional_data = self.recommender.nutrition_calculator.calculate_nutritional_targets(user_profile)
            self.current_nutritional_data = nutritional_data
            
            # Calculate life element
            birth_month = user_profile['birth_month']
            birth_year = user_profile['birth_year']
            life_element = self.recommender.life_element_calculator.calculate_life_element(birth_month, birth_year)
            element_info = self.recommender.life_element_calculator.get_element_info(life_element)
            self.current_life_element = life_element
            
            # Display in targets tab
            self.targets_text.delete(1.0, tk.END)
            
            # Add header
            self.targets_text.insert(tk.END, "PERSONALIZED NUTRITIONAL TARGETS + THAI LIFE ELEMENT\n", "header")
            self.targets_text.insert(tk.END, "="*60 + "\n\n", "separator")
            
            # Add life element section
            self.targets_text.insert(tk.END, f"Thai Life Element (ธาตุ): {life_element} ({element_info['thai_name']})\n", "thai_header")
            self.targets_text.insert(tk.END, "-"*40 + "\n", "separator")
            self.targets_text.insert(tk.END, f"Characteristics: {', '.join(element_info['characteristics'])}\n")
            self.targets_text.insert(tk.END, f"Dietary Focus: {', '.join(element_info['dietary_focus'])}\n")
            self.targets_text.insert(tk.END, f"Avoid: {', '.join(element_info['avoid'])}\n")
            self.targets_text.insert(tk.END, f"Beneficial Nutrients: {', '.join(element_info['beneficial_nutrients'])}\n")
            self.targets_text.insert(tk.END, f"Meal Timing: {element_info['meal_timing']}\n\n")
            
            # Add calculations
            calc = nutritional_data['calculations']
            self.targets_text.insert(tk.END, "Medical Calculations:\n", "subheader")
            self.targets_text.insert(tk.END, f"• {calc['bmr_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['tdee_formula']}\n")
            self.targets_text.insert(tk.END, f"• {calc['target_formula']}\n\n")
            
            # Add daily targets
            daily = nutritional_data['daily_targets']
            self.targets_text.insert(tk.END, "Daily Nutritional Targets:\n", "subheader")
            self.targets_text.insert(tk.END, f"• Calories: {daily['calories']:.0f} kcal\n")
            self.targets_text.insert(tk.END, f"• Protein: {daily['protein_min']:.0f}-{daily['protein_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Carbohydrates: {daily['carbs_min']:.0f}-{daily['carbs_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Sugar (max): {daily['sugar_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Fiber (min): {daily['fiber_min']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Fat: {daily['fat_min']:.0f}-{daily['fat_max']:.0f} g\n")
            self.targets_text.insert(tk.END, f"• Sodium (max): {daily['sodium_max']:.0f} mg\n\n")
            
            # Add meal targets
            meal_type = self.meal_type_var.get().lower()
            if meal_type in nutritional_data['meal_targets']:
                meal = nutritional_data['meal_targets'][meal_type]
                self.targets_text.insert(tk.END, f"{self.meal_type_var.get()} Targets:\n", "subheader")
                self.targets_text.insert(tk.END, f"• Calories: {meal['calories']:.0f} kcal\n")
                self.targets_text.insert(tk.END, f"• Protein: {meal['protein_min']:.0f}-{meal['protein_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Carbohydrates: {meal['carbs_min']:.0f}-{meal['carbs_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Sugar (max): {meal['sugar_max']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fiber (min): {meal['fiber_min']:.0f} g\n")
                self.targets_text.insert(tk.END, f"• Fat: {meal['fat_min']:.0f}-{meal['fat_max']:.0f} g\n\n")
            
            # Add health condition info
            health_conditions = user_profile['health_conditions']
            if health_conditions:
                self.targets_text.insert(tk.END, "Health Condition Adjustments:\n", "subheader")
                for condition in health_conditions:
                    self.targets_text.insert(tk.END, f"• {condition}: Applied medical guidelines\n")
            
            # Configure text tags for formatting
            self.targets_text.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.targets_text.tag_configure("thai_header", font=self.fonts['thai'], foreground=self.colors['thai_gold'])
            self.targets_text.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.targets_text.tag_configure("separator", foreground=self.colors['text_light'])
            
            # Switch to targets tab
            self.notebook.select(1)
            
            self.status_var.set(f"Nutritional targets calculated for {life_element} element using medical guidelines")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating nutrition targets: {str(e)}")
    
    def get_recommendations(self):
        """Get food recommendations based on calculated targets and life element"""
        try:
            if not self.current_nutritional_data:
                messagebox.showwarning("Warning", "Please calculate nutrition targets and life element first!")
                return
            
            user_profile = self.get_user_profile()
            meal_type = self.meal_type_var.get().lower()
            max_results = int(self.max_results_var.get())
            
            self.status_var.set(f"Finding optimal food matches for {self.current_life_element} element...")
            
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
                for rec in recommendations:
                    suitable_text = ", ".join(rec['suitable_for_conditions']) if rec['suitable_for_conditions'] else "General"
                    
                    self.tree.insert('', 'end', values=(
                        rec['name'][:18],  # Truncate long names
                        rec['category'],
                        f"{rec['calories']:.0f}",
                        f"{rec['protein']:.1f}",
                        f"{rec['carbs']:.1f}",
                        f"{rec['sugar']:.1f}",
                        f"{rec['fiber']:.1f}",
                        f"{rec['combined_score']:.1f}",
                        rec['element_match'],
                        suitable_text
                    ))
                
                # Select first item
                if self.tree.get_children():
                    first_item = self.tree.get_children()[0]
                    self.tree.selection_set(first_item)
                    self.tree.focus(first_item)
                    self.show_food_details(None)
                
                # Update charts
                self.update_charts(recommendations)
                
                # Switch to recommendations tab
                self.notebook.select(0)
                
                health_conditions = user_profile['health_conditions']
                condition_text = ", ".join(health_conditions) if health_conditions else "general health"
                self.status_var.set(f"Found {len(recommendations)} optimal foods for {self.current_life_element} element + {condition_text}")
                
            else:
                messagebox.showinfo("No Results", "No suitable foods found matching your criteria.")
                self.status_var.set("No suitable foods found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {str(e)}")
            self.status_var.set("Error occurred during recommendation")
    
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
            self.details_text.insert(tk.END, f"{rec['name']}\n", "title")
            self.details_text.insert(tk.END, f"Category: {rec['category']}\n\n", "subtitle")
            
            # Add life element compatibility
            self.details_text.insert(tk.END, f"Life Element Compatibility ({rec['life_element']}):\n", "thai_header")
            self.details_text.insert(tk.END, f"Match Quality: {rec['element_match']}\n", "element")
            self.details_text.insert(tk.END, f"Element Bonus: {rec['element_bonus']:.1f}\n\n", "element")
            
            # Add nutritional information
            self.details_text.insert(tk.END, "Nutritional Information (per 100g):\n", "header")
            self.details_text.insert(tk.END, f"• Energy: {rec['calories']:.0f} kcal\n")
            self.details_text.insert(tk.END, f"• Protein: {rec['protein']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Carbohydrates: {rec['carbs']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sugar: {rec['sugar']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Dietary Fiber: {rec['fiber']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Fat: {rec['fat']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sodium: {rec['sodium']:.0f} mg\n")
            self.details_text.insert(tk.END, f"• Potassium: {rec['potassium']:.0f} mg\n")
            self.details_text.insert(tk.END, f"• Cholesterol: {rec['cholesterol']:.0f} mg\n\n")
            
            # Add health suitability
            if rec['suitable_for_conditions']:
                self.details_text.insert(tk.END, "Suitable for Health Conditions:\n", "good")
                for condition in rec['suitable_for_conditions']:
                    self.details_text.insert(tk.END, f"✓ {condition}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Add health warnings
            if rec['health_warnings']:
                self.details_text.insert(tk.END, "Health Considerations:\n", "warning")
                for warning in rec['health_warnings']:
                    self.details_text.insert(tk.END, f"⚠ {warning}\n", "warning")
                self.details_text.insert(tk.END, "\n")
            
            # Add targets met
            if rec['targets_met']:
                self.details_text.insert(tk.END, "Nutritional Targets Met:\n", "good")
                for target in rec['targets_met']:
                    self.details_text.insert(tk.END, f"✓ {target}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Add life element information
            if 'life_element_info' in rec:
                element_info = rec['life_element_info']
                self.details_text.insert(tk.END, f"Life Element ({rec['life_element']}) Recommendations:\n", "thai_header")
                self.details_text.insert(tk.END, f"• Focus Foods: {', '.join(element_info['dietary_focus'])}\n")
                self.details_text.insert(tk.END, f"• Avoid: {', '.join(element_info['avoid'])}\n")
                self.details_text.insert(tk.END, f"• Meal Timing: {element_info['meal_timing']}\n\n")
            
            # Add recommendation explanation
            self.details_text.insert(tk.END, "Why This Food Was Recommended:\n", "header")
            self.details_text.insert(tk.END, f"{rec['explanation']}\n\n")
            
            # Add scoring details
            self.details_text.insert(tk.END, "Recommendation Score Details:\n", "header")
            self.details_text.insert(tk.END, f"• Nutritional Match Score: {rec['nutrition_score']:.2f}\n")
            self.details_text.insert(tk.END, f"• Health Condition Penalty: {rec['health_penalty']:.2f}\n")
            self.details_text.insert(tk.END, f"• Life Element Bonus: {rec['element_bonus']:.2f}\n")
            self.details_text.insert(tk.END, f"• Combined Score: {rec['combined_score']:.2f} (lower is better)\n")
            
            # Configure text tags
            self.details_text.tag_configure("title", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.details_text.tag_configure("subtitle", font=self.fonts['subheading'], foreground=self.colors['text'])
            self.details_text.tag_configure("header", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.details_text.tag_configure("thai_header", font=self.fonts['thai'], foreground=self.colors['thai_gold'])
            self.details_text.tag_configure("element", foreground=self.colors['thai_red'])
            self.details_text.tag_configure("good", foreground=self.colors['success'])
            self.details_text.tag_configure("warning", foreground=self.colors['warning'])
    
    def update_charts(self, recommendations):
        """Update visualization charts including life element analysis"""
        # Clear previous charts
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.clear()
        
        if not recommendations:
            # Show placeholder text
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
                ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        try:
            # Chart 1: Macronutrient Distribution
            avg_protein = np.mean([r['protein'] for r in recommendations])
            avg_carbs = np.mean([r['carbs'] for r in recommendations])
            avg_fat = np.mean([r['fat'] for r in recommendations])
            
            labels = ['Protein', 'Carbs', 'Fat']
            sizes = [avg_protein * 4, avg_carbs * 4, avg_fat * 9]  # Convert to calories
            colors = [self.colors['success'], self.colors['warning'], self.colors['danger']]
            
            if sum(sizes) > 0:
                self.ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                self.ax1.set_title('Avg Macronutrient Distribution', fontsize=10, fontweight='bold')
            
            # Chart 2: Health Score Distribution
            scores = [r['combined_score'] for r in recommendations]
            self.ax2.hist(scores, bins=5, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            self.ax2.set_title('Recommendation Score Distribution', fontsize=10, fontweight='bold')
            self.ax2.set_xlabel('Score (lower is better)')
            self.ax2.set_ylabel('Count')
            
            # Chart 3: Category Distribution
            categories = {}
            for r in recommendations:
                cat = r['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            if categories:
                cats = list(categories.keys())
                counts = list(categories.values())
                colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cats)))
                
                bars = self.ax3.bar(cats, counts, color=colors_cat)
                self.ax3.set_title('Food Categories', fontsize=10, fontweight='bold')
                self.ax3.set_ylabel('Count')
                
                # Rotate labels if needed
                if len(max(cats, key=len)) > 8:
                    self.ax3.tick_params(axis='x', rotation=45)
            
            # Chart 4: Health Condition Suitability
            condition_counts = {'Diabetes': 0, 'Obesity': 0, 'Hypertension': 0, 'High_Cholesterol': 0}
            
            for r in recommendations:
                for condition in r['suitable_for_conditions']:
                    if condition in condition_counts:
                        condition_counts[condition] += 1
            
            # Calculate percentages
            total = len(recommendations)
            condition_labels = [c.replace('_', ' ') for c in condition_counts.keys()]
            condition_percentages = [count/total*100 for count in condition_counts.values()]
            
            bars = self.ax4.bar(condition_labels, condition_percentages, color=self.colors['success'], alpha=0.7)
            self.ax4.set_title('Condition Suitability (%)', fontsize=10, fontweight='bold')
            self.ax4.set_ylabel('Percentage of Foods')
            self.ax4.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, condition_percentages):
                if pct > 0:
                    self.ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                 f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)
            
            # Rotate labels
            self.ax4.tick_params(axis='x', rotation=45)
            
            # Chart 5: Life Element Match Distribution
            element_matches = {}
            for r in recommendations:
                match = r['element_match']
                element_matches[match] = element_matches.get(match, 0) + 1
            
            if element_matches:
                match_labels = list(element_matches.keys())
                match_counts = list(element_matches.values())
                # Use Thai-inspired colors
                thai_colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#95E1D3'][:len(match_labels)]
                
                bars = self.ax5.bar(match_labels, match_counts, color=thai_colors)
                self.ax5.set_title(f'Life Element Match ({self.current_life_element})', fontsize=10, fontweight='bold')
                self.ax5.set_ylabel('Count')
                
                # Add count labels on bars
                for bar, count in zip(bars, match_counts):
                    self.ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                 str(count), ha='center', va='bottom', fontsize=8)
            
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
                widget.destroy()
            
            # Create stats labels
            total_items = stats['total_items']
            num_categories = len(stats['categories'])
            loading_time = stats['loading_time']
            
            stats_text = f"Loaded: {total_items} foods | Categories: {num_categories} | Time: {loading_time:.1f}s"
            
            if 'model_performance' in stats and stats['model_performance']:
                # Show model performance for overall health
                if 'Overall_Health' in stats['model_performance']:
                    perf = stats['model_performance']['Overall_Health']
                    stats_text += f" | Model R²: {perf['r2_score']:.3f}"
            
            if self.current_life_element:
                stats_text += f" | Element: {self.current_life_element}"
            
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
        self.birth_month_var.set(1)
        self.birth_year_var.set(1990)
        
        self.diabetes_var.set(False)
        self.obesity_var.set(False)
        self.hypertension_var.set(False)
        self.cholesterol_var.set(False)
        
        # Clear results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.targets_text.delete(1.0, tk.END)
        self.targets_text.insert(tk.END, "Click 'Calculate Nutrition + Element' to see your personalized nutritional targets and Thai life element analysis.")
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information, health suitability, and life element compatibility.")
        
        self.update_charts([])
        self.current_nutritional_data = None
        self.current_life_element = None
        self.last_recommendations = []
        
        self.status_var.set("Form reset to defaults")


def main():
    """Main function to run the application"""
    print("Starting Health-Driven Food Recommendation System with Thai Life Elements...")
    
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
             font=('Segoe UI', 16, 'bold')).pack(pady=10)
    ttk.Label(splash_frame, text="Thai Life Elements (ธาตุ) + Medical Guidelines", 
             font=('Segoe UI', 14, 'bold'), foreground='#FFD700').pack()
    ttk.Label(splash_frame, text="Using Random Forest & Traditional Medicine", 
             font=('Segoe UI', 12)).pack(pady=5)
    ttk.Label(splash_frame, text="Prince of Songkla University", 
             font=('Segoe UI', 10)).pack(pady=5)
    
    # Thai elements display
    elements_frame = ttk.Frame(splash_frame)
    elements_frame.pack(pady=20)
    
    element_colors = {'Earth': '#8B4513', 'Water': '#4169E1', 'Wind': '#32CD32', 'Fire': '#FF4500'}
    elements = [('Earth (ธาตุดิน)', 'earth'), ('Water (ธาตุน้ำ)', 'water'), 
                ('Wind (ธาตุลม)', 'wind'), ('Fire (ธาตุไฟ)', 'fire')]
    
    for i, (element_text, element_key) in enumerate(elements):
        row = i // 2
        col = i % 2
        color = element_colors.get(element_key.title(), '#333333')
        ttk.Label(elements_frame, text=element_text, 
                 font=('Segoe UI', 10, 'bold'), foreground=color).grid(row=row, column=col, padx=20, pady=5)
    
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
            
            update_splash_status("Calculating Thai life element scores...")
            time.sleep(0.3)
            
            update_splash_status("Building user interface...")
            
            # Create main UI
            app = HealthDrivenFoodRecommenderUI(root, recommender)
            
            # Show main window
            root.deiconify()
            root.lift()
            root.focus_force()
            
            # Close splash
            splash.destroy()
            
            print("System initialized successfully with Thai life elements!")
            
        except Exception as e:
            splash.destroy()
            messagebox.showerror("Initialization Error", f"Failed to initialize system: {str(e)}")
            root.quit()
    
    # Schedule initialization
    root.after(100, initialize_system)
    
    # Configure main window
    root.title("Health-Driven Food Recommendation System - Thai Life Elements")
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