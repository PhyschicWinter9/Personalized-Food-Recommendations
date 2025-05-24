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
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
from scipy.spatial.distance import euclidean

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


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
        
        # Medical guidelines for different health conditions (based on research)
        self.medical_guidelines = {
            'Diabetes': {
                'sugar_max_percent': 5,          # <5% of total calories (WHO recommendation)
                'sugar_max_grams': 25,           # Maximum 25g/day
                'carb_percent': (40, 50),        # 40-50% of calories (lower than general)
                'protein_percent': (15, 25),     # 15-25% of calories
                'fat_percent': (25, 35),         # 25-35% of calories
                'saturated_fat_percent': 7,      # <7% of calories
                'fiber_min': 25,                 # Minimum 25g/day
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
                'fiber_min': 25,                 # Minimum 25g/day
                'sodium_max': 2300,              # <2300mg/day
                'calorie_deficit': 500           # 500 kcal deficit for 1lb/week loss
            },
            'Hypertension': {
                'sugar_max_percent': 10,         # <10% of total calories
                'carb_percent': (45, 65),        # 45-65% of calories
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories (DASH)
                'fiber_min': 30,                 # Minimum 30g/day (DASH)
                'sodium_max': 1500,              # <1500mg/day (ideal for BP)
                'potassium_min': 4700            # Minimum 4700mg/day
            },
            'High_Cholesterol': {
                'sugar_max_percent': 10,         # <10% of total calories
                'carb_percent': (45, 65),        # 45-65% of calories
                'protein_percent': (15, 20),     # 15-20% of calories
                'fat_percent': (25, 30),         # 25-30% of calories
                'saturated_fat_percent': 6,      # <6% of calories
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
            if 'Obesity' in health_conditions:
                target_calories = tdee - self.medical_guidelines['Obesity']['calorie_deficit']
            else:
                target_calories = tdee - 300  # Moderate deficit
        elif weight_goal == 'Gain Weight':
            target_calories = tdee + 300  # Moderate surplus
        
        return max(1200, target_calories)  # Ensure minimum safe calories
    
    def calculate_nutritional_targets(self, user_profile):
        """Calculate personalized nutritional targets based on medical guidelines"""
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
            targets = self._apply_health_condition_modifications(targets, health_conditions, target_calories, weight_kg)
        
        # Calculate meal-specific targets
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
                    targets['protein_min'] = max(targets['protein_min'], protein_from_weight_min)
                    targets['protein_max'] = max(targets['protein_max'], protein_from_weight_max)
                
                # Fat modifications
                if 'fat_percent' in guidelines:
                    fat_min, fat_max = guidelines['fat_percent']
                    targets['fat_min'] = (target_calories * fat_min / 100) / 9
                    targets['fat_max'] = (target_calories * fat_max / 100) / 9
                
                # Other modifications
                if 'fiber_min' in guidelines:
                    targets['fiber_min'] = max(targets['fiber_min'], guidelines['fiber_min'])
                if 'sodium_max' in guidelines:
                    targets['sodium_max'] = min(targets['sodium_max'], guidelines['sodium_max'])
                if 'cholesterol_max' in guidelines:
                    targets['cholesterol_max'] = min(targets['cholesterol_max'], guidelines['cholesterol_max'])
        
        return targets
    
    def _calculate_meal_targets(self, daily_targets):
        """Calculate targets for individual meals"""
        meal_distribution = {
            'breakfast': 0.25,
            'lunch': 0.30,
            'dinner': 0.30,
            'snack': 0.075
        }
        
        meal_targets = {}
        for meal_type, proportion in meal_distribution.items():
            meal_targets[meal_type] = {}
            for nutrient, value in daily_targets.items():
                meal_targets[meal_type][nutrient] = value * proportion
        
        return meal_targets


class HealthAwareKNNRecommender:
    """KNN-based food recommender with health condition awareness"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        
        # Stats tracking
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0,
            'model_performance': {}
        }
        
        # Nutritional features for KNN
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
        try:
            if self.status_callback:
                self.status_callback(message)
            print(message)
        except Exception as e:
            # If callback fails, just print the message
            print(f"Status: {message}")
            print(f"Status callback error: {e}")
    
    def load_data(self):
        """Load and combine all food datasets from CSV files"""
        try:
            # Look for CSV files in current directory
            dataset_folder = './datasets'  # Adjust path as needed
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                self.update_status("No CSV files found in current directory!")
                self.food_data = pd.DataFrame()
                return
            
            dataframes = []
            
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    # Standardize category names
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
                    
                    base_name = os.path.splitext(filename)[0].lower()
                    category = category_map.get(base_name, 
                        os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title())
                    
                    self.update_status(f"Loading {category} data...")
                    
                    df = pd.read_csv(file_path)
                    
                    # Always set category from filename
                    df['Category'] = category
                    
                    # Clean and standardize data
                    df = self._clean_nutritional_data(df)
                    
                    if len(df) > 0:  # Only add if data exists
                        dataframes.append(df)
                        self.update_status(f"Loaded {len(df)} items from {filename} as {category}")
                    else:
                        self.update_status(f"No valid data in {filename}")
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.update_status("Combining all food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                
                # Debug: print categories found
                unique_categories = self.food_data['Category'].unique()
                self.update_status(f"Categories found: {', '.join(unique_categories)}")
                
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
            
            # Calculate overall health score
            overall_score = (diabetes_score + obesity_score + hypertension_score + cholesterol_score) / 4
            self.food_data.at[idx, 'Overall_Health_Score'] = overall_score
    
    def _calculate_diabetes_score(self, food):
        """Calculate diabetes suitability score (lower is better)"""
        score = 0
        
        # Sugar penalty
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
        
        # Protein and fiber bonus
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
        
        # Sodium penalty
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
        """Prepare features for KNN model"""
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
        
        self.update_status(f"Prepared {len(self.features)} features for KNN model")
    
    def train_models(self):
        """Train KNN models for different recommendation tasks"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for training")
            return
        
        try:
            # Prepare feature matrix
            X = self.food_data[self.features].fillna(0)
            
            # Standardize features for KNN
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train KNN models for different health conditions
            self.knn_models = {}
            
            # General nutritional similarity KNN
            self.update_status("Training general KNN model...")
            general_knn = NearestNeighbors(n_neighbors=20, metric='euclidean', algorithm='auto')
            general_knn.fit(X_scaled)
            self.knn_models['general'] = general_knn
            
            # Health-condition-specific KNN models
            health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
            
            for condition in health_conditions:
                self.update_status(f"Training KNN model for {condition}...")
                
                # Create condition-specific feature weights
                condition_features = self._get_condition_feature_weights(condition)
                weighted_features = X_scaled * condition_features
                
                condition_knn = NearestNeighbors(n_neighbors=20, metric='euclidean', algorithm='auto')
                condition_knn.fit(weighted_features)
                self.knn_models[condition] = condition_knn
            
            # Evaluate model performance
            self._evaluate_knn_performance(X_scaled)
            
            self.update_status("KNN models trained successfully!")
            
        except Exception as e:
            self.update_status(f"Error training KNN models: {e}")
    
    def _get_condition_feature_weights(self, condition):
        """Get feature weights specific to health conditions"""
        base_weights = np.ones(len(self.features))
        
        # Map features to indices
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
        if condition == 'Diabetes':
            # Emphasize sugar and carbohydrate features
            if 'SUGAR(g)' in feature_map:
                base_weights[feature_map['SUGAR(g)']] = 3.0
            if 'CHOCDF (g) Carbohydrate' in feature_map:
                base_weights[feature_map['CHOCDF (g) Carbohydrate']] = 2.0
            if 'FIBTG (g) Dietary fibre' in feature_map:
                base_weights[feature_map['FIBTG (g) Dietary fibre']] = 2.0
        
        elif condition == 'Hypertension':
            # Emphasize sodium and potassium features
            if 'Na(mg)' in feature_map:
                base_weights[feature_map['Na(mg)']] = 3.0
            if 'K(mg)' in feature_map:
                base_weights[feature_map['K(mg)']] = 2.0
        
        elif condition == 'High_Cholesterol':
            # Emphasize fat and cholesterol features
            if 'Fat(g)' in feature_map:
                base_weights[feature_map['Fat(g)']] = 2.0
            if 'CHOLE(mg) Cholesterol' in feature_map:
                base_weights[feature_map['CHOLE(mg) Cholesterol']] = 3.0
            if 'FIBTG (g) Dietary fibre' in feature_map:
                base_weights[feature_map['FIBTG (g) Dietary fibre']] = 2.0
        
        elif condition == 'Obesity':
            # Emphasize calorie density features
            if 'Energy(kcal) by calculation' in feature_map:
                base_weights[feature_map['Energy(kcal) by calculation']] = 3.0
            if 'Fat(g)' in feature_map:
                base_weights[feature_map['Fat(g)']] = 2.0
            if 'Protein(g)' in feature_map:
                base_weights[feature_map['Protein(g)']] = 1.5
        
        return base_weights
    
    def _evaluate_knn_performance(self, X_scaled):
        """Evaluate KNN model performance"""
        try:
            # Simple evaluation metrics
            n_samples = len(X_scaled)
            n_neighbors = min(10, n_samples - 1) if n_samples > 1 else 1
            
            if n_samples > 10:
                # Sample-based evaluation
                sample_indices = np.random.choice(n_samples, min(100, n_samples), replace=False)
                
                for model_name, knn_model in self.knn_models.items():
                    distances_list = []
                    
                    for idx in sample_indices:
                        query_point = X_scaled[idx:idx+1]
                        distances, _ = knn_model.kneighbors(query_point, n_neighbors=n_neighbors)
                        distances_list.extend(distances.flatten())
                    
                    avg_distance = np.mean(distances_list)
                    std_distance = np.std(distances_list)
                    
                    self.stats['model_performance'][model_name] = {
                        'avg_distance': avg_distance,
                        'std_distance': std_distance,
                        'n_neighbors': n_neighbors
                    }
            
        except Exception as e:
            self.update_status(f"Error evaluating KNN performance: {e}")
    
    def get_recommendations(self, user_profile, meal_type='meal', max_recommendations=10):
        """Get personalized food recommendations using KNN"""
        if not hasattr(self, 'knn_models') or len(self.food_data) == 0:
            return []
        
        try:
            # Calculate nutritional targets
            nutritional_data = self.nutrition_calculator.calculate_nutritional_targets(user_profile)
            daily_targets = nutritional_data['daily_targets']
            meal_targets = nutritional_data['meal_targets'].get(meal_type.lower(), 
                                                              nutritional_data['meal_targets']['lunch'])
            
            # Get user's health conditions
            health_conditions = user_profile.get('health_conditions', [])
            
            # Start with full dataset
            candidates = self.food_data.copy()
            self.update_status(f"Starting with {len(candidates)} total food items")
            
            # Apply category filter if specified
            category_filter = user_profile.get('category_filter', 'All')
            self.update_status(f"Category filter selected: {category_filter}")
            
            if category_filter != 'All' and category_filter is not None:
                # Debug: show available categories
                available_categories = candidates['Category'].unique()
                self.update_status(f"Available categories: {list(available_categories)}")
                
                # Filter by category
                before_filter = len(candidates)
                candidates = candidates[candidates['Category'] == category_filter]
                after_filter = len(candidates)
                
                self.update_status(f"After category filter: {after_filter} items (was {before_filter})")
                
                # If no items found with exact match, try partial match
                if len(candidates) == 0:
                    self.update_status(f"No exact match for '{category_filter}', trying partial match...")
                    candidates = self.food_data[self.food_data['Category'].str.contains(category_filter, case=False, na=False)]
                    self.update_status(f"Partial match found: {len(candidates)} items")
                
                # If still no matches, fall back to all data but warn user
                if len(candidates) == 0:
                    self.update_status(f"No items found for category '{category_filter}', using all categories")
                    candidates = self.food_data.copy()
            
            # Final check
            if len(candidates) == 0:
                self.update_status("No candidate foods available after filtering")
                return []
            
            self.update_status(f"Processing {len(candidates)} candidate foods for KNN matching")
            
            # Create user profile vector based on calculated targets
            user_vector = self._create_user_vector(meal_targets)
            
            # Get KNN recommendations based on health conditions
            if health_conditions:
                recommendations = self._get_health_condition_recommendations(
                    user_vector, health_conditions, candidates, max_recommendations
                )
            else:
                recommendations = self._get_general_recommendations(
                    user_vector, candidates, max_recommendations
                )
            
            # Enhance recommendations with detailed information
            for rec in recommendations:
                rec['nutritional_data'] = nutritional_data
                rec['explanation'] = self._generate_explanation(rec, meal_targets, health_conditions)
                rec['health_warnings'] = self._generate_health_warnings(rec, health_conditions)
                rec['targets_met'] = self._check_targets_met(rec, meal_targets)
            
            self.update_status(f"KNN algorithm found {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            self.update_status(f"Error getting recommendations: {e}")
            return []
    
    def _create_user_vector(self, targets):
        """Create user profile vector from calculated nutritional targets"""
        user_vector = np.zeros(len(self.features))
        
        # Map targets to feature vector
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
        if 'Energy(kcal) by calculation' in feature_map:
            user_vector[feature_map['Energy(kcal) by calculation']] = targets.get('calories', 500)
        if 'Protein(g)' in feature_map:
            user_vector[feature_map['Protein(g)']] = (targets.get('protein_min', 0) + targets.get('protein_max', 50)) / 2
        if 'CHOCDF (g) Carbohydrate' in feature_map:
            user_vector[feature_map['CHOCDF (g) Carbohydrate']] = (targets.get('carbs_min', 0) + targets.get('carbs_max', 50)) / 2
        if 'SUGAR(g)' in feature_map:
            user_vector[feature_map['SUGAR(g)']] = targets.get('sugar_max', 10) * 0.5  # Aim for lower sugar
        if 'FIBTG (g) Dietary fibre' in feature_map:
            user_vector[feature_map['FIBTG (g) Dietary fibre']] = targets.get('fiber_min', 5)
        if 'Fat(g)' in feature_map:
            user_vector[feature_map['Fat(g)']] = (targets.get('fat_min', 0) + targets.get('fat_max', 20)) / 2
        if 'Na(mg)' in feature_map:
            user_vector[feature_map['Na(mg)']] = targets.get('sodium_max', 1000) * 0.3  # Aim for lower sodium
        
        # Set health scores to ideal values (0 = best)
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        for health_feature in health_features:
            if health_feature in feature_map:
                user_vector[feature_map[health_feature]] = 0  # Ideal health score
        
        return user_vector.reshape(1, -1)
    
    def _get_health_condition_recommendations(self, user_vector, health_conditions, candidates, max_recommendations):
        """Get recommendations considering health conditions"""
        all_recommendations = []
        
        # Get recommendations from condition-specific models
        for condition in health_conditions:
            if condition in self.knn_models:
                try:
                    # Create feature matrix for candidates only
                    candidate_features = candidates[self.features].fillna(0)
                    
                    if len(candidate_features) == 0:
                        continue
                    
                    # Scale candidate features
                    candidate_features_scaled = self.scaler.transform(candidate_features)
                    
                    # Scale user vector with condition weights
                    condition_weights = self._get_condition_feature_weights(condition)
                    weighted_user_vector = user_vector * condition_weights
                    scaled_user_vector = self.scaler.transform(weighted_user_vector)
                    
                    # Calculate distances manually since we're working with a subset
                    distances = []
                    for i, candidate_features_row in enumerate(candidate_features_scaled):
                        # Apply same weights to candidate
                        weighted_candidate = candidate_features_row * condition_weights
                        dist = euclidean(scaled_user_vector[0], weighted_candidate)
                        distances.append((dist, i))
                    
                    # Sort by distance and get top recommendations
                    distances.sort(key=lambda x: x[0])
                    top_distances = distances[:min(max_recommendations, len(distances))]
                    
                    # Create recommendations
                    for dist, candidate_idx in top_distances:
                        food = candidates.iloc[candidate_idx]
                        rec = self._create_recommendation_object(food, dist, condition)
                        all_recommendations.append(rec)
                        
                except Exception as e:
                    self.update_status(f"Error in {condition} KNN: {e}")
                    continue
        
        # If no condition-specific recommendations, use general model
        if not all_recommendations:
            return self._get_general_recommendations(user_vector, candidates, max_recommendations)
        
        # Remove duplicates and sort by distance
        unique_recommendations = {}
        for rec in all_recommendations:
            food_id = rec['food_id']
            if food_id not in unique_recommendations or rec['distance'] < unique_recommendations[food_id]['distance']:
                unique_recommendations[food_id] = rec
        
        recommendations = list(unique_recommendations.values())
        recommendations.sort(key=lambda x: x['distance'])
        
        return recommendations[:max_recommendations]
    
    def _get_general_recommendations(self, user_vector, candidates, max_recommendations):
        """Get general nutritional recommendations"""
        try:
            # Create feature matrix for candidates only
            candidate_features = candidates[self.features].fillna(0)
            
            if len(candidate_features) == 0:
                return []
            
            # Scale features
            candidate_features_scaled = self.scaler.transform(candidate_features)
            scaled_user_vector = self.scaler.transform(user_vector)
            
            # Calculate distances manually
            distances = []
            for i, candidate_features_row in enumerate(candidate_features_scaled):
                dist = euclidean(scaled_user_vector[0], candidate_features_row)
                distances.append((dist, i))
            
            # Sort by distance and get top recommendations
            distances.sort(key=lambda x: x[0])
            top_distances = distances[:min(max_recommendations, len(distances))]
            
            recommendations = []
            for dist, candidate_idx in top_distances:
                food = candidates.iloc[candidate_idx]
                rec = self._create_recommendation_object(food, dist, 'general')
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            self.update_status(f"Error in general KNN: {e}")
            return []
    
    def _create_recommendation_object(self, food, distance, model_type):
        """Create recommendation object from food data"""
        return {
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
            'distance': distance,
            'model_type': model_type,
            'diabetes_score': float(food.get('Diabetes_Score', 0)),
            'obesity_score': float(food.get('Obesity_Score', 0)),
            'hypertension_score': float(food.get('Hypertension_Score', 0)),
            'cholesterol_score': float(food.get('High_Cholesterol_Score', 0)),
            'suitable_for_conditions': self._check_condition_suitability(food)
        }
    
    def _check_condition_suitability(self, food):
        """Check which health conditions this food is suitable for"""
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
    
    def _check_targets_met(self, recommendation, targets):
        """Check which nutritional targets are met"""
        met = []
        
        # Check calorie range
        calories = recommendation['calories']
        target_calories = targets.get('calories', 500)
        if 0.7 * target_calories <= calories <= 1.3 * target_calories:
            met.append('Calories')
        
        # Check protein
        if recommendation['protein'] >= targets.get('protein_min', 0):
            met.append('Protein')
        
        # Check fiber
        if recommendation['fiber'] >= targets.get('fiber_min', 0):
            met.append('Fiber')
        
        # Check sugar limit
        if recommendation['sugar'] <= targets.get('sugar_max', 10):
            met.append('Sugar')
        
        return met
    
    def _generate_health_warnings(self, recommendation, health_conditions):
        """Generate health warnings for specific conditions"""
        warnings = []
        
        for condition in health_conditions:
            if condition == 'Diabetes' and recommendation['sugar'] > 10:
                warnings.append("High sugar content - monitor blood glucose")
            elif condition == 'Hypertension' and recommendation['sodium'] > 300:
                warnings.append("High sodium content - may increase blood pressure")
            elif condition == 'High_Cholesterol' and recommendation['saturated_fat'] > 4:
                warnings.append("High saturated fat - may raise cholesterol")
        
        return warnings
    
    def _generate_explanation(self, recommendation, targets, health_conditions):
        """Generate explanation for why this food was recommended"""
        explanations = []
        
        # Distance-based explanation
        distance = recommendation['distance']
        if distance < 0.5:
            explanations.append("Excellent nutritional match")
        elif distance < 1.0:
            explanations.append("Good nutritional match")
        else:
            explanations.append("Fair nutritional match")
        
        # Health condition suitability
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            conditions_text = ", ".join(suitable_conditions)
            explanations.append(f"Suitable for {conditions_text}")
        
        # Specific nutritional benefits
        if recommendation['fiber'] >= 5:
            explanations.append("High fiber content")
        if recommendation['protein'] >= 10:
            explanations.append("Good protein source")
        if recommendation['sugar'] <= 5:
            explanations.append("Low sugar content")
        
        return " | ".join(explanations)
    
    def get_stats(self):
        """Get statistics about the recommendation system"""
        return self.stats


class HealthDrivenKNNFoodRecommenderUI:
    """Modern GUI for the health-driven KNN food recommendation system"""
    
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Health-Driven KNN Food Recommendation System")
        self.master.geometry("1400x900")
        self.master.configure(bg="#f8f9fa")
        
        # Modern color scheme
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
        
        # Configure styles
        self.setup_styles()
        
        # Initialize recommender
        self.recommender = recommender or HealthAwareKNNRecommender()
        
        # Create UI
        self.create_main_interface()
        
        # Initialize variables
        self.last_recommendations = []
        self.current_nutritional_data = None
        
        # Update category list after recommender is initialized
        self.update_category_list()
        
        # Update stats
        self.update_stats_display()
    
    def setup_styles(self):
        """Setup modern UI styles"""
        self.style = ttk.Style()
        
        try:
            self.style.theme_use('clam')
        except:
            pass
        
        # Fonts
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
        self.style.configure('Health.Treeview', font=self.fonts['body'], rowheight=25)
        self.style.configure('Health.Treeview.Heading', font=self.fonts['subheading'])
    
    def create_main_interface(self):
        """Create the main interface"""
        # Main container
        main_container = ttk.Frame(self.master, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Three-panel layout
        self.create_input_panel(content_frame)
        self.create_results_panel(content_frame)
        self.create_charts_panel(main_container)
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self, parent):
        """Create header with title and controls"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(title_frame, text="Health-Driven KNN Food Recommendation System", 
                 style='Title.TLabel').pack(anchor=tk.W)
        ttk.Label(title_frame, text="Automatic Nutritional Calculation + K-Nearest Neighbors Matching", 
                 style='Subtitle.TLabel', foreground=self.colors['text_light']).pack(anchor=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Get Recommendations", 
                  command=self.get_recommendations).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_form).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Calculate Nutrition", 
                  command=self.show_nutrition_calculation).pack(side=tk.RIGHT, padx=5)
    
    def create_input_panel(self, parent):
        """Create input panel with health profile form"""
        left_panel = ttk.LabelFrame(parent, text="Health Profile", padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), ipadx=10)
        
        # Personal Information
        personal_frame = ttk.LabelFrame(left_panel, text="Personal Information", padding="10")
        personal_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Form fields
        self.create_form_field(personal_frame, "Weight (kg):", "weight_var", 70.0, 0, 0)
        self.create_form_field(personal_frame, "Height (cm):", "height_var", 170.0, 1, 0)
        self.create_form_field(personal_frame, "Age (years):", "age_var", 30, 2, 0)
        
        # Gender
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
        
        # Bind BMI calculation
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()
    
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
            height = self.height_var.get() / 100
            
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
    
    def update_category_list(self):
        """Update category dropdown"""
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
            if 'Category' in self.recommender.food_data.columns:
                unique_categories = sorted(self.recommender.food_data['Category'].dropna().unique())
                categories.extend(unique_categories)
                print(f"Categories available: {categories}")  # Debug info
        self.category_combo['values'] = categories
        
        # Reset to 'All' if current selection is not available
        current_selection = self.category_var.get()
        if current_selection not in categories:
            self.category_var.set('All')
    
    def create_results_panel(self, parent):
        """Create results panel with recommendations table"""
        right_panel = ttk.LabelFrame(parent, text="Food Recommendations", padding="15")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Recommendations tab
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="KNN Recommendations")
        
        # Treeview for recommendations
        columns = ('Name', 'Category', 'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Distance', 'Suitable For')
        self.tree = ttk.Treeview(rec_frame, columns=columns, show='headings', 
                               style='Health.Treeview', height=15)
        
        # Configure columns
        column_widths = {'Name': 150, 'Category': 100, 'Calories': 70, 'Protein': 60, 'Carbs': 60, 
                        'Sugar': 50, 'Fiber': 50, 'Distance': 60, 'Suitable For': 120}
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(rec_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(rec_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Nutritional targets tab
        targets_frame = ttk.Frame(self.notebook)
        self.notebook.add(targets_frame, text="Calculated Targets")
        
        self.targets_text = tk.Text(targets_frame, wrap=tk.WORD, font=self.fonts['body'])
        targets_scrollbar = ttk.Scrollbar(targets_frame, orient="vertical", command=self.targets_text.yview)
        self.targets_text.configure(yscrollcommand=targets_scrollbar.set)
        
        self.targets_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        targets_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Details tab
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="Food Details")
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD, font=self.fonts['body'])
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
        # Initial text
        self.targets_text.insert(tk.END, "Click 'Calculate Nutrition' to see your personalized nutritional targets based on medical guidelines.")
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information and KNN similarity analysis.")
    
    def create_charts_panel(self, parent):
        """Create charts panel for visualizations"""
        charts_frame = ttk.LabelFrame(parent, text="Nutritional Analysis & KNN Insights", padding="10")
        charts_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(14, 4), dpi=100)
        self.fig.patch.set_facecolor(self.colors['white'])
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(141)
        self.ax2 = self.fig.add_subplot(142)
        self.ax3 = self.fig.add_subplot(143)
        self.ax4 = self.fig.add_subplot(144)
        
        # Initialize charts
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_charts([])
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.master, relief=tk.SUNKEN, padding="5")
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready - Enter your health profile and click 'Calculate Nutrition'")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)
        
        # Stats display
        self.stats_frame = ttk.Frame(self.status_frame)
        self.stats_frame.pack(side=tk.RIGHT)
        
        # Initialize stats label
        self.stats_label = None
    
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
        """Show calculated nutritional targets"""
        try:
            user_profile = self.get_user_profile()
            
            # Calculate nutritional targets
            nutritional_data = self.recommender.nutrition_calculator.calculate_nutritional_targets(user_profile)
            self.current_nutritional_data = nutritional_data
            
            # Check if targets_text widget still exists
            if not hasattr(self, 'targets_text') or not self.targets_text.winfo_exists():
                messagebox.showerror("Error", "Text widget not available")
                return
            
            # Display in targets tab
            try:
                self.targets_text.delete(1.0, tk.END)
                
                # Add header
                self.targets_text.insert(tk.END, "PERSONALIZED NUTRITIONAL TARGETS (KNN-Based Matching)\n", "header")
                self.targets_text.insert(tk.END, "="*60 + "\n\n", "separator")
                
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
                    self.targets_text.insert(tk.END, f"{self.meal_type_var.get()} Targets (KNN Input Vector):\n", "subheader")
                    self.targets_text.insert(tk.END, f"• Calories: {meal['calories']:.0f} kcal\n")
                    self.targets_text.insert(tk.END, f"• Protein: {meal['protein_min']:.0f}-{meal['protein_max']:.0f} g\n")
                    self.targets_text.insert(tk.END, f"• Carbohydrates: {meal['carbs_min']:.0f}-{meal['carbs_max']:.0f} g\n")
                    self.targets_text.insert(tk.END, f"• Sugar (max): {meal['sugar_max']:.0f} g\n")
                    self.targets_text.insert(tk.END, f"• Fiber (min): {meal['fiber_min']:.0f} g\n")
                    self.targets_text.insert(tk.END, f"• Fat: {meal['fat_min']:.0f}-{meal['fat_max']:.0f} g\n\n")
                
                # Add health condition info
                health_conditions = user_profile['health_conditions']
                if health_conditions:
                    self.targets_text.insert(tk.END, "Health Condition Adjustments (KNN Model Selection):\n", "subheader")
                    for condition in health_conditions:
                        self.targets_text.insert(tk.END, f"• {condition}: Using specialized KNN model with weighted features\n")
                    self.targets_text.insert(tk.END, "\n")
                
                self.targets_text.insert(tk.END, "KNN Algorithm Process:\n", "subheader")
                self.targets_text.insert(tk.END, "1. Convert your targets into a feature vector\n")
                self.targets_text.insert(tk.END, "2. Apply health condition feature weights\n")
                self.targets_text.insert(tk.END, "3. Find nearest neighbors in nutritional space\n")
                self.targets_text.insert(tk.END, "4. Rank by Euclidean distance (lower = better match)\n")
                
                # Configure text tags
                self.targets_text.tag_configure("header", font=self.fonts['heading'], foreground=self.colors['primary'])
                self.targets_text.tag_configure("subheader", font=self.fonts['subheading'], foreground=self.colors['secondary'])
                self.targets_text.tag_configure("separator", foreground=self.colors['text_light'])
                
                # Switch to targets tab
                if hasattr(self, 'notebook') and self.notebook.winfo_exists():
                    self.notebook.select(1)
                
                self.status_var.set("Nutritional targets calculated - ready for KNN food matching")
                
            except tk.TclError as e:
                print(f"Text widget error: {e}")
                messagebox.showerror("Error", "Error updating text display")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating nutrition targets: {str(e)}")
            print(f"Nutrition calculation error: {e}")
    
    def get_recommendations(self):
        """Get KNN-based food recommendations"""
        try:
            if not self.current_nutritional_data:
                messagebox.showwarning("Warning", "Please calculate nutrition targets first!")
                return
            
            # Update category list in case data was loaded after UI creation
            self.update_category_list()
            
            user_profile = self.get_user_profile()
            meal_type = self.meal_type_var.get().lower()
            max_results = int(self.max_results_var.get())
            
            self.status_var.set("Running KNN algorithm to find optimal food matches...")
            
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
                        rec['name'][:20],
                        rec['category'],
                        f"{rec['calories']:.0f}",
                        f"{rec['protein']:.1f}",
                        f"{rec['carbs']:.1f}",
                        f"{rec['sugar']:.1f}",
                        f"{rec['fiber']:.1f}",
                        f"{rec['distance']:.3f}",
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
                self.status_var.set(f"KNN found {len(recommendations)} optimal foods for {condition_text}")
                
            else:
                # Provide more helpful error message
                category_filter = user_profile.get('category_filter', 'All')
                if category_filter != 'All':
                    error_msg = f"No suitable foods found in '{category_filter}' category. Try:\n"
                    error_msg += "1. Select 'All' categories\n"
                    error_msg += "2. Choose a different category\n"
                    error_msg += "3. Adjust your health profile settings"
                    
                    # Show available categories
                    if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
                        available_cats = list(self.recommender.food_data['Category'].unique())
                        error_msg += f"\n\nAvailable categories: {', '.join(available_cats)}"
                else:
                    error_msg = "No suitable foods found matching your criteria.\n"
                    error_msg += "Try adjusting your health profile or meal type settings."
                
                messagebox.showinfo("No Results", error_msg)
                self.status_var.set("No suitable foods found - try different filter settings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {str(e)}")
            self.status_var.set("Error occurred during KNN recommendation")
    
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
            
            # Add KNN similarity information
            self.details_text.insert(tk.END, "KNN Analysis:\n", "header")
            self.details_text.insert(tk.END, f"• Euclidean Distance: {rec['distance']:.4f} (lower = better match)\n")
            self.details_text.insert(tk.END, f"• Model Used: {rec['model_type']}\n")
            if rec['distance'] < 0.5:
                similarity = "Excellent"
                color = "good"
            elif rec['distance'] < 1.0:
                similarity = "Good"
                color = "good"
            elif rec['distance'] < 2.0:
                similarity = "Fair"
                color = "warning"
            else:
                similarity = "Poor"
                color = "warning"
            self.details_text.insert(tk.END, f"• Similarity Level: {similarity}\n\n", color)
            
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
            
            # Add health suitability scores
            self.details_text.insert(tk.END, "Health Condition Scores (lower = better):\n", "header")
            self.details_text.insert(tk.END, f"• Diabetes Score: {rec['diabetes_score']:.1f}\n")
            self.details_text.insert(tk.END, f"• Obesity Score: {rec['obesity_score']:.1f}\n")
            self.details_text.insert(tk.END, f"• Hypertension Score: {rec['hypertension_score']:.1f}\n")
            self.details_text.insert(tk.END, f"• High Cholesterol Score: {rec['cholesterol_score']:.1f}\n\n")
            
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
            
            # Add recommendation explanation
            self.details_text.insert(tk.END, "Why This Food Was Recommended (KNN):\n", "header")
            self.details_text.insert(tk.END, f"{rec['explanation']}\n")
            
            # Configure text tags
            self.details_text.tag_configure("title", font=self.fonts['heading'], foreground=self.colors['primary'])
            self.details_text.tag_configure("subtitle", font=self.fonts['subheading'], foreground=self.colors['text'])
            self.details_text.tag_configure("header", font=self.fonts['subheading'], foreground=self.colors['secondary'])
            self.details_text.tag_configure("good", foreground=self.colors['success'])
            self.details_text.tag_configure("warning", foreground=self.colors['warning'])
    
    def update_charts(self, recommendations):
        """Update visualization charts with KNN insights"""
        # Clear previous charts
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        if not recommendations:
            # Show placeholder text
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
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
            
            # Chart 2: KNN Distance Distribution
            distances = [r['distance'] for r in recommendations]
            self.ax2.hist(distances, bins=5, color=self.colors['secondary'], alpha=0.7, edgecolor='black')
            self.ax2.set_title('KNN Distance Distribution', fontsize=10, fontweight='bold')
            self.ax2.set_xlabel('Euclidean Distance')
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
            
            self.ax4.tick_params(axis='x', rotation=45)
            
        except Exception as e:
            print(f"Error updating charts: {e}")
        
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_stats_display(self):
        """Update statistics display"""
        try:
            if not hasattr(self, 'stats_frame') or not self.stats_frame.winfo_exists():
                return
                
            stats = self.recommender.get_stats()
            
            # Clear previous stats safely
            if hasattr(self, 'stats_label') and self.stats_label:
                try:
                    self.stats_label.destroy()
                except tk.TclError:
                    pass
            
            # Clear all children of stats_frame
            for widget in self.stats_frame.winfo_children():
                try:
                    widget.destroy()
                except tk.TclError:
                    pass
            
            # Create new stats display
            total_items = stats.get('total_items', 0)
            num_categories = len(stats.get('categories', {}))
            loading_time = stats.get('loading_time', 0)
            
            stats_text = f"Loaded: {total_items} foods | Categories: {num_categories} | Time: {loading_time:.1f}s"
            
            if 'model_performance' in stats and stats['model_performance']:
                # Show KNN performance
                if 'general' in stats['model_performance']:
                    perf = stats['model_performance']['general']
                    avg_dist = perf.get('avg_distance', 0)
                    stats_text += f" | KNN Avg Distance: {avg_dist:.3f}"
            
            # Create new label
            self.stats_label = ttk.Label(self.stats_frame, text=stats_text, 
                                       font=self.fonts['caption'],
                                       foreground=self.colors['text_light'])
            self.stats_label.pack()
            
        except Exception as e:
            print(f"Error updating stats: {e}")
            # Don't show error to user, just log it
    
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
            
            self.diabetes_var.set(False)
            self.obesity_var.set(False)
            self.hypertension_var.set(False)
            self.cholesterol_var.set(False)
            
            # Clear results safely
            try:
                if hasattr(self, 'tree') and self.tree.winfo_exists():
                    for item in self.tree.get_children():
                        self.tree.delete(item)
            except tk.TclError:
                pass
            
            # Clear text widgets safely
            try:
                if hasattr(self, 'targets_text') and self.targets_text.winfo_exists():
                    self.targets_text.delete(1.0, tk.END)
                    self.targets_text.insert(tk.END, "Click 'Calculate Nutrition' to see your personalized nutritional targets based on medical guidelines.")
            except tk.TclError:
                pass
            
            try:
                if hasattr(self, 'details_text') and self.details_text.winfo_exists():
                    self.details_text.delete(1.0, tk.END)
                    self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information and KNN similarity analysis.")
            except tk.TclError:
                pass
            
            # Update charts safely
            try:
                self.update_charts([])
            except Exception:
                pass
            
            # Reset data
            self.current_nutritional_data = None
            self.last_recommendations = []
            
            self.status_var.set("Form reset to defaults")
            
        except Exception as e:
            print(f"Error during reset: {e}")
            self.status_var.set("Reset completed with minor issues")


def main():
    """Main function to run the application"""
    print("Starting Health-Driven KNN Food Recommendation System...")
    
    # Create main window
    root = tk.Tk()
    root.title("Health-Driven KNN Food Recommendation System")
    root.geometry("1400x900")
    
    # Center main window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (700)  # Half of 1400
    y = (screen_height // 2) - (450)  # Half of 900
    root.geometry(f"1400x900+{x}+{y}")
    
    # Hide main window initially
    root.withdraw()
    
    # Create splash screen as a separate Toplevel
    splash = None
    progress = None
    status_label = None
    
    def create_splash():
        nonlocal splash, progress, status_label
        
        splash = tk.Toplevel(root)
        splash.title("Loading KNN System...")
        splash.geometry("500x300")
        splash.resizable(False, False)
        splash.grab_set()  # Make splash modal
        
        # Center splash screen
        splash_x = (screen_width // 2) - 250
        splash_y = (screen_height // 2) - 150
        splash.geometry(f"500x300+{splash_x}+{splash_y}")
        
        # Splash content
        splash_frame = ttk.Frame(splash, padding="50")
        splash_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(splash_frame, text="Health-Driven KNN Food Recommendation System", 
                 font=('Segoe UI', 16, 'bold')).pack(pady=20)
        ttk.Label(splash_frame, text="Using K-Nearest Neighbors & Medical Guidelines", 
                 font=('Segoe UI', 12)).pack()
        ttk.Label(splash_frame, text="Prince of Songkla University", 
                 font=('Segoe UI', 10)).pack(pady=10)
        
        # Progress bar
        progress = ttk.Progressbar(splash_frame, mode='indeterminate')
        progress.pack(fill=tk.X, pady=20)
        progress.start()
        
        status_label = ttk.Label(splash_frame, text="Initializing KNN models...")
        status_label.pack()
        
        splash.update()
    
    def update_splash_status(message):
        if splash and splash.winfo_exists() and status_label:
            try:
                status_label.config(text=message)
                splash.update()
            except tk.TclError:
                pass  # Ignore if splash is being destroyed
    
    def close_splash_safely():
        nonlocal splash, progress, status_label
        try:
            if progress:
                progress.stop()
            if splash and splash.winfo_exists():
                splash.grab_release()
                splash.destroy()
        except tk.TclError:
            pass  # Splash already destroyed
        finally:
            splash = None
            progress = None
            status_label = None
    
    def initialize_system():
        try:
            # Initialize recommender system
            update_splash_status("Loading Thai food database...")
            recommender = HealthAwareKNNRecommender(update_splash_status)
            
            update_splash_status("Training KNN models...")
            root.update()  # Process any pending events
            time.sleep(0.5)
            
            update_splash_status("Building user interface...")
            root.update()
            
            # Create main UI
            app = HealthDrivenKNNFoodRecommenderUI(root, recommender)
            
            # Close splash safely
            close_splash_safely()
            
            # Show main window
            root.deiconify()
            root.lift()
            root.focus_force()
            
            print("KNN System initialized successfully!")
            
        except Exception as e:
            close_splash_safely()
            root.deiconify()  # Show main window even if there's an error
            messagebox.showerror("Initialization Error", f"Failed to initialize KNN system: {str(e)}")
            print(f"Error: {e}")
            # Don't quit, let user see the error
    
    def start_initialization():
        create_splash()
        # Use after to avoid blocking
        root.after(100, initialize_system)
    
    # Start the initialization process
    root.after(50, start_initialization)
    
    # Start application
    try:
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        # Cleanup
        close_splash_safely()


if __name__ == "__main__":
    main()