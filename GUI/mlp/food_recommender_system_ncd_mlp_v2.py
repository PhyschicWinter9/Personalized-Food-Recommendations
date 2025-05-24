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
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from scipy.spatial.distance import euclidean

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

# Set DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


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
                
                if 'protein_percent' in guidelines:
                    protein_min, protein_max = guidelines['protein_percent']
                    targets['protein_min'] = (target_calories * protein_min / 100) / 4
                    targets['protein_max'] = (target_calories * protein_max / 100) / 4
                
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


class HealthAwareMLPRecommender:
    """MLP-based food recommender with health awareness"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.nutrition_calculator = MedicalNutritionCalculator()
        
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
        
        self.stats = {'total_items': 0, 'categories': {}, 'loading_time': 0}
        
        start_time = time.time()
        self.load_data()
        self.prepare_features()
        self.train_models()
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
            # Try datasets folder first, then current directory
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
    
    def prepare_features(self):
        """Prepare features for MLP"""
        if len(self.food_data) == 0:
            return
        
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features = available_features + health_features
    
    def train_models(self):
        """Train MLP models"""
        if len(self.food_data) == 0 or not hasattr(self, 'features'):
            return
        
        try:
            X = self.food_data[self.features].fillna(0)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.mlp_models = {}
            
            # General model
            self.update_status("Training general MLP...")
            y_general = (self.food_data['Diabetes_Score'] + self.food_data['Obesity_Score'] + 
                        self.food_data['Hypertension_Score'] + self.food_data['High_Cholesterol_Score']) / 4
            
            general_mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            general_mlp.fit(X_scaled, y_general)
            self.mlp_models['general'] = general_mlp
            
            # Condition-specific models
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                self.update_status(f"Training {condition} MLP...")
                y_condition = self.food_data[f'{condition}_Score']
                
                condition_mlp = MLPRegressor(
                    hidden_layer_sizes=(80, 40),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
                condition_mlp.fit(X_scaled, y_condition)
                self.mlp_models[condition] = condition_mlp
            
            self.update_status("MLP training completed!")
            
        except Exception as e:
            self.update_status(f"Error training MLPs: {e}")
    
    def get_recommendations(self, user_profile, meal_type='lunch', max_recommendations=10):
        """Get food recommendations using MLP"""
        if not hasattr(self, 'mlp_models') or len(self.food_data) == 0:
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
            
            # Create user vector
            user_vector = self._create_user_vector(meal_targets)
            
            # Get recommendations
            if health_conditions:
                recommendations = self._get_health_recommendations(user_vector, health_conditions, candidates, max_recommendations)
            else:
                recommendations = self._get_general_recommendations(user_vector, candidates, max_recommendations)
            
            # Add metadata
            for rec in recommendations:
                rec['nutritional_data'] = nutritional_data
                rec['explanation'] = self._generate_explanation(rec, health_conditions)
            
            return recommendations
            
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
    
    def _get_health_recommendations(self, user_vector, health_conditions, candidates, max_recommendations):
        """Get recommendations for health conditions"""
        all_recommendations = []
        
        for condition in health_conditions:
            if condition in self.mlp_models:
                try:
                    candidate_features = candidates[self.features].fillna(0)
                    candidate_features_scaled = self.scaler.transform(candidate_features)
                    scaled_user_vector = self.scaler.transform(user_vector)
                    
                    predicted_scores = self.mlp_models[condition].predict(candidate_features_scaled)
                    
                    similarity_scores = []
                    for i, candidate_row in enumerate(candidate_features_scaled):
                        nutritional_distance = euclidean(scaled_user_vector[0], candidate_row)
                        health_score = predicted_scores[i]
                        combined_score = 0.6 * nutritional_distance + 0.4 * health_score
                        similarity_scores.append((combined_score, i))
                    
                    similarity_scores.sort(key=lambda x: x[0])
                    top_scores = similarity_scores[:max_recommendations]
                    
                    for score, idx in top_scores:
                        food = candidates.iloc[idx]
                        rec = self._create_recommendation_object(food, score, condition, predicted_scores[idx])
                        all_recommendations.append(rec)
                        
                except Exception as e:
                    continue
        
        if not all_recommendations:
            return self._get_general_recommendations(user_vector, candidates, max_recommendations)
        
        # Remove duplicates
        unique_recommendations = {}
        for rec in all_recommendations:
            food_id = rec['food_id']
            if food_id not in unique_recommendations or rec['mlp_score'] < unique_recommendations[food_id]['mlp_score']:
                unique_recommendations[food_id] = rec
        
        recommendations = list(unique_recommendations.values())
        recommendations.sort(key=lambda x: x['mlp_score'])
        
        return recommendations[:max_recommendations]
    
    def _get_general_recommendations(self, user_vector, candidates, max_recommendations):
        """Get general recommendations"""
        try:
            candidate_features = candidates[self.features].fillna(0)
            candidate_features_scaled = self.scaler.transform(candidate_features)
            scaled_user_vector = self.scaler.transform(user_vector)
            
            predicted_scores = self.mlp_models['general'].predict(candidate_features_scaled)
            
            similarity_scores = []
            for i, candidate_row in enumerate(candidate_features_scaled):
                nutritional_distance = euclidean(scaled_user_vector[0], candidate_row)
                health_score = predicted_scores[i]
                combined_score = 0.6 * nutritional_distance + 0.4 * health_score
                similarity_scores.append((combined_score, i))
            
            similarity_scores.sort(key=lambda x: x[0])
            top_scores = similarity_scores[:max_recommendations]
            
            recommendations = []
            for score, idx in top_scores:
                food = candidates.iloc[idx]
                rec = self._create_recommendation_object(food, score, 'general', predicted_scores[idx])
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            return []
    
    def _create_recommendation_object(self, food, mlp_score, model_type, predicted_health_score):
        """Create recommendation object"""
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
            'mlp_score': mlp_score,
            'predicted_health_score': predicted_health_score,
            'model_type': model_type,
            'suitable_for_conditions': self._check_suitability(food)
        }
    
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
    
    def _generate_explanation(self, recommendation, health_conditions):
        """Generate explanation"""
        explanations = []
        
        mlp_score = recommendation['mlp_score']
        if mlp_score < 1.0:
            explanations.append("Excellent MLP match")
        elif mlp_score < 2.0:
            explanations.append("Good MLP match")
        else:
            explanations.append("Fair MLP match")
        
        suitable_conditions = recommendation['suitable_for_conditions']
        if suitable_conditions:
            explanations.append(f"Suitable for {', '.join(suitable_conditions)}")
        
        if recommendation['fiber'] >= 5:
            explanations.append("High fiber")
        if recommendation['protein'] >= 10:
            explanations.append("Good protein")
        if recommendation['sugar'] <= 5:
            explanations.append("Low sugar")
        
        return " | ".join(explanations)
    
    def get_stats(self):
        return self.stats


class ModernFoodRecommenderApp:
    """Modern, intuitive GUI for MLP food recommendation system"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Food Recommendations - MLP Neural Network")
        self.master.geometry("1200x800")
        self.master.configure(bg="#f5f6fa")
        
        # Modern color palette
        self.colors = {
            'primary': '#2f3542',
            'secondary': '#3742fa',
            'accent': '#2ed573',
            'warning': '#ffa502',
            'danger': '#ff4757',
            'light': '#f1f2f6',
            'white': '#ffffff',
            'gray': '#a4b0be',
            'dark_gray': '#57606f'
        }
        
        self.setup_styles()
        self.recommender = None
        self.current_recommendations = []
        
        self.create_interface()
        self.initialize_system()
    
    def setup_styles(self):
        """Setup modern styling"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure modern styles
        self.style.configure('Modern.TFrame', background=self.colors['white'], relief='flat')
        self.style.configure('Card.TFrame', background=self.colors['white'], relief='solid', borderwidth=1)
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=self.colors['primary'], background=self.colors['white'])
        self.style.configure('Heading.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.colors['primary'], background=self.colors['white'])
        self.style.configure('Body.TLabel', font=('Segoe UI', 10), foreground=self.colors['dark_gray'], background=self.colors['white'])
        self.style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'))
        self.style.configure('Modern.Treeview', font=('Segoe UI', 10), rowheight=30)
        self.style.configure('Modern.Treeview.Heading', font=('Segoe UI', 11, 'bold'))
    
    def create_interface(self):
        """Create modern interface"""
        # Main container
        main_container = tk.Frame(self.master, bg=self.colors['light'], padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(main_container)
        
        # Content area
        content_area = tk.Frame(main_container, bg=self.colors['light'])
        content_area.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Create left panel (inputs) and right panel (results)
        self.create_input_section(content_area)
        self.create_results_section(content_area)
        
        # Status bar
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """Create modern header"""
        header_frame = tk.Frame(parent, bg=self.colors['white'], height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title section
        title_section = tk.Frame(header_frame, bg=self.colors['white'])
        title_section.pack(side=tk.LEFT, fill=tk.Y, padx=30, pady=20)
        
        title_label = tk.Label(title_section, text="🧠 Smart Food Recommendations", 
                              font=('Segoe UI', 20, 'bold'), 
                              fg=self.colors['primary'], bg=self.colors['white'])
        title_label.pack(anchor=tk.W)
        
        subtitle_label = tk.Label(title_section, text="AI-Powered Nutrition with MLP Neural Networks", 
                                 font=('Segoe UI', 11), 
                                 fg=self.colors['gray'], bg=self.colors['white'])
        subtitle_label.pack(anchor=tk.W)
        
        # Action buttons
        button_section = tk.Frame(header_frame, bg=self.colors['white'])
        button_section.pack(side=tk.RIGHT, fill=tk.Y, padx=30, pady=15)
        
        self.recommend_btn = tk.Button(button_section, text="🎯 Get Recommendations", 
                                      font=('Segoe UI', 11, 'bold'),
                                      bg=self.colors['secondary'], fg='white',
                                      relief='flat', padx=20, pady=10,
                                      command=self.get_recommendations)
        self.recommend_btn.pack(side=tk.RIGHT, padx=5)
        
        self.analyze_btn = tk.Button(button_section, text="📊 Analyze Profile", 
                                    font=('Segoe UI', 11, 'bold'),
                                    bg=self.colors['accent'], fg='white',
                                    relief='flat', padx=20, pady=10,
                                    command=self.analyze_profile)
        self.analyze_btn.pack(side=tk.RIGHT, padx=5)
        
        reset_btn = tk.Button(button_section, text="🔄 Reset", 
                             font=('Segoe UI', 11),
                             bg=self.colors['gray'], fg='white',
                             relief='flat', padx=15, pady=10,
                             command=self.reset_form)
        reset_btn.pack(side=tk.RIGHT, padx=5)
    
    def create_input_section(self, parent):
        """Create modern input section"""
        input_frame = tk.Frame(parent, bg=self.colors['white'], width=400)
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        input_frame.pack_propagate(False)
        
        # Scrollable frame for inputs
        canvas = tk.Canvas(input_frame, bg=self.colors['white'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['white'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        scrollbar.pack(side="right", fill="y")
        
        # Personal Info Section
        self.create_personal_info_section(scrollable_frame)
        
        # Health Conditions Section
        self.create_health_conditions_section(scrollable_frame)
        
        # Preferences Section
        self.create_preferences_section(scrollable_frame)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_personal_info_section(self, parent):
        """Create personal information section"""
        section_frame = tk.Frame(parent, bg=self.colors['white'])
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Section header
        header_label = tk.Label(section_frame, text="👤 Personal Information", 
                               font=('Segoe UI', 14, 'bold'),
                               fg=self.colors['primary'], bg=self.colors['white'])
        header_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Create input fields with modern styling
        self.weight_var = tk.DoubleVar(value=70.0)
        self.height_var = tk.DoubleVar(value=170.0)
        self.age_var = tk.IntVar(value=30)
        self.gender_var = tk.StringVar(value="Male")
        self.activity_var = tk.StringVar(value="Moderate")
        self.weight_goal_var = tk.StringVar(value="Maintain Weight")
        
        self.create_modern_input(section_frame, "Weight (kg)", self.weight_var)
        self.create_modern_input(section_frame, "Height (cm)", self.height_var)
        self.create_modern_input(section_frame, "Age (years)", self.age_var)
        
        self.create_modern_dropdown(section_frame, "Gender", self.gender_var, 
                                   ['Male', 'Female'])
        self.create_modern_dropdown(section_frame, "Activity Level", self.activity_var,
                                   ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'])
        self.create_modern_dropdown(section_frame, "Weight Goal", self.weight_goal_var,
                                   ['Lose Weight', 'Maintain Weight', 'Gain Weight'])
        
        # BMI display
        self.bmi_var = tk.StringVar(value="BMI will be calculated")
        bmi_frame = tk.Frame(section_frame, bg=self.colors['light'], padx=15, pady=10)
        bmi_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(bmi_frame, text="📏 BMI Status:", font=('Segoe UI', 10, 'bold'),
                fg=self.colors['primary'], bg=self.colors['light']).pack(anchor=tk.W)
        self.bmi_label = tk.Label(bmi_frame, textvariable=self.bmi_var, 
                                 font=('Segoe UI', 12, 'bold'),
                                 fg=self.colors['secondary'], bg=self.colors['light'])
        self.bmi_label.pack(anchor=tk.W)
        
        # Bind BMI calculation
        for var in [self.weight_var, self.height_var]:
            var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()
    
    def create_health_conditions_section(self, parent):
        """Create health conditions section"""
        section_frame = tk.Frame(parent, bg=self.colors['white'])
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = tk.Label(section_frame, text="🏥 Health Conditions", 
                               font=('Segoe UI', 14, 'bold'),
                               fg=self.colors['primary'], bg=self.colors['white'])
        header_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        conditions = [
            ("🩺 Diabetes", self.diabetes_var),
            ("⚖️ Obesity", self.obesity_var),
            ("💔 Hypertension", self.hypertension_var),
            ("🧪 High Cholesterol", self.cholesterol_var)
        ]
        
        for text, var in conditions:
            cb_frame = tk.Frame(section_frame, bg=self.colors['white'])
            cb_frame.pack(fill=tk.X, pady=5)
            
            cb = tk.Checkbutton(cb_frame, text=text, variable=var,
                               font=('Segoe UI', 11), fg=self.colors['dark_gray'],
                               bg=self.colors['white'], activebackground=self.colors['white'],
                               selectcolor=self.colors['light'])
            cb.pack(anchor=tk.W)
    
    def create_preferences_section(self, parent):
        """Create preferences section"""
        section_frame = tk.Frame(parent, bg=self.colors['white'])
        section_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = tk.Label(section_frame, text="⚙️ Preferences", 
                               font=('Segoe UI', 14, 'bold'),
                               fg=self.colors['primary'], bg=self.colors['white'])
        header_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.meal_type_var = tk.StringVar(value="Lunch")
        self.category_var = tk.StringVar(value="All")
        self.max_results_var = tk.StringVar(value="10")
        
        self.create_modern_dropdown(section_frame, "Meal Type", self.meal_type_var,
                                   ['Breakfast', 'Lunch', 'Dinner', 'Snack'])
        
        # Category dropdown will be updated after data loads
        self.category_dropdown = self.create_modern_dropdown(section_frame, "Food Category", self.category_var, ['All'])
        
        self.create_modern_dropdown(section_frame, "Number of Results", self.max_results_var,
                                   ['5', '10', '15', '20', '25'])
    
    def create_modern_input(self, parent, label_text, variable):
        """Create modern styled input field"""
        field_frame = tk.Frame(parent, bg=self.colors['white'])
        field_frame.pack(fill=tk.X, pady=8)
        
        label = tk.Label(field_frame, text=label_text, font=('Segoe UI', 10, 'bold'),
                        fg=self.colors['primary'], bg=self.colors['white'])
        label.pack(anchor=tk.W)
        
        entry = tk.Entry(field_frame, textvariable=variable, font=('Segoe UI', 11),
                        bg=self.colors['light'], fg=self.colors['primary'],
                        relief='flat', bd=0, highlightthickness=1,
                        highlightcolor=self.colors['secondary'])
        entry.pack(fill=tk.X, pady=(5, 0), ipady=8)
        
        return entry
    
    def create_modern_dropdown(self, parent, label_text, variable, values):
        """Create modern styled dropdown"""
        field_frame = tk.Frame(parent, bg=self.colors['white'])
        field_frame.pack(fill=tk.X, pady=8)
        
        label = tk.Label(field_frame, text=label_text, font=('Segoe UI', 10, 'bold'),
                        fg=self.colors['primary'], bg=self.colors['white'])
        label.pack(anchor=tk.W)
        
        combo = ttk.Combobox(field_frame, textvariable=variable, values=values,
                            state="readonly", font=('Segoe UI', 11))
        combo.pack(fill=tk.X, pady=(5, 0), ipady=5)
        
        return combo
    
    def create_results_section(self, parent):
        """Create modern results section"""
        results_frame = tk.Frame(parent, bg=self.colors['white'])
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results header
        results_header = tk.Frame(results_frame, bg=self.colors['white'])
        results_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(results_header, text="🎯 Personalized Recommendations", 
                font=('Segoe UI', 16, 'bold'),
                fg=self.colors['primary'], bg=self.colors['white']).pack(anchor=tk.W)
        
        # Notebook for different views
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Recommendations tab
        self.create_recommendations_tab()
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Charts tab
        self.create_charts_tab()
    
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="🍽️ Food Recommendations")
        
        # Recommendations list
        columns = ('Food', 'Category', 'Calories', 'Protein', 'Carbs', 'Score', 'Health Match')
        self.rec_tree = ttk.Treeview(rec_frame, columns=columns, show='headings', 
                                    style='Modern.Treeview', height=12)
        
        # Configure columns
        widths = {'Food': 200, 'Category': 120, 'Calories': 80, 'Protein': 80, 
                 'Carbs': 80, 'Score': 80, 'Health Match': 150}
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            self.rec_tree.column(col, width=widths.get(col, 100), minwidth=60)
        
        # Scrollbar for recommendations
        rec_scroll = ttk.Scrollbar(rec_frame, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scroll.set)
        
        self.rec_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0), pady=20)
        rec_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        
        # Bind selection
        self.rec_tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
        # Details panel
        self.details_frame = tk.Frame(rec_frame, bg=self.colors['light'], width=300)
        self.details_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        self.details_frame.pack_propagate(False)
        
        # Details content
        details_label = tk.Label(self.details_frame, text="📋 Food Details", 
                                font=('Segoe UI', 12, 'bold'),
                                fg=self.colors['primary'], bg=self.colors['light'])
        details_label.pack(anchor=tk.W, padx=15, pady=(15, 10))
        
        self.details_text = tk.Text(self.details_frame, wrap=tk.WORD, 
                                   font=('Segoe UI', 10), bg=self.colors['white'],
                                   fg=self.colors['dark_gray'], relief='flat',
                                   padx=15, pady=15)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information and AI analysis.")
    
    def create_analysis_tab(self):
        """Create analysis tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Nutritional Analysis")
        
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, 
                                    font=('Segoe UI', 11), bg=self.colors['white'],
                                    fg=self.colors['dark_gray'], padx=20, pady=20)
        analysis_scroll = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scroll.set)
        
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.analysis_text.insert(tk.END, "Click 'Analyze Profile' to see your personalized nutritional targets and medical calculations.")
    
    def create_charts_tab(self):
        """Create charts tab"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="📈 Visual Insights")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100, facecolor=self.colors['white'])
        self.canvas = FigureCanvasTkAgg(self.fig, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Initialize empty charts
        self.update_charts([])
    
    def create_status_bar(self, parent):
        """Create modern status bar"""
        status_frame = tk.Frame(parent, bg=self.colors['primary'], height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="🚀 Initializing AI system...")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Segoe UI', 10), fg='white', bg=self.colors['primary'])
        status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # System stats on the right
        self.stats_var = tk.StringVar(value="")
        stats_label = tk.Label(status_frame, textvariable=self.stats_var,
                              font=('Segoe UI', 9), fg=self.colors['gray'], bg=self.colors['primary'])
        stats_label.pack(side=tk.RIGHT, padx=20, pady=10)
    
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
                    color = self.colors['accent']
                elif bmi < 30:
                    category = "Overweight"
                    color = self.colors['warning']
                else:
                    category = "Obese"
                    color = self.colors['danger']
                
                self.bmi_var.set(f"{bmi:.1f} - {category}")
                self.bmi_label.configure(fg=color)
            else:
                self.bmi_var.set("Enter valid weight and height")
                self.bmi_label.configure(fg=self.colors['gray'])
        except:
            self.bmi_var.set("BMI will be calculated")
            self.bmi_label.configure(fg=self.colors['gray'])
    
    def initialize_system(self):
        """Initialize the recommendation system"""
        def init_in_background():
            try:
                self.status_var.set("🔄 Loading food database...")
                self.master.update()
                
                self.recommender = HealthAwareMLPRecommender(self.update_status)
                
                # Update category dropdown
                if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
                    categories = ['All'] + sorted(self.recommender.food_data['Category'].unique())
                    self.category_dropdown['values'] = categories
                
                # Update stats
                stats = self.recommender.get_stats()
                self.stats_var.set(f"📊 {stats['total_items']} foods | {len(stats['categories'])} categories | {stats['loading_time']:.1f}s")
                
                self.status_var.set("✅ AI system ready! Enter your profile and get recommendations.")
                
                # Enable buttons
                self.recommend_btn.configure(state='normal')
                self.analyze_btn.configure(state='normal')
                
            except Exception as e:
                self.status_var.set(f"❌ Error initializing system: {str(e)}")
                messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{str(e)}")
        
        # Disable buttons during initialization
        self.recommend_btn.configure(state='disabled')
        self.analyze_btn.configure(state='disabled')
        
        # Run initialization in a separate thread-like manner
        self.master.after(100, init_in_background)
    
    def update_status(self, message):
        """Update status message"""
        self.status_var.set(f"🔄 {message}")
        self.master.update()
    
    def get_user_profile(self):
        """Get user profile from inputs"""
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
    
    def analyze_profile(self):
        """Analyze user profile and show nutritional targets"""
        if not self.recommender:
            messagebox.showwarning("Warning", "System not ready yet!")
            return
        
        try:
            user_profile = self.get_user_profile()
            
            # Calculate nutritional targets
            nutritional_data = self.recommender.nutrition_calculator.calculate_nutritional_targets(user_profile)
            
            # Display analysis
            self.analysis_text.delete(1.0, tk.END)
            
            # Header
            self.analysis_text.insert(tk.END, "🧠 AI NUTRITIONAL ANALYSIS\n", "header")
            self.analysis_text.insert(tk.END, "=" * 50 + "\n\n", "separator")
            
            # Personal metrics
            self.analysis_text.insert(tk.END, "👤 Your Health Profile:\n", "subheader")
            self.analysis_text.insert(tk.END, f"• Weight: {user_profile['weight']} kg\n")
            self.analysis_text.insert(tk.END, f"• Height: {user_profile['height']} cm\n")
            self.analysis_text.insert(tk.END, f"• Age: {user_profile['age']} years\n")
            self.analysis_text.insert(tk.END, f"• Gender: {user_profile['gender']}\n")
            self.analysis_text.insert(tk.END, f"• Activity: {user_profile['activity_level']}\n")
            self.analysis_text.insert(tk.END, f"• Goal: {user_profile['weight_goal']}\n\n")
            
            # Health conditions
            if user_profile['health_conditions']:
                self.analysis_text.insert(tk.END, "🏥 Health Conditions:\n", "subheader")
                for condition in user_profile['health_conditions']:
                    self.analysis_text.insert(tk.END, f"• {condition}\n")
                self.analysis_text.insert(tk.END, "\n")
            
            # Metabolic calculations
            self.analysis_text.insert(tk.END, "🔬 Metabolic Calculations:\n", "subheader")
            self.analysis_text.insert(tk.END, f"• BMR (Basal Metabolic Rate): {nutritional_data['bmr']:.0f} kcal/day\n")
            self.analysis_text.insert(tk.END, f"• TDEE (Total Daily Energy): {nutritional_data['tdee']:.0f} kcal/day\n")
            self.analysis_text.insert(tk.END, f"• Target Calories: {nutritional_data['daily_targets']['calories']:.0f} kcal/day\n\n")
            
            # Daily targets
            daily = nutritional_data['daily_targets']
            self.analysis_text.insert(tk.END, "🎯 Daily Nutritional Targets:\n", "subheader")
            self.analysis_text.insert(tk.END, f"• Protein: {daily['protein_min']:.0f}-{daily['protein_max']:.0f} g\n")
            self.analysis_text.insert(tk.END, f"• Carbohydrates: {daily['carbs_min']:.0f}-{daily['carbs_max']:.0f} g\n")
            self.analysis_text.insert(tk.END, f"• Fat: {daily['fat_min']:.0f}-{daily['fat_max']:.0f} g\n")
            self.analysis_text.insert(tk.END, f"• Sugar (max): {daily['sugar_max']:.0f} g\n")
            self.analysis_text.insert(tk.END, f"• Fiber (min): {daily['fiber_min']:.0f} g\n")
            self.analysis_text.insert(tk.END, f"• Sodium (max): {daily['sodium_max']:.0f} mg\n\n")
            
            # Meal breakdown
            meal_type = self.meal_type_var.get().lower()
            if meal_type in nutritional_data['meal_targets']:
                meal = nutritional_data['meal_targets'][meal_type]
                self.analysis_text.insert(tk.END, f"🍽️ {self.meal_type_var.get()} Targets:\n", "subheader")
                self.analysis_text.insert(tk.END, f"• Calories: {meal['calories']:.0f} kcal\n")
                self.analysis_text.insert(tk.END, f"• Protein: {meal['protein_min']:.0f}-{meal['protein_max']:.0f} g\n")
                self.analysis_text.insert(tk.END, f"• Carbs: {meal['carbs_min']:.0f}-{meal['carbs_max']:.0f} g\n")
                self.analysis_text.insert(tk.END, f"• Fat: {meal['fat_min']:.0f}-{meal['fat_max']:.0f} g\n\n")
            
            # AI insights
            self.analysis_text.insert(tk.END, "🤖 AI Insights:\n", "subheader")
            if user_profile['health_conditions']:
                self.analysis_text.insert(tk.END, "• MLP models will prioritize foods suitable for your health conditions\n")
                self.analysis_text.insert(tk.END, "• Nutritional targets have been adjusted based on medical guidelines\n")
            else:
                self.analysis_text.insert(tk.END, "• Using general health optimization model\n")
                self.analysis_text.insert(tk.END, "• Recommendations will focus on overall nutritional balance\n")
            
            self.analysis_text.insert(tk.END, "• Neural network will find foods matching your calculated needs\n")
            self.analysis_text.insert(tk.END, "• Lower MLP scores indicate better nutritional matches\n")
            
            # Configure text styling
            self.analysis_text.tag_configure("header", font=('Segoe UI', 14, 'bold'), foreground=self.colors['primary'])
            self.analysis_text.tag_configure("subheader", font=('Segoe UI', 12, 'bold'), foreground=self.colors['secondary'])
            self.analysis_text.tag_configure("separator", foreground=self.colors['gray'])
            
            # Switch to analysis tab
            self.notebook.select(1)
            
            self.status_var.set("✅ Profile analyzed! Ready for recommendations.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error analyzing profile: {str(e)}")
    
    def get_recommendations(self):
        """Get food recommendations"""
        if not self.recommender:
            messagebox.showwarning("Warning", "System not ready yet!")
            return
        
        try:
            user_profile = self.get_user_profile()
            meal_type = self.meal_type_var.get().lower()
            max_results = int(self.max_results_var.get())
            
            self.status_var.set("🧠 AI is analyzing thousands of foods...")
            self.master.update()
            
            # Get recommendations
            recommendations = self.recommender.get_recommendations(user_profile, meal_type, max_results)
            
            if recommendations:
                self.current_recommendations = recommendations
                self.display_recommendations(recommendations)
                self.update_charts(recommendations)
                
                # Switch to recommendations tab
                self.notebook.select(0)
                
                health_conditions = user_profile['health_conditions']
                condition_text = ", ".join(health_conditions) if health_conditions else "general health"
                self.status_var.set(f"✅ Found {len(recommendations)} perfect matches for {condition_text}!")
                
            else:
                messagebox.showinfo("No Results", 
                    "No suitable foods found. Try:\n"
                    "• Selecting 'All' categories\n"
                    "• Adjusting your health profile\n"
                    "• Choosing a different meal type")
                self.status_var.set("❌ No matches found - try different settings")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {str(e)}")
            self.status_var.set("❌ Error occurred during recommendation")
    
    def display_recommendations(self, recommendations):
        """Display recommendations in the tree view"""
        # Clear previous results
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        # Add recommendations
        for i, rec in enumerate(recommendations, 1):
            suitable_conditions = rec['suitable_for_conditions']
            health_match = ", ".join(suitable_conditions) if suitable_conditions else "General"
            
            self.rec_tree.insert('', 'end', values=(
                f"{i}. {rec['name'][:25]}{'...' if len(rec['name']) > 25 else ''}",
                rec['category'],
                f"{rec['calories']:.0f}",
                f"{rec['protein']:.1f}g",
                f"{rec['carbs']:.1f}g",
                f"{rec['mlp_score']:.3f}",
                health_match
            ))
        
        # Select first item
        if self.rec_tree.get_children():
            first_item = self.rec_tree.get_children()[0]
            self.rec_tree.selection_set(first_item)
            self.rec_tree.focus(first_item)
            self.show_food_details(None)
    
    def show_food_details(self, event):
        """Show details for selected food"""
        selected_items = self.rec_tree.selection()
        if not selected_items or not self.current_recommendations:
            return
        
        item = selected_items[0]
        item_index = self.rec_tree.index(item)
        
        if item_index < len(self.current_recommendations):
            rec = self.current_recommendations[item_index]
            
            # Clear and update details
            self.details_text.delete(1.0, tk.END)
            
            # Food header
            self.details_text.insert(tk.END, f"🍽️ {rec['name']}\n", "title")
            self.details_text.insert(tk.END, f"Category: {rec['category']}\n\n", "subtitle")
            
            # MLP Analysis
            self.details_text.insert(tk.END, "🧠 AI Analysis:\n", "header")
            self.details_text.insert(tk.END, f"• MLP Score: {rec['mlp_score']:.4f}\n")
            self.details_text.insert(tk.END, f"• Model Used: {rec['model_type']}\n")
            self.details_text.insert(tk.END, f"• Health Prediction: {rec['predicted_health_score']:.3f}\n")
            
            if rec['mlp_score'] < 1.0:
                match_quality = "🟢 Excellent Match"
            elif rec['mlp_score'] < 2.0:
                match_quality = "🟡 Good Match"
            else:
                match_quality = "🟠 Fair Match"
            
            self.details_text.insert(tk.END, f"• Match Quality: {match_quality}\n\n")
            
            # Nutrition per 100g
            self.details_text.insert(tk.END, "📊 Nutrition (per 100g):\n", "header")
            self.details_text.insert(tk.END, f"• Energy: {rec['calories']:.0f} kcal\n")
            self.details_text.insert(tk.END, f"• Protein: {rec['protein']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Carbohydrates: {rec['carbs']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sugar: {rec['sugar']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Fiber: {rec['fiber']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Fat: {rec['fat']:.1f} g\n")
            self.details_text.insert(tk.END, f"• Sodium: {rec['sodium']:.0f} mg\n\n")
            
            # Health compatibility
            if rec['suitable_for_conditions']:
                self.details_text.insert(tk.END, "✅ Suitable For:\n", "good")
                for condition in rec['suitable_for_conditions']:
                    self.details_text.insert(tk.END, f"• {condition}\n", "good")
                self.details_text.insert(tk.END, "\n")
            
            # Why recommended
            self.details_text.insert(tk.END, "💡 Why Recommended:\n", "header")
            self.details_text.insert(tk.END, f"{rec['explanation']}\n")
            
            # Configure styling
            self.details_text.tag_configure("title", font=('Segoe UI', 12, 'bold'), foreground=self.colors['primary'])
            self.details_text.tag_configure("subtitle", font=('Segoe UI', 10), foreground=self.colors['gray'])
            self.details_text.tag_configure("header", font=('Segoe UI', 10, 'bold'), foreground=self.colors['secondary'])
            self.details_text.tag_configure("good", foreground=self.colors['accent'])
    
    def update_charts(self, recommendations):
        """Update visualization charts"""
        self.fig.clear()
        
        if not recommendations:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '📊 Charts will appear here after getting recommendations', 
                   ha='center', va='center', fontsize=14, color=self.colors['gray'])
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return
        
        # Create 2x2 subplot layout
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)
        
        # 1. Macronutrient distribution
        avg_protein = np.mean([r['protein'] for r in recommendations])
        avg_carbs = np.mean([r['carbs'] for r in recommendations])
        avg_fat = np.mean([r['fat'] for r in recommendations])
        
        sizes = [avg_protein * 4, avg_carbs * 4, avg_fat * 9]
        labels = ['Protein', 'Carbs', 'Fat']
        colors = ['#2ed573', '#ffa502', '#ff4757']
        
        if sum(sizes) > 0:
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Average Macronutrients', fontweight='bold')
        
        # 2. MLP score distribution
        scores = [r['mlp_score'] for r in recommendations]
        ax2.hist(scores, bins=5, color='#3742fa', alpha=0.7, edgecolor='black')
        ax2.set_title('MLP Score Distribution', fontweight='bold')
        ax2.set_xlabel('MLP Score (lower = better)')
        ax2.set_ylabel('Count')
        
        # 3. Category distribution
        categories = {}
        for r in recommendations:
            cat = r['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            cats = list(categories.keys())
            counts = list(categories.values())
            ax3.bar(cats, counts, color='#2ed573', alpha=0.7)
            ax3.set_title('Food Categories', fontweight='bold')
            ax3.set_ylabel('Count')
            if len(max(cats, key=len)) > 8:
                ax3.tick_params(axis='x', rotation=45)
        
        # 4. Health suitability
        condition_counts = {'Diabetes': 0, 'Obesity': 0, 'Hypertension': 0, 'High_Cholesterol': 0}
        
        for r in recommendations:
            for condition in r['suitable_for_conditions']:
                if condition in condition_counts:
                    condition_counts[condition] += 1
        
        total = len(recommendations)
        condition_labels = [c.replace('_', ' ') for c in condition_counts.keys()]
        condition_percentages = [count/total*100 for count in condition_counts.values()]
        
        bars = ax4.bar(condition_labels, condition_percentages, color='#2ed573', alpha=0.7)
        ax4.set_title('Health Condition Suitability (%)', fontweight='bold')
        ax4.set_ylabel('Percentage')
        ax4.set_ylim(0, 100)
        ax4.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, pct in zip(bars, condition_percentages):
            if pct > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def reset_form(self):
        """Reset form to defaults"""
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
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information and AI analysis.")
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Click 'Analyze Profile' to see your personalized nutritional targets and medical calculations.")
        
        self.update_charts([])
        self.current_recommendations = []
        
        self.status_var.set("✅ Form reset to defaults - ready for new profile!")


def main():
    """Main application entry point"""
    print("🚀 Starting Modern MLP Food Recommendation System...")
    
    root = tk.Tk()
    app = ModernFoodRecommenderApp(root)
    
    # Center window
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - root.winfo_width()) // 2
    y = (screen_height - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    # Set minimum size
    root.minsize(1000, 600)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()