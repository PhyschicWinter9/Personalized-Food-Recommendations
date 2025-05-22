import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')  # Use non-interactive backend for compatibility

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


class FoodRecommendationSystem:
    
    def __init__(self, status_callback=None):
        # Callback function to update loading status
        self.status_callback = status_callback
        
        # Set default stats
        self.stats = {
            'total_items': 0,
            'categories': {},
            'loading_time': 0
        }
        
        # Load datasets
        start_time = time.time()
        self.load_data()
        self.stats['loading_time'] = time.time() - start_time
        
        # Features for nutritional analysis
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
        
        # Initialize list for available features
        self.features = []
        
        # Check which features are available in the data
        if hasattr(self, 'food_data') and not self.food_data.empty:
            available_features = []
            for feature in self.nutritional_features:
                if feature in self.food_data.columns:
                    available_features.append(feature)
            
            self.features = available_features
            
            # Make sure we have at least one feature
            if not self.features:
                self.update_status("Warning: No nutritional features found in data")
                # Try to use any numeric columns as features
                for col in self.food_data.columns:
                    if pd.api.types.is_numeric_dtype(self.food_data[col]):
                        self.features.append(col)
        
        # Dietary guidelines for each condition (based on research)
        self.dietary_guidelines = {
            'Diabetes': {
                'sugar_limit': 25,  # grams per day
                'carb_percent': (45, 60),  # % of total calories
                'fiber_per_1000kcal': 14,  # grams per 1000 kcal
                'protein_grams_per_kg': (0.8, 1.5),  # grams per kg body weight
                'fat_percent': (20, 35),  # % of total calories
                'sat_fat_percent': 7,  # % of total calories
            },
            'Obesity': {
                'carb_percent': (45, 65),  # % of total calories
                'protein_percent': (10, 35),  # % of total calories
                'protein_grams_per_kg': (1.2, 2.0),  # grams per kg body weight for weight loss
                'fat_percent': (20, 35),  # % of total calories
                'added_sugar_percent': 10,  # % of total calories
                'fiber_grams': (25, 30),  # grams per day
            },
            'Hypertension': {
                'sodium_limit': 2300,  # mg per day (ideal: 1500 mg)
                'saturated_fat_percent': 6,  # % of total calories
                'total_fat_percent': 27,  # % of total calories
                'cholesterol_limit': 150,  # mg per day
                'carb_percent': 55,  # % of total calories
                'protein_percent': 18,  # % of total calories
                'fiber_grams': 30,  # grams per day
            },
            'High_Cholesterol': {
                'total_fat_percent': 28,  # % of total calories
                'saturated_fat_percent': 8,  # % of total calories
                'added_sugar_percent': 10,  # % of total calories
                'cholesterol_limit': 200,  # mg per day
                'fiber_grams': (20, 35),  # grams per day
            }
        }
        
        # Prepare the data for Random Forest
        self.prepare_rf_data()
        
    def update_status(self, message):
        """Update loading status if callback is provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)  # Also print to console for debugging
        
    def load_data(self):
        """Load and combine all food datasets from CSV files in datasets folder"""
        try:
            # Path to the datasets folder
            dataset_folder = './datasets'
            
            # Get all CSV files in the datasets folder
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                self.update_status("No CSV files found in the datasets folder!")
                self.food_data = pd.DataFrame()
                return
                
            # List to store individual dataframes
            dataframes = []
            
            # Load each CSV file
            for file_path in csv_files:
                try:
                    # Get filename without extension to use as category
                    filename = os.path.basename(file_path)
                    category = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                    
                    self.update_status(f"Loading {category} data...")
                    
                    # Load CSV file
                    df = pd.read_csv(file_path)
                    
                    # Add category column if it doesn't exist
                    if 'Category' not in df.columns:
                        df['Category'] = category
                    
                    dataframes.append(df)
                    self.update_status(f"Loaded {len(df)} items from {filename}")
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            # Combine all dataframes
            if dataframes:
                self.update_status("Combining all food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.stats['total_items'] = len(self.food_data)
                
                # Get category counts
                if 'Category' in self.food_data.columns:
                    self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
                
                self.update_status(f"Successfully loaded {len(self.food_data)} food items")
                
                # Print column names for debugging
                self.update_status(f"Columns available: {', '.join(self.food_data.columns)}")
            else:
                self.update_status("No valid data files could be loaded")
                self.food_data = pd.DataFrame()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            # Create empty DataFrame if loading fails
            self.food_data = pd.DataFrame()

    def generate_suitability_scores(self, user_profile):
        """Generate suitability scores based on nutritional profiles and health metrics"""
        
        # Simplified version that calculates a suitability score for each food item
        # based on nutritional content and health guidelines
        
        scores = []
        weight_kg = user_profile.get('weight', 70)
        height_cm = user_profile.get('height', 170)
        age = user_profile.get('age', 45)
        
        # Calculate BMI
        bmi = weight_kg / ((height_cm/100) ** 2)
        
        # Determine calorie needs (very simplified)
        if user_profile.get('gender', 'Male') == 'Male':
            bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        
        # Adjust based on activity level (assuming moderate activity)
        daily_calories = bmr * 1.55
        
        # Get target percentages based on health profile
        carb_percent_target = (50, 60)  # Default carb percentage range
        protein_percent_target = (15, 20)  # Default protein percentage range
        fat_percent_target = (20, 30)  # Default fat percentage range
        
        # Adjust based on health conditions (simplified approach)
        if user_profile.get('diabetes', False):
            carb_percent_target = (40, 50)
            sugar_limit = 25  # grams
        else:
            sugar_limit = 50  # grams
            
        if user_profile.get('obesity', False) or bmi >= 30:
            fat_percent_target = (20, 25)
            daily_calories *= 0.8  # Calorie deficit for weight loss
            
        if user_profile.get('hypertension', False):
            sodium_limit = 1500  # mg
        else:
            sodium_limit = 2300  # mg
            
        if user_profile.get('high_cholesterol', False):
            cholesterol_limit = 200  # mg
            saturated_fat_percent = 5  # % of total calories
        else:
            cholesterol_limit = 300  # mg
            saturated_fat_percent = 10  # % of total calories
        
        # Calculate target macros in grams
        carb_grams_target = (daily_calories * carb_percent_target[0] / 100 / 4,
                             daily_calories * carb_percent_target[1] / 100 / 4)
        protein_grams_target = (daily_calories * protein_percent_target[0] / 100 / 4,
                               daily_calories * protein_percent_target[1] / 100 / 4)
        fat_grams_target = (daily_calories * fat_percent_target[0] / 100 / 9,
                           daily_calories * fat_percent_target[1] / 100 / 9)
        
        # Calculate meal-sized targets (assuming 3 meals + 2 snacks)
        meal_factor = 0.3  # 30% of daily intake for a meal
        snack_factor = 0.1  # 10% of daily intake for a snack
        
        if user_profile.get('meal_type', 'meal') == 'meal':
            factor = meal_factor
        else:
            factor = snack_factor
            
        meal_calories = daily_calories * factor
        meal_carbs = (carb_grams_target[0] * factor, carb_grams_target[1] * factor)
        meal_protein = (protein_grams_target[0] * factor, protein_grams_target[1] * factor)
        meal_fat = (fat_grams_target[0] * factor, fat_grams_target[1] * factor)
        meal_sugar = sugar_limit * factor
        meal_sodium = sodium_limit * factor
        meal_cholesterol = cholesterol_limit * factor
        
        # Process each food item
        for _, food in self.food_data.iterrows():
            score = 0
            
            # Get nutritional values with safe defaults
            calories = float(food.get('Energy(kcal) by calculation', 0))
            protein = float(food.get('Protein(g)', 0))
            carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
            sugar = float(food.get('SUGAR(g)', 0))
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            fat = float(food.get('Fat(g)', 0))
            sodium = float(food.get('Na(mg)', 0))
            cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            # 1. Calorie appropriateness (0-10 score)
            if calories <= meal_calories * 0.8:
                calorie_score = 10  # Under target
            elif calories <= meal_calories * 1.2:
                calorie_score = 7   # Near target
            elif calories <= meal_calories * 1.5:
                calorie_score = 4   # Above target
            else:
                calorie_score = 1   # Far above target
                
            # 2. Macronutrient balance (0-10 score)
            macro_score = 0
            
            # Protein score
            if meal_protein[0] <= protein <= meal_protein[1]:
                macro_score += 3  # Ideal range
            elif protein < meal_protein[0]:
                macro_score += 1  # Too low
            else:
                macro_score += 2  # Above range but protein is generally good
                
            # Carbs score
            if meal_carbs[0] <= carbs <= meal_carbs[1]:
                macro_score += 3  # Ideal range
            elif carbs < meal_carbs[0]:
                macro_score += 2  # Below range
            else:
                macro_score += 1  # Above range
                
            # Fat score
            if meal_fat[0] <= fat <= meal_fat[1]:
                macro_score += 3  # Ideal range
            elif fat < meal_fat[0]:
                macro_score += 2  # Below range
            else:
                macro_score += 1  # Above range
                
            # 3. Nutritional quality (0-10 score)
            quality_score = 0
            
            # Fiber quality
            if fiber >= 5:
                quality_score += 3  # High fiber
            elif fiber >= 2:
                quality_score += 2  # Moderate fiber
            else:
                quality_score += 1  # Low fiber
                
            # Sugar penalty
            if sugar <= meal_sugar * 0.3:
                quality_score += 3  # Low sugar
            elif sugar <= meal_sugar * 0.7:
                quality_score += 2  # Moderate sugar
            else:
                quality_score += 0  # High sugar
                
            # Sodium consideration
            if sodium <= meal_sodium * 0.3:
                quality_score += 2  # Low sodium
            elif sodium <= meal_sodium * 0.7:
                quality_score += 1  # Moderate sodium
            else:
                quality_score += 0  # High sodium
                
            # Cholesterol consideration
            if cholesterol <= meal_cholesterol * 0.3:
                quality_score += 2  # Low cholesterol
            elif cholesterol <= meal_cholesterol * 0.7:
                quality_score += 1  # Moderate cholesterol
            else:
                quality_score += 0  # High cholesterol
                
            # Calculate final score (weighted average)
            final_score = (calorie_score * 0.4) + (macro_score * 0.3) + (quality_score * 0.3)
            
            # Normalize to 0-100 scale
            normalized_score = (final_score / 10) * 100
            
            # Add to scores list
            scores.append({
                'food_id': len(scores),
                'food_name': food.get('Thai_Name', food.get('English_Name', f"Food {len(scores)}")),
                'category': food.get('Category', 'Unknown'),
                'calories': calories,
                'protein': protein,
                'carbs': carbs,
                'sugar': sugar,
                'fiber': fiber,
                'fat': fat,
                'sodium': sodium,
                'cholesterol': cholesterol,
                'score': normalized_score,
                'calorie_score': calorie_score,
                'macro_score': macro_score,
                'quality_score': quality_score
            })
            
        return scores
    
    def prepare_rf_data(self):
        """Prepare data for Random Forest model"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for Random Forest model")
            return
            
        # Check if all features exist in the dataset
        missing_features = [f for f in self.features if f not in self.food_data.columns]
        if missing_features:
            self.update_status(f"Warning: Missing features in dataset: {missing_features}")
            # Use only available features
            self.features = [f for f in self.features if f in self.food_data.columns]
            
        if not self.features:
            self.update_status("Error: No valid features available for recommendation")
            return
            
        try:
            # Extract features for Random Forest
            X = self.food_data[self.features].fillna(0)
            
            # Generate target labels for training
            # For regression: Use a health score based on nutritional guidelines
            y = self.generate_health_scores()
            
            # Standardize features
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, y, test_size=0.2, random_state=42)
            
            # Initialize and train Random Forest model
            self.rf_model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate model performance
            train_score = self.rf_model.score(X_train, y_train)
            test_score = self.rf_model.score(X_test, y_test)
            
            self.update_status(f"Random Forest model trained. R² on training: {train_score:.3f}, test: {test_score:.3f}")
            
            # Get feature importances
            feature_importances = self.rf_model.feature_importances_
            importance_dict = {feature: importance for feature, importance in zip(self.features, feature_importances)}
            
            # Sort features by importance
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            importance_str = ", ".join([f"{feature}: {importance:.3f}" for feature, importance in sorted_importances[:5]])
            
            self.update_status(f"Top 5 important features: {importance_str}")
            
        except Exception as e:
            self.update_status(f"Error preparing Random Forest model: {e}")
    
    def generate_health_scores(self):
        """Generate health scores for each food item based on nutritional guidelines"""
        scores = []
        
        for _, food in self.food_data.iterrows():
            score = 50  # Start with a neutral score
            
            # Get nutritional values with safe defaults
            calories = float(food.get('Energy(kcal) by calculation', 0))
            protein = float(food.get('Protein(g)', 0))
            carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
            sugar = float(food.get('SUGAR(g)', 0))
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            fat = float(food.get('Fat(g)', 0))
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            sodium = float(food.get('Na(mg)', 0))
            potassium = float(food.get('K(mg)', 0))
            calcium = float(food.get('Ca(mg)', 0))
            cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            # Apply general nutritional guidelines
            
            # Protein quality (higher is better)
            if protein >= 15:
                score += 10
            elif protein >= 8:
                score += 5
                
            # Fiber content (higher is better)
            if fiber >= 5:
                score += 10
            elif fiber >= 3:
                score += 5
                
            # Sugar content (lower is better)
            if sugar > 20:
                score -= 15
            elif sugar > 10:
                score -= 5
                
            # Fat profile
            if fat > 20:
                score -= 10
            elif fat > 10:
                score -= 5
            
            # Saturated fat (if available)
            if sat_fat > 8:
                score -= 15
            elif sat_fat > 4:
                score -= 5
                
            # Sodium content
            if sodium > 500:
                score -= 15
            elif sodium > 250:
                score -= 5
                
            # Potassium content (beneficial)
            if potassium > 300:
                score += 5
                
            # Calcium content (beneficial)
            if calcium > 200:
                score += 5
                
            # Cholesterol content
            if cholesterol > 100:
                score -= 10
            elif cholesterol > 50:
                score -= 5
                
            # Balance between calories and nutrients
            if calories > 0:
                # Protein density
                protein_density = protein / calories * 100
                if protein_density > 5:
                    score += 5
                
                # Fiber density
                fiber_density = fiber / calories * 100
                if fiber_density > 1.5:
                    score += 5
            
            # Clamp score to 0-100 range
            score = max(0, min(100, score))
            scores.append(score)
            
        return np.array(scores)
    
    def get_recommendations(self, user_preferences, health_profile, category_filter="All", max_recommendations=10):
        """Get food recommendations based on user preferences and health profile"""
        if not hasattr(self, 'rf_model') or len(self.food_data) == 0:
            return []
          
        try:
            # Extract user features for prediction
            user_features = np.array([user_preferences.get(feature, 0) for feature in self.features]).reshape(1, -1)
            
            # Scale user features
            user_scaled = self.scaler.transform(user_features)
            
            # Generate suitability scores based on health profile
            suitability_scores = self.generate_suitability_scores(health_profile)
            
            # Sort scores by descending order
            sorted_scores = sorted(suitability_scores, key=lambda x: x['score'], reverse=True)
            
            # Apply category filter if needed
            if category_filter != "All":
                filtered_scores = [item for item in sorted_scores if item['category'] == category_filter]
            else:
                filtered_scores = sorted_scores
                
            # Return top recommendations
            recommendations = filtered_scores[:max_recommendations]
            
            # Enhance recommendations with personalization insights
            for rec in recommendations:
                # Calculate distance from user preferences
                food_idx = rec['food_id']
                
                # Add suitability flags for different health conditions
                rec['suitable_for'] = []
                
                # Diabetes suitability
                if rec['sugar'] <= 10 and rec['fiber'] >= 3:
                    rec['suitable_for'].append('Diabetes')
                    
                # Weight management suitability
                if rec['calories'] <= 300 and rec['fiber'] >= 3:
                    rec['suitable_for'].append('Obesity')
                    
                # Hypertension suitability
                if rec['sodium'] <= 200:
                    rec['suitable_for'].append('Hypertension')
                    
                # Cholesterol management suitability
                if rec.get('cholesterol', 0) <= 50 and rec['fat'] <= 10:
                    rec['suitable_for'].append('High_Cholesterol')
            
            return recommendations
            
        except Exception as e:
            self.update_status(f"Error getting recommendations: {e}")
            return []
    
    def analyze_nutritional_profile(self, food_item):
        """Analyze a food item's nutritional profile and generate health recommendations"""
        if not isinstance(food_item, dict) and not hasattr(food_item, 'to_dict'):
            return {"error": "Invalid food item format"}
        
        # Convert to dictionary if it's a pandas Series
        if hasattr(food_item, 'to_dict'):
            food_item = food_item.to_dict()
        
        # Extract key nutritional values with safe defaults
        analysis = {
            "name": food_item.get('food_name', food_item.get('Thai_Name', food_item.get('English_Name', 'Unknown Food'))),
            "category": food_item.get('category', food_item.get('Category', 'Uncategorized')),
            "energy_kcal": float(food_item.get('calories', food_item.get('Energy(kcal) by calculation', 0))),
            "protein_g": float(food_item.get('protein', food_item.get('Protein(g)', 0))),
            "carbs_g": float(food_item.get('carbs', food_item.get('CHOCDF (g) Carbohydrate', 0))),
            "sugar_g": float(food_item.get('sugar', food_item.get('SUGAR(g)', 0))),
            "fiber_g": float(food_item.get('fiber', food_item.get('FIBTG (g) Dietary fibre', 0))),
            "fat_g": float(food_item.get('fat', food_item.get('Fat(g)', 0))),
            "sat_fat_g": float(food_item.get('FASAT (g) Saturated FA', 0)),
            "sodium_mg": float(food_item.get('sodium', food_item.get('Na(mg)', 0))),
            "potassium_mg": float(food_item.get('K(mg)', 0)),
            "cholesterol_mg": float(food_item.get('cholesterol', food_item.get('CHOLE(mg) Cholesterol', 0))),
            "suitable_for": food_item.get('suitable_for', []),
            "warnings": []
        }
        
        # Calculate macronutrient distribution
        total_calories_from_macros = 0
        if analysis["protein_g"] > 0:
            protein_calories = analysis["protein_g"] * 4
            total_calories_from_macros += protein_calories
        
        if analysis["carbs_g"] > 0:
            carbs_calories = analysis["carbs_g"] * 4
            total_calories_from_macros += carbs_calories
        
        if analysis["fat_g"] > 0:
            fat_calories = analysis["fat_g"] * 9
            total_calories_from_macros += fat_calories
        
        # Calculate percentages if we have energy information
        if total_calories_from_macros > 0:
            analysis["protein_percent"] = (analysis["protein_g"] * 4 / total_calories_from_macros) * 100
            analysis["carbs_percent"] = (analysis["carbs_g"] * 4 / total_calories_from_macros) * 100
            analysis["fat_percent"] = (analysis["fat_g"] * 9 / total_calories_from_macros) * 100
        else:
            analysis["protein_percent"] = 0
            analysis["carbs_percent"] = 0
            analysis["fat_percent"] = 0
        
        # Calculate sugar as percentage of carbs
        if analysis["carbs_g"] > 0:
            analysis["sugar_percent_of_carbs"] = (analysis["sugar_g"] / analysis["carbs_g"]) * 100
        else:
            analysis["sugar_percent_of_carbs"] = 0
        
        # Calculate saturated fat as percentage of total fat
        if analysis["fat_g"] > 0:
            analysis["sat_fat_percent"] = (analysis["sat_fat_g"] / analysis["fat_g"]) * 100
        else:
            analysis["sat_fat_percent"] = 0
        
        # Calculate fiber per 1000 kcal
        if analysis["energy_kcal"] > 0:
            analysis["fiber_per_1000kcal"] = (analysis["fiber_g"] / analysis["energy_kcal"]) * 1000
        else:
            analysis["fiber_per_1000kcal"] = 0
        
        # Generate warnings based on nutritional content
        if analysis["sugar_g"] > 15:
            analysis["warnings"].append("High sugar content")
        
        if analysis["sodium_mg"] > 500:
            analysis["warnings"].append("High sodium content")
            
        if analysis["fat_g"] > 20:
            analysis["warnings"].append("High fat content")
            
        if analysis["sat_fat_g"] > 5:
            analysis["warnings"].append("High saturated fat content")
            
        if analysis["cholesterol_mg"] > 100:
            analysis["warnings"].append("High cholesterol content")
        
        # Generate summary and recommendations
        analysis["summary"] = self.generate_food_summary(analysis)
        analysis["recommendations"] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_food_summary(self, analysis):
        """Generate a summary of the food's nutritional profile"""
        summary = f"{analysis['name']} is a {analysis['category']} food with {analysis['energy_kcal']:.0f} kcal, "
        summary += f"{analysis['protein_g']:.1f}g protein, {analysis['carbs_g']:.1f}g carbs, and {analysis['fat_g']:.1f}g fat. "
        
        # Add notable nutritional characteristics
        notable_points = []
        
        if analysis["fiber_g"] >= 5:
            notable_points.append("high in fiber")
        elif analysis["fiber_g"] >= 3:
            notable_points.append("good source of fiber")
        
        if analysis["sodium_mg"] >= 500:
            notable_points.append("high in sodium")
        elif analysis["sodium_mg"] <= 140:
            notable_points.append("low in sodium")
        
        if analysis["sugar_g"] >= 15:
            notable_points.append("high in sugar")
        elif analysis["sugar_g"] <= 5:
            notable_points.append("low in sugar")
        
        if analysis["sat_fat_g"] >= 5:
            notable_points.append("high in saturated fat")
        elif analysis["sat_fat_g"] <= 1:
            notable_points.append("low in saturated fat")
        
        if notable_points:
            summary += "It is " + ", ".join(notable_points) + "."
        
        return summary
    
    def generate_recommendations(self, analysis):
        """Generate health recommendations based on the food analysis"""
        recommendations = []
        
        # Generate recommendations based on the nutritional profile
        if "Diabetes" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with diabetes due to its balanced nutritional profile.")
        elif analysis["sugar_g"] > 15:
            recommendations.append("People with diabetes should limit consumption due to high sugar content.")
            
        if "Obesity" in analysis["suitable_for"]:
            recommendations.append("Suitable for weight management due to its lower calorie profile.")
        elif analysis["energy_kcal"] > 300:
            recommendations.append("Those managing their weight should limit portion size due to high calorie content.")
            
        if "Hypertension" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with hypertension due to its lower sodium content.")
        elif analysis["sodium_mg"] > 400:
            recommendations.append("People with hypertension should limit consumption due to high sodium content.")
            
        if "High_Cholesterol" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with high cholesterol due to its heart-healthy profile.")
        elif analysis["sat_fat_g"] > 5 or analysis["cholesterol_mg"] > 100:
            recommendations.append("People with high cholesterol should limit consumption due to saturated fat/cholesterol content.")
        
        # General recommendations
        if not recommendations:
            if len(analysis["warnings"]) <= 1:
                recommendations.append("Generally acceptable for most diets in moderation.")
            else:
                recommendations.append("Best consumed occasionally as part of a varied and balanced diet.")
        
        return recommendations
    
    def get_stats(self):
        """Get statistics about loaded data"""
        stats = {
            'total_items': len(self.food_data) if hasattr(self, 'food_data') else 0,
            'categories': {},
            'loading_time': 0
        }
        
        # Count items by category
        if hasattr(self, 'food_data') and 'Category' in self.food_data.columns:
            category_counts = self.food_data['Category'].value_counts().to_dict()
            stats['categories'] = category_counts
            
        # Add loading time
        stats['loading_time'] = self.stats.get('loading_time', 0)
            
        return stats
    
    def get_food_details(self, food_name):
        """Get detailed information about a specific food item"""
        if not hasattr(self, 'food_data') or self.food_data.empty:
            return {}
            
        # Find the food item by name
        food_items = self.food_data[
            (self.food_data['Thai_Name'] == food_name) | 
            (self.food_data['English_Name'] == food_name)
        ]
        
        if food_items.empty:
            return {}
            
        # Return the first matching food item as a dictionary
        return food_items.iloc[0].to_dict()


class FoodRecommenderUI:
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Personalized Food Recommendation System")
        self.master.geometry("1200x750")
        self.master.configure(bg="#f5f5f5")
        
        # Get screen dimensions to allow responsive scaling
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        # Set theme
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            # Fall back to default theme if clam is not available
            pass
        
        # Configure colors with better accessibility
        self.primary_color = "#2980b9"  # Darker blue for better contrast
        self.secondary_color = "#27ae60"  # Darker green for better contrast
        self.accent_color = "#e74c3c"  # Red
        self.bg_color = "#f5f5f5"  # Light gray
        self.text_color = "#2c3e50"  # Dark blue/gray
        self.light_text_color = "#7f8c8d"  # For less important text
        
        # Configure styles with better fonts and sizes
        default_font_size = self.calculate_font_size(10)
        header_font_size = self.calculate_font_size(16)
        subheader_font_size = self.calculate_font_size(12)
        
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Arial', default_font_size))
        self.style.configure('TButton', font=('Arial', default_font_size, 'bold'))
        self.style.configure('Primary.TButton', background=self.primary_color, foreground='white')
        self.style.configure('Secondary.TButton', background=self.secondary_color, foreground='white')
        self.style.configure('Header.TLabel', font=('Arial', header_font_size, 'bold'), background=self.bg_color, foreground=self.text_color)
        self.style.configure('Subheader.TLabel', font=('Arial', subheader_font_size, 'bold'), background=self.bg_color, foreground=self.text_color)
        self.style.configure('Stats.TLabel', font=('Arial', 9), background=self.bg_color, foreground=self.light_text_color)
        
        # Configure Treeview with improved styling
        self.style.configure('Treeview', font=('Arial', default_font_size), rowheight=30)
        self.style.configure('Treeview.Heading', font=('Arial', default_font_size, 'bold'))
        self.style.map('Treeview', 
                     background=[('selected', self.primary_color)],
                     foreground=[('selected', 'white')])
        
        # Initialize the recommendation system if not provided
        self.recommender = recommender or FoodRecommendationSystem()
        
        # Create UI elements
        self.create_main_layout()
        
        # Nutritional target values
        self.targets = {
            'Energy(kcal) by calculation': 500,
            'Protein(g)': 20,
            'CHOCDF (g) Carbohydrate': 30,
            'SUGAR(g)': 5,
            'FIBTG (g) Dietary fibre': 8,
            'Fat(g)': 15
        }
        
        # Update stats display
        self.update_stats_display()
        
    def calculate_font_size(self, base_size):
        """Calculate font size based on screen resolution for better scaling"""
        # Simple scaling logic - can be adjusted as needed
        scaling_factor = min(self.screen_width / 1920, self.screen_height / 1080)
        return max(int(base_size * scaling_factor), 8)  # Ensure minimum size of 8
        
    def create_main_layout(self):
        """Create the main layout with modern UI"""
        # Main container with padding that scales with window size
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Header with title and recommendation button
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title on the left with better styling
        title_label = ttk.Label(header_frame, 
                              text="Personalized Food Recommendation System", 
                              style='Header.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Subtitle for approach
        subtitle_label = ttk.Label(header_frame, 
                                 text="Using Health and Biometric Data Analysis", 
                                 foreground=self.light_text_color,
                                 font=('Arial', self.calculate_font_size(12)))
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Buttons on the right
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Get Recommendations button with better styling
        recommend_btn = ttk.Button(button_frame, text="Get Recommendations", 
                                  command=self.show_recommendations, style='Primary.TButton')
        recommend_btn.pack(side=tk.RIGHT, padx=5)
        
        # Reset button
        reset_btn = ttk.Button(button_frame, text="Reset Settings", 
                              command=self.reset_preferences)
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create two panels with flexible sizing
        panel_container = ttk.Frame(main_container)
        panel_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input controls - now with proportion-based width
        left_panel_container = ttk.Frame(panel_container)
        left_panel_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 15), pady=0, ipadx=5)
        
        # Add canvas with scrollbar for left panel
        canvas = tk.Canvas(left_panel_container, borderwidth=0, highlightthickness=0, 
                         width=300)  # Fixed starting width
        scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas that fills the entire width
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar with better proportions
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create the actual content in the scrollable frame
        self.create_input_panel(scrollable_frame)
        
        # Right panel - Results with flexible width
        right_panel = ttk.Frame(panel_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_results_panel(right_panel)
        
        # Nutrition chart panel
        chart_panel = ttk.Frame(main_container, padding=10)
        chart_panel.pack(fill=tk.X, pady=(15, 0))
        
        self.create_chart_panel(chart_panel)
        
        # Create status bar at the bottom with better styling
        status_frame = ttk.Frame(self.master, relief=tk.GROOVE, padding=(5, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create stats display at the bottom right
        self.stats_frame = ttk.Frame(status_frame, style='Stats.TLabel')
        self.stats_frame.pack(side=tk.RIGHT, padx=5)
        
        # Create progress bar at the bottom right with better styling
        self.progress_frame = ttk.Frame(status_frame)
        self.progress_frame.pack(side=tk.RIGHT, padx=5)
        
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", 
                                      length=100, mode="determinate", 
                                      style="Horizontal.TProgressbar")
        self.progress.pack(side=tk.RIGHT)
        self.progress["value"] = 0
        
        # Style for progress bar
        self.style.configure("Horizontal.TProgressbar", 
                           background=self.secondary_color,
                           troughcolor=self.bg_color)
        
        # Make the canvas resize with the window
        self.master.bind("<Configure>", self.on_window_configure)
        
        # Handle mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def on_window_configure(self, event=None):
        """Handle window resize events for responsive UI"""
        # Minimum width for input panel
        min_width = 280
        
        # Get 25% of window width but not less than min_width
        desired_width = max(min_width, int(self.master.winfo_width() * 0.25))
        
        # Find the canvas in the left panel
        for widget in self.master.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Frame):
                                for great_grandchild in grandchild.winfo_children():
                                    if isinstance(great_grandchild, tk.Canvas):
                                        # Resize the canvas width
                                        great_grandchild.configure(width=desired_width)
                                        
                                        # Also resize the scrollable frame inside
                                        for item_id in great_grandchild.find_all():
                                            if great_grandchild.type(item_id) == "window":
                                                great_grandchild.itemconfigure(item_id, width=desired_width)
        
    def create_input_panel(self, parent):
        """Create the input panel with nutrition preferences and health conditions"""
        # Title with better spacing
        ttk.Label(parent, text="Health & Biometric Profile", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Health information frame with better styling
        health_frame = ttk.LabelFrame(parent, text="Personal Information", padding=10)
        health_frame.pack(fill=tk.X, pady=10)
        
        # Use grid with consistent spacing
        grid_padx = 5
        grid_pady = 8
        
        # Weight with validation
        ttk.Label(health_frame, text="Weight (kg):").grid(row=0, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.weight_var = tk.DoubleVar(value=70.0)
        weight_entry = ttk.Entry(health_frame, textvariable=self.weight_var, width=10)
        weight_entry.grid(row=0, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.add_validation(weight_entry, "float")
        
        # Height with validation
        ttk.Label(health_frame, text="Height (cm):").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.height_var = tk.DoubleVar(value=170.0)
        height_entry = ttk.Entry(health_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.add_validation(height_entry, "float")
        
        # Age with validation
        ttk.Label(health_frame, text="Age:").grid(row=2, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.age_var = tk.IntVar(value=45)
        age_entry = ttk.Entry(health_frame, textvariable=self.age_var, width=10)
        age_entry.grid(row=2, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.add_validation(age_entry, "int")
        
        # Gender
        ttk.Label(health_frame, text="Gender:").grid(row=3, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.gender_var = tk.StringVar(value="Male")
        gender_combo = ttk.Combobox(health_frame, textvariable=self.gender_var, values=['Male', 'Female', 'Other'], width=10)
        gender_combo.grid(row=3, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Add BMI calculation
        ttk.Label(health_frame, text="BMI:").grid(row=4, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.bmi_var = tk.StringVar(value="Computing...")
        ttk.Label(health_frame, textvariable=self.bmi_var).grid(row=4, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Activity level
        ttk.Label(health_frame, text="Activity Level:").grid(row=5, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.activity_var = tk.StringVar(value="Moderate")
        activity_combo = ttk.Combobox(health_frame, textvariable=self.activity_var, 
                                    values=['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'], width=12)
        activity_combo.grid(row=5, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Meal type preference
        ttk.Label(health_frame, text="Searching for:").grid(row=6, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.meal_type_var = tk.StringVar(value="Meal")
        meal_type_combo = ttk.Combobox(health_frame, textvariable=self.meal_type_var, 
                                     values=['Meal', 'Snack', 'Beverage', 'Dessert'], width=12)
        meal_type_combo.grid(row=6, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Calculate BMI when weight or height changes
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()  # Initial calculation
        
        # Health conditions frame with improved styling
        conditions_frame = ttk.LabelFrame(parent, text="Health Conditions", padding=10)
        conditions_frame.pack(fill=tk.X, pady=(15, 10))
        
        # Create health condition checkboxes
        self.diabetes_var = tk.BooleanVar(value=False)
        self.obesity_var = tk.BooleanVar(value=False)
        self.hypertension_var = tk.BooleanVar(value=False)
        self.cholesterol_var = tk.BooleanVar(value=False)
        
        # Add tooltips with explanations
        diabetes_check = ttk.Checkbutton(conditions_frame, text="Diabetes", variable=self.diabetes_var)
        diabetes_check.pack(anchor=tk.W, padx=5, pady=5)
        self.create_tooltip(diabetes_check, "Recommendations for blood sugar management")
        
        obesity_check = ttk.Checkbutton(conditions_frame, text="Weight Management", variable=self.obesity_var)
        obesity_check.pack(anchor=tk.W, padx=5, pady=5)
        self.create_tooltip(obesity_check, "Recommendations for weight management and calorie control")
        
        hypertension_check = ttk.Checkbutton(conditions_frame, text="Hypertension (High Blood Pressure)", variable=self.hypertension_var)
        hypertension_check.pack(anchor=tk.W, padx=5, pady=5)
        self.create_tooltip(hypertension_check, "Foods with lower sodium and heart-healthy nutrients")
        
        cholesterol_check = ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var)
        cholesterol_check.pack(anchor=tk.W, padx=5, pady=5)
        self.create_tooltip(cholesterol_check, "Foods with better fat profile and heart-healthy nutrients")
        
        # Frame for nutritional inputs with better styling
        input_frame = ttk.LabelFrame(parent, text="Nutritional Preferences", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Energy/Calories preference
        self.create_slider(input_frame, "Energy (kcal):", "calories_var", 500, 0, 1000, 0)
        
        # Protein preference
        self.create_slider(input_frame, "Protein (g):", "protein_var", 20, 0, 50, 1)
        
        # Carbs preference
        self.create_slider(input_frame, "Carbohydrates (g):", "carbs_var", 30, 0, 100, 2)
        
        # Sugar preference
        self.create_slider(input_frame, "Sugar (g):", "sugar_var", 5, 0, 30, 3)
        
        # Fiber preference
        self.create_slider(input_frame, "Dietary Fiber (g):", "fiber_var", 8, 0, 20, 4)
        
        # Fat preference
        self.create_slider(input_frame, "Fat (g):", "fat_var", 15, 0, 40, 5)
        
        # Recommendation settings with better styling
        settings_frame = ttk.LabelFrame(parent, text="Recommendation Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Number of recommendations
        ttk.Label(settings_frame, text="Number of recommendations:").grid(row=0, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.num_recommendations_var = tk.IntVar(value=10)
        ttk.Combobox(settings_frame, textvariable=self.num_recommendations_var, values=[5, 10, 15, 20, 25], width=5).grid(row=0, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Filter by category with tooltip
        ttk.Label(settings_frame, text="Filter by category:").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_filter_var, width=15)
        self.category_combo.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.create_tooltip(self.category_combo, "Select a specific food category or 'All' to show foods from all categories")
        
        # Get all available categories
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty and 'Category' in self.recommender.food_data.columns:
            categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
        
        # Help/Info button
        help_btn = ttk.Button(parent, text="About NCDs", command=self.show_ncd_info)
        help_btn.pack(pady=15)
        
    def add_validation(self, entry_widget, validation_type):
        """Add validation to entry widgets"""
        if validation_type == "float":
            vcmd = (self.master.register(self.validate_float), '%P')
            entry_widget.configure(validate="key", validatecommand=vcmd)
        elif validation_type == "int":
            vcmd = (self.master.register(self.validate_int), '%P')
            entry_widget.configure(validate="key", validatecommand=vcmd)
    
    def validate_float(self, new_value):
        """Validate float input"""
        if new_value == "":
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False
    
    def validate_int(self, new_value):
        """Validate integer input"""
        if new_value == "":
            return True
        try:
            int(new_value)
            return True
        except ValueError:
            return False
            
    def calculate_bmi(self, *args):
        """Calculate BMI from weight and height"""
        try:
            weight = self.weight_var.get()
            height = self.height_var.get() / 100  # convert cm to m
            if weight > 0 and height > 0:
                bmi = weight / (height * height)
                
                # Determine BMI category
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
            self.bmi_var.set("Computing...")
    
    def show_ncd_info(self):
        """Show information about NCDs"""
        info_window = tk.Toplevel(self.master)
        info_window.title("About Non-Communicable Diseases (NCDs)")
        info_window.geometry("600x500")
        info_window.transient(self.master)  # Make window modal
        
        # Add content in a scrollable frame
        main_frame = ttk.Frame(info_window, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add content
        ttk.Label(scrollable_frame, text="Understanding Non-Communicable Diseases (NCDs)", 
                 font=('Arial', 16, 'bold')).pack(pady=(0, 15), anchor=tk.W)
        
        # Diabetes info
        ttk.Label(scrollable_frame, text="Diabetes", font=('Arial', 12, 'bold')).pack(pady=(10, 5), anchor=tk.W)
        ttk.Label(scrollable_frame, text="A chronic condition affecting how your body processes blood sugar (glucose). "
                                        "Dietary recommendations include limiting sugar and simple carbohydrates, "
                                        "choosing high-fiber foods, and maintaining consistency in carbohydrate intake.",
                 wraplength=550, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Obesity info
        ttk.Label(scrollable_frame, text="Weight Management", font=('Arial', 12, 'bold')).pack(pady=(10, 5), anchor=tk.W)
        ttk.Label(scrollable_frame, text="Managing weight involves balancing caloric intake with energy expenditure. "
                                        "Dietary recommendations include controlling portion sizes, choosing foods high in "
                                        "protein and fiber for satiety, and limiting foods high in added sugars and fats.",
                 wraplength=550, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Hypertension info
        ttk.Label(scrollable_frame, text="Hypertension (High Blood Pressure)", font=('Arial', 12, 'bold')).pack(pady=(10, 5), anchor=tk.W)
        ttk.Label(scrollable_frame, text="A condition in which the force of the blood against the artery walls is too high. "
                                        "Dietary recommendations include reducing sodium intake, increasing potassium-rich foods, "
                                        "limiting alcohol, and following the DASH (Dietary Approaches to Stop Hypertension) eating plan.",
                 wraplength=550, justify=tk.LEFT).pack(anchor=tk.W)
        
        # High Cholesterol info
        ttk.Label(scrollable_frame, text="High Cholesterol", font=('Arial', 12, 'bold')).pack(pady=(10, 5), anchor=tk.W)
        ttk.Label(scrollable_frame, text="Occurs when you have too much cholesterol in your blood. Dietary recommendations include "
                                        "reducing saturated and trans fats, increasing soluble fiber, choosing lean proteins, and "
                                        "incorporating foods containing plant sterols/stanols.",
                 wraplength=550, justify=tk.LEFT).pack(anchor=tk.W)
        
        # General recommendations
        ttk.Label(scrollable_frame, text="General Dietary Recommendations", font=('Arial', 12, 'bold')).pack(pady=(15, 5), anchor=tk.W)
        ttk.Label(scrollable_frame, text="• Eat plenty of fruits, vegetables, whole grains, and lean proteins\n"
                                        "• Choose foods rich in fiber, vitamins, and minerals\n"
                                        "• Limit processed foods, added sugars, and unhealthy fats\n"
                                        "• Practice portion control and mindful eating\n"
                                        "• Stay hydrated by drinking plenty of water\n"
                                        "• Consult with healthcare professionals for personalized advice",
                 justify=tk.LEFT).pack(anchor=tk.W)
        
        # Close button
        ttk.Button(scrollable_frame, text="Close", command=info_window.destroy).pack(pady=20)
        
        # Handle mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set focus to the new window
        info_window.focus_set()
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=text, background="#ffffe0", relief="solid", borderwidth=1, padding=5)
            label.pack()
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        
    def create_slider(self, parent, label_text, var_name, default_value, min_val, max_val, row):
        """Create a slider with label and value display"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=8, sticky=tk.W)
        
        # Create slider variable
        slider_var = tk.IntVar(value=default_value)
        setattr(self, var_name, slider_var)
        
        # Create frame for slider and value to ensure proper layout
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, columnspan=2, padx=5, pady=8, sticky=tk.W)
        
        # Create slider with improved styling and proportional length
        slider = ttk.Scale(slider_frame, from_=min_val, to=max_val, variable=slider_var, 
                         orient=tk.HORIZONTAL, length=180)
        slider.pack(side=tk.LEFT)
        
        # Value label with fixed width to prevent layout shifts
        value_label = ttk.Label(slider_frame, textvariable=slider_var, width=3)
        value_label.pack(side=tk.LEFT, padx=(5, 0))
        
    def create_results_panel(self, parent):
        """Create the results panel with recommendations"""
        # Title
        ttk.Label(parent, text="Food Recommendations", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create notebook (tabbed interface) for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Table View
        table_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(table_frame, text="Table View")
        
        # Create Treeview with scrollbar for recommendations
        columns = ('Name', 'Category', 'Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'Score')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Add scrollbars - both vertical and horizontal
        y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        
        # Define headings with better labels
        column_texts = {
            'Name': 'Food Name', 
            'Category': 'Category', 
            'Energy': 'Energy (kcal)', 
            'Protein': 'Protein (g)', 
            'Carbs': 'Carbs (g)',
            'Sugar': 'Sugar (g)',
            'Fiber': 'Fiber (g)',
            'Fat': 'Fat (g)',
            'Score': 'Suitability Score'
        }
        
        # Configure columns with proportional widths
        self.tree.column('Name', width=180, minwidth=120, stretch=True)
        self.tree.column('Category', width=120, minwidth=80, stretch=True)
        for col in columns[2:]:  # Numeric columns
            self.tree.column(col, width=80, minwidth=60, stretch=False)
        
        # Set headings
        for col in columns:
            self.tree.heading(col, text=column_texts[col], 
                            command=lambda c=col: self.sort_treeview(self.tree, c, False))
        
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 2: Health Analysis
        health_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(health_frame, text="Health Analysis")
        
        # Create a canvas for the health analysis chart
        self.health_canvas_frame = ttk.Frame(health_frame)
        self.health_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Nutrition Comparison
        comparison_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(comparison_frame, text="Nutrition Comparison")
        
        # Create a canvas for the comparison chart
        self.comparison_canvas_frame = ttk.Frame(comparison_frame)
        self.comparison_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Details panel below notebook with improved styling
        details_frame = ttk.LabelFrame(parent, text="Nutritional Details")
        details_frame.pack(fill=tk.X, pady=10)
        
        # Selected food details with better formatting
        self.details_text = tk.Text(details_frame, height=7, width=40, wrap=tk.WORD,
                                  font=('Arial', self.calculate_font_size(10)))
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information")
        self.details_text.config(state=tk.DISABLED)
        
        # Add tag configuration for formatting
        self.details_text.tag_configure("title", font=('Arial', self.calculate_font_size(11), 'bold'))
        self.details_text.tag_configure("subtitle", font=('Arial', self.calculate_font_size(10), 'bold'))
        self.details_text.tag_configure("good", foreground="green")
        self.details_text.tag_configure("warning", foreground="red")
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
    def sort_treeview(self, tree, col, reverse):
        """Sort treeview content when clicking on a column"""
        # Get all items in the tree
        data = [(tree.set(item, col), item) for item in tree.get_children('')]
        
        # Sort based on type of data
        try:
            # Try to sort as numbers
            data.sort(key=lambda x: float(x[0]), reverse=reverse)
        except ValueError:
            # Fall back to string sort
            data.sort(reverse=reverse)
            
        # Rearrange items in sorted positions
        for index, (_, item) in enumerate(data):
            tree.move(item, '', index)
            
        # Switch the heading to the opposite sort order
        tree.heading(col, command=lambda: self.sort_treeview(tree, col, not reverse))
        
    def create_chart_panel(self, parent):
        """Create a panel with charts/graphs"""
        # Create figure with appropriate size and improved resolution
        self.fig = Figure(figsize=(12, 3), dpi=100)
        self.fig.subplots_adjust(wspace=0.3)  # Add more space between subplots
        
        # Initialize the chart with empty data
        self.nutrition_chart = FigureCanvasTkAgg(self.fig, parent)
        self.nutrition_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create chart axes
        self.ax1 = self.fig.add_subplot(141)  # Macronutrient distribution
        self.ax2 = self.fig.add_subplot(142)  # Category distribution
        self.ax3 = self.fig.add_subplot(143)  # Health condition scores
        self.ax4 = self.fig.add_subplot(144)  # Nutritional balance
        
        # Set titles with better styling
        self.ax1.set_title('Macronutrient Distribution', fontsize=10, fontweight='bold')
        self.ax2.set_title('Food Categories', fontsize=10, fontweight='bold')
        self.ax3.set_title('Health Impact', fontsize=10, fontweight='bold')
        self.ax4.set_title('Suitable for Conditions', fontsize=10, fontweight='bold')
        
        # Update chart with empty data initially
        self.update_charts([])
        
    def update_nutrition_comparison_chart(self, recommendations):
        """Update the nutrition comparison chart in the third tab"""
        # Clear the previous chart if it exists
        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()
            
        if not recommendations or len(recommendations) < 2:
            # Show a message if not enough recommendations
            msg = ttk.Label(self.comparison_canvas_frame, 
                          text="Select at least two food items in the table view to compare", 
                          font=('Arial', 12))
            msg.pack(expand=True, pady=50)
            return
            
        # Create a Figure for the comparison chart
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for comparison - we'll compare selected items in the treeview
        selected_items = self.tree.selection()
        if not selected_items or len(selected_items) < 2:
            msg = ttk.Label(self.comparison_canvas_frame, 
                          text="Select at least two food items in the table view to compare", 
                          font=('Arial', 12))
            msg.pack(expand=True, pady=50)
            return
            
        # Get selected food items
        selected_foods = []
        for item_id in selected_items:
            item_values = self.tree.item(item_id, 'values')
            if item_values:
                food_name = item_values[0]
                for rec in recommendations:
                    if rec['food_name'] == food_name:
                        selected_foods.append(rec)
                        break
        
        # Prepare data for chart
        nutrients = ['Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat']
        
        # Position of bars on x-axis
        x = np.arange(len(nutrients))
        width = 0.8 / len(selected_foods)  # Width of bars adjusted for number of foods
        
        # Plot bars for each food
        for i, food in enumerate(selected_foods):
            values = [
                food.get('calories', 0),
                food.get('protein', 0),
                food.get('carbs', 0),
                food.get('sugar', 0),
                food.get('fiber', 0),
                food.get('fat', 0)
            ]
            pos = x - 0.4 + (i + 0.5) * width  # Position bars side by side
            ax.bar(pos, values, width=width, label=food['food_name'][:15])  # Truncate long names
        
        # Set chart properties
        ax.set_title('Nutrient Comparison of Selected Foods')
        ax.set_ylabel('Amount')
        ax.set_xticks(x)
        ax.set_xticklabels(nutrients)
        ax.legend(loc='upper right', fontsize='small')
        
        # Add units to y-axis labels
        y_labels = [f"{int(y)}" + (" kcal" if idx == 0 else " g") 
                   for idx, y in enumerate(ax.get_yticks())]
        ax.set_yticklabels(y_labels)
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, self.comparison_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add explanation text
        explanation = "This chart compares the nutritional content of selected foods. Select multiple items in the Table View to compare."
        ttk.Label(self.comparison_canvas_frame, text=explanation, 
                foreground=self.light_text_color).pack(pady=5)
        
    def update_health_analysis_chart(self, recommendations):
        """Update the health analysis chart in the second tab"""
        # Clear the previous chart if it exists
        for widget in self.health_canvas_frame.winfo_children():
            widget.destroy()
            
        if not recommendations:
            # Show a message if no recommendations
            msg = ttk.Label(self.health_canvas_frame, text="No data available for analysis", 
                          font=('Arial', 12))
            msg.pack(expand=True, pady=50)
            return
            
        # Create a Figure for the score breakdown
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for chart
        max_foods_to_display = min(10, len(recommendations))  # Limit number of foods to display
        food_names = [rec['food_name'] for rec in recommendations[:max_foods_to_display]]
        
        # Get scores for each component
        calorie_scores = [rec.get('calorie_score', 0) * 10 for rec in recommendations[:max_foods_to_display]]
        macro_scores = [rec.get('macro_score', 0) * 10 for rec in recommendations[:max_foods_to_display]]
        quality_scores = [rec.get('quality_score', 0) * 10 for rec in recommendations[:max_foods_to_display]]
        
        # Set up chart
        x = np.arange(len(food_names))
        width = 0.25
        
        # Plot bars for each score component
        calorie_bars = ax.barh(x - width, calorie_scores, width, label='Calorie Appropriateness', color='#3498db')
        macro_bars = ax.barh(x, macro_scores, width, label='Macronutrient Balance', color='#2ecc71')
        quality_bars = ax.barh(x + width, quality_scores, width, label='Nutritional Quality', color='#e74c3c')
        
        # Add food names as y-tick labels
        ax.set_yticks(x)
        ax.set_yticklabels(food_names)
        
        # Set labels and title
        ax.set_xlabel('Score (0-100)')
        ax.set_title('Food Recommendation Score Breakdown')
        ax.legend()
        
        # Set x-axis limits
        ax.set_xlim(0, 100)
        
        # Add grid lines for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, self.health_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add explanation of scoring
        explanation = (
            "Score Breakdown Explanation:\n"
            "• Calorie Appropriateness: How well the food's calories fit your needs\n"
            "• Macronutrient Balance: Balance of protein, carbs, and fats\n"
            "• Nutritional Quality: Beneficial nutrients and limited problematic ones"
        )
        ttk.Label(self.health_canvas_frame, text=explanation, 
                justify=tk.LEFT, foreground=self.text_color).pack(pady=10, anchor=tk.W)
        
    def update_charts(self, recommendations):
        """Update the charts with recommendation data"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Reset titles
        self.ax1.set_title('Macronutrient Distribution', fontsize=10, fontweight='bold')
        self.ax2.set_title('Food Categories', fontsize=10, fontweight='bold')
        self.ax3.set_title('Scoring Components', fontsize=10, fontweight='bold')
        self.ax4.set_title('Suitable for Conditions (%)', fontsize=10, fontweight='bold')
        
        if recommendations and len(recommendations) > 0:
            try:
                # 1. Macronutrient distribution chart
                labels = ['Protein', 'Carbs', 'Fat']
                values = [
                    sum(rec.get('protein', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('carbs', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('fat', 0) for rec in recommendations) / len(recommendations)
                ]
                
                # Filter out zero values for pie chart
                filtered_labels = []
                filtered_values = []
                colors = ['#ff9999','#66b3ff','#99ff99']
                
                for i, value in enumerate(values):
                    if value > 0:
                        filtered_labels.append(labels[i])
                        filtered_values.append(value)
                
                if sum(filtered_values) > 0:
                    # Format percentages with one decimal place
                    def autopct_format(values):
                        def my_format(pct):
                            total = sum(values)
                            val = int(round(pct*total/100.0))
                            return '{:.1f}%\n({:d}g)'.format(pct, val)
                        return my_format
                    
                    self.ax1.pie(filtered_values, labels=filtered_labels, 
                               colors=colors[:len(filtered_labels)], 
                               autopct=autopct_format(filtered_values), 
                               shadow=False, startangle=90)
                    self.ax1.set_title('Average Macronutrient Distribution')
                else:
                    self.ax1.text(0.5, 0.5, 'No macronutrient data available', 
                                ha='center', va='center')
                
                # 2. Category distribution chart with improved colors
                categories = {}
                for rec in recommendations:
                    cat = rec.get('category', 'Unknown')
                    if cat in categories:
                        categories[cat] += 1
                    else:
                        categories[cat] = 1
                
                if categories:
                    # Sort categories by count
                    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
                    cat_names = [item[0] for item in sorted_categories]
                    cat_counts = [item[1] for item in sorted_categories]
                    
                    # Limit to top 6 categories to prevent overcrowding
                    if len(cat_names) > 6:
                        other_count = sum(cat_counts[5:])
                        cat_names = cat_names[:5] + ['Other']
                        cat_counts = cat_counts[:5] + [other_count]
                    
                    # Use a colorful palette
                    colors = plt.cm.tab10(np.linspace(0, 1, len(cat_names)))
                    
                    # Create horizontal bar chart
                    bars = self.ax2.barh(cat_names, cat_counts, color=colors)
                    self.ax2.set_title('Food Categories')
                    self.ax2.set_xlabel('Count')
                    
                    # Add count labels to bars
                    for bar in bars:
                        width = bar.get_width()
                        self.ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                                    f'{int(width)}', ha='left', va='center', fontsize=8)
                    
                    # Set y-limit to ensure all categories are visible
                    self.ax2.set_ylim(-0.5, len(cat_names)-0.5)
                else:
                    self.ax2.text(0.5, 0.5, 'No category data available', 
                                ha='center', va='center')
                
                # 3. Scoring components chart
                score_components = ['Calorie', 'Macro', 'Quality']
                avg_scores = [
                    sum(rec.get('calorie_score', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('macro_score', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('quality_score', 0) for rec in recommendations) / len(recommendations)
                ]
                
                if any(score > 0 for score in avg_scores):
                    # Use a color palette that provides good contrast
                    bar_colors = ['#3498db', '#2ecc71', '#e74c3c']
                    
                    bars = self.ax3.bar(score_components, avg_scores, color=bar_colors)
                    self.ax3.set_title('Average Score Components')
                    self.ax3.set_ylabel('Score (0-10)')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        self.ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
                    # Set y-limit with some padding for the labels
                    self.ax3.set_ylim(0, 12)
                    
                    # Add a subtle horizontal grid for better readability
                    self.ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
                    
                else:
                    self.ax3.text(0.5, 0.5, 'No score data available', 
                                ha='center', va='center')
                
                # 4. Suitable conditions chart with improved visualization
                suitable_counts = {
                    'Diabetes': 0,
                    'Obesity': 0,
                    'Hypertension': 0,
                    'High_Cholesterol': 0
                }
                
                for rec in recommendations:
                    for condition in rec.get('suitable_for', []):
                        suitable_counts[condition] = suitable_counts.get(condition, 0) + 1
                
                # Calculate percentage suitable for each condition
                suitable_percentages = {}
                total_items = len(recommendations)
                for condition, count in suitable_counts.items():
                    suitable_percentages[condition] = (count / total_items) * 100 if total_items > 0 else 0
                
                if suitable_counts and any(suitable_counts.values()):
                    labels = [condition.replace('_', ' ') for condition in suitable_counts.keys()]
                    values = list(suitable_percentages.values())
                    
                    # Use a distinct color palette
                    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
                    
                    bars = self.ax4.bar(labels, values, color=colors)
                    self.ax4.set_title('Suitable for Conditions (%)')
                    self.ax4.set_ylabel('Percentage of Foods')
                    self.ax4.set_ylim(0, 100)
                    
                    # Add percentage labels
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:  # Only show label if value is non-zero
                            self.ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                                       f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
                    
                    # Add grid lines for easier reading
                    self.ax4.yaxis.grid(True, linestyle='--', alpha=0.3)
                    
                    # Rotate labels for better display
                    plt.setp(self.ax4.get_xticklabels(), rotation=30, ha='right')
                else:
                    self.ax4.text(0.5, 0.5, 'No suitability data available', 
                                ha='center', va='center')
                
            except Exception as e:
                print(f"Error updating charts: {e}")
                self.ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                self.ax2.text(0.5, 0.5, 'Chart error', ha='center', va='center')
                self.ax3.text(0.5, 0.5, 'Chart error', ha='center', va='center')
                self.ax4.text(0.5, 0.5, 'Chart error', ha='center', va='center')
        else:
            # Show placeholder text with better styling
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.text(0.5, 0.5, 'No data to display\nSelect health conditions and click "Get Recommendations"',
                      ha='center', va='center', fontsize=9, color=self.light_text_color,
                      multialignment='center')
        
        # Adjust layout and update canvas
        self.fig.tight_layout()
        self.nutrition_chart.draw()
        
    def show_food_details(self, event):
        """Show details for selected food item"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        # Get the first selected item (for detailed view)
        item = selected_items[0]
        values = self.tree.item(item, 'values')
        
        # Update details text
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        # Get food name from the first column
        food_name = values[0]
        
        # Get more detailed information if available
        food_details = self.recommender.get_food_details(food_name)
        
        if food_details:
            # Format a more comprehensive details text with tags for styling
            self.details_text.insert(tk.END, f"{food_name}\n", "title")
            self.details_text.insert(tk.END, f"Category: {values[1]}\n\n")
            
            # Nutritional content section
            self.details_text.insert(tk.END, "Nutritional Content:\n", "subtitle")
            
            # Format each nutrient with appropriate styling
            energy = float(values[2])
            self.details_text.insert(tk.END, f"• Energy: {energy:.0f} kcal")
            if energy > 300:
                self.details_text.insert(tk.END, " (high)\n", "warning")
            elif energy < 100:
                self.details_text.insert(tk.END, " (low)\n", "good")
            else:
                self.details_text.insert(tk.END, "\n")
                
            protein = float(values[3])
            self.details_text.insert(tk.END, f"• Protein: {protein:.1f}g")
            if protein >= 10:
                self.details_text.insert(tk.END, " (good source)\n", "good")
            else:
                self.details_text.insert(tk.END, "\n")
                
            carbs = float(values[4])
            self.details_text.insert(tk.END, f"• Carbohydrates: {carbs:.1f}g\n")
            
            sugar = float(values[5])
            self.details_text.insert(tk.END, f"• Sugar: {sugar:.1f}g")
            if sugar > 15:
                self.details_text.insert(tk.END, " (high)\n", "warning")
            elif sugar <= 5:
                self.details_text.insert(tk.END, " (low)\n", "good")
            else:
                self.details_text.insert(tk.END, "\n")
                
            fiber = float(values[6])
            self.details_text.insert(tk.END, f"• Fiber: {fiber:.1f}g")
            if fiber >= 5:
                self.details_text.insert(tk.END, " (high)\n", "good")
            elif fiber >= 3:
                self.details_text.insert(tk.END, " (good source)\n", "good")
            else:
                self.details_text.insert(tk.END, "\n")
                
            fat = float(values[7])
            self.details_text.insert(tk.END, f"• Fat: {fat:.1f}g\n")
            
            # Add sodium information if available
            if 'Na(mg)' in food_details and not pd.isna(food_details['Na(mg)']):
                sodium = float(food_details['Na(mg)'])
                self.details_text.insert(tk.END, f"\nSodium: {sodium:.0f} mg")
                if sodium > 400:
                    self.details_text.insert(tk.END, " (high)\n", "warning")
                elif sodium <= 140:
                    self.details_text.insert(tk.END, " (low)\n", "good")
                else:
                    self.details_text.insert(tk.END, "\n")
            
            # Add potassium information if available
            if 'K(mg)' in food_details and not pd.isna(food_details['K(mg)']):
                potassium = float(food_details['K(mg)'])
                self.details_text.insert(tk.END, f"Potassium: {potassium:.0f} mg")
                if potassium > 300:
                    self.details_text.insert(tk.END, " (good source)\n", "good")
                else:
                    self.details_text.insert(tk.END, "\n")
                    
            # Add cholesterol information if available
            if 'CHOLE(mg) Cholesterol' in food_details and not pd.isna(food_details['CHOLE(mg) Cholesterol']):
                cholesterol = float(food_details['CHOLE(mg) Cholesterol'])
                self.details_text.insert(tk.END, f"Cholesterol: {cholesterol:.0f} mg")
                if cholesterol > 100:
                    self.details_text.insert(tk.END, " (high)\n", "warning")
                elif cholesterol <= 20:
                    self.details_text.insert(tk.END, " (low)\n", "good")
                else:
                    self.details_text.insert(tk.END, "\n")
                    
        else:
            # Basic information from the table with improved formatting
            self.details_text.insert(tk.END, f"{food_name}\n", "title")
            self.details_text.insert(tk.END, f"Category: {values[1]}\n\n")
            
            self.details_text.insert(tk.END, "Nutritional Content:\n", "subtitle")
            self.details_text.insert(tk.END, f"• Energy: {values[2]} kcal\n")
            self.details_text.insert(tk.END, f"• Protein: {values[3]} g\n")
            self.details_text.insert(tk.END, f"• Carbs: {values[4]} g\n")
            self.details_text.insert(tk.END, f"• Sugar: {values[5]} g\n")
            self.details_text.insert(tk.END, f"• Fiber: {values[6]} g\n")
            self.details_text.insert(tk.END, f"• Fat: {values[7]} g\n")
            self.details_text.insert(tk.END, f"• Suitability Score: {values[8]}")
        
        self.details_text.config(state=tk.DISABLED)
        
        # If multiple items are selected, update the comparison chart
        if len(selected_items) >= 2:
            self.notebook.select(2)  # Switch to the comparison tab
            self.update_nutrition_comparison_chart(self.last_recommendations)
        
    def show_recommendations(self):
        """Show food recommendations based on user preferences and health profile"""
        # Get user preferences
        user_prefs = {
            'Energy(kcal) by calculation': self.calories_var.get(),
            'Protein(g)': self.protein_var.get(),
            'CHOCDF (g) Carbohydrate': self.carbs_var.get(),
            'SUGAR(g)': self.sugar_var.get(),
            'FIBTG (g) Dietary fibre': self.fiber_var.get(),
            'Fat(g)': self.fat_var.get()
        }
        
        # Create health profile
        health_profile = {
            'weight': self.weight_var.get(),
            'height': self.height_var.get(),
            'age': self.age_var.get(),
            'gender': self.gender_var.get(),
            'activity_level': self.activity_var.get(),
            'meal_type': self.meal_type_var.get().lower(),
            'diabetes': self.diabetes_var.get(),
            'obesity': self.obesity_var.get(),
            'hypertension': self.hypertension_var.get(),
            'high_cholesterol': self.cholesterol_var.get()
        }
        
        # Show loading in progress bar
        self.status_var.set("Finding recommendations...")
        self.progress["value"] = 10
        self.master.update_idletasks()
        
        # Get category filter
        category_filter = self.category_filter_var.get()
        num_recommendations = self.num_recommendations_var.get()
        
        # Get recommendations
        self.progress["value"] = 30
        self.master.update_idletasks()
        
        try:
            # Check if recommender has the get_recommendations method
            if hasattr(self.recommender, 'get_recommendations'):
                recommendations = self.recommender.get_recommendations(
                    user_prefs,
                    health_profile=health_profile,
                    category_filter=category_filter,
                    max_recommendations=num_recommendations
                )
                
                # Store recommendations for comparison chart
                self.last_recommendations = recommendations
            else:
                messagebox.showerror("Error", "Recommendation system not properly initialized")
                self.progress["value"] = 0
                return
                
            self.progress["value"] = 60
            self.master.update_idletasks()
            
            # Clear previous recommendations
            for i in self.tree.get_children():
                self.tree.delete(i)
                
            # Display recommendations
            if recommendations:
                for rec in recommendations:
                    # Get the suitability score
                    suitability_score = rec.get('score', 0)
                    suitability_score_str = f"{suitability_score:.1f}"
                    
                    # Insert into treeview
                    item_id = self.tree.insert('', 'end', values=(
                        rec.get('food_name', 'Unknown'),
                        rec.get('category', 'Unknown'),
                        rec.get('calories', 0),
                        rec.get('protein', 0),
                        rec.get('carbs', 0),
                        rec.get('sugar', 0),
                        rec.get('fiber', 0),
                        rec.get('fat', 0),
                        suitability_score_str
                    ))
                    
                # Selection for the first item to display details
                if self.tree.get_children():
                    first_item = self.tree.get_children()[0]
                    self.tree.selection_set(first_item)
                    self.tree.focus(first_item)
                    self.show_food_details(None)  # Show details for the first item
                
                # Automatic width adjustment for columns
                for col in self.tree['columns']:
                    self.tree.column(col, width=tk.font.Font().measure(col) + 20)
                
                # Update status
                health_profile_text = []
                if health_profile['diabetes']:
                    health_profile_text.append("Diabetes")
                if health_profile['obesity']:
                    health_profile_text.append("Weight Management")
                if health_profile['hypertension']:
                    health_profile_text.append("Hypertension")
                if health_profile['high_cholesterol']:
                    health_profile_text.append("High Cholesterol")
                
                profile_text = ", ".join(health_profile_text) if health_profile_text else "general health profile"
                self.status_var.set(f"Found {len(recommendations)} recommendations for {profile_text}")
                
                # Update charts with new data
                self.progress["value"] = 80
                self.master.update_idletasks()
                self.update_charts(recommendations)
                
                # Update health analysis chart in second tab
                self.update_health_analysis_chart(recommendations)
                
                # Clear the comparison chart initially
                for widget in self.comparison_canvas_frame.winfo_children():
                    widget.destroy()
                msg = ttk.Label(self.comparison_canvas_frame, 
                              text="Select multiple food items in the Table View to compare", 
                              font=('Arial', 12))
                msg.pack(expand=True, pady=50)
            else:
                messagebox.showinfo("No Recommendations", 
                                   "No recommendations found matching your criteria.")
                self.status_var.set("No recommendations found")
                
                # Clear charts
                self.update_charts([])
                self.update_health_analysis_chart([])
                
                # Clear comparison chart
                for widget in self.comparison_canvas_frame.winfo_children():
                    widget.destroy()
                msg = ttk.Label(self.comparison_canvas_frame, 
                              text="No data available for comparison", 
                              font=('Arial', 12))
                msg.pack(expand=True, pady=50)
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Error getting recommendations: {e}")
            
        # Reset progress bar
        self.progress["value"] = 100
        self.master.update_idletasks()
        self.master.after(500, lambda: self.progress.configure(value=0))
    
    def reset_preferences(self):
        """Reset preferences to default values"""
        # Reset health conditions
        self.diabetes_var.set(False)
        self.obesity_var.set(False)
        self.hypertension_var.set(False)
        self.cholesterol_var.set(False)
        
        # Reset nutrition sliders
        self.calories_var.set(500)
        self.protein_var.set(20)
        self.carbs_var.set(30)
        self.sugar_var.set(5)
        self.fiber_var.set(8)
        self.fat_var.set(15)
        
        # Reset other settings
        self.weight_var.set(70.0)
        self.height_var.set(170.0)
        self.age_var.set(45)
        self.gender_var.set("Male")
        self.activity_var.set("Moderate")
        self.meal_type_var.set("Meal")
        self.num_recommendations_var.set(10)
        self.category_filter_var.set("All")
        
        # Recalculate BMI
        self.calculate_bmi()
        
        # Clear previous recommendations
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Reset details text
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information")
        self.details_text.config(state=tk.DISABLED)
        
        # Reset charts
        self.update_charts([])
        self.update_health_analysis_chart([])
        
        # Clear comparison chart
        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()
        msg = ttk.Label(self.comparison_canvas_frame, 
                      text="Select multiple food items in the Table View to compare", 
                      font=('Arial', 12))
        msg.pack(expand=True, pady=50)
        
        # Update status
        self.status_var.set("All settings reset to defaults")
    
    def update_stats_display(self):
        """Update the statistics display in the bottom right corner"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Create stats labels with better formatting
            total_items = stats['total_items']
            num_categories = len(stats['categories'])
            
            stats_text = f"{total_items} food items | {num_categories} categories"
            
            # Create a frame with better layout
            stats_label = ttk.Label(self.stats_frame, 
                                  text=stats_text,
                                  style='Stats.TLabel')
            stats_label.pack(side=tk.LEFT)
            
            # Add loading time if available
            if 'loading_time' in stats and stats['loading_time'] > 0:
                time_label = ttk.Label(self.stats_frame, 
                                     text=f" | Loaded in {stats['loading_time']:.2f}s", 
                                     style='Stats.TLabel')
                time_label.pack(side=tk.LEFT)
                
        except Exception as e:
            print(f"Error updating stats display: {e}")
            ttk.Label(self.stats_frame, text="Stats unavailable", style='Stats.TLabel').pack(side=tk.LEFT)


# Main function to run the application
def main():
    # Set up basic logging to console
    print("Starting Personalized Food Recommendation System...")
    
    # Create the root window first
    root = tk.Tk()
    root.title("Personalized Food Recommendation System")
    root.geometry("1200x750")
    
    # Set application icon if available
    try:
        # You can add an icon file to your project and use it here
        #root.iconbitmap("icon.ico")
        pass
    except:
        pass
    
    # Create a modern splash screen
    splash_frame = ttk.Frame(root)
    splash_frame.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Add a background color
    splash_bg = tk.Canvas(splash_frame, bg="#3498db", highlightthickness=0)
    splash_bg.place(relwidth=1, relheight=1)
    
    # Create content frame with white background for contrast
    content_frame = ttk.Frame(splash_frame, padding=20)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=500, height=400)
    
    # White background for content area
    content_bg = tk.Canvas(content_frame, bg="white", highlightthickness=0)
    content_bg.place(relwidth=1, relheight=1)
    
    # Create splash screen elements with better styling
    title_font = ('Arial', 22, 'bold')
    title_label = tk.Label(content_frame, text="Personalized Food Recommendation System",
                         font=title_font, bg="white", fg="#2c3e50")
    title_label.pack(pady=(40, 10))
    
    subtitle_font = ('Arial', 14)
    subtitle_label = tk.Label(content_frame, 
                           text="Using Health and Biometric Data Analysis",
                           font=subtitle_font, bg="white", fg="#7f8c8d")
    subtitle_label.pack(pady=(0, 30))
    
    # Custom progress bar with rounded corners
    progress_frame = ttk.Frame(content_frame)
    progress_frame.pack(pady=20)
    
    progress = ttk.Progressbar(progress_frame, orient="horizontal", 
                             length=350, mode="indeterminate", 
                             style="Splash.Horizontal.TProgressbar")
    progress.pack()
    
    # Configure progress bar style
    style = ttk.Style()
    style.configure("Splash.Horizontal.TProgressbar", 
                   background="#2ecc71",
                   troughcolor="#ecf0f1",
                   thickness=10)
    
    progress.start()
    
    # Status text
    status_var = tk.StringVar(value="Initializing system...")
    status_label = tk.Label(content_frame, textvariable=status_var, 
                          font=('Arial', 11), bg="white", fg="#7f8c8d")
    status_label.pack(pady=20)
    
    # Information text about the system
    info_text = "Powered by Random Forest Algorithm"
    info_label = tk.Label(content_frame, text=info_text,
                        font=('Arial', 10), bg="white", fg="#2980b9")
    info_label.pack(pady=10)
    
    # Add a footer with credits
    footer_text = "Prince of Songkla University, Surat Thani Campus"
    footer_label = tk.Label(content_frame, text=footer_text, 
                         font=('Arial', 8), bg="white", fg="#95a5a6")
    footer_label.pack(side=tk.BOTTOM, pady=20)
    
    def update_status(message):
        """Update status message and ensure UI updates"""
        print(f"Status: {message}")  # Debug print
        status_var.set(message)
        root.update_idletasks()  # Force update of the UI
    
    def initialize_app():
        """Initialize the application in a background thread"""
        try:
            # Create the food recommender
            update_status("Loading food database...")
            recommender = FoodRecommendationSystem(update_status)
            
            update_status("Training Random Forest model...")
            time.sleep(0.5)  # Small delay for visual effect
            
            update_status("Building user interface...")
            # Create main app UI
            app = FoodRecommenderUI(root, recommender)
            
            # Remove splash screen
            update_status("Ready! System loaded successfully.")
            root.after(1500, splash_frame.destroy)  # Destroy splash frame after delay
            
        except Exception as e:
            # Show error on splash screen
            update_status(f"Error initializing: {str(e)}")
            print(f"Initialization error: {e}")  # Debug print
            
            # Add a retry button
            retry_btn = ttk.Button(content_frame, text="Retry", 
                                  command=lambda: initialize_app())
            retry_btn.pack(pady=10)
    
    # Schedule the initialization to happen after the window is shown
    root.after(100, initialize_app)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Make window resizable with minimum size
    root.minsize(900, 600)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()