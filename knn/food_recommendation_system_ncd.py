import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox, font
import time
import threading
import os
import glob
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class FoodRecommendationSystem:
    def __init__(self, status_callback=None):
        # Callback function to update loading status
        self.status_callback = status_callback
        
        # Set default stats
        self.stats = {
            'total_items': 0,
            'categories': {},
            'condition_friendly': {
                'Diabetes': 0,
                'Obesity': 0,
                'Hypertension': 0,
                'High_Cholesterol': 0
            },
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
            'FASAT (g) Saturated FA',
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
        
        # Features for condition-specific analysis
        self.condition_features = {
            'Diabetes': ['Diabetes', 'SUGAR(g)', 'CHOCDF (g) Carbohydrate', 'FIBTG (g) Dietary fibre'],
            'Obesity': ['Obesity', 'Energy(kcal) by calculation', 'Fat(g)', 'FIBTG (g) Dietary fibre'],
            'Hypertension': ['Hypertension', 'Na(mg)', 'K(mg)', 'Fat(g)'],
            'High_Cholesterol': ['High Cholesterol', 'Fat(g)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol', 'FIBTG (g) Dietary fibre']
        }
        
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
        
        # Prepare the data for K-NN
        self.prepare_knn_data()
        
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
                    
                    # Count condition-friendly items
                    for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol']:
                        if condition in df.columns:
                            # Assuming values of 0 or 1 mean recommended
                            friendly_count = df[df[condition] <= 1].shape[0]
                            condition_key = condition.replace(' ', '_')
                            self.stats['condition_friendly'][condition_key] = \
                                self.stats['condition_friendly'].get(condition_key, 0) + friendly_count
                    
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
    
    def prepare_knn_data(self):
        """Prepare data for K-NN algorithm"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for K-NN model")
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
            
        # Extract features for K-NN
        X = self.food_data[self.features].fillna(0)
        
        # Standardize features (important for K-NN)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Initialize K-NN model (using 15 neighbors for more diverse recommendations)
        n_neighbors = min(15, len(X))
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        self.knn_model.fit(self.X_scaled)
        
        self.update_status(f"K-NN model prepared with {len(self.features)} features and {n_neighbors} neighbors")
    
    def calculate_condition_score(self, food_item, condition):
        """Calculate the suitability score for a specific health condition based on scientific guidelines"""
        score = 0
        
        if condition == 'Diabetes':
            # Direct condition rating if available
            if 'Diabetes' in self.food_data.columns and not pd.isna(food_item['Diabetes']):
                score += float(food_item['Diabetes']) * 2  # Weight the direct rating more heavily
            
            # Sugar content (ADA/WHO recommends limiting to 25g/day or less for added sugar)
            if 'SUGAR(g)' in self.food_data.columns and not pd.isna(food_item['SUGAR(g)']):
                sugar = float(food_item['SUGAR(g)'])
                # Higher penalty as sugar content approaches or exceeds daily limit
                if sugar <= 5:
                    score += sugar * 0.1  # Minimal impact for low sugar
                elif sugar <= 15:
                    score += 0.5 + (sugar - 5) * 0.3  # Moderate impact
                else:
                    score += 3.5 + (sugar - 15) * 0.5  # Significant impact for high sugar
            
            # Carbohydrate content (focus on quality and moderation)
            if 'CHOCDF (g) Carbohydrate' in self.food_data.columns and not pd.isna(food_item['CHOCDF (g) Carbohydrate']):
                carbs = float(food_item['CHOCDF (g) Carbohydrate'])
                # Progressive scoring based on carb content
                if carbs <= 15:  # Approximately one serving
                    score += carbs * 0.05
                elif carbs <= 30:
                    score += 0.75 + (carbs - 15) * 0.1
                else:
                    score += 2.25 + (carbs - 30) * 0.15
            
            # Fiber content (ADA recommends 14g/1000kcal; higher fiber is better)
            if 'FIBTG (g) Dietary fibre' in self.food_data.columns and not pd.isna(food_item['FIBTG (g) Dietary fibre']):
                fiber = float(food_item['FIBTG (g) Dietary fibre'])
                # Calculate energy if available
                energy = float(food_item.get('Energy(kcal) by calculation', 0))
                if energy > 0:
                    # Calculate fiber per 1000 kcal
                    fiber_ratio = (fiber / energy) * 1000
                    if fiber_ratio >= 14:  # Meets ADA recommendation
                        score -= min(fiber, 10) * 0.4  # Cap benefit at 10g fiber
                    else:
                        score -= min(fiber, 10) * 0.2  # Less benefit for lower fiber ratio
                else:
                    score -= min(fiber, 10) * 0.3  # Default benefit if energy unknown
            
            # Fat quality (prioritize unsaturated fats)
            if 'Fat(g)' in self.food_data.columns and not pd.isna(food_item['Fat(g)']):
                fat = float(food_item['Fat(g)'])
                # Check for saturated fat if available
                if 'FASAT (g) Saturated FA' in self.food_data.columns and not pd.isna(food_item['FASAT (g) Saturated FA']):
                    sat_fat = float(food_item['FASAT (g) Saturated FA'])
                    # Calculate percentage of saturated fat
                    if fat > 0:
                        sat_fat_percent = (sat_fat / fat) * 100
                        if sat_fat_percent > 30:  # High saturated fat ratio
                            score += sat_fat * 0.4
                        else:
                            score += sat_fat * 0.2
                    else:
                        score += sat_fat * 0.3
                else:
                    # If saturated fat info unavailable, use total fat as proxy
                    score += fat * 0.1
        
        elif condition == 'Obesity':
            # Direct condition rating if available
            if 'Obesity' in self.food_data.columns and not pd.isna(food_item['Obesity']):
                score += float(food_item['Obesity']) * 2
            
            # Energy density (primary factor for weight management)
            if 'Energy(kcal) by calculation' in self.food_data.columns and not pd.isna(food_item['Energy(kcal) by calculation']):
                energy = float(food_item['Energy(kcal) by calculation'])
                # Progressive scoring based on calorie content
                if energy <= 100:  # Low calorie
                    score += energy * 0.005
                elif energy <= 300:  # Moderate calorie
                    score += 0.5 + (energy - 100) * 0.01
                else:  # High calorie
                    score += 2.5 + (energy - 300) * 0.015
            
            # Fat content (20-35% of calories should come from fat)
            if 'Fat(g)' in self.food_data.columns and not pd.isna(food_item['Fat(g)']):
                fat = float(food_item['Fat(g)'])
                # Higher penalty for high fat foods
                score += fat * 0.2
            
            # Added sugar (should be <10% of total calories)
            if 'SUGAR(g)' in self.food_data.columns and not pd.isna(food_item['SUGAR(g)']):
                sugar = float(food_item['SUGAR(g)'])
                score += sugar * 0.3  # Higher penalty for sugar in obesity
            
            # Protein content (higher protein may aid weight management)
            if 'Protein(g)' in self.food_data.columns and not pd.isna(food_item['Protein(g)']):
                protein = float(food_item['Protein(g)'])
                # Benefit for protein content
                score -= min(protein, 30) * 0.15  # Cap benefit at 30g
            
            # Fiber content (promotes satiety)
            if 'FIBTG (g) Dietary fibre' in self.food_data.columns and not pd.isna(food_item['FIBTG (g) Dietary fibre']):
                fiber = float(food_item['FIBTG (g) Dietary fibre'])
                # Greater benefit for fiber in obesity management
                score -= min(fiber, 10) * 0.4  # Cap benefit at 10g
        
        elif condition == 'Hypertension':
            # Direct condition rating if available
            if 'Hypertension' in self.food_data.columns and not pd.isna(food_item['Hypertension']):
                score += float(food_item['Hypertension']) * 2
            
            # Sodium content (DASH diet recommends ≤2,300mg/day, ideally 1,500mg)
            if 'Na(mg)' in self.food_data.columns and not pd.isna(food_item['Na(mg)']):
                sodium = float(food_item['Na(mg)'])
                # Progressive scoring based on sodium content
                if sodium <= 140:  # Low sodium (FDA definition)
                    score += sodium * 0.002
                elif sodium <= 400:  # Moderate sodium
                    score += 0.28 + (sodium - 140) * 0.005
                else:  # High sodium
                    score += 1.58 + (sodium - 400) * 0.008
            
            # Potassium content (beneficial for blood pressure)
            if 'K(mg)' in self.food_data.columns and not pd.isna(food_item['K(mg)']):
                potassium = float(food_item['K(mg)'])
                # Benefit for potassium content
                score -= min(potassium, 1000) * 0.002  # Cap benefit at 1000mg
            
            # Saturated fat (DASH recommends ≤6% of calories)
            if 'FASAT (g) Saturated FA' in self.food_data.columns and not pd.isna(food_item['FASAT (g) Saturated FA']):
                sat_fat = float(food_item['FASAT (g) Saturated FA'])
                score += sat_fat * 0.4
            elif 'Fat(g)' in self.food_data.columns and not pd.isna(food_item['Fat(g)']):
                # If specific saturated fat info unavailable, use total fat as proxy
                fat = float(food_item['Fat(g)'])
                score += fat * 0.2
            
            # Fiber content (DASH recommends ≥30g/day)
            if 'FIBTG (g) Dietary fibre' in self.food_data.columns and not pd.isna(food_item['FIBTG (g) Dietary fibre']):
                fiber = float(food_item['FIBTG (g) Dietary fibre'])
                # Benefit for fiber content
                score -= min(fiber, 10) * 0.3  # Cap benefit at 10g
            
            # Calcium content (DASH emphasizes low-fat dairy)
            if 'Ca(mg)' in self.food_data.columns and not pd.isna(food_item['Ca(mg)']):
                calcium = float(food_item['Ca(mg)'])
                # Benefit for calcium content
                score -= min(calcium, 500) * 0.001  # Cap benefit at 500mg
        
        elif condition == 'High_Cholesterol':
            # Direct condition rating if available
            if 'High Cholesterol' in self.food_data.columns and not pd.isna(food_item['High Cholesterol']):
                score += float(food_item['High Cholesterol']) * 2
            
            # Saturated fat (should be <8% of daily calories)
            if 'FASAT (g) Saturated FA' in self.food_data.columns and not pd.isna(food_item['FASAT (g) Saturated FA']):
                sat_fat = float(food_item['FASAT (g) Saturated FA'])
                # Higher penalty for saturated fat
                score += sat_fat * 0.5
            
            # Total fat (should be <28% of daily calories)
            if 'Fat(g)' in self.food_data.columns and not pd.isna(food_item['Fat(g)']):
                fat = float(food_item['Fat(g)'])
                # Moderate penalty for total fat
                score += fat * 0.2
            
            # Dietary cholesterol (<150-200mg per day)
            if 'CHOLE(mg) Cholesterol' in self.food_data.columns and not pd.isna(food_item['CHOLE(mg) Cholesterol']):
                cholesterol = float(food_item['CHOLE(mg) Cholesterol'])
                # Progressive scoring based on cholesterol content
                if cholesterol <= 20:  # Very low
                    score += cholesterol * 0.01
                elif cholesterol <= 100:  # Moderate
                    score += 0.2 + (cholesterol - 20) * 0.02
                else:  # High
                    score += 1.8 + (cholesterol - 100) * 0.04
            
            # Fiber content (20-35g per day, especially soluble fiber)
            if 'FIBTG (g) Dietary fibre' in self.food_data.columns and not pd.isna(food_item['FIBTG (g) Dietary fibre']):
                fiber = float(food_item['FIBTG (g) Dietary fibre'])
                # Greater benefit for fiber in cholesterol management
                score -= min(fiber, 10) * 0.4  # Cap benefit at 10g
            
            # Added sugar (should be limited)
            if 'SUGAR(g)' in self.food_data.columns and not pd.isna(food_item['SUGAR(g)']):
                sugar = float(food_item['SUGAR(g)'])
                score += sugar * 0.15
        
        return max(0, score)  # Ensure score is non-negative
    
    def analyze_nutritional_profile(self, food_item):
        """Analyze a food item's nutritional profile and generate health recommendations"""
        if not isinstance(food_item, dict) and not hasattr(food_item, 'to_dict'):
            return {"error": "Invalid food item format"}
        
        # Convert to dictionary if it's a pandas Series
        if hasattr(food_item, 'to_dict'):
            food_item = food_item.to_dict()
        
        # Extract key nutritional values with safe defaults
        analysis = {
            "name": food_item.get('Thai_Name', food_item.get('English_Name', 'Unknown Food')),
            "category": food_item.get('Category', 'Uncategorized'),
            "energy_kcal": float(food_item.get('Energy(kcal) by calculation', 0)),
            "protein_g": float(food_item.get('Protein(g)', 0)),
            "carbs_g": float(food_item.get('CHOCDF (g) Carbohydrate', 0)),
            "sugar_g": float(food_item.get('SUGAR(g)', 0)),
            "fiber_g": float(food_item.get('FIBTG (g) Dietary fibre', 0)),
            "fat_g": float(food_item.get('Fat(g)', 0)),
            "sat_fat_g": float(food_item.get('FASAT (g) Saturated FA', 0)),
            "sodium_mg": float(food_item.get('Na(mg)', 0)),
            "potassium_mg": float(food_item.get('K(mg)', 0)),
            "cholesterol_mg": float(food_item.get('CHOLE(mg) Cholesterol', 0)),
            "condition_scores": {},
            "suitable_for": [],
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
        
        # Analyze suitability for different health conditions
        
        # Diabetes analysis
        diabetes_score = 0
        
        # Sugar analysis
        if analysis["sugar_g"] > 25:  # Above AHA/WHO daily limit
            diabetes_score += 3
            analysis["warnings"].append("High sugar content exceeds daily recommended limit for diabetes")
        elif analysis["sugar_g"] > 15:
            diabetes_score += 2
            analysis["warnings"].append("Moderately high sugar content")
        elif analysis["sugar_g"] > 5:
            diabetes_score += 1
        
        # Carbohydrate quality
        if analysis["carbs_g"] > 30 and analysis["fiber_g"] < 3:
            diabetes_score += 2
            analysis["warnings"].append("High carbohydrate with low fiber content")
        
        # Fiber content evaluation
        if analysis["fiber_per_1000kcal"] >= 14:  # Meets ADA recommendation
            diabetes_score -= 2
        
        # Overall assessment
        if diabetes_score <= 0 and analysis["fiber_g"] >= 3:
            analysis["suitable_for"].append("Diabetes")
        
        analysis["condition_scores"]["Diabetes"] = max(0, diabetes_score)
        
        # Obesity analysis
        obesity_score = 0
        
        # Energy density
        if analysis["energy_kcal"] > 300:
            obesity_score += 3
            analysis["warnings"].append("High calorie content")
        elif analysis["energy_kcal"] > 200:
            obesity_score += 2
        elif analysis["energy_kcal"] > 100:
            obesity_score += 1
        
        # Fat evaluation
        if analysis["fat_percent"] > 35:  # Above recommended range
            obesity_score += 2
            analysis["warnings"].append("High fat percentage")
        
        # Protein and fiber benefit
        if analysis["protein_g"] >= 10 and analysis["fiber_g"] >= 3:
            obesity_score -= 2  # Promotes satiety
        
        # Overall assessment
        if obesity_score <= 1 and analysis["energy_kcal"] < 200 and analysis["fiber_g"] >= 3:
            analysis["suitable_for"].append("Obesity")
        
        analysis["condition_scores"]["Obesity"] = max(0, obesity_score)
        
        # Hypertension analysis
        hypertension_score = 0
        
        # Sodium content (key factor)
        if analysis["sodium_mg"] > 400:  # High sodium
            hypertension_score += 3
            analysis["warnings"].append("High sodium content")
        elif analysis["sodium_mg"] > 140:  # Moderate sodium
            hypertension_score += 1
        
        # Potassium benefit
        if analysis["potassium_mg"] > 300:
            hypertension_score -= 1
        
        # Fat quality
        if analysis["sat_fat_percent"] > 30:
            hypertension_score += 1
            analysis["warnings"].append("High saturated fat percentage")
        
        # Overall assessment
        if hypertension_score <= 1 and analysis["sodium_mg"] < 140:  # Low sodium
            analysis["suitable_for"].append("Hypertension")
        
        analysis["condition_scores"]["Hypertension"] = max(0, hypertension_score)
        
        # High Cholesterol analysis
        cholesterol_score = 0
        
        # Saturated fat (primary concern)
        if analysis["sat_fat_g"] > 5:
            cholesterol_score += 3
            analysis["warnings"].append("High saturated fat content")
        elif analysis["sat_fat_g"] > 2:
            cholesterol_score += 2
        
        # Dietary cholesterol
        if analysis["cholesterol_mg"] > 100:
            cholesterol_score += 2
            analysis["warnings"].append("High cholesterol content")
        elif analysis["cholesterol_mg"] > 20:
            cholesterol_score += 1
        
        # Fiber benefit
        if analysis["fiber_g"] >= 5:
            cholesterol_score -= 2
        elif analysis["fiber_g"] >= 3:
            cholesterol_score -= 1
        
        # Overall assessment
        if cholesterol_score <= 1 and analysis["sat_fat_g"] < 2 and analysis["fiber_g"] >= 3:
            analysis["suitable_for"].append("High_Cholesterol")
        
        analysis["condition_scores"]["High_Cholesterol"] = max(0, cholesterol_score)
        
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
        
        if analysis["sodium_mg"] >= 400:
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
        
        # Diabetes recommendations
        if "Diabetes" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with diabetes due to its balanced nutritional profile.")
        elif analysis["condition_scores"]["Diabetes"] >= 3:
            if analysis["sugar_g"] > 15:
                recommendations.append("People with diabetes should limit consumption due to high sugar content.")
            elif analysis["carbs_g"] > 30 and analysis["fiber_g"] < 3:
                recommendations.append("People with diabetes should consume in moderation due to high carb and low fiber content.")
        
        # Obesity recommendations
        if "Obesity" in analysis["suitable_for"]:
            recommendations.append("Suitable for weight management due to its lower calorie profile.")
        elif analysis["condition_scores"]["Obesity"] >= 3:
            recommendations.append("Those managing their weight should limit portion size due to high calorie content.")
        
        # Hypertension recommendations
        if "Hypertension" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with hypertension due to its lower sodium content.")
        elif analysis["condition_scores"]["Hypertension"] >= 3:
            recommendations.append("People with hypertension should limit consumption due to high sodium content.")
        
        # High Cholesterol recommendations
        if "High_Cholesterol" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with high cholesterol due to its heart-healthy profile.")
        elif analysis["condition_scores"]["High_Cholesterol"] >= 3:
            recommendations.append("People with high cholesterol should limit consumption due to saturated fat/cholesterol content.")
        
        # General recommendations
        if not recommendations:
            if sum(analysis["condition_scores"].values()) <= 4:
                recommendations.append("Generally acceptable for most diets in moderation.")
            else:
                recommendations.append("Best consumed occasionally as part of a varied and balanced diet.")
        
        return recommendations
    
    def get_recommendations(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """Get food recommendations based on user preferences and health conditions"""
        if not hasattr(self, 'knn_model') or len(self.food_data) == 0:
            return []
        
        # Default to empty list if conditions is None
        if conditions is None:
            conditions = []
            
        # Extract user preferences for our features
        user_values = [user_preferences.get(feature, 0) for feature in self.features]
        
        # Scale user preferences
        user_scaled = self.scaler.transform([user_values])
        
        # Find K-nearest neighbors
        distances, indices = self.knn_model.kneighbors(user_scaled)
        
        # Get recommended food items
        recommendations = []
        for idx in indices[0]:
            food_item = self.food_data.iloc[idx]
            
            # Skip if not matching category filter
            if category_filter != "All" and food_item.get('Category', '') != category_filter:
                continue
            
            # Calculate condition scores
            condition_scores = {}
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                condition_scores[condition] = self.calculate_condition_score(food_item, condition)
            
            # Calculate combined condition score based on user's conditions
            combined_score = 0
            if conditions:
                for condition in conditions:
                    combined_score += condition_scores.get(condition, 0)
                combined_score /= len(conditions)  # Average across conditions
            
            # Basic nutritional data with safe extraction
            energy = float(food_item.get('Energy(kcal) by calculation', 0))
            protein = float(food_item.get('Protein(g)', 0))
            carbs = float(food_item.get('CHOCDF (g) Carbohydrate', 0))
            sugar = float(food_item.get('SUGAR(g)', 0))
            fiber = float(food_item.get('FIBTG (g) Dietary fibre', 0))
            fat = float(food_item.get('Fat(g)', 0))
            
            # Determine suitable conditions
            suitable_conditions = []
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                score = condition_scores[condition]
                
                # Different threshold for each condition
                threshold = 2
                if condition == 'Diabetes' and score <= threshold and sugar <= 10 and fiber >= 2:
                    suitable_conditions.append(condition)
                elif condition == 'Obesity' and score <= threshold and energy <= 250:
                    suitable_conditions.append(condition)
                elif condition == 'Hypertension' and score <= threshold and float(food_item.get('Na(mg)', 1000)) <= 200:
                    suitable_conditions.append(condition)
                elif condition == 'High_Cholesterol' and score <= threshold and float(food_item.get('FASAT (g) Saturated FA', 100)) <= 3:
                    suitable_conditions.append(condition)
            
            # Create recommendation item
            recommendation = {
                'Name': food_item.get('Thai_Name', food_item.get('English_Name', f"Food {idx}")),
                'Category': food_item.get('Category', 'Unknown'),
                'Energy': energy,
                'Protein': protein,
                'Carbs': carbs,
                'Sugar': sugar,
                'Fiber': fiber,
                'Fat': fat,
                'Distance': distances[0][list(indices[0]).index(idx)],
                'Combined_Score': combined_score,
                'Condition_Scores': condition_scores,
                'Suitable_For': suitable_conditions
            }
            
            recommendations.append(recommendation)
        
        # Sort recommendations
        if conditions:
            # Sort by condition score (lower is better)
            recommendations.sort(key=lambda x: (x['Combined_Score'], x['Distance']))
        else:
            # Sort by closest match to preferences
            recommendations.sort(key=lambda x: x['Distance'])
        
        # Return top recommendations (with at least some items)
        return recommendations[:max_recommendations]
    
    def get_stats(self):
        """Get statistics about loaded data"""
        stats = {
            'total_items': len(self.food_data) if hasattr(self, 'food_data') else 0,
            'categories': {},
            'condition_friendly': {
                'Diabetes': 0,
                'Obesity': 0,
                'Hypertension': 0,
                'High_Cholesterol': 0
            },
            'loading_time': 0
        }
        
        # Count items by category
        if hasattr(self, 'food_data') and 'Category' in self.food_data.columns:
            category_counts = self.food_data['Category'].value_counts().to_dict()
            stats['categories'] = category_counts
            
        # Count condition-friendly items (if columns exist)
        for condition, column_name in [
            ('Diabetes', 'Diabetes'),
            ('Obesity', 'Obesity'),
            ('Hypertension', 'Hypertension'),
            ('High_Cholesterol', 'High Cholesterol')
        ]:
            if hasattr(self, 'food_data') and column_name in self.food_data.columns:
                stats['condition_friendly'][condition] = self.food_data[self.food_data[column_name] <= 1].shape[0]
            
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
        self.master.title("Food Recommendation System for Multiple Health Conditions")
        self.master.geometry("1200x750")
        self.master.configure(bg="#f5f5f5")
        
        # Set theme
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except:
            # Fall back to default theme if clam is not available
            pass
        
        # Configure colors
        self.primary_color = "#3498db"  # Blue
        self.secondary_color = "#2ecc71"  # Green
        self.accent_color = "#e74c3c"  # Red
        self.bg_color = "#f5f5f5"  # Light gray
        self.text_color = "#2c3e50"  # Dark blue/gray
        
        # Configure styles
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Accent.TButton', background=self.secondary_color)
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background=self.bg_color, foreground=self.text_color)
        self.style.configure('Subheader.TLabel', font=('Arial', 12, 'bold'), background=self.bg_color, foreground=self.text_color)
        self.style.configure('Stats.TLabel', font=('Arial', 9), background=self.bg_color, foreground=self.text_color)
        
        # Configure Treeview
        self.style.configure('Treeview', font=('Arial', 9), rowheight=25)
        self.style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        
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
        
    def create_main_layout(self):
        """Create the main layout with modern UI"""
        # Main container
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header with title and recommendation button
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title on the left
        ttk.Label(header_frame, text="Personalized Food Recommendation System for NCDs", 
                 style='Header.TLabel').pack(side=tk.LEFT)
        
        # Recommendation button on the right
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Get Recommendations button with better styling
        recommend_btn = ttk.Button(button_frame, text="Get Recommendations", 
                                  command=self.show_recommendations, style='TButton')
        recommend_btn.pack(side=tk.RIGHT, padx=5)
        
        # Reset button
        reset_btn = ttk.Button(button_frame, text="Reset Settings", 
                              command=self.reset_preferences, style='TButton')
        reset_btn.pack(side=tk.RIGHT, padx=5)
        
        # Create two panels side by side
        panel_container = ttk.Frame(main_container)
        panel_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input controls with scrollbar
        left_panel_container = ttk.Frame(panel_container)
        left_panel_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Add canvas with scrollbar for left panel
        canvas = tk.Canvas(left_panel_container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create the actual content in the scrollable frame
        self.create_input_panel(scrollable_frame)
        
        # Right panel - Results
        right_panel = ttk.Frame(panel_container, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_results_panel(right_panel)
        
        # Nutrition chart panel
        chart_panel = ttk.Frame(main_container, padding=10)
        chart_panel.pack(fill=tk.X, pady=(15, 0))
        
        self.create_chart_panel(chart_panel)
        
        # Create status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create stats display at the bottom right
        self.stats_frame = ttk.Frame(self.status_bar, style='Stats.TLabel')
        self.stats_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create progress bar at the bottom right
        self.progress_frame = ttk.Frame(self.status_bar)
        self.progress_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(side=tk.RIGHT)
        self.progress["value"] = 0
        
        # Make the canvas resize with the window
        self.master.bind("<Configure>", lambda e: canvas.configure(width=left_panel_container.winfo_width()-20))
        
        # Handle mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def create_input_panel(self, parent):
        """Create the input panel with nutrition preferences and health conditions"""
        # Title
        ttk.Label(parent, text="Nutrition & Health Profile", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Health conditions frame
        conditions_frame = ttk.LabelFrame(parent, text="Health Conditions")
        conditions_frame.pack(fill=tk.X, pady=5)
        
        # Create health condition checkboxes
        self.diabetes_var = tk.BooleanVar(value=False)
        self.obesity_var = tk.BooleanVar(value=False)
        self.hypertension_var = tk.BooleanVar(value=False)
        self.cholesterol_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(conditions_frame, text="Diabetes", variable=self.diabetes_var).pack(anchor=tk.W, padx=10, pady=3)
        ttk.Checkbutton(conditions_frame, text="Obesity", variable=self.obesity_var).pack(anchor=tk.W, padx=10, pady=3)
        ttk.Checkbutton(conditions_frame, text="Hypertension (High Blood Pressure)", variable=self.hypertension_var).pack(anchor=tk.W, padx=10, pady=3)
        ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var).pack(anchor=tk.W, padx=10, pady=3)
        
        # Frame for nutritional inputs
        input_frame = ttk.LabelFrame(parent, text="Target Nutritional Values")
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
        
        # Health information frame
        health_frame = ttk.LabelFrame(parent, text="Personal Information")
        health_frame.pack(fill=tk.X, pady=10)
        
        # Weight
        ttk.Label(health_frame, text="Weight (kg):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.weight_var = tk.DoubleVar(value=70.0)
        ttk.Entry(health_frame, textvariable=self.weight_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Height
        ttk.Label(health_frame, text="Height (cm):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.DoubleVar(value=170.0)
        ttk.Entry(health_frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Age
        ttk.Label(health_frame, text="Age:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.age_var = tk.IntVar(value=45)
        ttk.Entry(health_frame, textvariable=self.age_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Gender
        ttk.Label(health_frame, text="Gender:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.gender_var = tk.StringVar(value="Male")
        gender_combo = ttk.Combobox(health_frame, textvariable=self.gender_var, values=['Male', 'Female', 'Other'], width=10)
        gender_combo.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Recommendation settings
        settings_frame = ttk.LabelFrame(parent, text="Recommendation Settings")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Number of recommendations
        ttk.Label(settings_frame, text="Number of recommendations:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_recommendations_var = tk.IntVar(value=10)
        ttk.Combobox(settings_frame, textvariable=self.num_recommendations_var, values=[5, 10, 15, 20, 25], width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Filter by category
        ttk.Label(settings_frame, text="Filter by category:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_filter_var, width=15)
        self.category_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Get all available categories
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty and 'Category' in self.recommender.food_data.columns:
            categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
        
    def create_slider(self, parent, label_text, var_name, default_value, min_val, max_val, row):
        """Create a slider with label and value display"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Create slider variable
        slider_var = tk.IntVar(value=default_value)
        setattr(self, var_name, slider_var)
        
        # Create slider
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=slider_var, 
                         orient=tk.HORIZONTAL, length=150)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Value label
        value_label = ttk.Label(parent, textvariable=slider_var, width=3)
        value_label.grid(row=row, column=2, padx=5, pady=5)
        
    def create_results_panel(self, parent):
        """Create the results panel with recommendations"""
        # Title
        ttk.Label(parent, text="Food Recommendations", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create notebook (tabbed interface) for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Table View
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="Table View")
        
        # Create Treeview with scrollbar for recommendations
        columns = ('Name', 'Category', 'Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'Health Score')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
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
            'Health Score': 'Health Score'
        }
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=column_texts[col])
            width = 120 if col == 'Name' else 70
            self.tree.column(col, width=width, minwidth=50)
        
        # Adjust name column to be wider
        self.tree.column('Name', width=150, minwidth=100)
        self.tree.column('Category', width=100, minwidth=80)
        
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Tab 2: Health Analysis
        health_frame = ttk.Frame(self.notebook)
        self.notebook.add(health_frame, text="Health Analysis")
        
        # Create a canvas for the health analysis chart (will be populated later)
        self.health_canvas_frame = ttk.Frame(health_frame)
        self.health_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Details panel below notebook
        details_frame = ttk.LabelFrame(parent, text="Nutritional Details")
        details_frame.pack(fill=tk.X, pady=10)
        
        # Selected food details
        self.details_text = tk.Text(details_frame, height=5, width=40, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X, padx=5, pady=5)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information")
        self.details_text.config(state=tk.DISABLED)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
    def create_chart_panel(self, parent):
        """Create a panel with charts/graphs"""
        self.fig = Figure(figsize=(12, 3), dpi=100)
        
        # Initialize the chart with empty data
        self.nutrition_chart = FigureCanvasTkAgg(self.fig, parent)
        self.nutrition_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create chart axes
        self.ax1 = self.fig.add_subplot(141)  # Macronutrient distribution
        self.ax2 = self.fig.add_subplot(142)  # Category distribution
        self.ax3 = self.fig.add_subplot(143)  # Health condition scores
        self.ax4 = self.fig.add_subplot(144)  # Nutritional balance
        
        # Update chart with empty data initially
        self.update_charts([])
        
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
            
        # Create a Figure for the heatmap
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for heatmap
        food_names = [rec['Name'] for rec in recommendations]
        conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        
        # Create data matrix
        data = np.zeros((len(food_names), len(conditions)))
        for i, rec in enumerate(recommendations):
            for j, cond in enumerate(conditions):
                data[i, j] = rec['Condition_Scores'].get(cond, 0)
        
        # Create labels for conditions
        condition_labels = ['Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol']
        
        # Create heatmap
        im = sns.heatmap(data, ax=ax, cmap='YlOrRd', linewidths=0.5, 
                    xticklabels=condition_labels, yticklabels=food_names)
        
        # Add colorbar label
        cbar = im.collections[0].colorbar
        cbar.set_label('Score (Lower is Better)', rotation=270, labelpad=15)
        
        # Add suitable markers
        # Check which foods are suitable for which conditions
        for i, rec in enumerate(recommendations):
            for j, cond in enumerate(conditions):
                # If suitable (in rec['Suitable_For']), add a marker
                if cond in rec.get('Suitable_For', []):
                    ax.text(j + 0.5, i + 0.5, '✓', 
                           ha='center', va='center', color='green',
                           fontweight='bold', fontsize=12)
        
        ax.set_title('Health Condition Analysis by Food Item')
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, self.health_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a legend for the checkmark
        legend_frame = ttk.Frame(self.health_canvas_frame)
        legend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(legend_frame, text="Legend: ", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="✓", foreground='green', font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text=" = Suitable for this condition").pack(side=tk.LEFT)
        
    def update_charts(self, recommendations):
        """Update the charts with recommendation data"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        if recommendations and len(recommendations) > 0:
            try:
                # 1. Macronutrient distribution chart
                labels = ['Protein', 'Carbs', 'Fat']
                values = [
                    sum(rec.get('Protein', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('Carbs', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('Fat', 0) for rec in recommendations) / len(recommendations)
                ]
                
                # Filter out zero values for pie chart
                filtered_labels = []
                filtered_values = []
                for i, value in enumerate(values):
                    if value > 0:
                        filtered_labels.append(labels[i])
                        filtered_values.append(value)
                
                if sum(filtered_values) > 0:
                    colors = ['#ff9999','#66b3ff','#99ff99']
                    self.ax1.pie(filtered_values, labels=filtered_labels, colors=colors[:len(filtered_labels)], 
                                autopct='%1.1f%%', shadow=False, startangle=90)
                    self.ax1.set_title('Average Macronutrient Distribution')
                else:
                    self.ax1.text(0.5, 0.5, 'No macronutrient data available', ha='center', va='center')
                
                # 2. Category distribution chart
                categories = {}
                for rec in recommendations:
                    cat = rec.get('Category', 'Unknown')
                    if cat in categories:
                        categories[cat] += 1
                    else:
                        categories[cat] = 1
                
                if categories:
                    cat_names = list(categories.keys())
                    cat_counts = list(categories.values())
                    self.ax2.barh(cat_names, cat_counts, color='#66b3ff')
                    self.ax2.set_title('Food Categories')
                    self.ax2.set_xlabel('Count')
                    self.ax2.set_ylim(-0.5, len(cat_names)-0.5)
                else:
                    self.ax2.text(0.5, 0.5, 'No category data available', ha='center', va='center')
                
                # 3. Health condition scores chart
                conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
                condition_labels = ['Diabetes', 'Obesity', 'HBP', 'Cholesterol']
                avg_scores = []
                
                for condition in conditions:
                    scores = [rec['Condition_Scores'].get(condition, 0) for rec in recommendations]
                    avg_scores.append(sum(scores) / len(scores) if scores else 0)
                
                if any(score > 0 for score in avg_scores):
                    bar_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                    self.ax3.bar(condition_labels, avg_scores, color=bar_colors)
                    self.ax3.set_title('Average Health Impact')
                    self.ax3.set_ylabel('Score (Lower is Better)')
                    self.ax3.set_ylim(0, max(avg_scores) * 1.2 if avg_scores else 1)
                else:
                    self.ax3.text(0.5, 0.5, 'No health score data available', ha='center', va='center')
                
                # 4. Suitable conditions chart
                suitable_counts = {
                    'Diabetes': 0,
                    'Obesity': 0,
                    'Hypertension': 0,
                    'High_Cholesterol': 0
                }
                
                for rec in recommendations:
                    for condition in rec.get('Suitable_For', []):
                        suitable_counts[condition] = suitable_counts.get(condition, 0) + 1
                
                # Calculate percentage suitable for each condition
                suitable_percentages = {}
                total_items = len(recommendations)
                for condition, count in suitable_counts.items():
                    suitable_percentages[condition] = (count / total_items) * 100 if total_items > 0 else 0
                
                if suitable_counts and any(suitable_counts.values()):
                    labels = list(suitable_counts.keys())
                    labels = [label.replace('_', ' ') for label in labels]  # Format labels
                    values = list(suitable_percentages.values())
                    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                    
                    self.ax4.bar(labels, values, color=colors)
                    self.ax4.set_title('Suitable for Conditions (%)')
                    self.ax4.set_ylim(0, 100)
                    self.ax4.set_ylabel('Percentage Suitable')
                    
                    # Rotate labels for better display
                    plt.setp(self.ax4.get_xticklabels(), rotation=30, ha='right')
                else:
                    self.ax4.text(0.5, 0.5, 'No suitability data available', ha='center', va='center')
                
            except Exception as e:
                print(f"Error updating charts: {e}")
                self.ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                self.ax2.text(0.5, 0.5, 'Chart error', ha='center', va='center')
                self.ax3.text(0.5, 0.5, 'Chart error', ha='center', va='center')
                self.ax4.text(0.5, 0.5, 'Chart error', ha='center', va='center')
        else:
            # Show placeholder text
            self.ax1.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            self.ax2.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            self.ax3.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            self.ax4.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            
            # Set empty titles
            self.ax1.set_title('Macronutrient Distribution')
            self.ax2.set_title('Food Categories')
            self.ax3.set_title('Health Impact')
            self.ax4.set_title('Suitable for Conditions (%)')
        
        # Adjust layout and update canvas
        self.fig.tight_layout()
        self.nutrition_chart.draw()
        
    def show_food_details(self, event):
        """Show details for selected food item"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
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
            # Format a more comprehensive details text
            details = f"Food: {food_name}\n"
            details += f"Category: {values[1]}\n"
            details += f"Nutritional Content: Energy: {values[2]}kcal, Protein: {values[3]}g, Carbs: {values[4]}g\n"
            details += f"Sugar: {values[5]}g, Fiber: {values[6]}g, Fat: {values[7]}g\n\n"
            
            # Add health information if available
            if 'Diabetes' in food_details:
                details += f"Diabetes Rating: {food_details['Diabetes']}\n"
            if 'Obesity' in food_details:
                details += f"Obesity Rating: {food_details['Obesity']}\n"
            if 'Hypertension' in food_details:
                details += f"Hypertension Rating: {food_details['Hypertension']}\n"
            if 'High Cholesterol' in food_details:
                details += f"High Cholesterol Rating: {food_details['High Cholesterol']}"
        else:
            # Basic information from the table
            details = f"Food: {food_name}\n"
            details += f"Category: {values[1]}\n"
            details += f"Nutritional Content: Energy: {values[2]}kcal, Protein: {values[3]}g, Carbs: {values[4]}g\n"
            details += f"Sugar: {values[5]}g, Fiber: {values[6]}g, Fat: {values[7]}g, Health Score: {values[8]}"
        
        self.details_text.insert(tk.END, details)
        self.details_text.config(state=tk.DISABLED)
        
    def show_recommendations(self):
        """Show food recommendations based on user preferences and health conditions"""
        # Get user preferences
        user_prefs = {
            'Energy(kcal) by calculation': self.calories_var.get(),
            'Protein(g)': self.protein_var.get(),
            'CHOCDF (g) Carbohydrate': self.carbs_var.get(),
            'SUGAR(g)': self.sugar_var.get(),
            'FIBTG (g) Dietary fibre': self.fiber_var.get(),
            'Fat(g)': self.fat_var.get()
        }
        
        # Show loading in progress bar
        self.status_var.set("Finding recommendations...")
        self.progress["value"] = 10
        self.master.update_idletasks()
        
        # Get selected health conditions
        conditions = []
        if self.diabetes_var.get():
            conditions.append('Diabetes')
        if self.obesity_var.get():
            conditions.append('Obesity')
        if self.hypertension_var.get():
            conditions.append('Hypertension')
        if self.cholesterol_var.get():
            conditions.append('High_Cholesterol')
            
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
                    conditions=conditions,
                    category_filter=category_filter,
                    max_recommendations=num_recommendations
                )
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
                    # Calculate a health score based on the selected conditions
                    health_score = rec.get('Combined_Score', 0)
                    health_score_str = f"{health_score:.1f}"
                    
                    self.tree.insert('', 'end', values=(
                        rec.get('Name', 'Unknown'),
                        rec.get('Category', 'Unknown'),
                        rec.get('Energy', 0),
                        rec.get('Protein', 0),
                        rec.get('Carbs', 0),
                        rec.get('Sugar', 0),
                        rec.get('Fiber', 0),
                        rec.get('Fat', 0),
                        health_score_str
                    ))
                self.status_var.set(f"Found {len(recommendations)} recommendations")
                
                # Update charts with new data
                self.progress["value"] = 80
                self.master.update_idletasks()
                self.update_charts(recommendations)
                
                # Update health analysis chart in second tab
                self.update_health_analysis_chart(recommendations)
            else:
                messagebox.showinfo("No Recommendations", 
                                "No recommendations found matching your criteria.")
                self.status_var.set("No recommendations found")
                
                # Clear charts
                self.update_charts([])
                self.update_health_analysis_chart([])
                
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
        self.num_recommendations_var.set(10)
        self.category_filter_var.set("All")
        
        # Update status
        self.status_var.set("All settings reset to defaults")
    
    def update_stats_display(self):
        """Update the statistics display in the bottom right corner"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Create stats labels
            stats_text = f"Loaded {stats['total_items']} food items | "
            stats_text += f"{len(stats['categories'])} categories"
            
            # Add condition-friendly counts
            condition_counts = []
            for condition, count in stats.get('condition_friendly', {}).items():
                if count > 0:
                    condition_counts.append(f"{count} {condition.replace('_', ' ')}-friendly")
            
            if condition_counts:
                stats_text += f" | {', '.join(condition_counts)}"
                
            if 'loading_time' in stats and stats['loading_time'] > 0:
                stats_text += f" | Loaded in {stats['loading_time']:.2f}s"
            
            ttk.Label(self.stats_frame, text=stats_text, style='Stats.TLabel').pack(side=tk.RIGHT)
        except Exception as e:
            print(f"Error updating stats display: {e}")
            ttk.Label(self.stats_frame, text="Stats unavailable", style='Stats.TLabel').pack(side=tk.RIGHT)


# Main function to run the application
def main():
    # Set up basic logging to console
    print("Starting application...")
    
    # Create the root window first
    root = tk.Tk()
    root.title("Food Recommendation System for NCDs")
    root.geometry("1200x750")
    
    # Set window icon if available
    try:
        # You can add an icon file to your project and use it here
        #root.iconbitmap("icon.ico")
        pass
    except:
        pass
    
    # Create a splash frame that covers the main window
    splash_frame = ttk.Frame(root)
    splash_frame.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Create splash screen elements
    splash_label = ttk.Label(splash_frame, text="Food Recommendation System", font=("Arial", 18, "bold"))
    splash_label.pack(pady=50)
    
    subtitle_label = ttk.Label(splash_frame, text="For Multiple Health Conditions (NCDs)", font=("Arial", 12))
    subtitle_label.pack(pady=10)
    
    progress = ttk.Progressbar(splash_frame, orient="horizontal", length=400, mode="indeterminate")
    progress.pack(pady=20)
    progress.start()
    
    status_var = tk.StringVar(value="Initializing system...")
    status_label = ttk.Label(splash_frame, textvariable=status_var, font=("Arial", 10))
    status_label.pack(pady=20)
    
    # Add information about supported conditions
    conditions_text = "Supports: Diabetes • Obesity • Hypertension • High Cholesterol"
    conditions_label = ttk.Label(splash_frame, text=conditions_text, font=("Arial", 11))
    conditions_label.pack(pady=30)
    
    # Add a footer with credits
    footer_text = "Developed by Surat Lawdi - Prince of Songkla University"
    footer_label = ttk.Label(splash_frame, text=footer_text, font=("Arial", 8))
    footer_label.pack(side=tk.BOTTOM, pady=20)
    
    def update_status(message):
        """Update status message and ensure UI updates"""
        print(f"Status: {message}")  # Debug print
        status_var.set(message)
        root.update_idletasks()  # Force update of the UI
    
    def initialize_app():
        try:
            # Create the food recommender
            recommender = FoodRecommendationSystem(update_status)
            
            # Create main app UI
            app = FoodRecommenderUI(root, recommender)
            
            # Remove splash screen
            update_status("Ready! System loaded successfully.")
            root.after(1500, splash_frame.destroy)  # Destroy splash frame after delay
            
        except Exception as e:
            # Show error on splash screen
            update_status(f"Error initializing: {str(e)}")
            print(f"Initialization error: {e}")  # Debug print
    
    # Schedule the initialization to happen after the window is shown
    root.after(100, initialize_app)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()