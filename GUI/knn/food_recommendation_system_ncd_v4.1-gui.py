import pandas as pd
import numpy as np
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

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
            'Diabetes': ['SUGAR(g)', 'CHOCDF (g) Carbohydrate', 'FIBTG (g) Dietary fibre'],
            'Obesity': ['Energy(kcal) by calculation', 'Fat(g)', 'FIBTG (g) Dietary fibre'],
            'Hypertension': ['Na(mg)', 'K(mg)', 'Fat(g)'],
            'High_Cholesterol': ['Fat(g)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol', 'FIBTG (g) Dietary fibre']
        }
        
        # Dietary guidelines based on NCD research
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
                # Create sample data for demonstration
                self.create_sample_data()
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
                self.create_sample_data()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            # Create sample data if loading fails
            self.create_sample_data()

    def create_sample_data(self):
        """Create sample food data for demonstration"""
        self.update_status("Creating sample food data for demonstration...")
        
        # Sample food data
        sample_foods = [
            # Fruits
            {'Thai_Name': 'แอปเปิ้ล', 'English_Name': 'Apple', 'Category': 'Fruit', 
             'Energy(kcal) by calculation': 52, 'Protein(g)': 0.3, 'CHOCDF (g) Carbohydrate': 14, 
             'SUGAR(g)': 10, 'FIBTG (g) Dietary fibre': 2.4, 'Fat(g)': 0.2, 'Na(mg)': 1, 'K(mg)': 107},
            {'Thai_Name': 'กล้วย', 'English_Name': 'Banana', 'Category': 'Fruit',
             'Energy(kcal) by calculation': 89, 'Protein(g)': 1.1, 'CHOCDF (g) Carbohydrate': 23,
             'SUGAR(g)': 12, 'FIBTG (g) Dietary fibre': 2.6, 'Fat(g)': 0.3, 'Na(mg)': 1, 'K(mg)': 358},
            {'Thai_Name': 'ส้ม', 'English_Name': 'Orange', 'Category': 'Fruit',
             'Energy(kcal) by calculation': 47, 'Protein(g)': 0.9, 'CHOCDF (g) Carbohydrate': 12,
             'SUGAR(g)': 9, 'FIBTG (g) Dietary fibre': 2.4, 'Fat(g)': 0.1, 'Na(mg)': 0, 'K(mg)': 181},
            
            # Vegetables
            {'Thai_Name': 'บรอกโคลี', 'English_Name': 'Broccoli', 'Category': 'Vegetable',
             'Energy(kcal) by calculation': 34, 'Protein(g)': 2.8, 'CHOCDF (g) Carbohydrate': 7,
             'SUGAR(g)': 1.5, 'FIBTG (g) Dietary fibre': 2.6, 'Fat(g)': 0.4, 'Na(mg)': 33, 'K(mg)': 316},
            {'Thai_Name': 'แครอท', 'English_Name': 'Carrot', 'Category': 'Vegetable',
             'Energy(kcal) by calculation': 41, 'Protein(g)': 0.9, 'CHOCDF (g) Carbohydrate': 10,
             'SUGAR(g)': 4.7, 'FIBTG (g) Dietary fibre': 2.8, 'Fat(g)': 0.2, 'Na(mg)': 69, 'K(mg)': 320},
            {'Thai_Name': 'ผักโขม', 'English_Name': 'Spinach', 'Category': 'Vegetable',
             'Energy(kcal) by calculation': 23, 'Protein(g)': 2.9, 'CHOCDF (g) Carbohydrate': 3.6,
             'SUGAR(g)': 0.4, 'FIBTG (g) Dietary fibre': 2.2, 'Fat(g)': 0.4, 'Na(mg)': 79, 'K(mg)': 558},
            
            # Grains
            {'Thai_Name': 'ข้าวกล้อง', 'English_Name': 'Brown Rice', 'Category': 'Grain',
             'Energy(kcal) by calculation': 111, 'Protein(g)': 2.6, 'CHOCDF (g) Carbohydrate': 23,
             'SUGAR(g)': 0.4, 'FIBTG (g) Dietary fibre': 1.8, 'Fat(g)': 0.9, 'Na(mg)': 5, 'K(mg)': 43},
            {'Thai_Name': 'ข้าวโอ๊ต', 'English_Name': 'Oats', 'Category': 'Grain',
             'Energy(kcal) by calculation': 389, 'Protein(g)': 16.9, 'CHOCDF (g) Carbohydrate': 66,
             'SUGAR(g)': 0.99, 'FIBTG (g) Dietary fibre': 10.6, 'Fat(g)': 6.9, 'Na(mg)': 2, 'K(mg)': 429},
            
            # Proteins
            {'Thai_Name': 'ไก่', 'English_Name': 'Chicken Breast', 'Category': 'Meat',
             'Energy(kcal) by calculation': 165, 'Protein(g)': 31, 'CHOCDF (g) Carbohydrate': 0,
             'SUGAR(g)': 0, 'FIBTG (g) Dietary fibre': 0, 'Fat(g)': 3.6, 'Na(mg)': 74, 'K(mg)': 256},
            {'Thai_Name': 'ปลาแซลมอน', 'English_Name': 'Salmon', 'Category': 'Fish',
             'Energy(kcal) by calculation': 208, 'Protein(g)': 25, 'CHOCDF (g) Carbohydrate': 0,
             'SUGAR(g)': 0, 'FIBTG (g) Dietary fibre': 0, 'Fat(g)': 12, 'Na(mg)': 59, 'K(mg)': 363},
            
            # Dairy
            {'Thai_Name': 'นมโยเกิร์ต', 'English_Name': 'Greek Yogurt', 'Category': 'Dairy',
             'Energy(kcal) by calculation': 59, 'Protein(g)': 10, 'CHOCDF (g) Carbohydrate': 3.6,
             'SUGAR(g)': 3.2, 'FIBTG (g) Dietary fibre': 0, 'Fat(g)': 0.4, 'Na(mg)': 36, 'K(mg)': 141},
            
            # Nuts
            {'Thai_Name': 'อัลมอนด์', 'English_Name': 'Almonds', 'Category': 'Nuts',
             'Energy(kcal) by calculation': 579, 'Protein(g)': 21, 'CHOCDF (g) Carbohydrate': 22,
             'SUGAR(g)': 4.4, 'FIBTG (g) Dietary fibre': 12.5, 'Fat(g)': 50, 'Na(mg)': 1, 'K(mg)': 733}
        ]
        
        # Add optional columns with default values
        for food in sample_foods:
            food['FASAT (g) Saturated FA'] = food['Fat(g)'] * 0.3  # Estimate
            food['Ca(mg)'] = 50  # Default calcium
            food['CHOLE(mg) Cholesterol'] = 0 if food['Category'] not in ['Meat', 'Fish', 'Dairy'] else 20
        
        self.food_data = pd.DataFrame(sample_foods)
        self.stats['total_items'] = len(self.food_data)
        self.stats['categories'] = self.food_data['Category'].value_counts().to_dict()
        
        self.update_status(f"Created sample data with {len(self.food_data)} food items")
    
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
        
        elif condition == 'Obesity':
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
        
        elif condition == 'High_Cholesterol':
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
        
        return max(0, score)  # Ensure score is non-negative
    
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
        
        # Subtitle for NCDs
        subtitle_label = ttk.Label(header_frame, 
                                 text="for Non-Communicable Diseases (NCDs)", 
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
        
        # Left panel - Input controls with proportion-based width
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
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        # Add after canvas.pack():
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        # Add this function to update canvas window width:
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind('<Configure>', configure_canvas)

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
        # Only handle main window resize events
        if event and event.widget != self.master:
            return
            
        # Minimum width for input panel
        min_width = 280
        
        # Get 25% of window width but not less than min_width
        desired_width = max(min_width, int(self.master.winfo_width() * 0.25))
        
        # Find and update canvas width
        def find_and_update_canvas(widget):
            for child in widget.winfo_children():
                if isinstance(child, tk.Canvas) and child.winfo_width() > 200:  # Identify our canvas
                    child.configure(width=desired_width)
                    return True
                elif hasattr(child, 'winfo_children'):
                    if find_and_update_canvas(child):
                        return True
            return False
        
        find_and_update_canvas(self.master)

        
    
    

    def create_input_panel(self, parent):
        """Create the input panel with nutrition preferences and health conditions"""
        # Title with better spacing
        ttk.Label(parent, text="Nutrition & Health Profile", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Health conditions frame with improved styling
        conditions_frame = ttk.LabelFrame(parent, text="Health Conditions", padding=10)
        conditions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create health condition checkboxes
        self.diabetes_var = tk.BooleanVar(value=False)
        self.obesity_var = tk.BooleanVar(value=False)
        self.hypertension_var = tk.BooleanVar(value=False)
        self.cholesterol_var = tk.BooleanVar(value=False)
        
        # Add tooltips with explanations
        diabetes_check = ttk.Checkbutton(conditions_frame, text="Diabetes", variable=self.diabetes_var)
        diabetes_check.pack(anchor=tk.W, padx=5, pady=5)
        
        obesity_check = ttk.Checkbutton(conditions_frame, text="Obesity/Weight Management", variable=self.obesity_var)
        obesity_check.pack(anchor=tk.W, padx=5, pady=5)
        
        hypertension_check = ttk.Checkbutton(conditions_frame, text="Hypertension (High Blood Pressure)", variable=self.hypertension_var)
        hypertension_check.pack(anchor=tk.W, padx=5, pady=5)
        
        cholesterol_check = ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var)
        cholesterol_check.pack(anchor=tk.W, padx=5, pady=5)
        
        # Frame for nutritional inputs with better styling
        input_frame = ttk.LabelFrame(parent, text="Target Nutritional Values", padding=10)
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
        
        # Height with validation
        ttk.Label(health_frame, text="Height (cm):").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.height_var = tk.DoubleVar(value=170.0)
        height_entry = ttk.Entry(health_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Age with validation
        ttk.Label(health_frame, text="Age:").grid(row=2, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.age_var = tk.IntVar(value=45)
        age_entry = ttk.Entry(health_frame, textvariable=self.age_var, width=10)
        age_entry.grid(row=2, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Gender
        ttk.Label(health_frame, text="Gender:").grid(row=3, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.gender_var = tk.StringVar(value="Male")
        gender_combo = ttk.Combobox(health_frame, textvariable=self.gender_var, values=['Male', 'Female', 'Other'], width=10)
        gender_combo.grid(row=3, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Add BMI calculation
        ttk.Label(health_frame, text="BMI:").grid(row=4, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.bmi_var = tk.StringVar(value="Computing...")
        ttk.Label(health_frame, textvariable=self.bmi_var).grid(row=4, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Calculate BMI when weight or height changes
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()  # Initial calculation
        
        # Recommendation settings with better styling
        settings_frame = ttk.LabelFrame(parent, text="Recommendation Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Number of recommendations
        ttk.Label(settings_frame, text="Number of recommendations:").grid(row=0, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.num_recommendations_var = tk.IntVar(value=10)
        ttk.Combobox(settings_frame, textvariable=self.num_recommendations_var, values=[5, 10, 15, 20, 25], width=5).grid(row=0, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Filter by category
        ttk.Label(settings_frame, text="Filter by category:").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_filter_var, width=15)
        self.category_combo.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky=tk.W)
        
        # Get all available categories
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty and 'Category' in self.recommender.food_data.columns:
            categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories

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
        
        # Create Treeview with scrollbar for recommendations
        columns = ('Name', 'Category', 'Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'Health Score')
        self.tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Add scrollbars - both vertical and horizontal
        y_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=self.tree.xview)
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
            'Health Score': 'Health Score'
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
        
        # Details panel
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
        self.ax3.set_title('Health Impact', fontsize=10, fontweight='bold')
        self.ax4.set_title('Suitable for Conditions (%)', fontsize=10, fontweight='bold')
        
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
                colors = ['#ff9999','#66b3ff','#99ff99']
                
                for i, value in enumerate(values):
                    if value > 0:
                        filtered_labels.append(labels[i])
                        filtered_values.append(value)
                
                if sum(filtered_values) > 0:
                    self.ax1.pie(filtered_values, labels=filtered_labels, 
                               colors=colors[:len(filtered_labels)], 
                               autopct='%1.1f%%', shadow=False, startangle=90)
                    self.ax1.set_title('Average Macronutrient Distribution')
                else:
                    self.ax1.text(0.5, 0.5, 'No macronutrient data available', 
                                ha='center', va='center')
                
                # 2. Category distribution chart
                categories = {}
                for rec in recommendations:
                    cat = rec.get('Category', 'Unknown')
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
                else:
                    self.ax2.text(0.5, 0.5, 'No category data available', 
                                ha='center', va='center')
                
                # 3. Health condition scores chart
                conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
                condition_labels = ['Diabetes', 'Obesity', 'Hypertension', 'Cholesterol']
                avg_scores = []
                
                for condition in conditions:
                    scores = [rec['Condition_Scores'].get(condition, 0) for rec in recommendations]
                    avg_scores.append(sum(scores) / len(scores) if scores else 0)
                
                if any(score > 0 for score in avg_scores):
                    # Use a color palette that provides good contrast
                    bar_colors = plt.cm.Paired(np.linspace(0, 1, len(condition_labels)))
                    
                    bars = self.ax3.bar(condition_labels, avg_scores, color=bar_colors)
                    self.ax3.set_title('Average Health Impact')
                    self.ax3.set_ylabel('Score (Lower is Better)')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        self.ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
                    # Set y-limit with some padding for the labels
                    self.ax3.set_ylim(0, max(avg_scores) * 1.2 if avg_scores else 1)
                    
                    # Add a subtle horizontal grid for better readability
                    self.ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
                    
                    # Slight rotation for x-labels to prevent overlap
                    plt.setp(self.ax3.get_xticklabels(), rotation=15, ha='right')
                else:
                    self.ax3.text(0.5, 0.5, 'No health score data available', 
                                ha='center', va='center')
                
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
                for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                    ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        else:
            # Show placeholder text
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
            self.details_text.insert(tk.END, f"• Health Score: {values[8]}")
        
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
                    # Calculate a health score based on the selected conditions
                    health_score = rec.get('Combined_Score', 0)
                    health_score_str = f"{health_score:.1f}"
                    
                    # Insert into treeview with color coding based on health score
                    item_id = self.tree.insert('', 'end', values=(
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
                
                # Selection for the first item to display details
                if self.tree.get_children():
                    first_item = self.tree.get_children()[0]
                    self.tree.selection_set(first_item)
                    self.tree.focus(first_item)
                    self.show_food_details(None)  # Show details for the first item
                
                # Update status
                condition_text = ", ".join(condition.replace("_", " ") for condition in conditions) if conditions else "general preferences"
                self.status_var.set(f"Found {len(recommendations)} recommendations for {condition_text}")
                
                # Update charts with new data
                self.progress["value"] = 80
                self.master.update_idletasks()
                self.update_charts(recommendations)
                
            else:
                messagebox.showinfo("No Recommendations", 
                                   "No recommendations found matching your criteria.")
                self.status_var.set("No recommendations found")
                
                # Clear charts
                self.update_charts([])
                
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
    print("Starting Food Recommendation System...")
    
    # Create the root window first
    root = tk.Tk()
    root.title("Food Recommendation System for NCDs")
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
    title_label = tk.Label(content_frame, text="Food Recommendation System",
                         font=title_font, bg="white", fg="#2c3e50")
    title_label.pack(pady=(40, 10))
    
    subtitle_font = ('Arial', 14)
    subtitle_label = tk.Label(content_frame, 
                           text="For Non-Communicable Diseases (NCDs)",
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
    
    # Add information about supported conditions with icons
    conditions_frame = ttk.Frame(content_frame)
    conditions_frame.pack(pady=20)
    
    # Each condition in its own label for better styling
    conditions = ["Diabetes", "Obesity", "Hypertension", "High Cholesterol"]
    for i, condition in enumerate(conditions):
        condition_label = tk.Label(conditions_frame, text=f"• {condition}",
                                font=('Arial', 11), bg="white", fg="#2980b9")
        condition_label.grid(row=i//2, column=i%2, padx=20, pady=5, sticky=tk.W)
    
    # Add a footer with credits
    footer_text = "Developed by Surat Lawdi - Prince of Songkla University"
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
            
            update_status("Preparing recommendation engine...")
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