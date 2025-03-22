import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import threading
import os
import glob
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import platform

# Import ttkbootstrap instead of ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import Style

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

class FoodRecommendationSystemRF:
    
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
            'model_performance': {
                'r2_score': 0,
                'feature_importance': {}
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
        
        # Prepare the data for Random Forest models
        self.prepare_rf_models()
        
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
    
    def prepare_rf_models(self):
        """Prepare Random Forest models for each health condition"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for Random Forest models")
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
            
        # Create Random Forest models for each condition
        self.rf_models = {}
        self.feature_importance = {}
        
        # Conditions to model
        conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        
        for condition in conditions:
            self.update_status(f"Training Random Forest model for {condition.replace('_', ' ')}...")
            
            # Check if direct condition column exists
            condition_col = condition if condition != 'High_Cholesterol' else 'High Cholesterol'
            
            if condition_col in self.food_data.columns:
                # Get features and target
                X = self.food_data[self.features].fillna(0)
                y = self.food_data[condition_col].fillna(3)  # Default to moderate if missing
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data for training and evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Create and train Random Forest model
                rf_model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
                
                rf_model.fit(X_train, y_train)
                
                # Evaluate model
                score = rf_model.score(X_test, y_test)
                self.update_status(f"{condition.replace('_', ' ')} model R² score: {score:.4f}")
                
                # Save the model and feature importance
                self.rf_models[condition] = {
                    'model': rf_model,
                    'scaler': scaler,
                    'score': score,
                    'features': self.features.copy()
                }
                
                # Update model performance stats
                self.stats['model_performance']['r2_score'] = score
                
                # Store feature importance
                feature_importance = rf_model.feature_importances_
                self.feature_importance[condition] = dict(zip(self.features, feature_importance))
                self.stats['model_performance']['feature_importance'][condition] = \
                    dict(sorted(zip(self.features, feature_importance), key=lambda x: x[1], reverse=True)[:5])
                
                # Print top 5 important features
                important_features = sorted(
                    zip(self.features, feature_importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                importance_msg = f"Top features for {condition.replace('_', ' ')}: "
                importance_msg += ", ".join([f"{f[0]} ({f[1]:.3f})" for f in important_features])
                self.update_status(importance_msg)
            else:
                # If condition column doesn't exist, create a synthetic model based on nutritional guidelines
                self.update_status(f"No direct {condition} column found. Creating synthetic model based on guidelines.")
                
                # Create a new target column based on nutritional values
                synthetic_targets = self.generate_synthetic_targets(condition)
                
                if synthetic_targets is not None:
                    # Get features 
                    X = self.food_data[self.features].fillna(0)
                    y = synthetic_targets
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                    
                    # Create and train Random Forest model
                    rf_model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )
                    
                    rf_model.fit(X_train, y_train)
                    
                    # Evaluate model
                    score = rf_model.score(X_test, y_test)
                    self.update_status(f"Synthetic {condition.replace('_', ' ')} model R² score: {score:.4f}")
                    
                    # Save the model
                    self.rf_models[condition] = {
                        'model': rf_model,
                        'scaler': scaler,
                        'score': score,
                        'features': self.features.copy(),
                        'synthetic': True
                    }
                    
                    # Store feature importance
                    feature_importance = rf_model.feature_importances_
                    self.feature_importance[condition] = dict(zip(self.features, feature_importance))
                else:
                    self.update_status(f"Could not create model for {condition}")
    
    def generate_synthetic_targets(self, condition):
        """Generate synthetic target scores for a condition based on nutritional guidelines"""
        if len(self.food_data) == 0:
            return None
            
        # Create a Series to hold the synthetic scores
        scores = pd.Series(index=self.food_data.index, dtype=float)
        
        # For each food item, calculate a score based on condition-specific rules
        for idx, food in self.food_data.iterrows():
            score = self.calculate_condition_score(food, condition)
            scores[idx] = score
        
        return scores
    
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
    
    def get_recommendations(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """Get food recommendations based on user preferences and health conditions using Random Forest models"""
        if not hasattr(self, 'rf_models') or len(self.food_data) == 0 or not self.rf_models:
            return []
        
        # Default to empty list if conditions is None
        if conditions is None:
            conditions = []
        
        # Filter foods by category if specified
        if category_filter != "All":
            filtered_data = self.food_data[self.food_data['Category'] == category_filter].copy()
        else:
            filtered_data = self.food_data.copy()
            
        if len(filtered_data) == 0:
            return []
            
        # Calculate condition scores for all foods
        recommendations = []
        
        for idx, food in filtered_data.iterrows():
            # Basic food info
            food_info = {
                'Name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'Category': food.get('Category', 'Unknown'),
                'Energy': float(food.get('Energy(kcal) by calculation', 0)),
                'Protein': float(food.get('Protein(g)', 0)),
                'Carbs': float(food.get('CHOCDF (g) Carbohydrate', 0)),
                'Sugar': float(food.get('SUGAR(g)', 0)),
                'Fiber': float(food.get('FIBTG (g) Dietary fibre', 0)),
                'Fat': float(food.get('Fat(g)', 0)),
                'Condition_Scores': {},
                'Suitability_Scores': {},
                'Suitable_For': []
            }
            
            # Calculate condition scores
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                # Calculate base score using nutritional guidelines
                base_score = self.calculate_condition_score(food, condition)
                food_info['Condition_Scores'][condition] = base_score
                
                # If we have a model for this condition, use it to predict suitability
                if condition in self.rf_models:
                    model_info = self.rf_models[condition]
                    model = model_info['model']
                    scaler = model_info['scaler']
                    features = model_info['features']
                    
                    # Extract feature values and handle missing features
                    feature_values = []
                    for feature in features:
                        if feature in food and not pd.isna(food[feature]):
                            feature_values.append(float(food[feature]))
                        else:
                            feature_values.append(0)  # Default to 0 for missing values
                    
                    # Scale features
                    feature_values_scaled = scaler.transform([feature_values])
                    
                    # Predict suitability score
                    suitability = model.predict(feature_values_scaled)[0]
                    food_info['Suitability_Scores'][condition] = suitability
                    
                    # Determine if food is suitable for this condition
                    # Lower scores are better (0-1 typically means suitable)
                    if suitability <= 1.5:
                        food_info['Suitable_For'].append(condition)
                else:
                    # Fall back to base score if no model exists
                    food_info['Suitability_Scores'][condition] = base_score
                    if base_score <= 1.5:
                        food_info['Suitable_For'].append(condition)
            
            # Calculate combined score for sorting based on user conditions
            if conditions:
                combined_score = 0
                for condition in conditions:
                    if condition in food_info['Suitability_Scores']:
                        combined_score += food_info['Suitability_Scores'][condition]
                combined_score /= len(conditions)  # Average across conditions
                food_info['Combined_Score'] = combined_score
            else:
                # If no specific conditions, score based on overall healthiness
                food_info['Combined_Score'] = sum(food_info['Condition_Scores'].values()) / len(food_info['Condition_Scores'])
            
            # Calculate preference match score based on user preferences
            pref_match_score = 0
            for feature, target_value in user_preferences.items():
                if feature in food and not pd.isna(food[feature]):
                    # Calculate how close the food's value is to the target
                    actual_value = float(food[feature])
                    # Normalize the difference by the possible range of values
                    feature_range = 1  # Default
                    if feature in self.food_data.columns:
                        feature_max = self.food_data[feature].max()
                        feature_min = self.food_data[feature].min() 
                        feature_range = max(1, feature_max - feature_min)
                    
                    # Calculate normalized difference
                    diff = abs(actual_value - target_value) / feature_range
                    pref_match_score += 1 - min(1, diff)  # Higher score = better match
            
            # Normalize preference score
            if user_preferences:
                pref_match_score /= len(user_preferences)
                food_info['Preference_Match'] = pref_match_score
            else:
                food_info['Preference_Match'] = 0.5  # Default neutral score
            
            recommendations.append(food_info)
        
        # Sort recommendations based on conditions and preferences
        if conditions:
            # Sort by condition score (lower is better) and preference match (higher is better)
            recommendations.sort(key=lambda x: (x['Combined_Score'], -x['Preference_Match']))
        else:
            # Sort primarily by preference match
            recommendations.sort(key=lambda x: (-x['Preference_Match'], x['Combined_Score']))
        
        # Return top recommendations (with at least some items)
        return recommendations[:max_recommendations]
    
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
            "model_predictions": {},
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
        
        # Use Random Forest models to predict suitability for each condition
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            # Calculate base score using nutritional guidelines
            base_score = self.calculate_condition_score(food_item, condition)
            analysis["condition_scores"][condition] = base_score
            
            # If we have a model for this condition, use it to predict suitability
            if condition in self.rf_models:
                model_info = self.rf_models[condition]
                model = model_info['model']
                scaler = model_info['scaler']
                features = model_info['features']
                
                # Extract feature values and handle missing features
                feature_values = []
                for feature in features:
                    if feature in food_item and not pd.isna(food_item[feature]):
                        feature_values.append(float(food_item[feature]))
                    else:
                        feature_values.append(0)  # Default to 0 for missing values
                
                # Scale features
                feature_values_scaled = scaler.transform([feature_values])
                
                # Predict suitability score
                suitability = model.predict(feature_values_scaled)[0]
                analysis["model_predictions"][condition] = suitability
                
                # Determine if food is suitable for this condition
                if suitability <= 1.5:
                    analysis["suitable_for"].append(condition.replace('_', ' '))
            else:
                # Fall back to base score if no model exists
                analysis["model_predictions"][condition] = base_score
                if base_score <= 1.5:
                    analysis["suitable_for"].append(condition.replace('_', ' '))
        
        # Add warnings based on nutritional content
        if analysis["sugar_g"] > 25:
            analysis["warnings"].append("High sugar content exceeds daily recommended limit")
        
        if analysis["sodium_mg"] > 400:
            analysis["warnings"].append("High sodium content")
        
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
        elif analysis["model_predictions"].get("Diabetes", 0) >= 3:
            if analysis["sugar_g"] > 15:
                recommendations.append("People with diabetes should limit consumption due to high sugar content.")
            elif analysis["carbs_g"] > 30 and analysis["fiber_g"] < 3:
                recommendations.append("People with diabetes should consume in moderation due to high carb and low fiber content.")
        
        # Obesity recommendations
        if "Obesity" in analysis["suitable_for"]:
            recommendations.append("Suitable for weight management due to its lower calorie profile.")
        elif analysis["model_predictions"].get("Obesity", 0) >= 3:
            recommendations.append("Those managing their weight should limit portion size due to high calorie content.")
        
        # Hypertension recommendations
        if "Hypertension" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with hypertension due to its lower sodium content.")
        elif analysis["model_predictions"].get("Hypertension", 0) >= 3:
            recommendations.append("People with hypertension should limit consumption due to high sodium content.")
        
        # High Cholesterol recommendations
        if "High Cholesterol" in analysis["suitable_for"]:
            recommendations.append("Suitable for people with high cholesterol due to its heart-healthy profile.")
        elif analysis["model_predictions"].get("High_Cholesterol", 0) >= 3:
            recommendations.append("People with high cholesterol should limit consumption due to saturated fat/cholesterol content.")
        
        # General recommendations
        if not recommendations:
            if sum(analysis["model_predictions"].values()) <= 4:
                recommendations.append("Generally acceptable for most diets in moderation.")
            else:
                recommendations.append("Best consumed occasionally as part of a varied and balanced diet.")
        
        return recommendations
    
    def get_stats(self):
        """Get statistics about loaded data and model performance"""
        stats = {
            'total_items': len(self.food_data) if hasattr(self, 'food_data') else 0,
            'categories': {},
            'condition_friendly': {
                'Diabetes': 0,
                'Obesity': 0,
                'Hypertension': 0,
                'High_Cholesterol': 0
            },
            'model_performance': {},
            'loading_time': self.stats.get('loading_time', 0)
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
        
        # Add model performance metrics
        if hasattr(self, 'rf_models'):
            for condition, model_info in self.rf_models.items():
                stats['model_performance'][condition] = {
                    'r2_score': model_info.get('score', 0),
                    'feature_importance': self.feature_importance.get(condition, {})
                }
        
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
    
    def evaluate_model_performance(self):
        """Evaluate and compare model performance metrics"""
        if not hasattr(self, 'rf_models') or not self.rf_models:
            return "No models available for evaluation"
            
        performance = {}
        for condition, model_info in self.rf_models.items():
            if 'score' in model_info:
                performance[condition] = {
                    'r2_score': model_info['score'],
                    'is_synthetic': model_info.get('synthetic', False)
                }
                
        return performance
    
    def plot_feature_importance(self, condition, top_n=10):
        """Plot feature importance for a specific condition"""
        if condition not in self.feature_importance:
            return None
            
        importance = self.feature_importance[condition]
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot
        features = [f[0] for f in top_features]
        values = [f[1] for f in top_features]
        
        ax.barh(features, values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Features for {condition.replace("_", " ")}')
        ax.invert_yaxis()  # Display the most important feature at the top
        
        return fig


class FoodRecommenderUI:
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Food Recommendation System for Multiple Health Conditions")
        self.master.geometry("1200x750")
        
        # Set the ttkbootstrap theme
        self.style = ttk.Style(theme="cosmo")
        
        # Get screen dimensions to allow responsive scaling
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        
        # Configure colors
        self.primary_color = "#2980b9"  # Darker blue for better contrast
        self.secondary_color = "#27ae60"  # Darker green for better contrast
        self.accent_color = "#e74c3c"  # Red
        self.bg_color = "#f5f5f5"  # Light gray
        self.text_color = "#2c3e50"  # Dark blue/gray
        self.light_text_color = "#7f8c8d"  # For less important text
        
        # Calculate font sizes based on screen resolution
        self.default_font_size = self.calculate_font_size(10)
        self.header_font_size = self.calculate_font_size(16)
        self.subheader_font_size = self.calculate_font_size(12)
        
        # Initialize the recommendation system if not provided
        self.recommender = recommender or FoodRecommendationSystemRF()
        
        # Create main layout
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
        main_container.pack(fill="both", expand=True, padx=20, pady=15)
        
        # Header with title and recommendation button
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill="x", pady=(0, 15))
        
        # Title on the left with better styling
        title_label = ttk.Label(
            header_frame, 
            text="Personalized Food Recommendation System", 
            font=("Helvetica", self.header_font_size, "bold"),
            bootstyle="primary"
        )
        title_label.pack(side="left")
        
        # Subtitle for NCDs
        subtitle_label = ttk.Label(
            header_frame, 
            text="for Non-Communicable Diseases (NCDs)", 
            font=("Helvetica", self.calculate_font_size(12))
        )
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Buttons on the right
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side="right")
        
        # Get Recommendations button with better styling
        recommend_btn = ttk.Button(
            button_frame, 
            text="Get Recommendations", 
            command=self.show_recommendations, 
            bootstyle="success"
        )
        recommend_btn.pack(side="right", padx=5)
        
        # Reset button
        reset_btn = ttk.Button(
            button_frame, 
            text="Reset Settings", 
            command=self.reset_preferences,
            bootstyle="secondary"
        )
        reset_btn.pack(side="right", padx=5)
        
        # Model performance button
        model_btn = ttk.Button(
            button_frame, 
            text="Model Metrics", 
            command=self.show_model_metrics,
            bootstyle="info"
        )
        model_btn.pack(side="right", padx=5)
        
        # Create two panels with flexible sizing
        panel_container = ttk.Frame(main_container)
        panel_container.pack(fill="both", expand=True)
        
        # Left panel - Input controls - now with proportion-based width
        left_panel_container = ttk.Frame(panel_container)
        left_panel_container.pack(side="left", fill="both", expand=False, padx=(0, 15), pady=0, ipadx=5)
        
        # Create a PanedWindow for resizable panels
        paned_window = ttk.PanedWindow(panel_container, orient="horizontal")
        paned_window.pack(fill="both", expand=True)
        
        # Add canvas with scrollbar for left panel
        canvas = ttk.Canvas(left_panel_container, width=300)
        scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas that fills the entire width
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar with better proportions
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create the actual content in the scrollable frame
        self.create_input_panel(scrollable_frame)
        
        # Right panel - Results with flexible width
        right_panel = ttk.Frame(panel_container)
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.create_results_panel(right_panel)
        
        # Nutrition chart panel
        chart_panel = ttk.Frame(main_container, padding=10)
        chart_panel.pack(fill="x", pady=(15, 0))
        
        self.create_chart_panel(chart_panel)
        
        # Create status bar at the bottom
        status_frame = ttk.Frame(self.master, padding=(5, 2))
        status_frame.pack(side="bottom", fill="x")
        
        self.status_var = ttk.StringVar(value="Ready")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_bar.pack(side="left", fill="x", expand=True)
        
        # Create stats display at the bottom right
        self.stats_frame = ttk.Frame(status_frame)
        self.stats_frame.pack(side="right", padx=5)
        
        # Create progress bar at the bottom right
        self.progress_frame = ttk.Frame(status_frame)
        self.progress_frame.pack(side="right", padx=5)
        
        self.progress = ttk.Progressbar(
            self.progress_frame, 
            orient="horizontal", 
            length=100, 
            mode="determinate", 
            bootstyle="success-striped"
        )
        self.progress.pack(side="right")
        self.progress["value"] = 0
        
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
                                    if isinstance(great_grandchild, ttk.Canvas):
                                        # Resize the canvas width
                                        great_grandchild.configure(width=desired_width)
                                        
                                        # Also resize the scrollable frame inside
                                        for item_id in great_grandchild.find_all():
                                            if great_grandchild.type(item_id) == "window":
                                                great_grandchild.itemconfigure(item_id, width=desired_width)
        
    def create_input_panel(self, parent):
        """Create the input panel with nutrition preferences and health conditions"""
        # Title with better spacing
        ttk.Label(
            parent, 
            text="Nutrition & Health Profile", 
            font=("Helvetica", self.subheader_font_size, "bold"),
            bootstyle="primary"
        ).pack(anchor="w", pady=(0, 15))
        
        # Health conditions frame with improved styling
        conditions_frame = ttk.Labelframe(parent, text="Health Conditions", padding=10)
        conditions_frame.pack(fill="x", pady=(0, 10))
        
        # Create health condition checkboxes
        self.diabetes_var = ttk.BooleanVar(value=False)
        self.obesity_var = ttk.BooleanVar(value=False)
        self.hypertension_var = ttk.BooleanVar(value=False)
        self.cholesterol_var = ttk.BooleanVar(value=False)
        
        # Add tooltips with explanations
        diabetes_check = ttk.Checkbutton(
            conditions_frame, 
            text="Diabetes", 
            variable=self.diabetes_var,
            bootstyle="round-toggle"
        )
        diabetes_check.pack(anchor="w", padx=5, pady=5)
        self.create_tooltip(diabetes_check, "Recommendations for blood sugar management")
        
        obesity_check = ttk.Checkbutton(
            conditions_frame, 
            text="Obesity/Weight Management", 
            variable=self.obesity_var,
            bootstyle="round-toggle"
        )
        obesity_check.pack(anchor="w", padx=5, pady=5)
        self.create_tooltip(obesity_check, "Recommendations for weight management and calorie control")
        
        hypertension_check = ttk.Checkbutton(
            conditions_frame, 
            text="Hypertension (High Blood Pressure)", 
            variable=self.hypertension_var,
            bootstyle="round-toggle"
        )
        hypertension_check.pack(anchor="w", padx=5, pady=5)
        self.create_tooltip(hypertension_check, "Foods with lower sodium and heart-healthy nutrients")
        
        cholesterol_check = ttk.Checkbutton(
            conditions_frame, 
            text="High Cholesterol", 
            variable=self.cholesterol_var,
            bootstyle="round-toggle"
        )
        cholesterol_check.pack(anchor="w", padx=5, pady=5)
        self.create_tooltip(cholesterol_check, "Foods with better fat profile and heart-healthy nutrients")
        
        # Frame for nutritional inputs with better styling
        input_frame = ttk.Labelframe(parent, text="Target Nutritional Values", padding=10)
        input_frame.pack(fill="x", pady=10)
        
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
        health_frame = ttk.Labelframe(parent, text="Personal Information", padding=10)
        health_frame.pack(fill="x", pady=10)
        
        # Use grid with consistent spacing
        grid_padx = 5
        grid_pady = 8
        
        # Weight with validation
        ttk.Label(health_frame, text="Weight (kg):").grid(row=0, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.weight_var = ttk.DoubleVar(value=70.0)
        weight_entry = ttk.Entry(health_frame, textvariable=self.weight_var, width=10)
        weight_entry.grid(row=0, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        self.add_validation(weight_entry, "float")
        
        # Height with validation
        ttk.Label(health_frame, text="Height (cm):").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.height_var = ttk.DoubleVar(value=170.0)
        height_entry = ttk.Entry(health_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        self.add_validation(height_entry, "float")
        
        # Age with validation
        ttk.Label(health_frame, text="Age:").grid(row=2, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.age_var = ttk.IntVar(value=45)
        age_entry = ttk.Entry(health_frame, textvariable=self.age_var, width=10)
        age_entry.grid(row=2, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        self.add_validation(age_entry, "int")
        
        # Gender
        ttk.Label(health_frame, text="Gender:").grid(row=3, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.gender_var = ttk.StringVar(value="Male")
        gender_combo = ttk.Combobox(health_frame, textvariable=self.gender_var, values=['Male', 'Female', 'Other'], width=10)
        gender_combo.grid(row=3, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        
        # Add BMI calculation
        ttk.Label(health_frame, text="BMI:").grid(row=4, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.bmi_var = ttk.StringVar(value="Computing...")
        ttk.Label(health_frame, textvariable=self.bmi_var).grid(row=4, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        
        # Calculate BMI when weight or height changes
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        self.calculate_bmi()  # Initial calculation
        
        # Recommendation settings with better styling
        settings_frame = ttk.Labelframe(parent, text="Recommendation Settings", padding=10)
        settings_frame.pack(fill="x", pady=10)
        
        # Number of recommendations
        ttk.Label(settings_frame, text="Number of recommendations:").grid(row=0, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.num_recommendations_var = ttk.IntVar(value=10)
        ttk.Combobox(settings_frame, textvariable=self.num_recommendations_var, values=[5, 10, 15, 20, 25], width=5).grid(row=0, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        
        # Filter by category with tooltip
        ttk.Label(settings_frame, text="Filter by category:").grid(row=1, column=0, padx=grid_padx, pady=grid_pady, sticky="w")
        self.category_filter_var = ttk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_filter_var, width=15)
        self.category_combo.grid(row=1, column=1, padx=grid_padx, pady=grid_pady, sticky="w")
        self.create_tooltip(self.category_combo, "Select a specific food category or 'All' to show foods from all categories")
        
        # Get all available categories
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty and 'Category' in self.recommender.food_data.columns:
            categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
        
        # Help/Info button
        help_btn = ttk.Button(
            parent, 
            text="About NCDs", 
            command=self.show_ncd_info,
            bootstyle="info"
        )
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
                    bootstyle = "info"
                elif bmi < 25:
                    category = "Normal"
                    bootstyle = "success"
                elif bmi < 30:
                    category = "Overweight"
                    bootstyle = "warning"
                else:
                    category = "Obese"
                    bootstyle = "danger"
                    
                self.bmi_var.set(f"{bmi:.1f} ({category})")
                
                # Update obesity checkbox based on BMI if it's not explicitly set
                if not hasattr(self, 'obesity_manually_set') or not self.obesity_manually_set:
                    if bmi >= 30:  # Obese
                        self.obesity_var.set(True)
                    elif bmi < 25:  # Normal or underweight
                        self.obesity_var.set(False)
            else:
                self.bmi_var.set("Invalid input")
        except:
            self.bmi_var.set("Computing...")
    
    def show_ncd_info(self):
        """Show information about NCDs"""
        info_window = ttk.Toplevel(self.master)
        info_window.title("About Non-Communicable Diseases (NCDs)")
        info_window.geometry("600x500")
        info_window.transient(self.master)  # Make window modal
        
        # Add content in a scrollable frame
        main_frame = ttk.Frame(info_window, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        # Create scrollable canvas
        canvas = ttk.Canvas(main_frame)
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
        ttk.Label(
            scrollable_frame, 
            text="Understanding Non-Communicable Diseases (NCDs)", 
            font=("Helvetica", 16, "bold"),
            bootstyle="primary"
        ).pack(pady=(0, 15), anchor="w")
        
        # Diabetes info
        ttk.Label(
            scrollable_frame, 
            text="Diabetes", 
            font=("Helvetica", 12, "bold"),
            bootstyle="success"
        ).pack(pady=(10, 5), anchor="w")
        
        ttk.Label(
            scrollable_frame, 
            text="A chronic condition affecting how your body processes blood sugar (glucose). "
                 "Dietary recommendations include limiting sugar and simple carbohydrates, "
                 "choosing high-fiber foods, and maintaining consistency in carbohydrate intake.",
            wraplength=550, 
            justify="left"
        ).pack(anchor="w")
        
        # Obesity info
        ttk.Label(
            scrollable_frame, 
            text="Obesity", 
            font=("Helvetica", 12, "bold"),
            bootstyle="warning"
        ).pack(pady=(10, 5), anchor="w")
        
        ttk.Label(
            scrollable_frame, 
            text="A complex disease involving an excessive amount of body fat. Dietary recommendations "
                 "include controlling portion sizes, reducing calorie intake, choosing foods high in "
                 "protein and fiber for satiety, and limiting foods high in added sugars and fats.",
            wraplength=550, 
            justify="left"
        ).pack(anchor="w")
        
        # Hypertension info
        ttk.Label(
            scrollable_frame, 
            text="Hypertension (High Blood Pressure)", 
            font=("Helvetica", 12, "bold"),
            bootstyle="danger"
        ).pack(pady=(10, 5), anchor="w")
        
        ttk.Label(
            scrollable_frame, 
            text="A condition in which the force of the blood against the artery walls is too high. "
                 "Dietary recommendations include reducing sodium intake, increasing potassium-rich foods, "
                 "limiting alcohol, and following the DASH (Dietary Approaches to Stop Hypertension) eating plan.",
            wraplength=550, 
            justify="left"
        ).pack(anchor="w")
        
        # High Cholesterol info
        ttk.Label(
            scrollable_frame, 
            text="High Cholesterol", 
            font=("Helvetica", 12, "bold"),
            bootstyle="info"
        ).pack(pady=(10, 5), anchor="w")
        
        ttk.Label(
            scrollable_frame, 
            text="Occurs when you have too much cholesterol in your blood. Dietary recommendations include "
                 "reducing saturated and trans fats, increasing soluble fiber, choosing lean proteins, and "
                 "incorporating foods containing plant sterols/stanols.",
            wraplength=550, 
            justify="left"
        ).pack(anchor="w")
        
        # General recommendations
        ttk.Label(
            scrollable_frame, 
            text="General Dietary Recommendations", 
            font=("Helvetica", 12, "bold"),
            bootstyle="primary"
        ).pack(pady=(15, 5), anchor="w")
        
        ttk.Label(
            scrollable_frame, 
            text="• Eat plenty of fruits, vegetables, whole grains, and lean proteins\n"
                 "• Choose foods rich in fiber, vitamins, and minerals\n"
                 "• Limit processed foods, added sugars, and unhealthy fats\n"
                 "• Practice portion control and mindful eating\n"
                 "• Stay hydrated by drinking plenty of water\n"
                 "• Consult with healthcare professionals for personalized advice",
            justify="left"
        ).pack(anchor="w")
        
        # Close button
        ttk.Button(
            scrollable_frame, 
            text="Close", 
            command=info_window.destroy,
            bootstyle="secondary"
        ).pack(pady=20)
        
        # Handle mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set focus to the new window
        info_window.focus_set()
    
    def show_model_metrics(self):
        """Show Random Forest model performance metrics"""
        metrics_window = ttk.Toplevel(self.master)
        metrics_window.title("Random Forest Model Performance Metrics")
        metrics_window.geometry("800x600")
        metrics_window.transient(self.master)  # Make window modal
        
        # Add content in a scrollable frame
        main_frame = ttk.Frame(metrics_window, padding=15)
        main_frame.pack(fill="both", expand=True)
        
        # Create scrollable canvas
        canvas = ttk.Canvas(main_frame)
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
        ttk.Label(
            scrollable_frame, 
            text="Random Forest Model Performance Metrics", 
            font=("Helvetica", 16, "bold"),
            bootstyle="primary"
        ).pack(pady=(0, 15), anchor="w")
        
        # Get model performance data
        performance = self.recommender.evaluate_model_performance()
        
        if isinstance(performance, dict) and performance:
            # Create metrics table
            metrics_frame = ttk.Frame(scrollable_frame)
            metrics_frame.pack(fill="x", pady=10)
            
            # Table headers
            ttk.Label(metrics_frame, text="Condition", font=("Helvetica", 11, "bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_frame, text="R² Score", font=("Helvetica", 11, "bold")).grid(row=0, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_frame, text="Model Type", font=("Helvetica", 11, "bold")).grid(row=0, column=2, padx=10, pady=5, sticky="w")
            
            # Add separator
            ttk.Separator(metrics_frame, orient="horizontal").grid(row=1, column=0, columnspan=3, sticky="ew", pady=5)
            
            # Fill table with data
            row = 2
            for condition, data in performance.items():
                condition_name = condition.replace('_', ' ')
                r2_score = data.get('r2_score', 0)
                is_synthetic = data.get('is_synthetic', False)
                model_type = "Synthetic (Rule-Based)" if is_synthetic else "Trained on Data"
                
                # Score color based on value
                if r2_score >= 0.7:
                    score_style = "success"
                elif r2_score >= 0.5:
                    score_style = "warning"
                else:
                    score_style = "danger"
                
                ttk.Label(metrics_frame, text=condition_name).grid(row=row, column=0, padx=10, pady=5, sticky="w")
                ttk.Label(metrics_frame, text=f"{r2_score:.4f}", bootstyle=score_style).grid(row=row, column=1, padx=10, pady=5, sticky="w")
                ttk.Label(metrics_frame, text=model_type).grid(row=row, column=2, padx=10, pady=5, sticky="w")
                row += 1
            
            # Explanation of metrics
            explanation_frame = ttk.Frame(scrollable_frame)
            explanation_frame.pack(fill="x", pady=20)
            
            ttk.Label(
                explanation_frame, 
                text="About R² Score", 
                font=("Helvetica", 14, "bold"),
                bootstyle="primary"
            ).pack(anchor="w", pady=(0, 10))
            
            ttk.Label(
                explanation_frame, 
                text="R² (coefficient of determination) measures how well the model predicts the target variable. "
                     "Values range from 0 to 1, where 1 indicates perfect prediction and 0 indicates that the model "
                     "is not performing better than a simple mean. Negative values (not shown here) would indicate "
                     "a model that performs worse than using a simple mean.",
                wraplength=700, 
                justify="left"
            ).pack(anchor="w")
            
            # Feature importance
            importance_frame = ttk.Frame(scrollable_frame)
            importance_frame.pack(fill="both", expand=True, pady=20)
            
            ttk.Label(
                importance_frame, 
                text="Feature Importance", 
                font=("Helvetica", 14, "bold"),
                bootstyle="primary"
            ).pack(anchor="w", pady=(0, 10))
            
            # Add tabs for each condition
            importance_tabs = ttk.Notebook(importance_frame)
            importance_tabs.pack(fill="both", expand=True)
            
            for condition in performance.keys():
                condition_name = condition.replace('_', ' ')
                condition_frame = ttk.Frame(importance_tabs)
                importance_tabs.add(condition_frame, text=condition_name)
                
                # Get feature importance data
                if hasattr(self.recommender, 'feature_importance') and condition in self.recommender.feature_importance:
                    importance = self.recommender.feature_importance[condition]
                    
                    # Create figure for this condition
                    fig = self.recommender.plot_feature_importance(condition)
                    
                    if fig:
                        # Embed the figure in the tab
                        canvas = FigureCanvasTkAgg(fig, master=condition_frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill="both", expand=True)
                    else:
                        ttk.Label(condition_frame, text=f"No feature importance data available for {condition_name}").pack(pady=50)
                else:
                    ttk.Label(condition_frame, text=f"No feature importance data available for {condition_name}").pack(pady=50)
        else:
            ttk.Label(scrollable_frame, text="No model performance data available").pack(pady=50)
        
        # Model comparison
        comparison_frame = ttk.Frame(scrollable_frame)
        comparison_frame.pack(fill="x", pady=20)
        
        ttk.Label(
            comparison_frame, 
            text="Random Forest vs. K-NN Comparison", 
            font=("Helvetica", 14, "bold"),
            bootstyle="primary"
        ).pack(anchor="w", pady=(0, 10))
        
        comparison_text = (
            "Random Forest models generally provide several advantages over K-NN for food recommendation systems:\n\n"
            "• Better interpretability through feature importance analysis\n"
            "• More robust to noise and missing values in nutrition data\n"
            "• Ability to capture complex, non-linear relationships in food nutrition data\n"
            "• Less sensitive to outliers in food composition\n"
            "• Better prediction performance with more sophisticated learning capabilities"
        )
        
        ttk.Label(
            comparison_frame, 
            text=comparison_text,
            wraplength=700, 
            justify="left"
        ).pack(anchor="w")
        
        # Close button
        ttk.Button(
            scrollable_frame, 
            text="Close", 
            command=metrics_window.destroy,
            bootstyle="secondary"
        ).pack(pady=20)
        
        # Handle mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Set focus to the new window
        metrics_window.focus_set()
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = ttk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(
                self.tooltip, 
                text=text, 
                bootstyle="secondary-inverse",
                padding=5
            )
            label.pack()
            
        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)
        
    def create_slider(self, parent, label_text, var_name, default_value, min_val, max_val, row):
        """Create a slider with label and value display"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=8, sticky="w")
        
        # Create slider variable
        slider_var = ttk.IntVar(value=default_value)
        setattr(self, var_name, slider_var)
        
        # Create frame for slider and value to ensure proper layout
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, columnspan=2, padx=5, pady=8, sticky="w")
        
        # Create slider with improved styling and proportional length
        slider = ttk.Scale(
            slider_frame, 
            from_=min_val, 
            to=max_val, 
            variable=slider_var, 
            orient="horizontal", 
            length=180,
            bootstyle="success"
        )
        slider.pack(side="left")
        
        # Value label with fixed width to prevent layout shifts
        value_label = ttk.Label(slider_frame, textvariable=slider_var, width=3)
        value_label.pack(side="left", padx=(5, 0))
        
    def create_results_panel(self, parent):
        """Create the results panel with recommendations"""
        # Title
        ttk.Label(
            parent, 
            text="Food Recommendations", 
            font=("Helvetica", self.subheader_font_size, "bold"),
            bootstyle="primary"
        ).pack(anchor="w", pady=(0, 10))
        
        # Create notebook (tabbed interface) for different views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        
        # Tab 1: Table View
        table_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(table_frame, text="Table View")
        
        # Create Treeview with scrollbar for recommendations
        columns = ('Name', 'Category', 'Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'Health Score')
        self.tree = ttk.Treeview(
            table_frame, 
            columns=columns, 
            show='headings', 
            height=15,
            bootstyle="light"
        )
        
        # Add scrollbars - both vertical and horizontal
        y_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        y_scrollbar.pack(side="right", fill="y")
        
        x_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        x_scrollbar.pack(side="bottom", fill="x")
        
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
        
        self.tree.pack(fill="both", expand=True, pady=5)
        
        # Tab 2: Health Analysis
        health_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(health_frame, text="Health Analysis")
        
        # Create a canvas for the health analysis chart
        self.health_canvas_frame = ttk.Frame(health_frame)
        self.health_canvas_frame.pack(fill="both", expand=True)
        
        # Tab 3: Nutrition Comparison
        comparison_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(comparison_frame, text="Nutrition Comparison")
        
        # Create a canvas for the comparison chart
        self.comparison_canvas_frame = ttk.Frame(comparison_frame)
        self.comparison_canvas_frame.pack(fill="both", expand=True)
        
        # Tab 4: Random Forest Analysis (New)
        rf_frame = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(rf_frame, text="Model Insights")
        
        # Create a canvas for the Random Forest analysis
        self.rf_canvas_frame = ttk.Frame(rf_frame)
        self.rf_canvas_frame.pack(fill="both", expand=True)
        
        # Add initial message
        ttk.Label(
            self.rf_canvas_frame, 
            text="Get recommendations to see Random Forest model insights", 
            font=("Helvetica", 12)
        ).pack(expand=True, pady=50)
        
        # Details panel below notebook with improved styling
        details_frame = ttk.Labelframe(parent, text="Nutritional Details")
        details_frame.pack(fill="x", pady=10)
        
        # Selected food details with better formatting
        self.details_text = ttk.Text(
            details_frame, 
            height=7, 
            width=40, 
            wrap="word",
            font=('Arial', self.calculate_font_size(10))
        )
        self.details_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.details_text.insert("end", "Select a food item to view detailed nutritional information")
        self.details_text.config(state="disabled")
        
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
        self.nutrition_chart.get_tk_widget().pack(fill="both", expand=True)
        
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
            msg = ttk.Label(
                self.comparison_canvas_frame, 
                text="Select at least two food items in the table view to compare", 
                font=('Helvetica', 12)
            )
            msg.pack(expand=True, pady=50)
            return
            
        # Create a Figure for the comparison chart
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for comparison - we'll compare selected items in the treeview
        selected_items = self.tree.selection()
        if not selected_items or len(selected_items) < 2:
            msg = ttk.Label(
                self.comparison_canvas_frame, 
                text="Select at least two food items in the table view to compare", 
                font=('Helvetica', 12)
            )
            msg.pack(expand=True, pady=50)
            return
            
        # Get selected food items
        selected_foods = []
        for item_id in selected_items:
            item_values = self.tree.item(item_id, 'values')
            if item_values:
                food_name = item_values[0]
                for rec in recommendations:
                    if rec['Name'] == food_name:
                        selected_foods.append(rec)
                        break
        
        # Prepare data for chart
        nutrients = ['Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat']
        
        # Position of bars on x-axis
        x = np.arange(len(nutrients))
        width = 0.8 / len(selected_foods)  # Width of bars adjusted for number of foods
        
        # Plot bars for each food
        for i, food in enumerate(selected_foods):
            values = [food.get(nutrient, 0) for nutrient in nutrients]
            pos = x - 0.4 + (i + 0.5) * width  # Position bars side by side
            ax.bar(pos, values, width=width, label=food['Name'][:15])  # Truncate long names
        
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
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add explanation text
        explanation = "This chart compares the nutritional content of selected foods. Select multiple items in the Table View to compare."
        ttk.Label(
            self.comparison_canvas_frame, 
            text=explanation, 
            bootstyle="secondary"
        ).pack(pady=5)
        
    def update_rf_analysis_chart(self, recommendations):
        """Update the Random Forest model analysis chart in the fourth tab"""
        # Clear the previous chart if it exists
        for widget in self.rf_canvas_frame.winfo_children():
            widget.destroy()
            
        if not recommendations:
            # Show a message if no recommendations
            msg = ttk.Label(
                self.rf_canvas_frame, 
                text="No data available for model analysis", 
                font=('Helvetica', 12)
            )
            msg.pack(expand=True, pady=50)
            return
            
        # Create a Figure for the feature importance chart
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Create two subplots: feature importance and prediction confidence
        ax1 = fig.add_subplot(211)  # Feature importance
        ax2 = fig.add_subplot(212)  # Prediction confidence
        
        # 1. Feature importance visualization - grab from the first selected condition
        conditions = []
        if self.diabetes_var.get():
            conditions.append('Diabetes')
        if self.obesity_var.get():
            conditions.append('Obesity')
        if self.hypertension_var.get():
            conditions.append('Hypertension')
        if self.cholesterol_var.get():
            conditions.append('High_Cholesterol')
            
        if not conditions:
            conditions = ['Diabetes']  # Default to show diabetes model if none selected
            
        condition = conditions[0]  # Use the first selected condition
        
        # Get feature importance data for this condition
        if hasattr(self.recommender, 'feature_importance') and condition in self.recommender.feature_importance:
            importance = self.recommender.feature_importance[condition]
            
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10 features
            top_features = sorted_features[:10]
            
            # Extract features and values
            features = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            
            # Plot horizontal bars
            bars = ax1.barh(features, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))))
            
            # Set title and labels
            ax1.set_title(f'Top Features for {condition.replace("_", " ")} Predictions', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Feature Importance')
            ax1.invert_yaxis()  # Display the most important feature at the top
            
            # Add value annotations
            for bar in bars:
                width = bar.get_width()
                ax1.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', va='center', fontsize=8)
        else:
            ax1.text(0.5, 0.5, f'No feature importance data available for {condition}', 
                   ha='center', va='center', fontsize=12)
        
        # 2. Prediction confidence visualization
        # Get the top 10 recommendations to visualize
        top_foods = recommendations[:10]
        food_names = [rec['Name'][:20] for rec in top_foods]  # Truncate long names
        
        # Get the suitable conditions for each food
        suitable_conditions = {
            'Diabetes': [], 
            'Obesity': [], 
            'Hypertension': [], 
            'High_Cholesterol': []
        }
        
        # Extract suitability scores
        for i, rec in enumerate(top_foods):
            for cond in conditions:
                if cond in rec.get('Suitable_For', []):
                    suitable_conditions[cond].append(i)
        
        # Create a scatter plot with colored points for suitability
        for i, food in enumerate(top_foods):
            # Default to gray if no conditions are met
            color = 'gray'
            marker = 'o'
            size = 50
            
            # Set color based on suitability for selected conditions
            is_suitable = False
            for cond in conditions:
                if cond in food.get('Suitable_For', []):
                    is_suitable = True
            
            if is_suitable:
                color = 'green'
                marker = '*'
                size = 100
            
            # Plot the point with confidence score (inverse of combined score)
            confidence = max(0, 1 - food.get('Combined_Score', 0) / 5)  # Normalize to 0-1
            ax2.scatter(confidence, i, color=color, marker=marker, s=size, 
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add vertical line at 0.5 confidence threshold
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        
        # Set title and labels
        ax2.set_title('Random Forest Prediction Confidence', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Confidence Score (Higher is Better)')
        ax2.set_yticks(range(len(food_names)))
        ax2.set_yticklabels(food_names)
        ax2.set_xlim(0, 1)
        
        # Add grid for better readability
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, 
                 label='Suitable for Selected Conditions'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, 
                 label='Not Suitable')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        # Adjust layout
        fig.tight_layout()
        
        # Embed the figure in the frame
        canvas = FigureCanvasTkAgg(fig, self.rf_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        
        # Add explanation text
        explanation = (
            "The top chart shows the most important features used by the Random Forest model. "
            "The bottom chart shows the model's confidence in its recommendations, with stars indicating foods "
            "suitable for selected health conditions."
        )
        
        ttk.Label(
            self.rf_canvas_frame, 
            text=explanation, 
            wraplength=700,
            bootstyle="secondary"
        ).pack(pady=5)
        
    def update_health_analysis_chart(self, recommendations):
        """Update the health analysis chart in the second tab"""
        # Clear the previous chart if it exists
        for widget in self.health_canvas_frame.winfo_children():
            widget.destroy()
            
        if not recommendations:
            # Show a message if no recommendations
            msg = ttk.Label(
                self.health_canvas_frame, 
                text="No data available for analysis", 
                font=('Helvetica', 12)
            )
            msg.pack(expand=True, pady=50)
            return
            
        # Create a Figure for the heatmap
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Prepare data for heatmap
        max_foods_to_display = 10  # Limit number of foods to display
        food_names = [rec['Name'] for rec in recommendations[:max_foods_to_display]]
        conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        condition_labels = ['Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol']
        
        # Create data matrix
        data = np.zeros((len(food_names), len(conditions)))
        for i, rec in enumerate(recommendations[:max_foods_to_display]):
            for j, cond in enumerate(conditions):
                data[i, j] = rec['Condition_Scores'].get(cond, 0)
        
        # Create heatmap with improved colors
        # Use a custom colormap that's colorblind-friendly
        cmap = plt.cm.YlOrRd
        
        # Create heatmap with improved formatting
        im = sns.heatmap(data, ax=ax, cmap=cmap, linewidths=0.5, 
                      xticklabels=condition_labels, yticklabels=food_names, 
                      annot=True, fmt=".1f", annot_kws={"size": 9})
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Add colorbar label
        cbar = im.collections[0].colorbar
        cbar.set_label('Score (Lower is Better)', rotation=270, labelpad=15)
        
        # Add suitable markers
        # Check which foods are suitable for which conditions
        for i, rec in enumerate(recommendations[:max_foods_to_display]):
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
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add a legend for the checkmark with better styling
        legend_frame = ttk.Frame(self.health_canvas_frame)
        legend_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(legend_frame, text="Legend: ", font=('Helvetica', 10, 'bold')).pack(side="left")
        ttk.Label(legend_frame, text="✓", foreground='green', font=('Helvetica', 12, 'bold')).pack(side="left")
        ttk.Label(legend_frame, text=" = Suitable for this condition").pack(side="left")
        
        # Add explanation of scores
        explanation = ("Lower scores indicate better suitability for the condition. "
                      "Foods with scores ≤ 2 and meeting nutrient criteria are marked as suitable (✓).")
        ttk.Label(
            self.health_canvas_frame, 
            text=explanation, 
            bootstyle="secondary"
        ).pack(pady=5)
        
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
                colors = ['#2a7b9b', '#3daf85', '#d95252']  # Updated color scheme
                
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
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_names)))
                    
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
                
                # 3. Health condition scores chart with improved styling
                conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
                condition_labels = ['Diabetes', 'Obesity', 'Hypertension', 'Cholesterol']
                avg_scores = []
                
                for condition in conditions:
                    scores = [rec['Condition_Scores'].get(condition, 0) for rec in recommendations]
                    avg_scores.append(sum(scores) / len(scores) if scores else 0)
                
                if any(score > 0 for score in avg_scores):
                    # Use a color palette that provides good contrast
                    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(condition_labels)))
                    
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
                
                # 4. Suitable conditions chart with improved visualization
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
        self.details_text.config(state="normal")
        self.details_text.delete(1.0, "end")
        
        # Get food name from the first column
        food_name = values[0]
        
        # Get more detailed information if available
        food_details = self.recommender.get_food_details(food_name)
        
        if food_details:
            # Format a more comprehensive details text with tags for styling
            self.details_text.insert("end", f"{food_name}\n", "title")
            self.details_text.insert("end", f"Category: {values[1]}\n\n")
            
            # Nutritional content section
            self.details_text.insert("end", "Nutritional Content:\n", "subtitle")
            
            # Format each nutrient with appropriate styling
            energy = float(values[2])
            self.details_text.insert("end", f"• Energy: {energy:.0f} kcal")
            if energy > 300:
                self.details_text.insert("end", " (high)\n", "warning")
            elif energy < 100:
                self.details_text.insert("end", " (low)\n", "good")
            else:
                self.details_text.insert("end", "\n")
                
            protein = float(values[3])
            self.details_text.insert("end", f"• Protein: {protein:.1f}g")
            if protein >= 10:
                self.details_text.insert("end", " (good source)\n", "good")
            else:
                self.details_text.insert("end", "\n")
                
            carbs = float(values[4])
            self.details_text.insert("end", f"• Carbohydrates: {carbs:.1f}g\n")
            
            sugar = float(values[5])
            self.details_text.insert("end", f"• Sugar: {sugar:.1f}g")
            if sugar > 15:
                self.details_text.insert("end", " (high)\n", "warning")
            elif sugar <= 5:
                self.details_text.insert("end", " (low)\n", "good")
            else:
                self.details_text.insert("end", "\n")
                
            fiber = float(values[6])
            self.details_text.insert("end", f"• Fiber: {fiber:.1f}g")
            if fiber >= 5:
                self.details_text.insert("end", " (high)\n", "good")
            elif fiber >= 3:
                self.details_text.insert("end", " (good source)\n", "good")
            else:
                self.details_text.insert("end", "\n")
                
            fat = float(values[7])
            self.details_text.insert("end", f"• Fat: {fat:.1f}g\n")
            
            # Health ratings section if available
            if any(condition in food_details for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol']):
                self.details_text.insert("end", "\nHealth Ratings:\n", "subtitle")
                
                # Add specific health condition ratings
                for condition, column in [
                    ('Diabetes', 'Diabetes'),
                    ('Obesity', 'Obesity'),
                    ('Hypertension', 'Hypertension'),
                    ('High Cholesterol', 'High Cholesterol')
                ]:
                    if column in food_details and not pd.isna(food_details[column]):
                        rating = float(food_details[column])
                        self.details_text.insert("end", f"• {condition}: {rating:.1f}")
                        
                        # Add visual indicator based on rating
                        if rating <= 1:
                            self.details_text.insert("end", " (suitable)\n", "good")
                        elif rating >= 3:
                            self.details_text.insert("end", " (use caution)\n", "warning")
                        else:
                            self.details_text.insert("end", " (moderate)\n")
            
            # Add sodium information if available
            if 'Na(mg)' in food_details and not pd.isna(food_details['Na(mg)']):
                sodium = float(food_details['Na(mg)'])
                self.details_text.insert("end", f"\nSodium: {sodium:.0f} mg")
                if sodium > 400:
                    self.details_text.insert("end", " (high)\n", "warning")
                elif sodium <= 140:
                    self.details_text.insert("end", " (low)\n", "good")
                else:
                    self.details_text.insert("end", "\n")
            
            # Add potassium information if available
            if 'K(mg)' in food_details and not pd.isna(food_details['K(mg)']):
                potassium = float(food_details['K(mg)'])
                self.details_text.insert("end", f"Potassium: {potassium:.0f} mg")
                if potassium > 300:
                    self.details_text.insert("end", " (good source)\n", "good")
                else:
                    self.details_text.insert("end", "\n")
                    
        else:
            # Basic information from the table with improved formatting
            self.details_text.insert("end", f"{food_name}\n", "title")
            self.details_text.insert("end", f"Category: {values[1]}\n\n")
            
            self.details_text.insert("end", "Nutritional Content:\n", "subtitle")
            self.details_text.insert("end", f"• Energy: {values[2]} kcal\n")
            self.details_text.insert("end", f"• Protein: {values[3]} g\n")
            self.details_text.insert("end", f"• Carbs: {values[4]} g\n")
            self.details_text.insert("end", f"• Sugar: {values[5]} g\n")
            self.details_text.insert("end", f"• Fiber: {values[6]} g\n")
            self.details_text.insert("end", f"• Fat: {values[7]} g\n")
            self.details_text.insert("end", f"• Health Score: {values[8]}")
        
        self.details_text.config(state="disabled")
        
        # If multiple items are selected, update the comparison chart
        if len(selected_items) >= 2:
            self.notebook.select(2)  # Switch to the comparison tab
            self.update_nutrition_comparison_chart(self.last_recommendations)
        
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
                ttk.Messagebox.show_error(
                    title="Error", 
                    message="Recommendation system not properly initialized",
                    parent=self.master
                )
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
                    
                    # Apply tag for suitable food items
                    is_suitable = False
                    if conditions:
                        for condition in conditions:
                            if condition in rec.get('Suitable_For', []):
                                is_suitable = True
                                break
                    
                    if is_suitable:
                        # Apply a tag for suitable items (could color them in the tree)
                        # Note: Would need to set up these tags with appropriate styles
                        pass
                
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
                
                # Update health analysis chart in second tab
                self.update_health_analysis_chart(recommendations)
                
                # Update RF analysis chart in fourth tab
                self.update_rf_analysis_chart(recommendations)
                
                # Clear the comparison chart initially
                for widget in self.comparison_canvas_frame.winfo_children():
                    widget.destroy()
                msg = ttk.Label(
                    self.comparison_canvas_frame, 
                    text="Select multiple food items in the Table View to compare", 
                    font=('Helvetica', 12)
                )
                msg.pack(expand=True, pady=50)
            else:
                ttk.Messagebox.show_info(
                    title="No Recommendations", 
                    message="No recommendations found matching your criteria.",
                    parent=self.master
                )
                self.status_var.set("No recommendations found")
                
                # Clear charts
                self.update_charts([])
                self.update_health_analysis_chart([])
                self.update_rf_analysis_chart([])
                
                # Clear comparison chart
                for widget in self.comparison_canvas_frame.winfo_children():
                    widget.destroy()
                msg = ttk.Label(
                    self.comparison_canvas_frame, 
                    text="No data available for comparison", 
                    font=('Helvetica', 12)
                )
                msg.pack(expand=True, pady=50)
                
        except Exception as e:
            ttk.Messagebox.show_error(
                title="Error", 
                message=f"An error occurred: {str(e)}",
                parent=self.master
            )
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
        self.details_text.config(state="normal")
        self.details_text.delete(1.0, "end")
        self.details_text.insert("end", "Select a food item to view detailed nutritional information")
        self.details_text.config(state="disabled")
        
        # Reset charts
        self.update_charts([])
        self.update_health_analysis_chart([])
        self.update_rf_analysis_chart([])
        
        # Clear comparison chart
        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()
        msg = ttk.Label(
            self.comparison_canvas_frame, 
            text="Select multiple food items in the Table View to compare", 
            font=('Helvetica', 12)
        )
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
            
            # Add condition-friendly counts
            condition_friendly = []
            for condition, count in stats.get('condition_friendly', {}).items():
                if count > 0:
                    condition_friendly.append(f"{count} {condition.replace('_', ' ')}")
            
            # Create a frame with better layout
            if condition_friendly:
                stats_label = ttk.Label(
                    self.stats_frame, 
                    text=stats_text,
                    bootstyle="secondary"
                )
                stats_label.pack(side="left")
                
                separator = ttk.Label(self.stats_frame, text=" | ", bootstyle="secondary")
                separator.pack(side="left")
                
                friendly_label = ttk.Label(
                    self.stats_frame, 
                    text="Condition-friendly items: " + 
                         ", ".join(condition_friendly),
                    bootstyle="secondary"
                )
                friendly_label.pack(side="left")
            else:
                # Simple version if no condition data
                stats_label = ttk.Label(self.stats_frame, text=stats_text, bootstyle="secondary")
                stats_label.pack(side="left")
            
            # Add loading time if available
            if 'loading_time' in stats and stats['loading_time'] > 0:
                time_label = ttk.Label(
                    self.stats_frame, 
                    text=f" | Loaded in {stats['loading_time']:.2f}s", 
                    bootstyle="secondary"
                )
                time_label.pack(side="left")
                
            # Add model type indicator
            separator = ttk.Label(self.stats_frame, text=" | ", bootstyle="secondary")
            separator.pack(side="left")
            
            model_label = ttk.Label(
                self.stats_frame, 
                text="Model: Random Forest", 
                bootstyle="info"
            )
            model_label.pack(side="left")
                
        except Exception as e:
            print(f"Error updating stats display: {e}")
            ttk.Label(self.stats_frame, text="Stats unavailable", bootstyle="danger").pack(side="left")


# Main function to run the application
def main():
    # Set up basic logging to console
    print("Starting Food Recommendation System with Random Forest...")
    
    # Create the root window with ttkbootstrap
    root = ttk.Window(
        title="Food Recommendation System for NCDs",
        themename="cosmo",  # Options: cosmo, flatly, litera, minty, lumen, sandstone, yeti, pulse, etc.
        size=(1200, 750),
        position=(100, 50),
        minsize=(900, 600),
        resizable=(True, True)
    )
    
    # Create a modern splash screen
    splash_frame = ttk.Frame(root)
    splash_frame.place(x=0, y=0, relwidth=1, relheight=1)
    
    # Add background color
    splash_bg = ttk.Label(splash_frame, bootstyle="primary", text="")
    splash_bg.place(relwidth=1, relheight=1)
    
    # Create content frame for better contrast
    content_frame = ttk.Frame(splash_frame, padding=20)
    content_frame.place(relx=0.5, rely=0.5, anchor="center", width=500, height=400)
    
    # Create splash screen elements with better styling
    title_label = ttk.Label(
        content_frame,
        text="Food Recommendation System",
        font=("Helvetica", 22, "bold"),
        bootstyle="primary"
    )
    title_label.pack(pady=(40, 10))
    
    subtitle_label = ttk.Label(
        content_frame,
        text="For Non-Communicable Diseases (NCDs)",
        font=("Helvetica", 14),
        bootstyle="secondary"
    )
    subtitle_label.pack(pady=(0, 10))
    
    model_label = ttk.Label(
        content_frame,
        text="Random Forest Model",
        font=("Helvetica", 16, "bold"),
        bootstyle="info"
    )
    model_label.pack(pady=(0, 30))
    
    # Progress meter
    progress_frame = ttk.Frame(content_frame)
    progress_frame.pack(pady=20)
    
    progress = ttk.Progressbar(
        progress_frame, 
        length=350, 
        mode="indeterminate",
        bootstyle="success-striped"
    )
    progress.pack()
    
    progress.start()
    
    # Status text
    status_var = ttk.StringVar(value="Initializing system...")
    status_label = ttk.Label(
        content_frame, 
        textvariable=status_var, 
        font=('Helvetica', 11),
        bootstyle="secondary"
    )
    status_label.pack(pady=20)
    
    # Add information about supported conditions
    conditions_frame = ttk.Frame(content_frame)
    conditions_frame.pack(pady=20)
    
    # Each condition in its own label with appropriate color
    conditions = [
        ("Diabetes", "success"),
        ("Obesity", "warning"),
        ("Hypertension", "danger"),
        ("High Cholesterol", "info")
    ]
    
    for i, (condition, style) in enumerate(conditions):
        condition_label = ttk.Label(
            conditions_frame, 
            text=f"• {condition}",
            font=('Helvetica', 11), 
            bootstyle=style
        )
        condition_label.grid(row=i//2, column=i%2, padx=20, pady=5, sticky="w")
    
    # Add a footer with credits
    footer_text = "Developed by Surat Lawdi - Prince of Songkla University"
    footer_label = ttk.Label(
        content_frame, 
        text=footer_text, 
        font=('Helvetica', 8),
        bootstyle="secondary"
    )
    footer_label.pack(side="bottom", pady=20)
    
    def update_status(message):
        """Update status message and ensure UI updates"""
        print(f"Status: {message}")  # Debug print
        status_var.set(message)
        root.update_idletasks()  # Force update of the UI
    
    def initialize_app():
        """Initialize the application"""
        try:
            # Create the food recommender
            update_status("Loading food database...")
            recommender = FoodRecommendationSystemRF(update_status)
            
            update_status("Training Random Forest models...")
            time.sleep(0.5)  # Small delay for visual effect
            
            update_status("Building user interface...")
            # Create main app UI
            app = FoodRecommenderUI(root, recommender)
            
            # Remove splash screen
            update_status("Ready! Random Forest models loaded successfully.")
            root.after(1500, splash_frame.destroy)  # Destroy splash frame after delay
            
        except Exception as e:
            # Show error on splash screen
            update_status(f"Error initializing: {str(e)}")
            print(f"Initialization error: {e}")  # Debug print
            
            # Add a retry button
            retry_btn = ttk.Button(
                content_frame, 
                text="Retry", 
                command=lambda: initialize_app(),
                bootstyle="warning"
            )
            retry_btn.pack(pady=10)
    
    # Schedule the initialization to happen after the window is shown
    root.after(100, initialize_app)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()