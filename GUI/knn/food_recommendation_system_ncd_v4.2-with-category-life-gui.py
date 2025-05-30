import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set higher DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

@dataclass
class HealthProfile:
    """User health profile for nutritional calculations"""
    age: int
    gender: str  # 'Male', 'Female'
    weight: float  # kg
    height: float  # cm
    activity_level: str  # 'Sedentary', 'Light', 'Moderate', 'Active', 'Very Active'
    weight_goal: str  # 'Maintain', 'Lose', 'Gain'
    diabetes: bool = False
    obesity: bool = False
    hypertension: bool = False
    high_cholesterol: bool = False
    # Thai Traditional Medicine
    use_thai_element: bool = False
    birth_month: int = 1
    birth_year: int = 1990
    life_element: str = ""

@dataclass
class NutritionalTargets:
    """Calculated nutritional targets based on medical guidelines"""
    energy_kcal: float
    protein_g: float
    fat_g: float
    saturated_fat_g: float
    carbohydrates_g: float
    sugar_g: float
    fiber_g: float
    sodium_mg: float
    potassium_mg: float
    explanation: str
    thai_element_guidance: str = ""

class ThaiElementCalculator:
    """Thai Traditional Medicine Life Element Calculator"""
    
    # Thai elements and their characteristics
    ELEMENTS = {
        'ดิน': {
            'name_thai': 'ดิน',
            'name_english': 'Earth (Din)',
            'characteristics': 'Stable, grounded, digestive strength',
            'favorable_foods': ['rice', 'root vegetables', 'grains', 'beans'],
            'avoid_foods': ['too spicy', 'too cold', 'raw foods'],
            'nutritional_focus': 'warm, cooked foods with moderate spices'
        },
        'น้ำ': {
            'name_thai': 'น้ำ',
            'name_english': 'Water (Nam)',
            'characteristics': 'Cooling, fluid balance, kidney health',
            'favorable_foods': ['coconut', 'cucumber', 'watermelon', 'fish'],
            'avoid_foods': ['too salty', 'too hot', 'fried foods'],
            'nutritional_focus': 'cooling foods, adequate hydration, low sodium'
        },
        'ลม': {
            'name_thai': 'ลม',
            'name_english': 'Wind (Lom)',
            'characteristics': 'Movement, circulation, nervous system',
            'favorable_foods': ['ginger', 'garlic', 'herbs', 'warm soups'],
            'avoid_foods': ['cold drinks', 'raw foods', 'gas-producing foods'],
            'nutritional_focus': 'warm, easily digestible, anti-inflammatory'
        },
        'ไฟ': {
            'name_thai': 'ไฟ',
            'name_english': 'Fire (Fai)',
            'characteristics': 'Heat, metabolism, energy transformation',
            'favorable_foods': ['cooling herbs', 'bitter vegetables', 'mild foods'],
            'avoid_foods': ['very spicy', 'fried', 'alcohol', 'excessive heat'],
            'nutritional_focus': 'cooling, bitter tastes, moderate portions'
        }
    }
    
    def calculate_life_element(self, birth_month: int, birth_year: int) -> str:
        """Calculate Thai life element based on birth month and year"""
        # Simplified calculation based on traditional Thai astrology
        # This is a basic implementation - actual calculations may be more complex
        
        # Convert to Buddhist year for traditional calculation
        buddhist_year = birth_year + 543
        
        # Element cycle based on year (simplified 4-year cycle)
        year_cycle = buddhist_year % 4
        
        # Month influence (seasons in Thailand)
        if birth_month in [12, 1, 2]:  # Cool season
            season_modifier = 0
        elif birth_month in [3, 4, 5]:  # Hot season  
            season_modifier = 1
        elif birth_month in [6, 7, 8, 9]:  # Rainy season
            season_modifier = 2
        else:  # Late rainy/transition
            season_modifier = 3
        
        # Combine year cycle and season
        element_index = (year_cycle + season_modifier) % 4
        
        elements = ['ดิน', 'น้ำ', 'ลม', 'ไฟ']
        return elements[element_index]
    
    def get_element_info(self, element: str) -> dict:
        """Get detailed information about a life element"""
        return self.ELEMENTS.get(element, {})
    
    def get_dietary_guidance(self, element: str, health_conditions: list) -> str:
        """Get dietary guidance based on life element and health conditions"""
        element_info = self.get_element_info(element)
        if not element_info:
            return ""
        
        guidance = f"Thai Traditional Medicine - {element_info['name_english']}:\n"
        guidance += f"Characteristics: {element_info['characteristics']}\n\n"
        
        guidance += "Traditional Dietary Recommendations:\n"
        guidance += f"• Focus: {element_info['nutritional_focus']}\n"
        guidance += f"• Favorable: {', '.join(element_info['favorable_foods'])}\n"
        guidance += f"• Minimize: {', '.join(element_info['avoid_foods'])}\n\n"
        
        # Combine with modern health conditions
        if health_conditions:
            guidance += "Integration with Health Conditions:\n"
            
            if 'Diabetes' in health_conditions:
                if element == 'ดิน':
                    guidance += "• Earth + Diabetes: Focus on complex carbs, avoid refined sugars\n"
                elif element == 'น้ำ':
                    guidance += "• Water + Diabetes: Cooling foods help reduce inflammation\n"
                elif element == 'ลม':
                    guidance += "• Wind + Diabetes: Warm spices help circulation and metabolism\n"
                elif element == 'ไฟ':
                    guidance += "• Fire + Diabetes: Cool, bitter foods help balance blood sugar\n"
            
            if 'Hypertension' in health_conditions and element == 'น้ำ':
                guidance += "• Water element naturally supports low-sodium approach\n"
            
            if 'Obesity' in health_conditions and element == 'ไฟ':
                guidance += "• Fire element benefits from cooling, lighter foods for weight management\n"
        
        return guidance

class MedicalNutritionCalculator:
    """Medical-grade nutritional calculation engine based on established formulas"""
    
    # Activity level multipliers for TDEE calculation
    ACTIVITY_MULTIPLIERS = {
        'Sedentary': 1.2,      # Little to no exercise
        'Light': 1.375,        # Light exercise 1-3 days/week
        'Moderate': 1.55,      # Moderate exercise 3-5 days/week
        'Active': 1.725,       # Hard exercise 6-7 days/week
        'Very Active': 1.9     # Very hard exercise, physical job
    }
    
    # Weight goal calorie adjustments
    WEIGHT_GOAL_ADJUSTMENTS = {
        'Lose': -500,    # 500 kcal deficit for ~1 lb/week loss
        'Maintain': 0,   # No adjustment
        'Gain': 300      # 300 kcal surplus for lean weight gain
    }
    
    def calculate_bmr(self, profile: HealthProfile) -> float:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor equation"""
        if profile.gender == 'Male':
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
        else:  # Female
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161
        return bmr
    
    def calculate_tdee(self, profile: HealthProfile) -> float:
        """Calculate Total Daily Energy Expenditure"""
        bmr = self.calculate_bmr(profile)
        activity_multiplier = self.ACTIVITY_MULTIPLIERS[profile.activity_level]
        tdee = bmr * activity_multiplier
        
        # Apply weight goal adjustments
        weight_adjustment = self.WEIGHT_GOAL_ADJUSTMENTS[profile.weight_goal]
        return tdee + weight_adjustment
    
    def calculate_protein_needs(self, profile: HealthProfile) -> float:
        """Calculate protein needs based on health conditions and goals"""
        # Base protein requirement
        if profile.diabetes or profile.weight_goal == 'Lose':
            # Higher protein for diabetes and weight loss (1.2-1.6 g/kg)
            protein_per_kg = 1.4
        elif profile.obesity:
            # Higher protein for obesity management
            protein_per_kg = 1.5
        else:
            # Standard recommendation (0.8-1.0 g/kg)
            protein_per_kg = 1.0
            
        return profile.weight * protein_per_kg
    
    def calculate_fat_limits(self, total_calories: float, profile: HealthProfile) -> Tuple[float, float]:
        """Calculate fat and saturated fat limits"""
        # Total fat: 20-35% of calories (using 30% as target)
        total_fat_calories = total_calories * 0.30
        total_fat_g = total_fat_calories / 9  # 9 kcal per gram of fat
        
        # Saturated fat limits based on health conditions
        if profile.high_cholesterol or profile.hypertension:
            # AHA recommendation: <6% of calories
            sat_fat_percent = 0.06
        else:
            # General recommendation: <7% of calories
            sat_fat_percent = 0.07
            
        sat_fat_calories = total_calories * sat_fat_percent
        sat_fat_g = sat_fat_calories / 9
        
        return total_fat_g, sat_fat_g
    
    def calculate_carbohydrate_needs(self, total_calories: float, protein_g: float, 
                                   fat_g: float, profile: HealthProfile) -> float:
        """Calculate carbohydrate needs based on health conditions"""
        # Calculate remaining calories after protein and fat
        protein_calories = protein_g * 4
        fat_calories = fat_g * 9
        remaining_calories = total_calories - protein_calories - fat_calories
        
        # Adjust for health conditions
        if profile.diabetes:
            # Lower carb percentage for diabetes (40-50% of total calories)
            carb_percent = 0.45
            carb_calories = total_calories * carb_percent
        else:
            # Use remaining calories for carbohydrates
            carb_calories = max(remaining_calories, total_calories * 0.45)
            
        return carb_calories / 4  # 4 kcal per gram of carbohydrate
    
    def calculate_sugar_limit(self, profile: HealthProfile) -> float:
        """Calculate added sugar limit based on WHO/AHA guidelines"""
        # WHO recommendation: <5% of total energy for NCDs
        if any([profile.diabetes, profile.obesity, profile.hypertension, profile.high_cholesterol]):
            return 25.0  # grams (strict limit for NCDs)
        else:
            return 50.0  # grams (general population)
    
    def calculate_fiber_needs(self, profile: HealthProfile) -> float:
        """Calculate fiber needs based on health conditions"""
        if profile.diabetes or profile.high_cholesterol:
            # Higher fiber for diabetes and cholesterol management
            return 35.0
        elif profile.hypertension:
            # DASH diet recommendation
            return 30.0
        else:
            # General recommendation
            return 25.0
    
    def calculate_sodium_limit(self, profile: HealthProfile) -> float:
        """Calculate sodium limit based on health conditions"""
        if profile.hypertension:
            # Strict limit for hypertension
            return 1500.0  # mg
        else:
            # General recommendation
            return 2300.0  # mg
    
    def calculate_nutritional_targets(self, profile: HealthProfile) -> NutritionalTargets:
        """Calculate complete nutritional targets based on medical guidelines"""
        # Calculate energy needs
        energy_kcal = self.calculate_tdee(profile)
        
        # Calculate macronutrients
        protein_g = self.calculate_protein_needs(profile)
        fat_g, sat_fat_g = self.calculate_fat_limits(energy_kcal, profile)
        carbs_g = self.calculate_carbohydrate_needs(energy_kcal, protein_g, fat_g, profile)
        
        # Calculate micronutrients and limits
        sugar_g = self.calculate_sugar_limit(profile)
        fiber_g = self.calculate_fiber_needs(profile)
        sodium_mg = self.calculate_sodium_limit(profile)
        potassium_mg = 3500.0  # General recommendation for all conditions
        
        # Generate explanation
        explanation = self._generate_explanation(profile, energy_kcal)
        
        # Generate Thai element guidance if enabled
        thai_element_guidance = ""
        if profile.use_thai_element:
            thai_calculator = ThaiElementCalculator()
            element = thai_calculator.calculate_life_element(profile.birth_month, profile.birth_year)
            
            # Get active health conditions
            active_conditions = []
            if profile.diabetes:
                active_conditions.append('Diabetes')
            if profile.obesity:
                active_conditions.append('Obesity')
            if profile.hypertension:
                active_conditions.append('Hypertension')
            if profile.high_cholesterol:
                active_conditions.append('High_Cholesterol')
            
            thai_element_guidance = thai_calculator.get_dietary_guidance(element, active_conditions)
        
        return NutritionalTargets(
            energy_kcal=energy_kcal,
            protein_g=protein_g,
            fat_g=fat_g,
            saturated_fat_g=sat_fat_g,
            carbohydrates_g=carbs_g,
            sugar_g=sugar_g,
            fiber_g=fiber_g,
            sodium_mg=sodium_mg,
            potassium_mg=potassium_mg,
            explanation=explanation,
            thai_element_guidance=thai_element_guidance
        )
    
    def _generate_explanation(self, profile: HealthProfile, energy_kcal: float) -> str:
        """Generate explanation for the calculated targets"""
        explanations = []
        
        # BMR and activity explanation
        bmr = self.calculate_bmr(profile)
        explanations.append(f"Your BMR (Mifflin-St Jeor): {bmr:.0f} kcal/day")
        explanations.append(f"Activity level ({profile.activity_level}): {self.ACTIVITY_MULTIPLIERS[profile.activity_level]}x multiplier")
        
        # Weight goal explanation
        if profile.weight_goal == 'Lose':
            explanations.append("Weight loss goal: 500 kcal deficit applied")
        elif profile.weight_goal == 'Gain':
            explanations.append("Weight gain goal: 300 kcal surplus applied")
        
        # Health condition modifications
        conditions = []
        if profile.diabetes:
            conditions.append("Diabetes: Lower carbs (45%), strict sugar limit (<25g)")
        if profile.obesity:
            conditions.append("Obesity: Higher protein (1.5g/kg)")
        if profile.hypertension:
            conditions.append("Hypertension: Low sodium (<1500mg), DASH principles")
        if profile.high_cholesterol:
            conditions.append("High Cholesterol: Low saturated fat (<6%), high fiber")
            
        if conditions:
            explanations.extend(conditions)
        else:
            explanations.append("No specific health conditions: Standard guidelines applied")
            
        return "\n".join(explanations)

class HealthDrivenKNN:
    """KNN implementation optimized for health-driven food recommendations"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.calculator = MedicalNutritionCalculator()
        self.scaler = StandardScaler()
        self.knn_model = None
        self.food_data = pd.DataFrame()
        self.feature_columns = []
        self.health_weights = {}
        
        # Load and prepare data
        self.load_thai_food_data()
        self.prepare_health_features()
        self.train_knn_model()
    
    def update_status(self, message: str):
        """Update status if callback provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_thai_food_data(self):
        """Load and combine Thai food datasets"""
        try:
            dataset_folder = './datasets'  # Adjust path as needed
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                # Create sample data if no files found
                self.create_sample_data()
                return
            
            dataframes = []
            for file_path in csv_files:
                try:
                    filename = os.path.basename(file_path)
                    category = os.path.splitext(filename)[0].replace('_', ' ').title()
                    
                    self.update_status(f"Loading {category}...")
                    df = pd.read_csv(file_path)
                    
                    if 'Category' not in df.columns:
                        df['Category'] = category
                    
                    dataframes.append(df)
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.update_status(f"Loaded {len(self.food_data)} food items")
            else:
                self.create_sample_data()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample Thai food data for testing"""
        self.update_status("Creating sample Thai food data...")
        
        sample_foods = [
            # Thai dishes with realistic nutritional values
            ['ข้าวผัด', 'Fried Rice', 'Rice', 520, 15, 18, 75, 2, 3, 850, 200, 45, 0],
            ['ต้มยำกุ้ง', 'Tom Yum Goong', 'Soup', 180, 20, 8, 12, 6, 2, 1200, 300, 15, 120],
            ['ส้มตำ', 'Som Tam', 'Salad', 120, 3, 2, 25, 15, 5, 800, 400, 8, 0],
            ['แกงเขียวหวาน', 'Green Curry', 'Curry', 280, 18, 20, 15, 8, 3, 950, 450, 35, 80],
            ['ผัดไทย', 'Pad Thai', 'Noodle', 450, 12, 15, 68, 12, 3, 1100, 250, 25, 85],
            ['มะม่วงข้าวเหนียว', 'Mango Sticky Rice', 'Dessert', 380, 6, 8, 78, 45, 2, 150, 180, 12, 0],
            ['ไก่ย่าง', 'Grilled Chicken', 'Meat', 250, 35, 12, 0, 0, 0, 420, 350, 2, 95],
            ['ยำวุ้นเส้น', 'Glass Noodle Salad', 'Salad', 180, 8, 3, 35, 8, 2, 680, 320, 5, 45],
        ]
        
        columns = ['Thai_Name', 'English_Name', 'Category', 'Energy(kcal) by calculation', 
                  'Protein(g)', 'Fat(g)', 'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 
                  'FIBTG (g) Dietary fibre', 'Na(mg)', 'K(mg)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol']
        
        self.food_data = pd.DataFrame(sample_foods, columns=columns)
        self.update_status(f"Created {len(self.food_data)} sample food items")
    
    def prepare_health_features(self):
        """Prepare features optimized for health-driven recommendations"""
        self.update_status("Preparing health-optimized features...")
        
        # Define critical nutritional features
        base_features = [
            'Energy(kcal) by calculation', 'Protein(g)', 'Fat(g)', 
            'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 'FIBTG (g) Dietary fibre'
        ]
        
        # Add optional features if available
        optional_features = ['Na(mg)', 'K(mg)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol']
        
        available_features = []
        for feature in base_features + optional_features:
            if feature in self.food_data.columns:
                available_features.append(feature)
        
        self.feature_columns = available_features
        
        # Handle missing values
        for col in self.feature_columns:
            self.food_data[col] = pd.to_numeric(self.food_data[col], errors='coerce')
            
            # Fill missing values with median or reasonable defaults
            if col == 'Na(mg)':
                self.food_data[col].fillna(500, inplace=True)  # Moderate sodium default
            elif col == 'K(mg)':
                self.food_data[col].fillna(200, inplace=True)  # Moderate potassium default
            elif col == 'FASAT (g) Saturated FA':
                # Estimate as 30% of total fat if missing
                self.food_data[col].fillna(self.food_data['Fat(g)'] * 0.3, inplace=True)
            elif col == 'CHOLE(mg) Cholesterol':
                self.food_data[col].fillna(0, inplace=True)  # Default to 0 for plant foods
            else:
                self.food_data[col].fillna(self.food_data[col].median(), inplace=True)
        
        # Calculate health scores for each condition
        self.calculate_health_scores()
        
        self.update_status(f"Prepared {len(self.feature_columns)} features for health analysis")
    
    def calculate_health_scores(self):
        """Calculate health suitability scores for each food item"""
        # Initialize health scores
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = 0.0
        
        # Initialize Thai element scores
        for element in ['ดิน', 'น้ำ', 'ลม', 'ไฟ']:
            self.food_data[f'Element_{element}_Score'] = 0.0
        
        for idx, food in self.food_data.iterrows():
            # Existing health condition scoring
            diabetes_score = 0
            diabetes_score += max(0, food.get('SUGAR(g)', 0) - 10) * 2  # Penalty for high sugar
            diabetes_score += max(0, food.get('CHOCDF (g) Carbohydrate', 0) - 30) * 0.5  # Penalty for high carbs
            diabetes_score -= min(food.get('FIBTG (g) Dietary fibre', 0), 10) * 0.5  # Bonus for fiber
            self.food_data.at[idx, 'Diabetes_Score'] = max(0, diabetes_score)
            
            # Obesity score (lower is better)
            obesity_score = 0
            obesity_score += max(0, food.get('Energy(kcal) by calculation', 0) - 300) * 0.01  # Penalty for high calories
            obesity_score += max(0, food.get('Fat(g)', 0) - 15) * 0.3  # Penalty for high fat
            obesity_score -= min(food.get('Protein(g)', 0), 25) * 0.2  # Bonus for protein
            obesity_score -= min(food.get('FIBTG (g) Dietary fibre', 0), 10) * 0.3  # Bonus for fiber
            self.food_data.at[idx, 'Obesity_Score'] = max(0, obesity_score)
            
            # Hypertension score (lower is better)
            hypertension_score = 0
            sodium = food.get('Na(mg)', 500)
            hypertension_score += max(0, sodium - 400) * 0.01  # Penalty for high sodium
            potassium = food.get('K(mg)', 200)
            hypertension_score -= min(potassium, 500) * 0.002  # Bonus for potassium
            self.food_data.at[idx, 'Hypertension_Score'] = max(0, hypertension_score)
            
            # High Cholesterol score (lower is better)
            cholesterol_score = 0
            sat_fat = food.get('FASAT (g) Saturated FA', food.get('Fat(g)', 0) * 0.3)
            cholesterol_score += sat_fat * 0.5  # Penalty for saturated fat
            cholesterol = food.get('CHOLE(mg) Cholesterol', 0)
            cholesterol_score += max(0, cholesterol - 50) * 0.02  # Penalty for dietary cholesterol
            cholesterol_score -= min(food.get('FIBTG (g) Dietary fibre', 0), 10) * 0.3  # Bonus for fiber
            self.food_data.at[idx, 'High_Cholesterol_Score'] = max(0, cholesterol_score)
            
            # Thai Element Scoring (traditional approach)
            category = food.get('Category', '').lower()
            
            # Earth (ดิน) - prefers grains, roots, cooked foods
            earth_score = 5.0  # neutral base
            if any(word in category for word in ['rice', 'grain', 'root']):
                earth_score -= 2.0  # bonus for earth-friendly foods
            if 'raw' in category or food.get('SUGAR(g)', 0) > 20:
                earth_score += 1.0  # penalty for raw or very sweet
            self.food_data.at[idx, 'Element_ดิน_Score'] = max(0, earth_score)
            
            # Water (น้ำ) - prefers cooling, low sodium foods  
            water_score = 5.0
            sodium = food.get('Na(mg)', 500)
            if sodium < 300:  # low sodium bonus
                water_score -= 2.0
            if sodium > 800:  # high sodium penalty
                water_score += 2.0
            if any(word in category for word in ['fruit', 'coconut', 'cucumber']):
                water_score -= 1.0  # cooling foods bonus
            self.food_data.at[idx, 'Element_น้ำ_Score'] = max(0, water_score)
            
            # Wind (ลม) - prefers warm, digestible foods
            wind_score = 5.0
            if any(word in category for word in ['soup', 'curry', 'warm']):
                wind_score -= 2.0  # warm foods bonus
            if category in ['salad'] and food.get('SUGAR(g)', 0) < 5:
                wind_score += 1.0  # raw foods penalty
            if food.get('FIBTG (g) Dietary fibre', 0) > 5:
                wind_score -= 1.0  # good digestion bonus
            self.food_data.at[idx, 'Element_ลม_Score'] = max(0, wind_score)
            
            # Fire (ไฟ) - prefers cooling, bitter, mild foods
            fire_score = 5.0
            spicy_categories = ['curry', 'spicy']
            if any(word in category for word in spicy_categories):
                fire_score += 1.5  # penalty for hot/spicy foods
            if any(word in category for word in ['vegetable', 'herb', 'bitter']):
                fire_score -= 1.5  # cooling foods bonus
            if food.get('Fat(g)', 0) > 20:  # high fat penalty for fire
                fire_score += 1.0
            self.food_data.at[idx, 'Element_ไฟ_Score'] = max(0, fire_score)
    
    def train_knn_model(self):
        """Train KNN model with health-optimized features"""
        if len(self.food_data) == 0:
            self.update_status("No data available for training")
            return
        
        self.update_status("Training health-driven KNN model...")
        
        # Prepare feature matrix
        X = self.food_data[self.feature_columns].values
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train KNN model
        n_neighbors = min(10, len(self.food_data))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='ball_tree',
            metric='manhattan'  # Manhattan distance works well for nutritional data
        )
        self.knn_model.fit(X_scaled)
        
        self.update_status(f"KNN model trained with {n_neighbors} neighbors")
    
    def get_recommendations(self, health_profile: HealthProfile, 
                          category_filter: str = "All", max_recommendations: int = 10) -> List[Dict]:
        """Get health-driven food recommendations with optional category filtering"""
        if self.knn_model is None:
            return []
        
        # Calculate nutritional targets
        targets = self.calculator.calculate_nutritional_targets(health_profile)
        
        # Filter data by category if specified
        if category_filter != "All":
            filtered_data = self.food_data[self.food_data['Category'] == category_filter]
            if len(filtered_data) == 0:
                return []  # No foods in selected category
            
            # Get indices of filtered foods in original dataset
            filtered_indices = filtered_data.index.tolist()
            
            # Filter scaled features for KNN search
            X_filtered = self.scaler.transform(filtered_data[self.feature_columns].values)
            
            # Create temporary KNN model for filtered data
            n_neighbors = min(max_recommendations * 2, len(filtered_data))  # Get more candidates
            temp_knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='manhattan')
            temp_knn.fit(X_filtered)
        else:
            filtered_data = self.food_data
            filtered_indices = list(range(len(self.food_data)))
            temp_knn = self.knn_model
        
        # Create target vector
        target_vector = []
        for feature in self.feature_columns:
            if feature == 'Energy(kcal) by calculation':
                # Scale down energy for per-meal recommendations
                target_vector.append(targets.energy_kcal / 3)  # Assuming 3 meals per day
            elif feature == 'Protein(g)':
                target_vector.append(targets.protein_g / 3)
            elif feature == 'Fat(g)':
                target_vector.append(targets.fat_g / 3)
            elif feature == 'CHOCDF (g) Carbohydrate':
                target_vector.append(targets.carbohydrates_g / 3)
            elif feature == 'SUGAR(g)':
                target_vector.append(targets.sugar_g / 3)
            elif feature == 'FIBTG (g) Dietary fibre':
                target_vector.append(targets.fiber_g / 3)
            elif feature == 'Na(mg)':
                target_vector.append(targets.sodium_mg / 3)
            elif feature == 'K(mg)':
                target_vector.append(targets.potassium_mg / 3)
            elif feature == 'FASAT (g) Saturated FA':
                target_vector.append(targets.saturated_fat_g / 3)
            elif feature == 'CHOLE(mg) Cholesterol':
                target_vector.append(100)  # Moderate cholesterol target
            else:
                target_vector.append(0)
        
        # Scale target vector
        target_scaled = self.scaler.transform([target_vector])
        
        # Find nearest neighbors
        distances, indices = temp_knn.kneighbors(target_scaled)
        
        # Create recommendations
        recommendations = []
        active_conditions = []
        if health_profile.diabetes:
            active_conditions.append('Diabetes')
        if health_profile.obesity:
            active_conditions.append('Obesity')
        if health_profile.hypertension:
            active_conditions.append('Hypertension')
        if health_profile.high_cholesterol:
            active_conditions.append('High_Cholesterol')
        
        for i, filtered_idx in enumerate(indices[0]):
            # Map back to original dataset index
            if category_filter != "All":
                original_idx = filtered_indices[filtered_idx]
                food = self.food_data.iloc[original_idx]
            else:
                food = self.food_data.iloc[filtered_idx]
            
            # Calculate health suitability
            health_scores = {}
            for condition in active_conditions:
                score_col = f'{condition}_Score'
                if score_col in self.food_data.columns:
                    health_scores[condition] = food[score_col]
            
            # Calculate Thai element suitability if enabled
            thai_element_score = 0
            if health_profile.use_thai_element:
                thai_calculator = ThaiElementCalculator()
                user_element = thai_calculator.calculate_life_element(
                    health_profile.birth_month, health_profile.birth_year
                )
                element_score_col = f'Element_{user_element}_Score'
                if element_score_col in self.food_data.columns:
                    thai_element_score = food[element_score_col]
            
            # Calculate overall health score
            total_health_scores = list(health_scores.values())
            if thai_element_score > 0:
                total_health_scores.append(thai_element_score)
            
            if total_health_scores:
                avg_health_score = np.mean(total_health_scores)
                health_rating = "Excellent" if avg_health_score < 2 else "Good" if avg_health_score < 5 else "Fair"
            else:
                avg_health_score = 0
                health_rating = "Good"
            
            recommendation = {
                'name': food.get('Thai_Name', food.get('English_Name', 'Unknown')),
                'english_name': food.get('English_Name', ''),
                'category': food.get('Category', 'Unknown'),
                'energy': food.get('Energy(kcal) by calculation', 0),
                'protein': food.get('Protein(g)', 0),
                'fat': food.get('Fat(g)', 0),
                'carbs': food.get('CHOCDF (g) Carbohydrate', 0),
                'sugar': food.get('SUGAR(g)', 0),
                'fiber': food.get('FIBTG (g) Dietary fibre', 0),
                'sodium': food.get('Na(mg)', 0),
                'distance': distances[0][i],
                'health_scores': health_scores,
                'thai_element_score': thai_element_score,
                'health_rating': health_rating,
                'avg_health_score': avg_health_score
            }
            
            recommendations.append(recommendation)
            
            # Stop when we have enough recommendations
            if len(recommendations) >= max_recommendations:
                break
        
        # Sort by combined score (distance + health score + thai element score)
        if active_conditions or health_profile.use_thai_element:
            recommendations.sort(key=lambda x: x['distance'] + x['avg_health_score'])
        else:
            recommendations.sort(key=lambda x: x['distance'])
        
        return recommendations[:max_recommendations]
    
    def get_available_categories(self) -> List[str]:
        """Get list of available food categories"""
        if 'Category' in self.food_data.columns:
            categories = ['All'] + sorted(self.food_data['Category'].unique().tolist())
            return categories
        return ['All']

class HealthDrivenFoodRecommenderGUI:
    """Modern GUI for health-driven food recommendation system"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Health-Driven Food Recommendation System")
        self.master.geometry("1400x900")
        self.master.configure(bg="#f8f9fa")
        
        # Initialize system
        self.calculator = MedicalNutritionCalculator()
        self.recommender = None
        self.current_targets = None
        self.current_recommendations = []
        
        # Setup UI
        self.setup_styles()
        self.create_ui()
        
        # Initialize recommender in background
        self.initialize_system()
    
    def setup_styles(self):
        """Setup modern UI styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color scheme
        self.colors = {
            'primary': '#007bff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Configure styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground=self.colors['primary'])
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), foreground=self.colors['dark'])
        self.style.configure('Primary.TButton', font=('Arial', 10, 'bold'))
        
    def create_ui(self):
        """Create the main UI layout"""
        # Main container
        main_frame = ttk.Frame(self.master, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Health-Driven Food Recommendation System", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Profile tab
        self.create_profile_tab()
        
        # Targets tab
        self.create_targets_tab()
        
        # Recommendations tab
        self.create_recommendations_tab()
        
        # Analysis tab
        self.create_analysis_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_profile_tab(self):
        """Create health profile input tab"""
        profile_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(profile_frame, text="Health Profile")
        
        # Create scrollable frame
        canvas = tk.Canvas(profile_frame)
        scrollbar = ttk.Scrollbar(profile_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Personal Information
        personal_frame = ttk.LabelFrame(scrollable_frame, text="Personal Information", padding="15")
        personal_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Age
        ttk.Label(personal_frame, text="Age (years):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.IntVar(value=30)
        age_spin = ttk.Spinbox(personal_frame, from_=18, to=100, textvariable=self.age_var, width=10)
        age_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Gender
        ttk.Label(personal_frame, text="Gender:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.gender_var = tk.StringVar(value="Male")
        gender_combo = ttk.Combobox(personal_frame, textvariable=self.gender_var, 
                                   values=["Male", "Female"], width=10)
        gender_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Weight
        ttk.Label(personal_frame, text="Weight (kg):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.weight_var = tk.DoubleVar(value=70.0)
        weight_spin = ttk.Spinbox(personal_frame, from_=30, to=200, increment=0.5, 
                                 textvariable=self.weight_var, width=10)
        weight_spin.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Height
        ttk.Label(personal_frame, text="Height (cm):").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.height_var = tk.DoubleVar(value=170.0)
        height_spin = ttk.Spinbox(personal_frame, from_=120, to=220, increment=0.5, 
                                 textvariable=self.height_var, width=10)
        height_spin.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # BMI display
        ttk.Label(personal_frame, text="BMI:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.bmi_var = tk.StringVar(value="Calculating...")
        bmi_label = ttk.Label(personal_frame, textvariable=self.bmi_var, font=('Arial', 10, 'bold'))
        bmi_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Bind BMI calculation
        self.weight_var.trace_add("write", self.calculate_bmi)
        self.height_var.trace_add("write", self.calculate_bmi)
        
        # Lifestyle Information
        lifestyle_frame = ttk.LabelFrame(scrollable_frame, text="Lifestyle Information", padding="15")
        lifestyle_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Activity Level
        ttk.Label(lifestyle_frame, text="Activity Level:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.activity_var = tk.StringVar(value="Moderate")
        activity_combo = ttk.Combobox(lifestyle_frame, textvariable=self.activity_var,
                                     values=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                     width=15)
        activity_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Weight Goal
        ttk.Label(lifestyle_frame, text="Weight Goal:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.weight_goal_var = tk.StringVar(value="Maintain")
        goal_combo = ttk.Combobox(lifestyle_frame, textvariable=self.weight_goal_var,
                                 values=["Lose", "Maintain", "Gain"], width=10)
        goal_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Food Category Filter
        ttk.Label(lifestyle_frame, text="Food Category:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(lifestyle_frame, textvariable=self.category_filter_var,
                                          values=["All"], width=15)
        self.category_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Max Recommendations
        ttk.Label(lifestyle_frame, text="Max Results:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.max_recs_var = tk.IntVar(value=10)
        max_recs_spin = ttk.Spinbox(lifestyle_frame, from_=5, to=20, textvariable=self.max_recs_var, width=10)
        max_recs_spin.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Health Conditions
        conditions_frame = ttk.LabelFrame(scrollable_frame, text="Health Conditions", padding="15")
        conditions_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.diabetes_var = tk.BooleanVar()
        self.obesity_var = tk.BooleanVar()
        self.hypertension_var = tk.BooleanVar()
        self.cholesterol_var = tk.BooleanVar()
        
        ttk.Checkbutton(conditions_frame, text="Diabetes", variable=self.diabetes_var).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(conditions_frame, text="Obesity", variable=self.obesity_var).grid(row=0, column=1, sticky=tk.W, padx=(20, 0), pady=5)
        ttk.Checkbutton(conditions_frame, text="Hypertension", variable=self.hypertension_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(conditions_frame, text="High Cholesterol", variable=self.cholesterol_var).grid(row=1, column=1, sticky=tk.W, padx=(20, 0), pady=5)
        
        # Thai Traditional Medicine (Optional)
        thai_frame = ttk.LabelFrame(scrollable_frame, text="Thai Traditional Medicine (Optional)", padding="15")
        thai_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Enable Thai element checkbox
        self.use_thai_element_var = tk.BooleanVar()
        enable_thai_check = ttk.Checkbutton(thai_frame, text="Use Thai Life Element Analysis", 
                                          variable=self.use_thai_element_var,
                                          command=self.toggle_thai_inputs)
        enable_thai_check.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        # Birth information (initially disabled)
        ttk.Label(thai_frame, text="Birth Month:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.birth_month_var = tk.IntVar(value=1)
        self.birth_month_combo = ttk.Combobox(thai_frame, textvariable=self.birth_month_var,
                                            values=list(range(1, 13)), width=8, state="disabled")
        self.birth_month_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(thai_frame, text="Birth Year:").grid(row=1, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.birth_year_var = tk.IntVar(value=1990)
        self.birth_year_spin = ttk.Spinbox(thai_frame, from_=1950, to=2010, 
                                         textvariable=self.birth_year_var, width=8, state="disabled")
        self.birth_year_spin.grid(row=1, column=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Life element display
        ttk.Label(thai_frame, text="Life Element:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.life_element_var = tk.StringVar(value="Not calculated")
        self.life_element_label = ttk.Label(thai_frame, textvariable=self.life_element_var, 
                                          font=('Arial', 10, 'bold'), foreground="#007bff")
        self.life_element_label.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Bind calculation
        self.birth_month_var.trace_add("write", self.calculate_life_element)
        self.birth_year_var.trace_add("write", self.calculate_life_element)
        
        # Calculate button
        calc_frame = ttk.Frame(scrollable_frame)
        calc_frame.pack(fill=tk.X, pady=20)
        
        calc_btn = ttk.Button(calc_frame, text="Calculate Nutritional Targets", 
                             command=self.calculate_targets, style='Primary.TButton')
        calc_btn.pack(side=tk.LEFT)
        
        recommend_btn = ttk.Button(calc_frame, text="Get Recommendations", 
                                  command=self.get_recommendations)
        recommend_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial BMI calculation
        self.calculate_bmi()
    
    def create_targets_tab(self):
        """Create nutritional targets display tab"""
        targets_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(targets_frame, text="Nutritional Targets")
        
        # Explanation frame
        explanation_frame = ttk.LabelFrame(targets_frame, text="Calculation Explanation", padding="15")
        explanation_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.explanation_text = tk.Text(explanation_frame, height=8, wrap=tk.WORD)
        self.explanation_text.pack(fill=tk.BOTH, expand=True)
        
        # Targets display frame
        targets_display_frame = ttk.LabelFrame(targets_frame, text="Daily Nutritional Targets", padding="15")
        targets_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create targets table
        columns = ("Nutrient", "Target", "Unit", "Guideline Source")
        self.targets_tree = ttk.Treeview(targets_display_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.targets_tree.heading(col, text=col)
            self.targets_tree.column(col, width=150)
        
        # Scrollbar for targets
        targets_scroll = ttk.Scrollbar(targets_display_frame, orient="vertical", command=self.targets_tree.yview)
        self.targets_tree.configure(yscrollcommand=targets_scroll.set)
        
        self.targets_tree.pack(side="left", fill="both", expand=True)
        targets_scroll.pack(side="right", fill="y")
    
    def create_recommendations_tab(self):
        """Create recommendations display tab"""
        rec_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(rec_frame, text="Food Recommendations")
        
        # Recommendations table
        rec_table_frame = ttk.LabelFrame(rec_frame, text="Recommended Foods", padding="15")
        rec_table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        columns = ("Food Name", "Category", "Energy", "Protein", "Carbs", "Sugar", "Fiber", "Health Rating")
        self.rec_tree = ttk.Treeview(rec_table_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.rec_tree.heading(col, text=col)
            if col in ["Energy", "Protein", "Carbs", "Sugar", "Fiber"]:
                self.rec_tree.column(col, width=80)
            else:
                self.rec_tree.column(col, width=120)
        
        rec_scroll = ttk.Scrollbar(rec_table_frame, orient="vertical", command=self.rec_tree.yview)
        self.rec_tree.configure(yscrollcommand=rec_scroll.set)
        
        self.rec_tree.pack(side="left", fill="both", expand=True)
        rec_scroll.pack(side="right", fill="y")
        
        # Food details frame
        details_frame = ttk.LabelFrame(rec_frame, text="Food Details", padding="15")
        details_frame.pack(fill=tk.X)
        
        self.details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.rec_tree.bind('<<TreeviewSelect>>', self.show_food_details)
    
    def create_analysis_tab(self):
        """Create nutritional analysis and charts tab"""
        analysis_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(analysis_frame, text="Nutritional Analysis")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, analysis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(analysis_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(control_frame, text="Update Charts", command=self.update_charts).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export Analysis", command=self.export_analysis).pack(side=tk.LEFT, padx=(10, 0))
    
    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.master)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def initialize_system(self):
        """Initialize the recommendation system in background"""
        def init_worker():
            self.status_var.set("Initializing recommendation system...")
            self.progress.start()
            
            try:
                self.recommender = HealthDrivenKNN(status_callback=self.update_status_threadsafe)
                
                # Update category filter options
                self.master.after(0, self.update_category_filter)
                
                self.status_var.set("System ready")
            except Exception as e:
                self.status_var.set(f"Error: {e}")
            finally:
                self.progress.stop()
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def update_category_filter(self):
        """Update category filter dropdown with available categories"""
        if self.recommender:
            categories = self.recommender.get_available_categories()
            self.category_combo['values'] = categories
    
    def update_status_threadsafe(self, message):
        """Thread-safe status update"""
        self.master.after(0, lambda: self.status_var.set(message))
    
    def calculate_bmi(self, *args):
        """Calculate and display BMI"""
        try:
            weight = self.weight_var.get()
            height = self.height_var.get() / 100
            
            if weight > 0 and height > 0:
                bmi = weight / (height ** 2)
                
                if bmi < 18.5:
                    category = "Underweight"
                elif bmi < 25:
                    category = "Normal"
                elif bmi < 30:
                    category = "Overweight"
                else:
                    category = "Obese"
                
                self.bmi_var.set(f"{bmi:.1f} ({category})")
                
                # Auto-set obesity condition
                if bmi >= 30:
                    self.obesity_var.set(True)
                else:
                    self.obesity_var.set(False)
            else:
                self.bmi_var.set("Invalid input")
        except:
            self.bmi_var.set("Calculating...")
    
    def get_health_profile(self) -> HealthProfile:
        """Get current health profile from UI"""
        return HealthProfile(
            age=self.age_var.get(),
            gender=self.gender_var.get(),
            weight=self.weight_var.get(),
            height=self.height_var.get(),
            activity_level=self.activity_var.get(),
            weight_goal=self.weight_goal_var.get(),
            diabetes=self.diabetes_var.get(),
            obesity=self.obesity_var.get(),
            hypertension=self.hypertension_var.get(),
            high_cholesterol=self.cholesterol_var.get(),
            use_thai_element=self.use_thai_element_var.get(),
            birth_month=self.birth_month_var.get(),
            birth_year=self.birth_year_var.get(),
            life_element=self.life_element_var.get()
        )
    
    def toggle_thai_inputs(self):
        """Enable/disable Thai element inputs"""
        if self.use_thai_element_var.get():
            self.birth_month_combo.configure(state="normal")
            self.birth_year_spin.configure(state="normal")
            self.calculate_life_element()
        else:
            self.birth_month_combo.configure(state="disabled")
            self.birth_year_spin.configure(state="disabled")
            self.life_element_var.set("Not calculated")
    
    def calculate_life_element(self, *args):
        """Calculate and display Thai life element"""
        if not self.use_thai_element_var.get():
            return
            
        try:
            thai_calculator = ThaiElementCalculator()
            element = thai_calculator.calculate_life_element(
                self.birth_month_var.get(), 
                self.birth_year_var.get()
            )
            
            element_info = thai_calculator.get_element_info(element)
            if element_info:
                display_text = f"{element_info['name_english']}"
                self.life_element_var.set(display_text)
            else:
                self.life_element_var.set("Calculation error")
                
        except Exception as e:
            self.life_element_var.set("Calculation error")
    
    def calculate_targets(self):
        """Calculate and display nutritional targets"""
        try:
            profile = self.get_health_profile()
            self.current_targets = self.calculator.calculate_nutritional_targets(profile)
            
            # Update explanation
            self.explanation_text.delete(1.0, tk.END)
            explanation_content = self.current_targets.explanation
            
            # Add Thai element guidance if enabled
            if self.current_targets.thai_element_guidance:
                explanation_content += "\n\n" + "="*50 + "\n"
                explanation_content += self.current_targets.thai_element_guidance
                
            self.explanation_text.insert(tk.END, explanation_content)
            
            # Update targets table
            for item in self.targets_tree.get_children():
                self.targets_tree.delete(item)
            
            targets_data = [
                ("Energy", f"{self.current_targets.energy_kcal:.0f}", "kcal/day", "Mifflin-St Jeor + Activity"),
                ("Protein", f"{self.current_targets.protein_g:.1f}", "g/day", "0.8-1.6 g/kg based on conditions"),
                ("Total Fat", f"{self.current_targets.fat_g:.1f}", "g/day", "20-35% of energy"),
                ("Saturated Fat", f"{self.current_targets.saturated_fat_g:.1f}", "g/day", "AHA <6-7% of energy"),
                ("Carbohydrates", f"{self.current_targets.carbohydrates_g:.1f}", "g/day", "45-60% of energy"),
                ("Added Sugar", f"{self.current_targets.sugar_g:.1f}", "g/day", "WHO <5% of energy"),
                ("Fiber", f"{self.current_targets.fiber_g:.1f}", "g/day", "14g per 1000 kcal"),
                ("Sodium", f"{self.current_targets.sodium_mg:.0f}", "mg/day", "AHA <2300mg (<1500mg HT)"),
                ("Potassium", f"{self.current_targets.potassium_mg:.0f}", "mg/day", "General recommendation"),
            ]
            
            # Add Thai element info if enabled
            if profile.use_thai_element and profile.life_element != "Not calculated":
                thai_calculator = ThaiElementCalculator()
                element = thai_calculator.calculate_life_element(profile.birth_month, profile.birth_year)
                element_info = thai_calculator.get_element_info(element)
                if element_info:
                    targets_data.append(("Thai Element", element_info['name_english'], "", "Traditional Medicine"))
                    targets_data.append(("Element Focus", element_info['nutritional_focus'], "", "Traditional Guidance"))
            
            for data in targets_data:
                self.targets_tree.insert("", tk.END, values=data)
            
            # Switch to targets tab
            self.notebook.select(1)
            
            self.status_var.set("Nutritional targets calculated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate targets: {e}")
    
    def get_recommendations(self):
        """Get and display food recommendations"""
        if self.recommender is None:
            messagebox.showwarning("Warning", "Recommendation system not ready yet")
            return
        
        if self.current_targets is None:
            messagebox.showwarning("Warning", "Please calculate nutritional targets first")
            return
        
        try:
            self.progress.start()
            
            # Get filter settings
            category_filter = self.category_filter_var.get()
            max_recs = self.max_recs_var.get()
            
            self.status_var.set(f"Getting {category_filter} recommendations...")
            
            def get_recs():
                try:
                    profile = self.get_health_profile()
                    recommendations = self.recommender.get_recommendations(
                        profile, 
                        category_filter=category_filter,
                        max_recommendations=max_recs
                    )
                    
                    # Update UI in main thread
                    self.master.after(0, lambda: self.display_recommendations(recommendations, category_filter))
                    
                except Exception as e:
                    self.master.after(0, lambda: messagebox.showerror("Error", f"Failed to get recommendations: {e}"))  # noqa: F821
                finally:
                    self.master.after(0, lambda: self.progress.stop())
            
            threading.Thread(target=get_recs, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {e}")
            self.progress.stop()
    
    def display_recommendations(self, recommendations, category_filter="All"):
        """Display recommendations in the UI"""
        self.current_recommendations = recommendations
        
        # Clear existing items
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        
        if not recommendations:
            # Show message if no recommendations found
            if category_filter != "All":
                self.rec_tree.insert("", tk.END, values=("No foods found", f"in {category_filter} category", "", "", "", "", "", ""))
            else:
                self.rec_tree.insert("", tk.END, values=("No recommendations", "available", "", "", "", "", "", ""))
            
            self.status_var.set(f"No recommendations found for {category_filter}")
            return
        
        # Add recommendations
        for rec in recommendations:
            values = (
                rec['name'],
                rec['category'],
                f"{rec['energy']:.0f}",
                f"{rec['protein']:.1f}",
                f"{rec['carbs']:.1f}",
                f"{rec['sugar']:.1f}",
                f"{rec['fiber']:.1f}",
                rec['health_rating']
            )
            self.rec_tree.insert("", tk.END, values=values)
        
        # Switch to recommendations tab
        self.notebook.select(2)
        
        # Update status message
        filter_text = f"({category_filter})" if category_filter != "All" else ""
        self.status_var.set(f"Found {len(recommendations)} recommendations {filter_text}")
    
    def show_food_details(self, event):
        """Show details for selected food"""
        selection = self.rec_tree.selection()
        if not selection:
            return
        
        item = self.rec_tree.item(selection[0])
        food_name = item['values'][0]
        
        # Handle case where no recommendations were found
        if food_name in ["No foods found", "No recommendations"]:
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, "No food selected or no recommendations available.\n\n")
            self.details_text.insert(tk.END, "Try:\n")
            self.details_text.insert(tk.END, "• Selecting 'All' categories\n")
            self.details_text.insert(tk.END, "• Adjusting your health profile\n")
            self.details_text.insert(tk.END, "• Increasing max recommendations")
            return
        
        # Find recommendation details
        rec_details = None
        for rec in self.current_recommendations:
            if rec['name'] == food_name:
                rec_details = rec
                break
        
        if rec_details:
            # Format details
            details = f"Food: {rec_details['name']}\n"
            if rec_details['english_name']:
                details += f"English: {rec_details['english_name']}\n"
            details += f"Category: {rec_details['category']}\n\n"
            
            details += "Nutritional Content (per serving):\n"
            details += f"• Energy: {rec_details['energy']:.0f} kcal\n"
            details += f"• Protein: {rec_details['protein']:.1f} g\n"
            details += f"• Fat: {rec_details['fat']:.1f} g\n"
            details += f"• Carbohydrates: {rec_details['carbs']:.1f} g\n"
            details += f"• Sugar: {rec_details['sugar']:.1f} g\n"
            details += f"• Fiber: {rec_details['fiber']:.1f} g\n"
            details += f"• Sodium: {rec_details['sodium']:.0f} mg\n\n"
            
            details += f"Health Rating: {rec_details['health_rating']}\n"
            details += f"Nutritional Match Score: {rec_details['distance']:.2f}\n\n"
            
            if rec_details['health_scores']:
                details += "Health Condition Suitability (lower is better):\n"
                for condition, score in rec_details['health_scores'].items():
                    suitability = "Excellent" if score < 2 else "Good" if score < 5 else "Fair"
                    details += f"• {condition}: {score:.1f} ({suitability})\n"
            
            # Add Thai element compatibility if enabled
            if 'thai_element_score' in rec_details and rec_details['thai_element_score'] > 0:
                profile = self.get_health_profile()
                if profile.use_thai_element:
                    thai_calculator = ThaiElementCalculator()
                    element = thai_calculator.calculate_life_element(profile.birth_month, profile.birth_year)
                    element_info = thai_calculator.get_element_info(element)
                    
                    details += f"\nThai Traditional Medicine:\n"
                    details += f"• Your Life Element: {element_info['name_english']}\n"
                    
                    element_score = rec_details['thai_element_score']
                    compatibility = "Excellent" if element_score < 3 else "Good" if element_score < 6 else "Fair"
                    details += f"• Element Compatibility: {compatibility} (Score: {element_score:.1f})\n"
            
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, details)
    
    def update_charts(self):
        """Update nutritional analysis charts"""
        if not self.current_recommendations:
            messagebox.showwarning("Warning", "No recommendations to analyze")
            return
        
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax4 = self.fig.add_subplot(2, 2, 4)
        
        # Chart 1: Macronutrient distribution
        nutrients = ['Protein', 'Carbs', 'Fat']
        avg_nutrients = [
            np.mean([rec['protein'] for rec in self.current_recommendations]),
            np.mean([rec['carbs'] for rec in self.current_recommendations]),
            np.mean([rec['fat'] for rec in self.current_recommendations])
        ]
        
        ax1.pie(avg_nutrients, labels=nutrients, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Average Macronutrient Distribution')
        
        # Chart 2: Health ratings distribution
        ratings = [rec['health_rating'] for rec in self.current_recommendations]
        rating_counts = {rating: ratings.count(rating) for rating in set(ratings)}
        
        ax2.bar(rating_counts.keys(), rating_counts.values(), 
               color=['green', 'orange', 'red'][:len(rating_counts)])
        ax2.set_title('Health Rating Distribution')
        ax2.set_ylabel('Number of Foods')
        
        # Chart 3: Calorie vs Fiber
        calories = [rec['energy'] for rec in self.current_recommendations]
        fiber = [rec['fiber'] for rec in self.current_recommendations]
        
        ax3.scatter(calories, fiber, alpha=0.7)
        ax3.set_xlabel('Calories (kcal)')
        ax3.set_ylabel('Fiber (g)')
        ax3.set_title('Calories vs Fiber Content')
        
        # Chart 4: Food categories
        categories = [rec['category'] for rec in self.current_recommendations]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        ax4.barh(list(category_counts.keys()), list(category_counts.values()))
        ax4.set_title('Food Categories')
        ax4.set_xlabel('Number of Foods')
        
        # Adjust layout and refresh
        self.fig.tight_layout()
        self.canvas.draw()
    
    def export_analysis(self):
        """Export analysis results"""
        if not self.current_recommendations or not self.current_targets:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Health-Driven Food Recommendation Analysis\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Profile summary
                    profile = self.get_health_profile()
                    f.write("Health Profile:\n")
                    f.write(f"Age: {profile.age}, Gender: {profile.gender}\n")
                    f.write(f"Weight: {profile.weight} kg, Height: {profile.height} cm\n")
                    f.write(f"Activity: {profile.activity_level}, Goal: {profile.weight_goal}\n")
                    
                    conditions = []
                    if profile.diabetes: conditions.append("Diabetes")
                    if profile.obesity: conditions.append("Obesity")
                    if profile.hypertension: conditions.append("Hypertension")
                    if profile.high_cholesterol: conditions.append("High Cholesterol")
                    
                    f.write(f"Health Conditions: {', '.join(conditions) if conditions else 'None'}\n\n")
                    
                    # Nutritional targets
                    f.write("Calculated Nutritional Targets:\n")
                    f.write(f"Energy: {self.current_targets.energy_kcal:.0f} kcal/day\n")
                    f.write(f"Protein: {self.current_targets.protein_g:.1f} g/day\n")
                    f.write(f"Fat: {self.current_targets.fat_g:.1f} g/day\n")
                    f.write(f"Carbohydrates: {self.current_targets.carbohydrates_g:.1f} g/day\n")
                    f.write(f"Sugar: {self.current_targets.sugar_g:.1f} g/day\n")
                    f.write(f"Fiber: {self.current_targets.fiber_g:.1f} g/day\n")
                    f.write(f"Sodium: {self.current_targets.sodium_mg:.0f} mg/day\n\n")
                    
                    # Recommendations
                    f.write("Food Recommendations:\n")
                    for i, rec in enumerate(self.current_recommendations, 1):
                        f.write(f"{i}. {rec['name']} ({rec['category']})\n")
                        f.write(f"   Energy: {rec['energy']:.0f} kcal, Health Rating: {rec['health_rating']}\n")
                        f.write(f"   Protein: {rec['protein']:.1f}g, Carbs: {rec['carbs']:.1f}g, Fat: {rec['fat']:.1f}g\n\n")
                
                messagebox.showinfo("Success", f"Analysis exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = HealthDrivenFoodRecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()