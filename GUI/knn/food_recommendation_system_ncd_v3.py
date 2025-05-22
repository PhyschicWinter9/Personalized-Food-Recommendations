import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class FoodRecommendationSystem:
    def __init__(self, data_folder='./datasets'):
        # Nutritional thresholds based on guidelines
        self.nutritional_thresholds = {
            'diabetes': {
                'sugar_limit': 25,  # grams per day
                'carb_percent': (45, 60),  # % of total calories
                'fiber_per_1000kcal': 14,  # grams per 1000 kcal
                'protein_grams_per_kg': (0.8, 1.5),  # grams per kg body weight
                'fat_percent': (20, 35),  # % of total calories
                'sat_fat_percent': 7,  # % of total calories
            },
            'obesity': {
                'carb_percent': (45, 60),  # % of total calories
                'protein_percent': (20, 35),  # % of total calories
                'protein_grams_per_kg': (1.2, 1.6),  # grams per kg body weight for weight loss
                'fat_percent': (20, 35),  # % of total calories
                'added_sugar_percent': 5,  # % of total calories
                'fiber_grams': (25, 30),  # grams per day
            },
            'hypertension': {
                'sodium_limit': 1500,  # mg per day (ideal)
                'saturated_fat_percent': 6,  # % of total calories
                'potassium_target': 4700,  # mg per day
                'fat_percent': (20, 35),  # % of total calories
                'carb_percent': (45, 65),  # % of total calories
                'fiber_grams': 30,  # grams per day
            },
            'high_cholesterol': {
                'fat_percent': (25, 35),  # % of total calories
                'saturated_fat_percent': 6,  # % of total calories
                'cholesterol_limit': 200,  # mg per day
                'fiber_grams': (25, 30),  # grams per day (emphasis on soluble fiber)
            }
        }
        
        # Define key nutritional features for analysis
        self.nutritional_features = [
            'Energy(kcal) by calculation', 
            'Protein(g)', 
            'CHOCDF (g) Carbohydrate',
            'SUGAR(g)', 
            'FIBTG (g) Dietary fibre', 
            'Fat(g)',
            'Na(mg)',  # Sodium - important for hypertension
            'K(mg)',   # Potassium - important for hypertension
            'Ca(mg)',  # Calcium
            'CHOLE(mg) Cholesterol'  # Cholesterol - important for heart health
        ]
        
        # Load and prepare data
        self.food_data = self.load_datasets(data_folder)
        self.preprocess_data()
        
        # Setup KNN model
        self.optimal_k = self.find_optimal_k()
        self.setup_knn_model()
        
    def load_datasets(self, data_folder):
        """Load and combine all datasets from CSV files"""
        all_data = []
        
        # Get all CSV files in the folder
        csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {data_folder}")
            return pd.DataFrame()
        
        for file_path in csv_files:
            try:
                # Extract category from filename
                category = os.path.basename(file_path).split('.')[0]
                
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Add category if not already in the data
                if 'Category' not in df.columns:
                    df['Category'] = category
                    
                all_data.append(df)
                print(f"Loaded {len(df)} items from {category}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Combine all dataframes
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined dataset contains {len(combined_df)} items")
            return combined_df
        else:
            print("No valid data loaded")
            return pd.DataFrame()
    
    def preprocess_data(self):
        """Preprocess data: handle missing values and add features"""
        if len(self.food_data) == 0:
            print("No data to preprocess")
            return
        
        print("Preprocessing data...")
        
        # Handle missing values - replace with column means for numerical columns
        for col in self.food_data.select_dtypes(include=['float64', 'int64']).columns:
            self.food_data[col].fillna(self.food_data[col].mean(), inplace=True)
        
        # Ensure all required features exist, add with zeros if missing
        for feature in self.nutritional_features:
            if feature not in self.food_data.columns:
                print(f"Adding missing feature: {feature}")
                self.food_data[feature] = 0
        
        # Calculate health condition scores
        self.calculate_all_condition_scores()
        
        # Scale features for KNN
        self.scale_features()
        
        print("Preprocessing complete.")
    
    def calculate_all_condition_scores(self):
        """Calculate and add health condition scores to the dataset"""
        print("Calculating health condition scores...")
        
        # Add condition scores to the dataset
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = self.food_data.apply(
                lambda x: self.calculate_condition_score(x, condition), axis=1)
        
        # Print the distribution of condition scores
        print("\nCondition score statistics:")
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            print(f"{condition}: Min={self.food_data[f'{condition}_Score'].min():.2f}, "
                  f"Max={self.food_data[f'{condition}_Score'].max():.2f}, "
                  f"Mean={self.food_data[f'{condition}_Score'].mean():.2f}")
    
    def calculate_condition_score(self, food_item, condition):
        """Calculate suitability score for a specific health condition"""
        score = 0
        
        if condition == 'Diabetes':
            # Sugar content penalty (higher sugar = higher score = worse)
            sugar = food_item.get('SUGAR(g)', 0)
            if not pd.isna(sugar):
                if sugar <= 5:
                    score += sugar * 0.1  # Minimal impact for low sugar
                elif sugar <= 15:
                    score += 0.5 + (sugar - 5) * 0.3  # Moderate impact
                else:
                    score += 3.5 + (sugar - 15) * 0.5  # Significant impact for high sugar
            
            # Carbohydrate impact
            carbs = food_item.get('CHOCDF (g) Carbohydrate', 0)
            if not pd.isna(carbs):
                if carbs <= 15:  # Moderate serving
                    score += carbs * 0.05
                elif carbs <= 30:
                    score += 0.75 + (carbs - 15) * 0.1
                else:
                    score += 2.25 + (carbs - 30) * 0.15
            
            # Fiber benefit (higher fiber = lower score = better)
            fiber = food_item.get('FIBTG (g) Dietary fibre', 0)
            if not pd.isna(fiber):
                score -= min(fiber, 10) * 0.3  # Cap benefit at 10g fiber
        
        elif condition == 'Obesity':
            # Calorie content
            calories = food_item.get('Energy(kcal) by calculation', 0)
            if not pd.isna(calories):
                if calories <= 100:
                    score += calories * 0.005
                elif calories <= 300:
                    score += 0.5 + (calories - 100) * 0.01
                else:
                    score += 2.5 + (calories - 300) * 0.015
            
            # Fat content
            fat = food_item.get('Fat(g)', 0)
            if not pd.isna(fat):
                score += fat * 0.2
            
            # Sugar penalty
            sugar = food_item.get('SUGAR(g)', 0)
            if not pd.isna(sugar):
                score += sugar * 0.3
            
            # Protein benefit
            protein = food_item.get('Protein(g)', 0)
            if not pd.isna(protein):
                score -= min(protein, 30) * 0.15
            
            # Fiber benefit
            fiber = food_item.get('FIBTG (g) Dietary fibre', 0)
            if not pd.isna(fiber):
                score -= min(fiber, 10) * 0.4
        
        elif condition == 'Hypertension':
            # Sodium content (critical for hypertension)
            sodium = food_item.get('Na(mg)', 0)
            if not pd.isna(sodium):
                if sodium <= 140:  # Low sodium (FDA definition)
                    score += sodium * 0.002
                elif sodium <= 400:  # Moderate sodium
                    score += 0.28 + (sodium - 140) * 0.005
                else:  # High sodium
                    score += 1.58 + (sodium - 400) * 0.008
            
            # Potassium benefit
            potassium = food_item.get('K(mg)', 0)
            if not pd.isna(potassium):
                score -= min(potassium, 1000) * 0.002
            
            # Fat consideration
            fat = food_item.get('Fat(g)', 0)
            if not pd.isna(fat):
                score += fat * 0.2
        
        elif condition == 'High_Cholesterol':
            # Saturated fat (if available, otherwise use total fat)
            sat_fat = food_item.get('FASAT (g) Saturated FA', None)
            fat = food_item.get('Fat(g)', 0)
            
            if sat_fat is not None and not pd.isna(sat_fat):
                score += sat_fat * 0.5
            elif not pd.isna(fat):
                # Estimate based on total fat if saturated fat is not available
                score += fat * 0.15
            
            # Dietary cholesterol
            dietary_chol = food_item.get('CHOLE(mg) Cholesterol', 0)
            if not pd.isna(dietary_chol):
                if dietary_chol <= 20:
                    score += dietary_chol * 0.01
                elif dietary_chol <= 100:
                    score += 0.2 + (dietary_chol - 20) * 0.02
                else:
                    score += 1.8 + (dietary_chol - 100) * 0.04
            
            # Fiber benefit
            fiber = food_item.get('FIBTG (g) Dietary fibre', 0)
            if not pd.isna(fiber):
                score -= min(fiber, 10) * 0.4
        
        return max(0, score)  # Ensure non-negative score
    
    def scale_features(self):
        """Scale nutritional features for KNN"""
        # Define features for KNN
        self.knn_features = self.nutritional_features + [
            'Diabetes_Score', 
            'Obesity_Score', 
            'Hypertension_Score', 
            'High_Cholesterol_Score'
        ]
        
        # Extract and scale features
        X = self.food_data[self.knn_features].fillna(0)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
    
    def find_optimal_k(self, k_range=range(3, 21, 2)):
        """Find optimal k value for KNN"""
        print("Finding optimal k value...")
        
        # Extract features
        X = self.food_data[self.nutritional_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cv_scores = []
        
        for k in k_range:
            knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
            knn.fit(X_scaled)
            
            # Calculate average distance to neighbors
            distances, _ = knn.kneighbors(X_scaled)
            avg_distance = np.mean(distances)
            cv_scores.append(avg_distance)
        
        # Find k with minimum average distance
        optimal_k = k_range[np.argmin(cv_scores)]
        print(f"Optimal k value: {optimal_k}")
        
        return optimal_k
    
    def setup_knn_model(self):
        """Initialize the KNN model with optimal k"""
        self.knn_model = NearestNeighbors(
            n_neighbors=self.optimal_k,
            algorithm='auto',
            metric='euclidean'
        )
        
        self.knn_model.fit(self.X_scaled)
        print(f"KNN model initialized with k={self.optimal_k}")
    
    def create_user_profile(self, personal_data, nutritional_preferences):
        """Create a complete user profile for KNN matching"""
        # Extract personal data with defaults
        weight = personal_data.get('weight', 70)  # kg
        height = personal_data.get('height', 170)  # cm
        age = personal_data.get('age', 45)
        gender = personal_data.get('gender', 'male')
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Determine calorie needs (simplified BMR x activity)
        if gender.lower() == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Activity factor (default to moderate)
        activity_factor = personal_data.get('activity_factor', 1.55)
        daily_calories = bmr * activity_factor
        
        # Meal-sized portion (default to 1/3 of daily needs)
        meal_factor = personal_data.get('meal_factor', 1/3)
        meal_calories = daily_calories * meal_factor
        
        # Create default nutritional profile based on health guidelines
        profile = {
            'Energy(kcal) by calculation': nutritional_preferences.get('Energy(kcal) by calculation', meal_calories),
            'Protein(g)': nutritional_preferences.get('Protein(g)', weight * 0.8 * meal_factor),  # 0.8g per kg
            'CHOCDF (g) Carbohydrate': nutritional_preferences.get('CHOCDF (g) Carbohydrate', meal_calories * 0.5 / 4),  # 50% of calories
            'SUGAR(g)': nutritional_preferences.get('SUGAR(g)', 5),  # Low sugar default
            'FIBTG (g) Dietary fibre': nutritional_preferences.get('FIBTG (g) Dietary fibre', 10),  # ~30g daily / 3 meals
            'Fat(g)': nutritional_preferences.get('Fat(g)', meal_calories * 0.3 / 9),  # 30% of calories
            'Na(mg)': nutritional_preferences.get('Na(mg)', 500),  # ~1500mg daily / 3 meals
            'K(mg)': nutritional_preferences.get('K(mg)', 1500),  # ~4500mg daily / 3 meals
            'Ca(mg)': nutritional_preferences.get('Ca(mg)', 300),  # ~1000mg daily / 3 meals
            'CHOLE(mg) Cholesterol': nutritional_preferences.get('CHOLE(mg) Cholesterol', 100)  # ~300mg daily / 3 meals
        }
        
        # Add personal data to profile
        profile.update({
            'weight': weight,
            'height': height,
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'daily_calories': daily_calories
        })
        
        return profile
    
    def get_recommendations(self, user_profile, health_conditions=None, n_recommendations=10, category_filter=None):
        """Get food recommendations based on user profile and health conditions"""
        if len(self.food_data) == 0:
            print("No data available for recommendations")
            return []
        
        # Default to empty list if conditions is None
        if health_conditions is None:
            health_conditions = []
        
        # Create user feature vector
        user_vector = []
        
        # Extract nutritional preferences from user profile
        for feature in self.nutritional_features:
            user_vector.append(user_profile.get(feature, 0))
        
        # Add placeholder values for condition scores (will be weighted later)
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            if condition in health_conditions:
                user_vector.append(0)  # Prefer lower scores for conditions of interest
            else:
                user_vector.append(5)  # Neutral for other conditions
        
        # Scale user vector
        user_scaled = self.scaler.transform([user_vector])
        
        # Find k nearest neighbors
        distances, indices = self.knn_model.kneighbors(user_scaled)
        
        # Get recommended food items
        recommendations = []
        
        for i, idx in enumerate(indices[0]):
            food_item = self.food_data.iloc[idx]
            
            # Skip if not matching category filter
            if category_filter and food_item.get('Category', '') != category_filter:
                continue
            
            # Calculate relevance score with health conditions in mind
            relevance_score = 1.0 / (distances[0][i] + 0.1)  # Avoid division by zero
            
            # Adjust relevance based on health conditions
            if health_conditions:
                condition_penalty = 0
                for condition in health_conditions:
                    # Higher score = worse for the condition
                    condition_penalty += food_item[f'{condition}_Score']
                
                # Reduce relevance for items with high condition scores
                relevance_score = relevance_score / (1 + condition_penalty * 0.1)
            
            # Add to recommendations with relevance score
            recommendations.append({
                'name': food_item.get('Thai_Name', food_item.get('English_Name', f"Item {idx}")),
                'category': food_item.get('Category', 'Unknown'),
                'relevance': relevance_score,
                'nutritional_data': {
                    'energy': food_item.get('Energy(kcal) by calculation', 0),
                    'protein': food_item.get('Protein(g)', 0),
                    'carbs': food_item.get('CHOCDF (g) Carbohydrate', 0),
                    'sugar': food_item.get('SUGAR(g)', 0),
                    'fiber': food_item.get('FIBTG (g) Dietary fibre', 0),
                    'fat': food_item.get('Fat(g)', 0),
                    'sodium': food_item.get('Na(mg)', 0),
                    'potassium': food_item.get('K(mg)', 0),
                    'cholesterol': food_item.get('CHOLE(mg) Cholesterol', 0)
                },
                'condition_scores': {
                    'diabetes': food_item.get('Diabetes_Score', 0),
                    'obesity': food_item.get('Obesity_Score', 0),
                    'hypertension': food_item.get('Hypertension_Score', 0),
                    'high_cholesterol': food_item.get('High_Cholesterol_Score', 0)
                },
                'is_suitable_for': self.determine_suitable_conditions(food_item)
            })
        
        # Sort by relevance and filter for top n
        recommendations.sort(key=lambda x: x['relevance'], reverse=True)
        return recommendations[:n_recommendations]
    
    def determine_suitable_conditions(self, food_item):
        """Determine which health conditions a food item is suitable for"""
        suitable_for = []
        
        # Check diabetes suitability
        diabetes_score = food_item.get('Diabetes_Score', 0)
        sugar = food_item.get('SUGAR(g)', 0)
        fiber = food_item.get('FIBTG (g) Dietary fibre', 0)
        
        if diabetes_score <= 2 and sugar <= 10 and fiber >= 2:
            suitable_for.append('Diabetes')
        
        # Check obesity suitability
        obesity_score = food_item.get('Obesity_Score', 0)
        calories = food_item.get('Energy(kcal) by calculation', 0)
        
        if obesity_score <= 2 and calories <= 250 and fiber >= 2:
            suitable_for.append('Obesity')
        
        # Check hypertension suitability
        hypertension_score = food_item.get('Hypertension_Score', 0)
        sodium = food_item.get('Na(mg)', 0)
        
        if hypertension_score <= 2 and sodium <= 200:
            suitable_for.append('Hypertension')
        
        # Check cholesterol suitability
        cholesterol_score = food_item.get('High_Cholesterol_Score', 0)
        fat = food_item.get('Fat(g)', 0)
        cholesterol = food_item.get('CHOLE(mg) Cholesterol', 0)
        
        if cholesterol_score <= 2 and fat <= 10 and cholesterol <= 50:
            suitable_for.append('High_Cholesterol')
        
        return suitable_for
    
    def evaluate_model(self, k_values=None):
        """Evaluate KNN performance using cross-validation"""
        if k_values is None:
            k_values = [3, 5, 7, 9, 11]
        
        print("Evaluating KNN performance...")
        
        # Prepare data for evaluation
        X_eval = self.food_data[self.knn_features].fillna(0).values
        
        # Metrics to track
        metrics = {k: {'mae': [], 'precision': []} for k in k_values}
        
        # Use KFold for cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(X_eval):
            X_train, X_test = X_eval[train_index], X_eval[test_index]
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Evaluate each k value
            for k in k_values:
                # Train KNN model
                knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
                knn.fit(X_train_scaled)
                
                # Get nearest neighbors for test set
                distances, indices = knn.kneighbors(X_test_scaled)
                
                # Calculate Mean Absolute Error (MAE) for nutritional features
                predictions = np.mean(X_train[indices], axis=1)
                mae = np.mean(np.abs(predictions - X_test), axis=1).mean()
                metrics[k]['mae'].append(mae)
                
                # Calculate precision (as a proxy)
                relevance = 1 / (distances + 0.1)  # Avoid division by zero
                precision = np.mean(relevance)
                metrics[k]['precision'].append(precision)
        
        # Average metrics across folds
        for k in k_values:
            metrics[k]['mae'] = np.mean(metrics[k]['mae'])
            metrics[k]['precision'] = np.mean(metrics[k]['precision'])
        
        # Display evaluation metrics
        print("\nKNN Evaluation Metrics:")
        print("K\tMAE\t\tPrecision")
        for k in k_values:
            print(f"{k}\t{metrics[k]['mae']:.4f}\t{metrics[k]['precision']:.4f}")
        
        # Find best k value based on MAE
        best_k = min(metrics.keys(), key=lambda k: metrics[k]['mae'])
        print(f"Best k value based on MAE: {best_k}")
        
        return metrics, best_k
    
    def plot_feature_importance(self):
        """Plot feature importance using Random Forest for diabetes score prediction"""
        if len(self.food_data) == 0:
            print("No data available for feature importance analysis")
            return
        
        # Use diabetes score as an example target
        y = self.food_data['Diabetes_Score']
        X = self.food_data[self.nutritional_features].fillna(0)
        
        # Train a random forest to determine feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature importance
        print("\nFeature ranking for diabetes score prediction:")
        for i in range(len(importances)):
            print(f"{i+1}. {self.nutritional_features[indices[i]]} ({importances[indices[i]]:.4f})")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance for Diabetes Score")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [self.nutritional_features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        print("Feature importance plot saved to 'feature_importance.png'")
    
    def get_food_categories(self):
        """Get all food categories in the dataset"""
        if 'Category' in self.food_data.columns:
            return sorted(self.food_data['Category'].unique())
        return []


# Main execution
if __name__ == "__main__":
    # Initialize the recommendation system
    print("Initializing Food Recommendation System...")
    recommender = FoodRecommendationSystem()
    
    # Plot feature importance
    recommender.plot_feature_importance()
    
    # Evaluate model
    recommender.evaluate_model()
    
    print("\n--- Example Recommendations ---")
    
    # Example 1: User with diabetes and hypertension
    print("\nExample 1: User with diabetes and hypertension")
    
    user_personal = {
        'weight': 75,  # kg
        'height': 165,  # cm
        'age': 55,
        'gender': 'male',
        'activity_factor': 1.3  # Sedentary
    }
    
    user_nutritional = {
        'Energy(kcal) by calculation': 400,  # Target calories per meal
        'Protein(g)': 25,
        'SUGAR(g)': 3,  # Low sugar preference
        'FIBTG (g) Dietary fibre': 10,  # High fiber preference
        'Na(mg)': 150  # Low sodium preference
    }
    
    user_profile = recommender.create_user_profile(user_personal, user_nutritional)
    
    # Print user profile
    print("\nUser Profile:")
    for key, value in sorted(user_profile.items()):
        if key in recommender.nutritional_features:
            print(f"{key}: {value:.1f}")
    
    # Get recommendations for diabetes and hypertension
    recommendations = recommender.get_recommendations(
        user_profile,
        health_conditions=['Diabetes', 'Hypertension'],
        n_recommendations=5
    )
    
    # Display recommendations
    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"\n{i+1}. {rec['name']} ({rec['category']})")
        print(f"   Energy: {rec['nutritional_data']['energy']:.1f} kcal, "
              f"Protein: {rec['nutritional_data']['protein']:.1f}g, "
              f"Carbs: {rec['nutritional_data']['carbs']:.1f}g, "
              f"Sugar: {rec['nutritional_data']['sugar']:.1f}g, "
              f"Sodium: {rec['nutritional_data']['sodium']:.1f}mg")
        print(f"   Condition Scores - Diabetes: {rec['condition_scores']['diabetes']:.2f}, "
              f"Hypertension: {rec['condition_scores']['hypertension']:.2f}")
        print(f"   Suitable for: {', '.join(rec['is_suitable_for'])}")
    
    # Example 2: User focusing on weight management
    print("\n\nExample 2: User focusing on weight management")
    
    user_personal2 = {
        'weight': 85,  # kg
        'height': 175,  # cm
        'age': 40,
        'gender': 'female',
        'activity_factor': 1.5  # Moderate activity
    }
    
    user_nutritional2 = {
        'Energy(kcal) by calculation': 350,  # Lower calorie target
        'Protein(g)': 30,  # Higher protein
        'Fat(g)': 10,  # Lower fat
        'FIBTG (g) Dietary fibre': 12  # Higher fiber
    }
    
    user_profile2 = recommender.create_user_profile(user_personal2, user_nutritional2)
    
    # Get recommendations for obesity
    recommendations2 = recommender.get_recommendations(
        user_profile2,
        health_conditions=['Obesity'],
        n_recommendations=5
    )
    
    # Display recommendations
    print("\nTop 5 Recommendations for Weight Management:")
    for i, rec in enumerate(recommendations2):
        print(f"\n{i+1}. {rec['name']} ({rec['category']})")
        print(f"   Energy: {rec['nutritional_data']['energy']:.1f} kcal, "
              f"Protein: {rec['nutritional_data']['protein']:.1f}g, "
              f"Fat: {rec['nutritional_data']['fat']:.1f}g, "
              f"Fiber: {rec['nutritional_data']['fiber']:.1f}g")
        print(f"   Condition Score - Obesity: {rec['condition_scores']['obesity']:.2f}")
        print(f"   Suitable for: {', '.join(rec['is_suitable_for'])}")
    
    # Example 3: Filter by category
    print("\n\nExample 3: Filter recommendations by food category")
    
    # Get available categories
    categories = recommender.get_food_categories()
    if categories:
        # Select a category (first in the list)
        selected_category = categories[0]
        print(f"Filtering for category: {selected_category}")
        
        # Get recommendations with category filter
        category_recommendations = recommender.get_recommendations(
            user_profile,
            health_conditions=['Diabetes'],
            category_filter=selected_category,
            n_recommendations=3
        )
        
        # Display recommendations
        print(f"\nTop 3 {selected_category} Recommendations:")
        if category_recommendations:
            for i, rec in enumerate(category_recommendations):
                print(f"\n{i+1}. {rec['name']} ({rec['category']})")
                print(f"   Energy: {rec['nutritional_data']['energy']:.1f} kcal, "
                      f"Sugar: {rec['nutritional_data']['sugar']:.1f}g")
                print(f"   Condition Score - Diabetes: {rec['condition_scores']['diabetes']:.2f}")
                print(f"   Suitable for: {', '.join(rec['is_suitable_for'])}")
        else:
            print(f"No items found in category: {selected_category}")