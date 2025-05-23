import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelPerformanceTracker:
    """Class to track and compare model performance"""
    def __init__(self):
        self.metrics = {
            'KNN': {
                'execution_time': [],
                'food_relevance_score': [],
                'condition_match_score': []
            },
            'RandomForest': {
                'execution_time': [],
                'food_relevance_score': [],
                'condition_match_score': [],
                'feature_importance': None
            }
        }
        
    def record_time(self, model_name, execution_time):
        """Record execution time for a model"""
        self.metrics[model_name]['execution_time'].append(execution_time)
        
    def record_relevance(self, model_name, relevance_score):
        """Record food relevance score (lower distance/error = better relevance)"""
        self.metrics[model_name]['food_relevance_score'].append(relevance_score)
    
    def record_condition_match(self, model_name, match_score):
        """Record condition match score (higher = better)"""
        self.metrics[model_name]['condition_match_score'].append(match_score)
        
    def record_feature_importance(self, feature_importance, feature_names):
        """Record feature importance from Random Forest"""
        self.metrics['RandomForest']['feature_importance'] = {
            'values': feature_importance,
            'names': feature_names
        }
        
    def get_average_metrics(self, model_name):
        """Get average metrics for a model"""
        metrics = self.metrics[model_name]
        avg_time = np.mean(metrics['execution_time']) if metrics['execution_time'] else 0
        avg_relevance = np.mean(metrics['food_relevance_score']) if metrics['food_relevance_score'] else 0
        avg_match = np.mean(metrics['condition_match_score']) if metrics['condition_match_score'] else 0
        
        return {
            'avg_execution_time': avg_time,
            'avg_food_relevance': avg_relevance,
            'avg_condition_match': avg_match
        }
    
    def compare_models(self):
        """Compare performance of KNN vs Random Forest"""
        knn_metrics = self.get_average_metrics('KNN')
        rf_metrics = self.get_average_metrics('RandomForest')
        
        comparison = {
            'execution_time_ratio': rf_metrics['avg_execution_time'] / knn_metrics['avg_execution_time'] 
                if knn_metrics['avg_execution_time'] > 0 else float('inf'),
            'relevance_improvement': (knn_metrics['avg_food_relevance'] - rf_metrics['avg_food_relevance']) / 
                knn_metrics['avg_food_relevance'] if knn_metrics['avg_food_relevance'] > 0 else 0,
            'condition_match_improvement': (rf_metrics['avg_condition_match'] - knn_metrics['avg_condition_match']) / 
                knn_metrics['avg_condition_match'] if knn_metrics['avg_condition_match'] > 0 else 0
        }
        
        return comparison
    
    def plot_performance_comparison(self, figure):
        """Plot performance comparison between KNN and Random Forest"""
        knn_metrics = self.get_average_metrics('KNN')
        rf_metrics = self.get_average_metrics('RandomForest')
        
        # Clear figure
        figure.clear()
        
        # Create subplots for comparisons
        ax1 = figure.add_subplot(131)  # Execution time comparison
        ax2 = figure.add_subplot(132)  # Food relevance comparison
        ax3 = figure.add_subplot(133)  # Condition match comparison
        
        # Data for comparison
        models = ['KNN', 'Random Forest']
        execution_times = [knn_metrics['avg_execution_time'], rf_metrics['avg_execution_time']]
        relevance_scores = [knn_metrics['avg_food_relevance'], rf_metrics['avg_food_relevance']]
        match_scores = [knn_metrics['avg_condition_match'], rf_metrics['avg_condition_match']]
        
        # Plot execution time (lower is better)
        ax1.bar(models, execution_times, color=['#3498db', '#e74c3c'])
        ax1.set_title('Execution Time (s)\n(Lower is better)')
        ax1.set_ylabel('Seconds')
        
        # Plot food relevance (lower is better)
        ax2.bar(models, relevance_scores, color=['#3498db', '#e74c3c'])
        ax2.set_title('Food Relevance Error\n(Lower is better)')
        ax2.set_ylabel('Average Error')
        
        # Plot condition match (higher is better)
        ax3.bar(models, match_scores, color=['#3498db', '#e74c3c'])
        ax3.set_title('Condition Match Score\n(Higher is better)')
        ax3.set_ylabel('Average Score')
        
        figure.tight_layout()
        
    def plot_feature_importance(self, figure):
        """Plot feature importance from Random Forest"""
        if not self.metrics['RandomForest']['feature_importance']:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No feature importance data available yet', 
                   ha='center', va='center', fontsize=12)
            figure.tight_layout()
            return
            
        # Get feature importance data
        importance = self.metrics['RandomForest']['feature_importance']['values']
        feature_names = self.metrics['RandomForest']['feature_importance']['names']
        
        # Sort by importance
        indices = np.argsort(importance)
        sorted_names = [feature_names[i] for i in indices]
        sorted_importance = [importance[i] for i in indices]
        
        # Clear figure
        figure.clear()
        ax = figure.add_subplot(111)
        
        # Plot feature importance
        ax.barh(sorted_names, sorted_importance, color='#2ecc71')
        ax.set_title('Feature Importance in Random Forest Model')
        ax.set_xlabel('Importance')
        
        figure.tight_layout()


class FoodRecommendationSystem:
    def __init__(self, status_callback=None):
        # Callback function to update loading status
        self.status_callback = status_callback
        
        # Performance tracker
        self.performance_tracker = ModelPerformanceTracker()
        
        # Model selection (default to KNN to match original code)
        self.model_type = "KNN"  # or "RandomForest"
        
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
        
        # Prepare both models
        self.prepare_knn_data()
        self.prepare_random_forest_data()
        
    def set_model(self, model_type):
        """Set the model type to use for recommendations"""
        if model_type in ["KNN", "RandomForest"]:
            self.model_type = model_type
            self.update_status(f"Model set to {model_type}")
        else:
            self.update_status("Invalid model type. Using KNN as default.")
            self.model_type = "KNN"
    
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
            
            import os
            import glob
            
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
    
    def prepare_random_forest_data(self):
        """Prepare data for Random Forest algorithms"""
        if len(self.food_data) == 0 or not self.features:
            self.update_status("Error: No data or features available for Random Forest model")
            return
        
        # Check if all features exist in the dataset
        missing_features = [f for f in self.features if f not in self.food_data.columns]
        if missing_features:
            self.update_status(f"Warning: Missing features in dataset for RF: {missing_features}")
            # Use only available features
            self.features = [f for f in self.features if f in self.food_data.columns]
        
        if not self.features:
            self.update_status("Error: No valid features available for Random Forest")
            return
        
        # Extract features
        X = self.food_data[self.features].fillna(0)
        
        # Initialize Random Forest models for each condition
        self.rf_models = {}
        
        # Train a model for each condition if the condition column exists
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol']:
            condition_key = condition.replace(' ', '_')
            # Check if we have labels for this condition
            if condition in self.food_data.columns:
                # Get labels (convert to binary)
                y = (self.food_data[condition] <= 1).astype(int)
                
                # Train a Random Forest classifier
                rf_model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=5,
                    min_samples_split=5,
                    random_state=42
                )
                rf_model.fit(X, y)
                
                # Store the model
                self.rf_models[condition_key] = {
                    'model': rf_model,
                    'type': 'classifier',
                    'feature_importance': rf_model.feature_importances_
                }
                
                self.update_status(f"Random Forest classifier trained for {condition} with accuracy: {rf_model.score(X, y):.2f}")
            else:
                # If we don't have labels, train a regressor to predict suitability scores
                # Create synthetic scores based on the condition_features
                synthetic_scores = np.zeros(len(X))
                
                if condition_key in self.condition_features:
                    # Use available relevant features to create a score
                    for feature in self.condition_features[condition_key]:
                        if feature in self.food_data.columns and feature != condition:
                            # Convert feature values to scores based on our domain knowledge
                            if feature == 'SUGAR(g)' and condition_key in ['Diabetes', 'Obesity']:
                                # Higher sugar = worse for diabetes and obesity
                                synthetic_scores += self.food_data[feature].fillna(0) * 0.1
                            elif feature == 'FIBTG (g) Dietary fibre':
                                # Higher fiber = better for all conditions
                                synthetic_scores -= self.food_data[feature].fillna(0) * 0.1
                            elif feature == 'Na(mg)' and condition_key == 'Hypertension':
                                # Higher sodium = worse for hypertension
                                synthetic_scores += self.food_data[feature].fillna(0) * 0.001
                            elif feature == 'FASAT (g) Saturated FA' and condition_key == 'High_Cholesterol':
                                # Higher saturated fat = worse for cholesterol
                                synthetic_scores += self.food_data[feature].fillna(0) * 0.2
                
                # Normalize to 0-1 range
                synthetic_scores = (synthetic_scores - synthetic_scores.min()) / (synthetic_scores.max() - synthetic_scores.min() + 1e-10)
                
                # Train a Random Forest regressor
                rf_model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=5,
                    min_samples_split=5,
                    random_state=42
                )
                rf_model.fit(X, synthetic_scores)
                
                # Store the model
                self.rf_models[condition_key] = {
                    'model': rf_model,
                    'type': 'regressor',
                    'feature_importance': rf_model.feature_importances_
                }
                
                self.update_status(f"Random Forest regressor trained for {condition} (synthetic scores)")
        
        # Record feature importance for visualization
        # Use the first model's feature importance if available
        if self.rf_models and len(self.features) > 0:
            first_condition = list(self.rf_models.keys())[0]
            feature_importance = self.rf_models[first_condition]['feature_importance']
            self.performance_tracker.record_feature_importance(feature_importance, self.features)
        
        self.update_status(f"Random Forest models prepared for {len(self.rf_models)} conditions")
    
    def get_recommendations_knn(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """Get food recommendations using K-NN algorithm"""
        if not hasattr(self, 'knn_model') or len(self.food_data) == 0:
            return []
        
        # Start timer for performance tracking
        start_time = time.time()
        
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
        
        # End timer and record performance
        end_time = time.time()
        execution_time = end_time - start_time
        self.performance_tracker.record_time('KNN', execution_time)
        
        # Calculate and record relevance score (using avg distance)
        if recommendations:
            avg_distance = np.mean([rec['Distance'] for rec in recommendations])
            self.performance_tracker.record_relevance('KNN', avg_distance)
            
            # Calculate and record condition match score
            if conditions:
                condition_match_percentage = np.mean([
                    len([c for c in rec['Suitable_For'] if c in conditions]) / len(conditions)
                    for rec in recommendations
                ]) * 100
                self.performance_tracker.record_condition_match('KNN', condition_match_percentage)
        
        # Return top recommendations (with at least some items)
        return recommendations[:max_recommendations]
    
    def get_recommendations_rf(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """Get food recommendations using Random Forest algorithm"""
        if not hasattr(self, 'rf_models') or len(self.food_data) == 0:
            return []
        
        # Start timer for performance tracking
        start_time = time.time()
        
        # Default to empty list if conditions is None
        if conditions is None:
            conditions = []
        
        # Extract features for all food items
        X = self.food_data[self.features].fillna(0)
        
        # Create user preference vector
        user_prefs_vector = np.array([user_preferences.get(feature, 0) for feature in self.features]).reshape(1, -1)
        
        # Calculate similarity to user preferences (using Euclidean distance)
        # Standardize first to ensure fair comparison
        X_std = self.scaler.transform(X)
        user_prefs_std = self.scaler.transform(user_prefs_vector)
        
        # Calculate distances
        distances = np.sqrt(np.sum((X_std - user_prefs_std) ** 2, axis=1))
        
        # Get all food items with their distances
        all_foods = []
        for i, (distance, food_item) in enumerate(zip(distances, self.food_data.iterrows())):
            # Skip if not matching category filter
            if category_filter != "All" and food_item[1].get('Category', '') != category_filter:
                continue
                
            # Use Random Forest models to predict suitability for each condition
            condition_scores = {}
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                condition_key = condition.replace(' ', '_')
                if condition_key in self.rf_models:
                    rf_info = self.rf_models[condition_key]
                    model = rf_info['model']
                    
                    # Get feature values for this food item
                    food_features = X.iloc[i].values.reshape(1, -1)
                    
                    if rf_info['type'] == 'classifier':
                        # For classifiers, use probability of suitable class (class 1)
                        score = model.predict_proba(food_features)[0][1]
                        # Convert to a score where higher = less suitable (to match KNN logic)
                        condition_scores[condition] = 5 * (1 - score)
                    else:
                        # For regressors, use predicted score directly
                        score = model.predict(food_features)[0]
                        # Convert to 0-5 scale where lower is better
                        condition_scores[condition] = 5 * score
                else:
                    # Fallback to traditional calculation if no model available
                    condition_scores[condition] = self.calculate_condition_score(food_item[1], condition)
            
            # Calculate combined condition score
            combined_score = 0
            if conditions:
                for condition in conditions:
                    combined_score += condition_scores.get(condition, 0)
                combined_score /= len(conditions)  # Average across conditions
            
            # Basic nutritional data with safe extraction
            energy = float(food_item[1].get('Energy(kcal) by calculation', 0))
            protein = float(food_item[1].get('Protein(g)', 0))
            carbs = float(food_item[1].get('CHOCDF (g) Carbohydrate', 0))
            sugar = float(food_item[1].get('SUGAR(g)', 0))
            fiber = float(food_item[1].get('FIBTG (g) Dietary fibre', 0))
            fat = float(food_item[1].get('Fat(g)', 0))
            
            # Determine suitable conditions using RF prediction
            suitable_conditions = []
            for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
                score = condition_scores[condition]
                # Lower threshold as RF gives more precise scores
                if score <= 2.5:  # Threshold for "suitable"
                    suitable_conditions.append(condition)
            
            # Create food item entry
            food_entry = {
                'Name': food_item[1].get('Thai_Name', food_item[1].get('English_Name', f"Food {i}")),
                'Category': food_item[1].get('Category', 'Unknown'),
                'Energy': energy,
                'Protein': protein,
                'Carbs': carbs,
                'Sugar': sugar,
                'Fiber': fiber,
                'Fat': fat,
                'Distance': distance,
                'Combined_Score': combined_score,
                'Condition_Scores': condition_scores,
                'Suitable_For': suitable_conditions,
                'Index': i
            }
            
            all_foods.append(food_entry)
        
        # Sort foods based on condition scores and distance
        if conditions:
            # Sort by condition score (lower is better) then by distance
            all_foods.sort(key=lambda x: (x['Combined_Score'], x['Distance']))
        else:
            # Sort by closest match to preferences
            all_foods.sort(key=lambda x: x['Distance'])
        
        # End timer and record performance
        end_time = time.time()
        execution_time = end_time - start_time
        self.performance_tracker.record_time('RandomForest', execution_time)
        
        # Calculate and record performance metrics
        if all_foods:
            # Record average distance as relevance score
            avg_distance = np.mean([food['Distance'] for food in all_foods[:max_recommendations]])
            self.performance_tracker.record_relevance('RandomForest', avg_distance)
            
            # Calculate and record condition match score
            if conditions:
                condition_match_percentage = np.mean([
                    len([c for c in food['Suitable_For'] if c in conditions]) / len(conditions)
                    for food in all_foods[:max_recommendations]
                ]) * 100
                self.performance_tracker.record_condition_match('RandomForest', condition_match_percentage)
        
        # Return top recommendations
        return all_foods[:max_recommendations]
    
    def get_recommendations(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """Get food recommendations using the selected model"""
        if self.model_type == "RandomForest":
            return self.get_recommendations_rf(user_preferences, conditions, category_filter, max_recommendations)
        else:  # Default to KNN
            return self.get_recommendations_knn(user_preferences, conditions, category_filter, max_recommendations)
    
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
    
    def analyze_food_item(self, food_index):
        """Analyze a food item by its index"""
        if not hasattr(self, 'food_data') or len(self.food_data) <= food_index:
            return {}
        
        food_item = self.food_data.iloc[food_index]
        return self.analyze_nutritional_profile(food_item)
    
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
    
    def get_performance_metrics(self):
        """Get performance metrics for model comparison"""
        return self.performance_tracker