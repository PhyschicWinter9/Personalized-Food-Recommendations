import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class ModelPerformanceComparison:
    """Class to compare KNN and Random Forest models for food recommendation"""
    
    def __init__(self, food_data, features, target_conditions=None):
        """
        Initialize the comparison class
        
        Parameters:
        -----------
        food_data : pandas DataFrame
            The food dataset containing nutritional information
        features : list
            List of feature column names to use for the models
        target_conditions : list, optional
            List of target condition columns (e.g., "Diabetes", "Obesity")
        """
        self.food_data = food_data
        self.features = [f for f in features if f in food_data.columns]
        
        # Default to major NCDs if no target conditions specified
        self.target_conditions = target_conditions or ["Diabetes", "Obesity", "Hypertension", "High Cholesterol"]
        self.available_targets = [t for t in self.target_conditions if t in food_data.columns]
        
        # Performance metrics storage
        self.metrics = {
            "knn": {"training_time": 0, "prediction_time": 0, "memory_usage": 0},
            "random_forest": {"training_time": 0, "prediction_time": 0, "memory_usage": 0, 
                             "precision": 0, "recall": 0, "f1": 0, "mse": 0}
        }
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for model training and evaluation"""
        # Extract features
        X = self.food_data[self.features].fillna(0)
        
        # Standardize for KNN (not needed for Random Forest)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.X = X  # Original features for Random Forest
        
        # Prepare targets if available
        self.y_dict = {}
        for condition in self.available_targets:
            if condition in self.food_data.columns:
                # Convert target to numeric if not already
                self.y_dict[condition] = pd.to_numeric(
                    self.food_data[condition], errors='coerce').fillna(0)
        
        # Create binary targets for classification
        self.y_binary_dict = {}
        for condition, y in self.y_dict.items():
            # Consider ratings of 0 and 1 as "suitable" (1), others as "not suitable" (0)
            self.y_binary_dict[condition] = (y <= 1).astype(int)
        
        # Split data for training and evaluation
        self.X_train, self.X_test, self.y_train_dict, self.y_test_dict = {}, {}, {}, {}
        for condition in self.available_targets:
            if condition in self.y_dict:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X, self.y_dict[condition], test_size=0.2, random_state=42)
                
                self.X_train[condition] = X_train
                self.X_test[condition] = X_test
                self.y_train_dict[condition] = y_train
                self.y_test_dict[condition] = y_test
        
        # Also split binary targets
        self.y_binary_train_dict, self.y_binary_test_dict = {}, {}
        for condition in self.available_targets:
            if condition in self.y_binary_dict:
                _, _, y_train, y_test = train_test_split(
                    self.X, self.y_binary_dict[condition], test_size=0.2, random_state=42)
                
                self.y_binary_train_dict[condition] = y_train
                self.y_binary_test_dict[condition] = y_test
    
    def train_knn_model(self, n_neighbors=5):
        """Train KNN model and measure performance"""
        print("Training KNN model...")
        start_time = time.time()
        
        # Train KNN model
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        self.knn_model.fit(self.X_scaled)
        
        # Record training time
        self.metrics["knn"]["training_time"] = time.time() - start_time
        self.metrics["knn"]["memory_usage"] = self._estimate_model_size(self.knn_model)
        
        print(f"KNN model trained in {self.metrics['knn']['training_time']:.3f} seconds")
        return self.knn_model
    
    def train_random_forest_models(self, n_estimators=100, max_depth=10):
        """Train Random Forest models for each condition and measure performance"""
        print("Training Random Forest models...")
        self.rf_models = {}
        self.rf_binary_models = {}
        
        start_time = time.time()
        total_memory = 0
        
        for condition in self.available_targets:
            if condition in self.y_train_dict:
                # Train regressor for scoring
                print(f"Training Random Forest regressor for {condition}...")
                rf_model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    random_state=42
                )
                rf_model.fit(self.X_train[condition], self.y_train_dict[condition])
                self.rf_models[condition] = rf_model
                total_memory += self._estimate_model_size(rf_model)
                
                # Train classifier for binary suitability
                print(f"Training Random Forest classifier for {condition}...")
                rf_binary_model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    random_state=42
                )
                rf_binary_model.fit(self.X_train[condition], self.y_binary_train_dict[condition])
                self.rf_binary_models[condition] = rf_binary_model
                total_memory += self._estimate_model_size(rf_binary_model)
        
        # Record training time and memory usage
        self.metrics["random_forest"]["training_time"] = time.time() - start_time
        self.metrics["random_forest"]["memory_usage"] = total_memory
        
        print(f"Random Forest models trained in {self.metrics['random_forest']['training_time']:.3f} seconds")
        return self.rf_models, self.rf_binary_models
    
    def evaluate_models(self, n_test_samples=20):
        """Evaluate both models using test data"""
        if not hasattr(self, 'knn_model') or not hasattr(self, 'rf_models'):
            print("Models must be trained before evaluation")
            return
        
        print("Evaluating models...")
        
        # Sample test data
        sample_indices = np.random.choice(len(self.X_test[self.available_targets[0]]), 
                                         min(n_test_samples, len(self.X_test[self.available_targets[0]])), 
                                         replace=False)
        
        # Evaluate KNN
        print("Evaluating KNN model...")
        start_time = time.time()
        
        X_sample_scaled = self.scaler.transform(
            self.X_test[self.available_targets[0]].iloc[sample_indices])
        
        _, _ = self.knn_model.kneighbors(X_sample_scaled)
        
        self.metrics["knn"]["prediction_time"] = (time.time() - start_time) / n_test_samples
        
        # Evaluate Random Forest
        print("Evaluating Random Forest models...")
        start_time = time.time()
        rf_predictions = {}
        rf_binary_predictions = {}
        
        for condition in self.available_targets:
            if condition in self.rf_models:
                X_sample = self.X_test[condition].iloc[sample_indices]
                
                # Score predictions
                rf_predictions[condition] = self.rf_models[condition].predict(X_sample)
                
                # Binary suitability predictions
                rf_binary_predictions[condition] = self.rf_binary_models[condition].predict(X_sample)
        
        self.metrics["random_forest"]["prediction_time"] = (time.time() - start_time) / (n_test_samples * len(self.available_targets))
        
        # Calculate additional RF metrics
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mse_scores = []
        
        for condition in self.available_targets:
            if condition in self.rf_models:
                # MSE for regression
                y_true = self.y_test_dict[condition].iloc[sample_indices]
                y_pred = rf_predictions[condition]
                mse = mean_squared_error(y_true, y_pred)
                mse_scores.append(mse)
                
                # Classification metrics
                y_true_binary = self.y_binary_test_dict[condition].iloc[sample_indices]
                y_pred_binary = rf_binary_predictions[condition]
                
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
        
        # Average metrics across conditions
        self.metrics["random_forest"]["precision"] = np.mean(precision_scores) if precision_scores else 0
        self.metrics["random_forest"]["recall"] = np.mean(recall_scores) if recall_scores else 0
        self.metrics["random_forest"]["f1"] = np.mean(f1_scores) if f1_scores else 0
        self.metrics["random_forest"]["mse"] = np.mean(mse_scores) if mse_scores else 0
        
        print("Evaluation complete")
        return self.metrics
    
    def _estimate_model_size(self, model):
        """Rough estimate of model memory usage in MB"""
        import sys
        try:
            size_bytes = sys.getsizeof(model)
            return size_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 0
    
    def plot_performance_comparison(self):
        """Plot comparison of KNN vs Random Forest performance"""
        if not self.metrics["knn"]["training_time"] or not self.metrics["random_forest"]["training_time"]:
            print("Models must be evaluated before plotting performance")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('KNN vs Random Forest Performance Comparison', fontsize=16)
        
        # Training time comparison
        axes[0, 0].bar(['KNN', 'Random Forest'], 
                      [self.metrics["knn"]["training_time"], 
                       self.metrics["random_forest"]["training_time"]], 
                      color=['blue', 'green'])
        axes[0, 0].set_title('Training Time (seconds)')
        axes[0, 0].set_ylabel('Seconds')
        
        # Prediction time comparison
        axes[0, 1].bar(['KNN', 'Random Forest'], 
                      [self.metrics["knn"]["prediction_time"] * 1000, 
                       self.metrics["random_forest"]["prediction_time"] * 1000], 
                      color=['blue', 'green'])
        axes[0, 1].set_title('Prediction Time per Sample (milliseconds)')
        axes[0, 1].set_ylabel('Milliseconds')
        
        # Memory usage comparison
        axes[1, 0].bar(['KNN', 'Random Forest'], 
                      [self.metrics["knn"]["memory_usage"], 
                       self.metrics["random_forest"]["memory_usage"]], 
                      color=['blue', 'green'])
        axes[1, 0].set_title('Model Size (MB)')
        axes[1, 0].set_ylabel('Megabytes')
        
        # Random Forest specific metrics
        if self.metrics["random_forest"]["precision"] > 0:
            rf_metrics = [
                self.metrics["random_forest"]["precision"],
                self.metrics["random_forest"]["recall"],
                self.metrics["random_forest"]["f1"]
            ]
            axes[1, 1].bar(['Precision', 'Recall', 'F1 Score'], rf_metrics, color='green')
            axes[1, 1].set_title('Random Forest Classification Metrics')
            axes[1, 1].set_ylim(0, 1)
            
            # Add MSE text
            mse_text = f"MSE: {self.metrics['random_forest']['mse']:.4f}"
            axes[1, 1].text(0.5, -0.15, mse_text, ha='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig

    def get_feature_importance(self):
        """Get feature importance from Random Forest models"""
        if not hasattr(self, 'rf_models') or not self.rf_models:
            print("Random Forest models must be trained first")
            return None
        
        # Average feature importance across all condition models
        importance_dict = {}
        
        for condition, model in self.rf_models.items():
            importances = model.feature_importances_
            
            for feature, importance in zip(self.features, importances):
                if feature not in importance_dict:
                    importance_dict[feature] = []
                importance_dict[feature].append(importance)
        
        # Average importances across conditions
        avg_importances = {feature: np.mean(values) for feature, values in importance_dict.items()}
        
        # Sort by importance
        sorted_importances = {k: v for k, v in sorted(avg_importances.items(), 
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        
        return sorted_importances
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance from Random Forest models"""
        importances = self.get_feature_importance()
        
        if not importances:
            return None
        
        # Get top N features
        top_features = list(importances.keys())[:top_n]
        top_importances = [importances[f] for f in top_features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {len(top_features)} Most Important Features')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        
        return plt.gcf()


class RandomForestRecommender:
    """Food recommendation system using Random Forest models"""
    
    def __init__(self, food_data, features, condition_cols=None):
        """
        Initialize the Random Forest recommender
        
        Parameters:
        -----------
        food_data : pandas DataFrame
            The food dataset containing nutritional information
        features : list
            List of feature column names to use for the models
        condition_cols : list, optional
            List of health condition columns
        """
        self.food_data = food_data
        self.features = [f for f in features if f in food_data.columns]
        
        # Default condition columns if not specified
        self.condition_cols = condition_cols or [
            'Diabetes', 'Obesity', 'Hypertension', 'High Cholesterol'
        ]
        self.available_conditions = [c for c in self.condition_cols if c in food_data.columns]
        
        # Initialize models
        self.rf_models = {}
        self.rf_binary_models = {}
        self.scaler = StandardScaler()
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for model training"""
        # Process features
        self.X = self.food_data[self.features].fillna(0)
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Process targets for each condition
        self.y_dict = {}
        self.y_binary_dict = {}
        
        for condition in self.available_conditions:
            if condition in self.food_data.columns:
                # Use raw scores for regression
                self.y_dict[condition] = pd.to_numeric(
                    self.food_data[condition], errors='coerce').fillna(0)
                
                # Create binary targets for classification (0-1 = suitable, >1 = not suitable)
                self.y_binary_dict[condition] = (self.y_dict[condition] <= 1).astype(int)
        
    def train_models(self, n_estimators=100, max_depth=10):
        """Train Random Forest models for all conditions"""
        print("Training Random Forest models...")
        
        for condition in self.available_conditions:
            if condition in self.y_dict:
                # Train regressor for scoring
                print(f"Training model for {condition}...")
                rf_model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    random_state=42
                )
                rf_model.fit(self.X, self.y_dict[condition])
                self.rf_models[condition] = rf_model
                
                # Train classifier for binary suitability
                rf_binary_model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    random_state=42
                )
                rf_binary_model.fit(self.X, self.y_binary_dict[condition])
                self.rf_binary_models[condition] = rf_binary_model
                
                print(f"Model for {condition} trained successfully")
        
        return self.rf_models
    
    def predict_condition_scores(self, nutritional_values):
        """
        Predict condition scores for a food item
        
        Parameters:
        -----------
        nutritional_values : dict or pandas Series
            Nutritional values for a food item
        
        Returns:
        --------
        dict
            Predicted scores and suitability for each condition
        """
        if not self.rf_models:
            raise ValueError("Models must be trained before prediction")
        
        # Convert input to proper format
        if isinstance(nutritional_values, dict):
            # Extract only features used by the model
            feature_values = [nutritional_values.get(feature, 0) for feature in self.features]
            X_input = np.array([feature_values])
        else:
            # Assume it's a pandas Series
            X_input = np.array([[nutritional_values.get(feature, 0) for feature in self.features]])
        
        # Make predictions for each condition
        predictions = {}
        
        for condition in self.available_conditions:
            if condition in self.rf_models:
                # Predict score
                score = float(self.rf_models[condition].predict(X_input)[0])
                
                # Predict binary suitability
                suitable = bool(self.rf_binary_models[condition].predict(X_input)[0])
                
                predictions[condition] = {
                    'score': score,
                    'suitable': suitable
                }
        
        return predictions
    
    def get_recommendations(self, user_preferences, conditions=None, category_filter="All", max_recommendations=10):
        """
        Get food recommendations based on user preferences and health conditions
        
        Parameters:
        -----------
        user_preferences : dict
            User's nutritional preferences
        conditions : list, optional
            List of health conditions to consider
        category_filter : str, optional
            Filter for food category
        max_recommendations : int, optional
            Maximum number of recommendations to return
            
        Returns:
        --------
        list
            Recommended food items
        """
        if not self.rf_models:
            raise ValueError("Models must be trained before getting recommendations")
        
        # Default to empty list if conditions is None
        if conditions is None:
            conditions = []
        
        # Extract feature values from user preferences
        user_values = np.array([[
            user_preferences.get(feature, 0) for feature in self.features
        ]])
        
        # Calculate nutritional similarity scores for all foods
        similarity_scores = np.zeros(len(self.food_data))
        
        for i, (_, food_item) in enumerate(self.food_data.iterrows()):
            food_values = np.array([[food_item.get(feature, 0) for feature in self.features]])
            # Calculate Euclidean distance (lower is more similar)
            similarity_scores[i] = -np.sqrt(np.sum((user_values - food_values) ** 2))
        
        # Get all indices sorted by similarity
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
        
        # Get recommended food items
        recommendations = []
        
        for idx in sorted_indices:
            food_item = self.food_data.iloc[idx]
            
            # Skip if not matching category filter
            if category_filter != "All" and food_item.get('Category', '') != category_filter:
                continue
            
            # Predict condition scores
            condition_scores = self.predict_condition_scores(food_item)
            
            # Determine suitable conditions
            suitable_conditions = []
            
            for condition, prediction in condition_scores.items():
                if prediction['suitable']:
                    suitable_conditions.append(condition)
            
            # Calculate combined condition score
            combined_score = 0
            
            if conditions:
                relevant_scores = []
                for condition in conditions:
                    if condition in condition_scores:
                        relevant_scores.append(condition_scores[condition]['score'])
                
                if relevant_scores:
                    combined_score = np.mean(relevant_scores)
            
            # Get basic nutritional data
            nutritional_data = {
                'Name': food_item.get('Thai_Name', food_item.get('English_Name', f"Food {idx}")),
                'Category': food_item.get('Category', 'Unknown'),
                'Energy': float(food_item.get('Energy(kcal) by calculation', 0)),
                'Protein': float(food_item.get('Protein(g)', 0)),
                'Carbs': float(food_item.get('CHOCDF (g) Carbohydrate', 0)),
                'Sugar': float(food_item.get('SUGAR(g)', 0)),
                'Fiber': float(food_item.get('FIBTG (g) Dietary fibre', 0)),
                'Fat': float(food_item.get('Fat(g)', 0)),
                'Distance': -similarity_scores[idx],  # Convert back to a distance
                'Combined_Score': combined_score,
                'Condition_Scores': {k: v['score'] for k, v in condition_scores.items()},
                'Suitable_For': suitable_conditions
            }
            
            recommendations.append(nutritional_data)
            
            # Stop once we have enough recommendations
            if len(recommendations) >= max_recommendations:
                break
        
        # Sort recommendations based on conditions
        if conditions:
            # Sort by condition score (lower is better)
            recommendations.sort(key=lambda x: (x['Combined_Score'], x['Distance']))
        else:
            # Sort by similarity (lower distance is better)
            recommendations.sort(key=lambda x: x['Distance'])
        
        return recommendations