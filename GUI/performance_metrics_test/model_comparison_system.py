import glob
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
import warnings
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Import model components
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import euclidean

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

class ModelComparison:
    """Comprehensive comparison of KNN, Random Forest, and MLP models"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.food_data = pd.DataFrame()
        self.models = {}
        self.scalers = {}
        self.features = []
        self.results = {}
        
        # Performance tracking
        self.performance_metrics = {
            'KNN': {},
            'RandomForest': {},
            'MLP': {}
        }
        
        # Recommendation tracking
        self.recommendation_results = {
            'KNN': [],
            'RandomForest': [],
            'MLP': []
        }
        
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
        
    def update_status(self, message):
        """Update status message"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self):
        """Load and prepare dataset for all models"""
        try:
            self.update_status("Loading dataset...")
            
            # Try to find CSV files
            csv_files = glob.glob('./datasets/*.csv')
            if not csv_files:
                csv_files = glob.glob('*.csv')
            
            if not csv_files:
                raise Exception("No CSV files found!")
            
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
                    
                    df = pd.read_csv(file_path)
                    df['Category'] = category
                    df = self._clean_data(df)
                    
                    if len(df) > 0:
                        dataframes.append(df)
                        
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            if dataframes:
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.update_status(f"Loaded {len(self.food_data)} food items")
                self._calculate_health_scores()
                self._prepare_features()
            else:
                raise Exception("No valid data loaded")
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            raise
    
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
        """Calculate health scores for all conditions"""
        self.update_status("Calculating health scores...")
        
        # Initialize score columns
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = 0.0
        
        for idx, food in self.food_data.iterrows():
            # Diabetes score
            score = 0
            sugar = float(food.get('SUGAR(g)', 0))
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
            
            if sugar > 15: score += 3
            elif sugar > 8: score += 2
            elif sugar > 3: score += 1
            
            if carbs > 20 and fiber < 3: score += 2
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'Diabetes_Score'] = max(0, score)
            
            # Obesity score
            score = 0
            calories = float(food.get('Energy(kcal) by calculation', 0))
            protein = float(food.get('Protein(g)', 0))
            fat = float(food.get('Fat(g)', 0))
            
            if calories > 300: score += 3
            elif calories > 200: score += 2
            elif calories > 150: score += 1
            
            if fat > 15: score += 2
            elif fat > 10: score += 1
            
            if sugar > 10: score += 2
            elif sugar > 5: score += 1
            
            if protein >= 10: score -= 1
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'Obesity_Score'] = max(0, score)
            
            # Hypertension score
            score = 0
            sodium = float(food.get('Na(mg)', 0))
            potassium = float(food.get('K(mg)', 0))
            
            if sodium > 400: score += 3
            elif sodium > 200: score += 2
            elif sodium > 100: score += 1
            
            if potassium > 300: score -= 1
            elif potassium > 200: score -= 0.5
            
            if fiber >= 5: score -= 0.5
            
            self.food_data.at[idx, 'Hypertension_Score'] = max(0, score)
            
            # High cholesterol score
            score = 0
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            if sat_fat > 5: score += 3
            elif sat_fat > 3: score += 2
            elif sat_fat > 1: score += 1
            
            if cholesterol > 100: score += 2
            elif cholesterol > 50: score += 1
            
            if fiber >= 5: score -= 1
            elif fiber >= 3: score -= 0.5
            
            self.food_data.at[idx, 'High_Cholesterol_Score'] = max(0, score)
        
        # Calculate overall health score
        self.food_data['Overall_Health_Score'] = (
            self.food_data['Diabetes_Score'] + 
            self.food_data['Obesity_Score'] + 
            self.food_data['Hypertension_Score'] + 
            self.food_data['High_Cholesterol_Score']
        ) / 4
    
    def _prepare_features(self):
        """Prepare features for all models"""
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features = available_features + health_features
        
        self.update_status(f"Prepared {len(self.features)} features")
    
    def train_all_models(self):
        """Train all three models and collect performance metrics"""
        if len(self.food_data) == 0:
            raise Exception("No data available for training")
        
        X = self.food_data[self.features].fillna(0)
        
        # Multiple target variables for comprehensive evaluation
        targets = {
            'Overall_Health': self.food_data['Overall_Health_Score'],
            'Diabetes': self.food_data['Diabetes_Score'],
            'Obesity': self.food_data['Obesity_Score'],
            'Hypertension': self.food_data['Hypertension_Score'],
            'High_Cholesterol': self.food_data['High_Cholesterol_Score']
        }
        
        # Split data
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, X.index, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Train and evaluate each model
        for model_name in ['KNN', 'RandomForest', 'MLP']:
            self.update_status(f"Training {model_name} model...")
            self.performance_metrics[model_name] = {}
            
            for target_name, y in targets.items():
                y_train = y.iloc[indices_train]
                y_test = y.iloc[indices_test]
                
                # Train model
                if model_name == 'KNN':
                    model = self._train_knn_model(X_train_scaled, y_train)
                elif model_name == 'RandomForest':
                    model = self._train_rf_model(X_train, y_train)
                elif model_name == 'MLP':
                    model = self._train_mlp_model(X_train_scaled, y_train)
                
                # Evaluate model
                if model is not None:
                    metrics = self._evaluate_model(model, X_test_scaled if model_name in ['KNN', 'MLP'] else X_test, y_test, model_name)
                    self.performance_metrics[model_name][target_name] = metrics
                    
                    # Store model
                    if target_name not in self.models:
                        self.models[target_name] = {}
                    self.models[target_name][model_name] = model
        
        self.update_status("All models trained successfully!")
    
    def _train_knn_model(self, X_train, y_train):
        """Train KNN model"""
        try:
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor(n_neighbors=5, weights='distance')
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.update_status(f"Error training KNN: {e}")
            return None
    
    def _train_rf_model(self, X_train, y_train):
        """Train Random Forest model"""
        try:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.update_status(f"Error training Random Forest: {e}")
            return None
    
    def _train_mlp_model(self, X_train, y_train):
        """Train MLP model"""
        try:
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            self.update_status(f"Error training MLP: {e}")
            return None
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Regression metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Classification metrics (convert to binary classification)
            y_test_binary = (y_test > y_test.median()).astype(int)
            y_pred_binary = (y_pred > y_test.median()).astype(int)
            
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            
            return {
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'actual': y_test.values
            }
            
        except Exception as e:
            self.update_status(f"Error evaluating {model_name}: {e}")
            return {}
    
    def get_recommendations_comparison(self, user_profile, max_recommendations=10):
        """Get recommendations from all models for comparison"""
        self.update_status("Getting recommendations from all models...")
        
        # Create user vector
        user_vector = self._create_user_vector(user_profile)
        
        for model_name in ['KNN', 'RandomForest', 'MLP']:
            try:
                if model_name == 'KNN':
                    recs = self._get_knn_recommendations(user_vector, max_recommendations)
                elif model_name == 'RandomForest':
                    recs = self._get_rf_recommendations(user_vector, max_recommendations)
                elif model_name == 'MLP':
                    recs = self._get_mlp_recommendations(user_vector, max_recommendations)
                
                self.recommendation_results[model_name] = recs
                
            except Exception as e:
                self.update_status(f"Error getting {model_name} recommendations: {e}")
                self.recommendation_results[model_name] = []
    
    def _create_user_vector(self, user_profile):
        """Create user profile vector"""
        user_vector = np.zeros(len(self.features))
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
        # Sample nutritional targets (simplified)
        target_calories = 500  # Meal target
        target_protein = 25
        target_carbs = 60
        target_sugar = 10
        target_fiber = 8
        target_fat = 15
        target_sodium = 600
        
        if 'Energy(kcal) by calculation' in feature_map:
            user_vector[feature_map['Energy(kcal) by calculation']] = target_calories
        if 'Protein(g)' in feature_map:
            user_vector[feature_map['Protein(g)']] = target_protein
        if 'CHOCDF (g) Carbohydrate' in feature_map:
            user_vector[feature_map['CHOCDF (g) Carbohydrate']] = target_carbs
        if 'SUGAR(g)' in feature_map:
            user_vector[feature_map['SUGAR(g)']] = target_sugar
        if 'FIBTG (g) Dietary fibre' in feature_map:
            user_vector[feature_map['FIBTG (g) Dietary fibre']] = target_fiber
        if 'Fat(g)' in feature_map:
            user_vector[feature_map['Fat(g)']] = target_fat
        if 'Na(mg)' in feature_map:
            user_vector[feature_map['Na(mg)']] = target_sodium
        
        # Set ideal health scores
        for health_feature in ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']:
            if health_feature in feature_map:
                user_vector[feature_map[health_feature]] = 0
        
        return user_vector
    
    def _get_knn_recommendations(self, user_vector, max_recommendations):
        """Get KNN recommendations"""
        if 'Overall_Health' not in self.models or 'KNN' not in self.models['Overall_Health']:
            return []
        
        model = self.models['Overall_Health']['KNN']
        X = self.food_data[self.features].fillna(0)
        X_scaled = self.scalers['main'].transform(X)
        user_vector_scaled = self.scalers['main'].transform(user_vector.reshape(1, -1))
        
        # Find nearest neighbors manually
        distances = []
        for i, food_features in enumerate(X_scaled):
            dist = euclidean(user_vector_scaled[0], food_features)
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        top_indices = [idx for _, idx in distances[:max_recommendations]]
        
        recommendations = []
        for idx in top_indices:
            food = self.food_data.iloc[idx]
            recommendations.append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'score': distances[distances.index((min([d for d, i in distances if i == idx]), idx))][0],
                'model': 'KNN'
            })
        
        return recommendations
    
    def _get_rf_recommendations(self, user_vector, max_recommendations):
        """Get Random Forest recommendations"""
        if 'Overall_Health' not in self.models or 'RandomForest' not in self.models['Overall_Health']:
            return []
        
        model = self.models['Overall_Health']['RandomForest']
        X = self.food_data[self.features].fillna(0)
        
        # Predict health scores for all foods
        predicted_scores = model.predict(X)
        
        # Sort by best predicted health scores
        food_scores = list(zip(predicted_scores, range(len(self.food_data))))
        food_scores.sort(key=lambda x: x[0])  # Lower scores are better
        
        recommendations = []
        for score, idx in food_scores[:max_recommendations]:
            food = self.food_data.iloc[idx]
            recommendations.append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'score': score,
                'model': 'RandomForest'
            })
        
        return recommendations
    
    def _get_mlp_recommendations(self, user_vector, max_recommendations):
        """Get MLP recommendations"""
        if 'Overall_Health' not in self.models or 'MLP' not in self.models['Overall_Health']:
            return []
        
        model = self.models['Overall_Health']['MLP']
        X = self.food_data[self.features].fillna(0)
        X_scaled = self.scalers['main'].transform(X)
        user_vector_scaled = self.scalers['main'].transform(user_vector.reshape(1, -1))
        
        # Predict health scores and calculate combined score
        predicted_scores = model.predict(X_scaled)
        
        combined_scores = []
        for i, food_features in enumerate(X_scaled):
            nutritional_distance = euclidean(user_vector_scaled[0], food_features)
            health_score = predicted_scores[i]
            combined_score = 0.6 * nutritional_distance + 0.4 * health_score
            combined_scores.append((combined_score, i))
        
        combined_scores.sort(key=lambda x: x[0])
        
        recommendations = []
        for score, idx in combined_scores[:max_recommendations]:
            food = self.food_data.iloc[idx]
            recommendations.append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'score': score,
                'model': 'MLP'
            })
        
        return recommendations
    
    def get_comparison_summary(self):
        """Get comprehensive comparison summary"""
        summary = {
            'dataset_info': {
                'total_foods': len(self.food_data),
                'features': len(self.features),
                'categories': len(self.food_data['Category'].unique()) if 'Category' in self.food_data.columns else 0
            },
            'model_performance': self.performance_metrics,
            'recommendations': self.recommendation_results
        }
        
        return summary


class ModelComparisonGUI:
    """GUI for comparing all three models"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Food Recommendation Models Comparison - KNN vs Random Forest vs MLP")
        self.master.geometry("1600x1000")
        self.master.configure(bg="#f8f9fa")
        
        self.comparison = ModelComparison(self.update_status)
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the comparison GUI"""
        # Main container
        main_container = tk.Frame(self.master, bg="#f8f9fa", padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = tk.Frame(main_container, bg="#ffffff", relief="solid", bd=1)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(title_frame, text="🔬 Model Performance Comparison", 
                font=('Segoe UI', 20, 'bold'), fg="#2c3e50", bg="#ffffff", pady=20).pack()
        
        tk.Label(title_frame, text="K-Nearest Neighbors vs Random Forest vs Multi-Layer Perceptron", 
                font=('Segoe UI', 12), fg="#7f8c8d", bg="#ffffff", pady=(0, 20)).pack()
        
        # Control buttons
        button_frame = tk.Frame(main_container, bg="#f8f9fa")
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Button(button_frame, text="🚀 Load Data & Train Models", 
                 font=('Segoe UI', 12, 'bold'), bg="#3498db", fg="white",
                 command=self.train_models, padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="🎯 Get Recommendations", 
                 font=('Segoe UI', 12, 'bold'), bg="#27ae60", fg="white",
                 command=self.get_recommendations, padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📊 Generate Report", 
                 font=('Segoe UI', 12, 'bold'), bg="#e74c3c", fg="white",
                 command=self.generate_report, padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Performance metrics tab
        self.create_performance_tab()
        
        # Recommendations comparison tab
        self.create_recommendations_tab()
        
        # Visual comparison tab
        self.create_visual_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Click 'Load Data & Train Models' to begin comparison")
        status_frame = tk.Frame(main_container, bg="#34495e", relief="sunken", bd=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(20, 0))
        
        tk.Label(status_frame, textvariable=self.status_var, 
                font=('Segoe UI', 10), fg="white", bg="#34495e", pady=5).pack()
    
    def create_performance_tab(self):
        """Create performance metrics comparison tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="📈 Performance Metrics")
        
        # Performance table
        columns = ('Model', 'Target', 'R² Score', 'MSE', 'MAE', 'RMSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score')
        self.perf_tree = ttk.Treeview(perf_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=100, minwidth=80)
        
        # Scrollbars
        perf_v_scroll = ttk.Scrollbar(perf_frame, orient="vertical", command=self.perf_tree.yview)
        perf_h_scroll = ttk.Scrollbar(perf_frame, orient="horizontal", command=self.perf_tree.xview)
        self.perf_tree.configure(yscrollcommand=perf_v_scroll.set, xscrollcommand=perf_h_scroll.set)
        
        self.perf_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        perf_v_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=20)
        perf_h_scroll.pack(side=tk.BOTTOM, fill=tk.X, padx=20)
    
    def create_recommendations_tab(self):
        """Create recommendations comparison tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="🍽️ Recommendations")
        
        # Create three sections for each model
        models_frame = tk.Frame(rec_frame, bg="#f8f9fa")
        models_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.rec_trees = {}
        
        for i, model_name in enumerate(['KNN', 'Random Forest', 'MLP']):
            model_frame = tk.LabelFrame(models_frame, text=f"{model_name} Recommendations", 
                                      font=('Segoe UI', 12, 'bold'), padx=10, pady=10)
            model_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            columns = ('Food', 'Category', 'Calories', 'Protein', 'Score')
            tree = ttk.Treeview(model_frame, columns=columns, show='headings', height=10)
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=80, minwidth=60)
            
            tree.pack(fill=tk.BOTH, expand=True)
            self.rec_trees[model_name] = tree
    
    def create_visual_tab(self):
        """Create visual comparison tab"""
        visual_frame = ttk.Frame(self.notebook)
        self.notebook.add(visual_frame, text="📊 Visual Analysis")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(16, 10), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, visual_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.master.update()
    
    def train_models(self):
        """Train all models and display performance"""
        try:
            self.update_status("Loading dataset...")
            self.comparison.load_data()
            
            self.update_status("Training all models...")
            self.comparison.train_all_models()
            
            self.display_performance_metrics()
            self.update_status("✅ All models trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training models: {str(e)}")
            self.update_status(f"❌ Error: {str(e)}")
    
    def display_performance_metrics(self):
        """Display performance metrics in the table"""
        # Clear previous results
        for item in self.perf_tree.get_children():
            self.perf_tree.delete(item)
        
        # Add performance data
        for model_name, model_metrics in self.comparison.performance_metrics.items():
            for target_name, metrics in model_metrics.items():
                if metrics:  # Only if metrics exist
                    self.perf_tree.insert('', 'end', values=(
                        model_name,
                        target_name,
                        f"{metrics.get('r2_score', 0):.4f}",
                        f"{metrics.get('mse', 0):.4f}",
                        f"{metrics.get('mae', 0):.4f}",
                        f"{metrics.get('rmse', 0):.4f}",
                        f"{metrics.get('accuracy', 0):.4f}",
                        f"{metrics.get('precision', 0):.4f}",
                        f"{metrics.get('recall', 0):.4f}",
                        f"{metrics.get('f1_score', 0):.4f}"
                    ))
    
    def get_recommendations(self):
        """Get recommendations from all models"""
        if not hasattr(self.comparison, 'models') or not self.comparison.models:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            # Sample user profile
            user_profile = {
                'weight': 70,
                'height': 170,
                'age': 30,
                'gender': 'Male',
                'health_conditions': ['Diabetes']
            }
            
            self.update_status("Getting recommendations from all models...")
            self.comparison.get_recommendations_comparison(user_profile, max_recommendations=10)
            
            self.display_recommendations()
            self.create_comparison_visualizations()
            self.update_status("✅ Recommendations generated!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error getting recommendations: {str(e)}")
            self.update_status(f"❌ Error: {str(e)}")
    
    def display_recommendations(self):
        """Display recommendations from all models"""
        for model_name, tree in self.rec_trees.items():
            # Clear previous results
            for item in tree.get_children():
                tree.delete(item)
            
            # Add recommendations
            recommendations = self.comparison.recommendation_results.get(model_name, [])
            for i, rec in enumerate(recommendations, 1):
                tree.insert('', 'end', values=(
                    f"{i}. {rec['name'][:15]}{'...' if len(rec['name']) > 15 else ''}",
                    rec['category'],
                    f"{rec['calories']:.0f}",
                    f"{rec['protein']:.1f}g",
                    f"{rec['score']:.4f}"
                ))
    
    def create_comparison_visualizations(self):
        """Create comparison visualizations"""
        self.fig.clear()
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Comparison
        ax1 = self.fig.add_subplot(gs[0, 0])
        self.plot_model_performance_comparison(ax1)
        
        # 2. R² Score Comparison
        ax2 = self.fig.add_subplot(gs[0, 1])
        self.plot_r2_comparison(ax2)
        
        # 3. Accuracy Comparison
        ax3 = self.fig.add_subplot(gs[0, 2])
        self.plot_accuracy_comparison(ax3)
        
        # 4. Recommendation Score Distribution
        ax4 = self.fig.add_subplot(gs[1, :])
        self.plot_recommendation_scores(ax4)
        
        self.canvas.draw()
    
    def plot_model_performance_comparison(self, ax):
        """Plot overall model performance comparison"""
        models = ['KNN', 'RandomForest', 'MLP']
        metrics = ['r2_score', 'accuracy', 'f1_score']
        
        avg_scores = {model: [] for model in models}
        
        for model in models:
            for metric in metrics:
                scores = []
                model_data = self.comparison.performance_metrics.get(model, {})
                for target_data in model_data.values():
                    if metric in target_data:
                        scores.append(target_data[metric])
                avg_scores[model].append(np.mean(scores) if scores else 0)
        
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            ax.bar(x + i*width, avg_scores[model], width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Average Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_r2_comparison(self, ax):
        """Plot R² score comparison"""
        models = []
        r2_scores = []
        
        for model_name, model_metrics in self.comparison.performance_metrics.items():
            for target_name, metrics in model_metrics.items():
                if 'r2_score' in metrics:
                    models.append(f"{model_name}\n{target_name}")
                    r2_scores.append(metrics['r2_score'])
        
        colors = ['#3498db', '#e74c3c', '#27ae60'] * (len(models) // 3 + 1)
        bars = ax.bar(range(len(models)), r2_scores, color=colors[:len(models)], alpha=0.7)
        
        ax.set_xlabel('Model-Target Combinations')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score Comparison')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    def plot_accuracy_comparison(self, ax):
        """Plot accuracy comparison"""
        models = ['KNN', 'RandomForest', 'MLP']
        avg_accuracies = []
        
        for model in models:
            accuracies = []
            model_data = self.comparison.performance_metrics.get(model, {})
            for target_data in model_data.values():
                if 'accuracy' in target_data:
                    accuracies.append(target_data['accuracy'])
            avg_accuracies.append(np.mean(accuracies) if accuracies else 0)
        
        colors = ['#3498db', '#e74c3c', '#27ae60']
        bars = ax.bar(models, avg_accuracies, color=colors, alpha=0.7)
        
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, avg_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_recommendation_scores(self, ax):
        """Plot recommendation score distributions"""
        models = ['KNN', 'RandomForest', 'MLP']
        colors = ['#3498db', '#e74c3c', '#27ae60']
        
        for i, model in enumerate(models):
            recommendations = self.comparison.recommendation_results.get(model, [])
            if recommendations:
                scores = [rec['score'] for rec in recommendations]
                ax.hist(scores, bins=10, alpha=0.6, label=model, color=colors[i])
        
        ax.set_xlabel('Recommendation Scores')
        ax.set_ylabel('Frequency')
        ax.set_title('Recommendation Score Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        if not hasattr(self.comparison, 'models') or not self.comparison.models:
            messagebox.showwarning("Warning", "Please train models first!")
            return
        
        try:
            summary = self.comparison.get_comparison_summary()
            
            # Create report window
            report_window = tk.Toplevel(self.master)
            report_window.title("Model Comparison Report")
            report_window.geometry("800x600")
            
            # Create text widget with scrollbar
            text_frame = tk.Frame(report_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Generate report content
            report_content = self._generate_report_content(summary)
            text_widget.insert(tk.END, report_content)
            
            # Make text widget read-only
            text_widget.configure(state='disabled')
            
            self.update_status("✅ Report generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating report: {str(e)}")
            self.update_status(f"❌ Error: {str(e)}")
    
    def _generate_report_content(self, summary):
        """Generate detailed report content"""
        report = "="*80 + "\n"
        report += "FOOD RECOMMENDATION MODELS COMPARISON REPORT\n"
        report += "="*80 + "\n\n"
        
        # Dataset info
        report += "DATASET INFORMATION:\n"
        report += "-"*40 + "\n"
        report += f"Total Food Items: {summary['dataset_info']['total_foods']}\n"
        report += f"Number of Features: {summary['dataset_info']['features']}\n"
        report += f"Food Categories: {summary['dataset_info']['categories']}\n\n"
        
        # Model performance summary
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-"*40 + "\n"
        
        for model_name in ['KNN', 'RandomForest', 'MLP']:
            report += f"\n{model_name} MODEL:\n"
            model_metrics = summary['model_performance'].get(model_name, {})
            
            if model_metrics:
                # Calculate averages
                avg_r2 = np.mean([m.get('r2_score', 0) for m in model_metrics.values()])
                avg_accuracy = np.mean([m.get('accuracy', 0) for m in model_metrics.values()])
                avg_precision = np.mean([m.get('precision', 0) for m in model_metrics.values()])
                avg_recall = np.mean([m.get('recall', 0) for m in model_metrics.values()])
                avg_f1 = np.mean([m.get('f1_score', 0) for m in model_metrics.values()])
                
                report += f"  Average R² Score: {avg_r2:.4f}\n"
                report += f"  Average Accuracy: {avg_accuracy:.4f}\n"
                report += f"  Average Precision: {avg_precision:.4f}\n"
                report += f"  Average Recall: {avg_recall:.4f}\n"
                report += f"  Average F1 Score: {avg_f1:.4f}\n"
                
                # Detailed metrics by target
                for target_name, metrics in model_metrics.items():
                    report += f"\n  {target_name} Target:\n"
                    report += f"    R² Score: {metrics.get('r2_score', 0):.4f}\n"
                    report += f"    MSE: {metrics.get('mse', 0):.4f}\n"
                    report += f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n"
                    report += f"    F1 Score: {metrics.get('f1_score', 0):.4f}\n"
        
        # Recommendations comparison
        report += "\n\nRECOMMENDATIONS COMPARISON:\n"
        report += "-"*40 + "\n"
        
        for model_name in ['KNN', 'RandomForest', 'MLP']:
            recommendations = summary['recommendations'].get(model_name, [])
            if recommendations:
                report += f"\n{model_name} TOP 5 RECOMMENDATIONS:\n"
                for i, rec in enumerate(recommendations[:5], 1):
                    report += f"  {i}. {rec['name']} ({rec['category']}) - Score: {rec['score']:.4f}\n"
        
        # Performance ranking
        report += "\n\nPERFORMANCE RANKING:\n"
        report += "-"*40 + "\n"
        
        model_scores = {}
        for model_name in ['KNN', 'RandomForest', 'MLP']:
            model_metrics = summary['model_performance'].get(model_name, {})
            if model_metrics:
                avg_score = np.mean([
                    m.get('r2_score', 0) + m.get('accuracy', 0) + m.get('f1_score', 0)
                    for m in model_metrics.values()
                ]) / 3
                model_scores[model_name] = avg_score
        
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(ranked_models, 1):
            report += f"{i}. {model}: {score:.4f}\n"
        
        # Conclusions
        report += "\n\nCONCLUSIONS:\n"
        report += "-"*40 + "\n"
        
        if ranked_models:
            best_model = ranked_models[0][0]
            report += f"• Best Overall Model: {best_model}\n"
            
            # Model-specific insights
            report += "\n• Model Characteristics:\n"
            report += "  - KNN: Good for similarity-based recommendations, interpretable\n"
            report += "  - Random Forest: Robust, handles feature importance well\n"
            report += "  - MLP: Can capture complex patterns, good for non-linear relationships\n"
            
            report += "\n• Recommendations for Usage:\n"
            report += f"  - For highest accuracy: Use {best_model}\n"
            report += "  - For interpretability: Use KNN or Random Forest\n"
            report += "  - For complex health conditions: Consider MLP\n"
        
        report += "\n" + "="*80 + "\n"
        report += "END OF REPORT\n"
        report += "="*80 + "\n"
        
        return report


def main():
    """Main function to run the comparison application"""
    print("🚀 Starting Model Comparison System...")
    
    root = tk.Tk()
    app = ModelComparisonGUI(root)
    
    # Center window
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - root.winfo_width()) // 2
    y = (screen_height - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()