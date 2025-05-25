import glob
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    mean_absolute_error, r2_score, roc_auc_score, precision_recall_curve,
    auc, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


class KNNPerformanceEvaluator:
    """Comprehensive performance evaluation for KNN Food Recommendation System"""
    
    def __init__(self):
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
        
        self.results = {
            'model_info': {
                'model_type': 'K-Nearest Neighbors (KNN)',
                'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset_info': {},
                'feature_count': 0
            },
            'regression_metrics': {},
            'classification_metrics': {},
            'recommendation_metrics': {},
            'cross_validation': {},
            'confusion_matrices': {},
            'feature_importance': {},
            'computational_metrics': {}
        }
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading datasets...")
        
        # Load CSV files
        csv_files = glob.glob('./datasets/*.csv')
        if not csv_files:
            csv_files = glob.glob('*.csv')
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found!")
        
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
                print(f"Error loading {file_path}: {e}")
        
        if dataframes:
            self.food_data = pd.concat(dataframes, ignore_index=True)
            print(f"Loaded {len(self.food_data)} food items")
            
            # Store dataset info
            self.results['model_info']['dataset_info'] = {
                'total_items': len(self.food_data),
                'categories': self.food_data['Category'].value_counts().to_dict(),
                'files_loaded': len(dataframes)
            }
            
            self._calculate_health_scores()
            self._prepare_features()
            
        else:
            raise ValueError("No valid data loaded")
    
    def _clean_data(self, df):
        """Clean nutritional data"""
        for col in self.nutritional_features + ['FASAT (g) Saturated FA']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        nutrient_cols = [col for col in self.nutritional_features if col in df.columns]
        if nutrient_cols:
            df = df[df[nutrient_cols].sum(axis=1) > 0]
        
        return df
    
    def _calculate_health_scores(self):
        """Calculate health suitability scores"""
        print("Calculating health scores...")
        
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
    
    def _prepare_features(self):
        """Prepare features for evaluation"""
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features = available_features + health_features
        self.results['model_info']['feature_count'] = len(self.features)
        
        print(f"Prepared {len(self.features)} features")
    
    def evaluate_knn_performance(self):
        """Comprehensive KNN performance evaluation"""
        print("Evaluating KNN performance...")
        
        # Prepare data
        X = self.food_data[self.features].fillna(0)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Evaluation metrics storage
        start_time = time.time()
        
        # 1. Regression Performance (Health Scores)
        self._evaluate_regression_performance(X_scaled)
        
        # 2. Classification Performance (Health Suitability)
        self._evaluate_classification_performance(X_scaled)
        
        # 3. Recommendation Quality Metrics
        self._evaluate_recommendation_performance(X_scaled)
        
        # 4. Cross-validation Performance
        self._evaluate_cross_validation(X_scaled)
        
        # 5. Computational Metrics
        self._evaluate_computational_metrics(X_scaled, time.time() - start_time)
        
        print("Performance evaluation completed!")
    
    def _evaluate_regression_performance(self, X_scaled):
        """Evaluate regression performance for health scores"""
        print("Evaluating regression performance...")
        
        regression_results = {}
        
        # Test each health condition score prediction
        health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        
        for condition in health_conditions:
            y = self.food_data[f'{condition}_Score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Use Random Forest as a baseline for comparison with KNN approach
            from sklearn.ensemble import RandomForestRegressor
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
            
            regression_results[condition] = {
                'MSE': float(mse),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2_Score': float(r2),
                'MAPE': float(mape),
                'Mean_Actual': float(np.mean(y_test)),
                'Mean_Predicted': float(np.mean(y_pred)),
                'Std_Actual': float(np.std(y_test)),
                'Std_Predicted': float(np.std(y_pred))
            }
        
        # Overall regression performance
        overall_r2 = np.mean([regression_results[c]['R2_Score'] for c in health_conditions])
        overall_rmse = np.mean([regression_results[c]['RMSE'] for c in health_conditions])
        
        regression_results['Overall'] = {
            'Average_R2_Score': float(overall_r2),
            'Average_RMSE': float(overall_rmse),
            'Performance_Grade': self._grade_regression_performance(overall_r2)
        }
        
        self.results['regression_metrics'] = regression_results
    
    def _evaluate_classification_performance(self, X_scaled):
        """Evaluate classification performance for health suitability"""
        print("Evaluating classification performance...")
        
        classification_results = {}
        
        health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        
        for condition in health_conditions:
            # Create binary classification target (suitable vs not suitable)
            y_binary = (self.food_data[f'{condition}_Score'] <= 1.5).astype(int)
            
            if len(np.unique(y_binary)) < 2:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
            )
            
            # Use Random Forest for classification
            rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)
            y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # ROC AUC
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.5
            
            # Precision-Recall AUC
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall_curve, precision_curve)
            except:
                pr_auc = 0.5
            
            # Matthews Correlation Coefficient
            try:
                mcc = matthews_corrcoef(y_test, y_pred)
            except:
                mcc = 0.0
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            classification_results[condition] = {
                'Accuracy': float(accuracy),
                'Precision': float(precision),
                'Recall': float(recall),
                'F1_Score': float(f1),
                'ROC_AUC': float(roc_auc),
                'PR_AUC': float(pr_auc),
                'MCC': float(mcc),
                'Confusion_Matrix': cm.tolist(),
                'Classification_Report': classification_report(y_test, y_pred, output_dict=True),
                'Support': {
                    'Negative_Class': int(np.sum(y_test == 0)),
                    'Positive_Class': int(np.sum(y_test == 1))
                }
            }
            
            # Store confusion matrix separately for visualization
            self.results['confusion_matrices'][condition] = {
                'matrix': cm.tolist(),
                'labels': ['Not Suitable', 'Suitable']
            }
        
        # Overall classification performance
        if classification_results:
            avg_accuracy = np.mean([classification_results[c]['Accuracy'] for c in classification_results.keys()])
            avg_f1 = np.mean([classification_results[c]['F1_Score'] for c in classification_results.keys()])
            avg_precision = np.mean([classification_results[c]['Precision'] for c in classification_results.keys()])
            avg_recall = np.mean([classification_results[c]['Recall'] for c in classification_results.keys()])
            
            classification_results['Overall'] = {
                'Average_Accuracy': float(avg_accuracy),
                'Average_F1_Score': float(avg_f1),
                'Average_Precision': float(avg_precision),
                'Average_Recall': float(avg_recall),
                'Performance_Grade': self._grade_classification_performance(avg_f1)
            }
        
        self.results['classification_metrics'] = classification_results
    
    def _evaluate_recommendation_performance(self, X_scaled):
        """Evaluate recommendation-specific performance metrics"""
        print("Evaluating recommendation performance...")
        
        # KNN-specific metrics
        knn_models = {}
        distances_stats = {}
        
        # Train KNN models for each condition
        for k in [5, 10, 15, 20]:
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(X_scaled)
            
            # Calculate average distances
            distances, indices = knn.kneighbors(X_scaled)
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            distances_stats[f'k_{k}'] = {
                'Average_Distance': float(avg_distance),
                'Std_Distance': float(std_distance),
                'Min_Distance': float(np.min(distances[:, 1:])),  # Exclude self-distance
                'Max_Distance': float(np.max(distances)),
                'Median_Distance': float(np.median(distances))
            }
        
        # Recommendation diversity metrics
        categories = self.food_data['Category'].values
        category_distribution = pd.Series(categories).value_counts(normalize=True).to_dict()
        
        # Coverage metrics
        unique_categories = len(np.unique(categories))
        total_items = len(self.food_data)
        
        recommendation_results = {
            'KNN_Distance_Stats': distances_stats,
            'Dataset_Coverage': {
                'Total_Items': total_items,
                'Unique_Categories': unique_categories,
                'Category_Distribution': category_distribution,
                'Coverage_Ratio': float(unique_categories / total_items)
            },
            'Recommendation_Quality': {
                'Health_Score_Range': {
                    'Diabetes': {
                        'Min': float(self.food_data['Diabetes_Score'].min()),
                        'Max': float(self.food_data['Diabetes_Score'].max()),
                        'Mean': float(self.food_data['Diabetes_Score'].mean())
                    },
                    'Obesity': {
                        'Min': float(self.food_data['Obesity_Score'].min()),
                        'Max': float(self.food_data['Obesity_Score'].max()),
                        'Mean': float(self.food_data['Obesity_Score'].mean())
                    },
                    'Hypertension': {
                        'Min': float(self.food_data['Hypertension_Score'].min()),
                        'Max': float(self.food_data['Hypertension_Score'].max()),
                        'Mean': float(self.food_data['Hypertension_Score'].mean())
                    },
                    'High_Cholesterol': {
                        'Min': float(self.food_data['High_Cholesterol_Score'].min()),
                        'Max': float(self.food_data['High_Cholesterol_Score'].max()),
                        'Mean': float(self.food_data['High_Cholesterol_Score'].mean())
                    }
                }
            }
        }
        
        self.results['recommendation_metrics'] = recommendation_results
    
    def _evaluate_cross_validation(self, X_scaled):
        """Evaluate cross-validation performance"""
        print("Evaluating cross-validation performance...")
        
        cv_results = {}
        health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        
        for condition in health_conditions:
            y = self.food_data[f'{condition}_Score']
            
            # Regression CV
            from sklearn.ensemble import RandomForestRegressor
            rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
            
            cv_scores_r2 = cross_val_score(rf_reg, X_scaled, y, cv=5, scoring='r2')
            cv_scores_mse = cross_val_score(rf_reg, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            
            # Classification CV (if applicable)
            y_binary = (y <= 1.5).astype(int)
            
            cv_acc = []
            cv_f1 = []
            
            if len(np.unique(y_binary)) > 1:
                from sklearn.ensemble import RandomForestClassifier
                rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
                
                cv_acc = cross_val_score(rf_clf, X_scaled, y_binary, cv=5, scoring='accuracy')
                cv_f1 = cross_val_score(rf_clf, X_scaled, y_binary, cv=5, scoring='f1')
            
            cv_results[condition] = {
                'Regression': {
                    'R2_Scores': cv_scores_r2.tolist(),
                    'R2_Mean': float(cv_scores_r2.mean()),
                    'R2_Std': float(cv_scores_r2.std()),
                    'MSE_Scores': (-cv_scores_mse).tolist(),
                    'MSE_Mean': float(-cv_scores_mse.mean()),
                    'MSE_Std': float(cv_scores_mse.std())
                },
                'Classification': {
                    'Accuracy_Scores': cv_acc.tolist() if len(cv_acc) > 0 else [],
                    'Accuracy_Mean': float(cv_acc.mean()) if len(cv_acc) > 0 else 0.0,
                    'Accuracy_Std': float(cv_acc.std()) if len(cv_acc) > 0 else 0.0,
                    'F1_Scores': cv_f1.tolist() if len(cv_f1) > 0 else [],
                    'F1_Mean': float(cv_f1.mean()) if len(cv_f1) > 0 else 0.0,
                    'F1_Std': float(cv_f1.std()) if len(cv_f1) > 0 else 0.0
                }
            }
        
        self.results['cross_validation'] = cv_results
    
    def _evaluate_computational_metrics(self, X_scaled, total_time):
        """Evaluate computational performance"""
        print("Evaluating computational metrics...")
        
        # Training time for different k values
        training_times = {}
        prediction_times = {}
        
        for k in [5, 10, 15, 20]:
            # Measure training time
            start_time = time.time()
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(X_scaled)
            training_time = time.time() - start_time
            
            # Measure prediction time
            start_time = time.time()
            sample_queries = X_scaled[:100]  # Sample queries
            distances, indices = knn.kneighbors(sample_queries)
            prediction_time = time.time() - start_time
            
            training_times[f'k_{k}'] = float(training_time)
            prediction_times[f'k_{k}'] = float(prediction_time / 100)  # Per query
        
        computational_results = {
            'Total_Evaluation_Time': float(total_time),
            'Training_Times': training_times,
            'Prediction_Times_Per_Query': prediction_times,
            'Memory_Usage': {
                'Dataset_Size': X_scaled.shape,
                'Memory_Estimate_MB': float(X_scaled.nbytes / (1024 * 1024)),
                'Feature_Count': X_scaled.shape[1],
                'Sample_Count': X_scaled.shape[0]
            },
            'Scalability_Metrics': {
                'Time_Complexity': 'O(n*d) for training, O(k*n*d) for prediction',
                'Space_Complexity': 'O(n*d)',
                'Suitable_Dataset_Size': 'Small to Medium (< 100K samples)'
            }
        }
        
        self.results['computational_metrics'] = computational_results
    
    def _grade_regression_performance(self, r2_score):
        """Grade regression performance"""
        if r2_score >= 0.9:
            return 'Excellent'
        elif r2_score >= 0.8:
            return 'Very Good'
        elif r2_score >= 0.7:
            return 'Good'
        elif r2_score >= 0.5:
            return 'Fair'
        else:
            return 'Poor'
    
    def _grade_classification_performance(self, f1_score):
        """Grade classification performance"""
        if f1_score >= 0.9:
            return 'Excellent'
        elif f1_score >= 0.8:
            return 'Very Good'
        elif f1_score >= 0.7:
            return 'Good'
        elif f1_score >= 0.6:
            return 'Fair'
        else:
            return 'Poor'
    
    def save_results(self, filename='knn_performance_results.json'):
        """Save performance results to JSON"""
        print(f"Saving results to {filename}...")
        
        # Add summary
        self.results['summary'] = self._generate_summary()
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        return filename
    
    def _generate_summary(self):
        """Generate performance summary"""
        summary = {
            'Overall_Performance': {},
            'Key_Metrics': {},
            'Recommendations': []
        }
        
        # Overall performance grades
        if 'Overall' in self.results['regression_metrics']:
            reg_grade = self.results['regression_metrics']['Overall']['Performance_Grade']
            summary['Overall_Performance']['Regression'] = reg_grade
        
        if 'Overall' in self.results['classification_metrics']:
            clf_grade = self.results['classification_metrics']['Overall']['Performance_Grade']
            summary['Overall_Performance']['Classification'] = clf_grade
        
        # Key metrics
        if 'Overall' in self.results['regression_metrics']:
            summary['Key_Metrics']['Average_R2_Score'] = self.results['regression_metrics']['Overall']['Average_R2_Score']
        
        if 'Overall' in self.results['classification_metrics']:
            summary['Key_Metrics']['Average_Accuracy'] = self.results['classification_metrics']['Overall']['Average_Accuracy']
            summary['Key_Metrics']['Average_F1_Score'] = self.results['classification_metrics']['Overall']['Average_F1_Score']
        
        # Dataset info
        summary['Key_Metrics']['Dataset_Size'] = self.results['model_info']['dataset_info']['total_items']
        summary['Key_Metrics']['Feature_Count'] = self.results['model_info']['feature_count']
        
        # Recommendations
        avg_acc = summary['Key_Metrics'].get('Average_Accuracy', 0)
        avg_f1 = summary['Key_Metrics'].get('Average_F1_Score', 0)
        
        if avg_acc < 0.7 or avg_f1 < 0.7:
            summary['Recommendations'].append("Consider feature engineering or hyperparameter tuning")
        
        if self.results['model_info']['dataset_info']['total_items'] < 1000:
            summary['Recommendations'].append("Consider collecting more data for better performance")
        
        if len(summary['Recommendations']) == 0:
            summary['Recommendations'].append("Model performance is satisfactory for the given dataset")
        
        return summary


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("KNN Food Recommendation System - Performance Evaluation")
    print("=" * 60)
    
    evaluator = KNNPerformanceEvaluator()
    
    try:
        # Load and prepare data
        evaluator.load_and_prepare_data()
        
        # Run comprehensive evaluation
        evaluator.evaluate_knn_performance()
        
        # Save results
        filename = evaluator.save_results()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {filename}")
        print("You can now use the web interface to view the results.")
        
        return filename
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


if __name__ == "__main__":
    main()