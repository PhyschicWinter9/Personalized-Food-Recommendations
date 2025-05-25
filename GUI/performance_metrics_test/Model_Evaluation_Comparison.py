import json
import time
import warnings
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score,
                           mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings('ignore')

class FoodRecommenderEvaluator:
    """Comprehensive evaluator for food recommendation models"""
    
    def __init__(self, dataset_folder='./datasets'):
        self.dataset_folder = dataset_folder
        self.results = {
            'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_info': {},
            'model_comparisons': {},
            'health_condition_results': {},
            'cross_validation_results': {},
            'feature_importance': {},
            'confusion_matrices': {},
            'training_times': {},
            'prediction_times': {}
        }
        
        # Nutritional features for models
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
        
        self.health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
    
    def load_and_prepare_data(self):
        """Load all food datasets and prepare for evaluation"""
        print("Loading datasets...")
        csv_files = glob.glob(os.path.join(self.dataset_folder, '*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_folder}")
        
        dataframes = []
        for file_path in csv_files:
            try:
                filename = os.path.basename(file_path)
                df = pd.read_csv(file_path)
                
                # Add category from filename
                category = os.path.splitext(filename)[0].replace('_', ' ').title()
                df['Category'] = category
                
                # Clean data
                for col in self.nutritional_features:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                dataframes.append(df)
                print(f"  Loaded {len(df)} items from {filename}")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        # Combine all data
        self.food_data = pd.concat(dataframes, ignore_index=True)
        
        # Calculate health scores
        self._calculate_health_scores()
        
        # Store dataset info
        self.results['dataset_info'] = {
            'total_samples': len(self.food_data),
            'total_features': len(self.nutritional_features),
            'categories': self.food_data['Category'].value_counts().to_dict(),
            'files_loaded': len(dataframes)
        }
        
        print(f"\nTotal samples loaded: {len(self.food_data)}")
        return self.food_data
    
    def _calculate_health_scores(self):
        """Calculate health suitability scores for each food"""
        print("Calculating health condition suitability scores...")
        
        # Initialize score columns
        for condition in self.health_conditions:
            self.food_data[f'{condition}_Suitable'] = 0
        
        for idx, food in self.food_data.iterrows():
            # Diabetes suitability (binary classification)
            sugar = float(food.get('SUGAR(g)', 0))
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
            
            diabetes_suitable = 1 if (sugar <= 8 and (fiber >= 3 or carbs <= 25)) else 0
            self.food_data.at[idx, 'Diabetes_Suitable'] = diabetes_suitable
            
            # Obesity suitability
            calories = float(food.get('Energy(kcal) by calculation', 0))
            protein = float(food.get('Protein(g)', 0))
            fat = float(food.get('Fat(g)', 0))
            
            obesity_suitable = 1 if (calories <= 250 and fat <= 10 and protein >= 5) else 0
            self.food_data.at[idx, 'Obesity_Suitable'] = obesity_suitable
            
            # Hypertension suitability
            sodium = float(food.get('Na(mg)', 0))
            potassium = float(food.get('K(mg)', 0))
            
            hypertension_suitable = 1 if (sodium <= 200 and potassium >= 100) else 0
            self.food_data.at[idx, 'Hypertension_Suitable'] = hypertension_suitable
            
            # High cholesterol suitability
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            cholesterol_suitable = 1 if (sat_fat <= 3 and cholesterol <= 50 and fiber >= 2) else 0
            self.food_data.at[idx, 'High_Cholesterol_Suitable'] = cholesterol_suitable
    
    def prepare_features_and_targets(self):
        """Prepare feature matrix and target variables"""
        # Get available features
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        
        # Add category encoding
        le = LabelEncoder()
        self.food_data['Category_Encoded'] = le.fit_transform(self.food_data['Category'])
        
        # Feature matrix
        feature_columns = available_features + ['Category_Encoded']
        X = self.food_data[feature_columns].fillna(0)
        
        # Target variables (multi-output classification)
        y_columns = [f'{condition}_Suitable' for condition in self.health_conditions]
        y = self.food_data[y_columns]
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_columns
    
    def evaluate_knn(self, X_train, X_test, y_train, y_test):
        """Evaluate K-Nearest Neighbors model"""
        print("\nEvaluating KNN model...")
        
        results = {}
        
        # Test different k values
        k_values = [3, 5, 7, 10, 15]
        best_k = 5
        best_score = 0
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn_multi = MultiOutputClassifier(knn)
            knn_multi.fit(X_train, y_train)
            score = knn_multi.score(X_test, y_test)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        # Train final model with best k
        start_time = time.time()
        knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
        knn_multi = MultiOutputClassifier(knn_model)
        knn_multi.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = knn_multi.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Evaluate each health condition
        for i, condition in enumerate(self.health_conditions):
            y_true_cond = y_test.iloc[:, i]
            y_pred_cond = y_pred[:, i]
            
            results[condition] = {
                'accuracy': accuracy_score(y_true_cond, y_pred_cond),
                'precision': precision_score(y_true_cond, y_pred_cond, zero_division=0),
                'recall': recall_score(y_true_cond, y_pred_cond, zero_division=0),
                'f1_score': f1_score(y_true_cond, y_pred_cond, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true_cond, y_pred_cond).tolist()
            }
        
        # Overall metrics
        overall_accuracy = np.mean([results[cond]['accuracy'] for cond in self.health_conditions])
        
        return {
            'model_name': 'KNN',
            'best_k': best_k,
            'overall_accuracy': overall_accuracy,
            'condition_results': results,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
    
    def evaluate_random_forest(self, X_train, X_test, y_train, y_test, feature_names):
        """Evaluate Random Forest model"""
        print("\nEvaluating Random Forest model...")
        
        results = {}
        
        # Train model
        start_time = time.time()
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_multi = MultiOutputClassifier(rf_model)
        rf_multi.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = rf_multi.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Feature importance (average across all estimators)
        feature_importance = {}
        for i, estimator in enumerate(rf_multi.estimators_):
            importances = estimator.feature_importances_
            for j, feature in enumerate(feature_names):
                if feature not in feature_importance:
                    feature_importance[feature] = 0
                feature_importance[feature] += importances[j] / len(rf_multi.estimators_)
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Evaluate each health condition
        for i, condition in enumerate(self.health_conditions):
            y_true_cond = y_test.iloc[:, i]
            y_pred_cond = y_pred[:, i]
            
            results[condition] = {
                'accuracy': accuracy_score(y_true_cond, y_pred_cond),
                'precision': precision_score(y_true_cond, y_pred_cond, zero_division=0),
                'recall': recall_score(y_true_cond, y_pred_cond, zero_division=0),
                'f1_score': f1_score(y_true_cond, y_pred_cond, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true_cond, y_pred_cond).tolist()
            }
        
        # Overall metrics
        overall_accuracy = np.mean([results[cond]['accuracy'] for cond in self.health_conditions])
        
        return {
            'model_name': 'Random Forest',
            'n_estimators': 100,
            'overall_accuracy': overall_accuracy,
            'condition_results': results,
            'feature_importance': dict(sorted_features[:10]),  # Top 10 features
            'training_time': training_time,
            'prediction_time': prediction_time
        }
    
    def evaluate_mlp(self, X_train, X_test, y_train, y_test):
        """Evaluate Multi-layer Perceptron model"""
        print("\nEvaluating MLP model...")
        
        results = {}
        
        # Train model
        start_time = time.time()
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        mlp_multi = MultiOutputClassifier(mlp_model)
        mlp_multi.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predictions
        start_time = time.time()
        y_pred = mlp_multi.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Evaluate each health condition
        for i, condition in enumerate(self.health_conditions):
            y_true_cond = y_test.iloc[:, i]
            y_pred_cond = y_pred[:, i]
            
            results[condition] = {
                'accuracy': accuracy_score(y_true_cond, y_pred_cond),
                'precision': precision_score(y_true_cond, y_pred_cond, zero_division=0),
                'recall': recall_score(y_true_cond, y_pred_cond, zero_division=0),
                'f1_score': f1_score(y_true_cond, y_pred_cond, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true_cond, y_pred_cond).tolist()
            }
        
        # Overall metrics
        overall_accuracy = np.mean([results[cond]['accuracy'] for cond in self.health_conditions])
        
        return {
            'model_name': 'MLP',
            'architecture': '100-50',
            'overall_accuracy': overall_accuracy,
            'condition_results': results,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
    
    def perform_cross_validation(self, X, y):
        """Perform cross-validation for all models"""
        print("\nPerforming 5-fold cross-validation...")
        
        cv_results = {}
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # For multi-output, we'll use the first target for stratification
        y_stratify = y.iloc[:, 0]
        
        # KNN
        print("  Cross-validating KNN...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_multi = MultiOutputClassifier(knn)
        knn_scores = []
        
        for train_idx, val_idx in kfold.split(X, y_stratify):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            knn_multi.fit(X_train_cv, y_train_cv)
            score = knn_multi.score(X_val_cv, y_val_cv)
            knn_scores.append(score)
        
        cv_results['KNN'] = {
            'mean_score': np.mean(knn_scores),
            'std_score': np.std(knn_scores),
            'scores': knn_scores
        }
        
        # Random Forest
        print("  Cross-validating Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_multi = MultiOutputClassifier(rf)
        rf_scores = []
        
        for train_idx, val_idx in kfold.split(X, y_stratify):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            rf_multi.fit(X_train_cv, y_train_cv)
            score = rf_multi.score(X_val_cv, y_val_cv)
            rf_scores.append(score)
        
        cv_results['Random Forest'] = {
            'mean_score': np.mean(rf_scores),
            'std_score': np.std(rf_scores),
            'scores': rf_scores
        }
        
        # MLP
        print("  Cross-validating MLP...")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp_multi = MultiOutputClassifier(mlp)
        mlp_scores = []
        
        for train_idx, val_idx in kfold.split(X, y_stratify):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            mlp_multi.fit(X_train_cv, y_train_cv)
            score = mlp_multi.score(X_val_cv, y_val_cv)
            mlp_scores.append(score)
        
        cv_results['MLP'] = {
            'mean_score': np.mean(mlp_scores),
            'std_score': np.std(mlp_scores),
            'scores': mlp_scores
        }
        
        return cv_results
    
    def generate_summary_statistics(self, model_results):
        """Generate summary statistics for all models"""
        summary = {
            'best_overall_model': '',
            'best_accuracy': 0,
            'model_rankings': {},
            'condition_best_models': {},
            'average_metrics': {}
        }
        
        # Find best overall model
        for model_name, results in model_results.items():
            accuracy = results['overall_accuracy']
            summary['model_rankings'][model_name] = accuracy
            
            if accuracy > summary['best_accuracy']:
                summary['best_accuracy'] = accuracy
                summary['best_overall_model'] = model_name
        
        # Find best model for each condition
        for condition in self.health_conditions:
            best_model = ''
            best_f1 = 0
            
            for model_name, results in model_results.items():
                f1 = results['condition_results'][condition]['f1_score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
            
            summary['condition_best_models'][condition] = {
                'model': best_model,
                'f1_score': best_f1
            }
        
        # Calculate average metrics
        for model_name, results in model_results.items():
            avg_precision = np.mean([results['condition_results'][c]['precision'] 
                                    for c in self.health_conditions])
            avg_recall = np.mean([results['condition_results'][c]['recall'] 
                                 for c in self.health_conditions])
            avg_f1 = np.mean([results['condition_results'][c]['f1_score'] 
                             for c in self.health_conditions])
            
            summary['average_metrics'][model_name] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1
            }
        
        return summary
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("="*60)
        print("FOOD RECOMMENDATION SYSTEM - MODEL EVALUATION")
        print("="*60)
        
        # Load data
        self.load_and_prepare_data()
        
        # Prepare features and targets
        X, y, feature_names = self.prepare_features_and_targets()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y.iloc[:, 0]
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Evaluate models
        model_results = {}
        
        # KNN
        knn_results = self.evaluate_knn(X_train, X_test, y_train, y_test)
        model_results['KNN'] = knn_results
        self.results['model_comparisons']['KNN'] = knn_results
        
        # Random Forest
        rf_results = self.evaluate_random_forest(X_train, X_test, y_train, y_test, feature_names)
        model_results['Random Forest'] = rf_results
        self.results['model_comparisons']['Random Forest'] = rf_results
        self.results['feature_importance'] = rf_results['feature_importance']
        
        # MLP
        mlp_results = self.evaluate_mlp(X_train, X_test, y_train, y_test)
        model_results['MLP'] = mlp_results
        self.results['model_comparisons']['MLP'] = mlp_results
        
        # Cross-validation
        cv_results = self.perform_cross_validation(X, y)
        self.results['cross_validation_results'] = cv_results
        
        # Generate summary
        summary = self.generate_summary_statistics(model_results)
        self.results['summary'] = summary
        
        # Save training and prediction times
        for model_name, results in model_results.items():
            self.results['training_times'][model_name] = results['training_time']
            self.results['prediction_times'][model_name] = results['prediction_time']
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        return self.results
    
    def save_results(self, filename='model_evaluation_report.json'):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nDataset: {self.results['dataset_info']['total_samples']} samples")
        print(f"Features: {self.results['dataset_info']['total_features']} nutritional features")
        
        print("\nModel Performance (Overall Accuracy):")
        for model, accuracy in self.results['summary']['model_rankings'].items():
            print(f"  {model}: {accuracy:.4f}")
        
        print(f"\nBest Overall Model: {self.results['summary']['best_overall_model']} "
              f"({self.results['summary']['best_accuracy']:.4f})")
        
        print("\nBest Model by Health Condition:")
        for condition, info in self.results['summary']['condition_best_models'].items():
            print(f"  {condition}: {info['model']} (F1: {info['f1_score']:.4f})")
        
        print("\nCross-Validation Results (Mean ± Std):")
        for model, cv_info in self.results['cross_validation_results'].items():
            print(f"  {model}: {cv_info['mean_score']:.4f} ± {cv_info['std_score']:.4f}")
        
        print("\nTraining Times:")
        for model, time_sec in self.results['training_times'].items():
            print(f"  {model}: {time_sec:.4f} seconds")
        
        print("\nTop 5 Important Features (Random Forest):")
        for i, (feature, importance) in enumerate(list(self.results['feature_importance'].items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = FoodRecommenderEvaluator(dataset_folder='./datasets')
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results('food_recommender_evaluation_report.json')
    
    # Also save a detailed report
    detailed_filename = f'detailed_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    evaluator.save_results(detailed_filename)
    
    print(f"\nDetailed report saved to {detailed_filename}")


if __name__ == "__main__":
    main()