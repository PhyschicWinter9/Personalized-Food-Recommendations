import pandas as pd
import numpy as np
import glob
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# Removed XGBoost import as requested


class FoodDatasetLoader:
    """Load and prepare food datasets for evaluation"""
    
    def __init__(self, datasets_path="./datasets"):
        self.datasets_path = datasets_path
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
        
    def load_all_datasets(self):
        """Load and combine all CSV datasets"""
        csv_files = glob.glob(os.path.join(self.datasets_path, "*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.datasets_path}")
        
        dataframes = []
        category_mapping = {
            'drinking': 'Beverages',
            'esan_food': 'Esan Food', 
            'fruit': 'Fruits',
            'meat': 'Meat & Poultry',
            'noodle': 'Noodles',
            'onedish': 'One Dish Meals',
            'processed_food': 'Processed Food',
            'vegetables': 'Vegetables',
            'cracker': 'Snacks & Crackers',
            'curry': 'Curries',
            'dessert': 'Desserts'
        }
        
        for file_path in csv_files:
            try:
                filename = os.path.basename(file_path)
                base_name = os.path.splitext(filename)[0].lower()
                category = category_mapping.get(base_name, base_name.title())
                
                df = pd.read_csv(file_path)
                df['Category'] = category
                df = self._clean_data(df)
                
                if len(df) > 0:
                    dataframes.append(df)
                    print(f"Loaded {len(df)} items from {filename} as {category}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not dataframes:
            raise ValueError("No valid datasets could be loaded")
        
        combined_data = pd.concat(dataframes, ignore_index=True)
        print(f"Total dataset size: {len(combined_data)} items")
        print(f"Categories: {combined_data['Category'].value_counts().to_dict()}")
        
        return combined_data
    
    def _clean_data(self, df):
        """Clean and prepare data"""
        # Handle nutritional columns
        additional_features = ['FASAT (g) Saturated FA']
        all_features = self.nutritional_features + additional_features
        
        for col in all_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove rows with all zero nutritional values
        nutrient_cols = [col for col in self.nutritional_features if col in df.columns]
        if nutrient_cols:
            df = df[df[nutrient_cols].sum(axis=1) > 0]
        
        return df
    
    def calculate_health_scores(self, data):
        """Calculate health condition scores for classification tasks"""
        data = data.copy()
        
        # Initialize health score columns
        health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        for condition in health_conditions:
            data[f'{condition}_Score'] = 0
            data[f'{condition}_Category'] = 'Suitable'
        
        for idx, food in data.iterrows():
            # Diabetes scoring
            diabetes_score = 0
            sugar = float(food.get('SUGAR(g)', 0))
            carbs = float(food.get('CHOCDF (g) Carbohydrate', 0))
            fiber = float(food.get('FIBTG (g) Dietary fibre', 0))
            
            if sugar > 15: diabetes_score += 3
            elif sugar > 8: diabetes_score += 2
            elif sugar > 3: diabetes_score += 1
            
            if carbs > 20 and fiber < 3: diabetes_score += 2
            if fiber >= 5: diabetes_score -= 1
            
            data.at[idx, 'Diabetes_Score'] = max(0, diabetes_score)
            data.at[idx, 'Diabetes_Category'] = 'Not_Suitable' if diabetes_score > 2 else 'Suitable'
            
            # Obesity scoring
            obesity_score = 0
            calories = float(food.get('Energy(kcal) by calculation', 0))
            fat = float(food.get('Fat(g)', 0))
            protein = float(food.get('Protein(g)', 0))
            
            if calories > 300: obesity_score += 3
            elif calories > 200: obesity_score += 2
            elif calories > 150: obesity_score += 1
            
            if fat > 15: obesity_score += 2
            elif fat > 10: obesity_score += 1
            
            if sugar > 10: obesity_score += 2
            elif sugar > 5: obesity_score += 1
            
            if protein >= 10: obesity_score -= 1
            if fiber >= 5: obesity_score -= 1
            
            data.at[idx, 'Obesity_Score'] = max(0, obesity_score)
            data.at[idx, 'Obesity_Category'] = 'Not_Suitable' if obesity_score > 2 else 'Suitable'
            
            # Hypertension scoring
            hypertension_score = 0
            sodium = float(food.get('Na(mg)', 0))
            potassium = float(food.get('K(mg)', 0))
            
            if sodium > 400: hypertension_score += 3
            elif sodium > 200: hypertension_score += 2
            elif sodium > 100: hypertension_score += 1
            
            if potassium > 300: hypertension_score -= 1
            elif potassium > 200: hypertension_score -= 0.5
            
            if fiber >= 5: hypertension_score -= 0.5
            
            data.at[idx, 'Hypertension_Score'] = max(0, hypertension_score)
            data.at[idx, 'Hypertension_Category'] = 'Not_Suitable' if hypertension_score > 2 else 'Suitable'
            
            # High Cholesterol scoring
            cholesterol_score = 0
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            cholesterol_mg = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            if sat_fat > 5: cholesterol_score += 3
            elif sat_fat > 3: cholesterol_score += 2
            elif sat_fat > 1: cholesterol_score += 1
            
            if cholesterol_mg > 100: cholesterol_score += 2
            elif cholesterol_mg > 50: cholesterol_score += 1
            
            if fiber >= 5: cholesterol_score -= 1
            elif fiber >= 3: cholesterol_score -= 0.5
            
            data.at[idx, 'High_Cholesterol_Score'] = max(0, cholesterol_score)
            data.at[idx, 'High_Cholesterol_Category'] = 'Not_Suitable' if cholesterol_score > 2 else 'Suitable'
        
        return data


class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, data, nutritional_features):
        self.data = data
        self.nutritional_features = nutritional_features
        self.available_features = [f for f in nutritional_features if f in data.columns]
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Prepare feature matrix
        self.X = data[self.available_features].fillna(0)
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Available features: {len(self.available_features)}")
    
    def evaluate_classification_task(self, task_name, target_column, models_to_test):
        """Evaluate models on a specific classification task"""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare target variable
        y = self.data[target_column].copy()
        
        # Handle different target types
        if y.dtype == 'object':
            if target_column not in self.label_encoders:
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                self.label_encoders[target_column] = le
            else:
                y_encoded = self.label_encoders[target_column].transform(y)
        else:
            y_encoded = y.values
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        print(f"\nTask: {task_name}")
        print(f"Classes: {unique_classes}")
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        # Skip if insufficient samples for any class
        if np.min(class_counts) < 2:
            print(f"Skipping {task_name} - insufficient samples for some classes")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, y_encoded, test_size=0.2, random_state=42, 
            stratify=y_encoded if len(unique_classes) > 1 else None
        )
        
        results = {}
        
        for model_name, model in models_to_test.items():
            try:
                print(f"Evaluating {model_name}...")
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                    except:
                        pass
                
                # Calculate metrics
                metrics = self._calculate_classification_metrics(
                    y_test, y_pred, y_pred_proba, len(unique_classes) > 2
                )
                
                # Cross-validation
                cv_scores = self._cross_validation_scores(model, self.X_scaled, y_encoded)
                
                # Training time
                training_time = time.time() - start_time
                
                results[model_name] = {
                    'metrics': metrics,
                    'cross_validation': cv_scores,
                    'training_time': training_time,
                    'feature_importance': self._get_feature_importance(model),
                    'model_parameters': str(model.get_params())
                }
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba, is_multiclass):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle multiclass vs binary classification
        average_method = 'weighted' if is_multiclass else 'binary'
        
        metrics['precision'] = float(precision_score(y_true, y_pred, average=average_method, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average=average_method, zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average=average_method, zero_division=0))
        
        # Additional metrics for binary classification
        if not is_multiclass and y_pred_proba is not None:
            try:
                if y_pred_proba.shape[1] == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                else:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
            except:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Confusion matrix metrics
        metrics['confusion_matrix_normalized'] = (cm / cm.sum(axis=1, keepdims=True)).tolist() if cm.sum() > 0 else cm.tolist()
        
        # Class-wise accuracy from confusion matrix
        if cm.shape[0] == cm.shape[1]:  # Square matrix
            class_accuracy = np.diag(cm) / cm.sum(axis=1) if cm.sum() > 0 else np.zeros(cm.shape[0])
            metrics['class_wise_accuracy'] = class_accuracy.tolist()
        
        # Per-class metrics
        if is_multiclass:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            metrics['per_class_metrics'] = {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist()
            }
        
        return metrics
    
    def _cross_validation_scores(self, model, X, y):
        """Perform cross-validation"""
        try:
            # Use stratified k-fold for classification
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Calculate CV scores for different metrics
            cv_accuracy = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
            cv_precision = cross_val_score(model, X, y, cv=cv_strategy, scoring='precision_weighted')
            cv_recall = cross_val_score(model, X, y, cv=cv_strategy, scoring='recall_weighted')
            cv_f1 = cross_val_score(model, X, y, cv=cv_strategy, scoring='f1_weighted')
            
            return {
                'accuracy': {
                    'mean': float(cv_accuracy.mean()),
                    'std': float(cv_accuracy.std()),
                    'scores': cv_accuracy.tolist()
                },
                'precision': {
                    'mean': float(cv_precision.mean()),
                    'std': float(cv_precision.std()),
                    'scores': cv_precision.tolist()
                },
                'recall': {
                    'mean': float(cv_recall.mean()),
                    'std': float(cv_recall.std()),
                    'scores': cv_recall.tolist()
                },
                'f1_score': {
                    'mean': float(cv_f1.mean()),
                    'std': float(cv_f1.std()),
                    'scores': cv_f1.tolist()
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_feature_importance(self, model):
        """Extract feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.available_features, importances.tolist()))
                return feature_importance
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                if model.coef_.ndim == 1:
                    coefficients = np.abs(model.coef_)
                else:
                    coefficients = np.abs(model.coef_).mean(axis=0)
                feature_importance = dict(zip(self.available_features, coefficients.tolist()))
                return feature_importance
            else:
                return None
        except:
            return None


def run_comprehensive_evaluation():
    """Run comprehensive model evaluation"""
    print("="*80)
    print("COMPREHENSIVE FOOD RECOMMENDATION MODEL EVALUATION")
    print("="*80)
    
    # Load datasets
    loader = FoodDatasetLoader()
    data = loader.load_all_datasets()
    data = loader.calculate_health_scores(data)
    
    print(f"\nDataset loaded: {len(data)} samples")
    print(f"Categories: {data['Category'].nunique()} unique categories")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(data, loader.nutritional_features)
    
    # Define models to test (your 3 models)
    models_to_test = {
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
            alpha=0.001, max_iter=500, random_state=42, early_stopping=True
        )
    }
    
    # Define classification tasks
    classification_tasks = {
        'Food_Category_Classification': 'Category',
        'Diabetes_Suitability': 'Diabetes_Category',
        'Obesity_Suitability': 'Obesity_Category', 
        'Hypertension_Suitability': 'Hypertension_Category',
        'High_Cholesterol_Suitability': 'High_Cholesterol_Category'
    }
    
    # Run evaluations
    all_results = {
        'evaluation_metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(data),
            'num_features': len(evaluator.available_features),
            'features_used': evaluator.available_features,
            'models_evaluated': list(models_to_test.keys()),
            'classification_tasks': list(classification_tasks.keys())
        },
        'task_results': {}
    }
    
    for task_name, target_column in classification_tasks.items():
        print(f"\n{'-'*60}")
        print(f"EVALUATING TASK: {task_name}")
        print(f"Target: {target_column}")
        print(f"{'-'*60}")
        
        task_results = evaluator.evaluate_classification_task(
            task_name, target_column, models_to_test
        )
        
        if task_results:
            all_results['task_results'][task_name] = task_results
    
    # Calculate overall model rankings
    model_rankings = calculate_overall_rankings(all_results['task_results'])
    all_results['model_rankings'] = model_rankings
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(all_results['task_results'])
    all_results['summary_statistics'] = summary_stats
    
    return all_results


def calculate_overall_rankings(task_results):
    """Calculate overall model rankings across all tasks"""
    model_scores = {}
    
    for task_name, task_data in task_results.items():
        for model_name, model_data in task_data.items():
            if 'error' in model_data:
                continue
                
            if model_name not in model_scores:
                model_scores[model_name] = {
                    'accuracy_scores': [],
                    'f1_scores': [],
                    'precision_scores': [],
                    'recall_scores': [],
                    'total_tasks': 0,
                    'successful_tasks': 0
                }
            
            model_scores[model_name]['total_tasks'] += 1
            
            metrics = model_data.get('metrics', {})
            if metrics:
                model_scores[model_name]['successful_tasks'] += 1
                model_scores[model_name]['accuracy_scores'].append(metrics.get('accuracy', 0))
                model_scores[model_name]['f1_scores'].append(metrics.get('f1_score', 0))
                model_scores[model_name]['precision_scores'].append(metrics.get('precision', 0))
                model_scores[model_name]['recall_scores'].append(metrics.get('recall', 0))
    
    # Calculate overall statistics
    rankings = {}
    for model_name, scores in model_scores.items():
        if scores['successful_tasks'] > 0:
            rankings[model_name] = {
                'overall_accuracy': float(np.mean(scores['accuracy_scores'])),
                'overall_f1_score': float(np.mean(scores['f1_scores'])),
                'overall_precision': float(np.mean(scores['precision_scores'])),
                'overall_recall': float(np.mean(scores['recall_scores'])),
                'accuracy_std': float(np.std(scores['accuracy_scores'])),
                'f1_std': float(np.std(scores['f1_scores'])),
                'precision_std': float(np.std(scores['precision_scores'])),
                'recall_std': float(np.std(scores['recall_scores'])),
                'success_rate': scores['successful_tasks'] / scores['total_tasks'],
                'tasks_completed': scores['successful_tasks'],
                'total_tasks': scores['total_tasks']
            }
    
    return rankings


def generate_summary_statistics(task_results):
    """Generate summary statistics across all evaluations"""
    summary = {
        'best_performers': {},
        'task_difficulty': {},
        'model_consistency': {}
    }
    
    # Find best performers for each metric
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    for metric in metrics:
        best_scores = {}
        for task_name, task_data in task_results.items():
            task_best = None
            task_best_score = 0
            
            for model_name, model_data in task_data.items():
                if 'error' in model_data:
                    continue
                score = model_data.get('metrics', {}).get(metric, 0)
                if score > task_best_score:
                    task_best_score = score
                    task_best = model_name
            
            if task_best:
                best_scores[task_name] = {
                    'model': task_best,
                    'score': task_best_score
                }
        
        summary['best_performers'][metric] = best_scores
    
    # Calculate task difficulty (based on highest achievable scores)
    for task_name, task_data in task_results.items():
        max_accuracy = 0
        min_accuracy = 1
        num_models = 0
        
        for model_name, model_data in task_data.items():
            if 'error' in model_data:
                continue
            accuracy = model_data.get('metrics', {}).get('accuracy', 0)
            max_accuracy = max(max_accuracy, accuracy)
            min_accuracy = min(min_accuracy, accuracy)
            num_models += 1
        
        if num_models > 0:
            summary['task_difficulty'][task_name] = {
                'max_accuracy': max_accuracy,
                'min_accuracy': min_accuracy,
                'accuracy_range': max_accuracy - min_accuracy,
                'models_evaluated': num_models,
                'difficulty_score': 1 - max_accuracy  # Higher means more difficult
            }
    
    return summary


def save_results_to_json(results, filename="model_evaluation_results.json"):
    """Save results to JSON file with pretty formatting"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {filename}")


def print_summary_report(results):
    """Print a human-readable summary report"""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY REPORT")
    print("="*80)
    
    # Overall rankings
    rankings = results.get('model_rankings', {})
    print(f"\n📊 OVERALL MODEL PERFORMANCE RANKINGS:")
    print("-" * 50)
    
    # Sort models by overall F1 score
    sorted_models = sorted(rankings.items(), key=lambda x: x[1]['overall_f1_score'], reverse=True)
    
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"{rank}. {model_name}:")
        print(f"   • Overall Accuracy: {stats['overall_accuracy']:.3f} (±{stats['accuracy_std']:.3f})")
        print(f"   • Overall F1-Score: {stats['overall_f1_score']:.3f} (±{stats['f1_std']:.3f})")
        print(f"   • Overall Precision: {stats['overall_precision']:.3f} (±{stats['precision_std']:.3f})")
        print(f"   • Overall Recall: {stats['overall_recall']:.3f} (±{stats['recall_std']:.3f})")
        print(f"   • Success Rate: {stats['success_rate']:.1%} ({stats['tasks_completed']}/{stats['total_tasks']} tasks)")
        print()
    
    # Best performers by metric
    best_performers = results.get('summary_statistics', {}).get('best_performers', {})
    print(f"\n🏆 BEST PERFORMERS BY METRIC:")
    print("-" * 40)
    
    for metric, tasks in best_performers.items():
        print(f"\n{metric.upper()}:")
        model_wins = {}
        for task, info in tasks.items():
            model = info['model']
            model_wins[model] = model_wins.get(model, 0) + 1
        
        sorted_wins = sorted(model_wins.items(), key=lambda x: x[1], reverse=True)
        for model, wins in sorted_wins:
            print(f"  • {model}: {wins} task(s) won")
    
    # Task difficulty analysis
    task_difficulty = results.get('summary_statistics', {}).get('task_difficulty', {})
    print(f"\n📈 TASK DIFFICULTY ANALYSIS:")
    print("-" * 35)
    
    sorted_tasks = sorted(task_difficulty.items(), key=lambda x: x[1]['difficulty_score'])
    
    for task, stats in sorted_tasks:
        difficulty_level = "Easy" if stats['difficulty_score'] < 0.2 else "Medium" if stats['difficulty_score'] < 0.4 else "Hard"
        print(f"• {task}: {difficulty_level}")
        print(f"  Best Accuracy: {stats['max_accuracy']:.3f}, Range: {stats['accuracy_range']:.3f}")


if __name__ == "__main__":
    try:
        # Run comprehensive evaluation
        results = run_comprehensive_evaluation()
        
        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_evaluation_results_{timestamp}.json"
        save_results_to_json(results, filename)
        
        # Print summary report
        print_summary_report(results)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📄 Detailed results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()