import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import os
import glob
import time
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class ModelPerformance:
    """Data class to store model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float = None
    confusion_matrix: np.ndarray = None
    classification_report: str = ""
    recommendations_quality: Dict = None
    execution_time: float = 0.0
    
class PerformanceEvaluator:
    """Comprehensive performance evaluation system for recommendation models"""
    
    def __init__(self):
        self.health_conditions = ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']
        self.performance_results = {}
        
    def create_ground_truth_labels(self, food_data: pd.DataFrame) -> pd.DataFrame:
        """Create ground truth labels for health condition suitability"""
        labeled_data = food_data.copy()
        
        for _, food in labeled_data.iterrows():
            idx = food.name
            
            # Get nutritional values
            energy = food.get('Energy(kcal) by calculation', 0)
            protein = food.get('Protein(g)', 0)
            carbs = food.get('CHOCDF (g) Carbohydrate', 0)
            sugar = food.get('SUGAR(g)', 0)
            fiber = food.get('FIBTG (g) Dietary fibre', 0)
            fat = food.get('Fat(g)', 0)
            sodium = food.get('Na(mg)', 500)  # Default moderate sodium
            sat_fat = food.get('FASAT (g) Saturated FA', fat * 0.3)  # Estimate if missing
            cholesterol = food.get('CHOLE(mg) Cholesterol', 0)
            
            # Create binary labels (1 = suitable, 0 = not suitable) for each condition
            
            # Diabetes suitability (ADA guidelines)
            diabetes_suitable = 1 if (
                sugar <= 10 and  # Low sugar
                fiber >= 3 and   # Good fiber
                carbs <= 30      # Moderate carbs per serving
            ) else 0
            labeled_data.at[idx, 'Diabetes_Label'] = diabetes_suitable
            
            # Obesity suitability (portion control, nutrient density)
            obesity_suitable = 1 if (
                energy <= 250 and  # Lower calorie
                fiber >= 3 and     # High satiety
                protein >= 8       # Good protein for satiety
            ) else 0
            labeled_data.at[idx, 'Obesity_Label'] = obesity_suitable
            
            # Hypertension suitability (DASH diet principles)
            hypertension_suitable = 1 if (
                sodium <= 300 and  # Low sodium
                fiber >= 2         # Heart healthy fiber
            ) else 0
            labeled_data.at[idx, 'Hypertension_Label'] = hypertension_suitable
            
            # High Cholesterol suitability (AHA guidelines)
            cholesterol_suitable = 1 if (
                sat_fat <= 3 and      # Low saturated fat
                cholesterol <= 50 and # Low dietary cholesterol
                fiber >= 3            # Cholesterol-lowering fiber
            ) else 0
            labeled_data.at[idx, 'High_Cholesterol_Label'] = cholesterol_suitable
        
        return labeled_data
    
    def evaluate_recommendations(self, true_labels: np.ndarray, predicted_labels: np.ndarray, 
                                condition: str) -> Dict:
        """Evaluate recommendation quality for a specific health condition"""
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
        
        # Generate classification report
        class_report = classification_report(true_labels, predicted_labels, 
                                           target_names=['Not Suitable', 'Suitable'],
                                           zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        return {
            'condition': condition,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'support': len(true_labels)
        }
    
    def calculate_recommendation_metrics(self, recommendations: List, true_labels: pd.DataFrame, 
                                       top_k: int = 10) -> Dict:
        """Calculate recommendation-specific metrics like precision@k, recall@k"""
        
        metrics = {}
        
        for condition in self.health_conditions:
            label_col = f'{condition}_Label'
            if label_col not in true_labels.columns:
                continue
                
            # Get indices of recommended foods
            rec_indices = []
            for rec in recommendations[:top_k]:
                # Find the index in the original dataset
                food_name = rec.get('name', '')
                matches = true_labels[
                    (true_labels['Thai_Name'] == food_name) | 
                    (true_labels['English_Name'] == food_name)
                ]
                if not matches.empty:
                    rec_indices.append(matches.index[0])
            
            if not rec_indices:
                continue
                
            # Calculate precision@k and recall@k
            relevant_items = true_labels[true_labels[label_col] == 1].index.tolist()
            recommended_relevant = [idx for idx in rec_indices if idx in relevant_items]
            
            precision_at_k = len(recommended_relevant) / len(rec_indices) if rec_indices else 0
            recall_at_k = len(recommended_relevant) / len(relevant_items) if relevant_items else 0
            
            metrics[condition] = {
                'precision_at_k': precision_at_k,
                'recall_at_k': recall_at_k,
                'recommended_items': len(rec_indices),
                'relevant_items': len(relevant_items),
                'recommended_relevant': len(recommended_relevant)
            }
        
        return metrics

class FoodRecommendationKNN:
    """Enhanced KNN Food Recommendation System with Performance Evaluation"""
    
    def __init__(self, status_callback=None):
        self.status_callback = status_callback
        self.scaler = StandardScaler()
        self.knn_model = None
        self.food_data = pd.DataFrame()
        self.labeled_data = pd.DataFrame()
        self.features = []
        self.evaluator = PerformanceEvaluator()
        self.performance_results = {}
        
        # Load and prepare data
        self.load_data()
        self.prepare_features()
        self.create_labels()
        self.train_model()
    
    def update_status(self, message: str):
        """Update status if callback provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self):
        """Load Thai food datasets"""
        try:
            dataset_folder = './datasets'
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
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
            ['ข้าวผัด', 'Fried Rice', 'Rice', 520, 15, 18, 75, 2, 3, 850, 200, 5, 0],
            ['ต้มยำกุ้ง', 'Tom Yum Goong', 'Soup', 180, 20, 8, 12, 6, 2, 1200, 300, 2, 120],
            ['ส้มตำ', 'Som Tam', 'Salad', 120, 3, 2, 25, 15, 5, 800, 400, 1, 0],
            ['แกงเขียวหวาน', 'Green Curry', 'Curry', 280, 18, 20, 15, 8, 3, 950, 450, 8, 80],
            ['ผัดไทย', 'Pad Thai', 'Noodle', 450, 12, 15, 68, 12, 3, 1100, 250, 4, 85],
            ['มะม่วงข้าวเหนียว', 'Mango Sticky Rice', 'Dessert', 380, 6, 8, 78, 45, 2, 150, 180, 3, 0],
            ['ไก่ย่าง', 'Grilled Chicken', 'Meat', 250, 35, 12, 0, 0, 0, 420, 350, 3, 95],
            ['ยำวุ้นเส้น', 'Glass Noodle Salad', 'Salad', 180, 8, 3, 35, 8, 2, 680, 320, 1, 45],
            ['ผักบุ้งไฟแดง', 'Stir-fried Water Spinach', 'Vegetable', 90, 4, 3, 12, 3, 4, 480, 380, 1, 0],
            ['ปลาเผา', 'Grilled Fish', 'Fish', 200, 30, 8, 0, 0, 0, 350, 400, 2, 70],
        ]
        
        columns = ['Thai_Name', 'English_Name', 'Category', 'Energy(kcal) by calculation', 
                  'Protein(g)', 'Fat(g)', 'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 
                  'FIBTG (g) Dietary fibre', 'Na(mg)', 'K(mg)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol']
        
        self.food_data = pd.DataFrame(sample_foods, columns=columns)
        self.update_status(f"Created {len(self.food_data)} sample food items")
    
    def prepare_features(self):
        """Prepare features for KNN model"""
        self.update_status("Preparing features...")
        
        # Define critical nutritional features
        feature_candidates = [
            'Energy(kcal) by calculation', 'Protein(g)', 'Fat(g)', 
            'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 'FIBTG (g) Dietary fibre',
            'Na(mg)', 'K(mg)', 'FASAT (g) Saturated FA', 'CHOLE(mg) Cholesterol'
        ]
        
        available_features = []
        for feature in feature_candidates:
            if feature in self.food_data.columns:
                available_features.append(feature)
        
        self.features = available_features
        
        # Handle missing values
        for col in self.features:
            self.food_data[col] = pd.to_numeric(self.food_data[col], errors='coerce')
            
            if col == 'Na(mg)':
                self.food_data[col].fillna(500, inplace=True)
            elif col == 'K(mg)':
                self.food_data[col].fillna(200, inplace=True)
            elif col == 'FASAT (g) Saturated FA':
                self.food_data[col].fillna(self.food_data['Fat(g)'] * 0.3, inplace=True)
            elif col == 'CHOLE(mg) Cholesterol':
                self.food_data[col].fillna(0, inplace=True)
            else:
                self.food_data[col].fillna(self.food_data[col].median(), inplace=True)
        
        self.update_status(f"Prepared {len(self.features)} features")
    
    def create_labels(self):
        """Create ground truth labels for evaluation"""
        self.update_status("Creating ground truth labels...")
        self.labeled_data = self.evaluator.create_ground_truth_labels(self.food_data)
        
        # Print label distribution for analysis
        for condition in self.evaluator.health_conditions:
            label_col = f'{condition}_Label'
            if label_col in self.labeled_data.columns:
                suitable_count = self.labeled_data[label_col].sum()
                total_count = len(self.labeled_data)
                self.update_status(f"{condition}: {suitable_count}/{total_count} ({suitable_count/total_count*100:.1f}%) suitable")
    
    def train_model(self):
        """Train KNN model"""
        if len(self.food_data) == 0:
            self.update_status("No data available for training")
            return
        
        self.update_status("Training KNN model...")
        
        # Prepare feature matrix
        X = self.food_data[self.features].values
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train KNN model
        n_neighbors = min(5, len(self.food_data))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='ball_tree',
            metric='manhattan'
        )
        self.knn_model.fit(X_scaled)
        
        self.update_status(f"KNN model trained with {n_neighbors} neighbors")
    
    def get_recommendations(self, user_preferences: Dict, health_conditions: List[str], 
                          max_recommendations: int = 10) -> List[Dict]:
        """Get food recommendations based on user preferences and health conditions"""
        if self.knn_model is None:
            return []
        
        # Create target vector based on user preferences
        target_vector = []
        for feature in self.features:
            if feature in user_preferences:
                target_vector.append(user_preferences[feature])
            else:
                # Use median values for missing preferences
                target_vector.append(self.food_data[feature].median())
        
        # Scale target vector
        target_scaled = self.scaler.transform([target_vector])
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(target_scaled)
        
        # Create recommendations
        recommendations = []
        for i, idx in enumerate(indices[0]):
            food = self.food_data.iloc[idx]
            
            # Calculate health scores for specified conditions
            health_scores = {}
            overall_suitability = 0
            
            for condition in health_conditions:
                label_col = f'{condition}_Label'
                if label_col in self.labeled_data.columns:
                    score = self.labeled_data.iloc[idx][label_col]
                    health_scores[condition] = score
                    overall_suitability += score
            
            # Calculate average suitability
            avg_suitability = overall_suitability / len(health_conditions) if health_conditions else 0
            
            recommendation = {
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
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
                'avg_suitability': avg_suitability,
                'food_index': idx
            }
            
            recommendations.append(recommendation)
            
            if len(recommendations) >= max_recommendations:
                break
        
        # Sort by health suitability and distance
        if health_conditions:
            recommendations.sort(key=lambda x: (-x['avg_suitability'], x['distance']))
        else:
            recommendations.sort(key=lambda x: x['distance'])
        
        return recommendations[:max_recommendations]
    
    def evaluate_model_performance(self, test_scenarios: List[Dict]) -> ModelPerformance:
        """Comprehensive model performance evaluation"""
        self.update_status("Evaluating model performance...")
        
        start_time = time.time()
        
        all_results = {}
        condition_performances = {}
        
        # Test multiple scenarios
        for scenario_idx, scenario in enumerate(test_scenarios):
            self.update_status(f"Testing scenario {scenario_idx + 1}/{len(test_scenarios)}")
            
            user_prefs = scenario['user_preferences']
            health_conditions = scenario['health_conditions']
            
            # Get recommendations
            recommendations = self.get_recommendations(user_prefs, health_conditions)
            
            # Evaluate for each health condition
            for condition in health_conditions:
                if condition not in condition_performances:
                    condition_performances[condition] = {
                        'true_labels': [],
                        'pred_labels': [],
                        'recommendation_metrics': []
                    }
                
                label_col = f'{condition}_Label'
                if label_col not in self.labeled_data.columns:
                    continue
                
                # Create predictions based on recommendations
                # Consider top recommended foods as "predicted suitable"
                rec_indices = [rec['food_index'] for rec in recommendations[:5]]  # Top 5
                
                # Get all food indices
                all_indices = list(range(len(self.labeled_data)))
                
                # Create binary predictions (1 if recommended, 0 if not)
                pred_labels = [1 if idx in rec_indices else 0 for idx in all_indices]
                true_labels = self.labeled_data[label_col].tolist()
                
                condition_performances[condition]['true_labels'].extend(true_labels)
                condition_performances[condition]['pred_labels'].extend(pred_labels)
                
                # Calculate recommendation metrics
                rec_metrics = self.evaluator.calculate_recommendation_metrics(
                    recommendations, self.labeled_data, top_k=10
                )
                condition_performances[condition]['recommendation_metrics'].append(rec_metrics)
        
        # Calculate overall performance metrics
        overall_metrics = {}
        detailed_results = {}
        
        for condition in self.evaluator.health_conditions:
            if condition in condition_performances:
                perf_data = condition_performances[condition]
                
                # Calculate classification metrics
                eval_result = self.evaluator.evaluate_recommendations(
                    np.array(perf_data['true_labels']),
                    np.array(perf_data['pred_labels']),
                    condition
                )
                
                detailed_results[condition] = eval_result
                
                # Store for overall calculation
                overall_metrics[f'{condition}_accuracy'] = eval_result['accuracy']
                overall_metrics[f'{condition}_precision'] = eval_result['precision']
                overall_metrics[f'{condition}_recall'] = eval_result['recall']
                overall_metrics[f'{condition}_f1'] = eval_result['f1_score']
        
        # Calculate average metrics across all conditions
        avg_accuracy = np.mean([overall_metrics[k] for k in overall_metrics if 'accuracy' in k])
        avg_precision = np.mean([overall_metrics[k] for k in overall_metrics if 'precision' in k])
        avg_recall = np.mean([overall_metrics[k] for k in overall_metrics if 'recall' in k])
        avg_f1 = np.mean([overall_metrics[k] for k in overall_metrics if 'f1' in k])
        
        execution_time = time.time() - start_time
        
        # Create performance object
        performance = ModelPerformance(
            model_name="KNN Food Recommender",
            accuracy=avg_accuracy,
            precision=avg_precision,
            recall=avg_recall,
            f1_score=avg_f1,
            execution_time=execution_time,
            recommendations_quality=detailed_results
        )
        
        self.performance_results = {
            'overall_performance': performance,
            'detailed_results': detailed_results,
            'condition_performances': condition_performances
        }
        
        return performance
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        if not self.performance_results:
            print("No performance results available. Run evaluate_model_performance() first.")
            return
        
        performance = self.performance_results['overall_performance']
        detailed_results = self.performance_results['detailed_results']
        
        print("=" * 80)
        print("KNN FOOD RECOMMENDATION SYSTEM - PERFORMANCE REPORT")
        print("=" * 80)
        
        print(f"\nModel: {performance.model_name}")
        print(f"Execution Time: {performance.execution_time:.2f} seconds")
        
        print(f"\nOVERALL PERFORMANCE METRICS:")
        print(f"Average Accuracy:  {performance.accuracy:.4f}")
        print(f"Average Precision: {performance.precision:.4f}")
        print(f"Average Recall:    {performance.recall:.4f}")
        print(f"Average F1-Score:  {performance.f1_score:.4f}")
        
        print(f"\nDETAILED RESULTS BY HEALTH CONDITION:")
        print("-" * 60)
        
        for condition, results in detailed_results.items():
            print(f"\n{condition.upper()}:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1_score']:.4f}")
            print(f"  Support:   {results['support']} samples")
            
            # Print confusion matrix
            cm = results['confusion_matrix']
            print(f"  Confusion Matrix:")
            print(f"    [[TN={cm[0,0]}, FP={cm[0,1]}],")
            print(f"     [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        # Print recommendation quality analysis
        print(f"\nRECOMMENDATION QUALITY ANALYSIS:")
        print("-" * 40)
        for condition in self.evaluator.health_conditions:
            if condition in detailed_results:
                suitable_foods = (self.labeled_data[f'{condition}_Label'] == 1).sum()
                total_foods = len(self.labeled_data)
                print(f"{condition}: {suitable_foods}/{total_foods} ({suitable_foods/total_foods*100:.1f}%) foods are suitable")
    
    def plot_performance_visualization(self):
        """Create performance visualization plots"""
        if not self.performance_results:
            print("No performance results available. Run evaluate_model_performance() first.")
            return
        
        detailed_results = self.performance_results['detailed_results']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('KNN Food Recommendation System - Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance metrics comparison
        conditions = list(detailed_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [detailed_results[cond][metric] for cond in conditions] for metric in metrics}
        
        x = np.arange(len(conditions))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(x + i * width, metric_values[metric], width, label=metric.capitalize())
        
        axes[0, 0].set_xlabel('Health Conditions')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics by Health Condition')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(conditions, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix Heatmap (for first condition)
        if conditions:
            first_condition = conditions[0]
            cm = detailed_results[first_condition]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Suitable', 'Suitable'],
                       yticklabels=['Not Suitable', 'Suitable'],
                       ax=axes[0, 1])
            axes[0, 1].set_title(f'Confusion Matrix - {first_condition}')
            axes[0, 1].set_ylabel('True Label')
            axes[0, 1].set_xlabel('Predicted Label')
        
        # Plot 3: Data distribution analysis
        label_counts = {}
        for condition in conditions:
            suitable_count = (self.labeled_data[f'{condition}_Label'] == 1).sum()
            total_count = len(self.labeled_data)
            label_counts[condition] = [total_count - suitable_count, suitable_count]
        
        bottom = np.zeros(len(conditions))
        for i, label in enumerate(['Not Suitable', 'Suitable']):
            values = [label_counts[cond][i] for cond in conditions]
            axes[1, 0].bar(conditions, values, bottom=bottom, label=label)
            bottom += values
        
        axes[1, 0].set_title('Data Distribution by Health Condition')
        axes[1, 0].set_ylabel('Number of Foods')
        axes[1, 0].set_xlabel('Health Conditions')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Overall performance radar chart
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        overall_perf = self.performance_results['overall_performance']
        values = [overall_perf.accuracy, overall_perf.precision, overall_perf.recall, overall_perf.f1_score]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        axes[1, 1].plot(angles, values, 'o-', linewidth=2, color='blue')
        axes[1, 1].fill(angles, values, alpha=0.25, color='blue')
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Overall Performance Profile')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def export_performance_results(self, filename: str = "knn_performance_results.csv"):
        """Export performance results to CSV"""
        if not self.performance_results:
            print("No performance results available.")
            return
        
        detailed_results = self.performance_results['detailed_results']
        overall_perf = self.performance_results['overall_performance']
        
        # Create results dataframe
        results_data = []
        
        # Add overall results
        results_data.append({
            'Model': 'KNN',
            'Condition': 'Overall',
            'Accuracy': overall_perf.accuracy,
            'Precision': overall_perf.precision,
            'Recall': overall_perf.recall,
            'F1_Score': overall_perf.f1_score,
            'Execution_Time': overall_perf.execution_time
        })
        
        # Add condition-specific results
        for condition, results in detailed_results.items():
            results_data.append({
                'Model': 'KNN',
                'Condition': condition,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'Support': results['support']
            })
        
        # Save to CSV
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filename, index=False)
        print(f"Performance results exported to {filename}")
        
        return results_df

def create_test_scenarios():
    """Create test scenarios for evaluation"""
    scenarios = [
        {
            'name': 'Diabetes Management',
            'user_preferences': {
                'Energy(kcal) by calculation': 200,
                'Protein(g)': 15,
                'Fat(g)': 8,
                'CHOCDF (g) Carbohydrate': 20,
                'SUGAR(g)': 5,
                'FIBTG (g) Dietary fibre': 5,
                'Na(mg)': 300
            },
            'health_conditions': ['Diabetes']
        },
        {
            'name': 'Weight Loss',
            'user_preferences': {
                'Energy(kcal) by calculation': 150,
                'Protein(g)': 20,
                'Fat(g)': 5,
                'CHOCDF (g) Carbohydrate': 15,
                'SUGAR(g)': 3,
                'FIBTG (g) Dietary fibre': 6,
                'Na(mg)': 250
            },
            'health_conditions': ['Obesity']
        },
        {
            'name': 'Heart Healthy',
            'user_preferences': {
                'Energy(kcal) by calculation': 180,
                'Protein(g)': 18,
                'Fat(g)': 6,
                'CHOCDF (g) Carbohydrate': 25,
                'SUGAR(g)': 4,
                'FIBTG (g) Dietary fibre': 5,
                'Na(mg)': 200
            },
            'health_conditions': ['Hypertension', 'High_Cholesterol']
        },
        {
            'name': 'Multiple Conditions',
            'user_preferences': {
                'Energy(kcal) by calculation': 160,
                'Protein(g)': 16,
                'Fat(g)': 6,
                'CHOCDF (g) Carbohydrate': 18,
                'SUGAR(g)': 3,
                'FIBTG (g) Dietary fibre': 6,
                'Na(mg)': 250
            },
            'health_conditions': ['Diabetes', 'Hypertension']
        }
    ]
    
    return scenarios

def main():
    """Main function to run the evaluation"""
    print("Initializing KNN Food Recommendation System with Performance Evaluation...")
    
    # Initialize system
    knn_system = FoodRecommendationKNN()
    
    # Create test scenarios
    test_scenarios = create_test_scenarios()
    
    print(f"\nRunning evaluation with {len(test_scenarios)} test scenarios...")
    
    # Evaluate performance
    performance = knn_system.evaluate_model_performance(test_scenarios)
    
    # Print results
    knn_system.print_performance_report()
    
    # Create visualizations
    print("\nGenerating performance visualizations...")
    knn_system.plot_performance_visualization()
    
    # Export results
    results_df = knn_system.export_performance_results()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return knn_system, performance, results_df

if __name__ == "__main__":
    knn_system, performance, results_df = main()