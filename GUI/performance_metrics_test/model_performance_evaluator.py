import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceEvaluator:
    """Comprehensive model performance evaluation for food recommendation systems"""
    
    def __init__(self):
        self.results = {}
        self.regression_metrics = [
            'r2_score', 'mse', 'rmse', 'mae', 'mape', 'cv_r2_mean', 'cv_r2_std'
        ]
        self.classification_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'cv_accuracy_mean', 'cv_accuracy_std'
        ]
    
    def evaluate_random_forest_model(self, recommender_system):
        """Extract and calculate comprehensive metrics for Random Forest model"""
        print("🔍 Evaluating Random Forest Model Performance...")
        print("=" * 60)
        
        if not hasattr(recommender_system, 'models') or not recommender_system.models:
            print("❌ No trained models found in the recommender system!")
            return None
        
        # Get the trained models and data
        models = recommender_system.models
        food_data = recommender_system.food_data
        features = recommender_system.features
        
        # Prepare feature matrix
        X = food_data[features].fillna(0)
        X_scaled = recommender_system.scaler.transform(X)
        
        model_results = {}
        
        # Evaluate each target model
        for target_name, model in models.items():
            print(f"\n📊 Evaluating {target_name} Model:")
            print("-" * 40)
            
            # Get target values
            if target_name == 'Overall_Health':
                y_true = food_data['Overall_Health_Score']
            elif target_name == 'Diabetes_Suitability':
                y_true = food_data['Diabetes_Score']
            elif target_name == 'Obesity_Suitability':
                y_true = food_data['Obesity_Score']
            elif target_name == 'Hypertension_Suitability':
                y_true = food_data['Hypertension_Score']
            elif target_name == 'Cholesterol_Suitability':
                y_true = food_data['High_Cholesterol_Score']
            else:
                continue
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Calculate regression metrics
            regression_results = self._calculate_regression_metrics(y_true, y_pred, model, X_scaled, y_true)
            
            # Calculate classification metrics (convert scores to categories)
            classification_results = self._calculate_classification_metrics(y_true, y_pred, model, X_scaled, y_true)
            
            # Combine results
            target_results = {**regression_results, **classification_results}
            model_results[target_name] = target_results
            
            # Display results
            self._display_target_results(target_name, target_results)
        
        # Calculate overall system performance
        overall_results = self._calculate_overall_performance(model_results)
        
        # Store results
        self.results['Random_Forest'] = {
            'individual_models': model_results,
            'overall_performance': overall_results,
            'feature_importance': self._get_feature_importance(models, features),
            'recommendation_accuracy': self._evaluate_recommendation_accuracy(recommender_system)
        }
        
        # Display summary
        self._display_overall_summary('Random_Forest', overall_results)
        
        return self.results['Random_Forest']
    
    def _calculate_regression_metrics(self, y_true, y_pred, model, X, y):
        """Calculate regression performance metrics"""
        metrics = {}
        
        # Basic regression metrics
        metrics['r2_score'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error (handle division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.inf
        
        # Cross-validation scores
        try:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        except:
            metrics['cv_r2_mean'] = metrics['r2_score']
            metrics['cv_r2_std'] = 0.0
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true, y_pred, model, X, y):
        """Convert regression to classification and calculate classification metrics"""
        metrics = {}
        
        # Convert continuous scores to categories
        # Define thresholds: 0-1.5 = "Good", 1.5-3 = "Fair", 3+ = "Poor"
        def score_to_category(scores):
            categories = np.zeros(len(scores), dtype=int)
            categories[(scores >= 0) & (scores < 1.5)] = 0  # Good
            categories[(scores >= 1.5) & (scores < 3.0)] = 1  # Fair
            categories[scores >= 3.0] = 2  # Poor
            return categories
        
        y_true_cat = score_to_category(y_true)
        y_pred_cat = score_to_category(y_pred)
        
        # Calculate classification metrics
        metrics['accuracy'] = accuracy_score(y_true_cat, y_pred_cat)
        metrics['precision'] = precision_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true_cat, y_pred_cat, average='weighted', zero_division=0)
        
        # Cross-validation for classification
        try:
            # Create a simple classifier version for CV
            from sklearn.ensemble import RandomForestClassifier
            clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            cv_scores = cross_val_score(clf_model, X, y_true_cat, cv=5, scoring='accuracy')
            metrics['cv_accuracy_mean'] = cv_scores.mean()
            metrics['cv_accuracy_std'] = cv_scores.std()
        except:
            metrics['cv_accuracy_mean'] = metrics['accuracy']
            metrics['cv_accuracy_std'] = 0.0
        
        # Store detailed classification report
        metrics['classification_report'] = classification_report(
            y_true_cat, y_pred_cat, 
            target_names=['Good (0-1.5)', 'Fair (1.5-3)', 'Poor (3+)'],
            output_dict=True
        )
        
        return metrics
    
    def _calculate_overall_performance(self, model_results):
        """Calculate overall system performance across all models"""
        overall = {}
        
        # Average metrics across all models
        all_metrics = {}
        for target, results in model_results.items():
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate averages
        for metric, values in all_metrics.items():
            overall[f'avg_{metric}'] = np.mean(values)
            overall[f'std_{metric}'] = np.std(values)
        
        return overall
    
    def _get_feature_importance(self, models, features):
        """Get feature importance from Random Forest models"""
        feature_importance = {}
        
        if 'Overall_Health' in models:
            importance_scores = models['Overall_Health'].feature_importances_
            feature_importance = dict(zip(features, importance_scores))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _evaluate_recommendation_accuracy(self, recommender_system):
        """Evaluate the accuracy of food recommendations"""
        try:
            # Create a test user profile
            test_profile = {
                'weight': 70,
                'height': 170,
                'age': 35,
                'gender': 'Male',
                'activity_level': 'Moderate',
                'weight_goal': 'Maintain Weight',
                'health_conditions': ['Diabetes'],
                'category_filter': 'All'
            }
            
            # Get recommendations
            recommendations = recommender_system.get_recommendations(test_profile, 'lunch', 20)
            
            if not recommendations:
                return {'error': 'No recommendations generated'}
            
            # Analyze recommendation quality
            suitable_count = 0
            avg_score = 0
            condition_matches = 0
            
            for rec in recommendations:
                avg_score += rec['combined_score']
                if rec['suitable_for_conditions']:
                    suitable_count += 1
                    if 'Diabetes' in rec['suitable_for_conditions']:
                        condition_matches += 1
            
            return {
                'total_recommendations': len(recommendations),
                'suitable_recommendations': suitable_count,
                'suitability_rate': suitable_count / len(recommendations),
                'condition_match_rate': condition_matches / len(recommendations),
                'avg_recommendation_score': avg_score / len(recommendations),
                'top_5_avg_score': sum(r['combined_score'] for r in recommendations[:5]) / 5
            }
            
        except Exception as e:
            return {'error': f'Failed to evaluate recommendations: {str(e)}'}
    
    def _display_target_results(self, target_name, results):
        """Display results for a specific target model"""
        print(f"📈 Regression Metrics:")
        print(f"   R² Score: {results['r2_score']:.4f}")
        print(f"   MSE: {results['mse']:.4f}")
        print(f"   RMSE: {results['rmse']:.4f}")
        print(f"   MAE: {results['mae']:.4f}")
        print(f"   MAPE: {results['mape']:.2f}%")
        print(f"   CV R² (mean±std): {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
        
        print(f"\n🎯 Classification Metrics (Good/Fair/Poor):")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")
        print(f"   CV Accuracy (mean±std): {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    
    def _display_overall_summary(self, model_name, overall_results):
        """Display overall model performance summary"""
        print(f"\n🏆 {model_name} Overall Performance Summary:")
        print("=" * 60)
        
        key_metrics = ['avg_r2_score', 'avg_accuracy', 'avg_f1_score', 'avg_mse']
        for metric in key_metrics:
            if metric in overall_results:
                print(f"   {metric.replace('avg_', '').upper()}: {overall_results[metric]:.4f}")
    
    def compare_models(self, model_results_dict):
        """Compare performance across multiple models"""
        print("\n🔄 Model Comparison Summary:")
        print("=" * 80)
        
        comparison_df = pd.DataFrame()
        
        for model_name, results in model_results_dict.items():
            if 'overall_performance' in results:
                model_row = {}
                overall = results['overall_performance']
                
                # Extract key metrics
                key_metrics = {
                    'Avg_R2': 'avg_r2_score',
                    'Avg_Accuracy': 'avg_accuracy', 
                    'Avg_F1': 'avg_f1_score',
                    'Avg_Precision': 'avg_precision',
                    'Avg_Recall': 'avg_recall',
                    'Avg_MSE': 'avg_mse',
                    'Avg_MAE': 'avg_mae'
                }
                
                for display_name, metric_key in key_metrics.items():
                    model_row[display_name] = overall.get(metric_key, np.nan)
                
                comparison_df[model_name] = pd.Series(model_row)
        
        if not comparison_df.empty:
            print("\n📊 Performance Metrics Comparison:")
            print(comparison_df.round(4).to_string())
            
            # Highlight best performing model for each metric
            print("\n🥇 Best Performing Model by Metric:")
            for metric in comparison_df.index:
                if metric in ['Avg_MSE', 'Avg_MAE']:  # Lower is better
                    best_model = comparison_df.loc[metric].idxmin()
                    best_value = comparison_df.loc[metric].min()
                else:  # Higher is better
                    best_model = comparison_df.loc[metric].idxmax()
                    best_value = comparison_df.loc[metric].max()
                
                if not np.isnan(best_value):
                    print(f"   {metric}: {best_model} ({best_value:.4f})")
        
        return comparison_df
    
    def generate_performance_report(self, output_file='model_performance_report.txt'):
        """Generate a comprehensive performance report"""
        with open(output_file, 'w') as f:
            f.write("FOOD RECOMMENDATION SYSTEM - MODEL PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"MODEL: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                # Overall performance
                if 'overall_performance' in results:
                    f.write("Overall Performance:\n")
                    for metric, value in results['overall_performance'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
                
                # Feature importance
                if 'feature_importance' in results:
                    f.write("Top 10 Feature Importance:\n")
                    for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
                        f.write(f"  {i+1}. {feature}: {importance:.4f}\n")
                    f.write("\n")
                
                # Recommendation accuracy
                if 'recommendation_accuracy' in results:
                    f.write("Recommendation System Performance:\n")
                    for metric, value in results['recommendation_accuracy'].items():
                        f.write(f"  {metric}: {value}\n")
                    f.write("\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"📄 Performance report saved to: {output_file}")
    
    def plot_performance_comparison(self, save_path='model_performance_plots.png'):
        """Create visualization plots for model performance"""
        if not self.results:
            print("❌ No results to plot!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        models = list(self.results.keys())
        
        # Plot 1: R² Scores by Model and Target
        ax1 = axes[0, 0]
        r2_data = []
        targets = []
        model_names = []
        
        for model_name, results in self.results.items():
            if 'individual_models' in results:
                for target, metrics in results['individual_models'].items():
                    r2_data.append(metrics.get('r2_score', 0))
                    targets.append(target.replace('_', ' '))
                    model_names.append(model_name)
        
        if r2_data:
            df_r2 = pd.DataFrame({'Model': model_names, 'Target': targets, 'R2_Score': r2_data})
            sns.barplot(data=df_r2, x='Target', y='R2_Score', hue='Model', ax=ax1)
            ax1.set_title('R² Score by Target Model')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Classification Metrics
        ax2 = axes[0, 1]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = []
        metric_names = []
        model_names_2 = []
        
        for model_name, results in self.results.items():
            if 'overall_performance' in results:
                for metric in metrics_to_plot:
                    avg_key = f'avg_{metric}'
                    if avg_key in results['overall_performance']:
                        metric_values.append(results['overall_performance'][avg_key])
                        metric_names.append(metric.title())
                        model_names_2.append(model_name)
        
        if metric_values:
            df_class = pd.DataFrame({'Model': model_names_2, 'Metric': metric_names, 'Value': metric_values})
            sns.barplot(data=df_class, x='Metric', y='Value', hue='Model', ax=ax2)
            ax2.set_title('Average Classification Metrics')
            ax2.set_ylabel('Score')
        
        # Plot 3: Feature Importance (for Random Forest)
        ax3 = axes[1, 0]
        if 'Random_Forest' in self.results and 'feature_importance' in self.results['Random_Forest']:
            importance_data = self.results['Random_Forest']['feature_importance']
            top_features = list(importance_data.items())[:10]
            
            if top_features:
                features, importances = zip(*top_features)
                y_pos = np.arange(len(features))
                
                ax3.barh(y_pos, importances)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels([f.replace('(', '\n(') if '(' in f else f for f in features])
                ax3.set_xlabel('Importance')
                ax3.set_title('Top 10 Feature Importance (Random Forest)')
        
        # Plot 4: Recommendation Quality
        ax4 = axes[1, 1]
        if 'Random_Forest' in self.results and 'recommendation_accuracy' in self.results['Random_Forest']:
            rec_data = self.results['Random_Forest']['recommendation_accuracy']
            
            if 'error' not in rec_data:
                metrics = ['suitability_rate', 'condition_match_rate']
                values = [rec_data.get(m, 0) for m in metrics]
                labels = ['Suitability Rate', 'Condition Match Rate']
                
                bars = ax4.bar(labels, values, color=['skyblue', 'lightcoral'])
                ax4.set_ylabel('Rate')
                ax4.set_title('Recommendation Quality Metrics')
                ax4.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance plots saved to: {save_path}")


# Usage example and main evaluation function
def evaluate_random_forest_system():
    """Main function to evaluate the Random Forest recommendation system"""
    
    # Import the Random Forest system
    import sys
    import os
    
    try:
        # Add the current directory to Python path to import the model
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
        # Import the Random Forest recommendation system
        from food_recommendation_system_using_rf_model_v2 import HealthAwareRandomForestRecommender
        
        print("🚀 Initializing Random Forest Food Recommendation System...")
        
        # Initialize the recommender system
        def status_callback(message):
            print(f"   {message}")
        
        recommender = HealthAwareRandomForestRecommender(status_callback=status_callback)
        
        print("\n📊 Starting Model Performance Evaluation...")
        
        # Initialize evaluator
        evaluator = ModelPerformanceEvaluator()
        
        # Evaluate Random Forest model
        rf_results = evaluator.evaluate_random_forest_model(recommender)
        
        if rf_results:
            print("\n📈 Generating Performance Report...")
            evaluator.generate_performance_report('random_forest_performance_report.txt')
            
            print("\n📊 Creating Performance Visualizations...")
            evaluator.plot_performance_comparison('random_forest_performance_plots.png')
            
            print("\n✅ Evaluation Complete!")
            print(f"📄 Report saved to: random_forest_performance_report.txt")
            print(f"📊 Plots saved to: random_forest_performance_plots.png")
            
            return evaluator.results
        else:
            print("❌ Failed to evaluate Random Forest model!")
            return None
            
    except ImportError as e:
        print(f"❌ Error importing Random Forest system: {e}")
        print("💡 Make sure 'food_recommendation_system_using_rf_model_v2.py' is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None


if __name__ == "__main__":
    # Run the evaluation
    results = evaluate_random_forest_system()
    
    if results:
        print("\n🏆 EVALUATION SUMMARY:")
        print("="*50)
        
        if 'Random_Forest' in results:
            rf_results = results['Random_Forest']
            
            # Display key metrics
            if 'overall_performance' in rf_results:
                overall = rf_results['overall_performance']
                print(f"📊 Overall R² Score: {overall.get('avg_r2_score', 'N/A'):.4f}")
                print(f"🎯 Overall Accuracy: {overall.get('avg_accuracy', 'N/A'):.4f}")
                print(f"🔍 Overall F1-Score: {overall.get('avg_f1_score', 'N/A'):.4f}")
                print(f"📉 Overall MSE: {overall.get('avg_mse', 'N/A'):.4f}")
            
            # Display recommendation performance
            if 'recommendation_accuracy' in rf_results:
                rec_acc = rf_results['recommendation_accuracy']
                if 'error' not in rec_acc:
                    print(f"✅ Recommendation Suitability Rate: {rec_acc.get('suitability_rate', 'N/A'):.4f}")
                    print(f"🎯 Condition Match Rate: {rec_acc.get('condition_match_rate', 'N/A'):.4f}")
        
        print("\n🔮 Framework ready for future model comparisons!")
        print("   - K-NN model evaluation can be added")
        print("   - XGBoost model evaluation can be added")
        print("   - Side-by-side performance comparison available")
