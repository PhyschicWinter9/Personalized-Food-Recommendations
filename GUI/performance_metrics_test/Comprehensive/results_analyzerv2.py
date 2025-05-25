#!/usr/bin/env python3
"""
Model Evaluation Results Analyzer
Analyze and visualize model evaluation results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class ModelResultsAnalyzer:
    """Comprehensive analyzer for model evaluation results"""
    
    def __init__(self, results_file):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.results = self.load_results()
        
    def load_results(self):
        """Load results from JSON file"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading results file: {e}")
    
    def get_performance_summary(self):
        """Get performance summary as DataFrame"""
        rankings = self.results.get('model_rankings', {})
        
        data = []
        for model_name, stats in rankings.items():
            data.append({
                'Model': model_name,
                'Accuracy': stats['overall_accuracy'],
                'F1_Score': stats['overall_f1_score'],
                'Precision': stats['overall_precision'],
                'Recall': stats['overall_recall'],
                'Accuracy_Std': stats['accuracy_std'],
                'F1_Std': stats['f1_std'],
                'Success_Rate': stats['success_rate'],
                'Tasks_Completed': stats['tasks_completed']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('F1_Score', ascending=False)
    
    def get_task_performance(self):
        """Get per-task performance as DataFrame"""
        task_results = self.results.get('task_results', {})
        
        data = []
        for task_name, task_data in task_results.items():
            for model_name, model_data in task_data.items():
                if 'error' in model_data:
                    continue
                
                metrics = model_data.get('metrics', {})
                cv_data = model_data.get('cross_validation', {})
                
                data.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'F1_Score': metrics.get('f1_score', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'CV_Accuracy_Mean': cv_data.get('accuracy', {}).get('mean', 0),
                    'CV_Accuracy_Std': cv_data.get('accuracy', {}).get('std', 0),
                    'Training_Time': model_data.get('training_time', 0)
                })
        
        return pd.DataFrame(data)
    
    def analyze_model_strengths(self):
        """Analyze each model's strengths and weaknesses"""
        task_df = self.get_task_performance()
        
        print("🔍 MODEL STRENGTHS & WEAKNESSES ANALYSIS")
        print("="*50)
        
        models = task_df['Model'].unique()
        
        for model in models:
            model_data = task_df[task_df['Model'] == model]
            
            print(f"\n📊 {model}:")
            print("-" * (len(model) + 5))
            
            # Best and worst tasks
            best_task = model_data.loc[model_data['F1_Score'].idxmax()]
            worst_task = model_data.loc[model_data['F1_Score'].idxmin()]
            
            print(f"✅ Best Performance: {best_task['Task']} (F1: {best_task['F1_Score']:.4f})")
            print(f"❌ Worst Performance: {worst_task['Task']} (F1: {worst_task['F1_Score']:.4f})")
            
            # Consistency analysis
            f1_std = model_data['F1_Score'].std()
            avg_f1 = model_data['F1_Score'].mean()
            consistency = 1 - (f1_std / avg_f1) if avg_f1 > 0 else 0
            
            print(f"📈 Average F1-Score: {avg_f1:.4f}")
            print(f"📊 Consistency Score: {consistency:.4f}")
            print(f"⏱️ Average Training Time: {model_data['Training_Time'].mean():.3f}s")
            
            # Speed vs Performance trade-off
            efficiency = avg_f1 / model_data['Training_Time'].mean() if model_data['Training_Time'].mean() > 0 else 0
            print(f"⚡ Efficiency (F1/Time): {efficiency:.2f}")
    
    def analyze_task_difficulty(self):
        """Analyze task difficulty based on model performance"""
        task_df = self.get_task_performance()
        
        print("\n📈 TASK DIFFICULTY ANALYSIS")
        print("="*35)
        
        task_stats = task_df.groupby('Task').agg({
            'Accuracy': ['mean', 'max', 'min', 'std'],
            'F1_Score': ['mean', 'max', 'min', 'std']
        }).round(4)
        
        # Flatten column names
        task_stats.columns = ['_'.join(col).strip() for col in task_stats.columns]
        
        # Calculate difficulty score (1 - max_accuracy)
        task_stats['Difficulty_Score'] = 1 - task_stats['Accuracy_max']
        task_stats['Performance_Variance'] = task_stats['F1_Score_std']
        
        # Sort by difficulty
        task_stats = task_stats.sort_values('Difficulty_Score')
        
        print("\nTask Rankings (Easiest to Hardest):")
        print("-" * 40)
        
        for i, (task, stats) in enumerate(task_stats.iterrows(), 1):
            difficulty = "Easy" if stats['Difficulty_Score'] < 0.2 else "Medium" if stats['Difficulty_Score'] < 0.4 else "Hard"
            print(f"{i}. {task}")
            print(f"   Difficulty: {difficulty} (Score: {stats['Difficulty_Score']:.3f})")
            print(f"   Best F1: {stats['F1_Score_max']:.3f}, Variance: {stats['Performance_Variance']:.3f}")
    
    def analyze_confusion_matrices(self):
        """Analyze confusion matrices for all tasks and models"""
        task_results = self.results.get('task_results', {})
        
        print("\n📊 CONFUSION MATRIX ANALYSIS")
        print("="*35)
        
        for task_name, task_data in task_results.items():
            print(f"\n🎯 Task: {task_name}")
            print("-" * (len(task_name) + 10))
            
            for model_name, model_data in task_data.items():
                if 'error' in model_data:
                    continue
                
                metrics = model_data.get('metrics', {})
                cm = metrics.get('confusion_matrix', [])
                
                if cm and len(cm) > 0:
                    cm_array = np.array(cm)
                    
                    print(f"\n🤖 {model_name}:")
                    print(f"   Confusion Matrix Shape: {cm_array.shape}")
                    
                    # Calculate metrics from confusion matrix
                    if cm_array.shape[0] == cm_array.shape[1]:  # Square matrix
                        total_correct = np.trace(cm_array)
                        total_samples = np.sum(cm_array)
                        accuracy = total_correct / total_samples if total_samples > 0 else 0
                        
                        print(f"   Total Predictions: {total_samples}")
                        print(f"   Correct Predictions: {total_correct}")
                        print(f"   Accuracy: {accuracy:.4f}")
                        
                        # Per-class accuracy
                        class_accuracy = metrics.get('class_wise_accuracy', [])
                        if class_accuracy:
                            print(f"   Class-wise Accuracy: {[f'{acc:.3f}' for acc in class_accuracy]}")
                        
                        # Most confused classes
                        if cm_array.shape[0] > 2:  # Multi-class
                            self._analyze_class_confusion(cm_array, model_name)
    
    def _analyze_class_confusion(self, cm_array, model_name):
        """Analyze which classes are most confused with each other"""
        # Find off-diagonal elements (misclassifications)
        np.fill_diagonal(cm_array, 0)  # Remove diagonal (correct predictions)
        
        # Find the highest confusion
        max_confusion_idx = np.unravel_index(np.argmax(cm_array), cm_array.shape)
        max_confusion_value = cm_array[max_confusion_idx]
        
        if max_confusion_value > 0:
            print(f"   Most Common Confusion: Class {max_confusion_idx[0]} → Class {max_confusion_idx[1]} ({max_confusion_value} cases)")
        
        # Find classes with highest error rates
        cm_with_diag = np.array(self.results['task_results'][list(self.results['task_results'].keys())[0]][model_name]['metrics']['confusion_matrix'])
        total_per_class = cm_with_diag.sum(axis=1)
        correct_per_class = np.diag(cm_with_diag)
        error_rates = 1 - (correct_per_class / total_per_class) if total_per_class.sum() > 0 else np.zeros_like(total_per_class)
        
        worst_class = np.argmax(error_rates)
        print(f"   Worst Performing Class: Class {worst_class} (Error Rate: {error_rates[worst_class]:.3f})")

        """Find the best model for each task"""
        task_df = self.get_task_performance()
        
        print("\n🏆 OPTIMAL MODEL PER TASK")
        print("="*30)
        
        optimal_models = {}
        
        for task in task_df['Task'].unique():
            task_data = task_df[task_df['Task'] == task]
            best_model = task_data.loc[task_data['F1_Score'].idxmax()]
            
            optimal_models[task] = {
                'model': best_model['Model'],
                'f1_score': best_model['F1_Score'],
                'accuracy': best_model['Accuracy']
            }
            
            print(f"📋 {task}:")
            print(f"   Best Model: {best_model['Model']}")
            print(f"   F1-Score: {best_model['F1_Score']:.4f}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        
    def find_optimal_model_per_task(self):
        """Find the best model for each task"""
        task_df = self.get_task_performance()
        
        print("\n🏆 OPTIMAL MODEL PER TASK")
        print("="*30)
        
        optimal_models = {}
        
        for task in task_df['Task'].unique():
            task_data = task_df[task_df['Task'] == task]
            best_model = task_data.loc[task_data['F1_Score'].idxmax()]
            
            optimal_models[task] = {
                'model': best_model['Model'],
                'f1_score': best_model['F1_Score'],
                'accuracy': best_model['Accuracy']
            }
            
            print(f"📋 {task}:")
            print(f"   Best Model: {best_model['Model']}")
            print(f"   F1-Score: {best_model['F1_Score']:.4f}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
        
        return optimal_models
    
    def create_confusion_matrix_visualizations(self, output_dir="./analysis_output", show_plots=False):
        """Create confusion matrix visualizations"""
        task_results = self.results.get('task_results', {})
        
        for task_name, task_data in task_results.items():
            # Create a figure for this task's confusion matrices
            models_with_cm = []
            cms = []
            
            for model_name, model_data in task_data.items():
                if 'error' in model_data:
                    continue
                
                metrics = model_data.get('metrics', {})
                cm = metrics.get('confusion_matrix', [])
                
                if cm and len(cm) > 0:
                    models_with_cm.append(model_name)
                    cms.append(np.array(cm))
            
            if not models_with_cm:
                continue
            
            # Create subplot for each model
            n_models = len(models_with_cm)
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
            
            if n_models == 1:
                axes = [axes]
            
            fig.suptitle(f'Confusion Matrices - {task_name}', fontsize=14, fontweight='bold')
            
            for i, (model_name, cm) in enumerate(zip(models_with_cm, cms)):
                ax = axes[i]
                
                # Normalize confusion matrix for better visualization
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
                
                # Create heatmap
                im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
                ax.set_title(f'{model_name}', fontweight='bold')
                
                # Add text annotations
                thresh = cm_normalized.max() / 2.
                for row in range(cm.shape[0]):
                    for col in range(cm.shape[1]):
                        ax.text(col, row, f'{cm[row, col]}\n({cm_normalized[row, col]:.2f})',
                               ha="center", va="center",
                               color="white" if cm_normalized[row, col] > thresh else "black",
                               fontsize=8)
                
                # Labels
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                
                # Ticks
                n_classes = cm.shape[0]
                ax.set_xticks(range(n_classes))
                ax.set_yticks(range(n_classes))
                ax.set_xticklabels(range(n_classes))
                ax.set_yticklabels(range(n_classes))
                
                # Colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save the figure
            safe_task_name = task_name.replace(' ', '_').replace('/', '_')
            plt.savefig(f"{output_dir}/confusion_matrices_{safe_task_name}.png", 
                       dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            plt.close()
        
        print(f"\n📊 Confusion matrix visualizations saved to {output_dir}/")
        print("   - confusion_matrices_[task_name].png for each task")
    
    def generate_model_recommendation(self):
        """Generate model recommendations based on analysis"""
        summary_df = self.get_performance_summary()
        task_df = self.get_task_performance()
        
        print("\n💡 MODEL RECOMMENDATIONS")
        print("="*30)
        
        # Overall best model
        best_overall = summary_df.iloc[0]
        print(f"🥇 Best Overall Model: {best_overall['Model']}")
        print(f"   Reason: Highest average F1-Score ({best_overall['F1_Score']:.4f})")
        print(f"   Success Rate: {best_overall['Success_Rate']:.1%}")
        
        # Most consistent model
        most_consistent = summary_df.loc[summary_df['F1_Std'].idxmin()]
        print(f"\n🎯 Most Consistent Model: {most_consistent['Model']}")
        print(f"   Reason: Lowest F1-Score variance ({most_consistent['F1_Std']:.4f})")
        
        # Fastest model (if training time data available)
        if 'Training_Time' in task_df.columns:
            avg_times = task_df.groupby('Model')['Training_Time'].mean()
            fastest_model = avg_times.idxmin()
            fastest_time = avg_times.min()
            
            print(f"\n⚡ Fastest Model: {fastest_model}")
            print(f"   Reason: Shortest average training time ({fastest_time:.3f}s)")
        
        # Balanced recommendation
        # Calculate efficiency score (performance / training_time)
        if 'Training_Time' in task_df.columns:
            model_efficiency = task_df.groupby('Model').apply(
                lambda x: x['F1_Score'].mean() / x['Training_Time'].mean() if x['Training_Time'].mean() > 0 else 0
            )
            most_efficient = model_efficiency.idxmax()
            
            print(f"\n⚖️ Most Balanced Model: {most_efficient}")
            print(f"   Reason: Best performance-to-speed ratio ({model_efficiency[most_efficient]:.2f})")
        
        # Task-specific recommendations
        print(f"\n📋 Task-Specific Recommendations:")
        optimal_per_task = self.find_optimal_model_per_task()
        
        # Count wins per model
        model_wins = {}
        for task, info in optimal_per_task.items():
            model = info['model']
            model_wins[model] = model_wins.get(model, 0) + 1
        
        print(f"\n🏆 Task Wins by Model:")
        for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
            tasks_won = [task for task, info in optimal_per_task.items() if info['model'] == model]
            print(f"   {model}: {wins} task(s) - {', '.join(tasks_won)}")
    
    def export_summary_tables(self, output_dir="./analysis_output"):
        """Export analysis tables to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance summary
        summary_df = self.get_performance_summary()
        summary_df.to_csv(f"{output_dir}/model_performance_summary.csv", index=False)
        
        # Task performance
        task_df = self.get_task_performance()
        task_df.to_csv(f"{output_dir}/task_performance_details.csv", index=False)
        
        # Optimal models per task
        optimal_models = self.find_optimal_model_per_task()
        optimal_df = pd.DataFrame([
            {'Task': task, 'Best_Model': info['model'], 'F1_Score': info['f1_score'], 'Accuracy': info['accuracy']}
            for task, info in optimal_models.items()
        ])
        optimal_df.to_csv(f"{output_dir}/optimal_models_per_task.csv", index=False)
        
        # Confusion matrix summary
        self._export_confusion_matrix_summary(output_dir)
        
        print(f"\n📁 Analysis tables exported to {output_dir}/")
        print("   - model_performance_summary.csv")
        print("   - task_performance_details.csv") 
        print("   - optimal_models_per_task.csv")
        print("   - confusion_matrix_summary.csv")
    
    def _export_confusion_matrix_summary(self, output_dir):
        """Export confusion matrix summary to CSV"""
        task_results = self.results.get('task_results', {})
        
        cm_data = []
        for task_name, task_data in task_results.items():
            for model_name, model_data in task_data.items():
                if 'error' in model_data:
                    continue
                
                metrics = model_data.get('metrics', {})
                cm = metrics.get('confusion_matrix', [])
                
                if cm and len(cm) > 0:
                    cm_array = np.array(cm)
                    total_correct = np.trace(cm_array)
                    total_samples = np.sum(cm_array)
                    accuracy = total_correct / total_samples if total_samples > 0 else 0
                    
                    # Calculate class-wise accuracy
                    class_accuracy = metrics.get('class_wise_accuracy', [])
                    min_class_acc = min(class_accuracy) if class_accuracy else 0
                    max_class_acc = max(class_accuracy) if class_accuracy else 0
                    avg_class_acc = np.mean(class_accuracy) if class_accuracy else 0
                    
                    cm_data.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Matrix_Shape': f"{cm_array.shape[0]}x{cm_array.shape[1]}",
                        'Total_Predictions': total_samples,
                        'Correct_Predictions': total_correct,
                        'CM_Accuracy': accuracy,
                        'Min_Class_Accuracy': min_class_acc,
                        'Max_Class_Accuracy': max_class_acc,
                        'Avg_Class_Accuracy': avg_class_acc,
                        'Num_Classes': cm_array.shape[0]
                    })
        
        if cm_data:
            cm_df = pd.DataFrame(cm_data)
            cm_df.to_csv(f"{output_dir}/confusion_matrix_summary.csv", index=False)
    
    def create_visualizations(self, output_dir="./analysis_output", show_plots=False):
        """Create visualization plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        summary_df = self.get_performance_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        axes[0,0].bar(summary_df['Model'], summary_df['Accuracy'], color='skyblue', alpha=0.7)
        axes[0,0].set_title('Overall Accuracy')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0,1].bar(summary_df['Model'], summary_df['F1_Score'], color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Overall F1-Score')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        axes[1,0].scatter(summary_df['Precision'], summary_df['Recall'], s=100, alpha=0.7)
        for i, model in enumerate(summary_df['Model']):
            axes[1,0].annotate(model, (summary_df.iloc[i]['Precision'], summary_df.iloc[i]['Recall']))
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_ylabel('Recall')
        axes[1,0].set_title('Precision vs Recall')
        
        # Success Rate
        axes[1,1].bar(summary_df['Model'], summary_df['Success_Rate'], color='orange', alpha=0.7)
        axes[1,1].set_title('Task Success Rate')
        axes[1,1].set_ylabel('Success Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_performance_comparison.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
        
        # 2. Task Performance Heatmap
        task_df = self.get_task_performance()
        pivot_df = task_df.pivot(index='Task', columns='Model', values='F1_Score')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.5, fmt='.3f',
                    cbar_kws={'label': 'F1-Score'})
        plt.title('F1-Score Performance by Task and Model', fontsize=14, fontweight='bold')
        plt.ylabel('Classification Task')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/task_performance_heatmap.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
        
        # 3. Model Consistency Analysis
        plt.figure(figsize=(10, 6))
        x = summary_df['F1_Score']
        y = summary_df['F1_Std']
        plt.scatter(x, y, s=100, alpha=0.7)
        
        for i, model in enumerate(summary_df['Model']):
            plt.annotate(model, (x.iloc[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average F1-Score')
        plt.ylabel('F1-Score Standard Deviation')
        plt.title('Model Performance vs Consistency', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add quadrant labels
        plt.axhline(y.mean(), color='red', linestyle='--', alpha=0.5)
        plt.axvline(x.mean(), color='red', linestyle='--', alpha=0.5)
        
        plt.text(x.max()*0.8, y.max()*0.9, 'High Performance\nLow Consistency', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        plt.text(x.max()*0.8, y.min()*1.1, 'High Performance\nHigh Consistency', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_consistency_analysis.png", dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()
        
        print(f"\n📊 Visualizations saved to {output_dir}/")
        print("   - model_performance_comparison.png")
        print("   - task_performance_heatmap.png")
        print("   - model_consistency_analysis.png")
        
        # Create confusion matrix visualizations
        self.create_confusion_matrix_visualizations(output_dir, show_plots)
    
    def run_full_analysis(self, export_tables=True, create_plots=True, show_plots=False):
        """Run complete analysis"""
        print("🔍 COMPREHENSIVE MODEL EVALUATION ANALYSIS")
        print("="*50)
        
        # Print metadata
        metadata = self.results.get('evaluation_metadata', {})
        print(f"\n📊 Evaluation Overview:")
        print(f"   Dataset Size: {metadata.get('dataset_size', 'N/A')} samples")
        print(f"   Features: {metadata.get('num_features', 'N/A')}")
        print(f"   Models Tested: {', '.join(metadata.get('models_evaluated', []))}")
        print(f"   Classification Tasks: {len(metadata.get('classification_tasks', []))}")
        
        # Run analyses
        self.analyze_model_strengths()
        self.analyze_task_difficulty()
        self.analyze_confusion_matrices()
        self.generate_model_recommendation()
        
        # Export tables and plots
        if export_tables:
            self.export_summary_tables()
        
        if create_plots:
            self.create_visualizations(show_plots=show_plots)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Analyze model evaluation results')
    parser.add_argument('results_file', help='Path to JSON results file')
    parser.add_argument('--export-tables', action='store_true', help='Export summary tables to CSV')
    parser.add_argument('--create-plots', action='store_true', help='Create visualization plots')
    parser.add_argument('--show-plots', action='store_true', help='Display plots interactively')
    parser.add_argument('--output-dir', default='./analysis_output', help='Output directory for exports')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    try:
        # Initialize analyzer
        analyzer = ModelResultsAnalyzer(args.results_file)
        
        # Run analysis
        analyzer.run_full_analysis(
            export_tables=args.export_tables,
            create_plots=args.create_plots,
            show_plots=args.show_plots
        )
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()