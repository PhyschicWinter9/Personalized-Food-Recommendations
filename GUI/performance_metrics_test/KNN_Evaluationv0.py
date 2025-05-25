import pandas as pd
import numpy as np
import glob
import os
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class KNNFoodRecommendationAnalysis:
    """Comprehensive KNN analysis for food recommendation system"""
    
    def __init__(self):
        self.food_data = pd.DataFrame()
        self.features = []
        self.knn_models = {}
        self.scaler = StandardScaler()
        self.performance_metrics = {}
        
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
        
        print("🔬 KNN Food Recommendation Analysis System")
        print("=" * 50)
    
    def load_data(self):
        """Load and prepare dataset"""
        print("📁 Loading dataset...")
        
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
                    print(f"  ✅ Loaded {len(df)} items from {category}")
                    
            except Exception as e:
                print(f"  ❌ Error loading {file_path}: {e}")
        
        if dataframes:
            self.food_data = pd.concat(dataframes, ignore_index=True)
            print(f"📊 Total dataset: {len(self.food_data)} food items")
            self._calculate_health_scores()
            self._prepare_features()
        else:
            raise Exception("No valid data loaded")
    
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
        """Calculate health suitability scores"""
        print("🏥 Calculating health scores...")
        
        for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
            self.food_data[f'{condition}_Score'] = 0.0
        
        for idx, food in self.food_data.iterrows():
            # Diabetes score calculation
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
            
            # Obesity score calculation
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
            
            # Hypertension score calculation
            score = 0
            sodium = float(food.get('Na(mg)', 0))
            potassium = float(food.get('K(mg)', 0))
            
            if sodium > 400: score += 3
            elif sodium > 200: score += 2
            elif sodium > 100: score += 1
            
            if potassium > 300: score -= 1
            if fiber >= 5: score -= 0.5
            
            self.food_data.at[idx, 'Hypertension_Score'] = max(0, score)
            
            # High cholesterol score calculation
            score = 0
            sat_fat = float(food.get('FASAT (g) Saturated FA', 0))
            cholesterol = float(food.get('CHOLE(mg) Cholesterol', 0))
            
            if sat_fat > 5: score += 3
            elif sat_fat > 3: score += 2
            elif sat_fat > 1: score += 1
            
            if cholesterol > 100: score += 2
            elif cholesterol > 50: score += 1
            
            if fiber >= 5: score -= 1
            
            self.food_data.at[idx, 'High_Cholesterol_Score'] = max(0, score)
        
        # Calculate overall health score
        self.food_data['Overall_Health_Score'] = (
            self.food_data['Diabetes_Score'] + 
            self.food_data['Obesity_Score'] + 
            self.food_data['Hypertension_Score'] + 
            self.food_data['High_Cholesterol_Score']
        ) / 4
        
        print("  ✅ Health scores calculated")
    
    def _prepare_features(self):
        """Prepare features for KNN"""
        available_features = [f for f in self.nutritional_features if f in self.food_data.columns]
        health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
        self.features = available_features + health_features
        
        print(f"🔬 Prepared {len(self.features)} features for KNN:")
        for i, feature in enumerate(self.features, 1):
            print(f"  {i}. {feature}")
    
    def train_knn_models(self):
        """Train KNN models for different health conditions"""
        print("\n🤖 Training KNN Models...")
        print("-" * 30)
        
        X = self.food_data[self.features].fillna(0)
        
        # Standardize features for KNN
        X_scaled = self.scaler.fit_transform(X)
        
        # Different target variables
        targets = {
            'General': self.food_data['Overall_Health_Score'],
            'Diabetes': self.food_data['Diabetes_Score'],
            'Obesity': self.food_data['Obesity_Score'],
            'Hypertension': self.food_data['Hypertension_Score'],
            'High_Cholesterol': self.food_data['High_Cholesterol_Score']
        }
        
        # Test different k values
        k_values = [3, 5, 7, 10, 15]
        best_k_results = {}
        
        for target_name, y in targets.items():
            print(f"\n📊 Training KNN for {target_name}...")
            
            best_k = 5
            best_score = -1
            k_scores = {}
            
            # Find optimal k value
            for k in k_values:
                try:
                    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='r2')
                    avg_score = cv_scores.mean()
                    k_scores[k] = avg_score
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_k = k
                        
                except Exception as e:
                    print(f"    ❌ Error with k={k}: {e}")
            
            print(f"  📈 K-value scores: {k_scores}")
            print(f"  🎯 Best k={best_k} with CV R² = {best_score:.4f}")
            
            # Train final model with best k
            best_knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
            best_knn.fit(X_scaled, y)
            self.knn_models[target_name] = best_knn
            best_k_results[target_name] = {'k': best_k, 'cv_score': best_score}
        
        print("\n✅ KNN model training completed!")
        return best_k_results
    
    def evaluate_knn_performance(self):
        """Comprehensive KNN performance evaluation"""
        print("\n📊 KNN Performance Evaluation")
        print("=" * 40)
        
        X = self.food_data[self.features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        targets = {
            'General': self.food_data['Overall_Health_Score'],
            'Diabetes': self.food_data['Diabetes_Score'],
            'Obesity': self.food_data['Obesity_Score'],
            'Hypertension': self.food_data['Hypertension_Score'],
            'High_Cholesterol': self.food_data['High_Cholesterol_Score']
        }
        
        results = []
        
        for target_name, y in targets.items():
            if target_name not in self.knn_models:
                continue
                
            print(f"\n🎯 Evaluating {target_name} KNN...")
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Get model
            knn = self.knn_models[target_name]
            
            # Re-train on training set for fair evaluation
            knn.fit(X_train, y_train)
            
            # Make predictions
            y_pred = knn.predict(X_test)
            
            # Calculate regression metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate classification metrics
            y_test_binary = (y_test > y_test.median()).astype(int)
            y_pred_binary = (y_pred > y_test.median()).astype(int)
            
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            
            # Cross-validation scores
            cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            metrics = {
                'Target': target_name,
                'R2_Score': r2,
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'CV_Mean': cv_mean,
                'CV_Std': cv_std,
                'K_Value': knn.n_neighbors
            }
            
            results.append(metrics)
            self.performance_metrics[target_name] = metrics
            
            # Print results
            print(f"  📈 R² Score: {r2:.4f}")
            print(f"  📉 MSE: {mse:.4f}")
            print(f"  🎯 Accuracy: {accuracy:.4f}")
            print(f"  🔄 F1 Score: {f1:.4f}")
            print(f"  ✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return results
    
    def get_knn_recommendations(self, user_profile, health_conditions=None, max_recommendations=10):
        """Get KNN-based food recommendations"""
        print(f"\n🎯 Getting KNN Recommendations...")
        print("-" * 35)
        
        # Create user nutritional target vector
        user_vector = self._create_user_vector(user_profile)
        user_vector_scaled = self.scaler.transform(user_vector.reshape(1, -1))
        
        X = self.food_data[self.features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        recommendations = {}
        
        # Get recommendations from different models
        model_types = ['General']
        if health_conditions:
            for condition in health_conditions:
                if condition in self.knn_models:
                    model_types.append(condition)
        
        for model_type in model_types:
            print(f"  🔍 {model_type} KNN recommendations...")
            
            if model_type not in self.knn_models:
                continue
            
            try:
                # Get feature weights for health conditions
                if model_type != 'General':
                    feature_weights = self._get_condition_feature_weights(model_type)
                    weighted_X = X_scaled * feature_weights
                    weighted_user = user_vector_scaled * feature_weights
                else:
                    weighted_X = X_scaled
                    weighted_user = user_vector_scaled
                
                # Calculate distances manually
                distances = []
                for i, food_features in enumerate(weighted_X):
                    dist = euclidean(weighted_user[0], food_features)
                    distances.append((dist, i))
                
                # Sort by distance and get top recommendations
                distances.sort(key=lambda x: x[0])
                top_distances = distances[:max_recommendations]
                
                model_recommendations = []
                for dist, idx in top_distances:
                    food = self.food_data.iloc[idx]
                    
                    rec = {
                        'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                        'category': food.get('Category', 'Unknown'),
                        'calories': float(food.get('Energy(kcal) by calculation', 0)),
                        'protein': float(food.get('Protein(g)', 0)),
                        'carbs': float(food.get('CHOCDF (g) Carbohydrate', 0)),
                        'sugar': float(food.get('SUGAR(g)', 0)),
                        'fiber': float(food.get('FIBTG (g) Dietary fibre', 0)),
                        'fat': float(food.get('Fat(g)', 0)),
                        'sodium': float(food.get('Na(mg)', 0)),
                        'distance': dist,
                        'health_scores': {
                            'diabetes': float(food.get('Diabetes_Score', 0)),
                            'obesity': float(food.get('Obesity_Score', 0)),
                            'hypertension': float(food.get('Hypertension_Score', 0)),
                            'cholesterol': float(food.get('High_Cholesterol_Score', 0))
                        },
                        'suitable_for': self._check_suitability(food)
                    }
                    
                    model_recommendations.append(rec)
                
                recommendations[model_type] = model_recommendations
                print(f"    ✅ Found {len(model_recommendations)} recommendations")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        return recommendations
    
    def _create_user_vector(self, user_profile):
        """Create user nutritional target vector"""
        # Sample nutritional targets (can be customized based on user profile)
        target_calories = user_profile.get('target_calories', 500)
        target_protein = user_profile.get('target_protein', 25)
        target_carbs = user_profile.get('target_carbs', 60)
        target_sugar = user_profile.get('target_sugar', 10)
        target_fiber = user_profile.get('target_fiber', 8)
        target_fat = user_profile.get('target_fat', 15)
        target_sodium = user_profile.get('target_sodium', 600)
        
        user_vector = np.zeros(len(self.features))
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
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
        
        # Set ideal health scores (0 = best)
        for health_feature in ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']:
            if health_feature in feature_map:
                user_vector[feature_map[health_feature]] = 0
        
        return user_vector
    
    def _get_condition_feature_weights(self, condition):
        """Get feature weights for specific health conditions"""
        base_weights = np.ones(len(self.features))
        feature_map = {feature: i for i, feature in enumerate(self.features)}
        
        if condition == 'Diabetes':
            if 'SUGAR(g)' in feature_map:
                base_weights[feature_map['SUGAR(g)']] = 3.0
            if 'CHOCDF (g) Carbohydrate' in feature_map:
                base_weights[feature_map['CHOCDF (g) Carbohydrate']] = 2.0
            if 'FIBTG (g) Dietary fibre' in feature_map:
                base_weights[feature_map['FIBTG (g) Dietary fibre']] = 2.0
        
        elif condition == 'Hypertension':
            if 'Na(mg)' in feature_map:
                base_weights[feature_map['Na(mg)']] = 3.0
            if 'K(mg)' in feature_map:
                base_weights[feature_map['K(mg)']] = 2.0
        
        elif condition == 'High_Cholesterol':
            if 'Fat(g)' in feature_map:
                base_weights[feature_map['Fat(g)']] = 2.0
            if 'CHOLE(mg) Cholesterol' in feature_map:
                base_weights[feature_map['CHOLE(mg) Cholesterol']] = 3.0
            if 'FIBTG (g) Dietary fibre' in feature_map:
                base_weights[feature_map['FIBTG (g) Dietary fibre']] = 2.0
        
        elif condition == 'Obesity':
            if 'Energy(kcal) by calculation' in feature_map:
                base_weights[feature_map['Energy(kcal) by calculation']] = 3.0
            if 'Fat(g)' in feature_map:
                base_weights[feature_map['Fat(g)']] = 2.0
            if 'Protein(g)' in feature_map:
                base_weights[feature_map['Protein(g)']] = 1.5
        
        return base_weights
    
    def _check_suitability(self, food):
        """Check which health conditions this food is suitable for"""
        suitable = []
        
        if float(food.get('Diabetes_Score', 0)) <= 1.5:
            suitable.append('Diabetes')
        if float(food.get('Obesity_Score', 0)) <= 1.5:
            suitable.append('Obesity')
        if float(food.get('Hypertension_Score', 0)) <= 1.5:
            suitable.append('Hypertension')
        if float(food.get('High_Cholesterol_Score', 0)) <= 1.5:
            suitable.append('High_Cholesterol')
        
        return suitable
    
    def print_performance_summary(self, results):
        """Print comprehensive performance summary"""
        print("\n" + "="*60)
        print("📊 KNN PERFORMANCE SUMMARY")
        print("="*60)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        print("\n🎯 DETAILED PERFORMANCE METRICS:")
        print("-" * 45)
        
        for _, row in df.iterrows():
            print(f"\n{row['Target']} KNN (k={row['K_Value']}):")
            print(f"  R² Score:     {row['R2_Score']:.4f}")
            print(f"  MSE:          {row['MSE']:.4f}")
            print(f"  MAE:          {row['MAE']:.4f}")
            print(f"  RMSE:         {row['RMSE']:.4f}")
            print(f"  Accuracy:     {row['Accuracy']:.4f}")
            print(f"  Precision:    {row['Precision']:.4f}")
            print(f"  Recall:       {row['Recall']:.4f}")
            print(f"  F1 Score:     {row['F1_Score']:.4f}")
            print(f"  CV Score:     {row['CV_Mean']:.4f} ± {row['CV_Std']:.4f}")
        
        # Overall statistics
        print(f"\n📈 AVERAGE PERFORMANCE ACROSS ALL TARGETS:")
        print("-" * 45)
        print(f"Average R² Score:    {df['R2_Score'].mean():.4f}")
        print(f"Average Accuracy:    {df['Accuracy'].mean():.4f}")
        print(f"Average Precision:   {df['Precision'].mean():.4f}")
        print(f"Average Recall:      {df['Recall'].mean():.4f}")
        print(f"Average F1 Score:    {df['F1_Score'].mean():.4f}")
        
        # Best performing targets
        print(f"\n🏆 BEST PERFORMING TARGETS:")
        print("-" * 30)
        best_r2 = df.loc[df['R2_Score'].idxmax()]
        best_acc = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1_Score'].idxmax()]
        
        print(f"Best R² Score:    {best_r2['Target']} ({best_r2['R2_Score']:.4f})")
        print(f"Best Accuracy:    {best_acc['Target']} ({best_acc['Accuracy']:.4f})")
        print(f"Best F1 Score:    {best_f1['Target']} ({best_f1['F1_Score']:.4f})")
        
        return df
    
    def print_recommendations_summary(self, recommendations):
        """Print recommendations summary"""
        print("\n" + "="*60)
        print("🍽️ KNN RECOMMENDATIONS SUMMARY")
        print("="*60)
        
        for model_type, recs in recommendations.items():
            print(f"\n🎯 {model_type.upper()} KNN RECOMMENDATIONS:")
            print("-" * 40)
            
            if not recs:
                print("  No recommendations available")
                continue
            
            for i, rec in enumerate(recs[:5], 1):  # Show top 5
                print(f"\n  {i}. {rec['name']}")
                print(f"     Category: {rec['category']}")
                print(f"     Calories: {rec['calories']:.0f} kcal")
                print(f"     Protein: {rec['protein']:.1f}g | Carbs: {rec['carbs']:.1f}g | Fat: {rec['fat']:.1f}g")
                print(f"     Sugar: {rec['sugar']:.1f}g | Fiber: {rec['fiber']:.1f}g | Sodium: {rec['sodium']:.0f}mg")
                print(f"     KNN Distance: {rec['distance']:.4f}")
                print(f"     Suitable for: {', '.join(rec['suitable_for']) if rec['suitable_for'] else 'General use'}")
                
                # Health scores
                health = rec['health_scores']
                print(f"     Health Scores - Diabetes: {health['diabetes']:.1f}, Obesity: {health['obesity']:.1f}")
                print(f"                     Hypertension: {health['hypertension']:.1f}, Cholesterol: {health['cholesterol']:.1f}")

def main():
    """Main function to run KNN analysis"""
    try:
        print("🚀 Starting KNN Food Recommendation Analysis")
        print("=" * 50)
        
        # Initialize analysis
        knn_analysis = KNNFoodRecommendationAnalysis()
        
        # Load and prepare data
        knn_analysis.load_data()
        
        # Train KNN models
        k_results = knn_analysis.train_knn_models()
        
        # Evaluate performance
        performance_results = knn_analysis.evaluate_knn_performance()
        
        # Print performance summary
        performance_df = knn_analysis.print_performance_summary(performance_results)
        
        # Get sample recommendations
        print("\n🎯 Getting Sample Recommendations...")
        user_profile = {
            'target_calories': 500,
            'target_protein': 25,
            'target_carbs': 60,
            'target_sugar': 8,
            'target_fiber': 10,
            'target_fat': 15,
            'target_sodium': 500
        }
        
        health_conditions = ['Diabetes', 'Hypertension']  # Sample conditions
        
        recommendations = knn_analysis.get_knn_recommendations(
            user_profile, 
            health_conditions=health_conditions,
            max_recommendations=10
        )
        
        # Print recommendations summary
        knn_analysis.print_recommendations_summary(recommendations)
        
        # Final summary
        print(f"\n" + "="*60)
        print("✅ KNN ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Dataset: {len(knn_analysis.food_data)} foods, {len(knn_analysis.features)} features")
        print(f"🤖 Models trained: {len(knn_analysis.knn_models)} KNN variants")
        print(f"📈 Average R² Score: {performance_df['R2_Score'].mean():.4f}")
        print(f"🎯 Average Accuracy: {performance_df['Accuracy'].mean():.4f}")
        print(f"🏆 Best model: {performance_df.loc[performance_df['F1_Score'].idxmax()]['Target']} KNN")
        
        # Recommendations generated
        total_recs = sum(len(recs) for recs in recommendations.values())
        print(f"🍽️ Recommendations generated: {total_recs}")
        
        print("\n💡 KNN Strengths for Food Recommendation:")
        print("   • Similarity-based matching finds foods with similar nutritional profiles")
        print("   • Interpretable results - can see why foods are recommended")
        print("   • Flexible feature weighting for different health conditions")
        print("   • No complex training process - works well with small datasets")
        print("   • Good performance for personalized nutrition matching")
        
    except Exception as e:
        print(f"❌ Error during KNN analysis: {e}")
        raise

if __name__ == "__main__":
    main()