import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the food dataset"""
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
    
    nutritional_features = [
        'Energy(kcal) by calculation', 'Protein(g)', 'CHOCDF (g) Carbohydrate',
        'SUGAR(g)', 'FIBTG (g) Dietary fibre', 'Fat(g)',
        'Na(mg)', 'K(mg)', 'Ca(mg)', 'CHOLE(mg) Cholesterol'
    ]
    
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0].lower()
            category = category_map.get(base_name, base_name.title())
            
            df = pd.read_csv(file_path)
            df['Category'] = category
            
            # Clean data
            for col in nutritional_features + ['FASAT (g) Saturated FA']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Remove rows with all zero nutritional values
            nutrient_cols = [col for col in nutritional_features if col in df.columns]
            if nutrient_cols:
                df = df[df[nutrient_cols].sum(axis=1) > 0]
            
            if len(df) > 0:
                dataframes.append(df)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not dataframes:
        raise Exception("No valid data loaded")
    
    food_data = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Loaded {len(food_data)} food items")
    
    # Calculate health scores
    print("🏥 Calculating health scores...")
    
    for condition in ['Diabetes', 'Obesity', 'Hypertension', 'High_Cholesterol']:
        food_data[f'{condition}_Score'] = 0.0
    
    for idx, food in food_data.iterrows():
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
        
        food_data.at[idx, 'Diabetes_Score'] = max(0, score)
        
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
        
        food_data.at[idx, 'Obesity_Score'] = max(0, score)
        
        # Hypertension score
        score = 0
        sodium = float(food.get('Na(mg)', 0))
        potassium = float(food.get('K(mg)', 0))
        
        if sodium > 400: score += 3
        elif sodium > 200: score += 2
        elif sodium > 100: score += 1
        
        if potassium > 300: score -= 1
        if fiber >= 5: score -= 0.5
        
        food_data.at[idx, 'Hypertension_Score'] = max(0, score)
        
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
        
        food_data.at[idx, 'High_Cholesterol_Score'] = max(0, score)
    
    # Calculate overall health score
    food_data['Overall_Health_Score'] = (
        food_data['Diabetes_Score'] + 
        food_data['Obesity_Score'] + 
        food_data['Hypertension_Score'] + 
        food_data['High_Cholesterol_Score']
    ) / 4
    
    # Prepare features
    available_features = [f for f in nutritional_features if f in food_data.columns]
    health_features = ['Diabetes_Score', 'Obesity_Score', 'Hypertension_Score', 'High_Cholesterol_Score']
    features = available_features + health_features
    
    print(f"🔬 Prepared {len(features)} features")
    
    return food_data, features

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance with comprehensive metrics"""
    y_pred = model.predict(X_test)
    
    # Regression metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Classification metrics (convert to binary)
    y_test_binary = (y_test > y_test.median()).astype(int)
    y_pred_binary = (y_pred > y_test.median()).astype(int)
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
    
    return {
        'Model': model_name,
        'R2_Score': r2,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }

def train_and_evaluate_models(food_data, features):
    """Train and evaluate all three models"""
    print("🤖 Training and evaluating models...")
    
    X = food_data[features].fillna(0)
    
    # Target variables
    targets = {
        'Overall_Health': food_data['Overall_Health_Score'],
        'Diabetes': food_data['Diabetes_Score'],
        'Obesity': food_data['Obesity_Score'],
        'Hypertension': food_data['Hypertension_Score'],
        'High_Cholesterol': food_data['High_Cholesterol_Score']
    }
    
    results = []
    
    for target_name, y in targets.items():
        print(f"  📊 Evaluating {target_name} target...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for KNN and MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. KNN Model
        try:
            knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn.fit(X_train_scaled, y_train)
            knn_metrics = evaluate_model(knn, X_test_scaled, y_test, 'KNN')
            knn_metrics['Target'] = target_name
            results.append(knn_metrics)
        except Exception as e:
            print(f"    ❌ KNN error: {e}")
        
        # 2. Random Forest Model
        try:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_metrics = evaluate_model(rf, X_test, y_test, 'Random Forest')
            rf_metrics['Target'] = target_name
            results.append(rf_metrics)
        except Exception as e:
            print(f"    ❌ Random Forest error: {e}")
        
        # 3. MLP Model
        try:
            mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
            mlp.fit(X_train_scaled, y_train)
            mlp_metrics = evaluate_model(mlp, X_test_scaled, y_test, 'MLP')
            mlp_metrics['Target'] = target_name
            results.append(mlp_metrics)
        except Exception as e:
            print(f"    ❌ MLP error: {e}")
    
    return results

def generate_recommendations_comparison(food_data, features):
    """Generate sample recommendations from all models"""
    print("🎯 Generating sample recommendations...")
    
    X = food_data[features].fillna(0)
    y = food_data['Overall_Health_Score']
    
    # Train models on full dataset for recommendations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sample user target (simplified)
    target_calories = 500
    target_protein = 25
    target_carbs = 60
    
    recommendations = {'KNN': [], 'Random Forest': [], 'MLP': []}
    
    try:
        # KNN Recommendations
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(X_scaled, y)
        
        # Create user vector
        user_vector = np.zeros(len(features))
        feature_map = {feature: i for i, feature in enumerate(features)}
        
        if 'Energy(kcal) by calculation' in feature_map:
            user_vector[feature_map['Energy(kcal) by calculation']] = target_calories
        if 'Protein(g)' in feature_map:
            user_vector[feature_map['Protein(g)']] = target_protein
        if 'CHOCDF (g) Carbohydrate' in feature_map:
            user_vector[feature_map['CHOCDF (g) Carbohydrate']] = target_carbs
        
        user_vector_scaled = scaler.transform(user_vector.reshape(1, -1))
        distances, indices = knn.kneighbors(user_vector_scaled, n_neighbors=5)
        
        for idx in indices[0]:
            food = food_data.iloc[idx]
            recommendations['KNN'].append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'health_score': float(food.get('Overall_Health_Score', 0))
            })
    
    except Exception as e:
        print(f"    ❌ KNN recommendations error: {e}")
    
    try:
        # Random Forest Recommendations
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        predicted_scores = rf.predict(X)
        best_indices = np.argsort(predicted_scores)[:5]  # Best 5 (lowest scores)
        
        for idx in best_indices:
            food = food_data.iloc[idx]
            recommendations['Random Forest'].append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'health_score': float(food.get('Overall_Health_Score', 0))
            })
    
    except Exception as e:
        print(f"    ❌ Random Forest recommendations error: {e}")
    
    try:
        # MLP Recommendations
        mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_scaled, y)
        
        predicted_scores = mlp.predict(X_scaled)
        best_indices = np.argsort(predicted_scores)[:5]  # Best 5 (lowest scores)
        
        for idx in best_indices:
            food = food_data.iloc[idx]
            recommendations['MLP'].append({
                'name': food.get('Thai_Name', food.get('English_Name', f"Food {idx}")),
                'category': food.get('Category', 'Unknown'),
                'calories': float(food.get('Energy(kcal) by calculation', 0)),
                'protein': float(food.get('Protein(g)', 0)),
                'health_score': float(food.get('Overall_Health_Score', 0))
            })
    
    except Exception as e:
        print(f"    ❌ MLP recommendations error: {e}")
    
    return recommendations

def print_results(results, recommendations):
    """Print comprehensive results"""
    print("\n" + "="*80)
    print("🔬 MODEL PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    summary_stats = df.groupby('Model').agg({
        'R2_Score': ['mean', 'std'],
        'Accuracy': ['mean', 'std'],
        'Precision': ['mean', 'std'],
        'Recall': ['mean', 'std'],
        'F1_Score': ['mean', 'std']
    }).round(4)
    
    print(summary_stats)
    
    # Detailed Results by Target
    print("\n📋 DETAILED RESULTS BY TARGET:")
    print("-" * 50)
    
    for target in df['Target'].unique():
        print(f"\n{target}:")
        target_data = df[df['Target'] == target]
        print(target_data[['Model', 'R2_Score', 'Accuracy', 'Precision', 'Recall', 'F1_Score']].to_string(index=False))
    
    # Model Ranking
    print("\n🏆 MODEL RANKING (by average performance):")
    print("-" * 50)
    
    model_averages = df.groupby('Model')[['R2_Score', 'Accuracy', 'F1_Score']].mean()
    model_averages['Combined_Score'] = (model_averages['R2_Score'] + model_averages['Accuracy'] + model_averages['F1_Score']) / 3
    model_ranking = model_averages.sort_values('Combined_Score', ascending=False)
    
    for i, (model, scores) in enumerate(model_ranking.iterrows(), 1):
        print(f"{i}. {model}: {scores['Combined_Score']:.4f}")
        print(f"   R² Score: {scores['R2_Score']:.4f} | Accuracy: {scores['Accuracy']:.4f} | F1 Score: {scores['F1_Score']:.4f}")
    
    # Best Model by Metric
    print("\n🎯 BEST MODELS BY METRIC:")
    print("-" * 50)
    
    metrics = ['R2_Score', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
    for metric in metrics:
        best_model = df.loc[df[metric].idxmax()]
        print(f"{metric}: {best_model['Model']} ({best_model[metric]:.4f}) on {best_model['Target']}")
    
    # Recommendations Comparison
    print("\n🍽️ SAMPLE RECOMMENDATIONS COMPARISON:")
    print("-" * 50)
    
    for model_name, recs in recommendations.items():
        print(f"\n{model_name} TOP 5 RECOMMENDATIONS:")
        if recs:
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['name']} ({rec['category']})")
                print(f"     Calories: {rec['calories']:.0f} | Protein: {rec['protein']:.1f}g | Health Score: {rec['health_score']:.3f}")
        else:
            print("  No recommendations available")
    
    # Conclusions
    print("\n💡 CONCLUSIONS:")
    print("-" * 50)
    
    best_overall = model_ranking.index[0]
    print(f"• Best Overall Model: {best_overall}")
    print(f"• Model Characteristics:")
    print("  - KNN: Good for similarity-based recommendations, interpretable")
    print("  - Random Forest: Robust, handles feature importance well, good generalization")
    print("  - MLP: Can capture complex non-linear patterns, good for complex health conditions")
    
    print(f"\n• Performance Insights:")
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        avg_acc = model_data['Accuracy'].mean()
        avg_f1 = model_data['F1_Score'].mean()
        print(f"  - {model}: Avg Accuracy = {avg_acc:.3f}, Avg F1 = {avg_f1:.3f}")
    
    print("\n" + "="*80)

def main():
    """Main function to run the comparison"""
    try:
        print("🚀 Starting Model Comparison Analysis...")
        
        # Load and prepare data
        food_data, features = load_and_prepare_data()
        
        # Train and evaluate models
        results = train_and_evaluate_models(food_data, features)
        
        # Generate recommendations
        recommendations = generate_recommendations_comparison(food_data, features)
        
        # Print results
        print_results(results, recommendations)
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()