import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import glob
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class HealthyFoodRecommender:
    """
    Personalized Food Recommendation System based on XGBoost
    that recommends food items based on user biometrics and
    nutritional composition without using disease labels.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.explainer = None
        self.foods_data = None
        self.scaler_user = StandardScaler()
        self.scaler_food = StandardScaler()
    
    def load_food_data(self, data_path):
        """
        Load food data from CSV files.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing food CSV files
        """
        print("Loading food data...")
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")
        
        # Load each file and add category field
        all_foods = []
        total_count = 0
        
        for file_path in csv_files:
            category = os.path.basename(file_path).replace(".csv", "")
            try:
                df = pd.read_csv(file_path)
                df['Category'] = category
                all_foods.append(df)
                total_count += len(df)
                print(f"  Loaded {len(df)} items from {category}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # Combine all data
        self.foods_data = pd.concat(all_foods, ignore_index=True)
        print(f"Successfully loaded {total_count} food items from {len(csv_files)} categories")
        
        # Display sample items from each category
        print("\nSample foods by category:")
        for category in self.foods_data['Category'].unique():
            sample = self.foods_data[self.foods_data['Category'] == category].sample(min(1, sum(self.foods_data['Category'] == category)))
            print(f"{category}: {sample['Thai_Name'].values[0]} ({sample['English_Name'].values[0]})")
        
        # Display column statistics
        print("\nNutritional data statistics:")
        numeric_cols = ['Energy(kcal) by calculation', 'Protein(g)', 'Fat(g)', 
                        'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 'FIBTG (g) Dietary fibre']
        
        for col in numeric_cols:
            if col in self.foods_data.columns:
                print(f"{col}: Mean={self.foods_data[col].mean():.1f}, Min={self.foods_data[col].min():.1f}, Max={self.foods_data[col].max():.1f}")
        
        return self.foods_data
    
    def engineer_user_features(self, user_data):
        """
        Engineer features from user biometric data.
        
        Parameters:
        -----------
        user_data : pd.DataFrame
            DataFrame containing user biometric information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered user features
        """
        features = user_data.copy()
        
        # Calculate BMI if not provided
        if 'BMI' not in features.columns:
            features['BMI'] = features['Weight'] / ((features['Height']/100) ** 2)
        
        # Create BMI categories
        features['BMI_Category'] = pd.cut(
            features['BMI'], 
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Waist-to-height ratio (health risk indicator)
        if 'Waist' in features.columns:
            features['Waist_Height_Ratio'] = features['Waist'] / features['Height']
        
        # Age groups
        features['Age_Group'] = pd.cut(
            features['Age'],
            bins=[0, 18, 35, 50, 65, 120],
            labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
        )
        
        # Blood pressure categories (if available)
        if 'Systolic' in features.columns and 'Diastolic' in features.columns:
            # Create blood pressure categories without using "hypertension" label
            conditions = [
                (features['Systolic'] < 120) & (features['Diastolic'] < 80),
                (features['Systolic'] < 130) & (features['Diastolic'] < 85),
                (features['Systolic'] < 140) & (features['Diastolic'] < 90),
                (features['Systolic'] >= 140) | (features['Diastolic'] >= 90)
            ]
            bp_categories = ['Optimal', 'Normal', 'High_Normal', 'Elevated']
            features['BP_Category'] = np.select(conditions, bp_categories, default='Unknown')
        
        # Glucose level categories (if available)
        if 'Glucose' in features.columns:
            features['Glucose_Category'] = pd.cut(
                features['Glucose'],
                bins=[0, 70, 100, 126, 500],
                labels=['Low', 'Normal', 'Elevated', 'High']
            )
        
        # One-hot encode categorical features
        categorical_cols = ['Gender', 'BMI_Category', 'Age_Group', 'BP_Category', 'Glucose_Category']
        for col in categorical_cols:
            if col in features.columns:
                dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
                features = pd.concat([features, dummies], axis=1)
        
        # Scale numerical features
        numeric_cols = [
            'Age', 'Height', 'Weight', 'BMI', 
            'Systolic', 'Diastolic', 'Glucose', 'Waist'
        ]
        
        # Only scale columns that exist
        cols_to_scale = [col for col in numeric_cols if col in features.columns]
        
        if cols_to_scale:
            # Only fit the scaler during training
            if not hasattr(self, 'user_features_fitted'):
                self.scaler_user.fit(features[cols_to_scale])
                self.user_features_fitted = True
            
            features[cols_to_scale] = self.scaler_user.transform(features[cols_to_scale])
        
        return features
    
    def engineer_food_features(self, food_data):
        """
        Engineer features from food nutritional data.
        
        Parameters:
        -----------
        food_data : pd.DataFrame
            DataFrame containing food nutritional information
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered food features
        """
        features = food_data.copy()
        
        # Remove suitability_score if present - this prevents the scaling error
        if 'suitability_score' in features.columns:
            features = features.drop(columns=['suitability_score'])
        
        # Handle missing values
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                features[col] = features[col].fillna(0)
        
        # Basic nutritional features
        basic_features = [
            'Energy(kcal) by calculation', 'Protein(g)', 'Fat(g)', 
            'CHOCDF (g) Carbohydrate', 'SUGAR(g)', 'FIBTG (g) Dietary fibre'
        ]
        
        # Additional nutritional features if available
        additional_features = [
            'FASAT (g) Saturated FA', 'Na(mg)', 'K(mg)', 'Ca(mg)', 'CHOLE(mg) Cholesterol'
        ]
        
        # Check which features exist
        existing_features = [f for f in basic_features + additional_features if f in features.columns]
        
        # Calculate macronutrient ratios
        if all(f in features.columns for f in ['Protein(g)', 'Energy(kcal) by calculation']):
            features['Protein_Calorie_Ratio'] = features['Protein(g)'] * 4 / features['Energy(kcal) by calculation'].replace(0, 1)
        
        if all(f in features.columns for f in ['Fat(g)', 'Energy(kcal) by calculation']):
            features['Fat_Calorie_Ratio'] = features['Fat(g)'] * 9 / features['Energy(kcal) by calculation'].replace(0, 1)
        
        if all(f in features.columns for f in ['CHOCDF (g) Carbohydrate', 'Energy(kcal) by calculation']):
            features['Carbs_Calorie_Ratio'] = features['CHOCDF (g) Carbohydrate'] * 4 / features['Energy(kcal) by calculation'].replace(0, 1)
        
        # Sugar and fiber ratios
        if all(f in features.columns for f in ['SUGAR(g)', 'CHOCDF (g) Carbohydrate']):
            features['Sugar_Carb_Ratio'] = features['SUGAR(g)'] / features['CHOCDF (g) Carbohydrate'].replace(0, 1)
        
        if all(f in features.columns for f in ['FIBTG (g) Dietary fibre', 'CHOCDF (g) Carbohydrate']):
            features['Fiber_Carb_Ratio'] = features['FIBTG (g) Dietary fibre'] / features['CHOCDF (g) Carbohydrate'].replace(0, 1)
        
        # Energy density features
        if all(f in features.columns for f in ['Protein(g)', 'Energy(kcal) by calculation']):
            features['Protein_Density'] = features['Protein(g)'] / features['Energy(kcal) by calculation'].replace(0, 1) * 1000
        
        if all(f in features.columns for f in ['FIBTG (g) Dietary fibre', 'Energy(kcal) by calculation']):
            features['Fiber_Density'] = features['FIBTG (g) Dietary fibre'] / features['Energy(kcal) by calculation'].replace(0, 1) * 1000
        
        # Advanced nutritional balance metrics
        if all(f in features.columns for f in ['Energy(kcal) by calculation', 'Protein(g)', 'FIBTG (g) Dietary fibre', 'SUGAR(g)']):
            # Create a simple nutritional quality score (higher is better)
            # Prioritizes protein and fiber, penalizes empty calories
            protein_score = features['Protein(g)'] / features['Energy(kcal) by calculation'].replace(0, 1) * 1000
            fiber_score = features['FIBTG (g) Dietary fibre'] / features['Energy(kcal) by calculation'].replace(0, 1) * 1000
            sugar_penalty = features['SUGAR(g)'] / features['Energy(kcal) by calculation'].replace(0, 1) * 500
            
            features['Nutritional_Quality_Score'] = protein_score + fiber_score - sugar_penalty
        
        # Handle saturated fat if available
        if all(f in features.columns for f in ['FASAT (g) Saturated FA', 'Fat(g)']):
            features['Saturated_Fat_Ratio'] = features['FASAT (g) Saturated FA'] / features['Fat(g)'].replace(0, 1)
        
        # Handle sodium and potassium if available
        if all(f in features.columns for f in ['Na(mg)', 'K(mg)']):
            features['Na_K_Ratio'] = features['Na(mg)'] / features['K(mg)'].replace(0, 1)
        
        # One-hot encode category
        if 'Category' in features.columns:
            dummies = pd.get_dummies(features['Category'], prefix='Category')
            features = pd.concat([features, dummies], axis=1)
        
        # Clean up infinite values and NaNs
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                features[col] = features[col].replace([np.inf, -np.inf], np.nan)
                features[col] = features[col].fillna(0)
        
    # Scale numerical features
        numeric_cols = [col for col in features.columns if 
                    features[col].dtype in [np.float64, np.float32] and 
                    col not in ['Food_Code'] and
                    not col.startswith('Category_')]
        
        if numeric_cols:
            # Only fit the scaler during training
            if not hasattr(self, 'food_features_fitted'):
                self.scaler_food.fit(features[numeric_cols])
                self.food_features_fitted = True
                # Store the column names seen during fit
                self.food_numeric_cols_at_fit = numeric_cols
            else:
                # Only transform columns that were present during fit
                common_cols = [col for col in numeric_cols if col in self.food_numeric_cols_at_fit]
                if common_cols:
                    features[common_cols] = self.scaler_food.transform(features[common_cols])
        
        return features
    
    def calculate_suitability_score(self, user, food):
        """
        Calculate a suitability score based on nutritional guidelines
        without explicitly using disease labels.
        
        Parameters:
        -----------
        user : pd.Series
            Series containing user biometric data
        food : pd.Series
            Series containing food nutritional data
        
        Returns:
        --------
        float
            Suitability score between 0 and 10
        """
        score = 5.0  # Base score (neutral)
        
        # Extract user metrics
        bmi = user['BMI'] if 'BMI' in user else user['Weight'] / ((user['Height']/100) ** 2)
        age = user['Age']
        gender = user['Gender']
        
        # Optional metrics
        systolic = user.get('Systolic', None)
        diastolic = user.get('Diastolic', None)
        glucose = user.get('Glucose', None)
        waist = user.get('Waist', None)
        
        # Extract food metrics
        energy = food['Energy(kcal) by calculation']
        protein = food['Protein(g)']
        fat = food['Fat(g)']
        carbs = food['CHOCDF (g) Carbohydrate']
        sugar = food['SUGAR(g)']
        fiber = food.get('FIBTG (g) Dietary fibre', 0)
        
        # Optional food metrics
        sodium = food.get('Na(mg)', None)
        sat_fat = food.get('FASAT (g) Saturated FA', None)
        cholesterol = food.get('CHOLE(mg) Cholesterol', None)
        
        # 1. BMI-based energy considerations
        if bmi > 30:  # Obesity range
            if energy > 500:
                score -= 1.5
            elif energy < 200:
                score += 0.5
        elif bmi > 25:  # Overweight range
            if energy > 400:
                score -= 1.0
            elif energy < 250:
                score += 0.5
        elif bmi < 18.5:  # Underweight range
            if energy > 500:
                score += 0.5
            elif energy < 200:
                score -= 0.5
        
        # 2. Blood pressure considerations (without mentioning hypertension)
        if systolic is not None and diastolic is not None:
            bp_elevated = (systolic >= 130 or diastolic >= 85)
            
            if sodium is not None:
                if bp_elevated and sodium > 400:
                    score -= 1.5
                elif sodium > 600:
                    score -= 1.0
                elif sodium < 140:
                    score += 0.5
        
        # 3. Glucose considerations (without mentioning diabetes)
        if glucose is not None:
            glucose_elevated = glucose > 100
            
            if glucose_elevated:
                if sugar > 15:
                    score -= 1.5
                elif sugar > 10:
                    score -= 1.0
                
                # Consider fiber-to-carb ratio for glycemic response
                if carbs > 0:
                    fiber_carb_ratio = fiber / carbs
                    if fiber_carb_ratio > 0.1:
                        score += 0.5
        
        # 4. General nutritional quality adjustments
        
        # Protein quality
        if protein > 15:
            score += 0.5
        
        # Fiber content is generally good
        if fiber > 5:
            score += 1.0
        elif fiber > 3:
            score += 0.5
        
        # Fat quality
        if sat_fat is not None:
            if fat > 0:
                sat_fat_ratio = sat_fat / fat
                if sat_fat_ratio > 0.4:
                    score -= 0.5
        elif fat > 20:
            score -= 0.5
        
        # Cholesterol considerations
        if cholesterol is not None:
            if cholesterol > 100:
                score -= 0.5
        
        # 5. Age and gender specific considerations
        if age > 60:
            # Older adults often need more protein and calcium
            if protein > 20:
                score += 0.5
                
            # May need to monitor sodium more carefully
            if sodium is not None and sodium > 300:
                score -= 0.5
        
        # Ensure score stays within reasonable range
        return max(0, min(score + 5, 10))  # Scale to 0-10
    
    def create_training_data(self, users, foods=None):
        """
        Create training data with user-food pairs and suitability scores.
        
        Parameters:
        -----------
        users : pd.DataFrame
            DataFrame containing user biometric data
        foods : pd.DataFrame, optional
            DataFrame containing food data (uses self.foods_data if not provided)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with user-food pairs and suitability scores
        """
        if foods is None:
            if self.foods_data is None:
                raise ValueError("No food data available. Please load food data first.")
            foods = self.foods_data
        
        print(f"Creating training data with {len(users)} users and {len(foods)} foods...")
        
        # For large datasets, sample foods for each user to avoid memory issues
        sample_size = min(200, len(foods))
        
        rows = []
        for user_id, user in users.iterrows():
            # For each user, take a stratified sample of foods by category
            sampled_foods = foods.groupby('Category').apply(
                lambda x: x.sample(min(max(1, sample_size // len(foods['Category'].unique())), len(x)), random_state=42)
            ).reset_index(drop=True)
            
            print(f"  Processing user {user_id+1}/{len(users)} with {len(sampled_foods)} food items")
            
            # Engineer user features once per user
            user_features = self.engineer_user_features(pd.DataFrame([user]))
            
            for _, food in sampled_foods.iterrows():
                # Create a row combining user and food features
                row = {'user_id': user_id, 'food_id': food.name}
                
                # Add basic identifiers
                row['food_name'] = food['Thai_Name'] if 'Thai_Name' in food else food['English_Name']
                row['food_category'] = food['Category']
                
                # Calculate suitability score based on health heuristics
                suitability_score = self.calculate_suitability_score(user, food)
                row['suitability_score'] = suitability_score
                
                # Add engineered features later in the prepare_model_features function
                rows.append(row)
        
        training_data = pd.DataFrame(rows)
        print(f"Created {len(training_data)} training samples")
        
        return training_data
    
    def prepare_model_features(self, training_data, users, foods=None):
        """
        Prepare features for the model by combining user and food features.
        
        Parameters:
        -----------
        training_data : pd.DataFrame
            DataFrame with user-food pairs and suitability scores
        users : pd.DataFrame
            DataFrame containing user biometric data
        foods : pd.DataFrame, optional
            DataFrame containing food data (uses self.foods_data if not provided)
        
        Returns:
        --------
        tuple
            (X, y) tuple with features and target variables
        """
        if foods is None:
            if self.foods_data is None:
                raise ValueError("No food data available. Please load food data first.")
            foods = self.foods_data
            
        print("Preparing model features...")
        
        X_data = []
        y = training_data['suitability_score'].values
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        batches = (len(training_data) + batch_size - 1) // batch_size
        
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(training_data))
            batch = training_data.iloc[start_idx:end_idx]
            
            print(f"  Processing batch {i+1}/{batches} ({start_idx}-{end_idx})")
            
            batch_features = []
            for _, row in batch.iterrows():
                user_id = row['user_id']
                food_id = row['food_id']
                
                # Get user and food data
                user = users.iloc[user_id]
                food = foods.iloc[food_id] if isinstance(food_id, int) else foods.loc[food_id]
                
                # Engineer features
                user_features = self.engineer_user_features(pd.DataFrame([user]))
                food_features = self.engineer_food_features(pd.DataFrame([food]))
                
                # Create a feature dictionary
                features = {}
                
                # Add user features
                for col in user_features:
                    if col not in ['user_id', 'BMI_Category', 'Age_Group', 'BP_Category', 'Glucose_Category', 'Gender']:
                        if isinstance(user_features[col].iloc[0], (np.bool_, bool)):
                            features[f'user_{col}'] = int(user_features[col].iloc[0])
                        else:
                            features[f'user_{col}'] = user_features[col].iloc[0]
                
                # Add food features
                for col in food_features:
                    if col not in ['Food_Code', 'Thai_Name', 'English_Name', 'Category']:
                        if isinstance(food_features[col].iloc[0], (np.bool_, bool)):
                            features[f'food_{col}'] = int(food_features[col].iloc[0])
                        else:
                            features[f'food_{col}'] = food_features[col].iloc[0]
                
                # Add category-user interaction features
                if 'Category' in food:
                    category = food['Category']
                    # Create interaction features between user metrics and food category
                    if 'BMI' in user:
                        features[f'interaction_BMI_{category}'] = user['BMI']
                    if 'Age' in user:
                        features[f'interaction_Age_{category}'] = user['Age']
                
                batch_features.append(features)
            
            # Convert batch to DataFrame
            batch_df = pd.DataFrame(batch_features)
            X_data.append(batch_df)
        
        # Combine all batches
        X = pd.concat(X_data, ignore_index=True)
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype not in [np.float64, np.float32, np.int64, np.int32, bool]:
                try:
                    X[col] = X[col].astype(float)
                except:
                    print(f"Warning: Dropping non-numeric column {col}")
                    X = X.drop(columns=[col])
        
        # Fill NaN values
        X = X.fillna(0)
        
        print(f"Final feature matrix shape: {X.shape}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train an XGBoost model for personalized food recommendations.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : array-like
            Target values (suitability scores)
        test_size : float, optional
            Proportion of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary with training results
        """
        print("Training XGBoost model...")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 200,
            'random_state': random_state
        }
        
        # Create and train the model - simple version compatible with all XGBoost versions
        self.model = xgb.XGBRegressor(**params)
        
        print("Training model...")
        # Basic fit call without eval_set or early_stopping_rounds
        self.model.fit(X_train, y_train)
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Rank correlation (Spearman's rho)
        train_rank_corr, _ = spearmanr(y_train, y_pred_train)
        test_rank_corr, _ = spearmanr(y_test, y_pred_test)
        
        print(f"\nModel Performance:")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f}")
        print(f"  Train Rank Correlation: {train_rank_corr:.4f}")
        print(f"  Test Rank Correlation: {test_rank_corr:.4f}")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(self.model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        print(f"  Cross-validation RMSE: {cv_rmse:.4f} (±{cv_scores.std():.4f})")
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 most important features:")
        print(importance_df.head(15))
        
        # Create SHAP explainer
        print("\nCreating SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Sample data for SHAP values
        shap_sample = X_test.iloc[:min(100, len(X_test))]
        shap_values = self.explainer.shap_values(shap_sample)
        
        # Return results
        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rank_corr': train_rank_corr,
            'test_rank_corr': test_rank_corr,
            'cv_rmse': cv_rmse,
            'importance_df': importance_df,
            'shap_values': shap_values,
            'shap_sample': shap_sample
        }
    
    def plot_feature_importance(self, importance_df=None, top_n=15):
        """
        Plot feature importance from the trained model.
        
        Parameters:
        -----------
        importance_df : pd.DataFrame, optional
            DataFrame with feature importance (uses model's importance if not provided)
        top_n : int, optional
            Number of top features to display
        """
        if importance_df is None:
            if self.model is None:
                raise ValueError("No trained model available")
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def plot_shap_summary(self, shap_values=None, X_sample=None):
        """
        Plot SHAP summary to explain model predictions.
        
        Parameters:
        -----------
        shap_values : array, optional
            Precomputed SHAP values
        X_sample : pd.DataFrame, optional
            Sample data for SHAP values
        """
        if shap_values is None or X_sample is None:
            if self.explainer is None:
                raise ValueError("No SHAP explainer available. Train model first.")
            
            # Sample data for visualization
            if hasattr(self, 'X_test'):
                X_sample = self.X_test.iloc[:min(100, len(self.X_test))]
            else:
                raise ValueError("No test data available for SHAP visualization")
                
            shap_values = self.explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns.tolist())
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()
    
    def get_recommendations(self, user, top_k=10, foods=None):
        """
        Get personalized food recommendations for a user.
        
        Parameters:
        -----------
        user : pd.Series or dict
            User biometric data
        top_k : int, optional
            Number of recommendations to return
        foods : pd.DataFrame, optional
            Food data to choose from (uses self.foods_data if not provided)
        
        Returns:
        --------
        pd.DataFrame
            Top K recommended foods with scores
        """
        if self.model is None:
            raise ValueError("No trained model available. Train the model first.")
        
        if foods is None:
            if self.foods_data is None:
                raise ValueError("No food data available. Please load food data first.")
            foods = self.foods_data
        
        # Convert user to Series if it's a dict
        if isinstance(user, dict):
            user = pd.Series(user)
        
        # Prepare features for all food items
        print(f"Generating recommendations for user from {len(foods)} food items...")
        
        # Process in batches to avoid memory issues
        batch_size = 500
        batches = (len(foods) + batch_size - 1) // batch_size
        
        all_predictions = []
        
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(foods))
            food_batch = foods.iloc[start_idx:end_idx]
            
            print(f"  Processing batch {i+1}/{batches} ({start_idx}-{end_idx})")
            
            # Engineer user features (once for all foods)
            user_features = self.engineer_user_features(pd.DataFrame([user]))
            
            # Prepare features for each food in the batch
            batch_features = []
            for _, food in food_batch.iterrows():
                # Engineer food features
                food_features = self.engineer_food_features(pd.DataFrame([food]))
                
                # Create a feature dictionary
                features = {}
                
                # Add user features
                for col in user_features:
                    if col not in ['user_id', 'BMI_Category', 'Age_Group', 'BP_Category', 'Glucose_Category', 'Gender']:
                        if isinstance(user_features[col].iloc[0], (np.bool_, bool)):
                            features[f'user_{col}'] = int(user_features[col].iloc[0])
                        else:
                            features[f'user_{col}'] = user_features[col].iloc[0]
                
                # Add food features
                for col in food_features:
                    if col not in ['Food_Code', 'Thai_Name', 'English_Name', 'Category']:
                        if isinstance(food_features[col].iloc[0], (np.bool_, bool)):
                            features[f'food_{col}'] = int(food_features[col].iloc[0])
                        else:
                            features[f'food_{col}'] = food_features[col].iloc[0]
                
                # Add category-user interaction features
                if 'Category' in food:
                    category = food['Category']
                    # Create interaction features between user metrics and food category
                    if 'BMI' in user:
                        features[f'interaction_BMI_{category}'] = user['BMI']
                    if 'Age' in user:
                        features[f'interaction_Age_{category}'] = user['Age']
                
                batch_features.append(features)
            
            # Convert batch to DataFrame
            X_batch = pd.DataFrame(batch_features)
            
            # Ensure all model features are present (fill with 0 if missing)
            for feature in self.feature_names:
                if feature not in X_batch.columns:
                    X_batch[feature] = 0
            
            # Make predictions for this batch
            X_batch = X_batch[self.feature_names]  # Ensure columns match model features
            predictions = self.model.predict(X_batch)
            
            # Store predictions with food indices
            for j, score in enumerate(predictions):
                idx = start_idx + j
                all_predictions.append((idx, score))
        
        # Sort predictions by score (descending)
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K recommendations
        top_indices = [idx for idx, _ in all_predictions[:top_k]]
        recommended_foods = foods.iloc[top_indices].copy()
        
        # Add prediction scores
        recommended_foods['suitability_score'] = [score for _, score in all_predictions[:top_k]]
        
        return recommended_foods
    
    def explain_recommendation(self, user, food):
        """
        Explain why a specific food was recommended to a user.
        
        Parameters:
        -----------
        user : pd.Series or dict
            User biometric data
        food : pd.Series or dict
            Food nutritional data
        
        Returns:
        --------
        dict
            Explanation of the recommendation
        """
        if self.model is None or self.explainer is None:
            raise ValueError("No trained model or explainer available")
        
        # Convert user and food to Series if they are dicts
        if isinstance(user, dict):
            user = pd.Series(user)
        if isinstance(food, dict):
            food = pd.Series(food)
        
        # Make a copy of food data and remove suitability_score if present
        food_copy = food.copy()
        if 'suitability_score' in food_copy:
            food_score = food_copy['suitability_score']  # Save for later
            food_copy = food_copy.drop('suitability_score')
        
        # Engineer features
        user_features = self.engineer_user_features(pd.DataFrame([user]))
        food_features = self.engineer_food_features(pd.DataFrame([food_copy]))
        
        # Create a feature dictionary
        features = {}
        
        # Add user features
        for col in user_features:
            if col not in ['user_id', 'BMI_Category', 'Age_Group', 'BP_Category', 'Glucose_Category', 'Gender']:
                if isinstance(user_features[col].iloc[0], (np.bool_, bool)):
                    features[f'user_{col}'] = int(user_features[col].iloc[0])
                else:
                    features[f'user_{col}'] = user_features[col].iloc[0]
        
        # Add food features
        for col in food_features:
            if col not in ['Food_Code', 'Thai_Name', 'English_Name', 'Category']:
                if isinstance(food_features[col].iloc[0], (np.bool_, bool)):
                    features[f'food_{col}'] = int(food_features[col].iloc[0])
                else:
                    features[f'food_{col}'] = food_features[col].iloc[0]
        
        # Add category-user interaction features
        if 'Category' in food:
            category = food['Category']
            # Create interaction features between user metrics and food category
            if 'BMI' in user:
                features[f'interaction_BMI_{category}'] = user['BMI']
            if 'Age' in user:
                features[f'interaction_Age_{category}'] = user['Age']
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure all model features are present
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Select only the features used by the model
        X = X[self.feature_names]
        
        # Get prediction
        prediction = self.model.predict(X)[0]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get top contributing factors
        feature_impacts = list(zip(self.feature_names, shap_values[0]))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Create explanation
        explanation = {
            'food_name': food.get('Thai_Name', food.get('English_Name', 'Unknown')),
            'suitability_score': prediction,
            'top_positive_factors': [],
            'top_negative_factors': []
        }
        
        # Add top 5 positive and negative factors
        for feature, impact in feature_impacts[:10]:
            # Create human-readable feature name
            readable_name = feature.replace('user_', '').replace('food_', '').replace('_', ' ').title()
            
            if impact > 0:
                explanation['top_positive_factors'].append({
                    'factor': readable_name,
                    'impact': float(impact)
                })
            else:
                explanation['top_negative_factors'].append({
                    'factor': readable_name,
                    'impact': float(impact)
                })
        
        # Sort by impact magnitude
        explanation['top_positive_factors'] = sorted(
            explanation['top_positive_factors'], 
            key=lambda x: x['impact'], 
            reverse=True
        )[:5]
        
        explanation['top_negative_factors'] = sorted(
            explanation['top_negative_factors'], 
            key=lambda x: x['impact'], 
            reverse=False
        )[:5]
        
        return explanation
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model available to save")
        
        # Create a dictionary with model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'scaler_user': self.scaler_user,
            'scaler_food': self.scaler_food,
            'user_features_fitted': hasattr(self, 'user_features_fitted'),
            'food_features_fitted': hasattr(self, 'food_features_fitted')
        }
        
        # Save model
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.scaler_user = model_data['scaler_user']
        self.scaler_food = model_data['scaler_food']
        
        if model_data['user_features_fitted']:
            self.user_features_fitted = True
        
        if model_data['food_features_fitted']:
            self.food_features_fitted = True
        
        # Create explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"Model loaded from {filepath}")


def demo():
    """Run a demonstration of the FoodRecommender."""
    # Create the recommender
    recommender = HealthyFoodRecommender()
    
    # Load food data
    print("=== Loading Food Data ===")
    foods = recommender.load_food_data('./datasets')
    
    # Create synthetic user data for demonstration
    print("\n=== Creating Synthetic User Data ===")
    users = pd.DataFrame([
        {
            'Age': 35, 'Gender': 'Male', 'Height': 175, 'Weight': 80,
            'BMI': 26.1, 'Systolic': 130, 'Diastolic': 85, 'Glucose': 105,
            'Waist': 90
        },
        {
            'Age': 65, 'Gender': 'Female', 'Height': 160, 'Weight': 60,
            'BMI': 23.4, 'Systolic': 150, 'Diastolic': 90, 'Glucose': 120,
            'Waist': 85
        },
        {
            'Age': 25, 'Gender': 'Male', 'Height': 180, 'Weight': 70,
            'BMI': 21.6, 'Systolic': 120, 'Diastolic': 75, 'Glucose': 85,
            'Waist': 80
        },
        {
            'Age': 45, 'Gender': 'Female', 'Height': 165, 'Weight': 90,
            'BMI': 33.1, 'Systolic': 140, 'Diastolic': 95, 'Glucose': 110,
            'Waist': 100
        }
    ])
    
    print(users)
    
    # Create training data
    print("\n=== Creating Training Data ===")
    training_data = recommender.create_training_data(users, foods)
    
    # Prepare features for model
    print("\n=== Preparing Model Features ===")
    X, y = recommender.prepare_model_features(training_data, users, foods)
    
    # Train the model
    print("\n=== Training XGBoost Model ===")
    results = recommender.train(X, y)
    
    # Plot feature importance
    print("\n=== Feature Importance Plot ===")
    recommender.plot_feature_importance(results['importance_df'])
    
    # Plot SHAP summary
    print("\n=== SHAP Explanation Plot ===")
    recommender.plot_shap_summary(results['shap_values'], results['shap_sample'])
    
    # Get recommendations for a user
    print("\n=== Sample Recommendations ===")
    test_user = {
        'Age': 50, 'Gender': 'Male', 'Height': 170, 'Weight': 85,
        'BMI': 29.4, 'Systolic': 135, 'Diastolic': 88, 'Glucose': 115,
        'Waist': 95
    }
    
    recommendations = recommender.get_recommendations(test_user, top_k=5)
    
    print("Top 5 recommended foods:")
    for idx, row in recommendations.iterrows():
        print(f"{row['Thai_Name']} ({row['English_Name']}) - Score: {row['suitability_score']:.2f}")
    
    # Explain a recommendation
    print("\n=== Recommendation Explanation ===")
    explanation = recommender.explain_recommendation(test_user, recommendations.iloc[0])
    
    print(f"Explanation for {explanation['food_name']} (Score: {explanation['suitability_score']:.2f}):")
    
    print("\nPositive factors:")
    for factor in explanation['top_positive_factors']:
        print(f"  • {factor['factor']}: +{factor['impact']:.4f}")
    
    print("\nNegative factors:")
    for factor in explanation['top_negative_factors']:
        print(f"  • {factor['factor']}: {factor['impact']:.4f}")
    
    # Save the model
    print("\n=== Saving Model ===")
    recommender.save_model('food_recommender_model.pkl')
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    demo()