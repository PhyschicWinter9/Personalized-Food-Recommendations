# Health-Driven Food Recommendation System Using KNN

I need help implementing a **health-driven K-Nearest Neighbors (KNN) model** for a personalized food recommendation system that automatically calculates nutritional needs based on medical guidelines rather than user guesses. This system will analyze Thai food datasets to provide scientifically-based recommendations using established medical formulas and health condition modifications.

**Key Improvement:** The system should automatically calculate personalized nutritional targets using BMR/TDEE formulas and health condition guidelines, then use KNN to find matching foods - eliminating the confusing "manual target setting" approach.

## Nutritional Guidelines Analysis

Please analyze these nutritional guideline files to establish medical-grade calculation rules:

1. `NCD_Nutritional_Guidelines_ChatGPT.md`
2. `NCD_Nutritional_Guidelines_Claude.md`  
3. `NCD_Nutritional_Guidelines_Gemini.md`
4. `NCD_Nutritional_Guidelines_Perplexity.md`

## Implementation Requirements

### 1. Medical-Grade Nutritional Calculation Engine

- BMR calculation using Mifflin-St Jeor equation (age, gender, weight, height)
- TDEE calculation with activity level multipliers
- Health condition modifications (diabetes → reduced sugar/carbs, hypertension → reduced sodium, etc.)
- Weight goal adjustments (deficit for weight loss, surplus for weight gain)
- Age and gender-specific adjustments based on DRI guidelines

### 2. Thai Food Dataset Preparation for Health-Driven KNN

- Processing datasets: `processed_food.csv`, `Fruit.csv`, `Drinking.csv`, `esan_food.csv`, `curry.csv`, `meat.csv`, `vegetables.csv`, `cracker.csv`, `dessert.csv`, `noodle.csv`, `onedish.csv`
- Feature engineering focused on health condition suitability scores
- Nutritional density calculations and portion size standardization
- Missing value handling for critical nutrients (Na, K, saturated fat, cholesterol)

### 3. Health-Condition-Aware KNN Implementation

- **Input:** User profile (age, gender, weight, height, activity, health conditions) → **Output:** Calculated nutritional targets → KNN matching
- Custom distance metrics that prioritize health-critical nutrients (sugar for diabetes, sodium for hypertension)
- Feature scaling that preserves medical significance of nutritional ratios
- Health suitability scoring algorithm based on established medical guidelines
- Multi-objective optimization: nutritional similarity + health condition compliance

### 4. Advanced KNN Optimization

- Cross-validation for optimal k values considering health condition clustering
- Weighted KNN with health condition penalties/bonuses
- Dimensionality reduction techniques that preserve medical interpretability
- Efficient user profile vector creation from calculated (not manually set) targets

### 5. Clinical Validation Approach

- Evaluation metrics: precision@k for health condition suitability
- Comparison with manual target-setting baseline
- Nutritional adequacy validation against DRI guidelines
- User satisfaction scoring for recommendation explanations

### 6. Complete Implementation Requirements

- **GUI Version:** Modern interface showing calculated targets with medical explanations
- **API Version:** RESTful service for integration with web/mobile applications
- **Explanation System:** Clear rationale for why specific foods are recommended
- **Real-time Updates:** Recommendations change when health profile changes

## Key Technical Innovation

The system should eliminate user confusion by automatically calculating what they need (based on medical science) rather than asking them to guess their nutritional targets. The KNN algorithm should then find foods that match these scientifically-calculated needs.

## Research Context

This is for my Master's thesis at **Prince of Songkla University, Surat Thani Campus**, focusing on *"Health and Biometric Data-Driven Food Recommendation System"* for Thai populations with NCDs. The system must be medically accurate, culturally appropriate, and eliminate the common UX problem of requiring users to set their own nutritional targets.

## Expected Deliverables

1. Complete health-driven recommendation engine with medical calculations
2. KNN implementation optimized for nutritional similarity and health compliance
3. Modern GUI showing the calculation process and explanations
4. API version for broader application integration
5. Evaluation framework comparing against manual target-setting approaches

## Project Goal

The goal is to create a system that works like a **digital nutritionist** - calculating what you need based on your health profile, then finding the best Thai foods to meet those needs.

---

*The details are in the Research Proposal - En - Edit_Gramma file.*


Raw text

Key Technical Innovation: The system should eliminate user confusion by automatically calculating what they need (based on medical science) rather than asking them to guess their nutritional targets. The KNN algorithm should then find foods that match these scientifically-calculated needs.

Research Context: This is for my Master's thesis at Prince of Songkla University, Surat Thani Campus, focusing on "Health and Biometric Data-Driven Food Recommendation System" for Thai populations with NCDs. The system must be medically accurate, culturally appropriate, and eliminate the common UX problem of requiring users to set their own nutritional targets.

Expected Deliverables:
1. Complete health-driven recommendation engine with medical calculations
2. KNN implementation optimized for nutritional similarity and health compliance
3. Modern GUI showing the calculation process and explanations
4. API version for broader application integration
5. Evaluation framework comparing against manual target-setting approaches

The goal is to create a system that works like a digital nutritionist - calculating what you need based on your health profile, then finding the best Thai foods to meet those needs.

The details are in the Research Proposal - En - Edit_Gramma file.