import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox, font
import time
import threading
import os
import glob
from tkinter import PhotoImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FoodRecommendationSystem:
    def __init__(self, status_callback=None):
        # Callback function to update loading status
        self.status_callback = status_callback
        
        # Load datasets - modify paths as needed
        self.load_data()
        
        # Features to use for K-NN - Updated to match your actual column names
        self.features = ['Energy(kcal) by calculation', 'Protein(g)', 'CHOCDF (g) Carbohydrate']
        
        # Prepare the data for K-NN
        self.prepare_knn_data()
        
    def update_status(self, message):
        """Update loading status if callback is provided"""
        if self.status_callback:
            self.status_callback(message)
        print(message)  # Also print to console for debugging
        
    def load_data(self):
        """Load and combine all food datasets from CSV files in datasets folder"""
        try:
            # Path to the datasets folder - matching your directory structure
            dataset_folder = './datasets'
            
            # Get all CSV files in the datasets folder
            csv_files = glob.glob(os.path.join(dataset_folder, '*.csv'))
            
            if not csv_files:
                self.update_status("No CSV files found in the datasets folder!")
                self.food_data = pd.DataFrame()
                return
                
            # List to store individual dataframes
            dataframes = []
            
            # Load each CSV file
            for file_path in csv_files:
                try:
                    # Get filename without extension to use as category
                    filename = os.path.basename(file_path)
                    category = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                    
                    self.update_status(f"Loading {category} data...")
                    
                    # Load CSV file
                    df = pd.read_csv(file_path)
                    
                    # Add category column if it doesn't exist
                    if 'Category' not in df.columns:
                        df['Category'] = category
                        
                    dataframes.append(df)
                    self.update_status(f"Loaded {len(df)} items from {filename}")
                    
                except Exception as e:
                    self.update_status(f"Error loading {file_path}: {e}")
            
            # Combine all dataframes
            if dataframes:
                self.update_status("Combining all food data...")
                self.food_data = pd.concat(dataframes, ignore_index=True)
                self.update_status(f"Successfully loaded {len(self.food_data)} food items")
                
                # Print column names for debugging
                self.update_status(f"Columns available: {', '.join(self.food_data.columns)}")
            else:
                self.update_status("No valid data files could be loaded")
                self.food_data = pd.DataFrame()
                
        except Exception as e:
            self.update_status(f"Error loading data: {e}")
            # Create empty DataFrame if loading fails
            self.food_data = pd.DataFrame()
    
    def prepare_knn_data(self):
        """Prepare data for K-NN algorithm"""
        if len(self.food_data) == 0:
            return
            
        # Check if all features exist in the dataset
        missing_features = [f for f in self.features if f not in self.food_data.columns]
        if missing_features:
            print(f"Warning: Missing features in dataset: {missing_features}")
            # Use only available features
            self.features = [f for f in self.features if f in self.food_data.columns]
            
        if not self.features:
            print("Error: No valid features available for recommendation")
            return
            
        # Extract features for K-NN
        X = self.food_data[self.features].fillna(0)
        
        # Standardize features (important for K-NN)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Initialize K-NN model (using 5 neighbors)
        self.knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.knn_model.fit(self.X_scaled)
    
    def get_recommendations(self, user_preferences):
        """Get food recommendations based on user preferences"""
        if not hasattr(self, 'knn_model') or len(self.food_data) == 0:
            return []
            
        # Extract user preferences for our features
        user_values = [user_preferences.get(feature, 0) for feature in self.features]
        
        # Scale user preferences
        user_scaled = self.scaler.transform([user_values])
        
        # Find K-nearest neighbors
        distances, indices = self.knn_model.kneighbors(user_scaled)
        
        # Get recommended food items
        recommendations = []
        for idx in indices[0]:
            food_item = self.food_data.iloc[idx]
            recommendations.append({
                'Name': food_item.get('Thai_Name', food_item.get('English_Name', f"Food {idx}")),
                'Category': food_item.get('Category', 'Unknown'),
                'Energy': food_item.get('Energy(kcal) by calculation', 0),
                'Protein': food_item.get('Protein(g)', 0),
                'Carbs': food_item.get('CHOCDF (g) Carbohydrate', 0),
                'Distance': distances[0][list(indices[0]).index(idx)]
            })
            
        return recommendations

# User Interface Class with Modern UI/UX
class FoodRecommenderUI:
    def __init__(self, master, recommender=None):
        self.master = master
        self.master.title("Food Recommendation System for Diabetic Patients")
        self.master.geometry("1000x700")
        self.master.configure(bg="#f0f0f0")
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'))
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        self.style.configure('Subheader.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        self.style.configure('Stats.TLabel', font=('Arial', 9), background='#f5f5f5')
        
        # Configure Treeview
        self.style.configure('Treeview', font=('Arial', 9))
        self.style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        
        # Initialize the recommendation system if not provided
        self.recommender = recommender or FoodRecommendationSystem()
        
        # Create UI elements
        self.create_main_layout()
        
        # Nutritional target values for diabetics (defaults)
        self.targets = {
            'Energy(kcal) by calculation': 500,
            'Protein(g)': 20,
            'CHOCDF (g) Carbohydrate': 30,
            'SUGAR(g)': 5,
            'FIBTG (g) Dietary fibre': 8,
            'Fat(g)': 15
        }
        
        # Update stats display
        self.update_stats_display()
        
    def create_main_layout(self):
        """Create the main layout with modern UI"""
        # Main container
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(header_frame, text="Personalized Food Recommendation System", 
                 style='Header.TLabel').pack(side=tk.LEFT)
        
        # Create two panels side by side
        panel_container = ttk.Frame(main_container)
        panel_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input controls
        left_panel = ttk.Frame(panel_container, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        self.create_input_panel(left_panel)
        
        # Right panel - Results
        right_panel = ttk.Frame(panel_container, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_results_panel(right_panel)
        
        # Nutrition chart panel
        chart_panel = ttk.Frame(main_container, padding=10)
        chart_panel.pack(fill=tk.X, pady=(15, 0))
        
        self.create_chart_panel(chart_panel)
        
        # Create status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create stats display at the bottom right
        self.stats_frame = ttk.Frame(self.status_bar, style='Stats.TLabel')
        self.stats_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Create progress bar at the bottom right
        self.progress_frame = ttk.Frame(self.status_bar)
        self.progress_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="determinate")
        self.progress.pack(side=tk.RIGHT)
        self.progress["value"] = 0
        
    def create_input_panel(self, parent):
        """Create the input panel with nutrition preferences"""
        # Title
        ttk.Label(parent, text="Nutrition Preferences", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Frame for nutritional inputs
        input_frame = ttk.LabelFrame(parent, text="Target Nutritional Values")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Energy/Calories preference
        self.create_slider(input_frame, "Energy (kcal):", "calories_var", 500, 0, 1000, 0)
        
        # Protein preference
        self.create_slider(input_frame, "Protein (g):", "protein_var", 20, 0, 50, 1)
        
        # Carbs preference
        self.create_slider(input_frame, "Carbohydrates (g):", "carbs_var", 30, 0, 100, 2)
        
        # Sugar preference
        self.create_slider(input_frame, "Sugar (g):", "sugar_var", 5, 0, 30, 3)
        
        # Fiber preference
        self.create_slider(input_frame, "Dietary Fiber (g):", "fiber_var", 8, 0, 20, 4)
        
        # Fat preference
        self.create_slider(input_frame, "Fat (g):", "fat_var", 15, 0, 40, 5)
        
        # Health information frame
        health_frame = ttk.LabelFrame(parent, text="Health Information")
        health_frame.pack(fill=tk.X, pady=10)
        
        # Weight
        ttk.Label(health_frame, text="Weight (kg):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.weight_var = tk.DoubleVar(value=70.0)
        ttk.Entry(health_frame, textvariable=self.weight_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Height
        ttk.Label(health_frame, text="Height (cm):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.height_var = tk.DoubleVar(value=170.0)
        ttk.Entry(health_frame, textvariable=self.height_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Diabetic condition
        ttk.Label(health_frame, text="Diabetic:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.diabetic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(health_frame, variable=self.diabetic_var).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Recommendation settings
        settings_frame = ttk.LabelFrame(parent, text="Recommendation Settings")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Number of recommendations
        ttk.Label(settings_frame, text="Number of recommendations:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.num_recommendations_var = tk.IntVar(value=10)
        ttk.Combobox(settings_frame, textvariable=self.num_recommendations_var, values=[5, 10, 15, 20, 25], width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Filter by category
        ttk.Label(settings_frame, text="Filter by category:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(settings_frame, textvariable=self.category_filter_var, width=15)
        self.category_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Get all available categories
        categories = ['All']
        if hasattr(self.recommender, 'food_data') and not self.recommender.food_data.empty:
            categories.extend(sorted(self.recommender.food_data['Category'].unique()))
        self.category_combo['values'] = categories
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=15)
        
        # Get Recommendations button with better styling
        recommend_btn = ttk.Button(button_frame, text="Get Recommendations", 
                                  command=self.show_recommendations, style='TButton')
        recommend_btn.pack(fill=tk.X, pady=5)
        
        # Reset button
        reset_btn = ttk.Button(button_frame, text="Reset Preferences", 
                              command=self.reset_preferences, style='TButton')
        reset_btn.pack(fill=tk.X, pady=5)
        
    def create_slider(self, parent, label_text, var_name, default_value, min_val, max_val, row):
        """Create a slider with label and value display"""
        ttk.Label(parent, text=label_text).grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Create slider variable
        slider_var = tk.IntVar(value=default_value)
        setattr(self, var_name, slider_var)
        
        # Create slider
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=slider_var, 
                         orient=tk.HORIZONTAL, length=150)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Value label
        value_label = ttk.Label(parent, textvariable=slider_var, width=3)
        value_label.grid(row=row, column=2, padx=5, pady=5)
        
    def create_results_panel(self, parent):
        """Create the results panel with recommendations"""
        # Title
        ttk.Label(parent, text="Food Recommendations", style='Subheader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Create Treeview with scrollbar for recommendations
        columns = ('Name', 'Category', 'Energy', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'Diabetic')
        self.tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Define headings with better labels
        column_texts = {
            'Name': 'Food Name', 
            'Category': 'Category', 
            'Energy': 'Energy (kcal)', 
            'Protein': 'Protein (g)', 
            'Carbs': 'Carbs (g)',
            'Sugar': 'Sugar (g)',
            'Fiber': 'Fiber (g)',
            'Fat': 'Fat (g)',
            'Diabetic': 'Diabetic Score'
        }
        
        # Configure columns
        for col in columns:
            self.tree.heading(col, text=column_texts[col])
            width = 100 if col == 'Name' else 70
            self.tree.column(col, width=width, minwidth=50)
        
        # Adjust name column to be wider
        self.tree.column('Name', width=150, minwidth=100)
        
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Details panel
        details_frame = ttk.LabelFrame(parent, text="Nutritional Details")
        details_frame.pack(fill=tk.X, pady=10)
        
        # Selected food details
        self.details_text = tk.Text(details_frame, height=4, width=40, wrap=tk.WORD)
        self.details_text.pack(fill=tk.X, padx=5, pady=5)
        self.details_text.insert(tk.END, "Select a food item to view detailed nutritional information")
        self.details_text.config(state=tk.DISABLED)
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.show_food_details)
        
    def create_chart_panel(self, parent):
        """Create a panel with charts/graphs"""
        self.fig = plt.Figure(figsize=(10, 3), dpi=100)
        
        # Initialize the chart with empty data
        self.nutrition_chart = FigureCanvasTkAgg(self.fig, parent)
        self.nutrition_chart.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create chart axes
        self.ax1 = self.fig.add_subplot(131)  # Macronutrient distribution
        self.ax2 = self.fig.add_subplot(132)  # Category distribution
        self.ax3 = self.fig.add_subplot(133)  # Diabetic suitability
        
        # Update chart with empty data initially
        self.update_charts([])
        
    def update_charts(self, recommendations):
        """Update the charts with recommendation data"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        if recommendations and len(recommendations) > 0:
            try:
                # Macronutrient distribution chart
                labels = ['Protein', 'Carbs', 'Fat']
                values = [
                    sum(rec.get('Protein', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('Carbs', 0) for rec in recommendations) / len(recommendations),
                    sum(rec.get('Fat', 0) for rec in recommendations) / len(recommendations)
                ]
                
                # Filter out zero values for pie chart
                filtered_labels = []
                filtered_values = []
                for i, value in enumerate(values):
                    if value > 0:
                        filtered_labels.append(labels[i])
                        filtered_values.append(value)
                
                if sum(filtered_values) > 0:
                    colors = ['#ff9999','#66b3ff','#99ff99']
                    self.ax1.pie(filtered_values, labels=filtered_labels, colors=colors[:len(filtered_labels)], 
                                autopct='%1.1f%%', shadow=False, startangle=90)
                    self.ax1.set_title('Average Macronutrient Distribution')
                else:
                    self.ax1.text(0.5, 0.5, 'No macronutrient data available', ha='center', va='center')
                
                # Category distribution chart
                categories = {}
                for rec in recommendations:
                    cat = rec.get('Category', 'Unknown')
                    if cat in categories:
                        categories[cat] += 1
                    else:
                        categories[cat] = 1
                
                if categories:
                    cat_names = list(categories.keys())
                    cat_counts = list(categories.values())
                    self.ax2.bar(cat_names, cat_counts, color='#66b3ff')
                    self.ax2.set_title('Food Categories')
                    self.ax2.tick_params(axis='x', rotation=45)
                    self.ax2.set_ylim(0, max(cat_counts) + 1)
                else:
                    self.ax2.text(0.5, 0.5, 'No category data available', ha='center', va='center')
                
                # Diabetic suitability chart
                diabetic_scores = [rec.get('Diabetic_Score', 0) for rec in recommendations]
                if any(score > 0 for score in diabetic_scores):
                    max_score = max(diabetic_scores)
                    bins = list(range(0, int(max_score) + 2))
                    if len(bins) < 2:
                        bins = [0, 1, 2]
                    
                    self.ax3.hist(diabetic_scores, bins=bins, color='#99ff99', edgecolor='black')
                    self.ax3.set_title('Diabetic Suitability (Lower = Better)')
                    self.ax3.set_xlabel('Score')
                    self.ax3.set_ylabel('Count')
                else:
                    self.ax3.text(0.5, 0.5, 'No diabetic score data available', ha='center', va='center')
            except Exception as e:
                print(f"Error updating charts: {e}")
                self.ax1.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
                self.ax2.text(0.5, 0.5, 'Chart error', ha='center', va='center')
                self.ax3.text(0.5, 0.5, 'Chart error', ha='center', va='center')
        else:
            # Show placeholder text
            self.ax1.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            self.ax2.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            self.ax3.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            
            # Set empty titles
            self.ax1.set_title('Macronutrient Distribution')
            self.ax2.set_title('Food Categories')
            self.ax3.set_title('Diabetic Suitability')
        
        # Adjust layout and update canvas
        self.fig.tight_layout()
        self.nutrition_chart.draw()
        
    def show_food_details(self, event):
        """Show details for selected food item"""
        selected_items = self.tree.selection()
        if not selected_items:
            return
            
        item = selected_items[0]
        values = self.tree.item(item, 'values')
        
        # Update details text
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        details = f"Food: {values[0]}\n"
        details += f"Category: {values[1]}\n"
        details += f"Nutritional Content: Energy: {values[2]}kcal, Protein: {values[3]}g, Carbs: {values[4]}g\n"
        details += f"Sugar: {values[5]}g, Fiber: {values[6]}g, Fat: {values[7]}g, Diabetic Score: {values[8]}"
        
        self.details_text.insert(tk.END, details)
        self.details_text.config(state=tk.DISABLED)
        
    def show_recommendations(self):
        """Show food recommendations based on user preferences"""
        # Get user preferences
        user_prefs = {
            'Energy(kcal) by calculation': self.calories_var.get(),
            'Protein(g)': self.protein_var.get(),
            'CHOCDF (g) Carbohydrate': self.carbs_var.get(),
            'SUGAR(g)': self.sugar_var.get(),
            'FIBTG (g) Dietary fibre': self.fiber_var.get(),
            'Fat(g)': self.fat_var.get()
        }
        
        # Show loading in progress bar
        self.status_var.set("Finding recommendations...")
        self.progress["value"] = 10
        self.master.update_idletasks()
        
        # Get category filter
        category_filter = self.category_filter_var.get()
        diabetic_focus = self.diabetic_var.get()
        num_recommendations = self.num_recommendations_var.get()
        
        # Get recommendations
        self.progress["value"] = 50
        self.master.update_idletasks()
        
        # Check if the recommender has the updated method
        if hasattr(self.recommender, 'get_recommendations'):
            if 'diabetic_focus' in self.recommender.get_recommendations.__code__.co_varnames:
                # Use the new method with all parameters
                recommendations = self.recommender.get_recommendations(
                    user_prefs, 
                    diabetic_focus=diabetic_focus,
                    max_recommendations=num_recommendations
                )
            else:
                # Use the old method without parameters
                recommendations = self.recommender.get_recommendations(user_prefs)
        else:
            messagebox.showerror("Error", "Recommendation system not properly initialized")
            return
        
        # Apply category filter if not "All"
        if category_filter != "All":
            recommendations = [r for r in recommendations if r['Category'] == category_filter]
        
        # Clear previous recommendations
        self.progress["value"] = 80
        self.master.update_idletasks()
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                self.tree.insert('', 'end', values=(
                    rec.get('Name', 'Unknown'),
                    rec.get('Category', 'Unknown'),
                    rec.get('Energy', 0),
                    rec.get('Protein', 0),
                    rec.get('Carbs', 0),
                    rec.get('Sugar', 0),
                    rec.get('Fiber', 0),
                    rec.get('Fat', 0),
                    f"{rec.get('Diabetic_Score', 0):.1f}"
                ))
            self.status_var.set(f"Found {len(recommendations)} recommendations")
            
            # Update charts
            self.update_charts(recommendations)
        else:
            messagebox.showinfo("No Recommendations", 
                               "No recommendations found matching your criteria.")
            self.status_var.set("No recommendations found")
            
            # Clear charts
            self.update_charts([])
            
        # Reset progress bar
        self.progress["value"] = 100
        self.master.update_idletasks()
        self.master.after(500, lambda: self.progress.configure(value=0))
    
    def reset_preferences(self):
        """Reset preferences to default values"""
        # Reset nutrition sliders
        self.calories_var.set(500)
        self.protein_var.set(20)
        self.carbs_var.set(30)
        self.sugar_var.set(5)
        self.fiber_var.set(8)
        self.fat_var.set(15)
        
        # Reset other settings
        self.weight_var.set(70.0)
        self.height_var.set(170.0)
        self.diabetic_var.set(True)
        self.num_recommendations_var.set(10)
        self.category_filter_var.set("All")
        
        # Update status
        self.status_var.set("Preferences reset to defaults")
    
    def update_stats_display(self):
        """Update the statistics display in the bottom right corner"""
        try:
            stats = self.recommender.get_stats()
            
            # Clear previous stats
            for widget in self.stats_frame.winfo_children():
                widget.destroy()
            
            # Create stats labels
            stats_text = f"Loaded {stats['total_items']} food items | "
            stats_text += f"{len(stats['categories'])} categories"
            
            if 'diabetic_friendly' in stats and stats['diabetic_friendly'] > 0:
                stats_text += f" | {stats['diabetic_friendly']} diabetic-friendly items"
                
            if 'loading_time' in stats and stats['loading_time'] > 0:
                stats_text += f" | Loaded in {stats['loading_time']:.2f}s"
            
            ttk.Label(self.stats_frame, text=stats_text, style='Stats.TLabel').pack(side=tk.RIGHT)
        except Exception as e:
            print(f"Error updating stats display: {e}")
            ttk.Label(self.stats_frame, text="Stats unavailable", style='Stats.TLabel').pack(side=tk.RIGHT)

# Main function to run the application
def main():
    # Set up basic logging to console
    print("Starting application...")
    
    # Create the root window first
    root = tk.Tk()
    root.title("Food Recommendation System for Diabetic Patients")
    root.geometry("1000x700")
    
    # Set window icon if available
    try:
        # You can add an icon file to your project and use it here
        #root.iconbitmap("icon.ico")
        pass
    except:
        pass
    
    # Create a splash frame that covers the main window
    splash_frame = ttk.Frame(root)
    splash_frame.place(x=0, y=0, relwidth=1, relheight=1)
    splash_frame.configure(style='TFrame')
    
    # Create splash screen elements
    splash_label = ttk.Label(splash_frame, text="Food Recommendation System", font=("Arial", 18, "bold"))
    splash_label.pack(pady=50)
    
    subtitle_label = ttk.Label(splash_frame, text="Optimized for Diabetic Patients", font=("Arial", 12))
    subtitle_label.pack(pady=10)
    
    progress = ttk.Progressbar(splash_frame, orient="horizontal", length=400, mode="indeterminate")
    progress.pack(pady=20)
    progress.start()
    
    status_var = tk.StringVar(value="Initializing system...")
    status_label = ttk.Label(splash_frame, textvariable=status_var, font=("Arial", 10))
    status_label.pack(pady=20)
    
    # Add a footer with credits
    footer_text = "Developed by Surat Lawdi - Prince of Songkla University"
    footer_label = ttk.Label(splash_frame, text=footer_text, font=("Arial", 8))
    footer_label.pack(side=tk.BOTTOM, pady=20)
    
    def update_status(message):
        """Update status message and ensure UI updates"""
        print(f"Status: {message}")  # Debug print
        status_var.set(message)
        root.update_idletasks()  # Force update of the UI
    
    def initialize_app():
        try:
            # Create the food recommender
            recommender = FoodRecommendationSystem(update_status)
            
            # Create main app UI
            app = FoodRecommenderUI(root, recommender)
            
            # Remove splash screen
            update_status("Ready! System loaded successfully.")
            root.after(1500, splash_frame.destroy)  # Destroy splash frame after delay
            
        except Exception as e:
            # Show error on splash screen
            update_status(f"Error initializing: {str(e)}")
            print(f"Initialization error: {e}")  # Debug print
    
    # Schedule the initialization to happen after the window is shown
    root.after(100, initialize_app)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()