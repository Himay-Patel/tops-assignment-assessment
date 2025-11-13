import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.predictions = {}
        self.results_df = None
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=== Loading Data ===")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")
            return True
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("\n=== Performing EDA ===")
        
        # Basic statistics
        print("\nBasic Statistics for Price:")
        print(self.df['Price'].describe())
        
        # Create EDA plots
        self._create_eda_plots()
        
        # Correlation analysis
        self._analyze_correlations()
        
        # Feature distributions
        self._analyze_feature_distributions()
    
    def _create_eda_plots(self):
        """Create various EDA plots"""
        plt.figure(figsize=(15, 12))
        
        # Price distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.df['Price'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        
        # Log Price distribution
        plt.subplot(2, 3, 2)
        plt.hist(np.log1p(self.df['Price']), bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        plt.title('Log Price Distribution')
        plt.xlabel('Log(Price)')
        plt.ylabel('Frequency')
        
        # Area vs Price
        plt.subplot(2, 3, 3)
        plt.scatter(self.df['Area'], self.df['Price'], alpha=0.6, color='green')
        plt.xlabel('Area (sq ft)')
        plt.ylabel('Price')
        plt.title('Area vs Price')
        
        # Price by Bedrooms
        plt.subplot(2, 3, 4)
        self.df.boxplot(column='Price', by='No. of Bedrooms', ax=plt.gca())
        plt.title('Price by Number of Bedrooms')
        plt.suptitle('')
        
        # Top locations
        plt.subplot(2, 3, 5)
        top_locations = self.df['Location'].value_counts().head(10)
        top_locations.plot(kind='bar', color='orange')
        plt.title('Top 10 Locations')
        plt.xticks(rotation=45)
        
        # Amenities correlation
        plt.subplot(2, 3, 6)
        amenities = ['Gymnasium', 'SwimmingPool', 'ClubHouse', '24X7Security', 'LiftAvailable']
        amenity_prices = [self.df[self.df[amenity] == 1]['Price'].mean() for amenity in amenities]
        plt.bar(amenities, amenity_prices, color='purple')
        plt.title('Average Price by Amenity')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_correlations(self):
        """Analyze correlations between features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Correlation with Price
        correlation_matrix = self.df[numeric_cols].corr()
        price_corr = correlation_matrix['Price'].sort_values(ascending=False)
        
        print("\nTop 10 features correlated with Price:")
        for feature, corr in price_corr.head(11).items():
            if feature != 'Price':
                print(f"  {feature}: {corr:.3f}")
        
        # Correlation heatmap for top features
        top_features = price_corr.head(11).index.tolist()
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df[top_features].corr(), annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Heatmap (Top Features)')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_feature_distributions(self):
        """Analyze distributions of important features"""
        print("\n=== Feature Analysis ===")
        
        # Categorical features analysis
        categorical_features = ['No. of Bedrooms', 'Resale']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                print(f"\n{feature} value counts:")
                print(self.df[feature].value_counts())
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n=== Preprocessing Data ===")
        
        # Handle categorical variables - Location
        le = LabelEncoder()
        self.df['Location_encoded'] = le.fit_transform(self.df['Location'])
        
        # Select features for modeling
        feature_columns = self._select_features()
        
        # Create feature matrix and target vector
        X = self.df[feature_columns]
        y = self.df['Price']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        
        return True
    
    def _select_features(self):
        """Select features for modeling based on correlation and domain knowledge"""
        # Using top correlated features and important amenities
        base_features = ['Area', 'No. of Bedrooms', 'Location_encoded']
        
        # Amenity features
        amenity_features = [
            'Gymnasium', 'SwimmingPool', 'ClubHouse', '24X7Security',
            'PowerBackup', 'LiftAvailable', 'Resale', 'MaintenanceStaff',
            'LandscapedGardens', 'JoggingTrack', 'RainWaterHarvesting'
        ]
        
        # Only include amenities that exist in the dataset
        available_amenities = [feature for feature in amenity_features if feature in self.df.columns]
        
        selected_features = base_features + available_amenities[:7]  # Limit to top 7 amenities
        
        print(f"Selected {len(selected_features)} features for modeling:")
        for feature in selected_features:
            print(f"  - {feature}")
        
        return selected_features
    
    def apply_pca(self, variance_threshold=0.95):
        """Apply PCA for dimensionality reduction"""
        pca = PCA(n_components=variance_threshold)
        X_train_pca = pca.fit_transform(self.X_train_scaled)
        X_test_pca = pca.transform(self.X_test_scaled)
        
        print(f"PCA: Original features {self.X_train_scaled.shape[1]} -> PCA features {X_train_pca.shape[1]}")
        print(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        return X_train_pca, X_test_pca, pca
    
    def build_all_models(self):
        """Build and train all machine learning models"""
        print("\n=== Building Models ===")
        
        # 1. Simple Linear Regression (Area only)
        print("1. Simple Linear Regression...")
        simple_lr = LinearRegression()
        simple_lr.fit(self.X_train[['Area']], self.y_train)
        self.models['Simple Linear Regression'] = simple_lr
        self.predictions['Simple Linear Regression'] = simple_lr.predict(self.X_test[['Area']])
        
        # 2. Multiple Linear Regression
        print("2. Multiple Linear Regression...")
        multiple_lr = LinearRegression()
        multiple_lr.fit(self.X_train_scaled, self.y_train)
        self.models['Multiple Linear Regression'] = multiple_lr
        self.predictions['Multiple Linear Regression'] = multiple_lr.predict(self.X_test_scaled)
        
        # 3. PCA + Multiple Linear Regression
        print("3. PCA + Linear Regression...")
        X_train_pca, X_test_pca, _ = self.apply_pca()
        pca_lr = LinearRegression()
        pca_lr.fit(X_train_pca, self.y_train)
        self.models['PCA + Linear Regression'] = pca_lr
        self.predictions['PCA + Linear Regression'] = pca_lr.predict(X_test_pca)
        
        # 4. Lasso Regression
        print("4. Lasso Regression...")
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(self.X_train_scaled, self.y_train)
        self.models['Lasso Regression'] = lasso
        self.predictions['Lasso Regression'] = lasso.predict(self.X_test_scaled)
        
        # 5. Ridge Regression
        print("5. Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(self.X_train_scaled, self.y_train)
        self.models['Ridge Regression'] = ridge
        self.predictions['Ridge Regression'] = ridge.predict(self.X_test_scaled)
        
        # 6. Support Vector Regression
        print("6. Support Vector Regression...")
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(self.X_train_scaled, self.y_train)
        self.models['Support Vector Regression'] = svr
        self.predictions['Support Vector Regression'] = svr.predict(self.X_test_scaled)
        
        # 7. Decision Tree Regressor
        print("7. Decision Tree Regressor...")
        dtree = DecisionTreeRegressor(random_state=42)
        dtree.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dtree
        self.predictions['Decision Tree'] = dtree.predict(self.X_test)
        
        # 8. Random Forest Regression
        print("8. Random Forest Regression...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = rf.predict(self.X_test)
        
        # 9. Hyperparameter Tuning for Random Forest
        print("9. Tuning Random Forest...")
        tuned_rf = self._tune_random_forest()
        self.models['Random Forest (Tuned)'] = tuned_rf
        self.predictions['Random Forest (Tuned)'] = tuned_rf.predict(self.X_test)
        
        # 10. Hyperparameter Tuning for SVR
        print("10. Tuning SVR...")
        tuned_svr = self._tune_svr()
        self.models['SVR (Tuned)'] = tuned_svr
        self.predictions['SVR (Tuned)'] = tuned_svr.predict(self.X_test_scaled)
        
        # Save all models
        for name, model in self.models.items():
            filename = f"model_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
            joblib.dump(model, filename)
        
        print("All models built and saved successfully!")
    
    def _tune_random_forest(self):
        """Tune Random Forest hyperparameters"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"  Best RF parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _tune_svr(self):
        """Tune SVR hyperparameters"""
        param_dist = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear']
        }
        
        svr = SVR()
        random_search = RandomizedSearchCV(svr, param_dist, n_iter=10, cv=5, 
                                       scoring='r2', random_state=42, n_jobs=-1, verbose=0)
        random_search.fit(self.X_train_scaled, self.y_train)
        
        print(f"  Best SVR parameters: {random_search.best_params_}")
        return random_search.best_estimator_
    
    def evaluate_models(self):
        """Evaluate all models and return results dataframe"""
        print("\n=== Model Evaluation ===")
        
        results = []    
        for model_name, y_pred in self.predictions.items():
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'R² Score': r2,
                'RMSE': rmse,
                'MAE': mae,
                'MSE': mse
            })
            
            print(f"{model_name}:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
        
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('R² Score', ascending=False)
        
        # Save results to CSV
        self.results_df.to_csv('model_results.csv', index=False)
        
        # Create comparison plot
        self._create_comparison_plot()
        
        return self.results_df
    
    def _create_comparison_plot(self):
        """Create model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score comparison
        axes[0, 0].barh(self.results_df['Model'], self.results_df['R² Score'], color='skyblue')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xlabel('R² Score')
        
        # RMSE comparison
        axes[0, 1].barh(self.results_df['Model'], self.results_df['RMSE'], color='lightcoral')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_xlabel('RMSE')
        
        # MAE comparison
        axes[1, 0].barh(self.results_df['Model'], self.results_df['MAE'], color='lightgreen')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].set_xlabel('MAE')
        
        # Actual vs Predicted for best model
        self.best_model_name = self.results_df.iloc[0]['Model']
        y_pred_best = self.predictions[self.best_model_name]
        
        axes[1, 1].scatter(self.y_test, y_pred_best, alpha=0.6, color='purple')
        axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Prices')
        axes[1, 1].set_ylabel('Predicted Prices')
        axes[1, 1].set_title(f'Actual vs Predicted - {self.best_model_name}')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def select_best_model(self):
        """Select and save the best model"""
        self.best_model_name = self.results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n=== Best Model Selected ===")
        print(f"Model: {self.best_model_name}")
        print(f"R² Score: {self.results_df.iloc[0]['R² Score']:.4f}")
        print(f"RMSE: {self.results_df.iloc[0]['RMSE']:.2f}")
        print(f"MAE: {self.results_df.iloc[0]['MAE']:.2f}")
        
        # Save best model
        joblib.dump(self.best_model, 'best_model.pkl')
        print("Best model saved as 'best_model.pkl'")
        
        return self.best_model_name, self.best_model
    
    def run_complete_pipeline(self):
        """Run the complete house price prediction pipeline"""
        print("=== House Price Prediction Pipeline ===")
        
        # Step 1: Load data
        if not self.load_data():
            return
        
        # Step 2: Perform EDA
        self.perform_eda()
        
        # Step 3: Preprocess data
        self.preprocess_data()
        
        # Step 4: Build models
        self.build_all_models()
        
        # Step 5: Evaluate models
        self.evaluate_models()
        
        # Step 6: Select best model
        self.select_best_model()
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Best Model: {self.best_model_name}")
        print(f"Results saved in 'model_results.csv'")
        print(f"Best model saved as 'best_model.pkl'")
        
        # Display feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            self._display_feature_importance()

    def _display_feature_importance(self):
        """Display feature importance for tree-based models"""
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Feature Importance - {self.best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

def main():
    """Main function to run the house price prediction"""
    # Initialize the predictor
    predictor = HousePricePredictor('task_1.csv')
    
    # Run the complete pipeline
    predictor.run_complete_pipeline()

if __name__ == "__main__":
    main()