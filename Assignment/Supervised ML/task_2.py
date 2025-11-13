# customer_churn_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class CustomerChurnPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.df = pd.read_csv(data_path)
        self.df_model = None
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nDataset Info:")
        print(self.df.info())
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        print("\nTarget variable distribution:")
        print(self.df['Attrition_Flag'].value_counts())
        
        # Visualizations
        plt.figure(figsize=(15, 12))
        
        # Target variable
        plt.subplot(3, 3, 1)
        self.df['Attrition_Flag'].value_counts().plot(kind='bar')
        plt.title('Target Variable Distribution')
        plt.xticks(rotation=45)
        
        # Numerical features
        plt.subplot(3, 3, 2)
        self.df['Customer_Age'].hist(bins=30)
        plt.title('Customer Age Distribution')
        
        plt.subplot(3, 3, 3)
        self.df['Credit_Limit'].hist(bins=30)
        plt.title('Credit Limit Distribution')
        
        plt.subplot(3, 3, 4)
        self.df['Total_Trans_Amt'].hist(bins=30)
        plt.title('Total Transaction Amount')
        
        # Categorical features
        plt.subplot(3, 3, 5)
        self.df['Gender'].value_counts().plot(kind='bar')
        plt.title('Gender Distribution')
        
        plt.subplot(3, 3, 6)
        self.df['Education_Level'].value_counts().plot(kind='bar')
        plt.title('Education Level')
        
        plt.subplot(3, 3, 7)
        self.df['Marital_Status'].value_counts().plot(kind='bar')
        plt.title('Marital Status')
        
        plt.subplot(3, 3, 8)
        self.df['Income_Category'].value_counts().plot(kind='bar')
        plt.title('Income Category')
        
        plt.subplot(3, 3, 9)
        self.df['Card_Category'].value_counts().plot(kind='bar')
        plt.title('Card Category')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        numerical_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 
                         'Total_Relationship_Count', 'Months_Inactive_12_mon', 
                         'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                         'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                         'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()
        
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create a copy for modeling
        self.df_model = self.df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['CLIENTNUM', 
                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 
                          'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
        self.df_model = self.df_model.drop(columns=columns_to_drop)
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_model[col] = le.fit_transform(self.df_model[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        self.df_model['Attrition_Flag'] = self.df_model['Attrition_Flag'].map({
            'Existing Customer': 0, 
            'Attrited Customer': 1
        })
        
        print("Target variable distribution after encoding:")
        print(self.df_model['Attrition_Flag'].value_counts())
        
    def handle_imbalance_and_split(self):
        """Handle class imbalance and split the data"""
        print("\n=== HANDLING CLASS IMBALANCE ===")
        
        X = self.df_model.drop('Attrition_Flag', axis=1)
        y = self.df_model['Attrition_Flag']
        
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        
        # Handle imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def build_logistic_regression(self):
        """Build Logistic Regression model"""
        print("\n=== LOGISTIC REGRESSION ===")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_lr = lr_model.predict(self.X_test_scaled)
        y_pred_proba_lr = lr_model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.models['Logistic Regression'] = lr_model
        self.results['Logistic Regression'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_lr),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_lr)
        }
        
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_lr):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_lr):.4f}")
        
    def build_naive_bayes(self):
        """Build Naive Bayes model"""
        print("\n=== NAIVE BAYES ===")
        nb_model = GaussianNB()
        nb_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_nb = nb_model.predict(self.X_test_scaled)
        y_pred_proba_nb = nb_model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.models['Naive Bayes'] = nb_model
        self.results['Naive Bayes'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_nb),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_nb)
        }
        
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_nb):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_nb):.4f}")
        
    def build_knn(self):
        """Build K-Nearest Neighbors model"""
        print("\n=== K-NEAREST NEIGHBORS ===")
        knn_model = KNeighborsClassifier()
        knn_model.fit(self.X_train_scaled, self.y_train)
        
        y_pred_knn = knn_model.predict(self.X_test_scaled)
        y_pred_proba_knn = knn_model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.models['K-Nearest Neighbors'] = knn_model
        self.results['K-Nearest Neighbors'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_knn),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_knn)
        }
        
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred_knn):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_knn):.4f}")
        
    def build_svm(self):
        """Build Support Vector Machine with GridSearchCV"""
        print("\n=== SUPPORT VECTOR MACHINE ===")
        svc_param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        svc_grid = GridSearchCV(
            SVC(probability=True, random_state=42), 
            svc_param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        svc_grid.fit(self.X_train_scaled, self.y_train)
        
        print(f"Best parameters: {svc_grid.best_params_}")
        print(f"Best cross-validation score: {svc_grid.best_score_:.4f}")
        
        svc_model = svc_grid.best_estimator_
        y_pred_svc = svc_model.predict(self.X_test_scaled)
        y_pred_proba_svc = svc_model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.models['Support Vector Machine'] = svc_model
        self.results['Support Vector Machine'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_svc),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_svc)
        }
        
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred_svc):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_svc):.4f}")
        
    def build_decision_tree(self):
        """Build Decision Tree with GridSearchCV"""
        print("\n=== DECISION TREE ===")
        dt_param_grid = {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        dt_grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42), 
            dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        dt_grid.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {dt_grid.best_params_}")
        print(f"Best cross-validation score: {dt_grid.best_score_:.4f}")
        
        dt_model = dt_grid.best_estimator_
        y_pred_dt = dt_model.predict(self.X_test)
        y_pred_proba_dt = dt_model.predict_proba(self.X_test)[:, 1]
        
        self.models['Decision Tree'] = dt_model
        self.results['Decision Tree'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_dt),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_dt)
        }
        
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred_dt):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_dt):.4f}")
        
    def build_random_forest(self):
        """Build Random Forest with RandomizedSearchCV"""
        print("\n=== RANDOM FOREST ===")
        rf_param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        rf_random = RandomizedSearchCV(
            RandomForestClassifier(random_state=42), 
            rf_param_dist, n_iter=20, cv=5, 
            scoring='accuracy', random_state=42, n_jobs=-1
        )
        rf_random.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {rf_random.best_params_}")
        print(f"Best cross-validation score: {rf_random.best_score_:.4f}")
        
        rf_model = rf_random.best_estimator_
        y_pred_rf = rf_model.predict(self.X_test)
        y_pred_proba_rf = rf_model.predict_proba(self.X_test)[:, 1]
        
        self.models['Random Forest'] = rf_model
        self.results['Random Forest'] = {
            'accuracy': accuracy_score(self.y_test, y_pred_rf),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_rf)
        }
        
        print(f"Test Accuracy: {accuracy_score(self.y_test, y_pred_rf):.4f}")
        print(f"Test ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba_rf):.4f}")
        
    def compare_models(self):
        """Compare all models and select the best one"""
        print("\n=== MODEL COMPARISON ===")
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        results_df = results_df.sort_values('roc_auc', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(results_df)
        
        # Visualize model comparison
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=results_df.index, y=results_df['accuracy'])
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=results_df.index, y=results_df['roc_auc'])
        plt.title('Model ROC-AUC Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('ROC-AUC')
        
        plt.tight_layout()
        plt.show()
        
        # Get best model
        best_model_name = results_df.index[0]
        best_model = self.models[best_model_name]
        
        print(f"\nBEST MODEL: {best_model_name}")
        print(f"Best Model Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
        print(f"Best Model ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Feature Importance - {best_model_name}')
            plt.tight_layout()
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return best_model_name, best_model, results_df
        
    def evaluate_best_model(self, best_model_name, best_model):
        """Perform detailed evaluation of the best model"""
        print(f"\n=== DETAILED EVALUATION OF BEST MODEL: {best_model_name} ===")
        
        # Get predictions
        if best_model_name in ['Decision Tree', 'Random Forest']:
            y_pred_best = best_model.predict(self.X_test)
            y_pred_proba_best = best_model.predict_proba(self.X_test)[:, 1]
        else:
            y_pred_best = best_model.predict(self.X_test_scaled)
            y_pred_proba_best = best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_best))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churn', 'Churn'], 
                   yticklabels=['Not Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("STARTING CUSTOMER CHURN PREDICTION ANALYSIS")
        print("=" * 50)
        
        # Step 1: EDA
        self.perform_eda()
        
        # Step 2: Preprocessing
        self.preprocess_data()
        
        # Step 3: Handle imbalance and split data
        self.handle_imbalance_and_split()
        
        # Step 4-8: Build all models
        self.build_logistic_regression()
        self.build_naive_bayes()
        self.build_knn()
        self.build_svm()
        self.build_decision_tree()
        self.build_random_forest()
        
        # Step 9: Model comparison and selection
        best_model_name, best_model, results_df = self.compare_models()
        
        # Detailed evaluation of best model
        self.evaluate_best_model(best_model_name, best_model)
        
        # Final summary
        self.print_final_summary(best_model_name, results_df)
        
    def print_final_summary(self, best_model_name, results_df):
        """Print final project summary"""
        print("\n" + "="*60)
        print("PROJECT SUMMARY")
        print("="*60)
        
        print(f"Dataset: {self.df.shape[0]} samples with {self.df.shape[1]} features")
        print(f"Target variable: {self.df['Attrition_Flag'].value_counts().to_dict()}")
        print(f"Class imbalance handled using SMOTE")
        print(f"Total models evaluated: {len(self.models)}")
        print(f"Best performing model: {best_model_name}")
        print(f"Best model accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
        print(f"Best model ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")
        
        print("\nModel Ranking (by ROC-AUC):")
        for i, (model_name, metrics) in enumerate(results_df.iterrows(), 1):
            print(f"{i}. {model_name}: Accuracy={metrics['accuracy']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)

def main():
    """Main function to run the customer churn prediction"""
    # Initialize the predictor
    predictor = CustomerChurnPredictor('task_2.csv')
    
    # Run complete analysis
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main()