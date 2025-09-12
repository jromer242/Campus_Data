import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import sqlite3
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StudentSuccessPredictor:
    """ML Model for predicting student academic success"""
    
    def __init__(self, db_path='campus_data.db'):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.model_metrics = {}
        
    def load_and_prepare_data(self):
        """Load data from database and create ML-ready features"""
        print("📊 Loading and preparing data...")
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        
        query = """
        SELECT 
            s.student_id,
            s.gpa,
            s.year_level,
            s.major,
            s.is_active,
            s.enrollment_date,
            
            -- Enrollment features
            COUNT(e.enrollment_id) as total_enrollments,
            COUNT(CASE WHEN e.status = 'Completed' THEN 1 END) as completed_courses,
            COUNT(CASE WHEN e.status = 'Dropped' THEN 1 END) as dropped_courses,
            COUNT(CASE WHEN e.status = 'Enrolled' THEN 1 END) as current_enrollments,
            
            -- Grade features
            COUNT(CASE WHEN e.grade IN ('A', 'A-') THEN 1 END) as excellent_grades,
            COUNT(CASE WHEN e.grade IN ('B+', 'B', 'B-') THEN 1 END) as good_grades,
            COUNT(CASE WHEN e.grade IN ('C+', 'C', 'C-') THEN 1 END) as average_grades,
            COUNT(CASE WHEN e.grade IN ('D', 'F') THEN 1 END) as poor_grades,
            
            -- Course load features
            AVG(c.credits) as avg_course_credits,
            SUM(c.credits) as total_credits_attempted,
            
            -- Time-based features
            julianday('now') - julianday(s.enrollment_date) as days_enrolled
            
        FROM students s
        LEFT JOIN enrollments e ON s.student_id = e.student_id
        LEFT JOIN courses c ON e.course_id = c.course_id
        WHERE s.is_active = 1
        GROUP BY s.student_id, s.gpa, s.year_level, s.major, s.is_active, s.enrollment_date
        HAVING COUNT(e.enrollment_id) > 0
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} student records")
        
        # Target Variable: Student Success
        # Criteria: GPA >= 3.0 AND completion rate >= 70%
        df['completion_rate'] = df['completed_courses'] / df['total_enrollments'] * 100
        df['is_successful'] = ((df['gpa'] >= 3.0) & (df['completion_rate'] >= 70)).astype(int)
        
        # additional features
        df['courses_per_semester'] = df['total_enrollments'] / (df['days_enrolled'] / 120 + 1)  # Approx 120 days per semester
        df['drop_rate'] = df['dropped_courses'] / df['total_enrollments'] * 100
        df['excellence_rate'] = df['excellent_grades'] / df['total_enrollments'] * 100
        df['struggle_rate'] = df['poor_grades'] / df['total_enrollments'] * 100
        
        # Fill NaN values
        df = df.fillna(0)
        
        print("✅ Feature engineering complete")
        return df
    
    def train_models(self, df):
        """Train multiple ML models and compare performance"""
        print("🤖 Training ML models...")
        
        # Prep features
        feature_columns = [
            'gpa', 'year_level', 'total_enrollments', 'completed_courses', 
            'dropped_courses', 'current_enrollments', 'excellent_grades',
            'good_grades', 'average_grades', 'poor_grades', 'avg_course_credits',
            'total_credits_attempted', 'days_enrolled', 'completion_rate',
            'courses_per_semester', 'drop_rate', 'excellence_rate', 'struggle_rate'
        ]
        
        X = df[feature_columns].copy()
        y = df['is_successful'].copy()
        
        # Encode categorical variables if needed
        if 'major' in df.columns:
            le_major = LabelEncoder()
            X['major_encoded'] = le_major.fit_transform(df['major'])
            self.encoders['major'] = le_major
            feature_columns.append('major_encoded')
        
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        print(f"📊 Success rate: {y.mean():.1%}")
        
        # Model configurations
        models_config = {
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=42))
                ]),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__max_iter': [1000]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [3, 5]
                }
            }
        }
        
        # Train and evaluate models
        for model_name, config in models_config.items():
            print(f"\n🔧 Training {model_name}...")
            
            # Grid search for best parameters
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Store model and metrics
            self.models[model_name] = best_model
            self.model_metrics[model_name] = {
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"✅ {model_name} - ROC AUC: {roc_auc:.3f}")
            print(f"   Cross-validation: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Select best model
        best_model_name = max(self.model_metrics.keys(), 
                             key=lambda x: self.model_metrics[x]['roc_auc'])
        
        print(f"\n🏆 Best model: {best_model_name}")
        return X_test, y_test
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for predictions"""
        print("\n🔍 Analyzing feature importance...")
        
        best_model_name = max(self.model_metrics.keys(), 
                             key=lambda x: self.model_metrics[x]['roc_auc'])
        
        best_model = self.models[best_model_name]
        
        # Get feature importance (works for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
                
            return importance_df
        
        # For logistic regression, use coefficients
        elif hasattr(best_model.named_steps['classifier'], 'coef_'):
            coef = best_model.named_steps['classifier'].coef_[0]
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            }).sort_values('abs_coefficient', ascending=False)
            
            print("\n📊 Top 10 Most Important Features (by coefficient):")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['coefficient']:.3f}")
                
            return importance_df
        
        return None
    
    def predict_student_success(self, student_data):
        """Make prediction for a single student"""
        best_model_name = max(self.model_metrics.keys(), 
                             key=lambda x: self.model_metrics[x]['roc_auc'])
        
        model = self.models[best_model_name]
        
        # Prepare features (same as training)
        features = np.array([student_data[feature] for feature in self.feature_names]).reshape(1, -1)
        
        # Make prediction
        success_probability = model.predict_proba(features)[0, 1]
        prediction = model.predict(features)[0]
        
        # Risk assessment
        if success_probability >= 0.8:
            risk_level = "Low"
            recommendations = [
                "Continue current academic plan",
                "Consider taking on leadership roles",
                "Explore advanced coursework opportunities"
            ]
        elif success_probability >= 0.6:
            risk_level = "Medium"
            recommendations = [
                "Meet with academic advisor quarterly",
                "Consider study groups for challenging courses",
                "Monitor course load balance"
            ]
        else:
            risk_level = "High"
            recommendations = [
                "Schedule immediate meeting with academic advisor",
                "Consider tutoring services",
                "Evaluate current course load",
                "Explore academic support resources"
            ]
        
        return {
            'success_probability': float(success_probability),
            'predicted_success': bool(prediction),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'model_used': best_model_name,
            'confidence': 'High' if max(success_probability, 1-success_probability) > 0.7 else 'Medium'
        }
    
    def save_models(self, model_dir='models'):
        """Save trained models to disk"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{model_name}.pkl")
        
        # Save encoders and feature names
        joblib.dump(self.encoders, f"{model_dir}/encoders.pkl")
        joblib.dump(self.feature_names, f"{model_dir}/feature_names.pkl")
        joblib.dump(self.model_metrics, f"{model_dir}/model_metrics.pkl")
        
        print(f"✅ Models saved to {model_dir}/")
    
    def load_models(self, model_dir='models'):
        """Load trained models from disk"""
        self.encoders = joblib.load(f"{model_dir}/encoders.pkl")
        self.feature_names = joblib.load(f"{model_dir}/feature_names.pkl")
        self.model_metrics = joblib.load(f"{model_dir}/model_metrics.pkl")
        
        # Load all models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl') and model_file not in ['encoders.pkl', 'feature_names.pkl', 'model_metrics.pkl']:
                model_name = model_file.replace('.pkl', '')
                self.models[model_name] = joblib.load(f"{model_dir}/{model_file}")
        
        print("✅ Models loaded successfully")
    
    def generate_model_report(self):
        """Generate a comprehensive model performance report"""
        print("\n" + "="*50)
        print("🤖 STUDENT SUCCESS PREDICTION MODEL REPORT")
        print("="*50)
        
        for model_name, metrics in self.model_metrics.items():
            print(f"\n📊 {model_name.upper()}")
            print("-" * 30)
            print(f"ROC AUC Score: {metrics['roc_auc']:.3f}")
            print(f"Cross-validation: {metrics['cv_mean']:.3f} (±{metrics['cv_std']:.3f})")
            print(f"Best parameters: {metrics['best_params']}")
            
            # Classification metrics
            report = metrics['classification_report']
            print(f"Precision: {report['1']['precision']:.3f}")
            print(f"Recall: {report['1']['recall']:.3f}")
            print(f"F1-Score: {report['1']['f1-score']:.3f}")
        
        # Best model summary
        best_model = max(self.model_metrics.keys(), 
                        key=lambda x: self.model_metrics[x]['roc_auc'])
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   Performance: {self.model_metrics[best_model]['roc_auc']:.3f} ROC AUC")

def main():
    """Main function to train and evaluate models"""
    print("🚀 Starting Student Success ML Pipeline")
    
    # Initialize predictor
    predictor = StudentSuccessPredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_data()
    
    # Train models
    X_test, y_test = predictor.train_models(df)
    
    # Analyze feature importance
    importance_df = predictor.analyze_feature_importance()
    
    # Generate report
    predictor.generate_model_report()
    
    # Save models
    predictor.save_models()
    
    # Example prediction
    print("\n🔮 Example Prediction:")
    sample_student = {
        'gpa': 2.3,
        'year_level': 3,
        'total_enrollments': 8,
        'completed_courses': 5,
        'dropped_courses': 1,
        'current_enrollments': 2,
        'excellent_grades': 1,
        'good_grades': 2,
        'average_grades': 2,
        'poor_grades': 1,
        'avg_course_credits': 3.0,
        'total_credits_attempted': 24,
        'days_enrolled': 600,
        'completion_rate': 62.5,
        'courses_per_semester': 1.6,
        'drop_rate': 12.5,
        'excellence_rate': 12.5,
        'struggle_rate': 12.5,
        'major_encoded': 0
    }
    
    prediction = predictor.predict_student_success(sample_student)
    print(f"Success Probability: {prediction['success_probability']:.1%}")
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Model Used: {prediction['model_used']}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()