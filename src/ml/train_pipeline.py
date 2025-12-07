"""
Main Training Pipeline

This script orchestrates the complete ML pipeline:
1. Feature engineering
2. Model training
3. Model evaluation
4. Model saving

Usage:
    python train_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.feature_engineering import FeatureEngineer
from src.ml.model_training import ModelTrainer
from src.ml.evaluation import ModelEvaluator, FeatureImportanceAnalyzer
import joblib
import warnings
warnings.filterwarnings('ignore')


class StudentSuccessPipeline:
    """
    Complete ML pipeline for student success prediction
    """
    
    def __init__(
        self, 
        db_path: str = 'campus_data.db',
        model_dir: str = 'models',
        random_state: int = 42
    ):
        self.db_path = db_path
        self.model_dir = model_dir
        self.random_state = random_state
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(db_path=db_path)
        self.model_trainer = ModelTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()
        self.importance_analyzer = FeatureImportanceAnalyzer()
        
        # Storage for results
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = []
        self.models_metrics = {}
    
    def run_feature_engineering(self):
        """Run the feature engineering pipeline"""
        print("\n" + "="*70)
        print("STEP 1: FEATURE ENGINEERING")
        print("="*70)
        
        X, y, self.feature_names = self.feature_engineer.full_pipeline()
        
        return X, y
    
    def run_data_splitting(self, X, y):
        """Split data into train/val/test"""
        print("\n" + "="*70)
        print("STEP 2: DATA SPLITTING")
        print("="*70)
        
        splits = self.model_trainer.split_data(X, y)
        self.X_train, self.X_val, self.X_test = splits[0], splits[1], splits[2]
        self.y_train, self.y_val, self.y_test = splits[3], splits[4], splits[5]
        
        return splits
    
    def run_model_training(self):
        """Train all models"""
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        
        results = self.model_trainer.train_all_models(
            self.X_train, 
            self.y_train,
            cv_folds=5,
            scoring='roc_auc'
        )
        
        self.model_trainer.print_training_summary()
        
        return results
    
    def run_model_evaluation(self):
        """Evaluate all trained models"""
        print("\n" + "="*70)
        print("STEP 4: MODEL EVALUATION")
        print("="*70)
        
        for model_name, model in self.model_trainer.models.items():
            print(f"\n📊 Evaluating {model_name}...")
            
            metrics = self.evaluator.evaluate_model(
                model, 
                self.X_test, 
                self.y_test,
                model_name
            )
            
            self.models_metrics[model_name] = metrics
        
        # Generate comprehensive summary
        self.evaluator.generate_performance_summary(self.models_metrics)
        
        return self.models_metrics
    
    def run_feature_importance_analysis(self):
        """Analyze feature importance for all models"""
        print("\n" + "="*70)
        print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        for model_name, model in self.model_trainer.models.items():
            # Check model type and analyze accordingly
            if model_name in ['random_forest', 'gradient_boosting']:
                self.importance_analyzer.analyze_tree_based_model(
                    model, 
                    self.feature_names,
                    model_name,
                    top_n=15
                )
            elif model_name == 'logistic_regression':
                self.importance_analyzer.analyze_linear_model(
                    model,
                    self.feature_names,
                    model_name,
                    top_n=15
                )
    
    def save_artifacts(self):
        """Save all models and metadata"""
        print("\n" + "="*70)
        print("STEP 6: SAVING ARTIFACTS")
        print("="*70)
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save trained models
        self.model_trainer.save_models(self.model_dir)
        
        # Save encoders and feature names
        encoders_path = os.path.join(self.model_dir, 'encoders.pkl')
        joblib.dump(self.feature_engineer.encoders, encoders_path)
        print(f"   ✅ Saved encoders")
        
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        joblib.dump(self.feature_names, features_path)
        print(f"   ✅ Saved feature names")
        
        # Save model metrics
        metrics_path = os.path.join(self.model_dir, 'model_metrics.pkl')
        joblib.dump(self.models_metrics, metrics_path)
        print(f"   ✅ Saved model metrics")
        
        # Save feature descriptions
        descriptions_path = os.path.join(self.model_dir, 'feature_descriptions.pkl')
        descriptions = self.feature_engineer.get_feature_descriptions()
        joblib.dump(descriptions, descriptions_path)
        print(f"   ✅ Saved feature descriptions")
        
        print(f"\n✅ All artifacts saved to {self.model_dir}/")
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        print("\n" + "="*70)
        print("🚀 STUDENT SUCCESS ML PIPELINE")
        print("="*70)
        
        # Step 1: Feature Engineering
        X, y = self.run_feature_engineering()
        
        # Step 2: Data Splitting
        self.run_data_splitting(X, y)
        
        # Step 3: Model Training
        self.run_model_training()
        
        # Step 4: Model Evaluation
        self.run_model_evaluation()
        
        # Step 5: Feature Importance
        self.run_feature_importance_analysis()
        
        # Step 6: Save Everything
        self.save_artifacts()
        
        # Final Summary
        self.print_final_summary()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
    
    def print_final_summary(self):
        """Print final summary with key takeaways"""
        print("\n" + "="*70)
        print("📋 FINAL SUMMARY")
        print("="*70)
        
        # Best model
        best_name, best_metrics = self.evaluator.get_best_model(self.models_metrics)
        
        print(f"\n🏆 Best Model: {best_name.upper()}")
        print(f"   ROC AUC: {best_metrics['roc_auc']:.3f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
        print(f"   F1 Score: {best_metrics['f1_score']:.3f}")
        
        # Dataset info
        print(f"\n📊 Dataset Information:")
        print(f"   Total samples: {len(self.X_train) + len(self.X_val) + len(self.X_test)}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Success rate: {self.y_test.mean():.1%}")
        
        # Top features
        if best_name in self.importance_analyzer.importance_data:
            print(f"\n🔍 Top 5 Most Important Features:")
            importance_df = self.importance_analyzer.importance_data[best_name]
            for i, row in enumerate(importance_df.head(5).iterrows(), 1):
                feature = row[1]['feature']
                if 'importance' in row[1]:
                    score = row[1]['importance']
                else:
                    score = row[1]['abs_coefficient']
                print(f"   {i}. {feature}: {score:.4f}")
        
        print(f"\n💾 Models saved to: {self.model_dir}/")
        print(f"\n📝 Next steps:")
        print(f"   - Review model performance")
        print(f"   - Use models for predictions")
        print(f"   - Deploy via API")


def main():
    """Main entry point"""
    # Configuration
    DB_PATH = 'campus_data.db'
    MODEL_DIR = 'models'
    RANDOM_STATE = 42
    
    # Initialize and run pipeline
    pipeline = StudentSuccessPipeline(
        db_path=DB_PATH,
        model_dir=MODEL_DIR,
        random_state=RANDOM_STATE
    )
    
    pipeline.run_full_pipeline()
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()