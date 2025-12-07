"""
Machine Learning Package for Student Success Prediction

This package contains modules for:
- Feature engineering and data preparation
- Model training and hyperparameter tuning
- Model evaluation and performance analysis
- Production predictions and inference

Usage:
    from src.ml import StudentSuccessPredictor, StudentSuccessPipeline
    
    # For training
    pipeline = StudentSuccessPipeline()
    pipeline.run_full_pipeline()
    
    # For predictions
    predictor = StudentSuccessPredictor()
    predictor.load_models('models/')
    result = predictor.predict(student_data)
"""

__version__ = '0.1.0'
__author__ = 'Your Name'

# Import main classes for easy access
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator, FeatureImportanceAnalyzer
from .student_predictor import StudentSuccessPredictor
from .train_pipeline import StudentSuccessPipeline

# Define what gets imported with "from src.ml import *"
__all__ = [
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'FeatureImportanceAnalyzer',
    'StudentSuccessPredictor',
    'StudentSuccessPipeline',
]

# Package-level configuration
DEFAULT_DB_PATH = 'campus_data.db'
DEFAULT_MODEL_DIR = 'models'
DEFAULT_RANDOM_STATE = 42

# Feature categories for reference
FEATURE_CATEGORIES = {
    'basic': ['gpa', 'year_level'],
    'enrollment': ['total_enrollments', 'completed_courses', 'dropped_courses', 'current_enrollments'],
    'grades': ['excellent_grades', 'good_grades', 'average_grades', 'poor_grades'],
    'rates': ['completion_rate', 'drop_rate', 'excellence_rate', 'struggle_rate'],
    'composite': ['engagement_score', 'grade_diversity', 'courses_per_semester']
}

# Model configuration defaults
MODEL_CONFIGS = {
    'default_cv_folds': 5,
    'default_scoring': 'roc_auc',
    'default_test_size': 0.2,
    'default_val_size': 0.2
}

def get_version():
    """Return package version"""
    return __version__

def quick_predict(student_data, model_dir='models'):
    """
    Convenience function for quick predictions
    
    Args:
        student_data: Dictionary of student features
        model_dir: Directory containing trained models
        
    Returns:
        Prediction dictionary
    """
    predictor = StudentSuccessPredictor(model_dir=model_dir)
    predictor.load_models()
    return predictor.predict(student_data)

def quick_train(db_path='campus_data.db', model_dir='models'):
    """
    Convenience function for quick training
    
    Args:
        db_path: Path to database
        model_dir: Directory to save models
        
    Returns:
        Trained pipeline object
    """
    pipeline = StudentSuccessPipeline(db_path=db_path, model_dir=model_dir)
    pipeline.run_full_pipeline()
    return pipeline