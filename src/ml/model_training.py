"""
Model Training Module

This module handles model training, hyperparameter tuning,
and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any, Optional, List
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training and tuning of multiple ML models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Define model configurations and hyperparameter grids
        
        Returns:
            Dictionary of model configurations
        """
        configs = {
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=self.random_state))
                ]),
                'params': {
                    'classifier__C': [0.01, 0.1, 1, 10],
                    'classifier__penalty': ['l2'],
                    'classifier__max_iter': [1000],
                    'classifier__solver': ['lbfgs']
                },
                'description': 'Logistic Regression with L2 regularization'
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'description': 'Random Forest Classifier'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 1.0]
                },
                'description': 'Gradient Boosting Classifier'
            }
        }
        
        return configs
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
               pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print(f"📊 Splitting data...")
        print(f"   Test size: {test_size:.0%}")
        print(f"   Validation size: {val_size:.0%}")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: separate validation from training
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"\n✅ Data split complete:")
        print(f"   Training set:   {len(X_train):>5} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation set: {len(X_val):>5} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test set:       {len(X_test):>5} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\n   Class distribution:")
        print(f"   Training:   {y_train.mean():.1%} positive")
        print(f"   Validation: {y_val.mean():.1%} positive")
        print(f"   Test:       {y_test.mean():.1%} positive")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = -1
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with hyperparameter tuning
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            n_jobs: Number of parallel jobs
            
        Returns:
            Tuple of (best_model, training_info)
        """
        print(f"\n🔧 Training {model_name}...")
        print(f"   Description: {model_config['description']}")
        print(f"   Hyperparameter combinations: {self._count_param_combinations(model_config['params'])}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Extract best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv_folds, scoring=scoring, n_jobs=n_jobs
        )
        
        # Store results
        self.models[model_name] = best_model
        self.best_params[model_name] = best_params
        self.cv_scores[model_name] = cv_scores
        
        training_info = {
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"   ✅ Best {scoring}: {grid_search.best_score_:.3f}")
        print(f"   ✅ CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        print(f"   ✅ Best params: {best_params}")
        
        return best_model, training_info
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Tuple[Any, Dict]]:
        """
        Train all configured models
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_names: List of model names to train (None = all)
            cv_folds: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary of model_name -> (model, training_info)
        """
        print("\n" + "="*60)
        print("🤖 TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        configs = self.get_model_configs()
        
        # Filter models if specified
        if model_names:
            configs = {k: v for k, v in configs.items() if k in model_names}
        
        results = {}
        
        for model_name, config in configs.items():
            model, info = self.train_single_model(
                model_name, config, X_train, y_train,
                cv_folds=cv_folds, scoring=scoring
            )
            results[model_name] = (model, info)
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*60)
        
        return results
    
    def _count_param_combinations(self, param_grid: Dict) -> int:
        """Count total number of parameter combinations"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on CV scores
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.cv_scores:
            raise ValueError("No models have been trained yet")
        
        best_model_name = max(
            self.cv_scores.keys(),
            key=lambda x: self.cv_scores[x].mean()
        )
        
        return best_model_name, self.models[best_model_name]
    
    def save_models(self, save_dir: str = 'models') -> None:
        """
        Save all trained models to disk
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving models to {save_dir}/")
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"   ✅ Saved {model_name}")
        
        # Save best parameters
        params_path = os.path.join(save_dir, "best_params.pkl")
        joblib.dump(self.best_params, params_path)
        
        # Save CV scores
        cv_path = os.path.join(save_dir, "cv_scores.pkl")
        joblib.dump(self.cv_scores, cv_path)
        
        print(f"✅ All models saved successfully")
    
    def load_models(self, save_dir: str = 'models') -> None:
        """
        Load trained models from disk
        
        Args:
            save_dir: Directory containing saved models
        """
        print(f"\n📂 Loading models from {save_dir}/")
        
        # Load all .pkl files except metadata
        for file in os.listdir(save_dir):
            if file.endswith('.pkl') and file not in ['best_params.pkl', 'cv_scores.pkl']:
                model_name = file.replace('.pkl', '')
                model_path = os.path.join(save_dir, file)
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✅ Loaded {model_name}")
        
        # Load metadata
        params_path = os.path.join(save_dir, "best_params.pkl")
        if os.path.exists(params_path):
            self.best_params = joblib.load(params_path)
        
        cv_path = os.path.join(save_dir, "cv_scores.pkl")
        if os.path.exists(cv_path):
            self.cv_scores = joblib.load(cv_path)
        
        print(f"✅ Models loaded successfully")
    
    def print_training_summary(self) -> None:
        """Print summary of all trained models"""
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY")
        print("="*70)
        
        summary_data = []
        for model_name in self.models.keys():
            cv_scores = self.cv_scores.get(model_name, [])
            summary_data.append({
                'Model': model_name,
                'CV Mean': f"{cv_scores.mean():.3f}",
                'CV Std': f"{cv_scores.std():.3f}",
                'Best Params': str(self.best_params.get(model_name, {}))[:50] + "..."
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Highlight best model
        if self.cv_scores:
            best_name, _ = self.get_best_model()
            print(f"\n🏆 Best Model: {best_name}")


# Example usage
if __name__ == "__main__":
    # This would typically be called from your main training pipeline
    print("🤖 Model Training Module - Use this in your training pipeline")