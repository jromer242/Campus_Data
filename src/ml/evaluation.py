"""
Model Evaluation and Reporting Module

This module handles all model evaluation, metrics calculation,
feature importance analysis, and report generation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    accuracy_score,
    precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Handles model evaluation and performance analysis"""
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model
        
        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1_score': f1_score(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_true': y_test
            }
        }
        
        # Store for later use
        self.metrics[model_name] = metrics
        
        return metrics
    
    def compare_models(self, models_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models side-by-side
        
        Args:
            models_metrics: Dictionary of model names to their metrics
            
        Returns:
            DataFrame comparing all models
        """
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'ROC AUC': metrics['roc_auc'],
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Precision': metrics['classification_report']['1']['precision'],
                'Recall': metrics['classification_report']['1']['recall']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('ROC AUC', ascending=False)
        
        return df
    
    def get_best_model(self, models_metrics: Dict[str, Dict]) -> Tuple[str, Dict]:
        """
        Identify the best performing model based on ROC AUC
        
        Args:
            models_metrics: Dictionary of model names to their metrics
            
        Returns:
            Tuple of (best_model_name, best_model_metrics)
        """
        best_model_name = max(
            models_metrics.keys(), 
            key=lambda x: models_metrics[x]['roc_auc']
        )
        
        return best_model_name, models_metrics[best_model_name]
    
    def print_classification_report(
        self, 
        model_name: str, 
        metrics: Dict[str, Any]
    ) -> None:
        """Print detailed classification report for a model"""
        print(f"\n{'='*60}")
        print(f"📊 CLASSIFICATION REPORT: {model_name.upper()}")
        print(f"{'='*60}")
        
        report = metrics['classification_report']
        
        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)
        
        for label in ['0', '1']:
            if label in report:
                print(
                    f"{('Unsuccessful' if label == '0' else 'Successful'):<15} "
                    f"{report[label]['precision']:<12.3f} "
                    f"{report[label]['recall']:<12.3f} "
                    f"{report[label]['f1-score']:<12.3f} "
                    f"{report[label]['support']:<10.0f}"
                )
        
        print("-" * 60)
        print(f"{'Accuracy':<15} {'':<12} {'':<12} {metrics['accuracy']:<12.3f} "
              f"{report['macro avg']['support']:<10.0f}")
        print(f"{'Macro Avg':<15} "
              f"{report['macro avg']['precision']:<12.3f} "
              f"{report['macro avg']['recall']:<12.3f} "
              f"{report['macro avg']['f1-score']:<12.3f} "
              f"{report['macro avg']['support']:<10.0f}")
        print(f"{'Weighted Avg':<15} "
              f"{report['weighted avg']['precision']:<12.3f} "
              f"{report['weighted avg']['recall']:<12.3f} "
              f"{report['weighted avg']['f1-score']:<12.3f} "
              f"{report['weighted avg']['support']:<10.0f}")
        
        print(f"\n🎯 ROC AUC Score: {metrics['roc_auc']:.3f}")
    
    def print_confusion_matrix(
        self, 
        model_name: str, 
        metrics: Dict[str, Any]
    ) -> None:
        """Print confusion matrix in readable format"""
        cm = metrics['confusion_matrix']
        
        print(f"\n📈 CONFUSION MATRIX: {model_name.upper()}")
        print("-" * 40)
        print(f"{'':>20} {'Predicted':>20}")
        print(f"{'':>20} {'Unsuccessful':>12} {'Successful':>12}")
        print(f"{'Actual':<15} {'Unsuccessful':<12} {cm[0][0]:>12} {cm[0][1]:>12}")
        print(f"{'':>15} {'Successful':<12} {cm[1][0]:>12} {cm[1][1]:>12}")
        
        # Calculate percentages
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        print(f"\n📊 Breakdown:")
        print(f"   True Negatives:  {tn:>5} ({tn/total*100:>5.1f}%)")
        print(f"   False Positives: {fp:>5} ({fp/total*100:>5.1f}%)")
        print(f"   False Negatives: {fn:>5} ({fn/total*100:>5.1f}%)")
        print(f"   True Positives:  {tp:>5} ({tp/total*100:>5.1f}%)")
    
    def generate_performance_summary(
        self, 
        models_metrics: Dict[str, Dict]
    ) -> None:
        """Generate comprehensive performance summary for all models"""
        print("\n" + "="*70)
        print("🤖 MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        # Comparison table
        comparison_df = self.compare_models(models_metrics)
        print("\n📊 MODEL COMPARISON:")
        print(comparison_df.to_string(index=False))
        
        # Best model
        best_name, best_metrics = self.get_best_model(models_metrics)
        print(f"\n🏆 BEST MODEL: {best_name.upper()}")
        print(f"   ROC AUC Score: {best_metrics['roc_auc']:.3f}")
        print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
        print(f"   F1 Score: {best_metrics['f1_score']:.3f}")
        
        # Detailed reports for each model
        for model_name, metrics in models_metrics.items():
            self.print_classification_report(model_name, metrics)
            self.print_confusion_matrix(model_name, metrics)
    
    def plot_roc_curves(
        self, 
        models_metrics: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curves for all models
        
        Args:
            models_metrics: Dictionary of model metrics
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in models_metrics.items():
            y_true = metrics['predictions']['y_true']
            y_pred_proba = metrics['predictions']['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = metrics['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ ROC curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curves(
        self, 
        models_metrics: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> None:
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, metrics in models_metrics.items():
            y_true = metrics['predictions']['y_true']
            y_pred_proba = metrics['predictions']['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Precision-Recall curve saved to {save_path}")
        
        plt.tight_layout()
        plt.show()


class FeatureImportanceAnalyzer:
    """Analyzes and visualizes feature importance"""
    
    def __init__(self):
        self.importance_data = {}
    
    def analyze_tree_based_model(
        self, 
        model, 
        feature_names: list,
        model_name: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models
        
        Args:
            model: Trained tree-based model
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            
        Returns:
            DataFrame with feature importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  {model_name} does not have feature_importances_ attribute")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.importance_data[model_name] = importance_df
        
        # Print top features
        print(f"\n🔍 TOP {top_n} FEATURES - {model_name.upper()}")
        print("-" * 50)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"   {row['feature']:<30} {row['importance']:>8.4f}")
        
        return importance_df
    
    def analyze_linear_model(
        self, 
        model, 
        feature_names: list,
        model_name: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract feature importance from linear models using coefficients
        
        Args:
            model: Trained linear model (with coefficients)
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            
        Returns:
            DataFrame with coefficient information
        """
        # Handle pipeline models
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        if not hasattr(classifier, 'coef_'):
            print(f"⚠️  {model_name} does not have coef_ attribute")
            return None
        
        coef = classifier.coef_[0]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        }).sort_values('abs_coefficient', ascending=False)
        
        self.importance_data[model_name] = importance_df
        
        # Print top features
        print(f"\n🔍 TOP {top_n} FEATURES - {model_name.upper()}")
        print("-" * 60)
        print(f"{'Feature':<30} {'Coefficient':>15} {'Abs Value':>15}")
        print("-" * 60)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<30} {row['coefficient']:>15.4f} {row['abs_coefficient']:>15.4f}")
        
        return importance_df
    
    def plot_feature_importance(
        self, 
        model_name: str, 
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance for a specific model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to plot
            save_path: Optional path to save the plot
        """
        if model_name not in self.importance_data:
            print(f"⚠️  No importance data found for {model_name}")
            return
        
        df = self.importance_data[model_name].head(top_n)
        
        plt.figure(figsize=(10, 8))
        
        # Determine which column to use
        value_col = 'importance' if 'importance' in df.columns else 'abs_coefficient'
        
        plt.barh(range(len(df)), df[value_col], color='steelblue')
        plt.yticks(range(len(df)), df['feature'])
        plt.xlabel('Importance Score' if value_col == 'importance' else 'Absolute Coefficient', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # This would typically be called from your main training script
    print("📊 Evaluation Module - Use this in your training pipeline")