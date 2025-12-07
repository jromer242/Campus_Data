"""
Student Success Predictor

This module handles loading trained models and making predictions
for new students. Used for production inference.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import os
import warnings
warnings.filterwarnings('ignore')


class StudentSuccessPredictor:
    """
    Production predictor for student success
    
    Usage:
        predictor = StudentSuccessPredictor()
        predictor.load_models('models/')
        prediction = predictor.predict(student_data)
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.models = {}
        self.encoders = {}
        self.feature_names = []
        self.model_metrics = {}
        self.feature_descriptions = {}
        self.best_model_name = None
        
    def load_models(self, model_dir: Optional[str] = None) -> None:
        """
        Load all trained models and metadata from disk
        
        Args:
            model_dir: Directory containing saved models (optional override)
        """
        if model_dir:
            self.model_dir = model_dir
            
        print(f"📂 Loading models from {self.model_dir}/")
        
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load all model files
        for file in os.listdir(self.model_dir):
            if file.endswith('.pkl'):
                filepath = os.path.join(self.model_dir, file)
                
                if file == 'encoders.pkl':
                    self.encoders = joblib.load(filepath)
                elif file == 'feature_names.pkl':
                    self.feature_names = joblib.load(filepath)
                elif file == 'model_metrics.pkl':
                    self.model_metrics = joblib.load(filepath)
                elif file == 'feature_descriptions.pkl':
                    self.feature_descriptions = joblib.load(filepath)
                elif file not in ['best_params.pkl', 'cv_scores.pkl']:
                    # Load actual model
                    model_name = file.replace('.pkl', '')
                    self.models[model_name] = joblib.load(filepath)
                    print(f"   ✅ Loaded {model_name}")
        
        # Determine best model
        if self.model_metrics:
            self.best_model_name = max(
                self.model_metrics.keys(),
                key=lambda x: self.model_metrics[x].get('roc_auc', 0)
            )
            print(f"\n🏆 Best model: {self.best_model_name}")
            print(f"   ROC AUC: {self.model_metrics[self.best_model_name]['roc_auc']:.3f}")
        
        print(f"✅ Models loaded successfully")
    
    def prepare_features(self, student_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features from raw student data
        
        Args:
            student_data: Dictionary of student attributes
            
        Returns:
            Feature array ready for prediction
        """
        # Ensure all required features are present
        features = []
        
        for feature_name in self.feature_names:
            if feature_name in student_data:
                features.append(student_data[feature_name])
            else:
                # Handle missing features
                if feature_name == 'major_encoded' and 'major' in student_data:
                    # Encode major if encoder is available
                    if 'major' in self.encoders:
                        encoded = self.encoders['major'].transform([student_data['major']])[0]
                        features.append(encoded)
                    else:
                        features.append(0)  # Default encoding
                else:
                    features.append(0)  # Default value for missing features
        
        return np.array(features).reshape(1, -1)
    
    def predict(
        self, 
        student_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction for a single student
        
        Args:
            student_data: Dictionary containing student features
            model_name: Specific model to use (None = use best model)
            
        Returns:
            Dictionary with prediction results and recommendations
        """
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Prepare features
        features = self.prepare_features(student_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        success_probability = probability[1]
        
        # Generate risk assessment
        risk_assessment = self._assess_risk(success_probability, student_data)
        
        return {
            'student_id': student_data.get('student_id', 'Unknown'),
            'predicted_success': bool(prediction),
            'success_probability': float(success_probability),
            'failure_probability': float(probability[0]),
            'risk_level': risk_assessment['risk_level'],
            'confidence': risk_assessment['confidence'],
            'recommendations': risk_assessment['recommendations'],
            'model_used': model_name,
            'model_performance': {
                'roc_auc': self.model_metrics.get(model_name, {}).get('roc_auc', None),
                'accuracy': self.model_metrics.get(model_name, {}).get('accuracy', None)
            }
        }
    
    def predict_batch(
        self, 
        students_data: List[Dict[str, Any]],
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions for multiple students
        
        Args:
            students_data: List of student data dictionaries
            model_name: Specific model to use
            
        Returns:
            DataFrame with predictions for all students
        """
        results = []
        
        for student_data in students_data:
            prediction = self.predict(student_data, model_name)
            results.append(prediction)
        
        return pd.DataFrame(results)
    
    def _assess_risk(
        self, 
        success_probability: float,
        student_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess risk level and generate recommendations
        
        Args:
            success_probability: Predicted probability of success
            student_data: Student attributes for context
            
        Returns:
            Dictionary with risk assessment
        """
        # Determine risk level
        if success_probability >= 0.8:
            risk_level = "Low"
            color = "🟢"
            recommendations = [
                "Continue current academic trajectory",
                "Consider taking on leadership roles or advanced coursework",
                "Explore research or internship opportunities",
                "Mentor other students who may be struggling"
            ]
        elif success_probability >= 0.6:
            risk_level = "Medium"
            color = "🟡"
            recommendations = [
                "Schedule quarterly check-ins with academic advisor",
                "Join study groups for challenging courses",
                "Consider tutoring for specific subjects if needed",
                "Monitor course load and adjust if feeling overwhelmed",
                "Utilize campus academic support resources"
            ]
        elif success_probability >= 0.4:
            risk_level = "High"
            color = "🟠"
            recommendations = [
                "Schedule immediate meeting with academic advisor",
                "Enroll in tutoring services for struggling courses",
                "Consider reducing course load next semester",
                "Attend academic skills workshops",
                "Connect with student success center",
                "Explore time management resources"
            ]
        else:
            risk_level = "Critical"
            color = "🔴"
            recommendations = [
                "URGENT: Meet with academic advisor immediately",
                "Consider academic intervention programs",
                "Evaluate if academic probation support is needed",
                "Reassess major and career goals with counselor",
                "Explore comprehensive support services",
                "Consider taking lighter course load or academic break"
            ]
        
        # Determine confidence
        confidence_score = max(success_probability, 1 - success_probability)
        if confidence_score > 0.85:
            confidence = "Very High"
        elif confidence_score > 0.7:
            confidence = "High"
        elif confidence_score > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Add context-specific recommendations
        if student_data.get('drop_rate', 0) > 15:
            recommendations.insert(0, "⚠️ High drop rate detected - investigate reasons for course withdrawals")
        
        if student_data.get('gpa', 4.0) < 2.0:
            recommendations.insert(0, "⚠️ GPA below 2.0 - immediate academic intervention recommended")
        
        return {
            'risk_level': f"{color} {risk_level}",
            'confidence': confidence,
            'recommendations': recommendations,
            'confidence_score': float(confidence_score)
        }
    
    def explain_prediction(
        self,
        student_data: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Provide explanation for a prediction
        
        Args:
            student_data: Student data
            model_name: Model to use
            
        Returns:
            Dictionary with explanation
        """
        prediction = self.predict(student_data, model_name)
        
        # Get feature values
        feature_values = {}
        for feature in self.feature_names[:10]:  # Top 10 features
            value = student_data.get(feature, 0)
            description = self.feature_descriptions.get(feature, "No description")
            feature_values[feature] = {
                'value': value,
                'description': description
            }
        
        return {
            'prediction': prediction,
            'key_features': feature_values,
            'interpretation': self._interpret_prediction(student_data, prediction)
        }
    
    def _interpret_prediction(
        self,
        student_data: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> str:
        """Generate human-readable interpretation"""
        prob = prediction['success_probability']
        
        if prob >= 0.8:
            return f"This student shows strong indicators of academic success with a {prob:.1%} probability. Their academic performance and engagement are excellent."
        elif prob >= 0.6:
            return f"This student has a {prob:.1%} probability of success. While generally on track, some areas may benefit from additional support."
        elif prob >= 0.4:
            return f"This student has a {prob:.1%} probability of success and may be at risk. Proactive intervention is recommended."
        else:
            return f"This student has only a {prob:.1%} probability of success and is at high risk. Immediate intervention is strongly recommended."
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_required_features(self) -> List[str]:
        """Get list of required feature names"""
        return self.feature_names.copy()


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StudentSuccessPredictor()
    predictor.load_models('models/')
    
    # Example student data
    sample_student = {
        'student_id': 'S12345',
        'gpa': 2.8,
        'year_level': 3,
        'total_enrollments': 10,
        'completed_courses': 7,
        'dropped_courses': 2,
        'current_enrollments': 1,
        'excellent_grades': 2,
        'good_grades': 3,
        'average_grades': 2,
        'poor_grades': 1,
        'avg_course_credits': 3.0,
        'total_credits_attempted': 30,
        'days_enrolled': 730,
        'semesters_enrolled': 6.0,
        'completion_rate': 70.0,
        'drop_rate': 20.0,
        'excellence_rate': 20.0,
        'good_grade_rate': 30.0,
        'struggle_rate': 10.0,
        'courses_per_semester': 1.7,
        'credits_per_semester': 5.0,
        'engagement_score': 0.65,
        'grade_diversity': 0.5,
        'high_performer': 0,
        'at_risk': 1,
        'major': 'Computer Science'
    }
    
    # Make prediction
    result = predictor.predict(sample_student)
    
    print("\n" + "="*60)
    print("🔮 STUDENT SUCCESS PREDICTION")
    print("="*60)
    print(f"\nStudent ID: {result['student_id']}")
    print(f"Success Probability: {result['success_probability']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']}")
    print(f"\n📋 Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"   {i}. {rec}")