"""
Unit tests for predictor module

Run with:
    pytest tests/test_student_predictor.py -v

be advised that the student id's here are synthetic and do not correspond to real students in the db.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import joblib
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml.student_predictor import StudentSuccessPredictor


class TestStudentSuccessPredictor:
    """Test suite for StudentSuccessPredictor"""
    
    @pytest.fixture
    def sample_student_data(self):
        """Sample student data for predictions"""
        return {
            'student_id': 'S12345',
            'gpa': 3.2,
            'year_level': 2,
            'total_enrollments': 10,
            'completed_courses': 8,
            'dropped_courses': 1,
            'current_enrollments': 1,
            'excellent_grades': 4,
            'good_grades': 3,
            'average_grades': 1,
            'poor_grades': 0,
            'avg_course_credits': 3.0,
            'total_credits_attempted': 30,
            'days_enrolled': 600,
            'semesters_enrolled': 5.0,
            'completion_rate': 80.0,
            'drop_rate': 10.0,
            'excellence_rate': 40.0,
            'good_grade_rate': 30.0,
            'struggle_rate': 0.0,
            'courses_per_semester': 2.0,
            'credits_per_semester': 6.0,
            'engagement_score': 0.75,
            'grade_diversity': 0.6,
            'high_performer': 1,
            'at_risk': 0,
            'major_encoded': 0
        }
    
    @pytest.fixture
    def mock_model_dir(self, tmp_path):
        """Create temporary directory with mock models"""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        # Create a simple mock model
        mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_train = np.random.rand(100, 25)
        y_train = np.random.randint(0, 2, 100)
        mock_model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(mock_model, model_dir / "random_forest.pkl")
        
        # Save feature names
        feature_names = [
            'gpa', 'year_level', 'total_enrollments', 'completed_courses',
            'dropped_courses', 'current_enrollments', 'excellent_grades',
            'good_grades', 'average_grades', 'poor_grades', 'avg_course_credits',
            'total_credits_attempted', 'days_enrolled', 'semesters_enrolled',
            'completion_rate', 'drop_rate', 'excellence_rate', 'good_grade_rate',
            'struggle_rate', 'courses_per_semester', 'credits_per_semester',
            'engagement_score', 'grade_diversity', 'high_performer', 'at_risk'
        ]
        joblib.dump(feature_names, model_dir / "feature_names.pkl")
        
        # Save model metrics
        model_metrics = {
            'random_forest': {
                'roc_auc': 0.85,
                'accuracy': 0.80
            }
        }
        joblib.dump(model_metrics, model_dir / "model_metrics.pkl")
        
        # Save encoders
        encoders = {}
        joblib.dump(encoders, model_dir / "encoders.pkl")
        
        # Save feature descriptions
        feature_descriptions = {
            'gpa': 'Student GPA',
            'completion_rate': 'Course completion rate'
        }
        joblib.dump(feature_descriptions, model_dir / "feature_descriptions.pkl")
        
        return str(model_dir)
    
    @pytest.fixture
    def predictor(self, mock_model_dir):
        """Create predictor with mock models loaded"""
        pred = StudentSuccessPredictor(model_dir=mock_model_dir)
        pred.load_models()
        return pred
    
    def test_initialization(self):
        """Test predictor initialization"""
        predictor = StudentSuccessPredictor(model_dir='models/')
        
        assert predictor.model_dir == 'models/'
        assert predictor.models == {}
        assert predictor.feature_names == []
    
    def test_load_models(self, mock_model_dir):
        """Test model loading"""
        predictor = StudentSuccessPredictor(model_dir=mock_model_dir)
        predictor.load_models()
        
        # Check models were loaded
        assert 'random_forest' in predictor.models
        assert len(predictor.feature_names) > 0
        assert predictor.best_model_name is not None
    
    def test_load_models_missing_directory(self):
        """Test error handling for missing model directory"""
        predictor = StudentSuccessPredictor(model_dir='nonexistent_dir/')
        
        with pytest.raises(FileNotFoundError):
            predictor.load_models()
    
    def test_prepare_features(self, predictor, sample_student_data):
        """Test feature preparation"""
        features = predictor.prepare_features(sample_student_data)
        
        # Check shape
        assert features.shape[0] == 1  # Single sample
        assert features.shape[1] == len(predictor.feature_names)
        
        # Check it's a numpy array
        assert isinstance(features, np.ndarray)
    
    def test_prepare_features_missing_values(self, predictor):
        """Test feature preparation with missing values"""
        incomplete_data = {
            'gpa': 3.0,
            'year_level': 2
            # Missing other features
        }
        
        features = predictor.prepare_features(incomplete_data)
        
        # Should still return array with default values
        assert features.shape[1] == len(predictor.feature_names)
    
    def test_predict(self, predictor, sample_student_data):
        """Test prediction on single student"""
        result = predictor.predict(sample_student_data)
        
        # Check result structure
        assert 'student_id' in result
        assert 'predicted_success' in result
        assert 'success_probability' in result
        assert 'risk_level' in result
        assert 'recommendations' in result
        assert 'model_used' in result
        
        # Check data types
        assert isinstance(result['predicted_success'], bool)
        assert isinstance(result['success_probability'], float)
        assert 0 <= result['success_probability'] <= 1
        assert isinstance(result['recommendations'], list)
    
    def test_predict_high_risk_student(self, predictor):
        """Test prediction for high-risk student"""
        high_risk_student = {
            'student_id': 'S99999',
            'gpa': 1.8,
            'completion_rate': 40.0,
            'drop_rate': 40.0,
            'struggle_rate': 50.0,
        }
        
        result = predictor.predict(high_risk_student)
        
        # Should have recommendations
        assert len(result['recommendations']) > 0
        
        # Risk level should indicate concern
        assert 'High' in result['risk_level'] or 'Critical' in result['risk_level']
    
    def test_predict_with_specific_model(self, predictor, sample_student_data):
        """Test prediction with specific model"""
        result = predictor.predict(sample_student_data, model_name='random_forest')
        
        assert result['model_used'] == 'random_forest'
    
    def test_predict_invalid_model(self, predictor, sample_student_data):
        """Test error handling for invalid model name"""
        with pytest.raises(ValueError):
            predictor.predict(sample_student_data, model_name='nonexistent_model')
    
    def test_predict_batch(self, predictor, sample_student_data):
        """Test batch predictions"""
        # Create multiple students
        students = [sample_student_data.copy() for _ in range(3)]
        for i, student in enumerate(students):
            student['student_id'] = f'S{i}'
        
        results = predictor.predict_batch(students)
        
        # Check result is DataFrame
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        
        # Check all required columns exist
        assert 'student_id' in results.columns
        assert 'success_probability' in results.columns
    
    def test_assess_risk_low(self, predictor, sample_student_data):
        """Test risk assessment for low-risk student"""
        assessment = predictor._assess_risk(0.85, sample_student_data)
        
        assert 'Low' in assessment['risk_level']
        assert assessment['confidence'] in ['High', 'Very High']
        assert isinstance(assessment['recommendations'], list)
    
    def test_assess_risk_medium(self, predictor, sample_student_data):
        """Test risk assessment for medium-risk student"""
        assessment = predictor._assess_risk(0.65, sample_student_data)
        
        assert 'Medium' in assessment['risk_level']
        assert isinstance(assessment['recommendations'], list)
    
    def test_assess_risk_high(self, predictor, sample_student_data):
        """Test risk assessment for high-risk student"""
        assessment = predictor._assess_risk(0.35, sample_student_data)
        
        assert 'High' in assessment['risk_level'] or 'Critical' in assessment['risk_level']
        assert len(assessment['recommendations']) > 0
    
    def test_assess_risk_critical(self, predictor, sample_student_data):
        """Test risk assessment for critical-risk student"""
        assessment = predictor._assess_risk(0.15, sample_student_data)
        
        assert 'Critical' in assessment['risk_level']
        # Should have urgent recommendations
        urgent_keywords = ['URGENT', 'immediate', 'Critical']
        has_urgent = any(
            any(keyword.lower() in rec.lower() for keyword in urgent_keywords)
            for rec in assessment['recommendations']
        )
        assert has_urgent
    
    def test_assess_risk_with_high_drop_rate(self, predictor):
        """Test that high drop rate triggers specific recommendations"""
        student_data = {'drop_rate': 25.0}
        assessment = predictor._assess_risk(0.70, student_data)
        
        # Should mention drop rate
        drop_mentioned = any(
            'drop' in rec.lower() 
            for rec in assessment['recommendations']
        )
        assert drop_mentioned
    
    def test_assess_risk_with_low_gpa(self, predictor):
        """Test that low GPA triggers specific recommendations"""
        student_data = {'gpa': 1.8}
        assessment = predictor._assess_risk(0.70, student_data)
        
        # Should mention GPA
        gpa_mentioned = any(
            'gpa' in rec.lower() 
            for rec in assessment['recommendations']
        )
        assert gpa_mentioned
    
    def test_explain_prediction(self, predictor, sample_student_data):
        """Test prediction explanation"""
        explanation = predictor.explain_prediction(sample_student_data)
        
        assert 'prediction' in explanation
        assert 'key_features' in explanation
        assert 'interpretation' in explanation
        
        # Check key features structure
        assert isinstance(explanation['key_features'], dict)
        for feature_info in explanation['key_features'].values():
            assert 'value' in feature_info
            assert 'description' in feature_info
    
    def test_get_available_models(self, predictor):
        """Test getting available models"""
        models = predictor.get_available_models()
        
        assert isinstance(models, list)
        assert 'random_forest' in models
    
    def test_get_required_features(self, predictor):
        """Test getting required features"""
        features = predictor.get_required_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert 'gpa' in features


class TestPredictorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_prediction_with_nan_values(self, mock_model_dir):
        """Test handling of NaN values in input"""
        predictor = StudentSuccessPredictor(model_dir=mock_model_dir)
        predictor.load_models()
        
        student_data = {
            'gpa': np.nan,
            'completion_rate': 75.0
        }
        
        # Should handle NaN by using defaults
        result = predictor.predict(student_data)
        assert 'success_probability' in result
    
    def test_prediction_boundary_probabilities(self, mock_model_dir):
        """Test predictions at probability boundaries"""
        predictor = StudentSuccessPredictor(model_dir=mock_model_dir)
        predictor.load_models()
        
        # Test at different probability thresholds
        for prob in [0.0, 0.4, 0.6, 0.8, 1.0]:
            student_data = {'completion_rate': prob * 100}
            assessment = predictor._assess_risk(prob, student_data)
            
            assert 'risk_level' in assessment
            assert 'confidence' in assessment
    
    def test_empty_student_data(self, mock_model_dir):
        """Test with minimal student data"""
        predictor = StudentSuccessPredictor(model_dir=mock_model_dir)
        predictor.load_models()
        
        minimal_data = {}
        
        # Should still make prediction with defaults
        result = predictor.predict(minimal_data)
        assert 'success_probability' in result


# Integration tests
class TestPredictorIntegration:
    """Integration tests for predictor"""
    
    @pytest.mark.integration
    def test_full_prediction_workflow(self, mock_model_dir, sample_student_data):
        """Test complete prediction workflow"""
        # Initialize
        predictor = StudentSuccessPredictor(model_dir=mock_model_dir)
        
        # Load models
        predictor.load_models()
        
        # Make prediction
        result = predictor.predict(sample_student_data)
        
        # Verify complete result
        assert result['student_id'] == 'S12345'
        assert isinstance(result['predicted_success'], bool)
        assert 0 <= result['success_probability'] <= 1
        assert len(result['recommendations']) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])