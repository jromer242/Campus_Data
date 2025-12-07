"""
Unit tests for feature engineering module

Run with:
    pytest tests/test_feature_engineering.py -v
    pytest tests/test_feature_engineering.py::TestFeatureEngineer::test_create_target_variable -v
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import the module to test
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'student_id': [1, 2, 3, 4, 5],
            'gpa': [3.5, 2.8, 3.9, 2.2, 3.1],
            'year_level': [2, 3, 4, 1, 2],
            'major': ['CS', 'Math', 'CS', 'Physics', 'Math'],
            'is_active': [1, 1, 1, 1, 1],
            'enrollment_date': ['2022-01-01'] * 5,
            'total_enrollments': [10, 12, 15, 8, 11],
            'completed_courses': [8, 8, 14, 4, 9],
            'dropped_courses': [1, 2, 0, 3, 1],
            'current_enrollments': [1, 2, 1, 1, 1],
            'excellent_grades': [5, 2, 10, 1, 4],
            'good_grades': [2, 4, 3, 2, 3],
            'average_grades': [1, 2, 1, 1, 2],
            'poor_grades': [0, 2, 0, 1, 1],
            'avg_course_credits': [3.0, 3.0, 3.0, 3.0, 3.0],
            'total_credits_attempted': [30, 36, 45, 24, 33],
            'days_enrolled': [700, 1000, 1400, 300, 800]
        })
    
    @pytest.fixture
    def temp_database(self, sample_dataframe):
        """Create temporary database for testing"""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
        temp_db.close()
        
        # Populate with test data
        conn = sqlite3.connect(temp_db.name)
        
        # Create students table
        sample_dataframe.to_sql('students', conn, if_exists='replace', index=False)
        
        # Create mock enrollments table
        enrollments = pd.DataFrame({
            'enrollment_id': range(1, 51),
            'student_id': [1]*10 + [2]*12 + [3]*15 + [4]*8 + [5]*6,
            'course_id': range(1, 51),
            'status': ['Completed']*40 + ['Dropped']*10,
            'grade': ['A']*10 + ['B']*15 + ['C']*10 + ['D']*5
        })
        enrollments.to_sql('enrollments', conn, if_exists='replace', index=False)
        
        # Create courses table
        courses = pd.DataFrame({
            'course_id': range(1, 51),
            'credits': [3]*50
        })
        courses.to_sql('courses', conn, if_exists='replace', index=False)
        
        conn.close()
        
        yield temp_db.name
        
        # Cleanup
        os.unlink(temp_db.name)
    
    @pytest.fixture
    def feature_engineer(self, temp_database):
        """Create FeatureEngineer instance with temp database"""
        return FeatureEngineer(db_path=temp_database)
    
    def test_initialization(self, temp_database):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(db_path=temp_database)
        
        assert engineer.db_path == temp_database
        assert engineer.encoders == {}
        assert engineer.feature_names == []
    
    def test_load_data_from_db(self, feature_engineer):
        """Test data loading from database"""
        df = feature_engineer.load_data_from_db()
        
        # Check that data was loaded
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns exist
        required_columns = [
            'student_id', 'gpa', 'year_level', 'major',
            'total_enrollments', 'completed_courses'
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_create_target_variable(self, feature_engineer, sample_dataframe):
        """Test target variable creation"""
        df = feature_engineer.create_target_variable(sample_dataframe)
        
        # Check that target column exists
        assert 'is_successful' in df.columns
        assert 'completion_rate' in df.columns
        
        # Check target is binary
        assert df['is_successful'].isin([0, 1]).all()
        
        # Verify logic: GPA >= 3.0 AND completion_rate >= 70%
        for idx, row in df.iterrows():
            expected = (row['gpa'] >= 3.0) and (row['completion_rate'] >= 70)
            assert row['is_successful'] == int(expected)
    
    def test_engineer_features(self, feature_engineer, sample_dataframe):
        """Test feature engineering"""
        # First create target variable
        df = feature_engineer.create_target_variable(sample_dataframe)
        
        # Then engineer features
        df = feature_engineer.engineer_features(df)
        
        # Check that new features were created
        expected_features = [
            'drop_rate', 'excellence_rate', 'good_grade_rate',
            'struggle_rate', 'courses_per_semester', 'engagement_score'
        ]
        
        for feature in expected_features:
            assert feature in df.columns, f"Missing feature: {feature}"
        
        # Check no NaN or infinite values
        assert not df.isnull().any().any(), "Found NaN values"
        assert not np.isinf(df.select_dtypes(include=[np.number])).any().any(), "Found infinite values"
    
    def test_encode_categorical_features(self, feature_engineer, sample_dataframe):
        """Test categorical encoding"""
        df = feature_engineer.encode_categorical_features(sample_dataframe)
        
        # Check that major was encoded
        assert 'major_encoded' in df.columns
        assert 'major' in feature_engineer.encoders
        
        # Check encoding is numeric
        assert df['major_encoded'].dtype in [np.int32, np.int64]
        
        # Check encoding is consistent
        assert len(df['major_encoded'].unique()) <= len(df['major'].unique())
    
    def test_get_feature_columns(self, feature_engineer):
        """Test feature column retrieval"""
        feature_columns = feature_engineer.get_feature_columns()
        
        assert isinstance(feature_columns, list)
        assert len(feature_columns) > 0
        assert 'gpa' in feature_columns
        assert 'major_encoded' in feature_columns
    
    def test_prepare_ml_data(self, feature_engineer, sample_dataframe):
        """Test ML data preparation"""
        # Prepare data first
        df = feature_engineer.create_target_variable(sample_dataframe)
        df = feature_engineer.engineer_features(df)
        df = feature_engineer.encode_categorical_features(df)
        
        # Prepare for ML
        X, y, feature_names = feature_engineer.prepare_ml_data(df)
        
        # Check shapes
        assert len(X) == len(y)
        assert len(X.columns) == len(feature_names)
        
        # Check X contains only features
        assert 'is_successful' not in X.columns
        assert 'student_id' not in X.columns
        
        # Check y is binary
        assert y.isin([0, 1]).all()
    
    def test_full_pipeline(self, feature_engineer):
        """Test complete pipeline"""
        X, y, feature_names = feature_engineer.full_pipeline()
        
        # Check outputs
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(feature_names, list)
        
        # Check consistency
        assert len(X) == len(y)
        assert len(X.columns) == len(feature_names)
        
        # Check no missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()
    
    def test_feature_descriptions(self, feature_engineer):
        """Test feature descriptions"""
        descriptions = feature_engineer.get_feature_descriptions()
        
        assert isinstance(descriptions, dict)
        assert 'gpa' in descriptions
        assert 'completion_rate' in descriptions
        
        # Check all descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
    
    def test_handles_missing_database(self):
        """Test handling of missing database"""
        engineer = FeatureEngineer(db_path='nonexistent.db')
        
        with pytest.raises(Exception):
            engineer.load_data_from_db()
    
    def test_completion_rate_calculation(self, feature_engineer, sample_dataframe):
        """Test completion rate calculation accuracy"""
        df = feature_engineer.create_target_variable(sample_dataframe)
        
        for idx, row in df.iterrows():
            expected_rate = (row['completed_courses'] / row['total_enrollments']) * 100
            assert abs(row['completion_rate'] - expected_rate) < 0.01
    
    def test_drop_rate_calculation(self, feature_engineer, sample_dataframe):
        """Test drop rate calculation"""
        df = feature_engineer.create_target_variable(sample_dataframe)
        df = feature_engineer.engineer_features(df)
        
        for idx, row in df.iterrows():
            expected_rate = (row['dropped_courses'] / row['total_enrollments']) * 100
            assert abs(row['drop_rate'] - expected_rate) < 0.01


class TestFeatureEngineeringEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_enrollments(self):
        """Test handling of students with zero enrollments"""
        df = pd.DataFrame({
            'student_id': [1],
            'total_enrollments': [0],
            'completed_courses': [0],
            'dropped_courses': [0],
        })
        
        engineer = FeatureEngineer()
        # Should handle division by zero
        df_result = engineer.create_target_variable(df)
        
        assert not np.isnan(df_result['completion_rate']).any()
    
    def test_all_dropped_courses(self):
        """Test student who dropped all courses"""
        df = pd.DataFrame({
            'student_id': [1],
            'gpa': [2.0],
            'total_enrollments': [5],
            'completed_courses': [0],
            'dropped_courses': [5],
        })
        
        engineer = FeatureEngineer()
        df_result = engineer.create_target_variable(df)
        
        assert df_result['completion_rate'].iloc[0] == 0
        assert df_result['is_successful'].iloc[0] == 0
    
    def test_perfect_student(self):
        """Test student with perfect record"""
        df = pd.DataFrame({
            'student_id': [1],
            'gpa': [4.0],
            'total_enrollments': [10],
            'completed_courses': [10],
            'dropped_courses': [0],
            'excellent_grades': [10],
            'good_grades': [0],
            'average_grades': [0],
            'poor_grades': [0],
        })
        
        engineer = FeatureEngineer()
        df = engineer.create_target_variable(df)
        df = engineer.engineer_features(df)
        
        assert df['completion_rate'].iloc[0] == 100
        assert df['excellence_rate'].iloc[0] == 100
        assert df['is_successful'].iloc[0] == 1


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v'])