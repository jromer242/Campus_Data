"""
conftest.py - Shared pytest fixtures and configuration

This file is automatically loaded by pytest and provides fixtures
that can be used across all test files.
"""

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def sample_students_data():
    """Sample student data for testing"""
    return pd.DataFrame({
        'student_id': list(range(1, 51)),
        'gpa': np.random.uniform(2.0, 4.0, 50),
        'year_level': np.random.randint(1, 5, 50),
        'major': np.random.choice(['CS', 'Math', 'Physics', 'Biology'], 50),
        'is_active': [1] * 50,
        'enrollment_date': ['2022-01-01'] * 50,
        'total_enrollments': np.random.randint(5, 20, 50),
        'completed_courses': np.random.randint(3, 18, 50),
        'dropped_courses': np.random.randint(0, 5, 50),
        'current_enrollments': np.random.randint(0, 4, 50),
        'excellent_grades': np.random.randint(0, 10, 50),
        'good_grades': np.random.randint(0, 8, 50),
        'average_grades': np.random.randint(0, 5, 50),
        'poor_grades': np.random.randint(0, 3, 50),
        'avg_course_credits': [3.0] * 50,
        'total_credits_attempted': np.random.randint(15, 60, 50),
        'days_enrolled': np.random.randint(200, 1500, 50)
    })


@pytest.fixture(scope='function')
def temp_database(sample_students_data):
    """
    Create temporary SQLite database with test data
    Scope: function (new database for each test)
    """
    # Create temporary file
    temp_db = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    
    # Create students table
    sample_students_data.to_sql('students', conn, if_exists='replace', index=False)
    
    # Create enrollments table
    enrollments_data = []
    for student_id in range(1, 51):
        num_enrollments = np.random.randint(5, 20)
        for _ in range(num_enrollments):
            enrollments_data.append({
                'enrollment_id': len(enrollments_data) + 1,
                'student_id': student_id,
                'course_id': np.random.randint(1, 100),
                'status': np.random.choice(['Completed', 'Enrolled', 'Dropped'], p=[0.7, 0.2, 0.1]),
                'grade': np.random.choice(['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F', None])
            })
    
    enrollments_df = pd.DataFrame(enrollments_data)
    enrollments_df.to_sql('enrollments', conn, if_exists='replace', index=False)
    
    # Create courses table
    courses_df = pd.DataFrame({
        'course_id': range(1, 101),
        'credits': np.random.choice([3, 4], 100)
    })
    courses_df.to_sql('courses', conn, if_exists='replace', index=False)
    
    conn.close()
    
    yield temp_db.name
    
    # Cleanup
    try:
        os.unlink(temp_db.name)
    except:
        pass


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def feature_names():
    """Standard feature names used across tests"""
    return [
        'gpa', 'year_level', 'total_enrollments', 'completed_courses',
        'dropped_courses', 'current_enrollments', 'excellent_grades',
        'good_grades', 'average_grades', 'poor_grades', 'avg_course_credits',
        'total_credits_attempted', 'days_enrolled', 'semesters_enrolled',
        'completion_rate', 'drop_rate', 'excellence_rate', 'good_grade_rate',
        'struggle_rate', 'courses_per_semester', 'credits_per_semester',
        'engagement_score', 'grade_diversity', 'high_performer', 'at_risk'
    ]


@pytest.fixture(scope='session')
def trained_models(feature_names):
    """Create trained mock models for testing"""
    models = {}
    
    # Create dummy training data
    n_samples = 100
    n_features = len(feature_names)
    X_train = np.random.rand(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    return models


@pytest.fixture(scope='function')
def temp_model_directory(trained_models, feature_names, tmp_path):
    """
    Create temporary directory with saved models
    Scope: function (new directory for each test)
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Save models
    for model_name, model in trained_models.items():
        joblib.dump(model, model_dir / f"{model_name}.pkl")
    
    # Save metadata
    joblib.dump(feature_names, model_dir / "feature_names.pkl")
    
    model_metrics = {
        'random_forest': {'roc_auc': 0.85, 'accuracy': 0.80},
        'logistic_regression': {'roc_auc': 0.82, 'accuracy': 0.78}
    }
    joblib.dump(model_metrics, model_dir / "model_metrics.pkl")
    
    joblib.dump({}, model_dir / "encoders.pkl")
    
    feature_descriptions = {name: f"Description of {name}" for name in feature_names[:10]}
    joblib.dump(feature_descriptions, model_dir / "feature_descriptions.pkl")
    
    return str(model_dir)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_student():
    """Single student data for testing predictions"""
    return {
        'student_id': 'TEST_001',
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
def high_risk_student():
    """High-risk student data for testing"""
    return {
        'student_id': 'RISK_001',
        'gpa': 1.8,
        'year_level': 3,
        'total_enrollments': 12,
        'completed_courses': 5,
        'dropped_courses': 5,
        'excellent_grades': 0,
        'good_grades': 2,
        'average_grades': 3,
        'poor_grades': 5,
        'completion_rate': 41.7,
        'drop_rate': 41.7,
        'struggle_rate': 41.7,
        'at_risk': 1
    }


@pytest.fixture
def excellent_student():
    """Excellent student data for testing"""
    return {
        'student_id': 'EXCELLENT_001',
        'gpa': 3.95,
        'year_level': 4,
        'total_enrollments': 15,
        'completed_courses': 15,
        'dropped_courses': 0,
        'excellent_grades': 12,
        'good_grades': 3,
        'average_grades': 0,
        'poor_grades': 0,
        'completion_rate': 100.0,
        'drop_rate': 0.0,
        'excellence_rate': 80.0,
        'struggle_rate': 0.0,
        'high_performer': 1,
        'at_risk': 0
    }


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_sklearn_model():
    """Mock sklearn model for testing"""
    from unittest.mock import Mock
    
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.feature_importances_ = np.random.rand(25)
    
    return model


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection
    Add markers automatically based on test names
    """
    for item in items:
        # Mark slow tests
        if 'slow' in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if 'integration' in item.nodeid or 'Integration' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default)
        else:
            item.add_marker(pytest.mark.unit)


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def assert_valid_probability(value):
    """Assert value is a valid probability"""
    assert isinstance(value, (int, float))
    assert 0 <= value <= 1


def assert_dataframe_structure(df, required_columns):
    """Assert DataFrame has required structure"""
    assert isinstance(df, pd.DataFrame)
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0


def create_mock_prediction_result():
    """Create a mock prediction result for testing"""
    return {
        'student_id': 'TEST',
        'predicted_success': True,
        'success_probability': 0.75,
        'risk_level': '🟢 Low',
        'confidence': 'High',
        'recommendations': ['Continue current path'],
        'model_used': 'random_forest'
    }