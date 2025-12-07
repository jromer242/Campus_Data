"""
Feature Engineering Module

This module handles all data loading, feature creation, 
and data preparation for ML models.
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Handles data loading and feature engineering"""
    
    def __init__(self, db_path: str = 'campus_data.db'):
        self.db_path = db_path
        self.encoders = {}
        self.feature_names = []
        
    def load_data_from_db(self) -> pd.DataFrame:
        """
        Load student data from database with enrollments and grades
        
        Returns:
            DataFrame with raw student data
        """
        print("📊 Loading data from database...")
        
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
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable: student success
        
        Success criteria: GPA >= 3.0 AND completion rate >= 70%
        
        Args:
            df: DataFrame with student data
            
        Returns:
            DataFrame with target variable added
        """
        print("🎯 Creating target variable...")
        
        # Calculate completion rate
        df['completion_rate'] = (df['completed_courses'] / df['total_enrollments'] * 100)
        
        # Define success
        df['is_successful'] = (
            (df['gpa'] >= 3.0) & 
            (df['completion_rate'] >= 70)
        ).astype(int)
        
        success_rate = df['is_successful'].mean()
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Successful students: {df['is_successful'].sum()}")
        print(f"   Unsuccessful students: {(~df['is_successful'].astype(bool)).sum()}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with engineered features
        """
        print("🔧 Engineering features...")
        
        # Rate-based features
        df['drop_rate'] = df['dropped_courses'] / df['total_enrollments'] * 100
        df['excellence_rate'] = df['excellent_grades'] / df['total_enrollments'] * 100
        df['good_grade_rate'] = df['good_grades'] / df['total_enrollments'] * 100
        df['struggle_rate'] = df['poor_grades'] / df['total_enrollments'] * 100
        
        # Academic load features
        # Approximate 120 days per semester
        df['semesters_enrolled'] = df['days_enrolled'] / 120
        df['courses_per_semester'] = df['total_enrollments'] / (df['semesters_enrolled'] + 1)
        df['credits_per_semester'] = df['total_credits_attempted'] / (df['semesters_enrolled'] + 1)
        
        # Performance indicators
        # Make sure to handle other conditions such as high GPA but low completion etc
        df['high_performer'] = ((df['gpa'] >= 3.5) & (df['excellence_rate'] >= 30)).astype(int)
        df['at_risk'] = ((df['gpa'] < 2.5) | (df['drop_rate'] > 20)).astype(int)
        
        # Grade distribution diversity (using standard deviation)
        grade_cols = ['excellent_grades', 'good_grades', 'average_grades', 'poor_grades']
        df['grade_diversity'] = df[grade_cols].std(axis=1)
        
        # Engagement score (composite metric)
        df['engagement_score'] = (
            df['total_enrollments'] / 10 * 0.3 +  # Normalized enrollment count
            df['completion_rate'] / 100 * 0.4 +   # Completion rate
            (100 - df['drop_rate']) / 100 * 0.3   # Inverse drop rate
        )
        
        # Fill NaN values that might have been created
        df = df.fillna(0)
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Created {len(df.columns)} total features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded features
        """
        print("🔤 Encoding categorical features...")
        
        # Encode major
        if 'major' in df.columns:
            le_major = LabelEncoder()
            df['major_encoded'] = le_major.fit_transform(df['major'])
            self.encoders['major'] = le_major
            print(f"   Encoded {len(le_major.classes_)} majors")
        
        # You can add more categorical encodings here if needed
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        Define which columns to use as features for ML
        
        Returns:
            List of feature column names
        """
        feature_columns = [
            # Basic features
            'gpa',
            'year_level',
            
            # Enrollment features
            'total_enrollments',
            'completed_courses',
            'dropped_courses',
            'current_enrollments',
            
            # Grade features
            'excellent_grades',
            'good_grades',
            'average_grades',
            'poor_grades',
            
            # Course load features
            'avg_course_credits',
            'total_credits_attempted',
            
            # Time features
            'days_enrolled',
            'semesters_enrolled',
            
            # Engineered rate features
            'completion_rate',
            'drop_rate',
            'excellence_rate',
            'good_grade_rate',
            'struggle_rate',
            
            # Engineered composite features
            'courses_per_semester',
            'credits_per_semester',
            'engagement_score',
            'grade_diversity',
            
            # Binary indicators
            'high_performer',
            'at_risk',
            
            # Encoded categorical
            'major_encoded'
        ]
        
        self.feature_names = feature_columns
        return feature_columns
    
    def prepare_ml_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare final X and y for machine learning
        
        Args:
            df: Full DataFrame with all features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        print("🎯 Preparing data for ML...")
        
        feature_columns = self.get_feature_columns()
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"⚠️  Warning: Missing features: {missing_features}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].copy()
        y = df['is_successful'].copy()
        
        print(f"✅ X shape: {X.shape}")
        print(f"✅ y shape: {y.shape}")
        print(f"✅ Features: {len(feature_columns)}")
        
        return X, y, feature_columns
    
    def full_pipeline(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Run the complete feature engineering pipeline
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        print("\n" + "="*60)
        print("🚀 FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        df = self.load_data_from_db()
        
        # Create target
        df = self.create_target_variable(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categoricals
        df = self.encode_categorical_features(df)
        
        # Prepare for ML
        X, y, feature_names = self.prepare_ml_data(df)
        
        print("\n✅ Feature engineering complete!\n")
        
        return X, y, feature_names
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all features for documentation
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'gpa': 'Student cumulative GPA',
            'year_level': 'Current year in program (1-4)',
            'total_enrollments': 'Total courses enrolled in',
            'completed_courses': 'Number of courses completed',
            'dropped_courses': 'Number of courses dropped',
            'current_enrollments': 'Currently enrolled courses',
            'excellent_grades': 'Count of A/A- grades',
            'good_grades': 'Count of B+/B/B- grades',
            'average_grades': 'Count of C+/C/C- grades',
            'poor_grades': 'Count of D/F grades',
            'avg_course_credits': 'Average credits per course',
            'total_credits_attempted': 'Total credits attempted',
            'days_enrolled': 'Days since enrollment',
            'semesters_enrolled': 'Approximate semesters enrolled',
            'completion_rate': 'Percentage of courses completed',
            'drop_rate': 'Percentage of courses dropped',
            'excellence_rate': 'Percentage of excellent grades',
            'good_grade_rate': 'Percentage of good grades',
            'struggle_rate': 'Percentage of poor grades',
            'courses_per_semester': 'Average courses per semester',
            'credits_per_semester': 'Average credits per semester',
            'engagement_score': 'Composite engagement metric',
            'grade_diversity': 'Variation in grade distribution',
            'high_performer': 'Binary indicator of high performance',
            'at_risk': 'Binary indicator of at-risk status',
            'major_encoded': 'Encoded student major'
        }


# Example usage
if __name__ == "__main__":
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.full_pipeline()
    
    print("\n📋 Feature Descriptions:")
    descriptions = engineer.get_feature_descriptions()
    for feature in feature_names[:10]:  # Show first 10
        print(f"   {feature}: {descriptions.get(feature, 'No description')}")