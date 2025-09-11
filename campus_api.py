from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select, func, and_, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List
import pandas as pd
from datetime import datetime
from my_models import Student, Course, Enrollment  # Import your SQLAlchemy models
from fastapi import FastAPI
# Ml Imports
from student_success_models import StudentSuccessPredictor
import os
from typing import Dict, Any
from contextlib import asynccontextmanager

# Database setup
DATABASE_URL = "sqlite:///campus_data.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# @app.lifespan("startup")
# @app.on_event("startup")

# Global ML predictor instance
ml_predictor = None

# FastAPI app
@asynccontextmanager
# async def load_ml_models():
async def lifespan(app: FastAPI):
    """Load ML models when the API starts"""
    global ml_predictor
    
    try:
        ml_predictor = StudentSuccessPredictor()
        
        # Check if models exist, if not train them
        if os.path.exists('models') and os.listdir('models'):
            ml_predictor.load_models()
            print("✅ ML models loaded successfully")
        else:
            print("🤖 No trained models found. Training new models...")
            df = ml_predictor.load_and_prepare_data()
            ml_predictor.train_models(df)
            ml_predictor.save_models()
            print("✅ ML models trained and saved")
            
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")
        ml_predictor = None
    
    yield

app = FastAPI(
    title="Trinity College Campus Data API",
    description="API for campus analytics, student data, and academic insights",
    version="1.0.0", 
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Pydantic models for API responses
class StudentResponse(BaseModel):
    student_id: str
    first_name: str
    last_name: str
    email: str
    major: str
    year_level: int
    gpa: float
    is_active: bool
    enrollment_date: datetime

class StudentSummary(BaseModel):
    total_students: int
    active_students: int
    average_gpa: float
    students_by_major: dict

class CourseAnalytics(BaseModel):
    course_id: str
    course_name: str
    department: str
    total_enrollments: int
    completion_rate: float
    average_grade_point: Optional[float]

class AtRiskStudent(BaseModel):
    student_id: str
    full_name: str
    major: str
    gpa: float
    active_enrollments: int

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Trinity College Campus Data API",
        "docs": "/docs",
        "endpoints": {
            "students": "/students",
            "analytics": "/analytics/students",
            "at-risk": "/students/at-risk",
            "courses": "/analytics/courses",
            "predictions": "/ml/predictions"
        }
    }

# Student endpoints - Order matters! Most specfic routes first
@app.get("/students/at-risk", response_model=List[AtRiskStudent])
def get_at_risk_students(
    gpa_threshold: float = Query(2.5, description="GPA threshold for at-risk classification"),
    db: Session = Depends(get_db)
):
    """Identify students at academic risk"""
    
    # Using raw SQL for complex query (like you'd do in the real job)
    query = """
    SELECT 
        s.student_id,
        s.first_name || ' ' || s.last_name as full_name,
        s.major,
        s.gpa,
        COUNT(e.enrollment_id) as active_enrollments
    FROM students s
    LEFT JOIN enrollments e ON s.student_id = e.student_id 
        AND e.status IN ('Enrolled', 'Completed')
    WHERE s.is_active = 1 AND s.gpa < :gpa_threshold
    GROUP BY s.student_id, s.first_name, s.last_name, s.major, s.gpa
    ORDER BY s.gpa ASC
    """
    params = {"gpa_threshold": gpa_threshold}
    result = db.execute(text(query), params)
    
    at_risk_students = []
    for row in result:
        at_risk_students.append(AtRiskStudent(
            student_id=row.student_id,
            full_name=row.full_name,
            major=row.major,
            gpa=row.gpa,
            active_enrollments=row.active_enrollments
        ))
    
    return at_risk_students


@app.get("/students", response_model=List[StudentResponse])
def get_students(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    major: Optional[str] = Query(None, description="Filter by major"),
    active_only: bool = Query(True, description="Return only active students"),
    min_gpa: Optional[float] = Query(None, ge=0.0, le=4.0, description="Minimum GPA filter"),
    db: Session = Depends(get_db)
):
    """Get students with filtering options"""
    
    query = db.query(Student)
    
    # Apply filters
    if active_only:
        query = query.filter(Student.is_active == True)
    if major:
        query = query.filter(Student.major.ilike(f"%{major}%"))
    if min_gpa:
        query = query.filter(Student.gpa >= min_gpa)
    
    # Apply pagination
    students = query.offset(skip).limit(limit).all()
    
    return students

@app.get("/students/{student_id}", response_model=StudentResponse)
def get_student(student_id: str, db: Session = Depends(get_db)):
    """Get a specific student by ID"""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student



# Analytics endpoints
@app.get("/analytics/students", response_model=StudentSummary)
def get_student_analytics(db: Session = Depends(get_db)):
    """Get student population analytics"""
    
    # Using pandas for complex analytics (showcase your pandas skills)
    students_df = pd.read_sql_query(
        "SELECT * FROM students", 
        con=engine
    )
    
    analytics = StudentSummary(
        total_students=len(students_df),
        active_students=len(students_df[students_df['is_active'] == 1]),
        average_gpa=round(students_df[students_df['is_active'] == 1]['gpa'].mean(), 2),
        students_by_major=students_df[students_df['is_active'] == 1]['major'].value_counts().to_dict()
    )
    
    return analytics

@app.get("/analytics/courses", response_model=List[CourseAnalytics])
def get_course_analytics(
    department: Optional[str] = Query(None, description="Filter by department"),
    db: Session = Depends(get_db)
):
    """Get course performance analytics"""
    
    # Complex query showcasing SQL skills
    query = """
    SELECT 
        c.course_id,
        c.course_name,
        c.department,
        COUNT(e.enrollment_id) as total_enrollments,
        ROUND(
            COUNT(CASE WHEN e.status = 'Completed' THEN 1 END) * 100.0 / 
            NULLIF(COUNT(e.enrollment_id), 0), 2
        ) as completion_rate,
        AVG(
            CASE e.grade
                WHEN 'A' THEN 4.0
                WHEN 'A-' THEN 3.7
                WHEN 'B+' THEN 3.3
                WHEN 'B' THEN 3.0
                WHEN 'B-' THEN 2.7
                WHEN 'C+' THEN 2.3
                WHEN 'C' THEN 2.0
                WHEN 'C-' THEN 1.7
                WHEN 'D' THEN 1.0
                WHEN 'F' THEN 0.0
            END
        ) as average_grade_point
    FROM courses c
    LEFT JOIN enrollments e ON c.course_id = e.course_id
    GROUP BY c.course_id, c.course_name, c.department
    HAVING COUNT(e.enrollment_id) > 0
    ORDER BY total_enrollments DESC
    """.format("WHERE c.department = :dept" if department else "")
    
    params = {"dept": department} if department else {}
    result = db.execute(text(query), params)
    
    course_analytics = []
    for row in result:
        course_analytics.append(CourseAnalytics(
            course_id=row.course_id,
            course_name=row.course_name,
            department=row.department,
            total_enrollments=row.total_enrollments,
            completion_rate=row.completion_rate or 0.0,
            average_grade_point=round(row.average_grade_point, 2) if row.average_grade_point else None
        ))
    
    return course_analytics

# ML/AI endpoint (placeholder for future ML models)
@app.get("/ml/predictions/student-success/{student_id}")
def predict_student_success(student_id: str, db: Session = Depends(get_db)):
    """Predict student success probability (ML placeholder)"""
    
    if not ml_predictor:
        # Get student data
        student = db.query(Student).filter(Student.student_id == student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
    
        # Simple rule-based prediction (replace with actual ML model later)
        success_probability = min(student.gpa / 4.0 + 0.1, 1.0)
        
        # Risk factors analysis
        risk_factors = []
        if student.gpa < 2.5:
            risk_factors.append("Low GPA")
        if student.year_level > 2 and student.gpa < 3.0:
            risk_factors.append("Upper-level with declining performance")
        
        return {
            "student_id": student_id,
            "success_probability": round(success_probability, 2),
            "risk_level": "High" if success_probability < 0.6 else "Medium" if success_probability < 0.8 else "Low",
            "risk_factors": risk_factors,
            "recommendations": [
                "Consider academic support services",
                "Meet with academic advisor",
                "Explore tutoring options"
            ] if success_probability < 0.7 else ["Continue current academic plan"]
        }
    
    try:
        query = text("""
        SELECT 
            s.student_id,
            s.gpa,
            s.year_level,
            s.major,
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
        WHERE s.student_id = :student_id
        GROUP BY s.student_id, s.gpa, s.year_level, s.major, s.enrollment_date
        """)

        result = db.execute(query, {"student_id": student_id}).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Student not found or no enrollment data")
        
        student_data = {
            'gpa': float(result[1]),
            'year_level': int(result[2]),
            'total_enrollments': int(result[5]) if result[5] else 0,
            'completed_courses': int(result[6]) if result[6] else 0,
            'dropped_courses': int(result[7]) if result[7] else 0,
            'current_enrollments': int(result[8]) if result[8] else 0,
            'excellent_grades': int(result[9]) if result[9] else 0,
            'good_grades': int(result[10]) if result[10] else 0,
            'average_grades': int(result[11]) if result[11] else 0,
            'poor_grades': int(result[12]) if result[12] else 0,
            'avg_course_credits': float(result[13]) if result[13] else 3.0,
            'total_credits_attempted': float(result[14]) if result[14] else 0,
            'days_enrolled': float(result[15]) if result[15] else 0
        }

        #  Calculate derived features
        total_enrollments = student_data['total_enrollments']
        if total_enrollments > 0:
            student_data['completion_rate'] = student_data['completed_courses'] / total_enrollments * 100
            student_data['drop_rate'] = student_data['dropped_courses'] / total_enrollments * 100
            student_data['excellence_rate'] = student_data['excellent_grades'] / total_enrollments * 100
            student_data['struggle_rate'] = student_data['poor_grades'] / total_enrollments * 100
        else:
            student_data['completion_rate'] = 0
            student_data['drop_rate'] = 0
            student_data['excellence_rate'] = 0
            student_data['struggle_rate'] = 0

        student_data['courses_per_semester'] = total_enrollments / (student_data['days_enrolled'] / 120 + 1)

        if 'major' in ml_predictor.encoders and result[3]:
            try:
                student_data['major_encoded'] = ml_predictor.encoders['major'].transform([result[3]])[0]
            except ValueError:
                student_data['major_encoded'] = 0
        else:
            student_data['major_encoded'] = 0

        # Make ML prediction
        prediction = ml_predictor.predict_student_success(student_data)

        #  Add additional context
        prediction['student_id'] = student_id
        prediction['features_used'] = list(student_data.keys())
        prediction['prediction_timestamp'] = datetime.now().isoformat()

        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed:{str(e)}")

# NEW ML ENDPOINTS - Add these after the existing ML endpoint above

# Model performance endpoint
@app.get("/ml/model-performance")
def get_model_performance():
    """Get ML model performance metrics"""
    
    if not ml_predictor or not ml_predictor.model_metrics:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    # Format metrics for API response
    formatted_metrics = {}
    
    for model_name, metrics in ml_predictor.model_metrics.items():
        formatted_metrics[model_name] = {
            'roc_auc_score': round(metrics['roc_auc'], 3),
            'cross_validation_mean': round(metrics['cv_mean'], 3),
            'cross_validation_std': round(metrics['cv_std'], 3),
            'precision': round(metrics['classification_report']['1']['precision'], 3),
            'recall': round(metrics['classification_report']['1']['recall'], 3),
            'f1_score': round(metrics['classification_report']['1']['f1-score'], 3),
            'best_parameters': metrics['best_params']
        }
    
    # Identify best model
    best_model = max(ml_predictor.model_metrics.keys(), 
                    key=lambda x: ml_predictor.model_metrics[x]['roc_auc'])
    
    return {
        'models': formatted_metrics,
        'best_model': best_model,
        'best_model_score': round(ml_predictor.model_metrics[best_model]['roc_auc'], 3),
        'total_models_trained': len(formatted_metrics),
        'last_updated': datetime.now().isoformat()
    }

# Feature importance endpoint
@app.get("/ml/feature-importance")
def get_feature_importance():
    """Get the most important features for student success prediction"""
    
    if not ml_predictor:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    try:
        # Get feature importance from the best model
        best_model_name = max(ml_predictor.model_metrics.keys(), 
                             key=lambda x: ml_predictor.model_metrics[x]['roc_auc'])
        
        best_model = ml_predictor.models[best_model_name]
        
        # Extract feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_data = [
                {
                    'feature': feature,
                    'importance': float(importance),
                    'rank': rank + 1
                }
                for rank, (feature, importance) in enumerate(
                    zip(ml_predictor.feature_names, best_model.feature_importances_)
                )
            ]
            importance_data.sort(key=lambda x: x['importance'], reverse=True)
            
        elif hasattr(best_model.named_steps['classifier'], 'coef_'):
            coef = best_model.named_steps['classifier'].coef_[0]
            importance_data = [
                {
                    'feature': feature,
                    'coefficient': float(coef_val),
                    'abs_coefficient': float(abs(coef_val)),
                    'rank': rank + 1
                }
                for rank, (feature, coef_val) in enumerate(zip(ml_predictor.feature_names, coef))
            ]
            importance_data.sort(key=lambda x: x['abs_coefficient'], reverse=True)
            
        else:
            raise HTTPException(status_code=500, detail="Feature importance not available for this model type")
        
        return {
            'model_used': best_model_name,
            'feature_importance': importance_data[:15],  # Top 15 features
            'total_features': len(ml_predictor.feature_names),
            'importance_type': 'feature_importance' if hasattr(best_model, 'feature_importances_') else 'coefficient'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

# Batch prediction endpoint
@app.post("/ml/predictions/batch")
def batch_predict_student_success(
    student_ids: List[str],
    db: Session = Depends(get_db)
):
    """Make predictions for multiple students at once"""
    
    if not ml_predictor:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    if len(student_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 students per batch")
    
    predictions = []
    
    for student_id in student_ids:
        try:
            # Reuse the single prediction logic
            prediction = predict_student_success(student_id, db)
            predictions.append(prediction)
        except HTTPException as e:
            predictions.append({
                'student_id': student_id,
                'error': e.detail,
                'success_probability': None
            })
    
    return {
        'predictions': predictions,
        'total_students': len(student_ids),
        'successful_predictions': len([p for p in predictions if 'error' not in p]),
        'batch_timestamp': datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

# Debug endpoint to check data
@app.get("/debug/tables")
def debug_tables(db: Session = Depends(get_db)):
    """Debug endpoint to check if tables have data"""
    try:
        student_count = db.execute("SELECT COUNT(*) FROM students").scalar()
        course_count = db.execute("SELECT COUNT(*) FROM courses").scalar()
        enrollment_count = db.execute("SELECT COUNT(*) FROM enrollments").scalar()
        
        # Sample data
        sample_course = db.execute("SELECT * FROM courses LIMIT 1").fetchone()
        sample_enrollment = db.execute("SELECT * FROM enrollments LIMIT 1").fetchone()
        
        return {
            "counts": {
                "students": student_count,
                "courses": course_count,
                "enrollments": enrollment_count
            },
            "sample_course": dict(sample_course._mapping) if sample_course else None,
            "sample_enrollment": dict(sample_enrollment._mapping) if sample_enrollment else None
        }
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)