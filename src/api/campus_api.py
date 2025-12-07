"""
Campus Data API
FastAPI endpoints for student success predictions and data access
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime

# ✅ NEW IMPORTS - Updated to use new database structure
from src.database import Student, Course, Enrollment
from src.database.connection import get_session, DatabaseSession, print_database_info

# Import ML predictor if available
try:
    from src.ml.student_predictor import StudentSuccessPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("⚠️ Warning: ML Predictor not available")


# Initialize FastAPI app
app = FastAPI(
    title="Campus Data API",
    description="API for student data and success predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML predictor
predictor = None
if PREDICTOR_AVAILABLE:
    try:
        predictor = StudentSuccessPredictor()
        predictor.load_models('models/')
        print("✅ ML Predictor loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load ML models: {e}")


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class StudentResponse(BaseModel):
    """Response model for student data"""
    student_id: str
    first_name: str
    last_name: str
    email: str
    major: Optional[str]
    year_level: Optional[int]
    gpa: Optional[float]
    is_active: bool
    enrollment_date: Optional[datetime]
    
    class Config:
        from_attributes = True


class StudentCreate(BaseModel):
    """Model for creating a new student"""
    student_id: str = Field(..., example="S12345")
    first_name: str = Field(..., example="John")
    last_name: str = Field(..., example="Doe")
    email: str = Field(..., example="john.doe@campus.edu")
    major: Optional[str] = Field(None, example="Computer Science")
    year_level: Optional[int] = Field(None, ge=1, le=4, example=2)
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, example=3.5)
    is_active: bool = Field(True)


class CourseResponse(BaseModel):
    """Response model for course data"""
    course_id: str
    course_name: str
    department: Optional[str]
    credits: Optional[int]
    course_type: Optional[str]
    semester: Optional[str]
    year: Optional[int]
    
    class Config:
        from_attributes = True


class EnrollmentResponse(BaseModel):
    """Response model for enrollment data"""
    enrollment_id: str
    student_id: str
    course_id: str
    enrollment_date: Optional[datetime]
    grade: Optional[str]
    status: str
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Request model for student success prediction"""
    gpa: float = Field(..., ge=0.0, le=4.0, example=3.2)
    year_level: int = Field(..., ge=1, le=4, example=2)
    total_enrollments: int = Field(..., ge=0, example=10)
    completed_courses: int = Field(..., ge=0, example=8)
    dropped_courses: int = Field(..., ge=0, example=1)
    current_enrollments: int = Field(default=0, ge=0, example=1)
    excellent_grades: int = Field(default=0, ge=0, example=4)
    good_grades: int = Field(default=0, ge=0, example=3)
    average_grades: int = Field(default=0, ge=0, example=1)
    poor_grades: int = Field(default=0, ge=0, example=0)
    completion_rate: float = Field(..., ge=0.0, le=100.0, example=80.0)
    drop_rate: float = Field(default=0.0, ge=0.0, le=100.0, example=10.0)
    excellence_rate: float = Field(default=0.0, ge=0.0, le=100.0, example=40.0)
    struggle_rate: float = Field(default=0.0, ge=0.0, le=100.0, example=0.0)


class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    student_id: Optional[str]
    predicted_success: bool
    success_probability: float
    failure_probability: float
    risk_level: str
    confidence: str
    recommendations: List[str]
    model_used: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    database_connected: bool
    ml_available: bool
    total_students: int
    total_courses: int
    total_enrollments: int


# ============================================================================
# Dependency Injection
# ============================================================================

def get_db():
    """
    Dependency for database session
    Automatically handles session lifecycle
    """
    db = get_session()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "Campus Data API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "students": "/students",
            "courses": "/courses",
            "enrollments": "/enrollments",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Test database connection
        student_count = db.query(Student).count()
        course_count = db.query(Course).count()
        enrollment_count = db.query(Enrollment).count()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "ml_available": predictor is not None,
            "total_students": student_count,
            "total_courses": course_count,
            "total_enrollments": enrollment_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# ============================================================================
# Student Endpoints
# ============================================================================

@app.get("/students", response_model=List[StudentResponse], tags=["Students"])
async def get_students(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records"),
    major: Optional[str] = Query(None, description="Filter by major"),
    year_level: Optional[int] = Query(None, ge=1, le=4, description="Filter by year"),
    min_gpa: Optional[float] = Query(None, ge=0.0, le=4.0, description="Minimum GPA"),
    db: Session = Depends(get_db)
):
    """Get list of students with optional filters"""
    query = db.query(Student)
    
    # Apply filters
    if major:
        query = query.filter(Student.major == major)
    if year_level:
        query = query.filter(Student.year_level == year_level)
    if min_gpa is not None:
        query = query.filter(Student.gpa >= min_gpa)
    
    # Apply pagination
    students = query.offset(skip).limit(limit).all()
    return students


@app.get("/students/{student_id}", response_model=StudentResponse, tags=["Students"])
async def get_student(student_id: str, db: Session = Depends(get_db)):
    """Get specific student by ID"""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    
    return student


@app.post("/students", response_model=StudentResponse, status_code=201, tags=["Students"])
async def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    """Create a new student"""
    # Check if student already exists
    existing = db.query(Student).filter(Student.student_id == student.student_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student ID already exists")
    
    # Create new student
    new_student = Student(
        student_id=student.student_id,
        first_name=student.first_name,
        last_name=student.last_name,
        email=student.email,
        major=student.major,
        year_level=student.year_level,
        gpa=student.gpa,
        enrollment_date=datetime.now(),
        is_active=student.is_active
    )
    
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    
    return new_student


@app.get("/students/{student_id}/enrollments", response_model=List[EnrollmentResponse], tags=["Students"])
async def get_student_enrollments(student_id: str, db: Session = Depends(get_db)):
    """Get all enrollments for a specific student"""
    # Verify student exists
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    
    enrollments = db.query(Enrollment).filter(Enrollment.student_id == student_id).all()
    return enrollments


# ============================================================================
# Course Endpoints
# ============================================================================

@app.get("/courses", response_model=List[CourseResponse], tags=["Courses"])
async def get_courses(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    department: Optional[str] = Query(None),
    semester: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get list of courses with optional filters"""
    query = db.query(Course)
    
    if department:
        query = query.filter(Course.department == department)
    if semester:
        query = query.filter(Course.semester == semester)
    
    courses = query.offset(skip).limit(limit).all()
    return courses


@app.get("/courses/{course_id}", response_model=CourseResponse, tags=["Courses"])
async def get_course(course_id: str, db: Session = Depends(get_db)):
    """Get specific course by ID"""
    course = db.query(Course).filter(Course.course_id == course_id).first()
    
    if not course:
        raise HTTPException(status_code=404, detail=f"Course {course_id} not found")
    
    return course


# ============================================================================
# Enrollment Endpoints
# ============================================================================

@app.get("/enrollments", response_model=List[EnrollmentResponse], tags=["Enrollments"])
async def get_enrollments(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db)
):
    """Get list of enrollments"""
    query = db.query(Enrollment)
    
    if status:
        query = query.filter(Enrollment.status == status)
    
    enrollments = query.offset(skip).limit(limit).all()
    return enrollments


# ============================================================================
# ML Prediction Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_success(request: PredictionRequest):
    """Predict student success probability"""
    if not predictor:
        raise HTTPException(
            status_code=503, 
            detail="ML Predictor not available. Models may not be loaded."
        )
    
    try:
        # Convert request to dict
        student_data = request.dict()
        
        # Make prediction
        result = predictor.predict(student_data)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/student/{student_id}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_student_success(student_id: str, db: Session = Depends(get_db)):
    """Predict success for an existing student"""
    if not predictor:
        raise HTTPException(status_code=503, detail="ML Predictor not available")
    
    # Get student data
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    
    # Get enrollment statistics
    from sqlalchemy import func
    
    stats = db.query(
        func.count(Enrollment.enrollment_id).label('total_enrollments'),
        func.sum(func.case((Enrollment.status == 'Completed', 1), else_=0)).label('completed_courses'),
        func.sum(func.case((Enrollment.status == 'Dropped', 1), else_=0)).label('dropped_courses'),
        func.sum(func.case((Enrollment.status == 'Enrolled', 1), else_=0)).label('current_enrollments'),
        func.sum(func.case((Enrollment.grade.in_(['A', 'A-']), 1), else_=0)).label('excellent_grades'),
        func.sum(func.case((Enrollment.grade.in_(['B+', 'B', 'B-']), 1), else_=0)).label('good_grades'),
        func.sum(func.case((Enrollment.grade.in_(['C+', 'C', 'C-']), 1), else_=0)).label('average_grades'),
        func.sum(func.case((Enrollment.grade.in_(['D', 'F']), 1), else_=0)).label('poor_grades'),
    ).filter(Enrollment.student_id == student_id).first()
    
    # Calculate rates
    total = stats.total_enrollments or 1
    completed = stats.completed_courses or 0
    dropped = stats.dropped_courses or 0
    
    student_data = {
        'student_id': student.student_id,
        'gpa': student.gpa or 0.0,
        'year_level': student.year_level or 1,
        'total_enrollments': total,
        'completed_courses': completed,
        'dropped_courses': dropped,
        'current_enrollments': stats.current_enrollments or 0,
        'excellent_grades': stats.excellent_grades or 0,
        'good_grades': stats.good_grades or 0,
        'average_grades': stats.average_grades or 0,
        'poor_grades': stats.poor_grades or 0,
        'completion_rate': (completed / total * 100) if total > 0 else 0,
        'drop_rate': (dropped / total * 100) if total > 0 else 0,
        'excellence_rate': ((stats.excellent_grades or 0) / total * 100) if total > 0 else 0,
        'struggle_rate': ((stats.poor_grades or 0) / total * 100) if total > 0 else 0,
    }
    
    # Make prediction
    result = predictor.predict(student_data)
    return result


# ============================================================================
# Statistics Endpoints
# ============================================================================

@app.get("/stats/overview", tags=["Statistics"])
async def get_overview_stats(db: Session = Depends(get_db)):
    """Get overview statistics"""
    from sqlalchemy import func
    
    stats = {
        'total_students': db.query(Student).count(),
        'active_students': db.query(Student).filter(Student.is_active == True).count(),
        'total_courses': db.query(Course).count(),
        'total_enrollments': db.query(Enrollment).count(),
        'average_gpa': db.query(func.avg(Student.gpa)).scalar() or 0.0,
        'students_by_year': {},
        'students_by_major': {}
    }
    
    # Students by year
    year_counts = db.query(
        Student.year_level, 
        func.count(Student.student_id)
    ).group_by(Student.year_level).all()
    
    stats['students_by_year'] = {year: count for year, count in year_counts if year}
    
    # Students by major
    major_counts = db.query(
        Student.major,
        func.count(Student.student_id)
    ).group_by(Student.major).all()
    
    stats['students_by_major'] = {major: count for major, count in major_counts if major}
    
    return stats


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "="*60)
    print("🚀 Campus Data API Starting...")
    print("="*60)
    print_database_info()
    print("✅ API Ready!")
    print("📝 API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\n👋 Campus Data API Shutting Down...")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )