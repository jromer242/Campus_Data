"""
SQLAlchemy ORM Models for Campus Data

Defines the database schema for students, courses, and enrollments.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Student(Base):
    """Student model representing enrolled students"""
    
    __tablename__ = 'students'
    
    student_id = Column(String, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String)
    major = Column(String)
    year_level = Column(Integer)
    gpa = Column(Float)
    enrollment_date = Column(DateTime)
    is_active = Column(Boolean)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="student")
    
    def __repr__(self):
        return f"<Student(id={self.student_id}, name={self.first_name} {self.last_name}, gpa={self.gpa})>"


class Course(Base):
    """Course model representing available courses"""
    
    __tablename__ = 'courses'
    
    course_id = Column(String, primary_key=True)
    course_name = Column(String)
    department = Column(String)
    credits = Column(Integer)
    course_type = Column(String)
    max_enrollment = Column(Integer)
    semester = Column(String)
    year = Column(Integer)
    
    # Relationships
    enrollments = relationship("Enrollment", back_populates="course")
    
    def __repr__(self):
        return f"<Course(id={self.course_id}, name={self.course_name}, credits={self.credits})>"


class Enrollment(Base):
    """Enrollment model representing student-course associations"""
    
    __tablename__ = 'enrollments'
    
    enrollment_id = Column(String, primary_key=True)
    student_id = Column(String, ForeignKey('students.student_id'))
    course_id = Column(String, ForeignKey('courses.course_id'))
    enrollment_date = Column(DateTime)
    grade = Column(String)
    status = Column(String)  # e.g., 'Enrolled', 'Completed', 'Dropped'
    
    # Relationships
    student = relationship("Student", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")
    
    def __repr__(self):
        return f"<Enrollment(id={self.enrollment_id}, student={self.student_id}, course={self.course_id}, status={self.status})>"