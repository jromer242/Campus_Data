from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Student(Base):
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
    
    # Relationship to enrollments
    enrollments = relationship("Enrollment", back_populates="student")

class Course(Base):
    __tablename__ = 'courses'
    
    course_id = Column(String, primary_key=True)
    course_name = Column(String)
    department = Column(String)
    credits = Column(Integer)
    course_type = Column(String)
    max_enrollment = Column(Integer)
    semester = Column(String)
    year = Column(Integer)
    
    # Relationship to enrollments
    enrollments = relationship("Enrollment", back_populates="course")

class Enrollment(Base):
    __tablename__ = 'enrollments'
    
    enrollment_id = Column(String, primary_key=True)
    student_id = Column(String, ForeignKey('students.student_id'))
    course_id = Column(String, ForeignKey('courses.course_id'))
    enrollment_date = Column(DateTime)
    grade = Column(String)
    status = Column(String)
    
    # Relationships
    student = relationship("Student", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")