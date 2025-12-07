import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3  # Using SQLite for immediate start - no setup needed!

from src.database import Student, Course, Enrollment, Base
from src.database.connection import get_session, create_tables

# Set random seed for reproducible data
np.random.seed(42)
random.seed(42)

def generate_students(n=1000):
    """Generate synthetic student data"""
    majors = ['Computer Science', 'Business', 'Psychology', 'Biology', 'English', 
              'Mathematics', 'History', 'Chemistry', 'Political Science', 'Art']
    
    students = []
    for i in range(n):
        student_id = f"STU{str(i+1).zfill(6)}"
        first_names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Avery', 'Quinn']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
        
        student = {
            'student_id': student_id,
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'email': f"{student_id.lower()}@college.edu",
            'major': random.choice(majors),
            'year_level': random.choice([1, 2, 3, 4]),
            'gpa': round(random.uniform(2.0, 4.0), 2),
            'enrollment_date': datetime(2020, 8, 15) + timedelta(days=random.randint(0, 1460)),
            'is_active': random.choice([True, True, True, False])  # 75% active
        }
        students.append(student)
    
    return pd.DataFrame(students)

def generate_courses(n=50):
    """Generate synthetic course data"""
    departments = ['CS', 'BUS', 'PSY', 'BIO', 'ENG', 'MATH', 'HIST', 'CHEM', 'POLS', 'ART']
    course_types = ['Lecture', 'Lab', 'Seminar', 'Workshop']
    
    courses = []
    for i in range(n):
        dept = random.choice(departments)
        course_num = random.randint(1000, 4999)
        
        course = {
            'course_id': f"{dept}-{course_num}",
            'course_name': f"{dept} Course {course_num}",
            'department': dept,
            'credits': random.choice([1, 2, 3, 4]),
            'course_type': random.choice(course_types),
            'max_enrollment': random.randint(15, 100),
            'semester': random.choice(['Fall', 'Spring', 'Summer']),
            'year': random.choice([2023, 2024, 2025])
        }
        courses.append(course)
    
    return pd.DataFrame(courses)

def generate_enrollments(students_df, courses_df, n=2000):
    """Generate synthetic enrollment data"""
    enrollments = []
    
    for i in range(n):
        student_id = random.choice(students_df['student_id'].tolist())
        course_id = random.choice(courses_df['course_id'].tolist())
        
        # Avoid duplicate enrollments
        if any(e['student_id'] == student_id and e['course_id'] == course_id for e in enrollments):
            continue
            
        enrollment = {
            'enrollment_id': f"ENR{str(i+1).zfill(6)}",
            'student_id': student_id,
            'course_id': course_id,
            'enrollment_date': datetime.now() - timedelta(days=random.randint(1, 365)),
            'grade': random.choice(['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F', None]),
            'status': random.choice(['Enrolled', 'Completed', 'Dropped', 'Withdrawn'])
        }
        enrollments.append(enrollment)
    
    return pd.DataFrame(enrollments)

def create_database_and_load_data():
    """Create SQLite database and load all data"""
    print("🚀 Generating synthetic campus data...")
    
    # Generate data
    students_df = generate_students(1000)
    courses_df = generate_courses(50)
    enrollments_df = generate_enrollments(students_df, courses_df, 2000)
    
    print(f"✅ Generated:")
    print(f"   - {len(students_df)} students")
    print(f"   - {len(courses_df)} courses") 
    print(f"   - {len(enrollments_df)} enrollments")
    
    # Create database
    conn = sqlite3.connect('campus_data.db')
    
    # Load data into database
    students_df.to_sql('students', conn, if_exists='replace', index=False)
    courses_df.to_sql('courses', conn, if_exists='replace', index=False)
    enrollments_df.to_sql('enrollments', conn, if_exists='replace', index=False)
    
    print("📊 Data loaded into campus_data.db")
    
    # Show some sample queries
    print("\n🔍 Sample Data Analysis:")
    
    # Query 1: Students by major
    query1 = """
    SELECT major, COUNT(*) as student_count, AVG(gpa) as avg_gpa
    FROM students 
    WHERE is_active = 1
    GROUP BY major 
    ORDER BY student_count DESC
    """
    result1 = pd.read_sql_query(query1, conn)
    print("\n📈 Active Students by Major:")
    print(result1.head())
    
    # Query 2: Course enrollment stats
    query2 = """
    SELECT c.department, COUNT(e.enrollment_id) as total_enrollments,
           COUNT(CASE WHEN e.status = 'Completed' THEN 1 END) as completed
    FROM courses c
    LEFT JOIN enrollments e ON c.course_id = e.course_id
    GROUP BY c.department
    ORDER BY total_enrollments DESC
    """
    result2 = pd.read_sql_query(query2, conn)
    print("\n📚 Enrollments by Department:")
    print(result2.head())
    
    # Query 3: Grade distribution
    query3 = """
    SELECT grade, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM enrollments WHERE grade IS NOT NULL), 1) as percentage
    FROM enrollments 
    WHERE grade IS NOT NULL
    GROUP BY grade 
    ORDER BY count DESC
    """
    result3 = pd.read_sql_query(query3, conn)
    print("\n🎓 Grade Distribution:")
    print(result3)
    
    conn.close()
    
    print("\n🎉 SUCCESS! You now have:")
    print("   - A SQLite database with realistic campus data")
    print("   - Example queries showing data analysis patterns")
    print("   - Foundation for building your portfolio project")
    
    print("\n📝 Next Steps:")
    print("   1. Explore the data with your own SQL queries")
    print("   2. Create visualizations with matplotlib or plotly")
    print("   3. Build a simple API to serve this data")
    print("   4. Add data pipeline automation")

if __name__ == "__main__":
    create_database_and_load_data()