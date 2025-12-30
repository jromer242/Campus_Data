

"""
Database connection and session management
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool



# Database configuration
DATABASE_URL = "sqlite:///./campus_data.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_session():
    """Get database session"""
    return SessionLocal()


class DatabaseSession:
    """Context manager for database sessions"""
    def __init__(self):
        self.session = None
    
    def __enter__(self):
        self.session = SessionLocal()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()


def print_database_info():
    """Print database information"""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\n📊 Database: {DATABASE_URL}")
    print(f"📋 Tables: {', '.join(tables) if tables else 'None'}")
    
    if tables:
        db = SessionLocal()
        try:
            # Import models to check counts
            from src.database import Student, Course, Enrollment
            print(f"👥 Students: {db.query(Student).count()}")
            print(f"📚 Courses: {db.query(Course).count()}")
            print(f"📝 Enrollments: {db.query(Enrollment).count()}")
        except Exception as e:
            print(f"⚠️  Could not query counts: {e}")
        finally:
            db.close()


def init_database():
    """Initialize database with tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


if __name__ == "__main__":
    init_database()
    print_database_info()