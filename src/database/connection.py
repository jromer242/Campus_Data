"""
Database connection utilities
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base


def get_engine(db_path='campus_data.db'):
    """Create and return database engine"""
    return create_engine(f'sqlite:///{db_path}', echo=False)


def get_session(db_path='campus_data.db'):
    """Create and return database session"""
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


def create_tables(db_path='campus_data.db'):
    """Create all tables in the database"""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    print(f"✅ Tables created in {db_path}")


def drop_tables(db_path='campus_data.db'):
    """Drop all tables from the database"""
    engine = get_engine(db_path)
    Base.metadata.drop_all(engine)
    print(f"⚠️  Tables dropped from {db_path}")