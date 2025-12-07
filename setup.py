from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="campus-data",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An end-to-end data platform for higher education data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jromer242/Campus_Data",
    project_urls={
        "Bug Tracker": "https://github.com/jromer242/Campus_Data/issues",
        "Documentation": "https://github.com/jromer242/Campus_Data/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",  # If using PostgreSQL
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        # Add other dependencies from your requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "campus-data=src.main:main",  # Creates a CLI command
            "campus-api=src.api.main:start_server",  # Command to start API
        ],
    },
    include_package_data=True,
    package_data={
        "campus_data": ["config/*.yaml", "data/*.json"],
    },
)