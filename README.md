# Smart Campus Analytics Platform

An end-to-end data platform that ingests, processes, and analyzes synthetic higher education data, complete with AI-powered insights and automated reporting.

![Campus Data Dashboard](https://github.com/user-attachments/assets/edb9d1e3-0977-4e6c-9712-493fec63a8f7)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jromer242/Campus_Data.git
cd Campus_Data

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install the package in development mode
pip install -e .

# 4. Start the API server
uvicorn src.api.campus_api:app --reload
```

Then visit **http://localhost:8000/docs** for interactive API documentation.

---

## 📋 Prerequisites

- Python 3.8 or higher
- SQLite (included with Python)
- pip (Python package installer)
- Git

---

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/jromer242/Campus_Data.git
cd Campus_Data
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

**Option A - Install with setup.py (Recommended):**
```bash
pip install -e .
```

**Option B - Install with requirements.txt:**
```bash
pip install -r requirements.txt
```

**Option C - Install development dependencies:**
```bash
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
python -c "from src.database import Student, Course, Enrollment; print('✅ Installation successful!')"
```

---

## 🚀 Running the Application

### Start the API Server

```bash
uvicorn src.api.campus_api:app --reload
```

**Alternative methods:**
```bash
# Using Python module syntax
python -m uvicorn src.api.campus_api:app --reload

# Custom port
uvicorn src.api.campus_api:app --reload --port 8080

# Allow external access
uvicorn src.api.campus_api:app --reload --host 0.0.0.0
```

The server will start at **http://localhost:8000**

### Access Points

- **Interactive API Documentation (Swagger UI):** http://localhost:8000/docs
- **Alternative API Documentation (ReDoc):** http://localhost:8000/redoc
- **API Root:** http://localhost:8000
- **Health Check:** http://localhost:8000/health

### Run the Dashboard

In a **new terminal window** (keep the API running):

```bash
cd Campus_Data
source .venv/bin/activate  # Activate venv
python dashboard.py
```

---

## 📊 Project Structure

```
Campus_Data/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── campus_api.py        # Main FastAPI application
│   ├── database/
│   │   ├── __init__.py          # Database models (Student, Course, Enrollment)
│   │   └── connection.py        # Database connection and session management
│   ├── ml/
│   │   ├── __init__.py
│   │   └── student_predictor.py # ML prediction models
│   └── data_generation/
├── models/                       # Trained ML models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── campus_data.db               # SQLite database
├── dashboard.py                 # Dashboard application
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 API Endpoints

### Health & Info
- `GET /` - API root with basic information
- `GET /health` - Health check and system status

### Students
- `GET /students` - List all students (with pagination and filters)
- `GET /students/{student_id}` - Get specific student
- `POST /students` - Create new student
- `GET /students/{student_id}/enrollments` - Get student enrollments

### Courses
- `GET /courses` - List all courses
- `GET /courses/{course_id}` - Get specific course

### Enrollments
- `GET /enrollments` - List all enrollments

### Predictions (ML)
- `POST /predict` - Predict student success with custom data
- `GET /predict/student/{student_id}` - Predict success for existing student

### Statistics
- `GET /stats/overview` - Get overview statistics

For complete API documentation with examples, visit http://localhost:8000/docs after starting the server.

---

## 🧪 Testing the API

### Using the Interactive Docs

Visit http://localhost:8000/docs and use the "Try it out" button on any endpoint.

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get students
curl http://localhost:8000/students?limit=5

# Get specific student
curl http://localhost:8000/students/S12345

# Predict success
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gpa": 3.5,
    "year_level": 2,
    "total_enrollments": 10,
    "completed_courses": 8,
    "dropped_courses": 1,
    "completion_rate": 80.0
  }'
```

### Using Python

```python
import requests

# Get health status
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get students
response = requests.get("http://localhost:8000/students?limit=5")
students = response.json()
print(f"Found {len(students)} students")
```

### Using the API Tester Script

```bash
python api_tester.py
```

---

## 🔍 Features

- **RESTful API** with FastAPI
- **SQLite Database** with SQLAlchemy ORM
- **Student Management** - CRUD operations for students, courses, and enrollments
- **ML Predictions** - Student success prediction with risk assessment
- **Data Analytics** - Statistical overview and insights
- **Interactive Documentation** - Auto-generated API docs with Swagger UI
- **CORS Support** - Ready for frontend integration
- **Type Validation** - Pydantic models for request/response validation

---

## 🐛 Troubleshooting

### ImportError: cannot import name 'DatabaseSession'

Make sure `src/database/connection.py` has the `DatabaseSession` class. If missing, check the installation.

### Port Already in Use

```bash
# Use a different port
uvicorn src.api.campus_api:app --reload --port 8080
```

### Database Not Found

The SQLite database should be created automatically. If issues persist:

```bash
python -c "from src.database.connection import init_database; init_database()"
```

### Module Not Found Errors

```bash
# Reinstall in development mode
pip install -e .
```

---

## 🧑‍💻 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

---

## 📝 Configuration

### Database Configuration

Edit `src/database/connection.py` to change the database:

```python
# For PostgreSQL
DATABASE_URL = "postgresql://user:password@localhost/dbname"

# For MySQL
DATABASE_URL = "mysql+pymysql://user:password@localhost/dbname"
```

### API Configuration

Modify settings in `src/api/campus_api.py`:

```python
app = FastAPI(
    title="Campus Data API",
    version="1.0.0",
    # Add your custom settings
)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@jromer242](https://github.com/jromer242)
- Email: jylesromer@gmail.com

---

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- SQLAlchemy for database ORM
- All contributors and supporters

---

## 📞 Support

If you have any questions or run into issues:

1. Check the [API Documentation](http://localhost:8000/docs) (after starting the server)
2. Review the [Troubleshooting](#-troubleshooting) section
3. Open an [Issue](https://github.com/jromer242/Campus_Data/issues)
4. Contact the maintainer

---

**Made with ❤️ for educational data analytics**
