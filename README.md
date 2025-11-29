# Smart Campus Analytics Platform


An end-to-end data platform that ingests, processes, and analyzes synthetic higher education data, complete with AI-powered insights and automated reporting.


# Dependencies

`pip install fastapi uvicorn`

<img width="1891" height="904" alt="Screenshot 2025-09-13 at 12 30 49 AM" src="https://github.com/user-attachments/assets/edb9d1e3-0977-4e6c-9712-493fec63a8f7" />

# Campus Data Platform
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]


## Prerequisites
- Python 3.8+
- PostgreSQL (or your database)
- [Other requirements]

## Installation

1. Clone the repository
```bash
   git clone https://github.com/jromer242/Campus_Data.git
   cd Campus_Data
```

2. Create virtual environment
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

## API

Start the API server:
```bash
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.
