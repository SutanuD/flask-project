# Flask Project

## Overview
This is a simple Flask project that demonstrates how to create a basic API with a "Hello World" endpoint.

## Project Structure
```
flask-project
├── app
│   ├── __init__.py
│   ├── routes.py
│   └── models.py
├── tests
│   ├── __init__.py
│   └── test_routes.py
├── requirements.txt
├── config.py
├── run.py
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-project
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Flask application, execute the following command:
```
python run.py
```

The application will be available at `http://127.0.0.1:5000/`.

## API Endpoints

- **GET /hello**
  - Returns a simple greeting message: "Hello, World!"