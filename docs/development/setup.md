# Development Setup

This document provides setup instructions and development guidelines for the Greyhound Racing Predictor project.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

- **Python 3.8+**: Required for running the application.
- **pip**: Python package installer.
- **Git**: Version control system.
- **SQLite/PostgreSQL**: Database for storing application data.
- **Docker** (optional): For containerized development.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/greyhound_racing_collector.git
cd greyhound_racing_collector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root and configure the following variables:

```env
FLASK_APP=app.py
FLASK_ENV=development
DATABASE_URL=sqlite:///greyhound_predictor.db
SECRET_KEY=your-secret-key-here
```

### 5. Initialize the Database

```bash
python -c "from app import db; db.create_all()"
```

### 6. Run the Application

```bash
flask run
```

The application will be available at `http://localhost:5000`.

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code style.
- Use meaningful variable and function names.
- Add docstrings to functions and classes.
- Keep functions small and focused on a single responsibility.

### Testing

#### Running Tests

```bash
pytest tests/
```

#### Writing Tests

- Write unit tests for all new functionality.
- Use the `pytest` framework for testing.
- Aim for high test coverage (>80%).
- Test both positive and negative scenarios.

#### Test Structure

```
tests/
├── unit/
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_ml_system.py
├── integration/
│   ├── test_api.py
│   └── test_database.py
└── fixtures/
    ├── sample_data.json
    └── test_config.py
```

### Version Control

#### Git Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add descriptive commit message"
   ```

3. Push your branch and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Messages

- Use clear, descriptive commit messages.
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update").
- Keep the first line under 50 characters.
- Provide additional context in the body if needed.

### Database Migrations

When making changes to the database schema:

1. Generate a migration:
   ```bash
   flask db migrate -m "Description of changes"
   ```

2. Apply the migration:
   ```bash
   flask db upgrade
   ```

### Debugging

#### Logging

The application uses Python's built-in logging module. Configure log levels in your development environment:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Debug Mode

Run the application in debug mode for development:

```bash
export FLASK_ENV=development
flask run --debug
```

### Performance

#### Profiling

Use profiling tools to identify performance bottlenecks:

```bash
python -m cProfile -o profile.stats your_script.py
```

#### Database Query Optimization

- Use database indexes appropriately.
- Monitor slow queries and optimize them.
- Use database query profiling tools.

## Contributing

### Pull Request Process

1. Ensure your code follows the style guidelines.
2. Add or update tests for your changes.
3. Update documentation as needed.
4. Ensure all tests pass.
5. Create a pull request with a clear description of changes.

### Code Review

- All code changes require review before merging.
- Address reviewer feedback promptly.
- Ensure your branch is up to date with `main` before merging.

### Issue Reporting

When reporting issues:

1. Use the issue tracker on GitHub.
2. Provide a clear description of the problem.
3. Include steps to reproduce the issue.
4. Add relevant error messages or logs.
5. Specify your environment (OS, Python version, etc.).

## Docker Development (Optional)

### Building the Docker Image

```bash
docker build -t greyhound-predictor .
```

### Running with Docker Compose

```bash
docker-compose up -d
```

This will start the application along with any required services (database, monitoring, etc.).

## Monitoring and Metrics

### Prometheus Integration

The application exposes metrics at `/metrics` endpoint. Configure Prometheus to scrape these metrics for monitoring.

### Health Checks

The application provides health check endpoints for monitoring:

- `/health` - Basic health check
- `/ready` - Readiness check for load balancers

## Documentation

### Updating Documentation

- Update relevant documentation when making changes.
- Use Markdown format for documentation files.
- Generate documentation using MkDocs:

```bash
mkdocs serve
```

### API Documentation

- Document all API endpoints with request/response examples.
- Use clear, consistent formatting.
- Include error handling information.

#### New Enhanced Prediction Endpoints

The system now includes enhanced prediction endpoints with intelligent pipeline selection:

**Single Race Prediction**:
```bash
curl -X POST http://localhost:5000/api/predict_single_race_enhanced \
  -H "Content-Type: application/json" \
  -d '{"race_filename": "Race 1 - GOSF - 2025-01-15.csv"}'
```

**Batch Race Prediction**:
```bash
curl -X POST http://localhost:5000/api/predict_all_upcoming_races_enhanced \
  -H "Content-Type: application/json" \
  -d '{"max_races": 5}'
```

**Key Features**:
- Intelligent prediction pipeline selection (PredictionPipelineV3 → UnifiedPredictor → ComprehensivePredictionPipeline)
- Comprehensive error handling and recovery
- Performance monitoring and detailed metrics
- Automatic file discovery and path resolution

## Troubleshooting

### Common Issues

#### Database Connection Errors

- Verify database URL in environment variables.
- Ensure database service is running.
- Check database permissions.

#### Import Errors

- Verify virtual environment is activated.
- Check that all dependencies are installed.
- Ensure Python path is configured correctly.

#### Performance Issues

- Check database query performance.
- Monitor memory usage with large datasets.
- Profile code to identify bottlenecks.

### Getting Help

- Check the documentation first.
- Review existing issues on GitHub.
- Create a new issue if your problem isn't addressed.
- Provide detailed information when seeking help.
