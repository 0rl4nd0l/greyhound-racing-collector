# Architecture Diagram

The following diagram illustrates the system architecture:

```mermaid
graph TB
    subgraph "Data Sources"
        A[FastTrack API] 
        B[Sportsbook APIs]
        C[Weather APIs]
    end
    
    subgraph "Data Layer"
        D[SQLite Database]
        E[File Storage]
    end
    
    subgraph "Processing Layer"
        F[Data Ingestion Pipeline]
        G[Feature Engineering]
        H[ML System V3]
    end
    
    subgraph "Application Layer"
        I[Flask Web App]
        J[Monitoring API]
        K[Prediction Pipeline]
    end
    
    subgraph "Model Governance"
        L[Champion Model]
        M[Challenger Model]
        N[Model Registry]
    end
    
    subgraph "Monitoring"
        O[Prometheus Metrics]
        P[Health Checks]
        Q[Performance Monitoring]
    end
    
    A --> F
    B --> F
    C --> F
    F --> D
    F --> E
    D --> G
    G --> H
    H --> L
    H --> M
    L --> N
    M --> N
    H --> K
    K --> I
    I --> J
    J --> O
    J --> P
    O --> Q
    
    classDef dataSource fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef application fill:#e8f5e8
    classDef monitoring fill:#fff3e0
    
    class A,B,C dataSource
    class F,G,H processing
    class I,J,K application
    class O,P,Q monitoring
```

## Component Descriptions

### Data Sources
- **FastTrack API**: Primary source for race data and form guides
- **Sportsbook APIs**: Real-time odds and betting information
- **Weather APIs**: Weather conditions for race venues

### Data Layer
- **SQLite Database**: Stores structured race data, predictions, and metrics
- **File Storage**: Raw data files and model artifacts

### Processing Layer
- **Data Ingestion Pipeline**: Processes and validates incoming data
- **Feature Engineering**: Creates ML features from raw data
- **ML System V3**: Core machine learning system with multiple models

### Application Layer
- **Flask Web App**: Main web interface and API
- **Monitoring API**: System health and metrics endpoints
- **Prediction Pipeline**: Orchestrates prediction generation

### Model Governance
- **Champion Model**: Current production model
- **Challenger Model**: Model under evaluation
- **Model Registry**: Manages model versions and metadata

### Monitoring
- **Prometheus Metrics**: System and application metrics
- **Health Checks**: Service availability monitoring
- **Performance Monitoring**: Real-time performance tracking
