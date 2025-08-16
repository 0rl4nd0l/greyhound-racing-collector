# Advisory API Contract Analysis

## Task Summary
✅ **Step 1 Complete**: Study `/api/generate_advisory` contract & prepare data model

## Key Findings

### 1. API Endpoint Structure
- **Endpoint**: `POST /api/generate_advisory`
- **Location**: Defined in `app.py` lines 5894-5946
- **Method**: POST with JSON payload

### 2. Request Payload Options
The API accepts **either** (not both):

**Option A: File Path**
```json
{
  "file_path": "/path/to/prediction_file.json"
}
```

**Option B: Direct Prediction Data**
```json
{
  "prediction_data": {
    "race_id": "race_123",
    "race_date": "2025-01-15",
    "race_time": "14:30",
    "extraction_time": "2025-01-15T12:00:00",
    "predictions": [
      {
        "dog_name": "Fast Dog",
        "box_number": 1,
        "win_prob": 0.25,
        "confidence": 0.85
      }
    ]
  }
}
```

### 3. Response Structure
The API returns a comprehensive advisory report:

```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "success": true,
  "file_path": "/path/to/file.json", // if file_path was provided
  "validation_result": { /* validation results */ },
  "analysis_results": { /* QA analysis results */ },
  "messages": [
    {
      "type": "CRITICAL|WARNING|INFO",
      "category": "validation|quality_assessment|confidence_analysis|...",
      "title": "Short Title",
      "message": "Detailed message content",
      "timestamp": "2025-01-15T10:30:01.000Z",
      "details": [] // optional additional data
    }
  ],
  "human_readable_summary": "Executive summary text...",
  "ml_json": { /* structured data for ML processing */ },
  "processing_time_ms": 150.5,
  "openai_used": true
}
```

### 4. Message Types & UI Color Mapping

| Message Type | UI Color Scheme | Use Case |
|--------------|-----------------|----------|
| `CRITICAL` | **Red** (`#dc2626`, `#fee2e2`, `#fca5a5`) | Validation errors, critical quality issues, data leakage |
| `WARNING` | **Orange/Yellow** (`#d97706`, `#fef3c7`, `#fcd34d`) | Moderate quality issues, low confidence predictions |
| `INFO` | **Blue/Neutral** (`#2563eb`, `#dbeafe`, `#93c5fd`) | High quality scores, informational messages |

### 5. Message Categories
- `validation` - Data validation issues
- `quality_assessment` - Overall prediction quality scores  
- `confidence_analysis` - Low confidence prediction warnings
- `class_imbalance` - Probability distribution issues
- `calibration` - Model calibration drift detection
- `data_leakage` - Data leakage or temporal issues
- `system_error` - System/processing errors

### 6. Error Handling
When `success: false`:
```json
{
  "success": false,
  "error": "Error message",
  "details": "Additional error details", // optional
  "timestamp": "2025-01-15T10:30:00.000Z",
  "messages": [
    {
      "type": "CRITICAL",
      "category": "system_error",
      "title": "Advisory Generation Error",
      "message": "Detailed error message"
    }
  ],
  "processing_time_ms": 50.2
}
```

## Files Created

1. **`advisory_data_model.ts`** - Complete TypeScript interfaces for:
   - Request payloads (`AdvisoryRequest`, `PredictionData`)
   - Response structures (`AdvisoryReport`, `AdvisoryMessage`) 
   - Message types and categories
   - UI color mapping reference
   - Usage examples with React/JavaScript

2. **`advisory_contract_analysis.md`** - This summary document

## Implementation Notes

### Backend Integration (advisory.py)
- Uses `AdvisoryGenerator` class
- Integrates with `validator.py` and `qa_analyzer.py`
- Supports OpenAI summarization with template fallback
- Generates structured ML-ready JSON output

### Frontend Integration Guidance
```typescript
// Example color utility function
function getMessageColorClass(type: MessageType): string {
  switch (type) {
    case 'CRITICAL': return 'text-red-600 bg-red-100 border-red-300';
    case 'WARNING': return 'text-amber-600 bg-yellow-100 border-yellow-300'; 
    case 'INFO': return 'text-blue-600 bg-blue-100 border-blue-300';
    default: return 'text-gray-600 bg-gray-100 border-gray-300';
  }
}
```

### Quality Score Interpretation
- **90-100**: Excellent (INFO messages)
- **70-89**: Moderate issues (WARNING messages)  
- **<70**: Significant issues (CRITICAL messages)

## Next Steps
The data model is ready for frontend integration. The TypeScript interfaces can be:
1. Imported into React components
2. Used for API request/response typing
3. Applied to UI components with the color mapping guide
4. Extended as needed for specific UI requirements
