/**
 * Advisory API Data Model
 * ======================
 * 
 * TypeScript-like interfaces for the /api/generate_advisory endpoint
 * Based on analysis of advisory.py and app.py
 * 
 * Author: AI Assistant
 * Date: January 2025
 */

// ===== REQUEST INTERFACES =====

/**
 * Request payload for /api/generate_advisory endpoint
 * Must provide either file_path OR prediction_data (not both)
 */
interface AdvisoryRequest {
    /** Path to JSON file containing prediction data to analyze */
    file_path?: string;
    
    /** Direct prediction data object to analyze (alternative to file_path) */
    prediction_data?: PredictionData;
}

/**
 * Prediction data structure when passed directly via prediction_data
 */
interface PredictionData {
    race_id: string;
    race_date: string;
    race_time?: string;
    extraction_time?: string;
    venue?: string;
    distance?: string;
    grade?: string;
    predictions: PredictionItem[];
    [key: string]: any; // Allow additional properties for analysis
}

/**
 * Individual prediction item within the prediction data
 */
interface PredictionItem {
    dog_name: string;
    box_number: number;
    win_prob?: number;
    place_prob?: number;
    confidence?: number;
    final_score?: number;
    [key: string]: any; // Allow additional prediction properties
}

// ===== RESPONSE INTERFACES =====

/**
 * Main response structure from /api/generate_advisory endpoint
 */
interface AdvisoryReport {
    /** Timestamp when advisory generation started */
    timestamp: string;
    
    /** Whether advisory generation was successful */
    success: boolean;
    
    /** File path that was analyzed (if file_path was provided) */
    file_path?: string;
    
    /** Raw validation results from validator.py */
    validation_result?: ValidationResult;
    
    /** Raw analysis results from qa_analyzer.py */
    analysis_results?: AnalysisResult;
    
    /** Array of structured advisory messages */
    messages: AdvisoryMessage[];
    
    /** Human-readable summary (OpenAI generated or template-based) */
    human_readable_summary: string;
    
    /** Structured JSON for downstream ML processing */
    ml_json: MLOutputData;
    
    /** Processing time in milliseconds */
    processing_time_ms: number;
    
    /** Whether OpenAI was used for summary generation */
    openai_used: boolean;
    
    /** Error message (only present when success: false) */
    error?: string;
}

/**
 * Individual advisory message structure
 */
interface AdvisoryMessage {
    /** Message severity level - determines UI color */
    type: MessageType;
    
    /** Category of the issue */
    category: MessageCategory;
    
    /** Short title/headline for the message */
    title: string;
    
    /** Detailed message content */
    message: string;
    
    /** ISO timestamp when message was generated */
    timestamp: string;
    
    /** Additional details (optional, for certain message types) */
    details?: any[];
}

/**
 * Message type enumeration - maps to UI colors
 */
type MessageType = 
    | 'INFO'      // Maps to: blue/neutral colors in UI
    | 'WARNING'   // Maps to: orange/yellow colors in UI  
    | 'CRITICAL'; // Maps to: red colors in UI

/**
 * Message category enumeration
 */
type MessageCategory = 
    | 'validation'           // Issues from data validation
    | 'quality_assessment'   // Overall prediction quality scores
    | 'confidence_analysis'  // Low confidence prediction issues
    | 'class_imbalance'     // Probability distribution issues
    | 'calibration'         // Model calibration drift issues
    | 'data_leakage'        // Data leakage or temporal issues
    | 'system_error';       // System/processing errors

/**
 * Validation result structure (from validator.py)
 */
interface ValidationResult {
    valid: boolean;
    errors?: string[];
    warnings?: string[];
    [key: string]: any;
}

/**
 * Analysis result structure (from qa_analyzer.py)
 */
interface AnalysisResult {
    overall_quality_score: number;
    issue_categories: string[];
    total_issues_detected?: number;
    individual_analyses?: {
        confidence_variance?: ConfidenceAnalysis;
        class_imbalance?: ClassImbalanceAnalysis;
        calibration_drift?: CalibrationAnalysis;
        leakage_date_drift?: LeakageAnalysis;
    };
    [key: string]: any;
}

/**
 * Individual analysis sub-structures
 */
interface ConfidenceAnalysis {
    issues_detected: boolean;
    low_confidence_count?: number;
    flagged_predictions?: any[];
    [key: string]: any;
}

interface ClassImbalanceAnalysis {
    issues_detected: boolean;
    normalized_entropy?: number;
    [key: string]: any;
}

interface CalibrationAnalysis {
    issues_detected: boolean;
    [key: string]: any;
}

interface LeakageAnalysis {
    issues_detected: boolean;
    errors?: string[];
    [key: string]: any;
}

/**
 * ML output data structure for downstream processing
 */
interface MLOutputData {
    version: string;
    timestamp: string;
    summary: {
        total_messages: number;
        critical_count: number;
        warning_count: number;
        info_count: number;
        validation_passed: boolean;
        quality_score: number;
    };
    messages: AdvisoryMessage[];
    raw_validation?: ValidationResult;
    raw_analysis?: AnalysisResult;
    feature_flags: {
        has_validation_errors: boolean;
        has_quality_issues: boolean;
        low_quality_score: boolean;
    };
}

// ===== UI COLOR MAPPING REFERENCE =====

/**
 * MESSAGE TYPE → UI COLOR MAPPING
 * ===============================
 * 
 * Use these color mappings in your UI components:
 * 
 * CRITICAL → Red colors
 *   - Background: #fee2e2 (red-100)
 *   - Border: #fca5a5 (red-300) 
 *   - Text: #dc2626 (red-600)
 *   - Icon: #ef4444 (red-500)
 * 
 * WARNING → Orange/Yellow colors
 *   - Background: #fef3c7 (yellow-100) or #fed7aa (orange-100)
 *   - Border: #fcd34d (yellow-300) or #fdba74 (orange-300)
 *   - Text: #d97706 (amber-600) or #ea580c (orange-600)
 *   - Icon: #f59e0b (amber-500) or #f97316 (orange-500)
 * 
 * INFO → Blue/Neutral colors
 *   - Background: #dbeafe (blue-100) or #f3f4f6 (gray-100)
 *   - Border: #93c5fd (blue-300) or #d1d5db (gray-300)
 *   - Text: #2563eb (blue-600) or #374151 (gray-700)
 *   - Icon: #3b82f6 (blue-500) or #6b7280 (gray-500)
 */

// ===== ERROR RESPONSE =====

/**
 * Error response structure (when success: false)
 */
interface AdvisoryErrorResponse {
    success: false;
    error: string;
    details?: string;
    timestamp?: string;
    messages?: AdvisoryMessage[]; // May include system error message
    processing_time_ms?: number;
}

// ===== EXPORT TYPES =====

export type {
    AdvisoryRequest,
    PredictionData,
    PredictionItem,
    AdvisoryReport,
    AdvisoryMessage,
    MessageType,
    MessageCategory,
    ValidationResult,
    AnalysisResult,
    ConfidenceAnalysis,
    ClassImbalanceAnalysis,
    CalibrationAnalysis,
    LeakageAnalysis,
    MLOutputData,
    AdvisoryErrorResponse
};

// ===== USAGE EXAMPLES =====

/**
 * Example usage in React/JavaScript:
 * 
 * ```typescript
 * // Making a request with file_path
 * const response = await fetch('/api/generate_advisory', {
 *   method: 'POST',
 *   headers: { 'Content-Type': 'application/json' },
 *   body: JSON.stringify({
 *     file_path: '/path/to/prediction_file.json'
 *   })
 * });
 * 
 * const advisory: AdvisoryReport = await response.json();
 * 
 * // Making a request with prediction_data
 * const response2 = await fetch('/api/generate_advisory', {
 *   method: 'POST',
 *   headers: { 'Content-Type': 'application/json' },
 *   body: JSON.stringify({
 *     prediction_data: {
 *       race_id: 'race_123',
 *       race_date: '2025-01-15',
 *       predictions: [
 *         { dog_name: 'Fast Dog', box_number: 1, win_prob: 0.25 },
 *         { dog_name: 'Quick Pup', box_number: 2, win_prob: 0.30 }
 *       ]
 *     }
 *   })
 * });
 * 
 * // Rendering messages with appropriate colors
 * advisory.messages.forEach(message => {
 *   const colorClass = getColorClass(message.type);
 *   console.log(`${message.type}: ${message.title} - ${message.message}`);
 * });
 * 
 * function getColorClass(type: MessageType): string {
 *   switch (type) {
 *     case 'CRITICAL': return 'text-red-600 bg-red-100 border-red-300';
 *     case 'WARNING': return 'text-amber-600 bg-yellow-100 border-yellow-300';
 *     case 'INFO': return 'text-blue-600 bg-blue-100 border-blue-300';
 *     default: return 'text-gray-600 bg-gray-100 border-gray-300';
 *   }
 * }
 * ```
 */
