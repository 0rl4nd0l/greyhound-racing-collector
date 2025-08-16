"""
Example implementations of GPTAssistant interfaces.
These demonstrate how the interfaces would be implemented in practice.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from gpt_assistant_interfaces import (
    LogIngestor, FormGuideValidator, PredictionQAAnalyzer, 
    AdvisoryDispatcher, OutputReporter
)
from models import PredictionRecord, GuideIssue, AdvisoryMessage, Severity, IssueType


class JSONLogIngestor(LogIngestor):
    """Example implementation for ingesting JSON prediction logs."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def ingest_logs(self, path: str) -> List[PredictionRecord]:
        """Ingest JSON logs from file or directory."""
        records = []
        
        if os.path.isfile(path):
            records.extend(self._process_file(path))
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    file_path = os.path.join(path, filename)
                    records.extend(self._process_file(file_path))
        else:
            raise FileNotFoundError(f"Path not found: {path}")
        
        self.logger.info(f"Ingested {len(records)} records from {path}")
        return records
    
    def _process_file(self, file_path: str) -> List[PredictionRecord]:
        """Process a single JSON log file."""
        records = []
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    record = self.parse_log_entry(line)
                    if record:
                        records.append(record)
                    else:
                        self.logger.warning(f"Failed to parse line {line_num} in {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return records
    
    def parse_log_entry(self, entry: str) -> Optional[PredictionRecord]:
        """Parse a JSON log entry."""
        try:
            data = json.loads(entry)
            
            if not self.validate_format(entry):
                return None
            
            return PredictionRecord(
                race_id=data['race_id'],
                dog_name=data['dog_name'],
                prediction_value=float(data['prediction_value']),
                confidence=float(data['confidence']),
                timestamp=datetime.fromisoformat(data['timestamp']),
                model_version=data['model_version'],
                form_data=data.get('form_data', {}),
                validation_errors=[]
            )
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Parse error: {str(e)}")
            return None
    
    def validate_format(self, entry: str) -> bool:
        """Validate JSON log entry format."""
        try:
            data = json.loads(entry)
            required_fields = ['race_id', 'dog_name', 'prediction_value', 
                             'confidence', 'timestamp', 'model_version']
            
            return all(field in data for field in required_fields)
        
        except json.JSONDecodeError:
            return False


class BasicFormGuideValidator(FormGuideValidator):
    """Example implementation for form guide validation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_form_data(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """Validate form data across multiple records."""
        issues = []
        
        for record in records:
            # Check individual record completeness
            completeness_issues = self.check_data_completeness(record)
            issues.extend(completeness_issues)
        
        # Check consistency across records
        consistency_issues = self.validate_consistency(records)
        issues.extend(consistency_issues)
        
        self.logger.info(f"Found {len(issues)} form validation issues")
        return issues
    
    def check_data_completeness(self, record: PredictionRecord) -> List[GuideIssue]:
        """Check completeness of a single record."""
        issues = []
        
        # Check for essential form data fields
        essential_fields = ['recent_form', 'trainer', 'weight', 'last_start']
        missing_fields = []
        
        for field in essential_fields:
            if field not in record.form_data or not record.form_data[field]:
                missing_fields.append(field)
        
        if missing_fields:
            issue = GuideIssue(
                issue_id=str(uuid.uuid4()),
                issue_type=IssueType.DATA_COMPLETENESS,
                severity=Severity.MEDIUM,
                description=f"Missing essential form data fields: {', '.join(missing_fields)}",
                affected_records=[record.race_id],
                recommendation=f"Ensure all essential form data is collected: {', '.join(missing_fields)}",
                metadata={'missing_fields': missing_fields, 'dog_name': record.dog_name}
            )
            issues.append(issue)
        
        return issues
    
    def validate_consistency(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """Validate consistency across records."""
        issues = []
        
        # Group records by race
        race_groups = {}
        for record in records:
            if record.race_id not in race_groups:
                race_groups[record.race_id] = []
            race_groups[record.race_id].append(record)
        
        # Check for inconsistent model versions within races
        for race_id, race_records in race_groups.items():
            model_versions = set(r.model_version for r in race_records)
            
            if len(model_versions) > 1:
                issue = GuideIssue(
                    issue_id=str(uuid.uuid4()),
                    issue_type=IssueType.DATA_CONSISTENCY,
                    severity=Severity.HIGH,
                    description=f"Inconsistent model versions in race {race_id}",
                    affected_records=[race_id],
                    recommendation="Ensure all predictions in a race use the same model version",
                    metadata={'model_versions': list(model_versions)}
                )
                issues.append(issue)
        
        return issues


class StatisticalPredictionQAAnalyzer(PredictionQAAnalyzer):
    """Example implementation for prediction quality analysis."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_predictions(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """Perform comprehensive prediction analysis."""
        issues = []
        
        # Check accuracy patterns
        accuracy_issues = self.check_accuracy_patterns(records)
        issues.extend(accuracy_issues)
        
        # Detect anomalies
        anomaly_issues = self.detect_anomalies(records)
        issues.extend(anomaly_issues)
        
        self.logger.info(f"Found {len(issues)} prediction quality issues")
        return issues
    
    def check_accuracy_patterns(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """Check for accuracy patterns."""
        issues = []
        
        # Check for suspiciously high confidence with extreme predictions
        suspicious_records = []
        for record in records:
            if record.confidence > 0.9 and (record.prediction_value < 0.1 or record.prediction_value > 0.9):
                suspicious_records.append(record.race_id)
        
        if len(suspicious_records) > len(records) * 0.2:  # More than 20% suspicious
            issue = GuideIssue(
                issue_id=str(uuid.uuid4()),
                issue_type=IssueType.PREDICTION_ACCURACY,
                severity=Severity.HIGH,
                description="High number of extreme predictions with high confidence",
                affected_records=suspicious_records[:10],  # Limit to first 10
                recommendation="Review model calibration and prediction thresholds",
                metadata={'suspicious_count': len(suspicious_records), 'total_count': len(records)}
            )
            issues.append(issue)
        
        return issues
    
    def detect_anomalies(self, records: List[PredictionRecord]) -> List[GuideIssue]:
        """Detect statistical anomalies."""
        issues = []
        
        if not records:
            return issues
        
        # Calculate prediction value statistics
        pred_values = [r.prediction_value for r in records]
        mean_pred = sum(pred_values) / len(pred_values)
        
        # Check for unusual distribution (all predictions clustered around mean)
        variance = sum((x - mean_pred) ** 2 for x in pred_values) / len(pred_values)
        
        if variance < 0.01:  # Very low variance suggests lack of discrimination
            issue = GuideIssue(
                issue_id=str(uuid.uuid4()),
                issue_type=IssueType.ANOMALY_DETECTION,
                severity=Severity.MEDIUM,
                description="Predictions show unusually low variance (lack of discrimination)",
                affected_records=[r.race_id for r in records[:5]],  # Sample affected records
                recommendation="Review model features and training data for discriminative power",
                metadata={'variance': variance, 'mean_prediction': mean_pred}
            )
            issues.append(issue)
        
        return issues


class StandardAdvisoryDispatcher(AdvisoryDispatcher):
    """Example implementation for advisory generation and dispatch."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.reporters = []
    
    def generate_advisories(self, issues: List[GuideIssue]) -> List[AdvisoryMessage]:
        """Generate advisories from issues."""
        prioritized_issues = self.prioritize_issues(issues)
        advisories = []
        
        for issue in prioritized_issues:
            advisory = self.create_advisory(issue)
            advisories.append(advisory)
        
        self.logger.info(f"Generated {len(advisories)} advisories from {len(issues)} issues")
        return advisories
    
    def prioritize_issues(self, issues: List[GuideIssue]) -> List[GuideIssue]:
        """Prioritize issues by severity and impact."""
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1, 
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        
        return sorted(issues, key=lambda x: (
            severity_order[x.severity],
            -len(x.affected_records)  # More affected records = higher priority
        ))
    
    def create_advisory(self, issue: GuideIssue) -> AdvisoryMessage:
        """Create advisory message from issue."""
        # Generate action items based on issue type
        action_items = self._generate_action_items(issue)
        
        return AdvisoryMessage(
            advisory_id=str(uuid.uuid4()),
            title=f"{issue.issue_type.value.replace('_', ' ').title()} Issue Detected",
            summary=issue.description,
            details=f"Recommendation: {issue.recommendation}",
            severity=issue.severity,
            action_items=action_items,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)  # Expire in 7 days
        )
    
    def _generate_action_items(self, issue: GuideIssue) -> List[str]:
        """Generate action items based on issue type."""
        base_actions = [issue.recommendation]
        
        if issue.issue_type == IssueType.DATA_COMPLETENESS:
            base_actions.extend([
                "Review data collection processes",
                "Implement data validation checks",
                "Update data quality monitoring"
            ])
        elif issue.issue_type == IssueType.PREDICTION_ACCURACY:
            base_actions.extend([
                "Review model performance metrics",
                "Consider model retraining",
                "Validate prediction thresholds"
            ])
        elif issue.issue_type == IssueType.ANOMALY_DETECTION:
            base_actions.extend([
                "Investigate root cause of anomaly",
                "Review recent system changes",
                "Consider additional monitoring"
            ])
        
        return base_actions
    
    def dispatch_to_reporters(self, advisories: List[AdvisoryMessage]) -> None:
        """Dispatch advisories to reporters."""
        self.logger.info(f"Dispatching {len(advisories)} advisories to reporters")
        # In a real implementation, this would coordinate with output reporters


class ConsoleOutputReporter(OutputReporter):
    """Example implementation for console output."""
    
    def format_advisory(self, advisory: AdvisoryMessage) -> str:
        """Format advisory for console output."""
        lines = [
            f"[{advisory.severity.value.upper()}] {advisory.title}",
            f"Summary: {advisory.summary}",
            f"Details: {advisory.details}",
            f"Action Items:",
        ]
        
        for item in advisory.action_items:
            lines.append(f"  - {item}")
        
        lines.append(f"Created: {advisory.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 50)
        
        return "\n".join(lines)
    
    def output_advisories(self, advisories: List[AdvisoryMessage]) -> bool:
        """Output advisories to console."""
        try:
            print("=== GPTAssistant Analysis Results ===\n")
            
            for advisory in advisories:
                print(self.format_advisory(advisory))
                print()
            
            return True
        
        except Exception as e:
            print(f"Error outputting to console: {str(e)}")
            return False
    
    def get_output_destination(self) -> str:
        """Get output destination."""
        return "console"


class FileOutputReporter(OutputReporter):
    """Example implementation for file output."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
    
    def format_advisory(self, advisory: AdvisoryMessage) -> str:
        """Format advisory for file output (JSON)."""
        return json.dumps(advisory.to_dict(), indent=2)
    
    def output_advisories(self, advisories: List[AdvisoryMessage]) -> bool:
        """Output advisories to file."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump([adv.to_dict() for adv in advisories], f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error writing to file {self.output_file}: {str(e)}")
            return False
    
    def get_output_destination(self) -> str:
        """Get output destination."""
        return f"file:{self.output_file}"


# Example usage function
def create_example_gpt_assistant() -> 'GPTAssistant':
    """Create an example GPTAssistant with all components."""
    from gpt_assistant_interfaces import GPTAssistant
    
    # Create components
    log_ingestor = JSONLogIngestor()
    form_validator = BasicFormGuideValidator()
    qa_analyzer = StatisticalPredictionQAAnalyzer()
    advisory_dispatcher = StandardAdvisoryDispatcher()
    
    # Create reporters
    reporters = [
        ConsoleOutputReporter(),
        FileOutputReporter("gpt_assistant_report.json")
    ]
    
    # Create and return GPTAssistant
    return GPTAssistant(
        log_ingestor=log_ingestor,
        form_validator=form_validator,
        qa_analyzer=qa_analyzer,
        advisory_dispatcher=advisory_dispatcher,
        reporters=reporters
    )
