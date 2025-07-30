
import json
import os
from pathlib import Path
import pandas as pd

def validate_predictions_and_betting():
    """
    Parses all prediction JSON files, checks for data integrity issues,
    and validates betting suggestion logic against documented strategy.
    """
    prediction_dir = Path('predictions')
    report_path = Path('reports/ml_output_issues.md')
    issues = []

    if not prediction_dir.exists():
        print("No predictions directory found.")
        return

    prediction_files = list(prediction_dir.glob('**/*.json'))

    for file_path in prediction_files:
        if not os.path.getsize(file_path) > 0:
            issues.append(f"- **{file_path.name}**: Empty file.")
            continue
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                issues.append(f"- **{file_path.name}**: Invalid JSON format.")
                continue

            predictions = None
            if 'predictions' in data and isinstance(data['predictions'], list):
                predictions = data['predictions']
            elif 'summary' in data and 'predictions' in data['summary'] and isinstance(data['summary']['predictions'], list):
                predictions = data['summary']['predictions']

            if predictions is None:
                issues.append(f"- **{file_path.name}**: Missing or invalid 'predictions' list.")
                continue

            ranks = set()
            for i, prediction in enumerate(predictions):
                dog_name = prediction.get('dog_name', f'Unknown Dog at index {i}')
                
                for key in ['win_probability', 'place_probability', 'prediction_score', 'final_score', 'win_prob', 'place_prob']:
                    if key in prediction:
                        prob = prediction[key]
                        if prob is not None and (not isinstance(prob, (int, float)) or not (0 <= prob <= 1)):
                            issues.append(f"- **{file_path.name}**: Invalid `{key}` for **{dog_name}**. Value: `{prob}` (not in [0, 1]).")
                        if pd.isna(prob):
                            issues.append(f"- **{file_path.name}**: NaN value for `{key}` for **{dog_name}**")

                if 'predicted_rank' in prediction:
                    rank = prediction['predicted_rank']
                    if rank in ranks:
                        issues.append(f"- **{file_path.name}**: Duplicate rank `{rank}` for **{dog_name}**.")
                    ranks.add(rank)

            # Betting logic validation
            if 'bet_suggestions' in data and isinstance(data['bet_suggestions'], list):
                for suggestion in data['bet_suggestions']:
                    bet_type = suggestion.get('type')
                    win_prob = suggestion.get('win_probability')
                    value = suggestion.get('value')
                    odds = suggestion.get('odds')

                    if bet_type == 'Win' and win_prob is not None and win_prob <= 0.25:
                        issues.append(f"- **{file_path.name}**: Low win probability ({win_prob}) for Win bet suggestion.")
                    
                    if value is not None and odds is not None and value <= odds:
                        issues.append(f"- **{file_path.name}**: Value ({value}) is not greater than odds ({odds}) for bet suggestion.")


    # Generate a clean report
    if issues:
        # Remove duplicates
        unique_issues = sorted(list(set(issues)))
        report_content = "## ML Output & Betting Logic Issues Report\n\n"
        report_content += "\n".join(unique_issues)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"Found {len(unique_issues)} unique issues. Report generated at {report_path}")
    else:
        print("No issues found in ML prediction outputs or betting logic.")
        if report_path.exists():
            os.remove(report_path)

if __name__ == '__main__':
    validate_predictions_and_betting()

