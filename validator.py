import json


def validate_output(file_path):
    required_fields = {
        "success": bool,
        "race_info": dict,
        "predictions": list
    }
    prediction_fields = {
        "dog_name": str,
        "box_number": int,
        "win_prob_raw": float,
        "win_prob_norm": float,
        "final_score": float,
        "confidence_level": float,
        "reasoning": str
    }
    outcome_fields = {"PLC", "BON", "finish_position"}

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check required fields in top level
        for field, field_type in required_fields.items():
            if field not in data or not isinstance(data[field], field_type):
                raise ValueError(f"Missing or invalid field: {field}")

        # Check each prediction
        for prediction in data["predictions"]:
            for field, field_type in prediction_fields.items():
                if field not in prediction or not isinstance(prediction[field], field_type):
                    raise ValueError(f"Missing or invalid field in prediction: {field}")
                    
            # Check for outcome fields in reasoning
            reasoning_text = prediction.get("reasoning", "")
            for outcome_field in outcome_fields:
                if outcome_field in reasoning_text:
                    raise ValueError(f"Reasoning contains outcome field: {outcome_field}")

            # Ensure no outcome field in feature list (if applicable)
            for feature in prediction:
                if feature in outcome_fields:
                    raise ValueError(f"Prediction contains outcome feature: {feature}")

        print("Validation passed!")
    
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Example usage
validate_output('output.json')

