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
    
    validation_result = {
        "valid": False,
        "errors": [],
        "warnings": []
    }

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check required fields in top level
        for field, field_type in required_fields.items():
            if field not in data:
                validation_result["errors"].append(f"Missing required field: {field}")
            elif not isinstance(data[field], field_type):
                validation_result["errors"].append(f"Invalid field type for {field}: expected {field_type.__name__}")

        # Check each prediction if predictions exist
        if "predictions" in data and isinstance(data["predictions"], list):
            for i, prediction in enumerate(data["predictions"]):
                for field, field_type in prediction_fields.items():
                    if field not in prediction:
                        validation_result["errors"].append(f"Missing field in prediction {i}: {field}")
                    elif not isinstance(prediction[field], field_type):
                        validation_result["errors"].append(f"Invalid field type in prediction {i} for {field}: expected {field_type.__name__}")
                        
                # Check for outcome fields in reasoning
                reasoning_text = prediction.get("reasoning", "")
                for outcome_field in outcome_fields:
                    if outcome_field in reasoning_text:
                        validation_result["errors"].append(f"Prediction {i} reasoning contains outcome field: {outcome_field}")

                # Ensure no outcome field in feature list (if applicable)
                for feature in prediction:
                    if feature in outcome_fields:
                        validation_result["errors"].append(f"Prediction {i} contains outcome feature: {feature}")
        
        # Set validation status
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        if validation_result["valid"]:
            print("Validation passed!")
        else:
            print(f"Validation failed with {len(validation_result['errors'])} errors")
            
        return validation_result
    
    except json.JSONDecodeError as e:
        validation_result["errors"].append(f"Error decoding JSON: {str(e)}")
        print("Error decoding JSON.")
        return validation_result
    except FileNotFoundError:
        validation_result["errors"].append(f"File not found: {file_path}")
        print(f"File not found: {file_path}")
        return validation_result
    except Exception as e:
        validation_result["errors"].append(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {e}")
        return validation_result


