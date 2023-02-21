import math

import numpy as np

from classification_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    
    expected_first_prediction_value = 1.0
    expected_no_predictions = 262

    
    result = make_prediction(input_data=sample_input_data)

    
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert math.isclose(predictions[0], expected_first_prediction_value)


# Path: AkashDeshwani_titanic-assignment\Production_lab_Assignment_2\tests\test_prediction.py