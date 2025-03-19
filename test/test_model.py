import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path so that app.py can be imported
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Register the custom class to avoid pickle errors
from app import Normal, load_model_mlflow, load_scaler, app
import __main__
__main__.Normal = Normal

#############################
# Endpoint Tests Using Flask Test Client
#############################

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage_response(client):
    response = client.get('/')
    assert response.status_code == 200
    print("Homepage Test Passed")

def test_a1_page(client):
    response = client.get('/a1')
    assert response.status_code == 200
    print("a1 Page Test Passed")

def test_a2_page(client):
    response = client.get('/a2')
    assert response.status_code == 200
    print("a2 Page Test Passed")

def test_a3_page(client):
    response = client.get('/a3')
    assert response.status_code == 200
    print("a3 Page Test Passed")

def test_receive_data_post_response(client):
    data = {
        'engine': '1500',
        'max_power': '100',   # Use underscore to match your app.py field name
        'year': '2010',
        'km_driven': '50000',
        'mileage': '15.0'
    }
    response = client.post('/receive_data', data=data)
    assert response.status_code == 200
    print("Receive Data Test Passed")

def test_receive_data_v2_post_response(client):
    data = {
        'engine': '1500',
        'max_power': '100',
        'year': '2010',
        'km_driven': '50000',
        'mileage': '15.0'
    }
    response = client.post('/receive_data_v2', data=data)
    assert response.status_code == 200
    print("Receive Data v2 Test Passed")

def test_receive_data_v3_post_response(client):
    data = {
        'engine': '1500',
        'max_power': '100',
        'year': '2010',
        'km_driven': '50000',
        'mileage': '15.0',
        'model_switch': 'ridge'
    }
    response = client.post('/receive_data_v3', data=data)
    assert response.status_code == 200
    print("Receive Data v3 Test Passed")

#############################
# Model Input/Output Tests
#############################

def get_transformed_input():
    """
    Create dummy input with 5 features, add an intercept column,
    and transform using the scaler (from the MLflow branch).
    """
    input_features = np.array([[2010, 50000, 15.0, 1500, 100]])
    input_with_intercept = np.concatenate(
        (np.ones((input_features.shape[0], 1)), input_features), axis=1
    )
    scaler = load_scaler()
    transformed = scaler.transform(input_with_intercept)
    return transformed

def test_model_accepts_expected_input():
    transformed_input = get_transformed_input()
    model_instance = load_model_mlflow(False)  # Change parameter if needed
    try:
        model_instance.predict(transformed_input)
    except Exception as e:
        pytest.fail(f"Model failed to process the expected input: {e}")

def test_model_output_shape():
    transformed_input = get_transformed_input()
    model_instance = load_model_mlflow(False)
    prediction = model_instance.predict(transformed_input)
    # Expect a single prediction with shape (1,)
    assert prediction.shape == (1,), f"Expected shape (1,), got {prediction.shape}"

if __name__ == '__main__':
    pytest.main()
