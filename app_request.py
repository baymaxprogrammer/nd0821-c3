"""
request app to fetch results directly using api without web access
"""
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# example
data_sample = {
    "age": 61,
    "workclass": 'Private',
    "fnlgt": 195453,
    "education": 'HS-grad',
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": 'United-States'
}

app_url = "https://udacity-deploy-pred.herokuapp.com/predict"

result = requests.post(app_url, json=data_sample)
assert result.status_code == 200

logging.info("Testing Heroku app using ''request'' function")
logging.info(f"Status code (success=200): {result.status_code}")
logging.info(f"Response body (success= '>50K'): {result.json()}")