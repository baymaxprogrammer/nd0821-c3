from fastapi.testclient import TestClient
from main import app

# Call Test-Client to run normal python tests
client = TestClient(app)


# Testing the main page message
def test_get_function():
    result = client.get("/")
    try:
        # Checking the status code (success is 200)
        assert result.status_code == 200
        # Checking the json message
        assert result.json() == {
            "message": "Welcome User! This is an app to predict whether or not someone's income will exceed $50,"
                       "000/year."
        }
    except AssertionError as err:
        print("GET function message is not what it is supposed to be!")
        raise err


# Testing the predict function and its message
def test_post_predict_below_50k():
    result = client.post("/predict", json={
        "age": 42,
        "workclass": 'Self-emp-not-inc',
        "fnlgt": 37618,
        "education": 'Some-college',
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Farming-fishing",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": 'United-States'
    })

    assert result.status_code == 200
    assert result.json() == {"Income category is: ": "<=50K"}


# Testing the predict function and its message
def test_post_predict_above_50k():
    result = client.post("/predict", json={
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
    })

    assert result.status_code == 200
    assert result.json() == {"Income category is: ": ">50K"}
