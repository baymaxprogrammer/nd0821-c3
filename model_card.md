# Model Card for Deploying a Machine Learning Model on Heroku with FastAPI


For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is based on a randomforest classifier and uses the following features to determine the salary of 
a person is more than 50k or not:

* age
* workclass
* fnlgt
* education
* education-num
* marital-status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week
* native-country


## Intended Use
The intended use of the model is to predict salary (is it more that 50k or not) based on the attributes mentioned above. This is a student/course level 
exercise and should NOT be used for any production purposes.

## Training Data
The data and related information (primary use, etc.) can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data
In this study, we used 20% of the data for testing and evaluation using `random_seed=42`

## Metrics
Here we selected the following metrics to measure the accuracy of the model:
* **Precision**:  0.74 
* **Recall**:  0.64
* **Fbeta**:  0.69

## Ethical Considerations
* Relying on features such as sex, ethnicity and race: These features, although meaningful, might not be the best ones 
to rely on classifying someone's income. Consider if a hiring manager asks if they should offer above or below 50k 
for a job, then the system will be biased towards minorities as they historically earned less dollar.

## Caveats and Recommendations
* Lack of important features: Our classifier does not rely on geographical information for this study. This is limiting 
in the sense that we may over generalize the income threshold for small and big cities altogether.
