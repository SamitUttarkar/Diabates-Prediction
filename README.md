![🏥_Diabetes_Prediction](https://user-images.githubusercontent.com/67644483/215288160-0703eb4e-e009-4dd8-9710-0c33ce109d74.png)

# Task
- Explain the issues in the data with data cleaning and preperation
- Predicting if the patient is going to develop diabetes based on three or more children indicator i.e. if the mother has more than three children or not and then calculating the probability of developing diabetes given the mother has more than three children and vice verca
- Predicting if the patient is going to develop diabetes based on multiple parameters and choosing the best model to predict it using ToPredict dataset as testing dataset

## Data Preperation
The first task is the analyse the data and performe some data cleaning steps 

<img width="793" alt="Screenshot 2023-01-28 at 8 02 20 PM" src="https://user-images.githubusercontent.com/67644483/215288372-d780a6b0-69f0-4e13-b617-78277443d413.png">

-From the above figure we can visualise that there are many zeros in the column Insulin and Skinthickness. It is not possible to get Insulin and Skinthickness as zero therefore I decided the drop these columns as they will not be useful in the prediction process
- We also replace the missing values with the median for the rest of the columns 

## EDA 
First, let's check for the correlation between the parameters 

![Screenshot 2023-01-28 at 8 12 25 PM](https://user-images.githubusercontent.com/67644483/215288770-b1cec56c-71e0-418e-85e8-6b679e8b52de.png)

The correlation between the paramters is not higher than 0.7 which is good as it will help in predicting.

Now, for the bivariate analysis

![Screenshot 2023-01-28 at 8 10 57 PM](https://user-images.githubusercontent.com/67644483/215288733-b0c89c2c-c31d-41a5-9e12-df2b3c093557.png)

From the pairplot it's difficult the classify based on the scatterplots

## Machine Learning

### Predicting using Three or more kids parameter

In the first step we will create a column called threeormore which indicated wether the patient has more than three children or not 
Then we calculate probability after the model fitting using logistic regression 

![Screenshot 2023-01-28 at 8 16 55 PM](https://user-images.githubusercontent.com/67644483/215288927-2fdaac45-1708-44e9-96bd-8b5f7b13cce9.png)

Bayes rule was used to calculate probability for this step

### Predicting using multiple paramateres

First we need to check which model performs the best 

![Screenshot 2023-01-28 at 8 19 17 PM](https://user-images.githubusercontent.com/67644483/215289018-ec5d0a34-cae3-4998-94aa-2559ef90eb53.png)

We can see that logistic regression performs better than DecisionTreeRegressor therefore we will use it for our further prediction


#### **Confusion Matrix using all the parameters for training**

![Screenshot 2023-01-28 at 8 20 17 PM](https://user-images.githubusercontent.com/67644483/215289098-19851516-1ad5-4524-85d9-74b4e4188db3.png)


#### **Feature Selection**

![Screenshot 2023-01-28 at 8 22 50 PM](https://user-images.githubusercontent.com/67644483/215289185-65741894-f9a7-46b9-a1a8-9d8c86b53cd5.png)

![Screenshot 2023-01-28 at 8 22 59 PM](https://user-images.githubusercontent.com/67644483/215289191-e07b1f28-df2e-47c6-9d4b-f3a96f47d3d5.png)


#### **Final Probability of getting Diabetes**
![Screenshot 2023-01-28 at 8 24 13 PM](https://user-images.githubusercontent.com/67644483/215289251-378dc60e-bdd1-44ff-8f77-6c53efaa6864.png)

## About the dataset
**The data used to create the dataset PimaDiabetes.cv, which is used in the coursework, was originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases in the United States. It includes a 0/1 variable, Outcome, which indicates whether the subject ultimately tested positive for diabetes, along with a list of numerous diagnostic measures recorded from 750 women.**



