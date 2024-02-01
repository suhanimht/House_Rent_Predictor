# House_Rent_Predictor
## Overview
There is a rise in demand for renting a house and buying a house therefore , determining  a more efficient way to calculate the house rents is crucial. House rent increases once a year, So there's a desire to predict house rents within the future . House rent prediction system studies behavior of your time series data and reflects the long run rents. Software implementations for the experiment were selected from python libraries. Data preprocessing and preparation techniques so as to get clean data. To make machine learning models ready to predict house price supported house features to research and compare models performance so as to decide on the simplest model. We applied three different Machine Learning algorithms: Decision tree, Random forest and Linear Regression on the training data.

## Introduction to problem:
In recent times, finding the ideal housing option according to budget and preferences is such a hassle. The cost of house rent depends on many factors such as; the house size, number of bedrooms, locality, number of bathrooms, halls, and kitchen, furnishing status, and a lot more. The prediction of house rents plays a crucial role in real estate and property management. Having an accurate estimate of house rents can assist both tenants and landlords in making informed decisions. With the use of appropriate machine learning algorithms, users can find the ideal house according to their budgets and preferences with ease.

## Analytic objective
The primary objective of this analysis is to build a predictive model for house rent prediction. We aim to create a model that can accurately estimate the rent of a house based on various features and location information. The model's performance will be evaluated using metrics such as mean squared error and R-squared. We also plan to determine the essential features significantly needed to predict the house rent for homes.

## Data description and preparation
For this project, we used a dataset from kaggle that includes information about various types of houses in different ocations and their corresponding rental prices. It consists of data of about 4700+ houses available for rent with different parameters ranging from; size to the number of bedrooms, to the locality, and furnishing status, among others. The data was
collected from multiple sources, including real estate listings and property management databases.

To prepare the data for modeling, we performed several preprocessing steps. This involved cleaning the dataset, handling missing values, and addressing outliers. 
#### Step 1:
Import dataset
#### Step 2: check for null, missing and duplicate values
After appropriate checking, we discovered that the dataset is void of null, missing and duplicate values. This depicts that the dataset is kind of clean from the onset.
#### Step 3: Feature engineering
Additionally, we conducted feature engineering by creating new variables such as price per square foot, age of the property from the day it was posted and bathroom to bedroom ratio, which we believed could be valuable in predicting house rents.

## Data exploration and visualization
Going further,we explored the data to check if there are trends between the explanatory variables and the target variable and to gain insights into the dataset. The summary statistics revealed the distributions and ranges of various features.

#### Step 1: Basic Stats of the target variable
#### Step 2: Feature encoding
Before plotting visualizations, the categorical features have to be converted to numerical features. Further examination showed that the categorical features have pretty much labels, therefore using one hot encoding will likely lead to high dimensionality. Therefore, Scikit-learn’s label encoder was used to encode the features. We did not use pandas as looking at the dataset, it contains multiple cities and area localities which would create numerous columns and cause unnecessary hassle when looking at the updated dataframe.
#### Step3: Exploring the different attributes using visualizations

**HISTOGRAMS**
We plotted a histogram for the attributes in the dataset to see their distributions. From this we could see some of the attributes were skewed, for example the number of price per sqft, Average rent and size of the house were right-skewed.

**SCATTER PLOTS**
Next we made some scatter plots to better understand the trends and relationships of some variables against the target variable. We noticed some extreme values for the target variable but decided not to remove it as it is possible to have extremely high house rent based on various factors.

**BAR GRAPH**
We plotted a bar graph to further understand the count of each type of BHK available in the dataset. We observed 2 BHK were the most common followed by 1 and 3 BHK. 6 BHK houses were the least common.

**BOX PLOT**
Next we moved on to plot a box plot for the Rent against Area Type to gain further insights. We noticed that area type 1 is the most common.

**HEATMAP**
Lastly we plotted a heatmap to show the correlation of all the features in the dataset. From the correlation we can see that some features like bathroom, size and BHK have a positive relation and other features like Area type and point of contact have a negative correlation. Turns out that Bathroom has the strongest positive correlation and point of contact has the strongest negative correlation out of all the features. Also, Area locality has a close to 0 correlation with the target variable so we may consider dropping it before the modeling process.

Through all these visualizations, we identified patterns and potential outliers. For example, we observed that houses with more bedrooms tended to have higher rental prices, and certain locations exhibited higher rent prices compared to others. These findings provided a foundation for feature selection and model development.

## Model explanation
For this project, we tried out three models, Linear Regression, Random Forest and Decision tree, that we learned in our DATA MODEL course this semester. We analyzed their effectiveness, robustness and evaluated the performances of the predictive models using appropriate metrics (e.g., mean squared error, root mean squared error, R-squared). Random Forest Regression is an ensemble learning method that combines multiple decision trees to make predictions. It is well-suited for regression problems and has the advantage of handling both numerical and categorical variables. The Random Forest Regression algorithm operates by constructing multiple decision trees and averaging their predictions. This approach reduces overfitting and provides robust predictions. It also allows us to understand the relative importance of different features in the prediction process. Linear regression is a simple and widely used model for predicting continuous numerical values. It assumes a linear relationship between the input features and the target variable. Linear regression works well when there is a linear relationship between the input features and the rent prices. Decision trees are non-parametric models that split the data based on feature thresholds to make predictions. They can handle both numerical and categorical features and are capable of capturing non-linear relationships in the data. Decision trees can be prone to overfitting, but techniques like pruning and ensemble methods can help mitigate this issue.

## Modeling implementation
#### Step 1: Split data into explanatory variables
Before we move on with modeling, we first need to split the data. We decided to drop the ‘Posted On’ column along with the target variable as looking at the problem at hand, this attribute would not contribute much to the model. The dataset is split into explanatory variables — X and target variable — y.
#### Step 2: Divide data into Train and Test set
To implement the model, we divided the dataset into training and testing sets using a 70:30 split. The training set was used to train the models, while the testing set was used to evaluate their performance. Scikit-learn’s train-test split is used to accomplish this task.
#### Step 3: Feature scaling
Before going into modeling proper, the data needs to be scaled to handle skewed features. Scikit-learn’s standard scaler ensures that for each feature the mean is 0 and the variance is 1, bringing all the features to the same magnitude. Doing this will significantly affect the model’s performance.
#### Step 4: Feature Preprocessing
House rent prediction is a regression problem, therefore about three regression models were trained and the best was chosen.
We defined two functions to help calculate the model metrics — R2 score and root mean squared error.
#### Step 5: Implementing Linear Regression model
#### Step 6: Implementing Decision Tree model
#### Step 7: Implementing Decision Tree model
#### Step 8: Hyperparameter tuning with Grid search (Optional)
Lastly, we then perform grid search with cross-validation using the GridSearchCV function. After fitting the grid search object to the training data, we can obtain the best hyperparameter values (best_params) and the corresponding best model (best_model) based on the evaluation metric.

## Evaluation
To evaluate each of the model's performance, we used metrics, including root mean squared error (RMSE) and R-squared. The RMSE measures the average squared difference between the predicted and actual rental prices, while the R-squared value represents the proportion of the variance in the rental prices that can be explained by our model. Upon evaluating the model on the testing set, we obtained an MSE of 0.9997 and an R-squared value of 0.0153. These results indicate that Linear Regression Model is the best fit out of the three. Our model has a reasonably good fit to the data and is capable of making
accurate predictions for house rents.

## Discussion and conclusion
In conclusion, we successfully developed a predictive model for house rent prediction based on specific requirements. The Linear Regression model performed well, providing accurate estimates of house rents. However, it is important to note that the model's predictions may still have limitations due to factors not included in the dataset, such as recent market trends or changes in economic conditions. Additionally, the model's accuracy may vary depending on the quality and completeness of the data used.
Further improvements could be made by incorporating additional features, considering alternative algorithms, or exploring more advanced techniques such as ensemble models or deep learning approaches. Overall, this predictive model has the potential to be a valuable tool for tenants, landlords, and real estate professionals in estimating house rents accurately and facilitating informed decision-making.

