## ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Reddit Post Classification
Author: Lee Wan Xian

---
### Executive Summary

The purpose of this project is to develop a machine learning (ML) classification model to assist a internet company in classifying forum posts to the streaming services (Netflix and DisneyPlus). There are 2 objectives in this project report. Firstly, it is to create a ML classification model that can predict and classify the relevant streaming service based on unique text features in Reddit posts. Next, it is to analyse user sentiments on famous shows in Reddit.<br> 

This report has 5 jupyter notebooks in total. Book 1 covers Data Collection. Book 2 covers Data Cleaning & Exploratory Data Analysis. Book 3 covers Preprocessing & Vectorization. Book 4 covers ML Modeling. Book 5 covers Sentiment Analysis, Conclusion & Recommendation. Do ensure to run these notebooks in an environment that has the relevant python packages/libraries stated in [dependancies.yml](./dependancies.yml) file.<br>

This report shows that the best performing classification model is Logistic Regression with TF-IDF Vectorization. To add on, joyful or surprising posts garner more user activity in Reddit, as compared to angry or disgusting posts.<br>

In conclusion, we recommend that the <font color="blue">Logistic Regression model</font> should be used to classify posts in a binary class classification scenario (i.e. Disneyplus or Netflix).

---
### Problem Statement

Our client is a firm that runs a streaming service discussion website. As part of their initiative to build an inhouse label tagging algorithm, they have tasked us to develop a machine learning (ML) classification model that tags posts to the right streaming service tag. Meanwhile, the client is also interested in users' sentiments towards famous shows. That way, they can evaluate how to improve their search and homepage recommendations for users.

---
### Data Dictionary

Parameter|Datatype|Description
---|---|---
subreddit|String|The subreddit where the post was scraped from
title|String|Title of the post
selftext|String|Body of the post
is_video|Boolean|True: The post contains video content<br>False: The post does not contain video content
created_utc|Integer|Unix epoch time of when the post was created

---
### Model Evaluation

Model|Vectorization|Accuracy (Train)|Accuracy (Test)
---|---|---|---
Naive Bayes (Baseline)|Count|0.7675|0.7748
Naive Bayes (Tuned)|Count|0.7665|0.7748
Extra Trees Classifier|TF-IDF|0.7196|0.7925
**Logistic Regression**|**TF-IDF**|**0.7837**|**0.7950**
SVM-Linear Kernel|TF-IDF|0.7678|0.7916|0.7916
Random Forest Classifier|TF-IDF|0.6888|0.7792
Naive Bayes|TF-IDF|0.7677|0.7770

For this classification problem, **accuracy score** is the most important metric. The model should be predict both DisneyPlus and Netflix posts as accurate as possible. In view of this, we will compare the accuracy score for all 7 models done in both training and test data.<br>

From the table above, the baseline model performed quite well in general (accuracy score was well above 0.5). The accuracy score in train set is lower than that of test set with a difference of around 0.01. This shows that the model is neither overfitted nor underfitted. It is noted that all of these models do not show signs of overfitting since all of their accuracy (test) scores are higher than accuracy (train) scores.<br>

Logistic Regression model with TF-IDF vectorization performed the best out of the 7 models. Even though accuracy (test) scores are similar amongst logistic regression, extra trees classifier and SVM-linear kernel models, the accuracy (train) score of logistic regression (0.7837) is significantly higher than the rest.

Model|Accuracy|F1 score|Recall|Precision|Specificity|AUC
---|---|---|---|---|---|---
**Logistic Regression**|**0.783**|**0.787**|**0.794**|**0.781**|**0.780**|0.871
SVM-Linear Kernel|0.767|0.760|0.716|0.810|0.834|0.000
Extra Trees Classifier|0.719|0.786|0.802|0.771|0.765|0.813

Considering all other metrics, the overall metrics for Logistic Regression model is the most balanced. All of them are at least above 0.75 and the variance across all these metrics is very small (less than 0.01). There is a very high likelihood that the logistic regression model fits well with the data on hand and still be able to fit well in unseen data. This means that Logistic Regression is able to predict the correct streaming service to the post for both DisneyPlus and Netflix with an accuracy of 0.783.

A limitation of Logistic Regression is that it assumes the features have at most moderate multicollinearity with each other in order to perform well. This might be hard to achieve with text words in real world scenario.

---
### Sentiment Analysis Summary

Most posts have a neutral sentiment towards all of the shows. It can be inferred that most of these posts are discussion about the show details and not expressing any strong sentiments. Excluding neutral sentiment, surprise was the next common emotion with Stranger Things being the most discussed show amongst the 6.<br>

The order of most common emotion shown in the subreddits are as follow:
1. Neutral
2. Surprise
3. Joy
4. Sadness
5. Fear
6. Anger
7. Disgust

Looking at the perspective of respective shows and excluding neutral sentiment, Joy is the most common emotion for Squid Game, Star Wars and Moon Knight. Surprise is the most common emotion for Stranger Things. Sadness is the most common emotion for Better Call Saul and Black Widow.

---
### Conclusion

**Classification Model**

Logistic Regression is the most suitable model to use for classifying posts to 2 different streaming services (DisneyPlus and Netflix). This model is able to predict the correct streaming service label tags better than leaving the label tagging by chance. Leaving the tagging by chance has a probabilty of it being correct at around 0.5.<br>

Reasons are stated as per below:
1. Accuracy score has consistently fall within the range of 0.78 to 0.795.
2. Accuracy score is well above 0.5, which is the rough probability of the label tagging being correct, if left to chance.
3. Recall score and Precision score do not differ too much (Difference of 0.01). This means that the model performs consistently with any data corpus in predicting Netflix posts.
4. Specificity score is very close to Recall score. This implies that the model performs consistently towards labeling DisneyPlus posts as well as Netflix posts.
5. AUC metric is relatively close to 1. This implies that the model fits well with the data corpus on hand.

**Sentiment Analysis**

Stranger Things is the most discussed show in the forum with most posts expressing surprise, followed by joy. Given that surprise and joy are generally the most common emotions across the shows, it shows most users prefer to publish joyful or surprising posts of their favourite shows. Posts filled with anger or disgust are unfavorable to users and are rarely published in Reddit.

**Limitation/Further Improvements**

Our model might not be able to predict well when there are more than 2 classes. For example, if we need to classify posts based on Netflix, DisneyPlus and Amazon Prime Video, we will need more data on posts related to Amazon Prime Videos to retrain the model. In addition, other classification models (i.e. k-Nearest Neighbors Classifier) might perform much better under multi-class classification scenarios.<br>
We restricted the maximum number of word vector features at 2500 when piping them into our model for training. The performance of the model might be different if we chose to increase or decrease the maximum number of features. We can run multiple experiments with varying maximum number of features on the baseline model to pinpoint the optimal maximum number of features.<br>
A limitation of Logistic Regression is that it assumes the features have at most moderate multicollinearity with each other in order to perform well. This might be hard to achieve with text words in real world scenario.<br>

Our sentiment analysis is quite rudimentary in nature as we analysed the title text. Ideally, we can run analysis in the body text to understand the intensity of emotion in each posts. Sentiment analysis on images and videos in the posts also can provide more signal to the model on the key emotion expressed.

---
### Recommendation

Going back to the problem statement, we recommend that the logistic regression model should be used for tagging posts to the right streaming service platform. This model would perform well when it needs to tag posts to either DisneyPlus or Netflix.

We recommend that the client should promote more joyful or surprising posts in their search and homepage recommendations to boost user activity on their website.