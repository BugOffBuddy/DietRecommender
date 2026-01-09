# DietRecommender
1. Abstract / Introduction

⦁	Need : In today’s busy world, most people don’t have time to properly plan their diet according to their health conditions. Many eat without knowing what is good or bad for their body, which leads to problems like obesity, diabetes, or cholesterol. To solve this issue, our project “AI-Based Personalized Diet Recommendation System” uses Machine Learning to suggest a balanced diet plan based on a person’s health details.

⦁	Objectives : The main objective of this project is to help users get personalized food recommendations using artificial intelligence. The system takes user inputs such as age, gender, weight, height, disease type, food preference (veg/non-veg), and allergies. It then calculates the BMI (Body Mass Index) and predicts the person’s goal — whether they should gain, lose, or maintain weight. Based on this goal, it suggests a list of suitable food items from a food dataset.

⦁	Methodology/Techniques : The methodology includes collecting data from two CSV files — one containing health details (user_health_dataset.csv) and the other containing food details (food_dataset.csv). We trained multiple machine learning models like Random Forest, SVM, and Logistic Regression, and selected the best one to predict the user’s diet goal. The selected model is saved as a .pkl file and used in the Streamlit web application to make real-time predictions.

⦁	Unique Contribution : The unique contribution of this project is that it not only predicts the user’s fitness goal but also filters food items according to their disease, allergies, and food preferences, ensuring that the diet plan is truly personalized.

⦁	Result (In short) :  the system predicts the right goal for the user and displays a list of at least eight food items suitable for them. For example, if someone wants to gain weight, it shows high-calorie food; if the goal is to lose weight, it shows low-calorie food.

⦁	Social/Professional Implications : The social and professional implication of this project is that it promotes healthy eating habits using modern technology. It can be useful for dieticians, fitness trainers, and individuals who want to maintain a healthy lifestyle. In the future, this project can be extended to include exercise suggestions, daily calorie tracking, and integration with wearable health devices.



2. Objective of the System

The main objective of this project is to create an intelligent system that recommends a suitable diet plan for a person based on their health details. The system uses machine learning to study health patterns and suggest food that matches the user’s goal and medical condition.

The specific objectives are:

1.	To analyze user health details like age, gender, weight, height, disease type, and food preference.

2.	To calculate BMI (Body Mass Index) and predict whether the person needs to gain, lose, or maintain weight.

3.	To recommend food items from the food dataset according to the user’s goal, disease, and allergy.

4.	To provide a personalized diet plan that fits the user’s lifestyle (veg/non-veg/both).

5.	To design a Streamlit-based web application that is simple, user-friendly, and interactive.

6.	To use multiple machine learning models (Random Forest, SVM, Logistic Regression) and choose the most accurate one for prediction.

7.	To help people follow a healthy and balanced diet with the help of AI technology.



3. Problem Definition

In today’s busy lifestyle, most people do not have proper knowledge about what kind of food they should eat according to their body type, health condition, and fitness goals. Many people face issues like obesity, diabetes, or cholesterol due to unhealthy eating habits.

Finding a personalized diet plan that suits each person’s needs is difficult and time-consuming. Most online diet plans are general and do not consider important factors like age, gender, BMI, medical condition, and allergies.

Therefore, there is a need for an AI-based diet recommendation system that can automatically analyze a person’s health details and provide a customized diet plan suitable for them.

The system should use machine learning to learn from health and food data, predict the user’s goal (gain, lose, or maintain weight), and suggest healthy food options accordingly.
