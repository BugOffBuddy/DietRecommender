import streamlit as st
import pandas as pd
from diet_utils import train_calorie_model, predict_calories, filter_foods, generate_meal_plan, split_meals

# Load dataset
df = pd.read_csv("usda_foods.csv")

st.title("AI Diet Recommender 🍎 - Weekly Planner")

# Sidebar Inputs
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age (years)", 10, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 250, 170)
activity = st.sidebar.selectbox(
    "Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
)
goal = st.sidebar.selectbox("Goal", ["Weight Loss", "Maintain", "Weight Gain"])
diet = st.sidebar.selectbox("Dietary Preference", ["vegan", "vegetarian", "non-vegetarian", "gluten-free"])

# --- Train ML model for calorie prediction ---
model = train_calorie_model()

# Predict daily calories using ML
calories_needed = predict_calories(model, age, gender, weight, height, activity, goal.lower())
st.write(f"### Your Daily Calorie Target: {int(calories_needed)} kcal (Predicted by ML Model)")

# Filter foods based on diet
filtered_df = filter_foods(df, diet)

# Weekly planner
st.write("## Your 7-Day Meal Plan")
for day in range(1, 8):
    daily_plan = generate_meal_plan(filtered_df, calories_needed)
    meals = split_meals(daily_plan)

    with st.expander(f"Day {day}"):
        for meal_name, meal_df in meals.items():
            st.write(f"### {meal_name}")
            if not meal_df.empty:
                st.dataframe(meal_df[["Food", "Category", "Calories", "Protein", "Carbs", "Fat"]])

                # Macro chart
                macro_totals = meal_df[["Calories", "Protein", "Carbs", "Fat"]].sum()
                st.bar_chart(macro_totals)
            else:
                st.write("⚠️ No foods available for this meal.")
