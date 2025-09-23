import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------------------
# Machine Learning Model
# ------------------------------
def train_calorie_model():
    """
    Train a simple calorie prediction model using dummy dataset.
    Replace with real training data if available.
    """
    data = {
        "age": [20, 30, 40, 50, 60],
        "weight": [60, 70, 80, 90, 100],
        "height": [160, 170, 180, 190, 200],
        "activity": [1.2, 1.375, 1.55, 1.725, 1.9],  # activity multipliers
        "calories": [2000, 2200, 2500, 2800, 3000],
    }
    df_train = pd.DataFrame(data)

    X = df_train[["age", "weight", "height", "activity"]]
    y = df_train["calories"]

    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_calories(model, age, gender, weight, height, activity, goal):
    """
    Predict calories based on user inputs.
    """
    # Convert activity level to multiplier
    activity_map = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "Active": 1.725,
        "Very Active": 1.9,
    }
    act_value = activity_map.get(activity, 1.2)

    # Prepare input
    X_new = np.array([[age, weight, height, act_value]])

    # Base prediction from ML model
    calories = model.predict(X_new)[0]

    # Gender adjustment
    if gender.lower() == "male":
        calories *= 1.05
    else:
        calories *= 0.95

    # Goal adjustment
    if goal == "weight loss":
        calories -= 500
    elif goal == "weight gain":
        calories += 500

    return calories


# ------------------------------
# Food Filtering
# ------------------------------
def filter_foods(df, diet):
    """
    Filter foods based on dietary preference.
    """
    if diet.lower() == "vegan":
        return df[df["Tags"].str.contains("vegan", case=False, na=False)]
    elif diet.lower() == "vegetarian":
        return df[df["Tags"].str.contains("vegetarian", case=False, na=False)]
    elif diet.lower() == "non-vegetarian":
        return df[df["Tags"].str.contains("non-vegetarian", case=False, na=False)]
    elif diet.lower() == "gluten-free":
        return df[df["Tags"].str.contains("gluten-free", case=False, na=False)]
    else:
        return df


# ------------------------------
# Meal Plan Generation
# ------------------------------
def generate_meal_plan(filtered_df, target_calories):
    """
    Generate a daily meal plan by randomly selecting foods
    until reaching target calories.
    """
    meal_plan = pd.DataFrame(columns=filtered_df.columns)
    total_calories = 0

    while total_calories < target_calories and not filtered_df.empty:
        food = filtered_df.sample(1)
        meal_plan = pd.concat([meal_plan, food])
        total_calories = meal_plan["Calories"].sum()

        if total_calories > target_calories + 200:
            break

    return meal_plan.reset_index(drop=True)


def split_meals(daily_plan):
    """
    Split daily plan into Breakfast, Lunch, Dinner, Snacks
    without repeating the same foods.
    """
    n = len(daily_plan)
    daily_plan = daily_plan.sample(frac=1).reset_index(drop=True)  # shuffle

    breakfast = daily_plan.iloc[: int(0.25 * n)]
    lunch = daily_plan.iloc[int(0.25 * n) : int(0.6 * n)]
    dinner = daily_plan.iloc[int(0.6 * n) : int(0.9 * n)]
    snacks = daily_plan.iloc[int(0.9 * n) :]

    meals = {
        "Breakfast": breakfast,
        "Lunch": lunch,
        "Dinner": dinner,
        "Snacks": snacks,
    }
    return meals
