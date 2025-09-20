import pandas as pd

# 1. Calorie Calculator (Harris-Benedict)
def calculate_calories(age, gender, weight, height, activity, goal):
    # BMR calculation
    if gender.lower() == "male":
        bmr = 88.36 + (13.4 * weight) + (4.8 * height) - (5.7 * age)
    else:  # female
        bmr = 447.6 + (9.2 * weight) + (3.1 * height) - (4.3 * age)

    # Activity multipliers
    activity_levels = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }
    maintenance = bmr * activity_levels.get(activity.lower(), 1.2)

    # Adjust based on goal
    if goal == "weight_loss":
        return maintenance - 500
    elif goal == "weight_gain":
        return maintenance + 500
    else:
        return maintenance  # maintenance

# 2. Filter foods by dietary preference
def filter_foods(df, dietary_pref):
    return df[df["Tags"].str.contains(dietary_pref, case=False, na=False)]

# 3. Generate meal plan
def generate_meal_plan(df, target_calories):
    # Pick foods until total calories ≈ target
    selected = []
    total_cal = 0

    for _, row in df.sample(frac=1).iterrows():  # shuffle foods
        if total_cal + row["Calories"] <= target_calories:
            selected.append(row)
            total_cal += row["Calories"]

        if total_cal >= target_calories * 0.95:  # stop when close to target
            break

    return pd.DataFrame(selected)
