import os
from unittest import result
import requests

from disease_predict import predict_disease
from datetime import date
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from database import init_db, get_db, save_history

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

app = Flask(__name__)

# Temporary guest session
app.secret_key = "agrisense-ai-class12"

# OpenWeather API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Initialize database
init_db()


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# GUEST PROFILE
# ==========================================

@app.route("/guest", methods=["GET", "POST"])
def guest():
    if request.method == "POST":
        name = request.form.get("name", "").strip()

        if name:
            session["guest_name"] = name
            return redirect(url_for("dashboard"))

    return render_template("guest.html")

# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    name = session.get("guest_name", "Guest")

    daily_tip = get_daily_tip()

    return render_template(
        "dashboard.html",
        name=name,
        daily_tip=daily_tip
    )


# ==========================================
# CROP RECOMMENDATION
# ==========================================

@app.route("/crops", methods=["GET", "POST"])
def crops():

    result = None
    error = None

    if request.method == "POST":

        try:

            # Get input values
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])

            temperature = float(
                request.form["temperature"]
            )

            humidity = float(
                request.form["humidity"]
            )

            ph = float(
                request.form["ph"]
            )

            rainfall = float(
                request.form["rainfall"]
            )

            # Import crop prediction function
            from crop_model import recommend_crop

            # Get prediction
            result = recommend_crop(
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            )

            # ==================================
            # SAVE CROP RESULT TO HISTORY
            # ==================================

            guest_name = session.get(
                "guest_name",
                "Guest"
            )

            save_history(
                guest_name,
                "Crop Recommendation",
                f"Recommended crop: {result}"
            )

        except (ValueError, KeyError):

            error = (
                "Please enter valid values "
                "in all fields."
            )

        except Exception as e:

            print(
                "Crop recommendation error:",
                e
            )

            error = (
                "Unable to generate crop "
                "recommendation."
            )

    return render_template(
        "crops.html",
        result=result,
        error=error
    )

# ==========================================
# DISEASE DETECTION
# ==========================================

from disease_predict import predict_disease

@app.route("/disease", methods=["GET", "POST"])
def disease():

    result = None
    suggestion = None
    confidence = None
    error = None

    if request.method == "POST":

        image = request.files.get("plant_image")

        if not image or image.filename == "":
            error = "Please select a plant image."

        else:

            try:

                # Run disease prediction
                result, confidence = predict_disease(image)

                # Suggestions
                suggestions = {

                    "Tomato_Early_blight":
                        "Remove heavily affected leaves and monitor the plant closely.",

                    "Tomato_Late_blight":
                        "Remove affected plant parts and avoid prolonged moisture on leaves.",

                    "Tomato_Bacterial_spot":
                        "Remove affected leaves and maintain good airflow around the plant.",

                    "Tomato_Leaf_Mold":
                        "Improve air circulation and avoid excessive moisture.",

                    "Tomato_Septoria_leaf_spot":
                        "Remove affected leaves and keep the surrounding area clean.",

                    "Tomato_healthy":
                        "The plant appears healthy. Continue regular monitoring.",

                    "Potato_Early_blight":
                        "Remove affected leaves and monitor the crop regularly.",

                    "Potato_Late_blight":
                        "Remove affected plant parts and avoid prolonged leaf wetness.",

                    "Potato_healthy":
                        "The plant appears healthy. Continue regular care.",

                    "Apple___Apple_scab":
                        "Remove affected leaves and maintain good airflow around the plant.",

                    "Apple___Black_rot":
                        "Remove affected plant material and monitor the tree closely.",

                    "Apple___Cedar_apple_rust":
                        "Remove affected leaves and monitor the plant closely.",

                    "Apple___healthy":
                        "The plant appears healthy. Continue regular monitoring.",

                    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
                        "Monitor the crop closely and remove severely affected leaves.",

                    "Corn_(maize)___Common_rust_":
                        "Monitor the crop and remove severely affected leaves.",

                    "Corn_(maize)___Northern_Leaf_Blight":
                        "Remove severely affected leaves and monitor crop health.",

                    "Corn_(maize)___healthy":
                        "The crop appears healthy. Continue regular monitoring.",

                    "Pepper__bell___Bacterial_spot":
                        "Remove affected leaves and maintain good airflow.",

                    "Pepper__bell___healthy":
                        "The plant appears healthy. Continue regular care."
                }

                suggestion = suggestions.get(
                    result,
                    "Monitor the plant closely and maintain proper crop care."
                )


                # ==========================================
                # SAVE RESULT TO HISTORY
                # ==========================================

                guest_name = session.get(
                    "guest_name",
                    "Guest"
                )

                history_result = (
                    f"{result} | "
                    f"Confidence: {confidence:.2f}% | "
                    f"{suggestion}"
                )

                save_history(
                    guest_name,
                    "Disease Detection",
                    history_result
                )


            except Exception as e:

                print(
                    "Disease prediction error:",
                    e
                )

                error = (
                    "Unable to analyze the image. "
                    "Please try another plant image."
                )


    return render_template(
        "disease.html",
        result=result,
        confidence=confidence,
        suggestion=suggestion,
        error=error
    )


# ==========================================
# WEATHER
# ==========================================

@app.route("/weather", methods=["GET", "POST"])
def weather():

    weather_data = None
    error = None
    city = ""

    if request.method == "POST":

        city = request.form.get(
            "city",
            ""
        ).strip()

        if not city:

            error = "Please enter a city."

        else:

            url = (
                "https://api.openweathermap.org/"
                "data/2.5/weather"
            )

            params = {
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }

            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=10
                )

                data = response.json()

                if response.status_code == 200:

                    # ==================================
                    # WEATHER DATA
                    # ==================================

                    weather_data = {

                        "city": data["name"],

                        "country": data["sys"]["country"],

                        "temperature": round(
                            data["main"]["temp"]
                        ),

                        "feels_like": round(
                            data["main"]["feels_like"]
                        ),

                        "humidity": data["main"]["humidity"],

                        "wind": round(
                            data["wind"]["speed"] * 3.6,
                            1
                        ),

                        "description": data["weather"][0][
                            "description"
                        ].title()
                    }

                    # ==================================
                    # SAVE WEATHER TO HISTORY
                    # ==================================

                    guest_name = session.get(
                        "guest_name",
                        "Guest"
                    )

                    connection = get_db()

                    connection.execute(
                        """
                        INSERT INTO history
                        (guest_name, activity, result)
                        VALUES (?, ?, ?)
                        """,
                        (
                            guest_name,
                            "Weather Forecast",
                            (
                                f"{weather_data['city']}: "
                                f"{weather_data['temperature']}°C, "
                                f"{weather_data['description']}"
                            )
                        )
                    )

                    connection.commit()
                    connection.close()

                else:

                    error = data.get(
                        "message",
                        "Unable to get weather information."
                    )

            except requests.RequestException:

                error = (
                    "Unable to connect to "
                    "the weather service."
                )

    return render_template(
        "weather.html",
        weather=weather_data,
        error=error,
        city=city
    )


# ==========================================
# HISTORY
# ==========================================

@app.route("/history")
def history():

    connection = get_db()

    records = connection.execute(
        """
        SELECT *
        FROM history
        ORDER BY created_at DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "history.html",
        records=records
    )


# ==========================================
# ABOUT
# ==========================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )

# ==========================================
# DAILY FARMING TIP
# ==========================================

FARMING_TIPS = [
    "Regularly monitor your crops for early signs of disease. Early detection can help protect crop health.",
    
    "Water your crops according to their needs and avoid unnecessary overwatering.",
    
    "Maintain proper soil moisture to support healthy crop growth.",
    
    "Check your plants regularly for unusual spots, discoloration, or other signs of disease.",
    
    "Keep the area around your crops clean to reduce the chance of pests and diseases.",
    
    "Good soil health is important for healthy plant growth. Use suitable nutrients based on soil conditions.",
    
    "Weather conditions can affect crop growth, so check the forecast before planning important farming activities.",
    
    "Remove severely damaged plant leaves when appropriate to help prevent problems from spreading.",
    
    "Record important farming activities and observations to help monitor crop performance.",
    
    "Use crop recommendations together with local farming knowledge and suitable agricultural practices."
]


def get_daily_tip():
    """Return a farming tip that changes automatically each day."""

    day_number = date.today().timetuple().tm_yday

    return FARMING_TIPS[day_number % len(FARMING_TIPS)]

# ==========================================
# LOGOUT / CHANGE GUEST
# ==========================================

@app.route("/logout")
def logout():

    session.pop("guest_name", None)

    return redirect(url_for("guest"))

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )