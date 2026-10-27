import base64
import io
import os
import time
import urllib.parse
import requests

from datetime import date
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

from database import init_db, get_db, save_history
from disease_predict import predict_disease_detailed

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

app = Flask(__name__)

# Temporary guest / user session
app.secret_key = os.getenv("FLASK_SECRET_KEY", "agrisense-ai-class12")

# API keys from .env
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

# Directory to save uploaded / camera-clicked plant photos for preview
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
init_db()


# Make Google Client ID and user session info available in all templates
@app.context_processor
def inject_auth_context():
    return {
        "google_client_id": GOOGLE_CLIENT_ID,
        "current_user_name": session.get("guest_name"),
        "current_user_email": session.get("user_email"),
        "current_user_picture": session.get("user_picture"),
        "auth_provider": session.get("auth_provider"),
    }


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# GUEST & GOOGLE SIGN-IN PAGE
# ==========================================

@app.route("/guest", methods=["GET", "POST"])
def guest():
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()

        if name:
            session["guest_name"] = name
            if email:
                session["user_email"] = email
                session["auth_provider"] = "google"
            else:
                session["auth_provider"] = "guest"
            return redirect(url_for("dashboard"))
        else:
            error = "Please enter your name or sign in with Google."

    return render_template("sign_in_agrisense_ai.html", error=error)


# ==========================================
# GOOGLE SIGN-IN (ID TOKEN & OAUTH2)
# ==========================================

@app.route("/auth/google", methods=["POST"])
def auth_google_token():
    """
    Handles Google Identity Services (Sign in with Google button / One Tap)
    ID token verification.
    """
    credential = request.form.get("credential", "").strip()

    # Fallback if quick Google email sign-in form is used in local demo mode
    if not credential:
        google_email = request.form.get("google_email", "").strip()
        google_name = request.form.get("google_name", "").strip()
        if google_email:
            display_name = google_name or google_email.split("@")[0].replace(".", " ").title()
            session["guest_name"] = display_name
            session["user_email"] = google_email
            session["user_picture"] = ""
            session["auth_provider"] = "google"
            save_history(display_name, "Google Sign-In", f"Signed in with Google ({google_email})")
            return redirect(url_for("dashboard"))
        return redirect(url_for("guest"))

    try:
        # Verify the Google ID token using Google's official tokeninfo endpoint
        verify_url = "https://oauth2.googleapis.com/tokeninfo"
        resp = requests.get(verify_url, params={"id_token": credential}, timeout=10)

        if resp.status_code == 200:
            user_info = resp.json()

            # Check client ID audience if configured
            if GOOGLE_CLIENT_ID and user_info.get("aud") != GOOGLE_CLIENT_ID:
                return render_template(
                    "sign_in_agrisense_ai.html",
                    error="Google Client ID mismatch. Please check GOOGLE_CLIENT_ID in .env."
                )

            name = (
                user_info.get("name")
                or user_info.get("given_name")
                or user_info.get("email", "User").split("@")[0]
            )
            email = user_info.get("email", "")
            picture = user_info.get("picture", "")

            session["guest_name"] = name
            session["user_email"] = email
            session["user_picture"] = picture
            session["auth_provider"] = "google"

            save_history(name, "Google Sign-In", f"Signed in with Google ({email})")
            return redirect(url_for("dashboard"))

    except Exception as e:
        print("Google ID token verification error:", e)

    return render_template(
        "sign_in_agrisense_ai.html",
        error="Unable to verify Google Sign-In. Please try again."
    )


@app.route("/login/google")
def google_login():
    """Initiates standard Google OAuth 2.0 redirect flow if Client ID & Secret are set."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return redirect(url_for("guest", mode="google"))

    redirect_uri = url_for("google_callback", _external=True)
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return redirect(auth_url)


@app.route("/login/google/callback")
def google_callback():
    """Handles the OAuth 2.0 callback from Google."""
    code = request.args.get("code")
    if not code:
        return redirect(url_for("guest"))

    try:
        token_url = "https://oauth2.googleapis.com/token"
        redirect_uri = url_for("google_callback", _external=True)
        token_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        token_resp = requests.post(token_url, data=token_data, timeout=10)
        token_json = token_resp.json()

        access_token = token_json.get("access_token")
        if access_token:
            userinfo_resp = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            if userinfo_resp.status_code == 200:
                profile = userinfo_resp.json()
                name = profile.get("name") or profile.get("email", "User").split("@")[0]
                email = profile.get("email", "")
                picture = profile.get("picture", "")

                session["guest_name"] = name
                session["user_email"] = email
                session["user_picture"] = picture
                session["auth_provider"] = "google"

                save_history(name, "Google Sign-In", f"Signed in with Google ({email})")
                return redirect(url_for("dashboard"))
    except Exception as e:
        print("Google OAuth callback error:", e)

    return render_template(
        "sign_in_agrisense_ai.html",
        error="Google Sign-In could not be completed. Please try again."
    )


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
        user_email=session.get("user_email"),
        user_picture=session.get("user_picture"),
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
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            from crop_model import recommend_crop

            result = recommend_crop(
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            )

            guest_name = session.get("guest_name", "Guest")
            save_history(
                guest_name,
                "Crop Recommendation",
                f"Recommended crop: {result}"
            )

        except (ValueError, KeyError):
            error = "Please enter valid values in all fields."

        except Exception as e:
            print("Crop recommendation error:", e)
            error = "Unable to generate crop recommendation."

    return render_template(
        "crops.html",
        result=result,
        error=error
    )


# ==========================================
# DISEASE DETECTION (UPLOAD + CAMERA CLICK)
# ==========================================

@app.route("/disease", methods=["GET", "POST"])
def disease():
    result = None
    suggestion = None
    confidence = None
    details = None
    image_path = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("plant_image")
        camera_data_url = request.form.get("camera_image", "").strip()

        image_bytes = None
        saved_filename = None

        # 1. Check if photo was clicked directly from the live camera
        if camera_data_url and camera_data_url.startswith("data:image"):
            try:
                header, encoded = camera_data_url.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                saved_filename = f"camera_leaf_{int(time.time())}.jpg"
            except Exception as e:
                print("Camera image decode error:", e)
                error = "Unable to read clicked camera photo. Please try again. / कैमरे की फोटो पढ़ने में त्रुटि हुई, कृपया दोबारा प्रयास करें।"

        # 2. Otherwise check if a photo file was uploaded
        elif uploaded_file and uploaded_file.filename != "":
            safe_name = secure_filename(uploaded_file.filename)
            saved_filename = f"leaf_{int(time.time())}_{safe_name}"
            image_bytes = uploaded_file.read()

        else:
            error = (
                "Please upload a plant image or click a photo using the camera. / "
                "कृपया पौधे की फोटो अपलोड करें या कैमरे से फोटो खींचें।"
            )

        if image_bytes and not error:
            try:
                full_save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
                with open(full_save_path, "wb") as f:
                    f.write(image_bytes)
                image_path = f"uploads/{saved_filename}"

                image_stream = io.BytesIO(image_bytes)
                result, confidence, details = predict_disease_detailed(image_stream)

                suggestion = (
                    f"{details['suggestion_en']} | "
                    f"हिंदी: {details['suggestion_hi']}"
                )

                guest_name = session.get("guest_name", "Guest")
                history_result = (
                    f"{details['name_en']} ({details['name_hi']}) | "
                    f"Confidence: {confidence:.2f}% | "
                    f"Problem: {details['problem_en']} / {details['problem_hi']} | "
                    f"Care: {details['suggestion_en']}"
                )

                save_history(
                    guest_name,
                    "Disease Detection",
                    history_result
                )

            except Exception as e:
                print("Disease prediction error:", e)
                error = (
                    "Unable to analyze the image. Please try another clear plant leaf photo. / "
                    "फोटो का विश्लेषण नहीं हो सका। कृपया पत्ती की साफ फोटो के साथ दोबारा प्रयास करें।"
                )

    return render_template(
        "disease.html",
        result=result,
        confidence=confidence,
        suggestion=suggestion,
        details=details,
        image_path=image_path,
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
        city = request.form.get("city", "").strip()

        if not city:
            error = "Please enter a city."
        elif not OPENWEATHER_API_KEY:
            error = "Weather service is not configured. Add OPENWEATHER_API_KEY to .env."
        else:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                if response.status_code == 200:
                    weather_data = {
                        "city": data["name"],
                        "country": data["sys"]["country"],
                        "temperature": round(data["main"]["temp"]),
                        "feels_like": round(data["main"]["feels_like"]),
                        "humidity": data["main"]["humidity"],
                        "wind": round(data["wind"]["speed"] * 3.6, 1),
                        "description": data["weather"][0]["description"].title()
                    }

                    guest_name = session.get("guest_name", "Guest")
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
                error = "Unable to connect to the weather service."
            except ValueError:
                error = "The weather service returned an invalid response."

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
    return render_template("about.html")


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
# LOGOUT / CHANGE USER
# ==========================================

@app.route("/logout")
def logout():
    session.pop("guest_name", None)
    session.pop("user_email", None)
    session.pop("user_picture", None)
    session.pop("auth_provider", None)
    return redirect(url_for("guest"))


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )