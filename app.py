
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    abort,
)

app = Flask(__name__)
app.secret_key = "careflow-demo-secret-key"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "events_cleaned.parquet"
REPORTS_DIR = BASE_DIR / "reports"

DEMO_USERNAME = "careflow_admin"
DEMO_PASSWORD = "CareFlow@123"


def load_dashboard_data():
    data = {
        "error": None,
        "total_cases": 0,
        "total_events": 0,
        "total_activities": 0,
        "avg_events": 0,
        "activities": [],
        "transitions": [],
        "recent_events": [],
    }

    if not DATA_FILE.exists():
        data["error"] = (
            "Dataset not found: data/processed/events_cleaned.parquet"
        )
        return data

    try:
        df = pd.read_parquet(DATA_FILE)

        case_col = "case:concept:name"
        activity_col = "concept:name"
        time_col = "time:timestamp"

        required = [case_col, activity_col, time_col]
        missing = [column for column in required if column not in df.columns]

        if missing:
            data["error"] = "Missing columns: " + ", ".join(missing)
            return data

        df[time_col] = pd.to_datetime(
            df[time_col], errors="coerce", utc=True
        )
        df = df.dropna(subset=required).copy()
        df = df.sort_values([case_col, time_col])

        total_cases = int(df[case_col].nunique())
        total_events = int(len(df))

        activities = (
            df[activity_col]
            .value_counts()
            .head(12)
            .rename_axis("activity")
            .reset_index(name="count")
        )

        df["next_activity"] = df.groupby(case_col)[activity_col].shift(-1)

        transitions = (
            df.dropna(subset=["next_activity"])
            .groupby([activity_col, "next_activity"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )

        recent = df.sort_values(time_col, ascending=False).head(10)

        data.update({
            "total_cases": total_cases,
            "total_events": total_events,
            "total_activities": int(df[activity_col].nunique()),
            "avg_events": round(total_events / total_cases, 2)
            if total_cases else 0,
            "activities": activities.to_dict("records"),
            "transitions": [
                {
                    "from_activity": str(row[activity_col]),
                    "to_activity": str(row["next_activity"]),
                    "count": int(row["count"]),
                }
                for _, row in transitions.iterrows()
            ],
            "recent_events": [
                {
                    "case": str(row[case_col]),
                    "activity": str(row[activity_col]),
                    "timestamp": row[time_col].strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                }
                for _, row in recent.iterrows()
            ],
        })

    except Exception as exc:
        data["error"] = f"Dataset loading failed: {exc}"

    return data


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            session.clear()
            session["logged_in"] = True
            return redirect(url_for("dashboard"))

        return render_template(
            "login.html",
            error="Incorrect username or password."
        )

    return render_template("login.html", error=None)


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        dashboard=load_dashboard_data()
    )


@app.route("/reports")
def reports():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    report_files = []

    if REPORTS_DIR.exists():
        allowed_extensions = {".csv", ".png"}
        report_files = sorted(
            file.name
            for file in REPORTS_DIR.iterdir()
            if file.is_file() and file.suffix.lower() in allowed_extensions
        )

    return render_template(
        "reports.html",
        report_files=report_files
    )


@app.route("/reports/file/<path:filename>")
def report_file(filename):
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    requested_file = Path(filename)

    if (
        requested_file.name != filename
        or requested_file.suffix.lower() not in {".csv", ".png"}
    ):
        abort(404)

    file_path = REPORTS_DIR / filename

    if not file_path.is_file():
        abort(404)

    if requested_file.suffix.lower() == ".png":
        return send_from_directory(
            REPORTS_DIR,
            filename,
            mimetype="image/png"
        )

    return send_from_directory(
        REPORTS_DIR,
        filename,
        as_attachment=True
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("welcome"))


if __name__ == "__main__":
    app.run(debug=True)
