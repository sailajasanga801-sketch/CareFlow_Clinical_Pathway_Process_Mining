
from pathlib import Path
from io import BytesIO

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
    send_file,
)

app = Flask(__name__)
app.secret_key = "careflow-demo-secret-key"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "events_cleaned.parquet"
REPORTS_DIR = BASE_DIR / "reports"

DEMO_USERNAME = "careflow_admin"
DEMO_PASSWORD = "CareFlow@123"

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"


def load_events():
    """Load and prepare clinical event data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found: data/processed/events_cleaned.parquet"
        )

    df = pd.read_parquet(DATA_FILE)

    required = [CASE_COL, ACTIVITY_COL, TIME_COL]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL], errors="coerce", utc=True
    )

    df = df.dropna(
        subset=[CASE_COL, ACTIVITY_COL, TIME_COL]
    ).copy()

    df[CASE_COL] = df[CASE_COL].astype(str)
    df[ACTIVITY_COL] = df[ACTIVITY_COL].astype(str)

    return df.sort_values(TIME_COL)


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

    try:
        df = load_events()

        total_cases = int(df[CASE_COL].nunique())
        total_events = int(len(df))

        activities = (
            df[ACTIVITY_COL]
            .value_counts()
            .head(12)
            .rename_axis("activity")
            .reset_index(name="count")
        )

        ordered = df.sort_values([CASE_COL, TIME_COL]).copy()

        ordered["next_activity"] = (
            ordered.groupby(CASE_COL)[ACTIVITY_COL].shift(-1)
        )

        transitions = (
            ordered.dropna(subset=["next_activity"])
            .groupby([ACTIVITY_COL, "next_activity"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )

        recent = df.sort_values(
            TIME_COL, ascending=False
        ).head(10)

        data.update({
            "total_cases": total_cases,
            "total_events": total_events,
            "total_activities": int(df[ACTIVITY_COL].nunique()),
            "avg_events": round(total_events / total_cases, 2)
            if total_cases else 0,
            "activities": activities.to_dict("records"),
            "transitions": [
                {
                    "from_activity": str(row[ACTIVITY_COL]),
                    "to_activity": str(row["next_activity"]),
                    "count": int(row["count"]),
                }
                for _, row in transitions.iterrows()
            ],
            "recent_events": [
                {
                    "case": str(row[CASE_COL]),
                    "activity": str(row[ACTIVITY_COL]),
                    "timestamp": row[TIME_COL].strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    ),
                }
                for _, row in recent.iterrows()
            ],
        })

    except Exception as exc:
        data["error"] = f"Dataset loading failed: {exc}"

    return data


def get_filtered_events(args):
    """Filter events by case, activity and inclusive date range."""
    df = load_events()

    case_id = args.get("case_id", "").strip()
    activity = args.get("activity", "").strip()
    start_date = args.get("start_date", "").strip()
    end_date = args.get("end_date", "").strip()

    if case_id:
        df = df[
            df[CASE_COL].str.contains(
                case_id, case=False, regex=False
            )
        ]

    if activity:
        df = df[df[ACTIVITY_COL] == activity]

    start = None
    end_exclusive = None

    if start_date:
        try:
            start = pd.Timestamp(start_date).tz_localize("UTC")
        except (ValueError, TypeError):
            raise ValueError("Please select a valid start date.")

    if end_date:
        try:
            end_exclusive = (
                pd.Timestamp(end_date).tz_localize("UTC")
                + pd.Timedelta(days=1)
            )
        except (ValueError, TypeError):
            raise ValueError("Please select a valid end date.")

    if (
        start is not None
        and end_exclusive is not None
        and start >= end_exclusive
    ):
        raise ValueError("Start date must be on or before end date.")

    if start is not None:
        df = df[df[TIME_COL] >= start]

    if end_exclusive is not None:
        df = df[df[TIME_COL] < end_exclusive]

    return df.sort_values(TIME_COL, ascending=False)


def get_patient_journey(case_id):
    """Return a single case's events in chronological order."""
    df = load_events()

    journey_df = df[
        df[CASE_COL] == case_id
    ].sort_values(TIME_COL).copy()

    if journey_df.empty:
        return None

    first_time = journey_df[TIME_COL].iloc[0]
    last_time = journey_df[TIME_COL].iloc[-1]
    duration = (last_time - first_time).total_seconds()

    timeline = []

    for number, (_, row) in enumerate(
        journey_df.iterrows(), start=1
    ):
        timeline.append({
            "number": number,
            "activity": str(row[ACTIVITY_COL]),
            "timestamp": row[TIME_COL].strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        })

    return {
        "case_id": case_id,
        "event_count": len(journey_df),
        "first_event": first_time.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "last_event": last_time.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "duration_hours": round(duration / 3600, 2),
        "timeline": timeline,
        "dataframe": journey_df,
    }


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if (
            username == DEMO_USERNAME
            and password == DEMO_PASSWORD
        ):
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


@app.route("/events", methods=["GET"])
def events():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    filters = {
        "case_id": request.args.get("case_id", ""),
        "activity": request.args.get("activity", ""),
        "start_date": request.args.get("start_date", ""),
        "end_date": request.args.get("end_date", ""),
    }

    try:
        df = get_filtered_events(filters)

        all_activities = sorted(
            load_events()[ACTIVITY_COL].dropna().unique().tolist()
        )

        display_df = df[
            [CASE_COL, ACTIVITY_COL, TIME_COL]
        ].copy()

        display_df.columns = [
            "case_id", "activity", "timestamp"
        ]

        display_df["timestamp"] = (
            display_df["timestamp"].dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )

        return render_template(
            "events.html",
            events=display_df.head(200).to_dict("records"),
            total_results=len(display_df),
            activities=all_activities,
            case_id=filters["case_id"],
            selected_activity=filters["activity"],
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            error=None,
        )

    except Exception as exc:
        try:
            all_activities = sorted(
                load_events()[ACTIVITY_COL]
                .dropna().unique().tolist()
            )
        except Exception:
            all_activities = []

        return render_template(
            "events.html",
            events=[],
            total_results=0,
            activities=all_activities,
            case_id=filters["case_id"],
            selected_activity=filters["activity"],
            start_date=filters["start_date"],
            end_date=filters["end_date"],
            error=str(exc),
        )


@app.route("/events/download")
def download_filtered_events():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        df = get_filtered_events(request.args)
        csv_data = df.to_csv(index=False).encode("utf-8")

        return send_file(
            BytesIO(csv_data),
            mimetype="text/csv",
            as_attachment=True,
            download_name="careflow_filtered_events.csv",
        )

    except Exception as exc:
        return f"Could not export events: {exc}", 400


@app.route("/journey", methods=["GET"])
def journey():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    try:
        df = load_events()
        case_ids = sorted(df[CASE_COL].unique().tolist())
        selected_case = request.args.get(
            "case_id", ""
        ).strip()

        journey_data = None
        error = None

        if selected_case:
            if selected_case not in set(case_ids):
                error = "That patient case ID was not found."
            else:
                journey_data = get_patient_journey(selected_case)

        return render_template(
            "journey.html",
            case_ids=case_ids,
            selected_case=selected_case,
            journey=journey_data,
            error=error,
        )

    except Exception as exc:
        return render_template(
            "journey.html",
            case_ids=[],
            selected_case="",
            journey=None,
            error=f"Could not load patient journey: {exc}",
        )


@app.route("/journey/download")
def download_journey():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    case_id = request.args.get("case_id", "").strip()

    if not case_id:
        return "Please select a patient case first.", 400

    try:
        result = get_patient_journey(case_id)

        if result is None:
            return "Patient case was not found.", 404

        csv_data = (
            result["dataframe"].to_csv(index=False).encode("utf-8")
        )

        return send_file(
            BytesIO(csv_data),
            mimetype="text/csv",
            as_attachment=True,
            download_name="careflow_journey.csv",
        )

    except Exception as exc:
        return f"Could not export patient journey: {exc}", 400


@app.route("/bottlenecks")
def bottlenecks():
    """Display the previously generated process bottleneck report."""
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    csv_file = REPORTS_DIR / "process_bottlenecks.csv"
    chart_file = REPORTS_DIR / "process_bottlenecks.png"

    try:
        if not csv_file.is_file():
            raise FileNotFoundError(
                "Bottleneck report not found. Run: "
                "python src/bottleneck_analysis.py"
            )

        df = pd.read_csv(csv_file)

        required = [
            "from_activity",
            "to_activity",
            "case_transitions",
            "average_gap_hours",
            "median_gap_hours",
            "maximum_gap_hours",
        ]

        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing report columns: " + ", ".join(missing)
            )

        df = df.sort_values(
            "average_gap_hours", ascending=False
        ).head(100)

        rows = df.to_dict("records")

        return render_template(
            "bottlenecks.html",
            bottlenecks=rows,
            total_transitions=len(rows),
            chart_exists=chart_file.is_file(),
            error=None,
        )

    except Exception as exc:
        return render_template(
            "bottlenecks.html",
            bottlenecks=[],
            total_transitions=0,
            chart_exists=False,
            error=str(exc),
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
            if file.is_file()
            and file.suffix.lower() in allowed_extensions
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
