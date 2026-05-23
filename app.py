import base64
import csv
import shutil
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).parent
ATTENDANCE_FILE = PROJECT_ROOT / "attendance" / "attendance.csv"
USERS_FILE = PROJECT_ROOT / "data" / "users.csv"
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "known_faces"
ACTIVE_SUBJECT_FILE = PROJECT_ROOT / "data" / "active_subject.txt"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.secret_key = "change-this-secret"

from main import load_model, get_face_detector, detect_faces, FACE_SIZE, mark_attendance, train_model


def ensure_dirs() -> None:
    (PROJECT_ROOT / "attendance").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)


def get_active_subject() -> str:
    """Get the currently active subject selected by teacher"""
    if ACTIVE_SUBJECT_FILE.exists():
        return ACTIVE_SUBJECT_FILE.read_text().strip()
    return "MOOC"  # Default subject


def set_active_subject(subject: str) -> None:
    """Set the active subject for students to mark attendance"""
    ensure_dirs()
    ACTIVE_SUBJECT_FILE.write_text(subject.strip())


def ensure_default_users() -> None:
    ensure_dirs()
    if not USERS_FILE.exists():
        with USERS_FILE.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["username", "password_hash", "role", "display_name"])
    users = []
    with USERS_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            users.append(row)
    existing = {u["username"].lower() for u in users}
    with USERS_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if "teacher" not in existing:
            writer.writerow(["teacher", generate_password_hash("teacher123"), "teacher", "Teacher"])
        if "student" not in existing:
            writer.writerow(["student", generate_password_hash("student123"), "student", "Student"])


def get_display_name(username: str) -> str:
    for user in load_users():
        if user["username"].lower() == username.lower():
            return user.get("display_name", username)
    return username


def set_display_name(username: str, display_name: str) -> None:
    users = load_users()
    updated = False
    for user in users:
        if user["username"].lower() == username.lower():
            user["display_name"] = display_name
            updated = True
            break
    if not updated:
        return
    with USERS_FILE.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["username", "password_hash", "role", "display_name"])
        writer.writeheader()
        writer.writerows(users)


def load_users():
    ensure_default_users()
    users = []
    with USERS_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            users.append(row)
    return users


def save_user(username: str, password: str, role: str, display_name: str = None) -> None:
    ensure_default_users()
    users = load_users()
    if any(u["username"].lower() == username.lower() for u in users):
        raise ValueError("User already exists")
    if not display_name:
        display_name = username
    with USERS_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([username, generate_password_hash(password), role, display_name])


def authenticate(username: str, password: str):
    for user in load_users():
        if user["username"].lower() == username.lower():
            if check_password_hash(user["password_hash"], password):
                return user
    return None


def get_user(username: str):
    """Get user information by username"""
    for user in load_users():
        if user["username"].lower() == username.lower():
            return user
    return None


def change_password(username: str, old_password: str, new_password: str) -> tuple:
    """Change user password. Returns (success, message)"""
    user = get_user(username)
    if not user:
        return False, "User not found"
    
    if not check_password_hash(user["password_hash"], old_password):
        return False, "Current password is incorrect"
    
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters"
    
    users = load_users()
    for u in users:
        if u["username"].lower() == username.lower():
            u["password_hash"] = generate_password_hash(new_password)
            break
    
    with USERS_FILE.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["username", "password_hash", "role", "display_name"])
        writer.writeheader()
        writer.writerows(users)
    
    return True, "Password changed successfully"


def read_attendance():
    if not ATTENDANCE_FILE.exists():
        return []
    rows = []
    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        raw_rows = list(reader)

    if not raw_rows:
        return rows

    data_rows = raw_rows[1:] if raw_rows else []
    default_subject = get_active_subject()

    for record in data_rows:
        if not record or not any((cell or "").strip() for cell in record):
            continue

        name = record[0].strip() if len(record) > 0 else ""
        date = record[1].strip() if len(record) > 1 else ""
        time = record[2].strip() if len(record) > 2 else ""
        subject = record[3].strip() if len(record) > 3 and record[3].strip() else default_subject

        if not name or not date or not time:
            continue

        rows.append({"name": name, "date": date, "time": time, "subject": subject})

    return rows


def get_attendance_version(rows=None):
    """Return a lightweight version token for attendance change detection."""
    if rows is None:
        rows = read_attendance()
    count = len(rows)
    if count == 0:
        return {"count": 0, "last": ""}
    latest = rows[-1]
    latest_token = "|".join([
        latest.get("name", ""),
        latest.get("date", ""),
        latest.get("time", ""),
        latest.get("subject", ""),
    ])
    return {"count": count, "last": latest_token}


def get_face_registration_version():
    """Return a lightweight version token for face registration changes."""
    total_images = 0
    latest_mtime = 0.0

    if KNOWN_FACES_DIR.exists():
        for image_path in KNOWN_FACES_DIR.rglob("*.jpg"):
            if not image_path.is_file():
                continue
            total_images += 1
            try:
                mtime = image_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
            except OSError:
                continue

    return {"total_images": total_images, "latest_mtime": latest_mtime}


def filter_attendance(rows, name=None, date=None, subject=None):
    results = rows
    if name:
        results = [r for r in results if r.get("name", "").lower() == name.lower()]
    if date:
        results = [r for r in results if r.get("date", "") == date]
    if subject:
        results = [r for r in results if r.get("subject", "General").lower() == subject.lower()]
    return results


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_data_url(data_url: str):
    if not data_url or "," not in data_url:
        return None
    encoded = data_url.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def save_student_image(username: str, image) -> bool:
    if image is None:
        return False
    student_dir = KNOWN_FACES_DIR / username
    student_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_path = student_dir / f"{timestamp}.jpg"
    return bool(cv2.imwrite(str(output_path), image))


def predict_name_from_image(image, threshold: float = 100.0):
    """Predict name from image with lenient threshold"""
    model_data = load_model()
    if model_data is None:
        return None, None
    recognizer, label_map = model_data
    if not label_map:
        return None, None
    
    id_to_name = {label: name for name, label in label_map.items()}
    detector = get_face_detector()

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        return None, None
    
    faces = detect_faces(gray, detector)
    if len(faces) == 0:
        return None, None

    # Use the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, FACE_SIZE)
    
    try:
        label, confidence = recognizer.predict(face)
    except:
        return None, None
    
    if confidence <= threshold and label in id_to_name:
        return id_to_name[label], float(confidence)
    return None, float(confidence)


def login_required(role=None):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if "user" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                return redirect(url_for("login"))
            return fn(*args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


@app.route("/")
def home():
    if session.get("role") == "teacher":
        return redirect(url_for("teacher_dashboard"))
    if session.get("role") == "student":
        return redirect(url_for("student_dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = authenticate(username, password)
        if not user:
            flash("Invalid credentials", "error")
            return redirect(url_for("login"))
        session["user"] = user["username"]
        session["role"] = user["role"]
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/profile")
@login_required()
def profile():
    """View user profile"""
    username = session.get("user")
    user = get_user(username)
    if not user:
        return redirect(url_for("login"))
    return render_template("profile.html", user=user)


@app.route("/profile/edit", methods=["GET", "POST"])
@login_required()
def edit_profile():
    """Edit user profile (display name)"""
    username = session.get("user")
    user = get_user(username)
    if not user:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip()
        if not display_name:
            flash("Display name cannot be empty", "error")
            return redirect(url_for("edit_profile"))
        
        try:
            # Update display name in CSV
            set_display_name(username, display_name)
            flash("Profile updated successfully", "success")
            return redirect(url_for("profile"))
        except Exception as e:
            flash(f"Error updating profile: {str(e)}", "error")
            return redirect(url_for("edit_profile"))
    
    return render_template("edit_profile.html", user=user)


@app.route("/profile/change-password", methods=["GET", "POST"])
@login_required()
def change_password_route():
    """Change user password"""
    username = session.get("user")
    user = get_user(username)
    if not user:
        return redirect(url_for("login"))
    
    if request.method == "POST":
        old_password = request.form.get("old_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        if not all([old_password, new_password, confirm_password]):
            flash("All fields are required", "error")
            return redirect(url_for("change_password_route"))
        
        if new_password != confirm_password:
            flash("New passwords do not match", "error")
            return redirect(url_for("change_password_route"))
        
        success, message = change_password(username, old_password, new_password)
        if success:
            flash(message, "success")
            return redirect(url_for("profile"))
        else:
            flash(message, "error")
            return redirect(url_for("change_password_route"))
    
    return render_template("change_password.html", user=user)


@app.route("/teacher")
@login_required(role="teacher")
def teacher_dashboard():
    rows = read_attendance()
    name_filter = request.args.get("name")
    date_filter = request.args.get("date")
    subject_filter = request.args.get("subject")
    filtered = filter_attendance(rows, name_filter, date_filter, subject_filter)
    unique_names = sorted({r["name"] for r in rows})
    unique_dates = sorted({r["date"] for r in rows})
    unique_subjects = sorted({r.get("subject", "General") for r in rows})
    return render_template(
        "teacher.html",
        rows=filtered,
        unique_names=unique_names,
        unique_dates=unique_dates,
        unique_subjects=unique_subjects,
        name_filter=name_filter,
        date_filter=date_filter,
        subject_filter=subject_filter,
    )


@app.route("/teacher/add-student", methods=["POST"])
@login_required(role="teacher")
def add_student():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    if not username or not password:
        flash("Username and password are required", "error")
        return redirect(url_for("teacher_dashboard"))
    try:
        save_user(username, password, "student", display_name=username)
    except ValueError:
        flash("Student already exists", "error")
        return redirect(url_for("teacher_dashboard"))
    (KNOWN_FACES_DIR / username).mkdir(parents=True, exist_ok=True)
    flash("Student added", "success")
    return redirect(url_for("teacher_dashboard"))


@app.route("/teacher/upload-faces", methods=["POST"])
@login_required(role="teacher")
def upload_faces():
    username = request.form.get("username", "").strip()
    files = request.files.getlist("photos")
    if not username or not files:
        flash("Select a student and upload at least one image", "error")
        return redirect(url_for("teacher_dashboard"))
    display_name = get_display_name(username)
    student_dir = KNOWN_FACES_DIR / display_name
    student_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            file.save(student_dir / f"{timestamp}_{filename}")
            saved += 1
    if saved > 0:
        train_model()
    flash(f"Uploaded {saved} photo(s) and retrained model", "success")
    return redirect(url_for("teacher_dashboard"))


@app.route("/teacher/download")
@login_required(role="teacher")
def download_attendance():
    if not ATTENDANCE_FILE.exists():
        flash("Attendance file not found", "error")
        return redirect(url_for("teacher_dashboard"))
    return send_file(ATTENDANCE_FILE, as_attachment=True, download_name="attendance.csv")


@app.route("/student")
@login_required(role="student")
def student_dashboard():
    rows = read_attendance()
    username = session.get("user")
    display_name = get_display_name(username)
    filtered = filter_attendance(rows, name=display_name)
    return render_template("student.html", rows=filtered, username=username, display_name=display_name)


@app.route("/student/set-name", methods=["POST"])
@login_required(role="student")
def student_set_name():
    display_name = request.form.get("display_name", "").strip()
    if not display_name:
        flash("Display name cannot be empty", "error")
        return redirect(url_for("student_dashboard"))
    username = session.get("user")
    set_display_name(username, display_name)
    flash(f"Name updated to {display_name}", "success")
    return redirect(url_for("student_dashboard"))


@app.route("/student/register-face-browser", methods=["POST"])
@login_required(role="student")
def student_register_face_browser():
    payload = request.get_json(silent=True) or {}
    images = payload.get("images", [])
    username = session.get("user")
    display_name = get_display_name(username)
    if not images:
        return {"ok": False, "error": "No images provided"}, 400
    saved = 0
    for data_url in images:
        image = decode_data_url(data_url)
        if save_student_image(display_name, image):
            saved += 1
    if saved > 0:
        train_model()
    return {"ok": True, "saved": saved}


@app.route("/student/mark-attendance-browser", methods=["POST"])
@login_required(role="student")
def student_mark_attendance_browser():
    payload = request.get_json(silent=True) or {}
    image = decode_data_url(payload.get("image"))
    subject = payload.get("subject", "General").strip()
    if image is None:
        return {"ok": False, "error": "Invalid image"}, 400
    username = session.get("user")
    display_name = get_display_name(username)

    # Verify the student has registered face images
    student_face_dir = KNOWN_FACES_DIR / display_name
    registered_images = list(student_face_dir.glob("*.jpg")) if student_face_dir.exists() else []
    if not registered_images:
        return {"ok": False, "error": "No registered face found for your account. Please register your face first."}, 400

    # Use a strict threshold — LBPH confidence: lower = better match.
    # Only one fallback is allowed; never escalate beyond 100 to prevent false matches.
    name = None
    confidence = None
    for threshold in [80.0, 100.0]:
        name, confidence = predict_name_from_image(image, threshold=threshold)
        if name is not None:
            break

    if name is None:
        return {"ok": False, "error": "Face not recognized. Ensure good lighting and face the camera directly."}, 400

    # Only allow the currently logged-in student to mark their own attendance
    if name.lower() != display_name.lower():
        return {"ok": False, "error": "Face does not match your registered face. Attendance not marked."}, 403

    mark_attendance(name, subject)
    version = get_attendance_version()
    return {"ok": True, "name": name, "confidence": confidence, "version": version}


@app.route("/api/attendance-version", methods=["GET"])
@login_required()
def get_attendance_version_api():
    """Get attendance version details to allow dashboards to auto-refresh on updates."""
    rows = read_attendance()
    version = get_attendance_version(rows)
    return {"ok": True, "version": version}


@app.route("/api/dashboard-version", methods=["GET"])
@login_required(role="teacher")
def get_dashboard_version_api():
    """Get combined dashboard version details to support auto-refresh on key changes."""
    attendance_version = get_attendance_version()
    face_version = get_face_registration_version()
    return {"ok": True, "attendance": attendance_version, "faces": face_version}


@app.route("/api/attendance/<name>", methods=["GET", "POST"])
@login_required()
def get_student_attendance(name):
    username = session.get("user")
    display_name = get_display_name(username)
    if session.get("role") == "student" and name.lower() != display_name.lower():
        return {"ok": False, "error": "Unauthorized"}, 403
    rows = read_attendance()
    filtered = filter_attendance(rows, name=name)
    return {"ok": True, "rows": filtered}


@app.route("/api/attendance-stats/<name>", methods=["GET"])
@login_required()
def get_attendance_stats(name):
    """Get attendance statistics by subject for a student"""
    username = session.get("user")
    display_name = get_display_name(username)
    if session.get("role") == "student" and name.lower() != display_name.lower():
        return {"ok": False, "error": "Unauthorized"}, 403
    
    rows = read_attendance()
    filtered = filter_attendance(rows, name=name)
    
    stats = {}
    for row in filtered:
        subject = row.get("subject", "General")
        if subject not in stats:
            stats[subject] = 0
        stats[subject] += 1
    
    return {"ok": True, "stats": stats, "subjects": list(stats.keys()), "counts": list(stats.values())}


@app.route("/api/all-subjects", methods=["GET"])
@login_required(role="teacher")
def get_all_subjects():
    """Get all subjects from attendance records"""
    rows = read_attendance()
    subjects = sorted({r.get("subject", "General") for r in rows})
    return {"ok": True, "subjects": subjects}


@app.route("/api/subject-stats/<subject>", methods=["GET"])
@login_required(role="teacher")
def get_subject_stats(subject):
    """Get attendance statistics for all students in a subject"""
    rows = read_attendance()
    filtered = filter_attendance(rows, subject=subject)
    
    stats = {}
    for row in filtered:
        name = row.get("name", "Unknown")
        if name not in stats:
            stats[name] = 0
        stats[name] += 1
    
    return {"ok": True, "subject": subject, "stats": stats, "names": list(stats.keys()), "counts": list(stats.values())}


@app.route("/api/active-subject", methods=["GET"])
@login_required()
def get_active_subject_api():
    """Get the currently active subject for attendance"""
    subject = get_active_subject()
    return {"ok": True, "subject": subject}


@app.route("/api/set-active-subject", methods=["POST"])
@login_required(role="teacher")
def set_active_subject_api():
    """Set the active subject for students to mark attendance"""
    payload = request.get_json(silent=True) or {}
    subject = payload.get("subject", "").strip()
    
    if not subject:
        return {"ok": False, "error": "Subject is required"}, 400
    
    set_active_subject(subject)
    return {"ok": True, "subject": subject}


@app.route("/api/train-model", methods=["POST"])
@login_required(role="teacher")
def train_model_api():
    """Manually trigger face model training after students register faces"""
    try:
        # Check if any faces exist
        total_images = 0
        student_count = 0
        if KNOWN_FACES_DIR.exists():
            for student_dir in KNOWN_FACES_DIR.iterdir():
                if student_dir.is_dir():
                    images = list(student_dir.glob("*.jpg"))
                    if images:
                        student_count += 1
                        total_images += len(images)
        
        if total_images == 0:
            return {"ok": False, "error": "No registered faces found. Students need to register faces first."}, 400
        
        model_data = train_model()
        if model_data is None:
            return {"ok": False, "error": "Training failed. No valid face data found."}, 400
        
        return {"ok": True, "message": f"Model trained successfully with {total_images} images from {student_count} students. Students can now mark attendance."}
    except Exception as e:
        return {"ok": False, "error": f"Training failed: {str(e)}"}, 500


@app.route("/api/face-registration-status", methods=["GET"])
@login_required(role="teacher")
def face_registration_status():
    """Check face registration status"""
    students_with_faces = {}
    if KNOWN_FACES_DIR.exists():
        for student_dir in KNOWN_FACES_DIR.iterdir():
            if student_dir.is_dir():
                images = list(student_dir.glob("*.jpg"))
                if images:
                    students_with_faces[student_dir.name] = len(images)
    
    return {"ok": True, "students": students_with_faces, "total_students": len(students_with_faces), "total_images": sum(students_with_faces.values())}


@app.route("/teacher/delete-student", methods=["POST"])
@login_required(role="teacher")
def delete_student():
    username = request.form.get("username", "").strip()
    if not username:
        flash("Username is required", "error")
        return redirect(url_for("teacher_dashboard"))

    users = load_users()
    target = next((u for u in users if u["username"].lower() == username.lower()), None)
    if not target:
        flash(f"Student '{username}' not found", "error")
        return redirect(url_for("teacher_dashboard"))
    if target.get("role") == "teacher":
        flash("Cannot delete a teacher account", "error")
        return redirect(url_for("teacher_dashboard"))

    # Remove from users.csv
    remaining = [u for u in users if u["username"].lower() != username.lower()]
    with USERS_FILE.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["username", "password_hash", "role", "display_name"])
        writer.writeheader()
        writer.writerows(remaining)

    # Remove face images
    display_name = target.get("display_name", username)
    for face_dir_name in [username, display_name]:
        face_dir = KNOWN_FACES_DIR / face_dir_name
        if face_dir.exists():
            shutil.rmtree(face_dir)

    flash(f"Student '{username}' deleted successfully", "success")
    return redirect(url_for("teacher_dashboard"))


@app.route("/teacher/clear-attendance", methods=["POST"])
@login_required(role="teacher")
def clear_attendance_route():
    name = request.form.get("name", "").strip()
    if not ATTENDANCE_FILE.exists():
        flash("No attendance records found", "error")
        return redirect(url_for("teacher_dashboard"))

    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)

    if name:
        keep = [r for r in records if r["name"].lower() != name.lower()]
        removed = len(records) - len(keep)
    else:
        keep = []
        removed = len(records)

    with ATTENDANCE_FILE.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "date", "time", "subject"])
        writer.writeheader()
        writer.writerows(keep)

    label = f"for '{name}'" if name else "all"
    flash(f"Cleared {removed} attendance record(s) {label}", "success")
    return redirect(url_for("teacher_dashboard"))


@app.route("/student/download-attendance")
@login_required(role="student")
def student_download_attendance():
    import io
    username = session.get("user")
    display_name = get_display_name(username)
    rows = filter_attendance(read_attendance(), name=display_name)
    if not rows:
        flash("No attendance records found for your account", "error")
        return redirect(url_for("student_dashboard"))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["name", "date", "time", "subject"])
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    from flask import Response
    filename = f"attendance_{display_name}_{datetime.now().strftime('%Y%m%d')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    ensure_default_users()
    app.run(debug=True)
