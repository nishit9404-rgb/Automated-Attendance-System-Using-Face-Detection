import argparse
import csv
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import shutil

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent
KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "known_faces"
ATTENDANCE_DIR = PROJECT_ROOT / "attendance"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"
MODEL_FILE = PROJECT_ROOT / "data" / "face_model.yml"
LABELS_FILE = PROJECT_ROOT / "data" / "labels.csv"
FACE_SIZE = (200, 200)


def ensure_dirs() -> None:
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)


def load_known_faces():
    images = []
    labels = []
    label_map = {}
    current_label = 0

    detector = get_face_detector()

    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = current_label
            current_label += 1
        for image_path in person_dir.glob("*.jpg"):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector)
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, FACE_SIZE)
            images.append(face)
            labels.append(label_map[name])

    return images, labels, label_map


def get_face_detector() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade face detector.")
    return detector


def detect_faces(gray_frame, detector):
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces


def save_labels(label_map) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_FILE.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["label", "name"])
        for name, label in sorted(label_map.items(), key=lambda item: item[1]):
            writer.writerow([label, name])


def load_labels():
    label_map = {}
    if not LABELS_FILE.exists():
        return label_map
    with LABELS_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_map[row["name"]] = int(row["label"])
    return label_map


def train_model():
    images, labels, label_map = load_known_faces()
    if not images:
        print("No training images found. Register users first.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(MODEL_FILE))
    save_labels(label_map)
    return recognizer, label_map


def is_model_stale() -> bool:
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        return True
    model_mtime = MODEL_FILE.stat().st_mtime
    for person_dir in KNOWN_FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        for image_path in person_dir.glob("*.jpg"):
            if image_path.stat().st_mtime > model_mtime:
                return True
    return False


def load_model():
    if is_model_stale():
        return train_model()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    label_map = load_labels()
    return recognizer, label_map


def register_user(name: str, num_samples: int = 5) -> None:
    ensure_dirs()
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    detector = get_face_detector()
    captured = 0
    print("Press SPACE to capture, ESC to exit.")
    print("Ensure only YOUR face is visible in the frame.")

    while captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)
        
        # Display status based on face count
        status_text = f"Samples: {captured}/{num_samples}"
        status_color = (0, 255, 0)
        warning_text = ""
        
        if len(faces) == 0:
            warning_text = "WARNING: No face detected!"
            status_color = (0, 0, 255)
        elif len(faces) > 1:
            warning_text = f"WARNING: {len(faces)} faces detected! Only one allowed!"
            status_color = (0, 0, 255)
        else:
            warning_text = "Ready to capture - Press SPACE"
            # Draw rectangle around the detected face
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            warning_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        
        cv2.imshow("Register", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            # Only capture if exactly one face is detected
            if len(faces) == 1:
                image_path = person_dir / f"{name}_{captured + 1}.jpg"
                cv2.imwrite(str(image_path), frame)
                captured += 1
                print(f"Captured {image_path.name}")
            else:
                if len(faces) == 0:
                    print("Cannot capture: No face detected!")
                else:
                    print(f"Cannot capture: {len(faces)} faces detected! Only one face allowed.")

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        print(f"Training model with new face data for '{name}'...")
        train_model()
        print("Model trained successfully.")


def mark_attendance(name: str, subject: str = "General") -> bool:
    ensure_dirs()
    is_new_file = not ATTENDANCE_FILE.exists()
    
    # Check if attendance already marked today
    today = datetime.now().strftime("%Y-%m-%d")
    if not is_new_file:
        with ATTENDANCE_FILE.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["name"] == name and row["date"] == today:
                    print(f"Attendance already marked for {name} today at {row['time']}")
                    return False
    
    with ATTENDANCE_FILE.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["name", "date", "time", "subject"])

        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), subject])
        print(f"Attendance marked successfully for {name}")
        return True


def attend_loop(threshold: float = 80.0) -> None:
    ensure_dirs()
    model_data = load_model()
    if model_data is None:
        return
    recognizer, label_map = model_data
    id_to_name = {label: name for name, label in label_map.items()}
    detector = get_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    print("Press ESC to exit.")
    print("Ensure only one face is visible for attendance.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)
        matched_name = None

        # Check if multiple faces are detected
        if len(faces) > 1:
            # Show warning for multiple faces
            warning_text = f"WARNING: {len(faces)} faces detected! Only one allowed!"
            cv2.putText(
                frame,
                warning_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            # Draw rectangles around all faces in red
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print("Multiple faces detected! Attendance not marked.")
        elif len(faces) == 1:
            # Only process if exactly one face is detected
            x, y, w, h = faces[0]
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, FACE_SIZE)
            label, confidence = recognizer.predict(face)

            name = "WRONG FACE"
            box_color = (0, 0, 255)  # Red for wrong face
            
            if confidence <= threshold and label in id_to_name:
                name = id_to_name[label]
                if mark_attendance(name):
                    matched_name = name
                box_color = (0, 255, 0)  # Green for recognized

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                frame,
                f"{name} ({confidence:.1f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
            )
        else:
            # No face detected
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

        cv2.imshow("Attendance", frame)
        if matched_name is not None:
            break
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def list_users() -> None:
    """List all registered users."""
    if not KNOWN_FACES_DIR.exists():
        print("No registered users found.")
        return
    
    users = [d.name for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]
    if not users:
        print("No registered users found.")
        return
    
    print(f"\n{'='*50}")
    print(f"Registered Users ({len(users)})")
    print(f"{'='*50}")
    for i, user in enumerate(sorted(users), 1):
        # Count number of images
        user_dir = KNOWN_FACES_DIR / user
        num_images = len(list(user_dir.glob("*.jpg")))
        print(f"{i}. {user} ({num_images} samples)")
    print(f"{'='*50}\n")


def view_attendance(name: str = None, date: str = None) -> None:
    """View attendance records with optional filters."""
    if not ATTENDANCE_FILE.exists():
        print("No attendance records found.")
        return
    
    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)
    
    if not records:
        print("No attendance records found.")
        return
    
    # Filter records
    filtered = records
    if name:
        filtered = [r for r in filtered if r["name"].lower() == name.lower()]
    if date:
        filtered = [r for r in filtered if r["date"] == date]
    
    if not filtered:
        print("No matching records found.")
        return
    
    print(f"\n{'='*70}")
    print(f"Attendance Records ({len(filtered)})")
    print(f"{'='*70}")
    print(f"{'Name':<15} {'Date':<12} {'Time':<10} {'Subject':<20}")
    print(f"{'-'*70}")
    for record in filtered:
        print(f"{record['name']:<15} {record['date']:<12} {record['time']:<10} {record.get('subject', 'N/A'):<20}")
    print(f"{'='*70}\n")


def delete_user(name: str) -> None:
    """Delete a registered user."""
    user_dir = KNOWN_FACES_DIR / name
    if not user_dir.exists():
        print(f"User '{name}' not found.")
        return
    
    # Delete user directory and all images
    shutil.rmtree(user_dir)
    print(f"User '{name}' deleted successfully.")
    
    # Mark model as stale by deleting it
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
    if LABELS_FILE.exists():
        LABELS_FILE.unlink()
    print("Model will be retrained on next use.")


def generate_report() -> None:
    """Generate attendance statistics report."""
    if not ATTENDANCE_FILE.exists():
        print("No attendance records found.")
        return
    
    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)
    
    if not records:
        print("No attendance records found.")
        return
    
    # Calculate statistics
    user_counts = Counter(r["name"] for r in records)
    subject_counts = Counter(r.get("subject", "N/A") for r in records)
    date_counts = Counter(r["date"] for r in records)
    
    # User-wise attendance by date
    user_dates = defaultdict(set)
    for r in records:
        user_dates[r["name"]].add(r["date"])
    
    print(f"\n{'='*70}")
    print("Attendance Statistics Report")
    print(f"{'='*70}")
    print(f"\nTotal Records: {len(records)}")
    print(f"Unique Users: {len(user_counts)}")
    print(f"Date Range: {min(date_counts.keys())} to {max(date_counts.keys())}")
    
    print(f"\n{'='*70}")
    print("User-wise Attendance Count")
    print(f"{'='*70}")
    print(f"{'Name':<20} {'Total Days':<15} {'Total Records':<15}")
    print(f"{'-'*70}")
    for user, count in user_counts.most_common():
        unique_days = len(user_dates[user])
        print(f"{user:<20} {unique_days:<15} {count:<15}")
    
    print(f"\n{'='*70}")
    print("Subject-wise Attendance")
    print(f"{'='*70}")
    for subject, count in subject_counts.most_common():
        print(f"{subject}: {count} records")
    
    print(f"\n{'='*70}")
    print("Date-wise Attendance (Last 10 Days)")
    print(f"{'='*70}")
    for date in sorted(date_counts.keys(), reverse=True)[:10]:
        print(f"{date}: {date_counts[date]} records")
    print(f"{'='*70}\n")


def rename_user(old_name: str, new_name: str) -> None:
    """Rename a registered user and invalidate the stale model."""
    old_dir = KNOWN_FACES_DIR / old_name
    if not old_dir.exists():
        print(f"User '{old_name}' not found.")
        return
    new_dir = KNOWN_FACES_DIR / new_name
    if new_dir.exists():
        print(f"User '{new_name}' already exists. Choose a different name.")
        return
    old_dir.rename(new_dir)
    print(f"User '{old_name}' renamed to '{new_name}' successfully.")
    if MODEL_FILE.exists():
        MODEL_FILE.unlink()
    if LABELS_FILE.exists():
        LABELS_FILE.unlink()
    print("Model will be retrained on next use.")


def verify_face(name: str, threshold: float = 80.0) -> None:
    """Live webcam verification — checks if the face matches a specific user without marking attendance."""
    model_data = load_model()
    if model_data is None:
        print("No model found. Register users first.")
        return
    recognizer, label_map = model_data
    if name not in label_map:
        print(f"User '{name}' is not registered. Register them first.")
        return

    id_to_name = {label: n for n, label in label_map.items()}
    detector = get_face_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    print(f"Verifying face against '{name}'. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, FACE_SIZE)
            label, confidence = recognizer.predict(face)
            predicted = id_to_name.get(label, "Unknown")
            is_match = predicted.lower() == name.lower() and confidence <= threshold
            color = (0, 255, 0) if is_match else (0, 0, 255)
            result = f"MATCH ({confidence:.1f})" if is_match else f"NO MATCH — {predicted} ({confidence:.1f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, result, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif len(faces) > 1:
            cv2.putText(frame, "Multiple faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.putText(
            frame,
            f"Verifying: {name}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Verify Face", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def export_report(output_file: str) -> None:
    """Export a per-student attendance summary to a CSV file."""
    if not ATTENDANCE_FILE.exists():
        print("No attendance records found.")
        return

    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)

    if not records:
        print("No attendance records found.")
        return

    user_dates: defaultdict = defaultdict(set)
    user_subjects: defaultdict = defaultdict(set)
    user_counts: Counter = Counter()

    for r in records:
        name = r["name"]
        user_dates[name].add(r["date"])
        user_subjects[name].add(r.get("subject", "General"))
        user_counts[name] += 1

    output_path = Path(output_file)
    with output_path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["name", "total_days", "total_records", "subjects"])
        for user, count in user_counts.most_common():
            subjects = ", ".join(sorted(user_subjects[user]))
            writer.writerow([user, len(user_dates[user]), count, subjects])

    print(f"Report exported to {output_path.resolve()} ({len(user_counts)} student(s)).")


def clear_attendance(name: str = None, date: str = None) -> None:
    """Clear attendance records with an optional name/date filter. Prompts for confirmation."""
    if not ATTENDANCE_FILE.exists():
        print("No attendance records found.")
        return

    with ATTENDANCE_FILE.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        records = list(reader)

    if not records:
        print("No attendance records found.")
        return

    to_remove = records
    if name:
        to_remove = [r for r in to_remove if r["name"].lower() == name.lower()]
    if date:
        to_remove = [r for r in to_remove if r["date"] == date]

    if not to_remove:
        print("No matching records found.")
        return

    print(f"About to delete {len(to_remove)} record(s). Confirm? [y/N]: ", end="", flush=True)
    if input().strip().lower() != "y":
        print("Cancelled.")
        return

    keep = [r for r in records if r not in to_remove]
    with ATTENDANCE_FILE.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "date", "time", "subject"])
        writer.writeheader()
        writer.writerows(keep)

    print(f"Cleared {len(to_remove)} record(s). {len(keep)} record(s) remaining.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Attendance System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register", help="Register a new user")
    register_parser.add_argument("--name", required=True, help="User name")
    register_parser.add_argument("--samples", type=int, default=5, help="Number of samples")

    attend_parser = subparsers.add_parser("attend", help="Start attendance")
    attend_parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="LBPH confidence threshold (lower is stricter)",
    )

    subparsers.add_parser("list", help="List all registered users")

    view_parser = subparsers.add_parser("view", help="View attendance records")
    view_parser.add_argument("--name", help="Filter by user name")
    view_parser.add_argument("--date", help="Filter by date (YYYY-MM-DD)")

    delete_parser = subparsers.add_parser("delete", help="Delete a registered user")
    delete_parser.add_argument("--name", required=True, help="User name to delete")

    subparsers.add_parser("report", help="Print attendance statistics report")

    rename_parser = subparsers.add_parser("rename", help="Rename a registered user")
    rename_parser.add_argument("--old-name", required=True, help="Current user name")
    rename_parser.add_argument("--new-name", required=True, help="New user name")

    verify_parser = subparsers.add_parser("verify", help="Verify face against a registered user (no attendance)")
    verify_parser.add_argument("--name", required=True, help="User name to verify against")
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="LBPH confidence threshold (lower is stricter)",
    )

    export_parser = subparsers.add_parser("export", help="Export per-student attendance summary to CSV")
    export_parser.add_argument(
        "--output",
        default="attendance_report.csv",
        help="Output file path (default: attendance_report.csv)",
    )

    clear_parser = subparsers.add_parser("clear", help="Clear attendance records (with confirmation)")
    clear_parser.add_argument("--name", help="Clear records for a specific user only")
    clear_parser.add_argument("--date", help="Clear records for a specific date (YYYY-MM-DD)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "register":
        register_user(args.name, args.samples)
    elif args.command == "attend":
        attend_loop(args.threshold)
    elif args.command == "list":
        list_users()
    elif args.command == "view":
        view_attendance(args.name, args.date)
    elif args.command == "delete":
        delete_user(args.name)
    elif args.command == "report":
        generate_report()
    elif args.command == "rename":
        rename_user(args.old_name, args.new_name)
    elif args.command == "verify":
        verify_face(args.name, args.threshold)
    elif args.command == "export":
        export_report(args.output)
    elif args.command == "clear":
        clear_attendance(args.name, args.date)


if __name__ == "__main__":
    main()
