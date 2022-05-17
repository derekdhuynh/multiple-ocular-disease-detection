"""
Creating the dashboard containing all the patient data associated with a
specific doctor

TODO:
    - Add an upload feature for images
    - CRUD for patient profiles
        - Images (left, right)
        - Doctor's notes
        - Delete button
"""
from flask import (
    g, session, request, Blueprint, redirect, render_template, flash, url_for
)
from ocularnn.db import get_db
from ocularnn.auth import requires_auth

bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")

def get_all_patients():
    db = get_db()
    patients = db.execute("SELECT * FROM patients WHERE doctor=?", (session['username'],)).fetchall()
    return patients

@bp.route("/<username>")
@requires_auth
def dashboard(username):
    patients = get_all_patients()
    print(patients)
    return render_template("dashboard/dashboard.html")

@bp.route("/<username>/<patient_id>")
def patient(username, patient_id):
    """
    View of a patient within a doctor's database
    """
    return
    
