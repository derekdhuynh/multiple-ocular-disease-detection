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
    patients = db.execute("SELECT * FROM patients WHERE doctor=?", (session['doctor'])).fetchall()
    return patients

@requires_auth
@bp.route("/<username>")
def dashboard(username):
    return render_template("dashboard/dashboard.html")
