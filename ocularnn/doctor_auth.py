from flask import (
    g, session, request, Blueprint, redirect, render_template, flash, url_for
)
from ocularnn.db import get_db

from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint("doctor_auth", __name__, url_prefix="/doctor-auth")

@bp.route("/register", methods=["GET", "POST"])
def register():
    """
    View for registering as a new user. An real-life implementation would use
    the doctor's email and have a more rigourous verification process.
    """
    if request.method == "POST":
        # Get username and password from form
        username = request.form['username']
        password = request.form['password']
        error = []

        # Catch any errors (i,e empty form)
        if username is None:
            error.append("You must enter a valid username")
        if password is None:
            error.append("You must enter a valid password")

        # Store username and password hash into db
        db = get_db()

        # Except any errors, like if the username is not unique
        try:
            db.execute(
                "INSERT INTO user (username, password) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )
            db.commit()
        except db.IntegrityError:
            error.append("That username was already taken")

        if not error:
            print("Registered successfully!")
            return redirect(url_for("doctor_auth.login"))
        else:
            print("Registration not successful!")
            for err in error:
                flash(err)

    return render_template("doctor_auth/register.html")

@bp.route("/login", methods=["GET", "POST"])
def login():
    """
    View for logging onto the site and accessing the doctor dashboard.

    Once the user has been authenticated store their username in a session
    so they can access restricted views.
    """
    if request.method == "POST":
        # Get username and password from user
        username = request.form["username"]
        password = request.form["password"]
        error = None

        # Query db to get user info if exists
        db = get_db()

        # Will be an empty list if no results
        user_info = db.execute(
            "SELECT * FROM user WHERE username=?",
            (username,)
        ).fetchone()

        # Throw error if no matches found
        if not user_info:
            error = "Username was not valid"

        # and check if username and password are valid
        authenticated = False
        if user_info is not None:
            authenticated = check_password_hash(user_info['password'], password)

        # Redirect to the main menu/dashboard if authentification is successful
        if not authenticated:
            error = "Password was not valid"
        else:
            session['username'] = username
            # Change to redirect to dashboard
            return redirect(url_for('home'))

        flash(error)

    return render_template("doctor_auth/login.html")


@bp.route("/logout", methods=["GET"])
def logout():
    """
    Logging the current user out
    """
    session.clear()
    return redirect(url_for('home'))

"""TODO: Add wrapper which locks views to only authenticated users"""
