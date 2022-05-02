from flask import g, session, request, Blueprint, redirect, render_template
from . import db 

from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint(__name__, "doctor_auth", url_prefix="/doctor-auth")

@bp.route("register", methods=["GET", "POST"])
def register():
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
        db = db.get_db()

        # Except any errors, like if the username is not unique
        try:
            db.execute(
                "INSERT INTO user (username, password) VALUES (?, ?)",
                (username, generate_password_hash(password))
            )
            db.commit()
        except db.IntegrityError:
            error = "That username was already taken"

        print(error is None)
        if error is None:
            return redirect(url_for("doctor_auth.login"))
        else:
            for err in error:
                flash(err)

    return render_template("doctor_auth/register.html")

@bp.route("login", methods=["GET", "POST"])
def login():
    return render_template("doctor_auth/login.html")

