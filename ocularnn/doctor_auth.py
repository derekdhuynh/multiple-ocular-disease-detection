from flask import g, session, request, Blueprint, redirect, render_template
from . import db 

from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint("doctor_auth", __name__, url_prefix="/doctor-auth")

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
            return redirect(url_for("doctor-auth.login"))
        else:
            for err in error:
                flash(err)

    return render_template("doctor_auth/register.html")

@bp.route("login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Get username and password from user
        username = request.form["username"]
        password = request.form["password"]
        error = None

        # Query db to get user info if exists
        db = get_db()
        cur = db.execute("SELECT (username, password) FROM user WHERE username=?", (username,))
        db.commit()
        user_info = cur.fetchone() # will be empty if username is incorrect

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
            session['logged_in'] = authenticated 
            g.user = username
            return redirect(url_for('home'))

        flash(error)

    return render_template("doctor_auth/login.html")

