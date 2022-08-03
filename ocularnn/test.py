from flask import Blueprint, render_template, url_for
from flask import current_app as app
from ocularnn.auth import requires_auth

bp = Blueprint('test', __name__, url_prefix="/test")

@bp.route('/test')
@requires_auth
def test():
    return render_template("test/test.html")
