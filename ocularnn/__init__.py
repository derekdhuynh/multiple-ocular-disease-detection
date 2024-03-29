import os
from flask import Flask, request, render_template

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # Setting configs
    app.config.from_mapping({
        "SECRET_KEY": "dev",
        "DATABASE": os.path.join(app.instance_path, "ocularnn.sqlite")
    })

    if test_config:
        app.config.from_mapping(test_config)
    else:
        app.config.from_pyfile("config.py", silent=True)

    try:
        os.mkdir(app.instance_path)
    except OSError:
        pass

    # Registering database initialization
    from . import db 
    db.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp)

    from . import dashboard
    app.register_blueprint(dashboard.bp)

    from . import test
    app.register_blueprint(test.bp)

    def test_template():
        return render_template('base.html')

    app.add_url_rule('/', endpoint='home', view_func=test_template)

    return app
