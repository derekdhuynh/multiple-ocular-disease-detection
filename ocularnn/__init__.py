import os
from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    # TODO: Add a database path, use either SQLite or SQLAlchemy
    app.config.from_mapping({"SECRET_KEY": "dev", "DATABASE"})

    print(dir(app))

    if test_config:
        app.config.from_mapping(test_config)
    else:
        app.config.from_pyfile("config.py", silent=True)

    try:
        os.mkdir(app.instance_path, "instance")
    except OSError:
        pass

    return app
