import sqlite3
from flask import current_app, g
from flask.cli import with_appcontext
import click

def get_db():
    """
    Creates a connection to the current database and stores it in the "g"
    namespace for reuse during the lifetime of the application context.

    This connection will be closed once the request has been completed. This is
    done automatically in the close_db function
    """
    if 'db' not in g:
        path = current_app.config['DATABASE']
        g.db = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def init_db():
    """
    Initializes and overwrites the current database
    """
    con = get_db()

    # Opening file resources the flask way and executing sql script
    with current_app.open_resource("schema.sql", 'r') as f:
        con.executescript(f.read()) 


@click.command("init-db")
@with_appcontext
def init_db_command():
    """
    A wrapper for the CLI command to initialize the database. The 
    """
    init_db()
    click.echo("Database initialized")

def close_db(e=None):
    """
    Pops the current database connection off the context stack and safely 
    closes the connection.

    Registered in __init__.py as a teardown_appcontext function which is called
    when the current application context ends or when a request context is 
    popped.

    Params
    ------
    e: Error
        Flask passes an error parameter to this function when it gets registered
        as the teardown routine (run
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_app(app):
    """
    A setup function meant to register the teardown routine for database
    connections and to add the init-db command to the app.
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
