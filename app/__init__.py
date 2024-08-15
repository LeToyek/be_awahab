from flask import Flask
from .controllers.regression_controller import regression_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Register Blueprints
    app.register_blueprint(regression_bp, url_prefix='/regression')
    
    return app
