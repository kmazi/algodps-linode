from config import config
from flasgger import Swagger
from flask import Flask, current_app, jsonify, make_response
from flask_cors import CORS
from flask_restful import Api

api = Api()


def create_app(config_name):

    from api import views

    app = Flask(__name__)

    app.config.from_object(config[config_name])

    CORS(app)
    api.init_app(app)

    Swagger(app=app)
    return app
