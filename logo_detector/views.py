import logging
import json

from flask import Flask, request, render_template
from flask_cors import CORS

from logo_detector.config import Config
from logo_detector.detector import detect_resnet, get_resnet


app = Flask(__name__)
log = logging.getLogger()
log.setLevel(logging.INFO)


# App Factory
def create_app(config_class=Config):
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(config_class)

    @app.route("/", methods=["GET"])
    def root():
        return """<form method="POST">
<input name="image_url">
<input type="submit">
</form>"""
    
    @app.route("/", methods=["POST"])
    def predict():
        image_url = request.form['image_url']
        logo = detect_resnet(image_url, model=config_class.model, config=config_class.model_config)
        return """<h1> This is "{}"!</h1>
<h1><img src="{}"></h1>
""".format(logo, image_url)
    return app


APP = create_app()
