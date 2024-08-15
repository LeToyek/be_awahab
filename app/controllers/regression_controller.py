from flask import Blueprint, request, jsonify
from app.services.regression_service import RegressionService

# Create a Blueprint for the regression routes
regression_bp = Blueprint('regression', __name__)

# Create an instance of RegressionService
regression_service = RegressionService()

@regression_bp.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Regression API'})

@regression_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Perform prediction from the file
        res_ti,res_si = regression_service.predict_from_file(file)
        json = {
            'TI': res_ti,
            'SI': res_si
        }
        return jsonify(json)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
