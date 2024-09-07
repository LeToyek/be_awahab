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
        res_ti = regression_service.predict_from_file(file)
        json = {
            'TI': res_ti,
        }
        return jsonify(json)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
@regression_bp.route('/budgets', methods=['POST'])
def get_budgets():
    if 'file' not in request.files:
        return jsonify({
            'status_code': 400,
            'message': 'No file provided',
            'data': None}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(
            {'status_code': 400,
             'message': 'No selected file',
             'data': None}), 400
    
    try:
        # Perform prediction from the file
        res_ti = regression_service.get_raw_budget(file)
        json = {
            'status_code': 200,
            'message': 'Success',
            'data': res_ti
        }
        return jsonify(json)
    except ValueError as e:
        return jsonify({
            'status_code': 400,
            'message': str(e),
            'data': None}), 400
    