import json
import os
import subprocess
import threading
import time
import urllib.parse
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variable to track Streamlit process
streamlit_process = None
streamlit_port = 8501

# Load NASA data on startup
def load_nasa_data():
    """Load the restructured NASA data from JSON file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'data/restructured_nasa_data.json')
        
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data['datasets']
    except Exception as e:
        print(f"Error loading NASA data: {e}")
        return {}

# Load data on startup
nasa_datasets = load_nasa_data()

def find_experiment_by_title(title):
    """Find experiment data by matching the title"""
    # Convert search title to lowercase for case-insensitive matching
    search_title = title.lower()
    
    for dataset_name, dataset_info in nasa_datasets.items():
        # Check if the title matches either the dataset name or the paper title
        if (search_title in dataset_name.lower() or 
            search_title in dataset_info.get('matched_paper', {}).get('title', '').lower()):
            
            # Format the data to match the structure from app.py
            experiment_info = {
                "osd_id": dataset_info.get('osd_id', ''),
                "glds_id": dataset_info.get('glds_id', ''),
                "matched_paper": {
                    "title": dataset_info.get('matched_paper', {}).get('title', ''),
                    "link": dataset_info.get('matched_paper', {}).get('link', ''),
                },
                "metadata": {
                    "organism": dataset_info.get('metadata', {}).get('organism', ''),
                    "tissue": dataset_info.get('metadata', {}).get('tissue', ''),
                    "assay_type": dataset_info.get('metadata', {}).get('assay_type', ''),
                    "mission": dataset_info.get('metadata', {}).get('mission', ''),
                    "description": dataset_info.get('metadata', {}).get('description', ''),
                    "categories": dataset_info.get('categories', []),
                },
                "visualizations": {
                    "files_url": dataset_info.get('visualizations', {}).get('files_url', '')
                },
            }
            return experiment_info
    
    return None

def start_streamlit_app(experiment_info):
    """Start Streamlit app with experiment data"""
    global streamlit_process
    
    try:
        # Stop existing Streamlit process if running
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait()
        
        # Encode experiment data for URL parameter
        experiment_data_encoded = urllib.parse.quote(json.dumps(experiment_info))
        
        # Get current directory and app.py path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_py_path = os.path.join(current_dir, 'app.py')
        
        # Start new Streamlit process
        cmd = [
            'streamlit', 'run', app_py_path,
            '--server.port', str(streamlit_port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ]
        
        streamlit_process = subprocess.Popen(
            cmd,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for Streamlit to start
        time.sleep(3)
        
        # Construct the URL with experiment data
        base_url = f"http://localhost:{streamlit_port}"
        url_with_params = f"{base_url}/?experiment_data={experiment_data_encoded}"
        
        return url_with_params
        
    except Exception as e:
        print(f"Error starting Streamlit app: {e}")
        return None

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': 'API server is running!', 'status': 'healthy'}), 200

@app.route('/streamlit/status', methods=['GET'])
def streamlit_status():
    """Check if Streamlit is running"""
    global streamlit_process
    
    if streamlit_process and streamlit_process.poll() is None:
        return jsonify({
            'status': 'running',
            'url': f"http://localhost:{streamlit_port}",
            'message': 'Streamlit app is running'
        }), 200
    else:
        return jsonify({
            'status': 'not_running',
            'message': 'Streamlit app is not running'
        }), 200

@app.route('/streamlit/stop', methods=['POST'])
def stop_streamlit():
    """Stop the Streamlit app"""
    global streamlit_process
    
    try:
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait()
            return jsonify({
                'message': 'Streamlit app stopped successfully',
                'status': 'stopped'
            }), 200
        else:
            return jsonify({
                'message': 'Streamlit app was not running',
                'status': 'not_running'
            }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/experiment', methods=['POST'])
def get_experiment_data():
    try:
        # Get JSON data from request body
        data = request.get_json()
        
        # Check if request body is valid JSON
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        # Check if title is provided
        if 'title' not in data:
            return jsonify({'error': 'Title is required'}), 400
        
        title = data['title']
        
        # Find experiment data by title
        experiment_info = find_experiment_by_title(title)
        
        if experiment_info:
            # Start Streamlit app with experiment data
            streamlit_url = start_streamlit_app(experiment_info)
            
            if streamlit_url:
                response_data = {
                    'message': 'Experiment data found and Streamlit app started successfully',
                    'experiment_info': experiment_info,
                    'streamlit_url': streamlit_url,
                    'status': 'success'
                }
                return jsonify(response_data), 200
            else:
                return jsonify({
                    'error': 'Failed to start Streamlit app',
                    'experiment_info': experiment_info,
                    'status': 'error'
                }), 500
        else:
            return jsonify({
                'error': 'No experiment found matching the provided title',
                'searched_title': title,
                'status': 'not_found'
            }), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def cleanup_streamlit():
    """Cleanup function to stop Streamlit process on exit"""
    global streamlit_process
    if streamlit_process and streamlit_process.poll() is None:
        streamlit_process.terminate()
        streamlit_process.wait()

if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_streamlit)
    
    print(f"Loaded {len(nasa_datasets)} NASA datasets")
    print(f"API server starting on http://localhost:5001")
    print(f"Streamlit apps will be served on port {streamlit_port}")
    app.run(debug=True, host='0.0.0.0', port=5001)