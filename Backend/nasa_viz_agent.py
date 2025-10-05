"""
NASA Publication Interactive Visualization REST API
Ready for Postman testing with complete error handling
"""

import json
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Optional
from openai import OpenAI
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class NASAInteractiveVisualizationAgent:
    def __init__(self, api_key: str, cache_dir: str = "./csv_cache"):
        """Initialize the agent with OpenAI API key and cache directory."""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.df_cache = {}
        
    def load_nasa_data(self, file_path: str) -> Dict:
        """Load the restructured NASA data JSON."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_publication_details(self, publication_title: str, nasa_data: Dict) -> Optional[Dict]:
        """Fetch publication details from the NASA data."""
        datasets = nasa_data.get('datasets', {})
        
        # First try exact match
        for title, data in datasets.items():
            if publication_title.lower() == title.lower():
                return self._extract_publication_info(data)
        
        # Then try partial match
        for title, data in datasets.items():
            if publication_title.lower() in title.lower() or title.lower() in publication_title.lower():
                return self._extract_publication_info(data)
        
        return None
    
    def _extract_publication_info(self, data: Dict) -> Dict:
        """Extract publication information from dataset."""
        return {
            'title': data['matched_paper']['title'],
            'link': data['matched_paper']['link'],
            'description': data['metadata'].get('description', ''),
            'files_url': data['visualizations']['files_url'],
            'osd_id': data.get('osd_id', ''),
            'organism': data['metadata'].get('organism', ''),
            'tissue': data['metadata'].get('tissue', ''),
            'assay_type': data['metadata'].get('assay_type', ''),
            'mission': data['metadata'].get('mission', '')
        }
    
    def fetch_visualization_files(self, files_url: str) -> Dict:
        """Fetch available CSV files from the visualization API."""
        try:
            response = requests.get(files_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            csv_files = {}
            for dataset_id, dataset_info in data.items():
                files = dataset_info.get('files', {})
                for filename, file_info in files.items():
                    if filename.endswith('.csv'):
                        csv_files[filename] = file_info['REST_URL']
            
            print(f"Found {len(csv_files)} CSV files")
            return csv_files
        except Exception as e:
            print(f"Error fetching files: {e}")
            return {}
    
    def download_csv_file(self, url: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Download and parse a CSV file with caching."""
        if url in self.df_cache:
            return self.df_cache[url]
        
        # Create safe filename for cache
        safe_filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
        cache_filename = os.path.join(self.cache_dir, safe_filename)
        
        if os.path.exists(cache_filename):
            try:
                df = pd.read_csv(cache_filename)
                self.df_cache[url] = df
                return df
            except:
                pass
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                
                self.df_cache[url] = df
                try:
                    df.to_csv(cache_filename, index=False)
                except:
                    pass
                
                return df
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to download {url}: {e}")
                    return None
        return None
    
    def generate_comprehensive_analysis(
        self, 
        publication_details: Dict, 
        csv_files: Dict[str, str]
    ) -> Dict:
        """Download all CSVs and generate comprehensive analysis."""
        
        print(f"\nAnalyzing {len(csv_files)} CSV files...")
        csv_samples = {}
        
        for i, (filename, url) in enumerate(csv_files.items(), 1):
            print(f"  [{i}/{len(csv_files)}] {filename}")
            df = self.download_csv_file(url)
            if df is not None:
                try:
                    csv_samples[filename] = {
                        'columns': df.columns.tolist(),
                        'shape': df.shape,
                        'dtypes': df.dtypes.astype(str).to_dict(),
                        'sample_rows': df.head(3).to_dict(),
                        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else None,
                    }
                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
        
        csv_context = self._format_csv_context(csv_samples)
        
        prompt = f"""
You are a NASA space biology data visualization expert.

**Publication Information:**
- Title: {publication_details['title']}
- Organism: {publication_details['organism']}
- Tissue: {publication_details['tissue']}
- Assay Type: {publication_details['assay_type']}
- Mission: {publication_details['mission']}

**Available Data Files ({len(csv_samples)} files):**
{csv_context}

Generate a JSON response:

{{
  "study_summary": {{
    "study_method": "Detailed experimental methods description",
    "results_conclusions": "Key findings and conclusions"
  }},
  "interactive_visualizations": [
    {{
      "csv_filename": "exact filename from available files",
      "plot_type": "plotly_scatter|plotly_line|plotly_bar|plotly_heatmap|plotly_box",
      "title": "Clear descriptive title",
      "description": "What this reveals scientifically",
      "x_axis": "column name",
      "y_axis": "column name",
      "color_by": "column name or null",
      "hover_data": ["col1", "col2"],
      "scientific_insight": "Key biological finding from this visualization"
    }}
  ]
}}

Requirements:
- Provide exactly 3 visualizations
- Use different CSV files
- Choose columns that exist in the data
- Focus on scientific significance

Return ONLY valid JSON.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a scientific visualization expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _format_csv_context(self, csv_samples: Dict) -> str:
        """Format CSV context for LLM."""
        parts = []
        for filename, info in list(csv_samples.items())[:15]:  # Limit to 15 files
            part = f"\n{filename}:\n"
            part += f"  Shape: {info['shape'][0]} rows × {info['shape'][1]} columns\n"
            part += f"  Columns: {', '.join(info['columns'][:8])}\n"
            parts.append(part)
        return '\n'.join(parts)
    
    def generate_interactive_plot_code(self, viz_rec: Dict, df: pd.DataFrame) -> str:
        """Generate Plotly code for interactive visualization."""
        
        df_info = {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'sample': df.head(5).to_dict()
        }
        
        prompt = f"""
Generate Plotly code for this visualization:

Type: {viz_rec['plot_type']}
Title: {viz_rec['title']}
X-axis: {viz_rec.get('x_axis', 'auto')}
Y-axis: {viz_rec.get('y_axis', 'auto')}
Color by: {viz_rec.get('color_by', 'none')}

DataFrame columns: {df_info['columns']}
Shape: {df_info['shape']}

Requirements:
1. Use plotly.express (px) or plotly.graph_objects (go)
2. Work with DataFrame 'df'
3. Create interactive plot with hover, zoom, pan
4. Professional styling with title and labels
5. Save to variable 'fig'
6. Do NOT call fig.show()
7. Handle missing data

Return ONLY Python code, no markdown.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Generate clean Plotly code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        code = response.choices[0].message.content
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        
        return code.strip()
    
    def execute_plotly_code(self, code: str, df: pd.DataFrame) -> Optional[Dict]:
        """Execute Plotly code and return HTML + image."""
        try:
            namespace = {
                'df': df,
                'pd': pd,
                'go': go,
                'px': px,
                'make_subplots': make_subplots,
                'np': __import__('numpy')
            }
            
            exec(code, namespace)
            
            if 'fig' not in namespace:
                return None
            
            fig = namespace['fig']
            
            # Generate HTML
            html = fig.to_html(
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            )
            
            # Generate static preview
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            except:
                # Fallback if kaleido not installed
                img_base64 = None
            
            return {
                'html': html,
                'image_base64': img_base64
            }
            
        except Exception as e:
            print(f"Error executing Plotly code: {e}")
            traceback.print_exc()
            return None
    
    def process_publication(self, publication_title: str, nasa_data_path: str) -> Dict:
        """Main processing pipeline."""
        
        print(f"\n{'='*80}")
        print(f"Processing: {publication_title}")
        print(f"{'='*80}\n")
        
        # Load publication details
        nasa_data = self.load_nasa_data(nasa_data_path)
        pub_details = self.get_publication_details(publication_title, nasa_data)
        
        if not pub_details:
            return {
                "error": "Publication not found",
                "message": "Please check the publication title and try again"
            }
        
        print(f"✓ Found: {pub_details['title']}")
        
        # Fetch CSV files
        csv_files = self.fetch_visualization_files(pub_details['files_url'])
        
        if not csv_files:
            return {
                "error": "No CSV files available",
                "message": "This publication has no visualization data"
            }
        
        # Generate analysis
        print("\nAnalyzing data with AI...")
        analysis = self.generate_comprehensive_analysis(pub_details, csv_files)
        
        # Generate plots
        print("\nGenerating interactive visualizations...")
        plots = []
        
        for i, viz_rec in enumerate(analysis['interactive_visualizations'], 1):
            print(f"\nPlot {i}/3: {viz_rec['title']}")
            
            filename = viz_rec['csv_filename']
            
            if filename not in csv_files:
                print(f"  ✗ File not found: {filename}")
                continue
            
            df = self.download_csv_file(csv_files[filename])
            if df is None:
                print(f"  ✗ Failed to download")
                continue
            
            print(f"  ✓ Data loaded: {df.shape}")
            
            plot_code = self.generate_interactive_plot_code(viz_rec, df)
            result = self.execute_plotly_code(plot_code, df)
            
            if result:
                print(f"  ✓ Plot created!")
                plots.append({
                    'plot_html': result['html'],
                    'plot_preview': result['image_base64'],
                    'plot_description': viz_rec['description'],
                    'title': viz_rec['title'],
                    'scientific_insight': viz_rec.get('scientific_insight', ''),
                    'plot_type': viz_rec['plot_type'],
                    'csv_filename': filename,
                    'code': plot_code
                })
        
        print(f"\n{'='*80}")
        print(f"✓ Complete! Generated {len(plots)} visualizations")
        print(f"{'='*80}\n")
        
        return {
            'summary_of_study_method': analysis['study_summary']['study_method'],
            'results_and_conclusions': analysis['study_summary']['results_conclusions'],
            'plots': plots,
            'metadata': {
                'publication_title': pub_details['title'],
                'publication_link': pub_details['link'],
                'osd_id': pub_details['osd_id'],
                'organism': pub_details['organism'],
                'tissue': pub_details['tissue'],
                'mission': pub_details['mission'],
                'total_csv_files_analyzed': len(csv_files),
                'visualizations_generated': len(plots)
            }
        }


# Global agent instance
agent = None

def get_agent():
    """Get or create agent instance."""
    global agent
    if agent is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        agent = NASAInteractiveVisualizationAgent(api_key=api_key)
    return agent


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'NASA Visualization API',
        'version': '1.0'
    })


@app.route('/analyze_publication', methods=['POST'])
def analyze_publication():
    """
    Main API endpoint to analyze a publication.
    
    POST /analyze_publication
    Content-Type: application/json
    
    Request Body:
    {
        "publicationTitle": "Toll Mediated Infection Response"
    }
    
    Response:
    {
        "summary_of_study_method": "...",
        "results_and_conclusions": "...",
        "plots": [
            {
                "plot_html": "...",
                "plot_preview": "base64_image",
                "plot_description": "...",
                "title": "...",
                "scientific_insight": "...",
                "plot_type": "...",
                "csv_filename": "...",
                "code": "..."
            }
        ],
        "metadata": {...}
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be JSON'
            }), 400
        
        publication_title = request.json.get('publicationTitle')
        
        if not publication_title:
            return jsonify({
                'error': 'Missing parameter',
                'message': 'publicationTitle is required'
            }), 400
        
        # Get agent
        try:
            agent_instance = get_agent()
        except ValueError as e:
            return jsonify({
                'error': 'Configuration error',
                'message': str(e)
            }), 500
        
        # Process publication
        result = agent_instance.process_publication(
            publication_title=publication_title,
            nasa_data_path='data/restructured_nasa_data.json'
        )
        
        # Check for errors in result
        if 'error' in result:
            return jsonify(result), 404
        
        return jsonify(result), 200
    
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON',
            'message': 'Request body is not valid JSON'
        }), 400
    
    except FileNotFoundError:
        return jsonify({
            'error': 'Configuration error',
            'message': 'data/restructured_nasa_data.json file not found'
        }), 500
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/publications', methods=['GET'])
def list_publications():
    """
    List all available publications.
    
    GET /publications
    
    Response:
    {
        "total": 144,
        "publications": [
            {
                "title": "...",
                "osd_id": "...",
                "organism": "...",
                "link": "..."
            }
        ]
    }
    """
    try:
        with open('data/restructured_nasa_data.json', 'r') as f:
            nasa_data = json.load(f)
        
        publications = []
        for title, data in nasa_data['datasets'].items():
            publications.append({
                'title': data['matched_paper']['title'],
                'osd_id': data.get('osd_id', ''),
                'organism': data['metadata'].get('organism', ''),
                'link': data['matched_paper']['link']
            })
        
        return jsonify({
            'total': len(publications),
            'publications': publications
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  NASA Interactive Visualization REST API                 ║
    ║  Ready for Postman Testing                               ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Endpoints:
    
    1. Health Check
       GET http://localhost:5000/health
    
    2. List Publications
       GET http://localhost:5000/publications
    
    3. Analyze Publication
       POST http://localhost:5000/analyze_publication
       Body: {"publicationTitle": "Toll Mediated Infection Response"}
    
    Required:
    - Set OPENAI_API_KEY environment variable
    - Place restructured_nasa_data.json in current directory
    
    Starting server on http://localhost:5000
    """)
    
    # Check for required files
    if not os.path.exists('data/restructured_nasa_data.json'):
        print("⚠️  WARNING: restructured_nasa_data.json not found!")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )