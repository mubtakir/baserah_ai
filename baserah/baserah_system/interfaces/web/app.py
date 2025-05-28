#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Interface for Basira System

This module implements a simple web interface for the Basira System using Flask.

Author: Basira System Development Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from io import BytesIO

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    logging.error("Flask not available, web interface will not work")
    FLASK_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    logging.error("PIL not available, image generation will not work")
    PIL_AVAILABLE = False

# Import Basira System components
try:
    from mathematical_core.general_shape_equation import GeneralShapeEquation, EquationType, LearningMode
    from arabic_nlp.morphology.root_extractor import ArabicRootExtractor
    from arabic_nlp.syntax.syntax_analyzer import ArabicSyntaxAnalyzer
    from arabic_nlp.rhetoric.rhetoric_analyzer import ArabicRhetoricAnalyzer
    from creative_generation.image.image_generator import ImageGenerator, GenerationParameters, GenerationMode
    from code_execution.code_executor import CodeExecutor, ProgrammingLanguage, ExecutionConfig
except ImportError as e:
    logging.error(f"Failed to import Basira System components: {e}")
    logging.error("Make sure you're running the script from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("basira_web.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('basira_web')

# Create Flask app
if FLASK_AVAILABLE:
    app = Flask(__name__, template_folder='templates', static_folder='static')
    CORS(app)  # Enable CORS for all routes
else:
    app = None

# Initialize Basira System components
root_extractor = ArabicRootExtractor()
syntax_analyzer = ArabicSyntaxAnalyzer()
rhetoric_analyzer = ArabicRhetoricAnalyzer()
image_generator = ImageGenerator()
code_executor = CodeExecutor()
general_shape_equation = GeneralShapeEquation()

# Add some basic shapes to the general shape equation
general_shape_equation.add_component("circle", "(x-cx)^2 + (y-cy)^2 - r^2")
general_shape_equation.add_component("cx", "0")
general_shape_equation.add_component("cy", "0")
general_shape_equation.add_component("r", "5")


@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/api/analyze_text', methods=['POST'])
def analyze_text():
    """
    Analyze Arabic text.

    Request JSON:
    {
        "text": "Arabic text to analyze"
    }

    Response JSON:
    {
        "roots": [{"word": "word", "root": "root"}, ...],
        "syntax": {...},
        "rhetoric": {...}
    }
    """
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400

    text = request.json['text']

    try:
        # Extract roots
        roots = root_extractor.extract_roots(text)
        roots_json = [{'word': word, 'root': root} for word, root in roots]

        # Analyze syntax
        syntax_analyses = syntax_analyzer.analyze(text)
        syntax_json = []
        for analysis in syntax_analyses:
            tokens_json = []
            for token in analysis.tokens:
                tokens_json.append({
                    'text': token.text,
                    'position': token.position,
                    'length': token.length,
                    'pos_tag': token.pos_tag
                })

            relations_json = []
            for relation in analysis.relations:
                head = analysis.tokens[relation.head_index].text
                dependent = analysis.tokens[relation.dependent_index].text
                relations_json.append({
                    'relation_type': relation.relation_type,
                    'head': head,
                    'dependent': dependent
                })

            syntax_json.append({
                'sentence': analysis.sentence,
                'tokens': tokens_json,
                'relations': relations_json
            })

        # Analyze rhetoric
        rhetoric_analysis = rhetoric_analyzer.analyze(text)
        features_json = []
        for feature in rhetoric_analysis.features:
            features_json.append({
                'device_type': feature.device_type,
                'text': feature.text,
                'start_index': feature.start_index,
                'end_index': feature.end_index,
                'confidence': feature.confidence
            })

        rhetoric_json = {
            'features': features_json,
            'summary': rhetoric_analysis.summary,
            'aesthetic_score': rhetoric_analysis.aesthetic_score,
            'style_profile': rhetoric_analysis.style_profile
        }

        return jsonify({
            'roots': roots_json,
            'syntax': syntax_json,
            'rhetoric': rhetoric_json
        })

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_image', methods=['POST'])
def generate_image():
    """
    Generate an image based on text description.

    Request JSON:
    {
        "text": "Text description",
        "width": 512,
        "height": 512,
        "mode": "text_to_image"
    }

    Response JSON:
    {
        "image": "base64 encoded image",
        "generation_time": 1.23
    }
    """
    if not request.json:
        return jsonify({'error': 'No input data provided'}), 400

    # Check for different input types based on mode
    mode_str = request.json.get('mode', 'text_to_image')
    width = request.json.get('width', 512)
    height = request.json.get('height', 512)

    try:
        # Set generation parameters
        mode = GenerationMode(mode_str)
        parameters = GenerationParameters(
            mode=mode,
            width=width,
            height=height,
            seed=42
        )

        # Determine input data based on mode
        if mode == GenerationMode.TEXT_TO_IMAGE:
            if 'text' not in request.json:
                return jsonify({'error': 'No text provided for text_to_image mode'}), 400
            input_data = request.json['text']

        elif mode == GenerationMode.ARABIC_TEXT_TO_IMAGE:
            if 'text' not in request.json:
                return jsonify({'error': 'No text provided for arabic_text_to_image mode'}), 400
            input_data = request.json['text']

        elif mode == GenerationMode.CALLIGRAPHY:
            if 'text' not in request.json:
                return jsonify({'error': 'No text provided for calligraphy mode'}), 400
            input_data = request.json['text']

        elif mode == GenerationMode.EQUATION_TO_IMAGE:
            if 'equation' not in request.json:
                return jsonify({'error': 'No equation provided for equation_to_image mode'}), 400
            input_data = request.json['equation']

        elif mode == GenerationMode.CONCEPT_TO_IMAGE:
            if 'concept' not in request.json:
                return jsonify({'error': 'No concept provided for concept_to_image mode'}), 400
            # Convert concept to numpy array if it's a list
            concept = request.json['concept']
            if isinstance(concept, list):
                input_data = np.array(concept, dtype=np.float32)
            else:
                input_data = concept

        elif mode == GenerationMode.STYLE_TRANSFER:
            if 'content_image' not in request.json or 'style_image' not in request.json:
                return jsonify({'error': 'Both content_image and style_image must be provided for style_transfer mode'}), 400

            # Decode base64 images
            try:
                content_image_data = base64.b64decode(request.json['content_image'])
                style_image_data = base64.b64decode(request.json['style_image'])

                content_image = Image.open(BytesIO(content_image_data))
                style_image = Image.open(BytesIO(style_image_data))

                # Convert to numpy arrays
                content_array = np.array(content_image) / 255.0
                style_array = np.array(style_image) / 255.0

                input_data = (content_array, style_array)
            except Exception as e:
                return jsonify({'error': f'Error decoding images: {str(e)}'}), 400

        elif mode == GenerationMode.HYBRID:
            # For hybrid mode, we need a dictionary of inputs
            hybrid_inputs = {}

            if 'text' in request.json:
                hybrid_inputs['text'] = request.json['text']

            if 'arabic_text' in request.json:
                hybrid_inputs['arabic_text'] = request.json['arabic_text']

            if 'equation' in request.json:
                hybrid_inputs['equation'] = request.json['equation']

            if 'concept' in request.json:
                concept = request.json['concept']
                if isinstance(concept, list):
                    hybrid_inputs['concept'] = np.array(concept, dtype=np.float32)
                else:
                    hybrid_inputs['concept'] = concept

            if not hybrid_inputs:
                return jsonify({'error': 'No valid inputs provided for hybrid mode'}), 400

            input_data = hybrid_inputs

        else:
            return jsonify({'error': f'Unsupported generation mode: {mode_str}'}), 400

        # Generate image
        result = image_generator.generate_image(input_data, parameters)

        # Convert image to base64
        if PIL_AVAILABLE:
            if isinstance(result.image, np.ndarray):
                # Convert numpy array to PIL Image
                image_array = (result.image * 255).astype(np.uint8)
                image = Image.fromarray(image_array)
            else:
                image = result.image

            # Convert PIL Image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'image': img_str,
                'generation_time': result.generation_time,
                'mode': mode_str
            })
        else:
            return jsonify({'error': 'PIL not available, cannot generate image'}), 500

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/execute_code', methods=['POST'])
def execute_code():
    """
    Execute code in a secure sandbox environment.

    Request JSON:
    {
        "code": "Code to execute",
        "language": "python",
        "timeout": 5.0,
        "input": "Optional input data"
    }

    Response JSON:
    {
        "stdout": "Standard output",
        "stderr": "Standard error",
        "exit_code": 0,
        "execution_time": 1.23
    }
    """
    if not request.json or 'code' not in request.json or 'language' not in request.json:
        return jsonify({'error': 'No code or language provided'}), 400

    code = request.json['code']
    language_str = request.json['language']
    timeout = request.json.get('timeout', 5.0)
    input_data = request.json.get('input')

    try:
        # Set execution configuration
        language = ProgrammingLanguage(language_str)
        config = ExecutionConfig(
            language=language,
            timeout=timeout,
            allow_network=False,
            allow_file_access=False
        )

        # Execute code
        result = code_executor.execute(code, language, config, input_data)

        return jsonify({
            'stdout': result.stdout,
            'stderr': result.stderr,
            'exit_code': result.exit_code,
            'execution_time': result.execution_time
        })

    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate_equation', methods=['POST'])
def evaluate_equation():
    """
    Evaluate a mathematical equation.

    Request JSON:
    {
        "equation": "Equation to evaluate",
        "variables": {"x": 1, "y": 2, ...}
    }

    Response JSON:
    {
        "result": {"component1": value1, "component2": value2, ...}
    }
    """
    if not request.json or 'equation' not in request.json:
        return jsonify({'error': 'No equation provided'}), 400

    equation_str = request.json['equation']
    variables = request.json.get('variables', {})

    try:
        # Create a new equation
        equation = GeneralShapeEquation()
        equation.add_component("user_equation", equation_str)

        # Evaluate equation
        result = equation.evaluate(variables)

        return jsonify({
            'result': result
        })

    except Exception as e:
        logger.error(f"Error evaluating equation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system_info', methods=['GET'])
def system_info():
    """
    Get information about the Basira System.

    Response JSON:
    {
        "version": "1.0.0",
        "components": ["mathematical_core", "arabic_nlp", ...],
        "capabilities": ["text_analysis", "image_generation", ...]
    }
    """
    try:
        return jsonify({
            'version': '1.0.0',
            'components': [
                'mathematical_core',
                'arabic_nlp',
                'creative_generation',
                'code_execution'
            ],
            'capabilities': [
                'text_analysis',
                'image_generation',
                'code_execution',
                'equation_evaluation'
            ]
        })

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500


def create_templates_directory():
    """Create templates directory and index.html if they don't exist."""
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)

    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نظام بصيرة</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
            direction: rtl;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            text-align: center;
        }

        h1 {
            margin: 0;
            font-size: 2.5em;
        }

        .section {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
            padding: 20px;
        }

        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: inherit;
            font-size: 1em;
            margin-bottom: 10px;
            direction: rtl;
        }

        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #2980b9;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            white-space: pre-wrap;
            direction: rtl;
        }

        .image-result {
            text-align: center;
            margin-top: 20px;
        }

        .image-result img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }

        footer {
            text-align: center;
            padding: 20px 0;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>نظام بصيرة</h1>
            <p>نموذج لغوي معرفي توليدي ونظام ذكاء اصطناعي مبتكر</p>
        </div>
    </header>

    <div class="container">
        <div class="section">
            <h2>تحليل النص العربي</h2>
            <textarea id="arabic-text" placeholder="أدخل النص العربي هنا..."></textarea>
            <button id="analyze-button">تحليل النص</button>
            <div id="analysis-result" class="result" style="display: none;"></div>
        </div>

        <div class="section">
            <h2>توليد الصور</h2>
            <textarea id="image-prompt" placeholder="أدخل وصفاً للصورة..."></textarea>
            <select id="image-mode-select">
                <option value="text_to_image">نص إلى صورة</option>
                <option value="arabic_text_to_image">نص عربي إلى صورة</option>
                <option value="calligraphy">خط عربي</option>
                <option value="equation_to_image">معادلة إلى صورة</option>
            </select>
            <button id="generate-button">توليد صورة</button>
            <div id="image-result" class="image-result" style="display: none;">
                <img id="generated-image" src="" alt="الصورة المولدة">
                <p id="generation-info"></p>
            </div>
        </div>

        <div class="section">
            <h2>تنفيذ الأكواد</h2>
            <textarea id="code-input" placeholder="أدخل الكود هنا..."></textarea>
            <select id="language-select">
                <option value="python">Python</option>
                <option value="javascript">JavaScript</option>
                <option value="bash">Bash</option>
            </select>
            <button id="execute-button">تنفيذ الكود</button>
            <div id="execution-result" class="result" style="display: none;"></div>
        </div>

        <div class="section">
            <h2>تقييم المعادلات</h2>
            <textarea id="equation-input" placeholder="أدخل المعادلة هنا..."></textarea>
            <textarea id="variables-input" placeholder='أدخل المتغيرات بصيغة JSON: {"x": 1, "y": 2}'></textarea>
            <button id="evaluate-button">تقييم المعادلة</button>
            <div id="equation-result" class="result" style="display: none;"></div>
        </div>
    </div>

    <footer>
        <div class="container">
            <p>نظام بصيرة - الإصدار 1.0.0</p>
            <p>فريق تطوير نظام بصيرة</p>
        </div>
    </footer>

    <script>
        document.getElementById('analyze-button').addEventListener('click', async () => {
            const text = document.getElementById('arabic-text').value;
            if (!text) return;

            try {
                const response = await fetch('/api/analyze_text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                const resultElement = document.getElementById('analysis-result');
                resultElement.textContent = JSON.stringify(data, null, 2);
                resultElement.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('حدث خطأ أثناء تحليل النص');
            }
        });

        document.getElementById('generate-button').addEventListener('click', async () => {
            const text = document.getElementById('image-prompt').value;
            const mode = document.getElementById('image-mode-select').value;
            if (!text) return;

            try {
                // Prepare request body based on mode
                let requestBody = {
                    width: 512,
                    height: 512,
                    mode: mode
                };

                if (mode === 'text_to_image' || mode === 'arabic_text_to_image' || mode === 'calligraphy') {
                    requestBody.text = text;
                } else if (mode === 'equation_to_image') {
                    requestBody.equation = text;
                }

                const response = await fetch('/api/generate_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                const imageElement = document.getElementById('generated-image');
                imageElement.src = `data:image/png;base64,${data.image}`;

                const infoElement = document.getElementById('generation-info');
                infoElement.textContent = `وقت التوليد: ${data.generation_time.toFixed(2)} ثانية | النمط: ${data.mode}`;

                const resultElement = document.getElementById('image-result');
                resultElement.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert(`حدث خطأ أثناء توليد الصورة: ${error.message}`);
            }
        });

        document.getElementById('execute-button').addEventListener('click', async () => {
            const code = document.getElementById('code-input').value;
            const language = document.getElementById('language-select').value;
            if (!code) return;

            try {
                const response = await fetch('/api/execute_code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        code,
                        language,
                        timeout: 5.0
                    })
                });

                const data = await response.json();

                const resultElement = document.getElementById('execution-result');
                resultElement.textContent = `الخرج القياسي:\n${data.stdout}\n\nالخطأ القياسي:\n${data.stderr}\n\nرمز الخروج: ${data.exit_code}\nوقت التنفيذ: ${data.execution_time} ثانية`;
                resultElement.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('حدث خطأ أثناء تنفيذ الكود');
            }
        });

        document.getElementById('evaluate-button').addEventListener('click', async () => {
            const equation = document.getElementById('equation-input').value;
            const variablesText = document.getElementById('variables-input').value;
            if (!equation) return;

            let variables = {};
            try {
                if (variablesText) {
                    variables = JSON.parse(variablesText);
                }

                const response = await fetch('/api/evaluate_equation', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        equation,
                        variables
                    })
                });

                const data = await response.json();

                const resultElement = document.getElementById('equation-result');
                resultElement.textContent = JSON.stringify(data.result, null, 2);
                resultElement.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('حدث خطأ أثناء تقييم المعادلة');
            }
        });
    </script>
</body>
</html>""")

    # Create static directory
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)


def main():
    """Run the Flask application."""
    if not FLASK_AVAILABLE:
        logger.error("Flask not available, cannot run web interface")
        return

    # Create templates directory and index.html if they don't exist
    create_templates_directory()

    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
