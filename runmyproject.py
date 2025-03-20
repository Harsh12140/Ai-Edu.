from flask import Flask, render_template, jsonify, request
from backend.imageandtextgen import generate_image, generate_text  # Import your image generation function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_prompt = request.json.get('prompt')
    text_response = generate_text(user_prompt)
    image_url = generate_image(user_prompt)
    return jsonify({'text': text_response, 'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)