import openai
import time
from flask import Flask, render_template, jsonify, request

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual API key

app = Flask(__name__)

def generate_text(prompt):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message['content']
        except Exception as e:
            if "Rate limit" in str(e):
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"An error occurred: {e}")
                break

def generate_image(prompt):
    while True:
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1,
            )
            return response['data'][0]['url']
        except Exception as e:
            if "Rate limit" in str(e):
                print("Rate limit exceeded. Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"An error occurred: {e}")
                break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_prompt = request.json.get('prompt')
    text_response = generate_text(user_prompt)
    image_url = generate_image(user_prompt)
    return jsonify({'text': text_response, 'image_url': image_url})

if __name__ == "__main__":
    app.run(debug=True)
