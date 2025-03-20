import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from flask import Flask, send_file, request
import io
import os

app = Flask(__name__)

device = "cuda"  # Use CUDA (GPU)
dtype = torch.float16  # Use FP16 precision

# Step for animation generation (options: 1, 2, 4, 8)
step = 4
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"  # Model checkpoint

# Choose base model for animation generation
base = "emilianJR/epiCRealism"  # Base model (can be replaced with other models)

# Load Motion Adapter and model
adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

# Initialize pipeline
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)

# Configure scheduler for smoother transitions
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

@app.route('/generate_animation', methods=['POST'])
def generate_animation():
    try:
        data = request.get_json()
        prompt = data.get('prompt', "car driving")  # Get prompt from request, default to "car driving"
        guidance_scale = data.get('guidance_scale', 1.0) #get guidance_scale from request, default to 1.0
        num_inference_steps = data.get('num_inference_steps', step) #get num_inference_steps from request, default to step

        output = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

        # Save the output as a GIF to memory
        gif_bytes = io.BytesIO()
        export_to_gif(output.frames[0], gif_bytes)
        gif_bytes.seek(0)  # Reset the buffer position

        return send_file(gif_bytes, mimetype='image/gif', as_attachment=True, download_name='animation.gif')

    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) #run app on port 5000 or the port set in the environment.