from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware for the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the diffusion model
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe.to("cuda")


@app.get("/generate")
async def generate(prompt: str):
    if not prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Generate the image
    images = pipe(prompt=prompt).images[0]

    # Convert the image to bytes
    img_byte_arr = BytesIO()
    images.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(BytesIO(img_byte_arr), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
