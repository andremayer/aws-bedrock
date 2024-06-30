import streamlit as st
import boto3
import json
import os
import random
import base64
from PIL import Image
from io import BytesIO

os.environ["AWS_PROFILE"] = "andre"

client = boto3.client("bedrock-runtime", region_name="us-west-2")

model_id = "amazon.titan-image-generator-v1"

def generate_image(prompt):
    try:
        seed = random.randint(0, 2147483647)

        native_request = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": prompt},
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "cfgScale": 8.0,
                "height": 512,
                "width": 512,
                "seed": seed,
            },
        }

        request = json.dumps(native_request)

        response = client.invoke_model(modelId=model_id, body=request)

        model_response = json.loads(response["body"].read())

        base64_image_data = model_response["images"][0]

        image_data = base64.b64decode(base64_image_data)

        image = Image.open(BytesIO(image_data))

        return image

    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

st.title("Demo: Amazon Titan Image Generator")

prompt = st.text_area("Enter your prompt for image generation:", height=100)
generate_button = st.button("Generate Image")

if generate_button and prompt:
    st.write("Generating image...")

    generated_image = generate_image(prompt)

    if generated_image:
        st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Failed to generate image. Please try again.")
