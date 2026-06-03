import os
import base64
from openai import OpenAI

# Configure client
client = OpenAI(
    api_key="sk-1bfd4c04b7a8423bb419704875b15f35",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def image_to_base64(image_path):
    """Convert local image to Base64 encoding"""
    try:
        with open(image_path, "rb") as image_file:
            # Encode as Base64 string
            base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
            # Format as Data URL
            return f"data:image/jpeg;base64,{base64_encoded}"
    except FileNotFoundError:
        print(f"Error: Cannot find image file {image_path}")
        return None
    except Exception as e:
        print(f"Image encoding failed: {e}")
        return None

def call_qwen3_vl_with_local_image():

    # Replace with your local image path
    local_image_path = r"C:\Users\zzq\Desktop\hello.jpg"

    # Convert to Base64
    image_base64 = image_to_base64(local_image_path)
    if not image_base64:
        return

    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Which three anime characters are in this image"},
                        {"type": "image_url", "image_url": {"url": image_base64}}
                    ]
                }
            ],
            temperature=0.7
        )
        print("Answer: ", completion.choices[0].message.content)

    except Exception as e:
        print(f"Call failed: {e}")

if __name__ == "__main__":
    call_qwen3_vl_with_local_image()