"""
/home/dzha866/Projects/ARK/scripts_evaluate/llm_client.py
Ollama API Client with retry mechanism and strict parameter control.
"""

import json
import base64
import logging
import io
from typing import Optional, Any

from PIL import Image
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
        timeout: int = 120,
    ):
        self.host = host
        self.model = model
        self.timeout = timeout
        self.api_endpoint = f"{host}/api/generate"
        self.headers = {"Content-Type": "application/json"}

        # Automatically disable proxies for localhost to avoid issues with system proxies
        self.proxies: Any = None
        if "localhost" in host or "127.0.0.1" in host:
            self.proxies = {"http": None, "https": None}

    @retry(
        stop=stop_after_attempt(5),  # Retry up to 5 times
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, json.JSONDecodeError)
        ),
        before_sleep=lambda retry_state: logging.warning(
            f"Request failed, retrying... (Attempt {retry_state.attempt_number})"
        ),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        options: Optional[dict] = None,
    ) -> dict:
        """
        Sends a generation request to the Ollama API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.
            images: Optional list of image file paths.
            options: Dictionary of model parameters (temperature, seed, etc.).

        Returns:
            The full JSON response from Ollama.
        """
        # Default strict parameters for reproducibility (Baseline)
        default_options = {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 8192,  # Optimized for speed
            "num_ctx": 8192,      # Increased context window for multi-image inputs
        }

        if options:
            default_options.update(options)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # We want the full response at once for batch processing
            "options": default_options,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if images:
            encoded_images = []
            for image_path in images:
                try:
                    with Image.open(image_path) as img:
                        # Resize image to balance detail and inference speed (max dim 1024).
                        # This is important for Re-ID to preserve fine-grained details.
                        img.thumbnail((1024, 1024))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG", quality=95) # Use high quality for details
                        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        encoded_images.append(encoded_string)
                except Exception as e:
                    logging.error(f"Failed to process image {image_path}: {e}")
                    raise e
            payload["images"] = encoded_images

        try:
            response = requests.post(
                self.api_endpoint,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=self.timeout,
                proxies=self.proxies,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if e.response is not None:
                logging.error(f"Ollama API Error Details: {e.response.text}")
            logging.error(f"Ollama API Connection Error: {e}")
            raise e

    def check_connection(self) -> bool:
        """Checks if the Ollama server is reachable."""
        try:
            # Ollama usually has a root endpoint or /api/tags
            requests.get(self.host, timeout=5, proxies=self.proxies)
            return True
        except requests.exceptions.RequestException:
            return False


if __name__ == "__main__":
    # Simple test to verify the client works
    logging.basicConfig(level=logging.INFO)
    client = OllamaClient()
    if client.check_connection():
        print("Successfully connected to Ollama.")
        try:
            res = client.generate("Hello, are you ready for evaluation?")
            print("Model Response:", res.get("response"))
        except Exception as e:
            print(f"Generation failed: {e}")
    else:
        print("Could not connect to Ollama. Is it running?")
