import openai
import os
import re
import time
import logging
from typing import Any
from dotenv import load_dotenv


from openai import OpenAI
client = OpenAI()

response = client.images.generate(
model="dall-e-3",
prompt="a yard, focuns on girl",
size="1024x1024",
quality="standard",
n=1,
)

image_url = response.data[0].url

print("Generated image URL:", image_url)
