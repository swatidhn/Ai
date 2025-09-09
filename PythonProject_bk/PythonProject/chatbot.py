import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Hugging Face OpenAI-compatible client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],  # Hugging Face token stored in environment
)

# Hardcoded system prompt (never shown in UI)
system_prompt = {
    "role": "system",
    "content": (
        "You are 'Sakhi' — a compassionate mental health companion by Beautifully Broken. "
        "You always respond in the same language the user uses. "
        "If the user types in Hindi, you respond in Hindi. "
        "If they type in English, respond in English. "
        "For code-mixed inputs (like Hindi+English), detect the dominant language and reply in that. "
        "Be empathetic, supportive, and never give medical advice or diagnosis. Feel free to suggest books , exercises or articles to the user. Be a good listener always. you can give them link of book"
    )
}

# Keep conversation history, always include system prompt first
chat_log = [system_prompt]

# Serve chat UI
@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    # Do not send system messages to frontend
    visible_log = [msg for msg in chat_log if msg["role"] != "system"]
    return templates.TemplateResponse("layout.html", {"request": request, "chat_log": visible_log})

# Handle user input
@app.post("/")
async def chat(user_input: str = Form(...)):
    chat_log.append({"role": "user", "content": user_input})

    # Call Hugging Face model
    response = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct",  # free Hugging Face model
        messages=chat_log,
        temperature=0.6
    )

    bot_response = response.choices[0].message.content
    chat_log.append({"role": "assistant", "content": bot_response})

    return {"bot_response": bot_response}
