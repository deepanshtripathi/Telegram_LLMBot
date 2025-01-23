from telegram.ext import Application, CommandHandler, MessageHandler, filters
from transformers import pipeline
import torch
import os
import logging
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

API_TOKEN = "7942357053:AAGYaJ1rasf_kuIYJVEwnpK_kcRmLi0Ri00"

# Path to the locally cached model
model_path = r"C:\Users\tripa\.cache\huggingface\hub\models--TinyLlama--TinyLlama-1.1B-Chat-v1.0\snapshots\fe8a4ea1ffedaf415f4da2f062534de366a451e6"

logging.info("Initializing the model pipeline")
pipe = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    device=0  # Force GPU usage
)
logging.info("Model pipeline initialized successfully")

def format_chat_message(system_message, user_message):
    logging.info("Formatting chat message")
    return f"<|system|>\n{system_message}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>"

async def start(update, context):
    logging.info("/start command received")
    await update.message.reply_text("Hello! I am your AI Assistant. How can I help you today?")

def process_message(user_message):
    logging.info(f"Processing user message: {user_message}")

    system_message = "You are an intelligent AI Assistant who is responding to the <|user|>. Respond in a short and concise sentence."
    prompt = format_chat_message(system_message, user_message)

    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        logging.info(f"Full model output: {generated_text}")

        response = generated_text.split("<|assistant|>")[-1].strip()

        logging.info(f"Cleaned response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        raise

async def handle_message(update, context):
    user_message = update.message.text
    await update.message.reply_text(f"Thinking....")

    try:
        response = process_message(user_message)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text("Error processing your request.")
        print(f"Error processing message: {e}")

def main():
    logging.info(f"Using device: {torch.cuda.get_device_name(0)}")

    app = Application.builder().token(API_TOKEN).build()
    logging.info("Telegram bot initialized.")

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Starting the bot polling...")
    app.run_polling()

if __name__ == "__main__":
    main()