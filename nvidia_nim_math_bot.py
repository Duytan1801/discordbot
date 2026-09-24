import os
import discord
from discord.ext import commands
import openai

# 1. Setup API clients and grab environment variables
# NVIDIA NIM uses the standard OpenAI-compatible client structure.
DISCORD_TOKEN = "MTQ3NzcxMDY4NjE0MTI4ODQ4OQ.G0k3J1.KPVhGQucNX9ZgjsqhBf73ZGdXD2S8P9reEkLc0
"
NVIDIA_API_KEY = "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"

# Defaulting to a strong math & reasoning model available on NIM (like llama-3.1-70b-instruct or deepseek)
# You can change this model string in your hosting provider's environment variables if desired.
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")

# Initialize OpenAI client pointing to NVIDIA NIM infrastructure
client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# 2. Configure Discord Bot Intents
intents = discord.Intents.default()
intents.message_content = True  # Crucial: Allows the bot to read math problems in chat
intents.members = True

bot = commands.Bot(command_list=[], intents=intents)

# 3. System Instructions to mimic MathGPT tutoring persona
MATH_GPT_SYSTEM_PROMPT = (
    "You are an expert, highly encouraging AI Math Tutor modeled after MathGPT. "
    "Your goal is to teach, not just give answers. When a user asks a math problem:\n"
    "1. Never present just the raw final answer up front.\n"
    "2. Walk through the solution systematically, step-by-step.\n"
    "3. Explain the logic, formulas, or rules used in each transition.\n"
    "4. Format expressions clearly using clean notation or markdown.\n"
    "5. Keep your explanations accessible, concise, and structured."
)

@bot.event
async def on_ready():
    print(f"Success! Logged in as {bot.user.name}")
    print(f"Routing queries to NVIDIA NIM model: {NVIDIA_MODEL}")

@bot.event
async def on_message(message):
    # Ignore messages sent by the bot itself to prevent infinite loops
    if message.author == bot.user:
        return

    # Check if the bot was mentioned or tagged in the chat
    if bot.user.mentioned_in(message):
        # Clean the message text by stripping out the bot's mention tag
        user_query = message.content.replace(f'<@{bot.user.id}>', '').strip()
        
        if not user_query:
            await message.channel.send("Hello! Mention me along with a math problem, and I'll break it down step-by-step for you.")
            return

        # Send a typing indicator to let users know the AI is processing the math problem
        async with message.channel.typing():
            try:
                # Call NVIDIA NIM endpoint using the OpenAI SDK structure
                completion = client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "system", "content":MATH_GPT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2, # Low temperature ensures structured, consistent mathematical reasoning
                    max_tokens=1024
                )
                
                # Extract and reply with the step-by-step solution
                response_text = completion.choices[0].message.content
                await message.reply(response_text)
                
            except Exception as e:
                print(f"Error querying NVIDIA NIM: {e}")
                await message.reply("Sorry, I encountered an error processing that math problem through NVIDIA NIM. Make sure your API key and model name are valid!")

# Launch the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN or not NVIDIA_API_KEY:
        print("CRITICAL ERROR: Please define DISCORD_TOKEN and NVIDIA_API_KEY variables in your environment.")
    else:
        bot.run(DISCORD_TOKEN)
