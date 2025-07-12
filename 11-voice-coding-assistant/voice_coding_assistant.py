from dotenv import load_dotenv
import os
import openai
import speech_recognition as sr

# Load environment variables from .env if present
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")

openai.api_key = OPENAI_API_KEY

recognizer = sr.Recognizer()

# Use the default system microphone
microphone = sr.Microphone()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers coding interview questions in a clear"
    " and concise manner."
)


def listen_and_transcribe() -> str | None:
    """Listen from the microphone and return the transcribed text."""
    with microphone as source:
        print("Listening... speak your question")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    print("Transcribing...")
    try:
        # Use Whisper API via speech_recognition for transcription
        text = recognizer.recognize_whisper_api(audio, api_key=OPENAI_API_KEY)
        return text
    except Exception as exc:
        print(f"Error during transcription: {exc}")
        return None


def ask_llm(question: str) -> str:
    """Send the question to the LLM and return the response."""
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    print("Coding Interview Voice Assistant. Press Ctrl+C to exit.")
    while True:
        try:
            question = listen_and_transcribe()
            if question:
                print(f"\nYou asked: {question}\n")
                answer = ask_llm(question)
                print(f"Assistant: {answer}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
