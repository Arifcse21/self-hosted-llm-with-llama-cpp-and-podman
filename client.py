import sys
from openai import OpenAI

# Connect directly to your local Podman llama-server instance
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed-for-local-inference"
)

# Maintain the chat logs so the model remembers context
conversation_history = [
    {"role": "system", "content": "You are a helpful, concise AI assistant."}
]

print("🤖 Qwen3 Local Chat Initialized! Type 'exit' or 'quit' to end.\n")

while True:
    try:
        user_input = input("\nYou 👤: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue

        # Append the new question to the active memory
        conversation_history.append({"role": "user", "content": user_input})
        
        print("AI 🤖: ", end="", flush=True)

        # Trigger the streaming API request
        stream = client.chat.completions.create(
            model="local-qwen3-4b",
            messages=conversation_history,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            # Safely grab the incoming text fragment
            token = chunk.choices[0].delta.content
            if token is not None:
                print(token, end="", flush=True)
                full_response += token
                
        print() # Newline after response finishes
        
        # Save the assistant's complete answer to history for context
        conversation_history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C to break out or reset
        print("\nChat session interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error contacting container: {e}")

