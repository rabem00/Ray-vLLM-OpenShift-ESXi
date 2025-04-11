from litellm_wrapper.litellm_wrapper import LitellmModel

print("Attempting to initialize LitellmModel...")
try:
    llm = LitellmModel(model="ollama/llama3", api_base="http://localhost:11434")
    print("LitellmModel initialized successfully!")

    # Optional: Test a simple call
    print("Testing a call...")
    response = llm(prompt="Why is the sky blue?", max_tokens=50)
    print("LLM Response:", response)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure the Ollama server is running and accessible.")
