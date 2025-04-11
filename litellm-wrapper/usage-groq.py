from litellm_wrapper.litellm_wrapper import LitellmModel

print("Attempting to initialize LitellmModel...")
try:
    llm = LitellmModel(model="groq/llama3-8b-8192", temperature=0.2)
    print("LitellmModel initialized successfully!")

    # Optional: Test a simple call
    print("Testing a call...")
    response = llm(prompt="Why is the grass green ?", max_tokens=50)
    print("LLM Response:", response)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure the groq is accessible and API key loaded.")