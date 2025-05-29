import os
import openai

# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chatgpt_response(prompt):
    """
    Get a response from ChatGPT API
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example prompt
    prompt = "What is artificial intelligence?"
    response = get_chatgpt_response(prompt)
    print("Prompt:", prompt)
    print("Response:", response) 