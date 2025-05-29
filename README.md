# ChatGPT API Implementation

This is a simple implementation of the ChatGPT API that can be run in Google Colab.

## Setup Instructions

1. Clone this repository to your local machine or Google Colab:
```bash
git clone https://github.com/officialhariprasath/youtube_automation2.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the script:
```bash
python chatgpt_api.py
```

## Google Colab Usage

1. Open Google Colab and create a new notebook
2. Clone the repository:
```python
!git clone https://github.com/officialhariprasath/youtube_automation2.git
%cd youtube_automation2
```

3. Install dependencies:
```python
!pip install -r requirements.txt
```

4. Set up your API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

5. Import and use the ChatGPT function:
```python
from chatgpt_api import get_chatgpt_response

# Example usage
response = get_chatgpt_response("What is artificial intelligence?")
print(response)
```

## Note
Make sure to keep your API key secure and never share it publicly. 