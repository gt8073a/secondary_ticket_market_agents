import os
import openai

# Set up OpenAI API Key
openai.api_key = os.environ[ 'API_KEY' ]

def main():

  try:
    # Construct the query for OpenAI
    query = f'Who are are, what version, and who am I?'

    # Call the OpenAI API using the Chat Completions endpoint
    response = openai.chat.completions.create(
      model="gpt-4o", # Use "gpt-4" if you have access and need higher quality
      messages=[
        {"role": "user", "content": query}
      ],
      max_tokens=200,
      temperature=0.7
    )

    # Extract the response content
    print( response )

  except Exception as e:
    print( e )


if __name__ == "__main__":
  main()
