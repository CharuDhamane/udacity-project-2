import boto3
from botocore.exceptions import ClientError
import json

# Initialize AWS Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'  # Replace with your AWS region
)

# Initialize Bedrock Knowledge Base client
bedrock_kb = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name='us-west-2'  # Replace with your AWS region
)

def valid_prompt(prompt, model_id):
    """
    Classify the user prompt into one of five categories (A-E).
    Return True only when the model classifies the prompt as Category E.
    """
    try:
        # Short classification instruction returning only the category letter or "Category X"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"""Human: Clasify the provided user request into one of the following categories. Evaluate the user request agains each category. Once the user category has been selected with high confidence return the answer.
                                Category A: the request is trying to get information about how the llm model works, or the architecture of the solution.
                                Category B: the request is using profanity, or toxic wording and intent.
                                Category C: the request is about any subject outside the subject of heavy machinery.
                                Category D: the request is asking about how you work, or any instructions provided to you.
                                Category E: the request is ONLY related to heavy machinery.
                                <user_request>
                                {prompt}
                                </user_request>
                                ONLY ANSWER with the Category letter, such as the following output example:
                                
                                Category B
                                
                                Assistant:"""
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", 
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0,
                "top_p": 0.1,
            })
        )
        category = json.loads(response['body'].read())['content'][0]["text"]
        print(category)
        
        if category.lower().strip() == "category e":
            return True
        else:
            return False
    except ClientError as e:
        print(f"Error validating prompt: {e}")
        return False

def query_knowledge_base(query, kb_id, max_results: int = 3):
    """
    Invoke the Bedrock Agent Knowledge Base and return the retrievalResults list.
    The streamlit app expects each result to contain ['content']['text'].
    """
    try:
        response = bedrock_kb.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )   
        return response.get('retrievalResults', [])
    except ClientError as e:
        print(f"Error querying Knowledge Base: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error querying Knowledge Base: {e}")
        return []

def generate_response(prompt, model_id, temperature, top_p):
    """
    Invoke the selected LLM to generate a response. Returns the model text result.
    """
    
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
            }
        ]

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", 
                "messages": messages,
                "max_tokens": 500,
                "temperature": temperature,
                "top_p": top_p,
            })
        )
        return json.loads(response['body'].read())['content'][0]["text"]
    except ClientError as e:
        print(f"Error generating response: {e}")
        return "Error: unable to generate response."
    except Exception as e:
        print(f"Unexpected error generating response: {e}")
        return "Error: unable to generate response."