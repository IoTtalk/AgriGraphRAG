import os
import json
import requests
import config

def query_ollama(prompt, model='deepseek-r1:70b'):
    url = 'http://140.113.164.115:32179/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,  # set to True if you want to handle streamed responses
        "keep_alive": -1
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data.get('response')
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None

# Example usage
if __name__ == '__main__':
    temp_dir = config.temp_directory
    entity_file = config.entity_file
    
    with open(temp_dir + entity_file, "r") as input_file:
        text_and_entities = json.load(input_file)
        
    claim_results = []
    
    claim_file = config.claim_file
    
    for item in text_and_entities:
        entities = item["entity_result"].split("```")[1].split("json ")[1]
        
        prompt_text = "A claim is a specific, factual, or inferential statement made about an entity, usually describing: what it does, what role it plays, how it interacts with other entities, or what properties or effects it has.\n"
        prompt_text += "I'm going to give you a text paragraph and a list including all the entities of this paragraph with there type.  Please find all the claims against those given entities and return all the entities and their claims in the json format.\n\n"
        prompt_text += f"The entities are {entities}.\n\n"
        prompt_text += "And the following is the text paragraph:\n"
        prompt_text += item["text_paragraph"]
        result = query_ollama(prompt_text)
        
        claim_results.append({
            "qid": item["qid"],
            "text_paragraph": item["text_paragraph"],
            "entity_result": entities,
            "claim_result": result
        })
        
    with open(temp_dir + claim_file, "w") as output_file:
        json.dump(claim_results, output_file, indent=4)
        output_file.close()
