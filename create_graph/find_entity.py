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
    enterprise_file = config.file_name
    
    texts = []
    
    with open(enterprise_file, "r") as input_file:
        data = json.load(input_file)
        for item in data:
            texts.append(item["content"])
        
    entity_results = []
    
    temp_dir = config.temp_directory
    entity_file = config.entity_file
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    
    for i, content in enumerate(texts):
        prompt_text = "An entity is any named, described, or significant element in the text that can be classified under a specified type and is relevant to the overarching activity.\n"
        prompt_text += "I'm going to give you a list including entity type and a text paragraph.  Please find all the entities and return them with their entity type in the json format.\n\n"
        prompt_text += "The entity types are [Pathogen, Host, Symptom, Environmental Factor, Control Method, Genetic Component, Location, Researcher, Diagnostic Method, Economic Impact, Agricultural Practice, Temporal Factor].\n\n"
        prompt_text += "And the following is the text paragraph:\n"
        prompt_text += content
        result = query_ollama(prompt_text)
        entity_results.append({
            "qid": i,
            "text_paragraph": content,
            "entity_result": result
        })
        
    with open(temp_dir + entity_file, "w") as output_file:
        json.dump(entity_results, output_file, indent=4)
        output_file.close()
