import os
import json
import requests

import sys
sys.path.append("..")
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
        
    relationship_results = []
    
    relationship_file = config.relationship_file
    
    except_ids = []
    
    for item in text_and_entities:
        entities = item["entity_result"]
        
        if "```" not in entities:
            except_ids.append(item["qid"])
            print(item["qid"])
            continue
        
        entities = entities.split("```")[1]
        
        if "json" not in entities:
            except_ids.append(item["qid"])
            continue
        
        entities = entities.split("json")[1]
        
        prompt_text = "A relationship is a meaningful connection or interaction between two distinct entities, as stated or implied in the text, which contributes to understanding their roles, dependencies, or effects in the context of the specified activity (e.g., understanding a disease system).\n"
        prompt_text += "I'm going to give you a text paragraph and a list including all the entities of this paragraph with there type.  Please find all the relationships against those given entities and return all the entity pairs and their relationships in the json format.\n\n"
        prompt_text += f"The entities are {entities}.\n\n"
        prompt_text += "And the following is the text paragraph:\n"
        prompt_text += item["text_paragraph"]
        result = query_ollama(prompt_text)
        
        relationship_results.append({
            "qid": item["qid"],
            "text_paragraph": item["text_paragraph"],
            "entity_result": entities,
            "relationship_result": result
        })
        
    with open(temp_dir + relationship_file, "w") as output_file:
        json.dump(relationship_results, output_file, indent=4)
        output_file.close()
        
    print(len(except_ids))
    print(except_ids)
