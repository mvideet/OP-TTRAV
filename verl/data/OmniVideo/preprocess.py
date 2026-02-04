base_dir = "/data/sls/scratch/mvideet/Videos/"

import os
import json
import pandas as pd


#list all the folders in the base_dir and get the questions and answwers from each file to put into a json file at the end
"""
"audio_file": "path/to/audio/file.wav",
"video_file": "path/to/video/file.mp4",
"question": "question", (with the mcq options formatted)
"answer": "answer" (the mcq answer),
"source": "source" (the source of the question, from what dataset),
"id": "id"
"""
all_folders = os.listdir(base_dir)


all_data = []
for folder in all_folders:
    folder_path = os.path.join(base_dir, folder)
    all_files = os.listdir(folder_path)
    
    # Initialize variables for this folder
    qa_file = None
    audio_file = None
    video_file = None
    
    # First pass: find all files
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        if file == "QAs_revise.json":
            qa_file = file_path
        elif file.endswith(".wav"):
            audio_file = file_path
        elif file.endswith(".mp4"):
            video_file = file_path

    # Second pass: parse QAs if found
    if qa_file:
        with open(qa_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse malformed JSON format: split by ][
        parts = content.split("][")
        qa_objects = []
        
        for i, part in enumerate(parts):
            part = part.strip()
            # Remove leading [ from first part
            if i == 0 and part.startswith("["):
                part = part[1:]
            # Remove trailing ] from last part
            if i == len(parts) - 1 and part.endswith("]"):
                part = part[:-1]
            
            # Wrap in array brackets to make valid JSON
            if part:
                try:
                    # Parse as a JSON array containing one object
                    json_str = "[" + part + "]"
                    obj_array = json.loads(json_str)
                    if obj_array and len(obj_array) > 0:
                        qa_objects.append(obj_array[0])
                except json.JSONDecodeError as e:
                    print(f"Error parsing part {i} in {qa_file}: {e}")
                    print(f"Problematic part: {part[:200]}...")
                    continue
        
        # Extract question and choices for each QA object
        for qa_obj in qa_objects:
            question_text = qa_obj.get("Question", "")
            choices = qa_obj.get("Choice", [])
            answer = qa_obj.get("Answer", "")
            
            # Format question with MCQ options
            formatted_question = question_text
            if choices:
                formatted_question += "\n" + "\n".join(choices)
            
            # Create data entry
            data_entry = {
                "audio_file": audio_file if audio_file else "",
                "video_file": video_file if video_file else "",
                "question": formatted_question,
                "answer": answer,
                "source": qa_obj.get("Type", ""),
                "id": f"{qa_obj.get('video_id', folder)}-{qa_obj.get('Type', 'unknown')}-{len(all_data)}",
                "video_id": qa_obj.get("video_id", ""),
                "question_type": qa_obj.get("Type", ""),
                "content_parent_category": qa_obj.get("content_parent_category", ""),
                "content_fine_category": qa_obj.get("content_fine_category", ""),
            }
            all_data.append(data_entry)

# Save all data to JSON file
output_file = os.path.join(base_dir, "all_qa_data.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"Processed {len(all_data)} questions from {len(all_folders)} folders")
print(f"Output saved to: {output_file}")

