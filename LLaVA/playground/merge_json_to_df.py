import pandas as pd
import json
import os
import re
from playground.constants import questions_person_bbox_json
# from constants import questions
import jsonlines
import csv

additional_keys = questions_person_bbox_json.keys()

output_data_format = {
    "question_id": [],
    "image_name": [],
}

for key in additional_keys:
    output_data_format[key] = []

def merge_data(question_jsonl, answer_jsonls, answer_base_path):
    # Read the question JSONL file
    with open(question_jsonl, 'r') as q_file:
        question_data = [json.loads(line) for line in q_file]
    # Extract image names and question IDs from the question JSONL
    question_id_to_image = {item["question_id"]: item["image"] for item in question_data}
    # Create an empty DataFrame with image names as rows
    df = pd.DataFrame(output_data_format)
    # Fill question id and image name from the question JSONL
    df["question_id"] = question_id_to_image.keys()
    df["image_name"] = question_id_to_image.values()
    # Iterate through answer JSONL files
    for ans_jsonl in answer_jsonls:
        answer_jsonl = os.path.join(answer_base_path, ans_jsonl)
        # Read the answer JSONL file line by line
        with open(answer_jsonl, 'r') as a_file:
            answer_data = [json.loads(line) for line in a_file]
        ans_id_to_image = {item["question_id"]: item["text"].split(",")[0] for item in answer_data}
        # Get the name of the answer file without extension
        ans_name = ans_jsonl.split(".")[0]
        column_to_fill = re.sub(r'^QuestionList_', '', ans_name)
        # number_part = ans_name.split("_")[-1]
        # column_to_fill = "a_" + number_part
        df[column_to_fill] = ans_id_to_image.values()
    # Fill missing values with NaN
    df = df.fillna("NaN")

    return df

# Example usage:
# question_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_red_cropped/QuestionList_dark_hair.jsonl"
# answer_base_path = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_answers_red_cropped"
# answer_files = os.listdir(answer_base_path)
# merged_data = merge_data(question_file, answer_files, answer_base_path)
# # Save the merged data to a CSV file
# merged_data.to_csv("red_cropped.csv", index=False)

# # Example usage:
# question_file = "/home/ubuntu/projects/LLaVA/playground/jsonl_questions_red_blurred_bbox/QuestionList_dark_hair.jsonl"
# answer_base_path = "/home/ubuntu/projects/LLaVA/playground/jsonl_answers_red_blurred_bbox"
# answer_files = os.listdir(answer_base_path)
# merged_data = merge_data(question_file, answer_files, answer_base_path)
# # Save the merged data to a CSV file
# merged_data.to_csv("blurred_bbox.csv", index=False)

# # Example usage:
# question_file = "/home/ubuntu/projects/LLaVA/playground/jsonl_questions_red_blurred/QuestionList_dark_hair.jsonl"
# answer_base_path = "/home/ubuntu/projects/LLaVA/playground/jsonl_answers_red_blurred"
# answer_files = os.listdir(answer_base_path)
# merged_data = merge_data(question_file, answer_files, answer_base_path)
# # Save the merged data to a CSV file
# merged_data.to_csv("red_blured.csv", index=False)

# # Example usage:
# question_file = "/home/ubuntu/projects/LLaVA/playground/jsonl_questions_red_bbox/QuestionList_dark_hair.jsonl"
# answer_base_path = "/home/ubuntu/projects/LLaVA/playground/jsonl_answers_red_bbox"
# answer_files = os.listdir(answer_base_path)
# merged_data = merge_data(question_file, answer_files, answer_base_path)
# # Save the merged data to a CSV file
# merged_data.to_csv("red_bbox.csv", index=False)






def merge_jsonl_files():
    # Input and output file paths
    question_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_red_bbox_cropped/QuestionList_q1.jsonl"
    jsonl_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_answers_red_bbox_cropped_13b/QuestionList_q1.jsonl"
    csv_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_answers_red_cropped_13b.csv"

    # Read the question JSONL file
    with open(question_file, 'r') as q_file:
        question_data = [json.loads(line) for line in q_file]
    # Extract image names and question IDs from the question JSONL
    question_id_to_image = {item["question_id"]: item["image"] for item in question_data}
    # Create an empty DataFrame with image names as rows
    df = pd.DataFrame(output_data_format)
    # Fill question id and image name from the question JSONL
    df["question_id"] = question_id_to_image.keys()
    df["image_name"] = question_id_to_image.values()

    # Open the JSONL file for reading and CSV file for writing
    with jsonlines.open(jsonl_file, mode='r') as reader, open(csv_file, mode='w', newline='') as writer:
        csv_writer = csv.writer(writer)

        # Initialize header as an empty list
        header = []

        # Read each JSONL entry and write to the CSV
        for ii, entry in enumerate(reader):
            text = entry.get("text", "{}")
            try:
                text_data = json.loads(text)
                # If the header is not yet determined, extract keys from the first entry
                if not header:
                    header_orig = list(text_data.keys())
                    header = ["question_id", "image_name"] + header_orig
                    csv_writer.writerow(header)

                # Create a data row based on the header
                data_row = [str(text_data.get(key, "N/A")) for key in header_orig]
                img_data = [df["question_id"][ii], df["image_name"][ii]]
                data_row = img_data + data_row
                csv_writer.writerow(data_row)
            except json.JSONDecodeError:
                print(f"Skipping entry due to JSON decoding error: {entry}")



    print(f"Data has been extracted from {jsonl_file} and saved to {csv_file}.")

def merge_description_json_file(question_file, jsonl_file, csv_file):
    # Input and output file paths


    # Read the question JSONL file
    with open(question_file, 'r') as q_file:
        question_data = [json.loads(line) for line in q_file]
    # Extract image names and question IDs from the question JSONL
    question_id_to_image = {item["question_id"]: item["image"] for item in question_data}
    # Create an empty DataFrame with image names as rows
    df = pd.DataFrame(output_data_format)
    # Fill question id and image name from the question JSONL
    df["question_id"] = question_id_to_image.keys()
    df["image_name"] = question_id_to_image.values()

    # Open the JSONL file for reading and CSV file for writing
    with jsonlines.open(jsonl_file, mode='r') as reader, open(csv_file, mode='w', newline='') as writer:
        csv_writer = csv.writer(writer)

        # Initialize header as an empty list
        header = []

        # Read each JSONL entry and write to the CSV
        for ii, entry in enumerate(reader):
            text = entry.get("text", "{}")
            try:
                # text_data = json.loads(text)
                # If the header is not yet determined, extract keys from the first entry
                if not header:
                    header_orig = ["description",] #list(text_data.keys())
                    header = ["question_id", "image_name",] + header_orig
                    csv_writer.writerow(header)

                # Create a data row based on the header
                # data_row = [str(text_data.get(key, "N/A")) for key in header_orig]
                img_data = [df["question_id"][ii], df["image_name"][ii]]
                data_row = img_data + [text]
                csv_writer.writerow(data_row)
            except json.JSONDecodeError:
                print(f"Skipping entry due to JSON decoding error: {entry}")

    print(f"Data has been extracted from {jsonl_file} and saved to {csv_file}.")

if __name__ == "__main__":
    # merge_jsonl_files()
    print("Merging")
    question_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_default/QuestionList_q1.jsonl"
    jsonl_file = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_answers_test/QuestionList_q1.jsonl"
    csv_file = "/home/ubuntu/Yoni/LLaVA/playground/description_13b.csv"
    merge_description_json_file(question_file, jsonl_file, csv_file)





