import json
import os
# from playground.constants import questions, question_prefix, return_questions_dict
from playground.constants import question_prefix, return_questions_dict


# from constants import questions, question_prefix


def generate_json_q_files(image_directory, questions, question_prefix, save_path):
    print(image_directory)
    image_files = os.listdir(image_directory)
    image_files_wo_hebrew = []
    for file in image_files:
        if not any(ch in "אבגדהוזחטיכלמנסעפצקרשת" for ch in file):
            image_files_wo_hebrew.append(file)
    image_files = image_files_wo_hebrew

    for question, key in zip(questions.values(), questions.keys()):
        # Create a JSON file for each question
        json_file_name = f"{save_path}/QuestionList_{key}.jsonl"
        # Create a list to store data for all images
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(json_file_name, 'w') as json_file:
            for jj, image_name in enumerate(image_files):
                json_data = {
                    "question_id": jj,
                    "image": image_name,
                    "text": question_prefix + question,
                    "category": "detail"
                }
                json_file.write(json.dumps(json_data, separators=(',', ':')))
                json_file.write('\n')  # Add a newline character
        print(f"Created jsonl questions at: {save_path}")
        print("JSON files generated successfully.")

import argparse

def parse_arguments():
    image_directory = "/home/ubuntu/Yoni/LLaVA/playground/red_bbox_cropped_8x"
    save_path = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_default"
    parser = argparse.ArgumentParser(description="Generate JSON question files")
    parser.add_argument("--image-directory", type=str, default=image_directory, required=True, help="Path to the image directory")
    parser.add_argument("--save-path", type=str, default=save_path, required=True, help="Path to save the JSON files")
    parser.add_argument("--questions", type=str, default='description', required=True, help="Questions to be asked")
    return parser.parse_args()



# if __name__ == "__main__":
#     image_directory = "/home/ubuntu/Yoni/LLaVA/playground/red_bbox_cropped_8x"
#     questions_type = return_questions_dict("description")
#     generate_json_q_files(image_directory, questions_type, question_prefix, save_path="/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_red_bbox_cropped_8x")
# #
if __name__ == "__main__":
    args = parse_arguments()

    # Call the function with command-line arguments
    questions_type = return_questions_dict(args.questions)

    # print( "============== args.image_directory ==============")
    generate_json_q_files(
        image_directory=args.image_directory,
        questions=questions_type,
        question_prefix=question_prefix,
        save_path=args.save_path
    )
