import os
import subprocess
from playground.constants import return_questions_dict
from playground.generate_json_questions import generate_json_q_files, question_prefix
from playground.yolo_utils import process_and_save_images
from playground.merge_json_to_df import merge_description_json_file
import argparse

def parse_arguments():
    # image_directory = "/home/ubuntu/Yoni/LLaVA/playground/red_bbox_cropped_8x"
    save_path = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_default"
    images_base_dir = "/home/ubuntu/red/imgs"
    processed_images_save_path = "/home/ubuntu/Yoni/LLaVA/playground/red_images_marked_01"
    save_json_path = "/home/ubuntu/Yoni/LLaVA/playground/jsonl_red_images_marked_01"
    
    parser = argparse.ArgumentParser(description="Generate JSON question files")
    parser.add_argument("--questions", type=str, default='person_json', required=False, help="Questions to be asked")

    parser.add_argument("--save_json_path", type=str, default=save_json_path, required=False, help="Path to save the JSON files")
    parser.add_argument("--images_input_dir", type=str, default=images_base_dir, required=False, help="base dir for input images")
    parser.add_argument("--processed_images_save_path", type=str, default=processed_images_save_path, required=False, help="processed_images_save_path")
    
    parser.add_argument("--save_cropped_images", type=bool, default=False, required=False, help="True for cropping image")
    parser.add_argument("--blur_images", type=bool, default=False, required=False, help="True for cropping image outside of detection")
    parser.add_argument("--mark_bbox", type=bool, default=True, required=False, help="add red bbox over the image")
    return parser.parse_args()


def main():#image_directory, questions, question_prefix, save_path):
    # generate blured images
    args = parse_arguments()    
    
    images_base_dir = args.images_input_dir #"/home/ubuntu/red/imgs"
    save_images_path = args.processed_images_save_path #"/home/ubuntu/Yoni/LLaVA/playground/red_images_marked_01"
    
    save_cropped = args.save_cropped_images
    blur_images = args.blur_images
    mark_bbox = args.mark_bbox
    
    process_and_save_images(images_base_dir, save_images_path, save_cropped, blur_images, mark_bbox)
    
    save_path= args.save_json_path 
        # generate json questions
        
    questions_type = return_questions_dict(args.questions)
    
    jsonl_questions = os.path.join(save_path, "Questions")
    
    jsonl_answers = os.path.join(save_path, "Answers")

    if not os.path.exists(jsonl_questions):
        os.makedirs(jsonl_questions)
    
    if not os.path.exists(jsonl_answers):
        os.makedirs(jsonl_answers)
        
    generate_json_q_files(save_images_path, questions_type, question_prefix,
                          save_path=jsonl_questions)
    
    question_file = os.path.join(jsonl_questions, "QuestionList_q1.jsonl")
    jsonl_answer_file = os.path.join(jsonl_answers, "QuestionList_q1.jsonl")
    
        
    csv_output_file = os.path.join(save_path, "merged_output.csv") 


    
    command = f'python -m llava.eval.model_vqa --model-path liuhaotian/llava-v1.5-7b --question-file "{question_file}" --image-folder "{save_images_path}" --answers-file "{jsonl_answer_file}"'
    print(command)
    

    subprocess.run(command, shell=True)

    merge_description_json_file(question_file, jsonl_answer_file, csv_output_file)        
    
    

main()



