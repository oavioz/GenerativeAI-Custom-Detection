#!/bin/bash

# Specify the folder containing question and answers files
# folder="/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_red_cropped"

# # Iterate over all files in the folder
# for question_file in "$folder"/QuestionList_*.jsonl; do
#     # Extract the base file name without the folder path
#     question_file=$(basename "$question_file")
#     echo $question_file
    
#     # Construct the corresponding answers file name
#     # answers_file="${question_file/question_/answers_}"

#     answers_file=$(basename "$question_file")

#     # Run your Python script for each combination
#     python -m llava.eval.model_vqa \
#         --model-path liuhaotian/llava-v1.5-13b \
#         --question-file "playground/jsonl_questions_red_cropped/$question_file" \
#         --image-folder "playground/red_cropped" \
#         --answers-file "playground/jsonl_answers_red_cropped/$answers_file"

#     # Optional: Add a separator between runs
#     echo "Finished processing $question_file and $answers_file"
# done

# folder="/home/ubuntu/projects/LLaVA/playground/jsonl_questions_red_blurred_bbox"

# # Iterate over all files in the folder
# for question_file in "$folder"/QuestionList_*.jsonl; do
#     # Extract the base file name without the folder path
#     question_file=$(basename "$question_file")
    
#     # Construct the corresponding answers file name
#     # answers_file="${question_file/question_/answers_}"

#     answers_file=$(basename "$question_file")

#     # Run your Python script for each combination
#     python -m llava.eval.model_vqa \
#         --model-path liuhaotian/llava-v1.5-7b \
#         --question-file "playground/jsonl_questions_red_blurred_bbox/$question_file" \
#         --image-folder "playground/red_blurred_bbox" \
#         --answers-file "playground/jsonl_answers_red_blurred_bbox/$answers_file"

#     # Optional: Add a separator between runs
#     echo "Finished processing $question_file and $answers_file"
# done


# folder="/home/ubuntu/projects/LLaVA/playground/jsonl_answers_red_blurred"

# # Iterate over all files in the folder
# for question_file in "$folder"/QuestionList_*.jsonl; do
#     # Extract the base file name without the folder path
#     question_file=$(basename "$question_file")
    
#     # Construct the corresponding answers file name
#     # answers_file="${question_file/question_/answers_}"

#     answers_file=$(basename "$question_file")

#     # Run your Python script for each combination
#     python -m llava.eval.model_vqa \
#         --model-path liuhaotian/llava-v1.5-7b \
#         --question-file "playground/jsonl_questions_red_blurred/$question_file" \
#         --image-folder "playground/red_blurred" \
#         --answers-file "playground/jsonl_answers_red_blurred/$answers_file"

#     # Optional: Add a separator between runs
#     echo "Finished processing $question_file and $answers_file"
# done

imgs_path="/data/imgs/"
json_questions_path="/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_default"
type_of_questions="description"

echo $imgs_path

python -m playground.generate_json_questions \
    --image-directory $imgs_path \
    --save-path $json_questions_path \
    --questions $type_of_questions

#folder="/home/ubuntu/Yoni/LLaVA/playground/jsonl_questions_default"

# Iterate over all files in the folder
for question_file in "$json_questions_path"/QuestionList_*.jsonl; do
    # Extract the base file name without the folder path
    question_file=$(basename "$question_file")
    
    # Construct the corresponding answers file name

    answers_file=$(basename "$question_file")

    # Run your Python script for each combination
    python -m llava.eval.model_vqa \
        --model-path liuhaotian/llava-v1.5-13b \
        --question-file "$json_questions_path/$question_file" \
        --image-folder "$imgs_path" \
        --answers-file "playground/jsonl_answers_test/$answers_file"

    # Optional: Add a separator between runs
    echo "Finished processing $question_file and $answers_file"
done
