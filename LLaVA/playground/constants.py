# questions = {
#     "male": "Is the person in the image male?",
#     # "female": "Is the person in the image female?",
#     "dark_hair": "Does the person in the image have dark hair?",
#     # "light hair": "Does the person in the image have light hair?",
#     # "long_hair": "Does the person in the image have long hair?",
#     "short_hair": "Does the person in the image have short hair?",
#     # "young": "Is the person in the image young?",
#     # "middle_aged": "Is the person in the image middle-aged?",
#     # "elderly": "Is the person in the image elderly?",
#     # "glasses": "Is the person in the image wearing glasses?",
#     "tattoo": "does the person in the image have a tattoo on their skin?",
#     "piercing": "Does the person in the image have piercing?",
#     # "watch": "Is the person in the image wearing a watch?",
#     # "earrings":"Is the person in the image wearing earrings?",
#     "necklace":"Is the person in the image wearing a necklace?",
#     # "with_others": "Is the person in the image with others?",
#     # "alone": "Is the person in the image alone?",
#     # "formal_outfit": "Is the person in the image in a formal outfit?",
#     # "casual_outfit": "Is the person in the image in a casual outfit?",
#     # "sportswear": "Is the person in the image in sportswear?",
#     # "thin": "Is the person in the image thin?",
#     # "fat": "Is the person in the image fat?",
#     "scars": "Does the person in the image have scars?",
#     "body_marks": "Does the person in the image have any body marks over their skin?",
#     "soldier": "Does the person in the image is a soldier?"
# }

# questions = {
#     "male": "Within the center of the red bounding box, Is the person is a male?",
#     "backpack": "Within the center of the red bounding box, Is the person wearing or holding a backpack?",
#     "gun": "Within the center of the red bounding box, Is the person has a gun or a rifle or any weapon?",
#     "watch": "Within the center of the red bounding box, Is the person wearing a watch?",
#     "soldier": "Within the center of the red bounding box, Does the person is a soldier?"
# }

# question_prefix = "Answer the question using a single word or phrase. "# Indicate for the person within the red bounding box, "


questions_person_bbox_json = {'q1': 'For the object within the center of the red bounding box. '
                   'Please return the information in the following JSON format '
                   '(note xxx is placeholder but can also get a list of items, '
                   'if the information is not available in the image, put "N/A" instead). '
                   'bbb is a placeholder for a binary answer (yes/no). '
                   '{"gender": xxx, '
                   '"cloths description": xxx, '
                   '"cloths color": xxx, '
                   '"is a soldier": bbb, '
                   '"wearing accessories": xxx, '
                   '"wearing a watch": bbb, '
                   '"has a backpack": bbb, '
                   '"has a large backpack": bbb, '
                   '"hair color": xxx, '
                   '"have tattoos": bbb, '
                   '"tattoos locations over the body": xxx, '
                   '"body type": xxx, '
                   '"other body marks": xxx, '
                   '"age": xxx, '
                   '"happy or sad": xxx,'
                   '"holding a weapon":xxx,'
                   '"other details": xxx}'}


question_prefix = ""

questions_image_description = {'q1': 'For the following image, please describe the image with details.'
                                     'focus specific on the people in the image. are the people holding weapons?'
                                     'are the people wearing any accessories? '
                                     'What is the atmosphere of the image? are people look happy or sad? '
                                     'what type of cars and car colors are in the image?'
                                     'does the cars have yellow or white license plates?'}

def return_questions_dict(questions_type):
    if questions_type=='description':
        return questions_image_description
    if questions_type=='person_json':
        return questions_person_bbox_json