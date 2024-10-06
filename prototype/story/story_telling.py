
import json

import story.prompts as prompts
# from utils.llm_utils import run_llm


STORY_PROMPT_TEMPLATE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "core_theme": {"type": "string"},
        "relationship_theme": {"type": "string"},
        "individuality_theme": {"type": "string"},
        "activity_theme": {"type": "string"},
        "setting_theme": {"type": "string"},
        "emotional_tone_theme": {"type": "string"},
        "accents_theme": {"type": "string"},
        "interesting_questions": {"type": "string"}
    },
    "required": [
        "core_theme",
        "relationship_theme",
        "individuality_theme",
        "activity_theme",
        "setting_theme",
        "emotional_tone_theme",
        "accents_theme",
        "interesting_questions"
    ]
}


PROMPT_QUESTIONS = {
    "core_theme": "What is the central theme we are trying to capture with the photo?",
    "relationship_theme": "List the people that you want to feature in this photo? What is the relationship to each other?",
    "individuality_theme": "What is the personality of each person like (e.g. share a story of each person that showcases their personality)?",
    "activity_theme": "What is are 1-3 activities that you could see everyone doing together on a Sunday afternoon?",
    "setting_theme": "What is the setting from which you all know each other? (family, home, school, work, sports)",
    "emotional_tone_theme": "What is the mood that you want to capture with the photo? (e.g. warm, cozy, playful, serious, etc.)",
    "accents_theme": "What specific details should be included to make the image feel personal or relevant? (e.g., an old family game, a shared tool, or something passed down through generations)",
    "interesting_questions": "Is there an interesting story you have of each person? (e.g. a memorable conversation, a shared experience, a significant moment)?"
}


def get_user_questions_for_story():
    return PROMPT_QUESTIONS


def generate_image_prompt_from_user_answers(json_data):

    if not json.validate(json_data, STORY_PROMPT_TEMPLATE_JSON_SCHEMA):
        raise json.ValidationError("Invalid JSON data")
    
    # prompt = prompts.IMAGE_PROMPT_INSTRUCTIONS.replace("<DATA>", str(json_data))
    # image_prompt = run_llm(prompt, "open_ai")
    prompt = prompts.SAMPLE_PROMPT
    return prompt
