from flask import Flask, jsonify, Response

from story.story_telling import (
    get_user_questions_for_story, 
    generate_image_prompt_from_user_answers
)

app = Flask(__name__)

@app.route('/get_story_themes_and_questions', methods=['GET'])
def get_story_themes_and_questions():
    return jsonify(get_user_questions_for_story())


@app.route('/generate_image_prompt', methods=['POST'])
def generate_image_prompt(data):
    try:
        response = generate_image_prompt_from_user_answers(data)
        return Response(response, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)

if __name__ == '__main__':
    app.run(debug=True)
