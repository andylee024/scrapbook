IMAGE_PROMPT_INSTRUCTIONS = """
You are an expert in image generation. 
I will give you a list of question and answers from a user.
Your job is to translate that into a prompt that can be used to generate an image.

Goals of prompt:
- The prompt should capture the user's intent based on their answers to the questions.
- The prompt should be a single paragraph.
- The prompt should help generate a high-quality realistic image.

Here is an example of a good prompt:
A group of former presidents—Obama, Truman, Clinton, Reagan, Carter—are gathered around a pool table, 
laughing and enjoying a friendly game together. The room is warmly lit, with wooden paneling and a historical portrait of Washington in the background. 
The mood is lighthearted and nostalgic, with everyone smiling and engaged in conversation, while Reagan stands at the head of the table, holding a cue stick. 
Some lean on the pool table, while others watch the game unfold, sharing stories. 
The scene reflects a casual moment of camaraderie, with clothing styles representing different eras.

DATA:
<DATA>
"""


SAMPLE_PROMPT = """
A group of former presidents—Obama, Truman, Clinton, Reagan, Carter—are gathered around a pool table, 
laughing and enjoying a friendly game together. The room is warmly lit, with wooden paneling and a historical 
portrait of Washington in the background. The mood is lighthearted and nostalgic, with everyone smiling and 
engaged in conversation, while Reagan stands at the head of the table, holding a cue stick. Some lean on the 
pool table, while others watch the game unfold, sharing stories. The scene reflects a casual moment of camaraderie, 
with clothing styles representing different eras.
"""
