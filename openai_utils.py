import openai


def api_key_setup(key_path):
    openai.api_key = open(key_path, "r").read()


def get_completion(prompt):
    message = [{"role": "user", "content": prompt}]
    responses = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1024,
        temperature=1,
    )
    return responses.choices[0]["message"]["content"]


def gpt_paraphrase(example):
    prompt = f"Generate a passage that is similar to the given text in length, domain, and style.\nGiven text:{example}\nPassage :"
    return get_completion(prompt)
