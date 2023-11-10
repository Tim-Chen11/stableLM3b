from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    num_words = int(request.form.get('num_words', 20))  # Default to 50 words if not specified

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", trust_remote_code=True, torch_dtype="auto")
    model.cuda()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(**inputs, max_new_tokens=num_words, temperature=0.3, top_p=0.95, do_sample=True)
    generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)

    return render_template('index.html', prompt=prompt, generated_text=generated_text, num_words=num_words)

if __name__ == '__main__':
    app.run(debug=True)
