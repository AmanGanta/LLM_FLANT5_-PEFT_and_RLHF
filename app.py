from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

app = Flask(__name__)
mdl_nm = 'google/flan-t5-base'
peft_mdl_pth = "./final_ppo_model"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tk = AutoTokenizer.from_pretrained(peft_mdl_pth)
base_mdl = AutoModelForSeq2SeqLM.from_pretrained(mdl_nm, torch_dtype=torch.bfloat16)
mdl = PeftModel.from_pretrained(base_mdl, peft_mdl_pth)
mdl.to(dev)
@app.route('/')
def idx():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def smrz():
    if request.method == 'POST':
        ip_txt = request.form['input_text']
        ip = tk(f"Summarize the conversation\n{ip_txt}", return_tensors="pt").to(dev)
        smry_ids = mdl.model.generate(ip['input_ids'], max_length=300)
        smry = tk.decode(smry_ids[0], skip_special_tokens=True)
        return render_template('result.html', summary=smry)

if __name__ == '__main__':
    app.run(debug=True)
