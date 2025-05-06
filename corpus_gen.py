import json
import pandas as pd

# Load JSON data with specified encoding
with open('hkl_faq.json', encoding='utf-8') as f:
    data = json.load(f)

data = data['hkl']
hkl_faq = pd.DataFrame(columns=['Question', 'Answer', 'Class'])

questions = []
answers = []
classes = []

for key in data.keys():
    for qnas in data[key]:
        questions.append(qnas[0])
        answers.append(qnas[1])
        classes.append(key)

hkl_faq['Question'] = pd.Series(questions)
hkl_faq['Answer'] = pd.Series(answers)
hkl_faq['Class'] = pd.Series(classes)

hkl_faq.to_csv("hklFAQs.csv", index=False)