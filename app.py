import gradio as gr

from transformers import pipeline

fill_mask = pipeline("fill-mask", model="./QuijoBERT", tokenizer = './QuijoBERT')

def predict(text):
    
    res_dict = {}
    x = fill_mask(text)
    print('x')
    for i in range(len(x)):
        k = x[i]['sequence']
        e = x[i]['score']
        print(k, e) 
        if e >= 0.05:
            res_dict[k] = e
    print (res_dict)
    return res_dict
    #return {x[0]["sequence"], x[0]["score"]}

# texto = 'en un lugar de la <mask>'
# print(predict(texto))

iface = gr.Interface(
    fn=predict, 
    inputs='text',
    outputs ='label',
    examples=['En un lugar de la <mask>', 'En verdad, <mask> Sancho', 'Cómo has estado, bien mío, <mask> de mis ojos, compañero mío']
)


iface.launch()