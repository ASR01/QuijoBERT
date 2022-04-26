# QuijoBERT

Using El Quijote in Spanish language a fill masked model.

## First Steps

As usual please use the reuirements.txt to make your virtual envideonment before training the model.

## Training

With the file quijoBERT.py you can train a transformer model and be able to infere the masked word of a passage you want.

The tokenizer used has been both bansd of El Quijote because the Spanisch of the XVII Centiry has words different as the one of nowadays. THis will also make thea for example the size of the roberta model tokenizer (with a tokenizer of 52000 words) is not used completely. But for compatibility issues i lef it like it is.

It is very recomendable to use a GPU, using a Tesla P-100 GPU it took nearly 24 hours to train the model for 100 Epochs.

Remember that if the model fails you car recover the info loading the intermediate saves in the checkpoint-xxxx folder generated from the trainer.

## Gradio

There is the possibility to run inference using gradio.

So when the model is ready run app.py and you can infere from your local machine. 


Enjoy (the book and) the model

## PS

Check out the model at work at https://huggingface.co/spaces/andersab/QuijoBERT. You can download the model from there if you do not want to train it.
