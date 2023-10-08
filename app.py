import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df['Place'] = df['Place'].astype(str)
df['Suffix'] = df['Suffix'].astype(str)
# print(len(df))
df = df[df['Suffix'] != 'nan']
# print(len(df))
X = [word.lower() for word in list(df['Place'])]
Y = [word.lower() for word in list(df['Suffix'])]
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=42)

class embeddnetwork():

    def __init__(self,embeddingdim, blocksize, suffixcorpus):
        self.embeddingdim = embeddingdim
        self.blocksize = blocksize
        self.suffixtoi = {s:i for i, s in enumerate(set(suffixcorpus))}
        self.itosuffix = {i:s for s, i in self.suffixtoi.items()}
        self.characters = """abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890`~!@#$%^&*()_-+={[}]|\:;"'<,>.?/ """
        self.C = torch.randn((len(self.characters), self.embeddingdim))
        self.W1 = torch.randn((self.blocksize*self.embeddingdim, 100))
        self.b1 = torch.rand(100)
        self.W2 = torch.randn((100, len(self.suffixtoi)))
        self.b2 = torch.randn(len(self.suffixtoi))
        self.stoi = {s:i for i, s in enumerate(self.characters)}
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
    
    def countrytoindx(self,country):
        context = []
        for char in country[-self.blocksize:]:
            ix = self.stoi[char]
            context.append(ix)
        if len(context) != self.blocksize:
            context = [self.stoi['_']]*(self.blocksize - len(context)) + context
        return context

    def fit(self,X, Y, lr, epochs):
        Xnew = []
        for country in X:
            context = self.countrytoindx(country)
            Xnew.append(context)
        Xnew = torch.tensor(Xnew)

        if self.suffixtoi == None:
            self.suffixtoi = {s:i for i, s in enumerate(set(Y))}
        else:
            for y in Y:
                if y not in self.suffixtoi:
                    self.suffixtoi[y] = len(self.suffixtoi)
        
        Ynew = []
        for suffix in Y:
            ix = self.suffixtoi[suffix]
            Ynew.append(ix)
        Ynew = torch.tensor(Ynew)

        for i in range(epochs):
            ix = torch.randint(0, Xnew.shape[0], (Xnew.shape[0],))
            emb = self.C[Xnew[ix]]
            h = torch.tanh(emb.view(-1, self.blocksize*self.embeddingdim)@self.W1 + self.b1)
            
            logits = h@self.W2 + self.b2

            loss = F.cross_entropy(logits, Ynew[ix])
            
            for p in self.parameters:
                p.grad = None
            loss.backward()
            for p in self.parameters:
                # print(len(p.grad))
                p.data -= lr*p.grad
            # print('loss', loss.item())

            
    
    def predict(self,X):
        idxes = [self.countrytoindx(x) for x in X]
        idxes = torch.tensor(idxes)
        Ys = []
        for ix in idxes:
            emb = self.C[ix]
            h = torch.tanh(emb.view(-1, self.blocksize*self.embeddingdim)@self.W1 + self.b1)
            logits = h@self.W2 + self.b2
            counts = logits.exp()
            P = counts/counts.sum(1, keepdims=True)
            suffixix = torch.argmax(P).item()
            Ys.append(self.itosuffix[suffixix])
        return Ys

import random

# Define the range of hyperparameters to search over
@st.cache(allow_output_mutation=True)
def tune_hyperparameters(X_train, Y_train):
    epochs = np.arange(50,1500,100)
    embeddingdim = np.arange(1,100,5)
    blocksize = np.arange(1,8,1)
    param_grid = {
        'lr': [0.001, 0.01, 0.1],
        'blocksize': blocksize,
        'embeddingdim': embeddingdim,
        'epochs': epochs
    }

    num_configs = 20
    x,y = [],[]
    best_accuracy = 0
    best_params = None

    for _ in range(num_configs):
        params = {
            'lr': random.choice(param_grid['lr']),
            'blocksize': random.choice(param_grid['blocksize']),
            'embeddingdim': random.choice(param_grid['embeddingdim']),
            'epochs': random.choice(param_grid['epochs'])
        }

        network = embeddnetwork(params['embeddingdim'], params['blocksize'], Y_train)
        network.fit(X_train, Y_train, params['lr'], params['epochs'])

        predictions = network.predict(X_test)
        accuracy = sum(1 for true, pred in zip(Y_test, predictions) if true == pred) / len(Y_test)
        x.append(epochs)
        y.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

    return best_params, best_accuracy, x,y

# Create a Streamlit app
st.title("Demonyms Predictor")

# Dropdown to select a place name
# selected_place = st.selectbox("Select a place name:", df['Place'])

selected_place = st.text_input('label')

# Button to trigger prediction
if st.button("Predict"):
    # Train a model with the best hyperparameters (you can use your tuning function here)
    best_params, best_accuracy, x,y = tune_hyperparameters(X_train, Y_train)
    # st.write("Best Hyperparameters:", best_params)
    # st.write("Best Accuracy:", best_accuracy)

    # Create an instance of the embeddnetwork with the best hyperparameters
    network = embeddnetwork(
        embeddingdim=best_params['embeddingdim'],
        blocksize=best_params['blocksize'],
        suffixcorpus=Y_train
    )

    # Fit the network with the training data
    network.fit(X_train, Y_train, lr=best_params['lr'], epochs=best_params['epochs'])

    # Predict the demonym for the selected place
    demonym = network.predict([selected_place])[0]

    st.write(f"The predicted demonym for '{selected_place}' is '{selected_place+demonym}'")
    # actual = list(df[df['Place']==selected_place]['Demonym'])[0]
    # st.write(f"The actual demonym for '{selected_place}' is '{actual}'")
    
    # # Create a Matplotlib figure and plot your data
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_title('Sample Plot')
    # # Display the Matplotlib plot in Streamlit
    # st.pyplot(fig)
    # # Save the Matplotlib plot to a file (e.g., as a PNG image)
    # fig.savefig('sample_plot.png')

    # Provide a link to download the saved plot
    # st.markdown('[Download the Plot](sample_plot.png)')



# embed = embeddnetwork(blocksize=2, embeddingdim=50, suffixcorpus = Y_train)
# embed.fit(X_train, Y_train, lr=0.1, epochs=500)

# embed.predict(['Pakistan'])
