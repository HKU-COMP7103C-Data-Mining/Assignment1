# Assignment1

#### 1. Install possible python requirements

The possible requirements are in [requirements.txt](./requirements.txt). To install, run the command below:

```bash
pip install -r requirements.txt
```

#### 2. Download IMDB dataset

The **download.sh** file will download the IMDB dataset directly from our Google Drive. The commands are below:

```bash
chmod 777 download.sh
./download.sh
```

#### 3. Traditional machine learning models training and evaluation

We first use some traditional machine learning models such as naive-bayes, knn, decision tree and linear SVC to do this task. The running details and logs are in the notebook [Bayes.ipynb](./Bayes.ipynb).

#### 4. RNN or LSTM training and evaluation

To run RNN and LSTM, please use following command:

```bash
python main.py --model_name RNN(LSTM) --learning_rate 1e-5 --batch_size 32 --num_layers 1 --embedding_dim 512 --epochs 20 --hidden_dim 64 --bidirectional True
```

#### 5. Bert training and evaluation

We also use a more advanced model BERT. To see the codes and running logs, please see [bert.ipynb](./bert.ipynb).



