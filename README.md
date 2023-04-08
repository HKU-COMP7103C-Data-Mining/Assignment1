# Assignment1

#### 1. Download IMDB dataset

The **download.sh** file will download the IMDB dataset directly from our Google Drive. The commands are below:

```bash
cd rnn_lstm
chmod 777 download.sh
./download.sh
```

#### 2. Run the training and evaluation process of RNN or LSTM

```bash
python main.py --model_name RNN(LSTM) --learning_rate 1e-5 --batch_size 32 --num_layers 1 --embedding_dim 512 --epochs 20 --hidden_dim 64 --bidirectional True
```

#### 3. 