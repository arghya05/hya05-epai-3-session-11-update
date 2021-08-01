
### Submission by Sachin Dangayach

# Objective

- Follow the similar strategy as we did in our baby-steps-code (Links to an external site.), but replace GRU with LSTM. In your code you must:
- Perform 1 full feed forward step for the encoder manually
- Perform 1 full feed forward step for the decoder manually.
- You can use any of the 3 attention mechanisms that we discussed.

# Solution -

[Colab Link](https://colab.research.google.com/drive/16rrx_ow0wbFDR2G1lyAlPkk7_S7_aoyE?usp=sharing)

## Data Preparation

    - Download the data

    - Create the lang class with methods to maintain word2index, word2count, index2word dictionaries and total number of words

    - Create methods to normalize and read dataset while provide list of sentences in languages as well as language pair as output

    - Prepare dataset of the French and the English translations of French sentences along with list of tuples as pairs while applying filter of max length 10 for sentences

## Feed forward steps for encoder

    - To feed the sentences to LSTM, we need to have the convert the input sentences to Embeddings and those Embeddings to tensors

      - Take the input language sentences, and split it into a list of words/tokens

      - Find the index of each words in the list to create a list of index, append EOS index and convert it into a tensor to get the input and output tensors

      - Define Embedding layer and LSTM layer for encoder

        embedding = nn.Embedding(input_size, hidden_size).to(device)

        lstm = nn.LSTM(hidden_size, hidden_size).to(device)

      -  Build LSTM, initialize the hidden state and cell state with Zeros(Empty state)

        (hidden,ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)

        embedded_input = embedding(input_tensor[0].view(-1, 1))

        output, (hidden,ct) = lstm(embedded_input, (hidden,ct))

      - Define a empty tensor with size MAX_LENGTH to store the Encoder outputs. Then we can get the encoder outputs for each of the word in the Sentence

        encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)

        (encoder_hidden,encoder_ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)

        for i in range(input_tensor.size()[0]):  

          embedded_input = embedding(input_tensor[i].view(-1, 1))

          output, (encoder_hidden,encoder_ct) = lstm(embedded_input, (encoder_hidden,encoder_ct))

          encoder_outputs[i] += output[0,0]

  ***output***
  ![Encoder output](https://github.com/SachinDangayach/END2.0/blob/main/Session11/images/i_1.PNG)


## Feed forward steps for decoder

- First input to the decoder will be SOS_token, later inputs would be the words it predicted (unless we implement teacher forcing).

- Decoder/LSTM's hidden state will be initialized with the encoder's last hidden state. We will use LSTM's hidden state and last prediction to generate attention weight using a FC layer.

- This attention weight will be used to weigh the encoder_outputs using batch matric multiplication. This will give us a NEW view on how to look at encoder_states. this attention applied encoder_states will then be concatenated with the input, and then sent a linear layer and then sent to the LSTM. LSTM's output will be sent to a FC layer to predict one of the output_language words

  - Define the first input for Decoder as -

    - decoder_input = tensor of index for SOS token

    - decoder_hidden = encoder_hidden and decoder_ct = encoder_ct

  - Create decoder embedding layer ( input size = number of words in output language, output dimension = 256)

      output, (decoder_hidden,decoder_ct) = lstm(input_to_lstm.unsqueeze(0), (decoder_hidden,decoder_ct))

      output.shape, decoder_hidden.shape


## Attention Mechanism

- We define attention layer by concatenating the embeddings and last decoder hidden state and giving as input to the fully connected layer

    attn_weight_layer = nn.Linear(256 * 2, 10).to(device)

    attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))

    attn_weights = F.softmax(attn_weights, dim = 1)


## Output

![Output](https://github.com/SachinDangayach/END2.0/blob/main/Session11/images/i_2.PNG)
