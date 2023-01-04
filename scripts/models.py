import torch

######################################
# SECTION
# LSTM MODEL
######################################

#LSTM with 
#Sequence length = 4  <= Check this part
#Input size = 768
#Batch size = __batch_size
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

class ExtractorLSTM(torch.nn.Module):
    def __init__(self, input_size=768, hidden_size=768*2, batch_size=8, batch_first=True):
        super(ExtractorLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.activation = torch.nn.Mish()
        self.linear = torch.nn.Linear(hidden_size, 2)
        self.softmax = torch.nn.LogSoftmax(dim=0)
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x):
      output = torch.Tensor([]).cuda()
      hidden = torch.zeros(1,1,self.hidden_size).cuda()
      cell = torch.zeros(1,1,self.hidden_size).cuda()
      for i in range(self.batch_size):
        curr_output, (hidden, cell) = self.lstm(x[i].unsqueeze(dim=0), (hidden, cell))
        curr_output = curr_output.squeeze()[-1]
        curr_output = self.activation(curr_output)
        curr_output = self.linear(curr_output)
        curr_output = self.softmax(curr_output)
        #curr_output.shape => [2]
        #to stack unsqueeze dim 0 => [1,2] => will accumulate into => [8,2]
        output = torch.cat((output, curr_output.unsqueeze(dim=0)), 0)

      # print(output)
      return output


######################################
# SECTION
# CNN MODEL
######################################

class ExtractorCNN(torch.nn.Module):
    def __init__(self, n_filter=100, batch_size=8):
        super(ExtractorCNN, self).__init__()
        self.cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (2, 768)),
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )
        self.cnn2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (3, 768)),
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )
        self.cnn3 = torch.nn.Sequential(
            torch.nn.Conv2d(1, n_filter, (4, 768)),
            torch.nn.LeakyReLU()
            # torch.nn.Mish()
        )

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_filter*3, 2)
        )
        self.softmax = torch.nn.LogSoftmax(dim=0)

        self.n_filter = n_filter
        self.batch_size = batch_size

    def forward(self, x):
      output = torch.Tensor([]).cuda()
      for i in range(self.batch_size):
        out1 = self.cnn1(x[i].unsqueeze(dim=0))
        out2 = self.cnn2(x[i].unsqueeze(dim=0))
        out3 = self.cnn3(x[i].unsqueeze(dim=0))

        out1_maxed = torch.max(out1, dim=2).values
        out2_maxed = torch.max(out2, dim=2).values
        out3_maxed = torch.max(out3, dim=2).values
        curr_output = torch.cat((out1_maxed, out2_maxed, out3_maxed), dim=1)
        curr_output = curr_output.squeeze()

        curr_output = self.linear(curr_output)
        curr_output = self.softmax(curr_output)

        #curr_output.shape => [2]
        #to stack unsqueeze dim 0 => [1,2] => will accumulate into => [8,2]
        output = torch.cat((output, curr_output.unsqueeze(dim=0)), 0)

      # print(output)
      return output  