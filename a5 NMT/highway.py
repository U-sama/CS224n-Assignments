#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, word_embed_size , dropout_rate = 0.3 ):

        """
        @param word_embed_size (int): Dimention of the word embeding from the brevious step conv step

        @param dropout (float) : dropout probability 
        """
        super(Highway,self).__init__()

        self.word_embed_size = word_embed_size
        self.dropout_rate = dropout_rate
        self.conv_proj = None
        self.conv_gate = None
        self.dropout = None

        self.conv_proj = nn.Linear(word_embed_size , word_embed_size)
        self.conv_gate = nn.Linear(word_embed_size , word_embed_size)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, conv_out:torch.Tensor ) -> torch.Tensor:
        

        """
        @param conv_out : previous conv layer output dim (s , b , e_word)
        where s: maxmun sentence length b: batch size and e_word: word embeding size

        output x_word_embed : the output of the Highway net which is the embeding vectors of words
        dim ( s , b , e) e: word embedding size
        (s, b , e_word) --> (s, b , e_word)
        """

        x_proj = F.relu(self.conv_proj(conv_out)) # (s, b , e_word)

        x_gate = torch.sigmoid( self.conv_gate(conv_out) ) # (s , b , e_word)
        
        x_high = x_gate * x_proj +  (1  - x_gate ) * conv_out   #  (s , b , e_word)

        x_word_embed = self.dropout(x_high) #  (s , b , e_word)

        return x_word_embed 


    ### END YOUR CODE

