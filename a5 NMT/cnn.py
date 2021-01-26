#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self , char_embed_size,  word_embed_size  , kernel_size = 5):
        '''
        @param max_word_length (int): The maximun length of a word
        $param n_chanell (int): Number of channels of the output of the conv1d layer equals to
                                wordEmbedd size e.
        @param kernal_size (int): kernal size of conv layer
        @param padding (int): Padding size of the conv1d layer
        '''
        super(CNN,self).__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(self.char_embed_size , self.word_embed_size , self.kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d( 1 )
        self.relu = nn.ReLU()

    def forward(self, x_reshabed : torch.Tensor) -> torch.Tensor :
        '''
        @param x_reshabed (torch.Tensor) : output of the embedding layer ( s * b , e , m ) where
        s: maxmun length of a sentence , b : batch size , e : word embedding size , m : maxmum wordlength
        ( s * b , e_char , m) --> ( s * b, e_word )
        '''
        
        
        x_conv = self.conv( x_reshabed ) # ( s * b , e_char , m) --> (s * b , f = e_word , m - k + 1)

        x_convout = self.maxpool(self.relu( x_conv )) #(s * b , f = e_word , m - k + 1) --> ( s * b, e_word , 1)
        
        return torch.squeeze(x_convout,-1) #( s * b, e_word)


    ### END YOUR CODE

