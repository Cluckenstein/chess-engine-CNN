#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Thu Jun 20 13:22:13 2019

@author: Cluckenstein
"""

import chess
import chess.pgn 
import numpy as np
import sys
import plaidml.keras
plaidml.keras.install_backend() #I used this in orde to be able to exploit AMD GPUs on Mac, one may use the normal tensorflow backend
import keras
keras.backend.backend()
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from keras.models import load_model



class chess_past(object):
    """
    We intiaize the model by either using a existing one from a file or a blank one which can be trained 
    The class takes the following parameters:
    weight_decay = float 
        -Gives the weight decay used by the convolutional layers in the model
        -default is set
    old_model_path = str
        -takes a string leading to the model predicting the field of the following move
        -default is set to false 
    piece_path = str
        -takes a string leading to the model predicting the field of the following piece used
        -default is false 
    """
    
    def __init__(self,weight_decay=5e-6,old_model_path=False,piece_path=False):
        
        if type(old_model_path)!=bool: #Install old model if one is already trained or initialize new model by giving no old model
            self.model = load_model(old_model_path)
            
        else:
            
            self.weight_decay=weight_decay
            weight_decay = self.weight_decay
            self.model=Sequential()
            self.model.add(Conv2D(32, (2,2), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(8,8,13)))
            self.model.add(Activation('elu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.15))
            self.model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            self.model.add(Activation('elu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.15))
            self.model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            self.model.add(Activation('elu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Flatten())
            
            self.model.add(Dense(32,activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(64, activation='softmax'))

        if type(piece_path)!=bool:
            self.piece = load_model(piece_path)
            
        else:

            self.weight_decay = weight_decay
            self.piece=Sequential()
            self.piece.add(Conv2D(32, (2,2), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(8,8,13)))
            self.piece.add(Activation('elu'))
            self.piece.add(BatchNormalization())
            
            self.piece.add(MaxPooling2D(pool_size=(2,2)))
            self.piece.add(Dropout(0.2))
            
            self.piece.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            self.piece.add(Activation('elu'))
            self.piece.add(BatchNormalization())
            
            self.piece.add(MaxPooling2D(pool_size=(2,2)))
            self.piece.add(Dropout(0.2))
    
            self.piece.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
            self.piece.add(Activation('elu'))
            self.piece.add(BatchNormalization())
        
            self.piece.add(Flatten())
            
            self.piece.add(Dense(32,activation='relu'))
            self.piece.add(BatchNormalization())
            self.piece.add(Dropout(0.25))
            self.piece.add(Dense(6, activation='softmax'))
            

        print("This is the model which will execute the moves:")  
        print(self.model.summary())
        print(self.piece.summary())
        self.piece.compile(loss=keras.losses.categorical_crossentropy,
                                   optimizer= 'adam',
                                   metrics=['accuracy'])
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                                   optimizer= 'adam',
                                   metrics=['accuracy'])
        
        self.X_train=None
        self.y_train=None

        self.X_piece=None
        self.y_piece=None



    """
    Usually a chess board is notated as a FEN notation (https://de.wikipedia.org/wiki/Forsyth-Edwards-Notation) but 
    we want to have it in matrix form of (8,8,13) where 8x8 coresponds to the board and 12 is the number of different
    possibilities a field can be inherrited by. e.g. own king, enemy kind or empty.
    I referr to own as the AI's figures
    this takes the following input:
    fen_as_string = str 
        -gives the current board as a string in FEN notation

    outputs:
    npy array of the form (8,8,13)
    """
    
    ########Function to translate FEN Notation to matrix notation
    def fen_to_chan(self,fen_as_string):
        end=fen_as_string.find(' ')
        fen_as_string=fen_as_string[:end+1]
        
        board_chan=np.zeros((8,8,13))
        
        figures=['p','b','r','n','q','k','','P','B','R','N','Q','K']
        numbers=['1','2','3','4','5','6','7','8']
        c=0
        for figure in figures:
            chan=np.zeros((8,8))
            field=1
            figure_in=[]

            for x in range(len(fen_as_string)): 
                
                if fen_as_string[x] in numbers:
                    field+=int(fen_as_string[x])
                    
                    
                if fen_as_string[x] in figures:
                    
                    if fen_as_string[x]==figure:
                        figure_in.append(field)
                        
                    field+=1
            
            for x in range(len(figure_in)):
                chan[int(np.ceil(figure_in[x]/8)-1),(figure_in[x]%8)-1]=1
                
            board_chan[:,:,c]=chan
            c+=1
            
        return board_chan
 
    """
    There is Databases out there (prefer:https://www.ficsgames.org ) where you can download those games to prepare this
    one needs to download a png file, change it to .txt and put it into a new folder, run chess_past.create_single_txt_games(...) on this folder
    and every game will be put into a single .txt file
    From those files we create a (N,8,8,13) Matrix where N is number of games* avg moves per game.
    This array holds N board layouts and we will create a resuklt of shape (N,64) which indicates the target field the
    player should move to
    inoputs:
    path_to_folder = str
        -takes the path to a folder containg 'number_of_games' .txt files of which very single on represents a chess game in png notation

    outputs:
    X_train with the form (number of moves,8,8,13)
    y_train with the form (number of moves,64)
    """
    
    def create_training(self,path_to_folder,number_of_games):
        trainx=[] 
        trainy=[]
        
        columns=['a','b','c','d','e','f','g','h']
        rows_f=['8','7','6','5','4','3','2','1']
        fene=['']
        for gam in range(number_of_games):       

            string=open(path_to_folder+'game_'+str(gam)+'.txt','r')

            pgn=chess.pgn.read_game(string)
            
            board=pgn.board()
            
            fens=[]
            moves=[]
            
            for move in pgn.mainline_moves(): #getting all FEN notations and the corresponding moves
                fens.append(board.fen())
                moves.append(str(move))
                board.push(move)
                
            fens_black=[]
            moves_black=[]

            for x in range(len(fens)):#we are only interested in the black moves
                if x % 2==1:
                    if fens[x] not in fene:
                        if x>6:
                            fens_black.append(fens[x])
                            moves_black.append((moves[x],moves[x-2],moves[x-4],moves[x-6]))
                            
                            fene.append(fens[x])
                        else:
                            fens_black.append(fens[x])
                            moves_black.append((moves[x],False))
                            fene.append(fens[x])
                            

                        
            for x in range(len(fens_black)):
                if moves_black[x][1]!=False:
                    trainx.append(self.fen_to_chan(fens_black[x]))
                    destin=moves_black[x][0][2:4]
                    past1=moves_black[x][0][2:4]
                    past2=moves_black[x][1][2:4]
                    past3=moves_black[x][2][2:4]
                    field=(int(destin[1])-1)*8 + int(columns.index(destin[0]))
                    past1_field=(rows_f.index(past1[1]),columns.index(past1[0]))
                    past2_field=(rows_f.index(past2[1]),columns.index(past2[0]))
                    past3_field=(rows_f.index(past3[1]),columns.index(past3[0]))
                    trainy.append((field,past1_field,past2_field,past3_field))
                else:
                    trainx.append(self.fen_to_chan(fens_black[x]))
                    destin=moves_black[x][0][2:4]
                    field=(int(destin[1])-1)*8 + int(columns.index(destin[0]))
                    trainy.append((field,0))
            

                
            
            print('Reading game: ',gam+1,' of ',number_of_games)
            
        X_train=np.zeros((len(trainx),8,8,13))
        y_train=np.zeros((len(trainy),64))
        
        for x in range(len(trainx)):
            if len(trainy[x])>2:
                time=trainx[x]                
                new_chan=np.zeros((8,8))
                new_chan[trainy[x][1]]=1
                new_chan[trainy[x][2]]=0.5
                new_chan[trainy[x][3]]=0.25  
                time[:,:,6]=new_chan               
                X_train[x,:,:,:]=time               
                alp=np.zeros((1,64))
                alp[0,int(trainy[x][0])]=1
                y_train[x,:]=alp
            else:
                X_train[x,:,:,:]=trainx[x]
                alp=np.zeros((1,64))
                alp[0,int(trainy[x][0])]=1
                y_train[x,:]=alp
    
        self.X_train=X_train
        self.y_train=y_train

    """
    same inputs as above which will output the piece used in the y_piece 
    """    

    def create_piece(self,path_to_folder,number_of_games): 



        piecex=[]
        piecey=[] #this will be a number between 0-63 corresponding to the field the player moves to given the current board
        
        columns=['a','b','c','d','e','f','g','h']
        rows=['1','2','3','4','5','6','7','8']
        figures=['p','b','r','n','q','k']
                    
        
        fene=['']
        for gam in range(number_of_games): #iterating through the number of games
            
            string=open(path_to_folder+'game_'+str(gam)+'.txt','r')
            
            pgn=chess.pgn.read_game(string)
            
            board=pgn.board()
            fens=[]
            moves=[]
            
            for move in pgn.mainline_moves(): #getting all FEN notations and the corresponding moves
                fens.append(board.fen())
                moves.append(str(move))
                board.push(move)
                
            fens_black=[]
            moves_black=[]
            
            for x in range(len(fens)):#we are only interested in the black moves
                if x % 2==1:
                    if fens[x] not in fene:
                        fens_black.append(fens[x])
                        fene.append(fens[x])
                        moves_black.append(moves[x])
            
            
            for x in range(len(fens_black)):
                origin = moves_black[x][0:2]
                num_field=columns.index(origin[0])+rows.index(origin[1])*8
                piece = str((chess.Board(fens_black[x])).piece_at(num_field))
                piecey.append(piece)
            
            
            print('Reading game: ',gam+1,' of ',number_of_games)
            

        y_piece=np.zeros((len(piecey),6))
        self.piecey=piecey
        for x in range(len(piecex)):
            alp=np.zeros((1,6))
            alp[0,figures.index(piecey[x])] = 1
            y_piece[x,:]=alp

        self.y_piece=y_piece

    """
    If you have a single txt file in which all the downloaded games are you cann ust give the path e.g. 'spiele.txt' and it will read it and put them in single txt
    files- works only for datasets downloaded from FICS
    You should store the txt file in a new folder so the new txt files are stored there too

    input:
    path_to_all_in_one = str
        -path to the folder in which the txt file with name spiele.txt, which holds all the games downloaded e.g. trainig_set/

    
    output:
        index = int
            -number of games created
        
        stores every game in a single txt file with the name gmae_x.txt
    """



    def create_single_txt_games(self,path_to_all_in_one):

        all_games=open(path_to_all_in_one+'spiele.txt','r')

        games_str=all_games.read()

        games=[]
        index=1
        number_of_games= games_str.count('[Eve')

        while index <=number_of_games:
            
            begin=games_str.find('[Event')
            
            end=games_str.find('} 0-1')
            end+=5
            
            games.append(games_str[begin:end])
            
            games_str=games_str[end:]
            
            index+=1
            
        for x in range(number_of_games):
            
            game=open(path_to_all_in_one+'game_'+str(x)+'.txt','w')
            
            game.write(games[x])
            
            game.close()

        return index

            
        

    """
    This function is used to schedule the learning rate by using keras callbacks, it depends on self.halvings_of_lr and self.epochs
    inputs:
    epoch = int
        - gives the current epoch we are in 

    output:
    lrate
        -is a float which is the leraning rate for the corresponding epoch 
    """      
    def lr_schedule(self,epoch):
        lrate = self.start_lr
        divider=self.halvings_of_lr+1
        progress=epoch/self.epochs
        pot = np.floor(progress*divider) 
        lrate /= np.power(2,pot)
        return lrate  

    def lr_schedule_piece(self,epoch):
        lrate = self.start_lr_piece
        divider=self.halvings_of_lr_piece+1
        progress=epoch/self.epochs_piece
        pot = np.floor(progress*divider)
        lrate /= np.power(2,pot)
        return lrate      
    
    """
    This is the main function to train the model predicting the field we should move to in our current move
    inputs:
    epochs = int
        -is the number of epochs we will train for, one epoch will use every single datapoint in the training set

    batch_size = int
        -is the number of iterations after which we will update our weights in the manner of the adam optimizer

    X_train = np.array
        -takes the training set of the form (number of moves,8,8,13) 
        -set to false as default, will take the self.X_train as input if self.create_training was run, if not asks for the path to the folder with the games 

    y_train = np.array
        -takes the labels of the form (number of moves,64)
        -set to false as default, will take the self.y_train as input if self.create_training was run, if not asks for the path to the folder with the games 

    X_val = np.array
        -takes a validation set
        -set to false by defaulkt, if nothing given it will use a validation split of 0.1 on the training set 

    y_val = np.array
        -takes a validation set
        -set to false by defaulkt, if nothing given it will use a validation split of 0.1 on the training set 

    start_lr = float
        -takes the start learning rate 

    halvings_of_lr = int
        - takes the number of times we halve the learning rate in constant steps

    output:
    model = keras.model
        -outputs a trained model
    """
    
    def train_chess(self,epochs,batch_size,
                    X_train=False,
                    y_train=False,
                    X_val=False,
                    y_val=False,
                    start_lr=1e-3,
                    halvings_of_lr=5):
        
        self.start_lr = start_lr
        self.halvings_of_lr = halvings_of_lr
        self.epochs = epochs
        validation_split=0.1

        if type(X_val)==np.ndarray and type(y_val)==np.ndarray:
            validation_set=(X_val,y_val)
            validation_split=0.0
        if type(X_val)==bool and type(y_val)==bool:
            validation_set=None
            validation_split=validation_split
             
        if type(X_train)==np.ndarray:
            X_train = X_train
        else:
            X_train=self.X_train
            
        if type(y_train)==np.ndarray:
            y_train = y_train
        else:
            y_train = self.y_train
        
        if X_train is None or y_train is None:
            path=input('Give a path to the folder with single games as .txt in order for me to be able to train: ')
            number=input('Now please give the number of games in that folder:')
            self.create_training(path,int(number))
        
        filepath='cache_model.h5'
        
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, 
                                             monitor='val_acc', 
                                             verbose=1, 
                                             save_best_only=True, 
                                             mode='max')
        
        callbacks_list = [checkpoint,keras.callbacks.TerminateOnNaN(),LearningRateScheduler(self.lr_schedule)]
        
        self.model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=callbacks_list,
                  validation_split=validation_split,
                  validation_data=validation_set)

        self.model.load_weights(filepath)

    """
    Same as above but trains the model piece which should predict which piece to use in the next move 
    """


    def train_piece(self,epochs,batch_size,
                    X_piece=False,
                    y_piece=False,
                    X_val=False,
                    y_val=False,
                    validation_split=0.1,
                    start_lr=1e-3,
                    halvings_of_lr=3):
        
        self.start_lr_piece = start_lr
        self.halvings_of_lr_piece = halvings_of_lr
        self.epochs_piece = epochs
         
        if type(X_val)==np.ndarray and type(y_val)==np.ndarray:
            validation_set=(X_val,y_val)
            validation_split=0.0
        if type(X_val)==bool and type(y_val)==bool:
            validation_set=None
            validation_split=validation_split
             
        if type(X_piece)==np.ndarray:
            X_piece = X_piece
        else:
            try:
                X_piece=self.X_piece
            except NameError:
                path=input('Give a path to the folder with single games as .txt in order for me to be able to train: ')
                number=input('Now please give the number of games in that folder:')
                self.create_piece(path,int(number))
            
        if type(y_piece)==np.ndarray:
            y_piece = y_piece
        else:
            try:
                y_piece = self.y_piece
            except:
                path=input('Give a path to the folder with single games as .txt in order for me to be able to train: ')
                number=input('Now please give the number of games in that folder:')
                self.create_piece(path,int(number))
        
        self.piece.fit(X_piece, y_piece,
                  batch_size=batch_size,
                  epochs=self.epochs_piece,
                  verbose=1,
                  callbacks=[keras.callbacks.TerminateOnNaN(),LearningRateScheduler(self.lr_schedule_piece)],
                  validation_split=validation_split,
                  validation_data=validation_set)    
        
    """
    Those functions are to save the models we trained above]
    inputs:
    file_name = str
        -takes the filename as string without the .h5 ending 
    """
  
    ##########Thisfunction will save a model 
    def save_model(self,file_name):
        self.model.save(file_name+'.h5')
    
    def save_piece(self,file_name):
        self.piece.save(file_name+'.h5')
    
    

    """
    This is the function which alternativly determines the next move for the chess engine using a minimax algorithm with alpha beta pruning
    input:
    depth = int
        -takes the depth of our serach tree, namely how many future moves we consider
        -note that this will increase exponentially if you put in higher numbers

    board = str 
        -takes the current board situation in fen notation as input

    maximizer = bool
        -takes the information which player is maximizin, True corresponds to white 

    output:
    best_final = str
        -outputs the optimal move ew should do
    """
    
    
    def minimaxPrune(self,depth, board, maximizer):
        possibleMoves = board.legal_moves
        best_val = -3000
        best_final = None
        for x in possibleMoves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            value = max(best_val, self.minimax(depth - 1, board,-10000,10000, not maximizer))
            board.pop()
            if value > best_val:
                #print("Best score: " ,str(best_val))
                #print("Best move: ",str(best_final))
                best_val = value
                best_final = move
        return best_final
    """
    Is the recursive follow up from the above function which will look in the depths >1 and recursivly go down the tree
    """

    def minimax(self,depth, board, alpha, beta, maximizer2):
        if(depth == 0):
            if maximizer2:
                return -self.evaluation(board)
            else:
                return self.evaluation(board)
            
        possibleMoves = board.legal_moves
        if depth % 1 == 0:
            bestMove = -3000
            for x in possibleMoves:
                move = chess.Move.from_uci(str(x))
                board.push(move)
                bestMove = max(bestMove,self.minimax(depth - 1, board,alpha,beta, not maximizer2))
                board.pop()
                alpha = max(alpha,bestMove)
                if beta <= alpha:
                    return bestMove
            return bestMove
        else:
            bestMove = 3000
            for x in possibleMoves:
                move = chess.Move.from_uci(str(x))
                board.push(move)
                bestMove = min(bestMove, self.minimax(depth - 1, board,alpha,beta, not maximizer2))
                board.pop()
                beta = min(beta,bestMove)
                if(beta < alpha):
                    return bestMove
            return bestMove
        
    """
    This function evaluates the board by weighing the pieces on the board
    input:
    board = str
        -takes a string of fen notation as input

    output:
    evaluation = int
        -gives the value of the board for the player looking at it
        -will be negaitve if the piece values of the enemy are higher 
    """
    
    def evaluation(self,board):
        i = 0
        turn=board.turn
        evaluation = 0
        x = True
        while i < 64:
            try:
                x = bool(board.piece_at(i).color)
            except AttributeError:
                x = x   
            if turn!=x:
                evaluation-=self.piece_val(str(board.piece_at(i)))
            if turn==x:
                evaluation+=self.piece_val(str(board.piece_at(i)))
            i += 1
        return evaluation
    
    """
    weihs the pieces with corresponding weights
    """
    
    def piece_val(self,piece):
        if(piece == None):
            return 0
        value = 0
        if piece == "P" or piece == "p":
            value = 7
        if piece == "N" or piece == "n":
            value = 28
        if piece == "B" or piece == "b":
            value = 33
        if piece == "R" or piece == "r":
            value = 60
        if piece == "Q" or piece == "q":
            value = 140
        if piece == "K" or piece == "k":
            value = 840
        return value