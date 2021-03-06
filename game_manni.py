#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Sun Jun 16 13:01:59 2019

@author: Cluckenstein
"""

"""
This file will be the acutal game in which the computer makes decisions on where to move in the next move and 
the player can interact in the console window IPython

In order to play the game, you need to run this file and then run game_AI() with the standard model or another one
youc also have to install the requirementds.txt
inputs:
path_to_model = str
    -if you have a model you can input it here with a file path
    -if not given uses the default model which comes with the file

piece_path = str
    -same goes for the model which is predicitng the piece to move with 
    -if not given uses the default model which comes with the file

game_mode = 'CNN' or 'Tree' 
    -if CNN acitvated the engine will move based on the CNN predicitions
    -if Tree acitvated it will move based only on the tree search minimax algorithm

board = str
    -takes a fen notation if you want to continue from an old game or so
    -if none given a new game will be started 

output:
result = str
    -tells who won

elapsed = float
    -tells how many seconds the game went on 

moves_black = list
moves_white = list
    -list of moves takend by the respective player

"""

#import build_enigne
import time
import chess
import numpy as np
import random
from IPython.display import SVG,display
from chess.svg import board

def game_AI(path_to_model=False,piece_path=False,
            game_mode='CNN',board_old=None):

    columns=['a','b','c','d','e','f','g','h']
    rows=['1','2','3','4','5','6','7','8']
    rows_f=['8','7','6','5','4','3','2','1']
    figures=['p','b','r','n','q','k']

    begin=time.time()
    
    moves_white=[]
    moves_black=[]
    
    if type(path_to_model)==bool:
        path_to_model='final_model.h5'
        
    if type(piece_path)==bool:
        path_to_piece='final_piece.h5'
    
    manfred=chess_past(old_model_path=path_to_model,piece_path=path_to_piece)
    
    print("You are white which is represented by the white figures")
    
    
    print("Your moves have to be of the form from square to square e.g. a1a3 which will move the figure standing on a1 to a3 if it is a legal move")
    print("You begin!")
    print("If you can and want to exchange a pawn for a Queen, type to move and add a 'Q' at the end e.g. a7a8q ")
    move_indicator = 1     
    if board_old==None:
        board=chess.Board()
    else:
        board=board_old
        if not board.turn:
            move_indicator+=1
    
    
    display(SVG(chess.svg.board(board,coordinates=True)))
    while not board.is_game_over():

        turn = (-1)**move_indicator
        
        if turn<0:###enemy players turn
            
            move_legal = False
            
            pos_moves=[]
            for move in board.legal_moves:
                pos_moves.append(str(move))
            
            while move_legal == False:
                
                white_move = input("Type your desired move: ")
                
                if ((len(white_move)==4 or len(white_move)==5) and white_move[0] in columns and white_move[2] in columns and white_move[1] in rows and white_move[3] in rows):
                    
                    if white_move in pos_moves:
                        
                        move_legal = True
                        
                        board.push(chess.Move.from_uci(white_move))
                        
                        moves_white.append(white_move)
                    
                    else:
                    
                        print("Invalid or illeagl move, try again!")
                else:
                    print("Invalid or illeagl move, try again!")
            
            if board.is_check()==True:
                print("Good Job, you put Manfred into check!")
                
                
        if turn>0:######Manfreds turn
            
            print("It is Manfred's move now, be aware!")
            
            pos_moves=[]
            change=False
            for move in board.legal_moves:
                pos_moves.append(str(move))
            
            for move in pos_moves:
                if len(move)>4:
                    change=True
                    changing_move=move
                
            black_move=None
            
            if game_mode == 'CNN':
                
                input_for_manni = np.zeros((1,8,8,13))
                input_for_manni[0,:,:,:] = manfred.fen_to_chan(board.fen())
                
                if move_indicator>6:
                    
                    old_moves_chan=np.zeros((8,8))
                    past1=moves_black[-1][2:4]
                    past2=moves_black[-2][2:4]
                    past3=moves_black[-3][2:4]
                    past1_field=(rows_f.index(past1[1]),columns.index(past1[0]))
                    past2_field=(rows_f.index(past2[1]),columns.index(past2[0]))
                    past3_field=(rows_f.index(past3[1]),columns.index(past3[0]))
                    old_moves_chan[past1_field]=1
                    old_moves_chan[past2_field]=0.5
                    old_moves_chan[past3_field]=0.25
                    input_for_manni[0,:,:,6]=old_moves_chan
                
                prediction = manfred.model.predict(input_for_manni)
                
                field = np.argmax(prediction)
                prediction[0,field]=0.
                field2 = np.argmax(prediction)
                prediction[0,field2]=0.
                field3 = np.argmax(prediction)
                """
                I will use the second and third biggest output to in order to make better predictions
                """
                if field-field2 > 5e-2:
                    field2 = False
                    field3 = False
                    
                if field-field3 > 5e-2:
                    field3 = False
                
                
                piece = figures[np.argmax(manfred.piece.predict(input_for_manni))]
                
                pos=[]
                pos2=[]
                pos3=[]
                
                y = int((np.floor(field/8))+1)
                x = columns[field%8]
                target=x+str(y)
                
                if field2!=False:
                    y2 = int((np.floor(field2/8))+1)
                    x2 = columns[field2%8]
                    target2=x2+str(y2)
                    
                    for x in range(len(pos_moves)):
                        if pos_moves[x][2:4]==target2:          
                            orig=columns.index(pos_moves[x][0])+rows.index(pos_moves[x][1])*8           
                            if str(board.piece_at(orig)) == piece:
                                pos2.append(pos_moves[x])
                    
                    
                if field3!=False:
                    y3= int((np.floor(field3/8))+1)
                    x3 = columns[field%8]
                    target3=x3+str(y3)
                    
                    for x in range(len(pos_moves)):
                        if pos_moves[x][2:4]==target3:          
                            orig=columns.index(pos_moves[x][0])+rows.index(pos_moves[x][1])*8           
                            if str(board.piece_at(orig)) == piece:
                                pos3.append(pos_moves[x])
                
                for x in range(len(pos_moves)):
                    if pos_moves[x][2:4]==target:          
                        orig=columns.index(pos_moves[x][0])+rows.index(pos_moves[x][1])*8           
                        if str(board.piece_at(orig)) == piece:
                            pos.append(pos_moves[x])
                
                if len(pos)>1:
                    provisional=random.choice(pos)
                    print("More than one choice for many, let the dices run")
                if len(pos)==1:
                    provisional=pos[0]
                    print("First choice")
                    
                if len(pos)==0:
                    
                    if len(pos2)>1:
                        provisional=random.choice(pos2)
                        print("More than one choice for many, let the dices run")
                        print("Second choice")
                    if len(pos2)==1:
                        provisional=pos2[0]
                        print("Second choice")

                    if len(pos2)==0:
                        if len(pos3)>=1:
                            provisional=pos3[0]
                            print("Third choice")
                            
                        if len(pos3)==0:## if there is no legal moves avalible calculated by the CNN, we use a minimax
                            black_move=manfred.minimaxPrune(3,board,False)
                            board.push(black_move)
                            moves_black.append(str(black_move))
                            print("Moved by Minimax")
                
                
                if black_move==None:
                    print("Manfred is thinking about his next move!")
                    #time.sleep(random.uniform(0,1)*4)
                    black_move=chess.Move.from_uci(provisional)
                    board.push(black_move)
                    moves_black.append(str(provisional))
                    print("Moved by CNN AI")
                
                if change:
                    board.pop()
                    board.push(chess.Move.from_uci(changing_move))
            
            if game_mode == 'Tree':
                black_move=manfred.minimaxPrune(3,board,False)
                board.push(black_move)
                moves_black.append(str(black_move))
                print("Moved by Minimax Treesearch")
                
            if board.is_check()==True:
                print("Watch out! Manfed put you into check!")
            


        display(SVG(chess.svg.board(board,coordinates=True)))
        
        move_indicator+=1
     
    end=time.time()
    
    result=board.result()
    if result == '1-0':
        print('You WON!')
    if result == '0-1':
        print('Manfred WON! Better luck next time!')
    if result == '1/2-1/2':
        print('It is a TIE!')
    
    elapsed=end-begin
    print("You played: ",elapsed/60 ,' min')   
    
    return result,elapsed,moves_black,moves_whiteg