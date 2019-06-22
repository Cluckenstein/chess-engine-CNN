#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Jun 19 16:23:01 2019

@author: Cluckenstein
"""

########################################################################
#In this file we will let the stockfish chess engine play against our CNN
#We are able to view the game in a notebook or Ipython console
#In order to run this code on must ensure to have a compiled stockfish
#engine and input the path below 
#
#Since the CNN cannot really handle late game we set a move limiter and 
#evaluate the boards position after a certain number of moves 
#
#To execute just type: gamevs()
########################################################################


import chess
import numpy as np
import chess.svg
import build_engine
import time
import random
from IPython.display import SVG,display
import chess.engine

def gamevs(path_to_model=False,piece_path=False,
            game_mode='CNN',board=None):
    
    engine = chess.engine.SimpleEngine.popen_uci("/Users/USER_NAME/Downloads/Stockfish-master/src/stockfish")

    columns=['a','b','c','d','e','f','g','h']
    rows=['1','2','3','4','5','6','7','8']
    figures=['p','b','r','n','q','k']
    moves_white=[]
    moves_black=[]
    
    if type(path_to_model)==bool:
        path_to_model='model_past_1_5epochs.h5'
        
    if type(piece_path)==bool:
        path_to_piece='piece_test_1_10epochs.h5'
    
    manfred=build_engine.chess_past(old_model_path=path_to_model,piece_path=path_to_piece)
    
    #print("You are white which is represented by the white figures")
    
    
    #print("Your moves have to be of the form from square to square e.g. a1a3 which will move the figure standing on a1 to a3 if it is a legal move")
    #print("You begin!")
    #print("If you can and want to exchange a pawn for a Queen, type to move and add a 'Q' at the end e.g. a7a8q ")
    move_indicator = 1     
    if board==None:
        board=chess.Board()
    else:
        if not board.turn:
            move_indicator+=1
    
    
    display(SVG(chess.svg.board(board,coordinates=True)))
    while not board.is_game_over() and move_indicator<8:

        turn = (-1)**move_indicator
        
        if turn<0:###enemuy players turn
                
            result = engine.play(board, chess.engine.Limit(time=0.100,depth=3))
            board.push(result.move)
            
            time.sleep(1)
        if turn>0:######Manfreds turn
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
                
                prediction = manfred.model.predict(input_for_manni)
                
                field = np.argmax(prediction)
                prediction[0,field]=0.
                field2 = np.argmax(prediction)
                prediction[0,field2]=0.
                field3 = np.argmax(prediction)
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
                if len(pos)==1:
                    provisional=pos[0]
                
                if len(pos)==0:
                    
                    if len(pos2)>1:
                        provisional=random.choice(pos2)
                    if len(pos2)==1:
                        provisional=pos2[0]
                    if len(pos2)==0:
                        if len(pos3)>=1:
                            provisional=pos3[0]
                        if len(pos3)==0:
                            black_move=manfred.minimaxPrune(3,board,False)
                            board.push(black_move)
                            moves_black.append(black_move)

                
                if black_move==None:
                    black_move=chess.Move.from_uci(provisional)
                    board.push(black_move)
                    moves_black.append(provisional)
                
                if change:
                    board.pop()
                    board.push(chess.Move.from_uci(changing_move))
            
            if game_mode == 'Tree':
                black_move=manfred.minimaxPrune(3,board,False)
                board.push(black_move)
                moves_black.append(black_move)
 
        display(SVG(chess.svg.board(board,coordinates=True)))       
        move_indicator+=1
    
    if board.turn==True:
        turn='white'
    else:
        turn='black'

    score=manfred.evaluation(board)

    result=board.result() 
    
    return (result,score,turn)