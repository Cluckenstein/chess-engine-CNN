**This is a chess engine based on a Convolutional Neural Network**

To run:
* Might need to pip install keras,python-chess etc.
* If you want to see it play against stockfish engine you need to download and make said
* Else run the game_manni.py file and type game_AI() in your IPython console

It is far from being a really good chess player, rather than it showing how one can utilize modern machine learning with only a few lines of code 
Problems with the algorithm are but are not limited to: 
*  Strategies cannot be palyed as a whole
*  The CNN is trained on games of Pro players with a ELO rating higher than 2000 this causes the problem that it is rather confused with the enemy player making 'stupid' moves
*  The endgame phase is hard to read for the machine since the informartions are very limited (having only ~6 pieces on the board)
*  Looking into the future moves of the enemy only happens from the experience it learned from the training games rather than a search tree
    
The board and the corresponding training sets are encoded in a so called bitmap which means a boardlayout is represented in a matrix of the form 8x8x13 
This 8x8x13 matrix holds the following information:

    
*  Channel 1 `[:,:,0]`: The corresponding positions pawns belonging to the machine itself 
*  Channel 2 `[:,:,1]`: The corresponding positions bishops belonging to the machine itself
*  Channel 3 `[:,:,2]`: The corresponding positions rooks belonging to the machine itself
*  Channel 4 `[:,:,3]`: The corresponding positions knights belonging to the machine itself
*  Channel 5 `[:,:,4]`: The corresponding positions queens belonging to the machine itself
*  Channel 6 `[:,:,5]`: The corresponding positions king belonging to the machine itself
    
*  Channel 7 `[:,:,6]`: In this channel the last three own moves are encoded in order to be able to have a bit of strategic understanding
    
*  Channel 8`[:,:,7]`: The corresponding positions pawns belonging to the enemy player
*  Channel 9 `[:,:,8]`: The corresponding positions bishops belonging to the enemy player
*  Channel 10 `[:,:,9]`: The corresponding positions rooks belonging to the enemy player
*  Channel 11 `[:,:,10]`: The corresponding positions knights belonging to the enemy player
*  Channel 12 `[:,:,11]`: The corresponding positions queens belonging to the enemy player
*  Channel 13 `[:,:,12]`: The corresponding positions king belonging to the enemy player
    
Sometimes it happens that the model predicts a field on which there is noch legal move possible, in thoses cases a classical minimax treesearch algrotihm helps out to predict a move
