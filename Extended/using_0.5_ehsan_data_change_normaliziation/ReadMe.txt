I checked the min and max new data (0.5s) and i found the maximum of momentary fuel about 7000 but in the 1s data its about 12000, So changed the target normalization from 0-20000 to 0 -10000 and check the sequence length 200 and 600, change normalization for prevent gradient vanishing and results show the effects but yet not good. Next step add layer to network.
test file has a problem but in other files that good and sl=600 better than sl=200.
Tested with sequence length 400 and results show the sequence length 600 is better than others.
SL 50 and 100 was good too, lr(no change) and epoch to 50.
