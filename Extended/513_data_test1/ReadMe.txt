We use 513 datas rom ipco cycle and shiraz ipco days. I added new feature first calculate the speed diff and named Acc1 and after that i shifted Acc and added ACC2 and trained with this and its better than one ACC without shifted.
after that I shifted up trip fuel consumption diff for train and test and other thing is use averegae fuel consumption rate and i dont which is better ?

sequence len is 600 and batch size is 64

after that we tested XUP nedc for 5 gear car and we predict 8.64 with fuel cons rate and 8.51 with avg rate 
The real NEDC for this car is 8.5 and error with rate under the 1% and with trip fuel cons is 1.5-2%


ضریب تصحیح برابر 1.02 تا 1.03

after that augmented file with forced that all trips start from cold and also we have last augment and after that tested and this augment methodes not good

after that we checked the sequence len with 600 60 and 10 and MAE mse and MAPE in seq10 is the best and we choose the low sequence

***************************************************-----------------**********************

In summary : 1- shifted acc is good for train
2- seq len lower is better 10 choose
3- Augment for cold start not good
4-2slice for aug is good
5-Trip fuel consumption with shifted up better that avg trip fuel rate


*************-----------------***************---------

I tested with BIGRU and its better than Bilstm but improvement not much