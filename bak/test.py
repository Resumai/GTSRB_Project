# I place and delete my junk as i test here(sometimes)
# Test n1
# from utils.df_compare import df_compare
# import pandas as pd

# df = pd.read_csv("best_cnn_model.csv", sep=";")

# df_compare(df)


# Test nr2
# var1 = "This"
# var2 = "is"
# var3 = "a"
# var4 = "test"

# def custom_func(var1, var2, var3, var4):
#     print(var1, var2, var3, var4)


# custom_func(
#     var1,
#     # test 
#     var2,
#     var3 = "sparta", # test
#     var4 = "!"
# )

# Clears gradient from previous itteration
optimizer.zero_grad() 
# Feeds batch of pictures
outputs = model(images) 
# Compares predictions to true labels
loss : torch.nn.CrossEntropyLoss = criterion(outputs, labels)
# Calculates gradients of the loss
loss.backward() 
# Updates models parameters using the grad
optimizer.step() 