import os, collections

rootdir = './ball-net/train/'
y_path = rootdir + "y/"
n_path = rootdir + "n/"

for root, subdir, files in os.walk(rootdir):
    for item in files:
        if item != ".DS_Store" and item.split("-")[0] == "b":
            original = os.path.join(root, item)
            new_name = os.path.join(rootdir, item)

            print(f'Moved {original} to {new_name}')
            os.rename(original, new_name)


# for item in os.listdir(rootdir):
#     original = os.path.join(rootdir, item)
    
#     if item != ".DS_Store" and os.path.isfile(original):
#         c_name = "c-" + item.split("-")[1]

#         if os.path.isfile(os.path.join(y_path, c_name)):
#             new_name = os.path.join(y_path, item)
#         elif os.path.isfile(os.path.join(n_path, c_name)):
#             new_name = os.path.join(n_path, item)
#         else:
#             print("Unexpected!")
        
        
#         os.rename(original, new_name)
#         print(f'Moved {original} to {new_name}')
