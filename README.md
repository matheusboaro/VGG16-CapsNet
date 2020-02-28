# VGG16-CapsNet
CapsNet adaptation to melanoma detection.
To run this model you need:
1-A data set with the features extracted by some other network (In this case VGG16) and save as .npy.
2-Run "create_name_list.py" with the directories of train, test and validation
3-Run "create_pickle_npy.py" to create batches of pickles to feed the Capsnet.
  obs: You need to save as testX.npy...trainX.npy...validX.npy...
4-Run "capsulenet.py"

obs: The "ft" file do the extraction, u just need to load it with your onw keras model
