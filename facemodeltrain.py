import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

round=["FACE","AGE","NATIONALITY","GENDER"]
path_round = ["./faces/user","./ages/age","./nationalities/nation","./genders/gender"]
yml_round = ["facemodel","agemodel","nationalitymodel","gendermodel"]

var=0
while var!=4:
    id=100
    onlyfiles = []
    i=0
    Training_Data,Labels = [],[]
    var = int(input("1)FACE \n2)AGE \n3)Nationality \n4)Gender \n5)EXIT \nCHOOSE THE MODEL TO CREATE : "))-1
    if(var==4):
        break
    while id!=0:
        id = int(input("Enter user ID to train : "))
        if(id==0):
            break
        data_path = path_round[var]+str(id)+'/'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
        print(onlyfiles)
        i=i+1
        for i,files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images,dtype=np.uint8))
            Labels.append(id)

    Labels = np.asarray(Labels,dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()    #cv2.createLBPHFaceRecognizer()
#    eigenmodel = cv2.face.EigenFaceRecognizer_create()

    print(round[var]+" being trained...")
    model.train(np.asarray(Training_Data),np.asarray(Labels))
    model.save(yml_round[var]+'/trainingData.yml')

#    eigenmodel.train(np.asarray(Training_Data), np.asarray(Labels))
#    eigenmodel.save("eigen"+yml_round[var] + '/trainingData.yml')

print("Now our LBPH MODEL is Ready to go ")
