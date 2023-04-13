import pandas as pd
import glob
import os
from csv import writer

dfkeyframe = pd.read_csv("KeyFrameCount.csv")
MovieName = dfkeyframe.VideoName.values
KeyFrameNo = dfkeyframe.KeyFrames.values

#The code will iterate through folder "Result", read each .csv which is one movie and construct the facial features.
#The final dataset will include movie trailer with their facial attributes, each row will be a movie and columns are facial features
Datasets = ['Result']
for name in Datasets:
    #os.chdir(Datasets)
    for f in glob.glob(name+"\*.csv"):
        print(f)

        OneRow = []
        print(f)  # f is a file name
        df = pd.read_csv(f, names=['videoname', 'link', 'frameno', 'totalframe','totalfps',
                               'faceno', 'age', 'gender', 'emotion', 'race',
                               'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral',
                               'asian', 'indian', 'black', 'white', 'middle', 'latino',
                               ' x', 'y', 'w', 'h'])
        videoname = df['videoname'].unique().item()
        videolink = df['link'].unique().item()
        OneRow.extend((videoname,videolink))

        keyframesno = df['totalfps'].unique().item()
        #Find keyframe Number
        for i in range(len(MovieName)):
            if(MovieName[i]==videoname):
                keyframesno = KeyFrameNo[i]
                break

        #faceno
        faceno = df['faceno'].count()
        ratiofaceno = faceno/keyframesno
        OneRow.extend((faceno,ratiofaceno))

        #Gender
        female = (df['gender']=='Woman').sum()
        if (female == None):
            female = 0
            ratiofemale = 0
        else:
            ratiofemale = female / faceno
        OneRow.extend((female,ratiofemale))

        male = (df['gender'] == 'Man').sum()
        if (male == None):
            male = 0
            ratiomale = 0
        else:
            ratiomale = male / faceno
        OneRow.extend((male,ratiomale))

        #Emotions
        emotions = df['emotion'].value_counts()
        emodict = emotions.to_dict()
        emotionlist = ['sad','happy','fear','angry','surprise','disgust','neutral']
        for emotion in emotionlist:
            numberemo = emodict.get(emotion)
            if (numberemo==None):
                numberemo = 0
                ratio = 0
            else:
                ratio = numberemo/faceno
            OneRow.extend((numberemo,ratio))


        #Races
        races = df['race'].value_counts()
        racedict = races.to_dict()
        racelist = ['asian','indian','black','white','middle eastern','latino hispanic']
        for race in racelist:
            numberrace = racedict.get(race)
            if (numberrace==None):
                numberrace = 0
                ratiorace = 0
            else:
                ratiorace = numberrace / faceno
            #print(race, numberrace)
            OneRow.extend((numberrace,ratiorace))


        #Age
        agemean = df['age'].mean()
        print("Age mean is: ",agemean )
        OneRow.append(agemean)

        #Face size
        df['facepixel'] = df['w']*df['h']
        sumface = df['facepixel'].sum()
        AvgFaceSize = sumface / faceno
        OneRow.append(AvgFaceSize)

        df['totalpixel'] = 720 * 1280
        TotalPixel = df['totalpixel'].sum()

        AveragePixels = sumface / TotalPixel

        OneRow.append(AveragePixels)

        print(len(OneRow))
        with open('FaceDataset.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(OneRow)
            f_object.close()






