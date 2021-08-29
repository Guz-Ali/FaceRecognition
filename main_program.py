import face_recognition
import cv2
import numpy as np
import os
import glob
import pandas as pd
import math
from datetime import datetime


faces_encodings = []
faces_names = []
cur_direc = os.getcwd()
path = os.path.join(cur_direc, 'data/faces/')
#path = os.path.join(cur_direc, 'data/test_faces/')
list_of_files = [f for f in glob.glob(path+'*.jpg')] + [f for f in glob.glob(path+'*.jpeg')]
number_files = len(list_of_files)
names = list_of_files.copy()


#TRAINING
for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    print(i)
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
# Create array of known names
    names[i] = names[i].replace(cur_direc, "")
    names[i] = names[i].replace('/data/faces/', "")
    #test faces
    #names[i] = names[i].replace('/data/test_faces/', "")
    names[i] = names[i].replace('.jpg', "")
    names[i] = names[i].replace('.jpeg', "")
    faces_names.append(names[i])
print('face ENCODINGSSSSSS')
print(faces_encodings)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#Starts the videocam
video_capture = cv2.VideoCapture(0)

frame_, different_face, frame_start, face_names_start, face_names_start_str, time_add, current_date_time\
= 0, True, 0, ['not_a_valid_face'], 'not_a_valid_face', 3, datetime.now(),

current_date = current_date_time.strftime("%m_%d_%Y")
current_time = current_date_time.strftime("%H:%M:%S")


def check_face(d_f, f_, f_n):
    global different_face, frame_start, face_names_start, face_names_start_str, time_add, current_date_time
    if d_f: # should happen when we see a new face, get the start of the time
        print('it was a different face')
        frame_start, face_names_start, different_face, time_add\
        = f_, f_n, False, 3
        if face_names_start != []:
            face_names_start_str = str(face_names_start[0])
        print('time_add')
        #what is face_names_start
        print('face_names_start ===== ', face_names_start)
    if f_ == frame_start+time_add:
        #if it's the same face and it has been two seconds that we have been seeing the face
        time_add+=3
        #print(face_names==face_names_start)
        if f_n == face_names_start and f_ >= frame_start+48:
            #register face to csv
            print('face Registered \n \n face Registered')
            #after this, either stop the video cam, or set the frame_start again
            #because we registered the person, for now, break
            print('we registered the face, closing the cam')
            #take date and time here
            current_date_time = datetime.now()
            current_date = current_date_time.strftime("%m_%d_%Y")
            current_time = current_date_time.strftime("%H:%M:%S")
            print(current_date)
            print(current_time)
            return True
            #will break with return True

        elif f_n == face_names_start:
            #we know it's the same face but not two seconds yet, wait for the next turn
            #cancel setting a new time counter and new face
            print('made different_face False')
            if f_n == []:
                different_face = True
            else:
                different_face = False
            return False #continue while loop
        else:
            #no face or another face, should change the face in the list, and restart the counter
            print('made different_face False')
            different_face = True
            return False




while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
                #testing
                print(name[-1] + name[-2])
                name = name.replace(name[-2] + name[-1], "")
            face_names.append(name)

            print(face_names)
    process_this_frame = not process_this_frame

#display the result

#add a condition to tell that this should run only when it sees a face, hence when face_names is not empty

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

#draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#Input text label with a name below the rectangle
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#Display the resulting image
        cv2.imshow('Video', frame)

#Add a time counter
    if check_face(different_face, frame_, face_names):
        break

#hit Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#add a frame
    frame_+=1
#while LOOP ED

print('adding the recognized face to csv')

#add the recognized face to the csv
#open csv
if os.path.isfile('data/check_in_data_' + current_date +'.csv'):
    df = pd.read_csv('data/check_in_data_' + current_date +'.csv')
else:
    df= pd.DataFrame([['', None, None, None, None]], columns = ['recognized_faces', 'date', 'time', 'check_in', 'check_out'])
    df.to_csv('data/check_in_data_' + current_date +'.csv', index=False)

    print('created an empty csv file')
#change the list
#df['recognized_faces'] = face_names_start
#df['date_time'] = current_date_time

#tried adding the name instead of changing the whole list

#df['recognized_faces'] = df['recognized_faces'].astype('string')
#df['date_time'] = df['date_time'].astype('datetime64[ns, US/Eastern]')
#df['time'] = df['time'].astype('time')
#df['recognized_faces'].append(face_names_start)
#df['date_time'].append(current_date_time)

#change the data at index, but cannot reach the one in the next index, since we want to add to that index
#df['recognized_faces'].iloc[len(df['recognized_faces'])] = face_names_start
#df['date_time'].iloc[-1] = current_date_time

#IF already checked_in, then add check_out time.
print('line 186')
print(df['recognized_faces'].empty)

person_index =0

if len(df) >=2:
    for i in  df['recognized_faces'].str.find(face_names_start_str):
        if i == 0:
            break
        person_index+=1
    if person_index == len(df) and df['recognized_faces'].str.find(face_names_start_str)[person_index-1] == -1:
        person_index = -1
    print(person_index)
else:
    person_index = -1


#print(person_index, df['check_in'][person_index].empty)


#print(df['check_in'][person_index])
#print(df['check_out'][person_index])
                        #future update wtd

if person_index >= 0: #and (df['check_out'][person_index]) == None
    print(pd.isnull(df['check_out'][person_index]))
    if pd.isnull(df['check_out'][person_index]):
        df['check_out'][person_index] = current_time
        print('added the check out time')
    else:
        print('You have already checked out')
else:
#append a new DataFrame IF NOT already checked_in
#print(df['recognized_faces'].get_loc(face_names_start))
#print(list(df).index(face_names_start))
    df2 = pd.DataFrame([[face_names_start_str, current_date, current_time, current_time, None]], columns = ['recognized_faces', 'date', 'time', 'check_in', 'check_out'])
#df_new = pd.DataFrame(np.array([face_names_start], [current_date_time]), columns=['recognized_faces','date_time'])
    print(face_names_start)
    print(df2['recognized_faces'][0])
    df = df.append(df2)
    print('added the face')

df.to_csv('data/check_in_data_' + current_date +'.csv', index=False)#don't write indeces

#release videocam
video_capture.release()
cv2.destroyAllWindows()
