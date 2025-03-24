# import cv2
# import os
# # from flask import Flask, request, render_template
# from datetime import date
# from datetime import datetime
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# import pandas as pd
# import joblib

# # Defining Flask App


# nimgs = 10

# # Saving Date today in 2 different formats
# datetoday=date.today().strftime("%m_%d_%y")
   
# def datetoday2():
#     return date.today().strftime('%B %d, %Y')  # Example format



# # Initializing VideoCapture object to access WebCam
# face_detector = cv2.CascadeClassifier('app\haarcascade_frontalface_default.xml')


# # If these directories don't exist, create them
# if not os.path.isdir('Attendance'):
#     os.makedirs('Attendance')
# if not os.path.isdir('static'):
#     os.makedirs('static')
# if not os.path.isdir('static/faces'):
#     os.makedirs('static/faces')
# if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
#     with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
#         f.write('Name,Roll,Time')


# # get a number of total registered users
# def totalreg():
#     return len(os.listdir('static/faces'))


# # extract the face from an image
# def extract_faces(img):
#     try:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
#         return face_points
#     except:
#         return []


# # Identify face using ML model
# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)


# # A function which trains the model on all the faces available in faces folder
# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             resized_face = cv2.resize(img, (50, 50))
#             faces.append(resized_face.ravel())
#             labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')


# # Extract info from today's attendance file in attendance folder
# from .models import StudentAttendance

# def extract_attendance(date,period):
#     """
#     Extracts attendance details for the given date from the StudentAttendance table.
    
#     Args:
#         date (datetime.date): The date for which attendance is to be fetched.
    
#     Returns:
#         names (list): List of student names.
#         rolls (list): List of student IDs (roll numbers).
#         times (list): List of attendance times.
#         l (int): Total number of attendance records for the given date.
#     """
#     # Query the StudentAttendance table for records matching the given date
#     attendance_records = StudentAttendance.objects.filter(date=date,period=period)

#     # Extract details from the query set
#     names = [record.studentName.studentName for record in attendance_records]
#     rolls = [record.studentId.studentId for record in attendance_records]
#     times = [record.time for record in attendance_records]
#     l = len(attendance_records)

#     return names, rolls, times, l



# from datetime import datetime
# from .models import Student, StudentAttendance

# from .models import Student, StudentAttendance
# from datetime import datetime

# def add_attendance(name, period):
#     # Extract the studentName and studentId from the provided name
#     try:
#         student_name = name.split('_')[0]
#         student_id = int(name.split('_')[1])  # Assuming studentId is an integer
#     except (IndexError, ValueError):
#         return {'status': 'error', 'message': f'Invalid name format: {name}. Expected "studentName_studentId".'}

#     current_date = datetime.now().date()  # Current date
#     current_time = datetime.now().time()  # Current time

#     try:
#         # Get the student object from the database
#         student = Student.objects.get(collegeId=student_id, studentName=student_name)
        
#         # Check if the student is active (status=1)
#         if student.status != 1:
#             return {'status': 'error', 'message': f'Student {student_name} (ID: {student_id}) is inactive. Attendance not marked.'}

#         # Check if attendance for this student on the same date and period already exists
#         attendance_exists = StudentAttendance.objects.filter(
#             studentId=student, date=current_date, period=period
#         ).exists()

#         if not attendance_exists:
#             # Save the attendance record in the StudentAttendance model
#             new_attendance = StudentAttendance(
#                 studentId=student,
#                 studentName=student.studentName,  # Use studentName from the Student object
#                 date=current_date,
#                 period=period,
#                 time=current_time,
#                 studentClass=student.studentClass
#             )
#             new_attendance.save()
#             print(f"Attendance added for {student_name} (ID: {student_id}) on {current_date} at {current_time}.")
#             return {'status': 'success', 'message': f'Attendance marked for {student_name} (ID: {student_id}).'}
#         else:
#             print(f"Attendance already exists for {student_name} (ID: {student_id}) on {current_date}.")
#             return {'status': 'info', 'message': f'Attendance already marked for {student_name} (ID: {student_id}).'}

#     except Student.DoesNotExist:
#         print(f"Student with ID {student_id} and name {student_name} does not exist.")
#         return {'status': 'error', 'message': f'Student with ID {student_id} and name {student_name} does not exist.'}
#     except Exception as e:
#         print(f"Error marking attendance: {e}")
#         return {'status': 'error', 'message': f'Error marking attendance: {str(e)}'}



# ## A function to get names and rol numbers of all users
# def getallusers():
#     userlist = os.listdir('static/faces')
#     names = []
#     rolls = []
#     l = len(userlist)

#     for i in userlist:
#         name, roll = i.split('_')
#         names.append(name)
#         rolls.append(roll)

#     return userlist, names, rolls, l


# ## A function to delete a user folder 
# def deletefolder(duser):
#     pics = os.listdir(duser)
#     for i in pics:
#         os.remove(duser+'/'+i)
#     os.rmdir(duser)


import cv2
import os
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import logging
from collections import Counter  # Added for potential use in other functions

logging.basicConfig(level=logging.DEBUG)

nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")

def datetoday2():
    return date.today().strftime('%B %d, %Y')

# Initialize face detector
face_detector = cv2.CascadeClassifier('app/haarcascade_frontalface_default.xml')
if face_detector.empty():
    raise ValueError("Failed to load Haar cascade file.")

# Directory setup
for dir_path in ['Attendance', 'static', 'static/faces']:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray

def extract_faces(img):
    try:
        gray = preprocess_image(img)
        face_points = face_detector.detectMultiScale(gray, 1.1, 6, minSize=(30, 30))
        return face_points
    except Exception as e:
        logging.error(f"Face extraction error: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    prediction = model.predict(facearray)
    logging.debug(f"Predicted: {prediction}")
    return prediction

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            gray = preprocess_image(img)
            resized_face = cv2.resize(gray, (100, 100))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    logging.info("Model training completed.")

from .models import Student, StudentAttendance

def extract_attendance(date, period):
    attendance_records = StudentAttendance.objects.filter(date=date, period=period)
    names = [record.studentName.studentName for record in attendance_records]
    rolls = [record.studentId.studentId for record in attendance_records]
    times = [record.time for record in attendance_records]
    l = len(attendance_records)
    return names, rolls, times, l

def add_attendance(name, period):
    try:
        student_name, student_id = name.split('_')
        student_id = int(student_id)
        current_date = datetime.now().date()
        current_time = datetime.now().time()
        student = Student.objects.get(collegeId=student_id, studentName=student_name)
        
        if student.status != 1:
            return {'status': 'error', 'message': f'Student {student_name} is inactive.'}

        if not StudentAttendance.objects.filter(studentId=student, date=current_date, period=period).exists():
            new_attendance = StudentAttendance(
                studentId=student,
                studentName=student.studentName,
                date=current_date,
                period=period,
                time=current_time,
                studentClass=student.studentClass
            )
            new_attendance.save()
            logging.info(f"Attendance added for {student_name} (ID: {student_id}).")
            return {'status': 'success', 'message': f'Attendance marked for {student_name}.'}
        return {'status': 'info', 'message': f'Attendance already marked for {student_name}.'}
    except Student.DoesNotExist:
        logging.error(f"Student {student_name}_{student_id} not found.")
        return {'status': 'error', 'message': f'Student {student_name}_{student_id} does not exist.'}
    except Exception as e:
        logging.error(f"Attendance error: {e}")
        return {'status': 'error', 'message': str(e)}

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)
    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)
    return userlist, names, rolls, l

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(os.path.join(duser, i))
    os.rmdir(duser)

