
import cv2
import os
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from datetime import date, datetime
from .utils import (
    extract_attendance,
    extract_faces,
    totalreg,
    datetoday2,
    getallusers,
    deletefolder,
    train_model,
    identify_face,
    add_attendance
)
from django.views.decorators.csrf import csrf_exempt
from .models import Student  # Import the Student model

nimgs = 10

# Home page
def home(request):
    try:
        names, rolls, times, l = extract_attendance()
        return render(request, 'home.html', {
            'names': names,
            'rolls': rolls,
            'times': times,
            'l': l,
            'totalreg': totalreg(),
            'datetoday2': datetoday2()
        })
    except Exception as e:
        print(f"Error loading home page: {e}")
        return HttpResponse("Error loading attendance data", status=500)


# List users page
def listusers(request):
    try:
        userlist, names, rolls, l = getallusers()
        return render(request, 'listusers.html', {
            'userlist': userlist,
            'names': names,
            'rolls': rolls,
            'l': l,
            'totalreg': totalreg(),
            'datetoday2': datetoday2()
        })
    except Exception as e:
        print(f"Error listing users: {e}")
        return HttpResponse("Error loading user list", status=500)

import json
import shutil

# Delete user functionality
@csrf_exempt
def deleteuser(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            data = json.loads(request.body)
            student_name = data.get('studentName')
            college_id = data.get('collegeId')

            if not student_name or not college_id:
                return JsonResponse({"error": "Both studentName and collegeId are required"}, status=400)

            # Generate the folder path to be deleted
            folder_path = f'static/faces/{student_name}_{college_id}'
            
            # Delete the folder if it exists
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)  # Delete folder and its contents

            # Delete the student record from the database
            deleted_count, _ = Student.objects.filter(studentName=student_name, collegeId=college_id).delete()

            if deleted_count == 0:
                return JsonResponse({"error": "No matching student found in the database"}, status=404)

            # If all face data is deleted, remove the trained model file
            if not os.listdir('static/faces/'):
                if os.path.exists('static/face_recognition_model.pkl'):
                    os.remove('static/face_recognition_model.pkl')

            # Retrain the model
            try:
                train_model()
            except Exception as e:
                print(f"Error retraining model: {e}")
                return JsonResponse({"error": "Student deleted, but model retraining failed"}, status=500)

            return JsonResponse({"message": "Student deleted successfully"}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            print(f"Error deleting user: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


import time  # Add this import at the top
import threading

# Global flag for stopping the process
stop_process_flag = False
# Start Face Recognition for attendance
@csrf_exempt
def start(request):
    try:
        global stop_process_flag
        stop_process_flag = False  # Reset stop flag at the start
        # Fetch today's date or get it from the request if provided
        data = json.loads(request.body)
        period = data.get('period', 1)
        # department = request.GET.get('department')
        # print(department)
        # classValue = request.GET.get('classValue')

        if 'date' in request.GET:
            attendance_date = request.GET['date']  # e.g., '2025-01-17'
            attendance_date = date.fromisoformat(attendance_date)  # Convert string to date
        else:
            attendance_date = date.today()  # Default to today's date

        # Extract attendance data for the given date
        # names, rolls, times, l = extract_attendance(attendance_date,period)

        # if 'face_recognition_model.pkl' not in os.listdir('static'):
        #     return render(request, 'attendance.html', {
        #         'names': names,
        #         'rolls': rolls,
        #         'times': times,
        #         'l': l,
        #         'totalreg': totalreg(),
        #         'datetoday2': datetoday2(),
        #         'mess': 'There is no trained model in the static folder. Please add a new face to continue.'
        #     })
                # Check for a trained model
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return render(request, 'attendance.html', {
                'mess': 'No trained model found. Please add a new face to continue.'
            })


        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot access the webcam")
            return HttpResponse("Cannot access the webcam", status=500)
        
        identified_person = None
        identified_start_time = None
        stable_threshold = 3  # Time in seconds for stability check

        while not stop_process_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame")
                break
            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                current_person = identify_face(face.reshape(1, -1))[0]
                # Stability check: Ensure the same face is identified for 3 seconds
                if current_person == identified_person:
                    if identified_start_time is None:
                        identified_start_time = time.time()
                    elif time.time() - identified_start_time >= stable_threshold:
                        # Mark attendance if the person is stable for 3 seconds
                        add_attendance(current_person,period)
                        cv2.putText(frame, f'{current_person} (Marked)', (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        identified_start_time = None  # Reset after marking attendance
                else:
                    identified_person = current_person
                    identified_start_time = None  # Reset timer if face changes
                # identified_person = identify_face(face.reshape(1, -1))[0]
                # add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return HttpResponse("Attendance process completed successfully.")
        # Update attendance list
        # names, rolls, times, l = extract_attendance()
        # return render(request, 'attendance.html', {
        #     'names': names,
        #     'rolls': rolls,
        #     'times': times,
        #     'l': l,
        #     'totalreg': totalreg(),
        #     'datetoday2': datetoday2()
        # })

    except Exception as e:
        print(f"Error in face recognition: {e}")
        return HttpResponse("Error in face recognition", status=500)

# Stop process endpoint
@csrf_exempt
def stop(request):
    global stop_process_flag
    stop_process_flag = True
    return HttpResponse("Process stopped successfully.")

import os
import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student
from .utils import extract_faces, train_model  # Assume utility functions are imported

@csrf_exempt
# Add a new student
def add(request):
    if request.method == 'POST':
        print(request.POST)  # Debugging line

        try:
            # Correct the way you extract values using `request.POST.get()`
            student_name = request.POST.get('studentName')
            print(student_name)
            collegeId = request.POST.get('collegeId')
            print(collegeId)
            dob = request.POST.get('dob', None)  # Date of birth (optional)
            print(dob)
            place = request.POST.get('place')  # Fixed typo from `POS.getT`
            print(place)
            department = request.POST.get('department')
            print(department)
            studentClass = request.POST.get('studentClass')
            print(studentClass)
            year_of_admission = request.POST.get('yearOfAdmission')
            password = request.POST.get('password')
            print(year_of_admission)
            print("hi")
            
            # Validate extracted fields
            if not all([student_name, collegeId, place, department, studentClass, year_of_admission,password]):
                return JsonResponse({'status': 'error', 'message': 'Missing required fields.'})
            print("hiii")
            # Check if the student already exists
            if Student.objects.filter(collegeId=collegeId).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': f"Student with college ID {collegeId} already exists."
                })

            # Prepare folder for saving images
            user_image_folder = f'static/faces/{student_name}_{collegeId}'
            print("hi2")
            if not os.path.isdir(user_image_folder):
                print("hi3")
                os.makedirs(user_image_folder)

            # Initialize variables for capturing images
            cap = cv2.VideoCapture(0)  # Open the webcam
            i, j = 0, 0
            nimgs = 20  # Number of images to capture
            saved_image_path = None  # To save the path of the first captured image

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                faces = extract_faces(frame)
                if len(faces) == 0:
                    print("No faces detected")
                    continue
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                    cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
                    if j % 5 == 0:  # Save every 5th frame
                        image_name = f'{student_name}_{i}.jpg'
                        image_path = f'{user_image_folder}/{image_name}'
                        cv2.imwrite(image_path, frame[y:y + h, x:x + w])
                        if saved_image_path is None:
                            saved_image_path = image_path  # Save the first image path
                        i += 1
                    j += 1
                if j == nimgs * 5:  # Stop after capturing required images
                    break
                cv2.imshow('Adding new User', frame)  # Show the webcam feed
                if cv2.waitKey(1) == 27:  # Exit if ESC key is pressed
                    break

            cap.release()
            cv2.destroyAllWindows()  # Close all OpenCV windows

            # Train the model after capturing images
            print('Training Model')
            train_model()

            # Save student details to the database
            new_student = Student(
                studentName=student_name,
                collegeId=collegeId,
                dob=dob,
                place=place,
                imagePath=saved_image_path,
                department=department,
                studentClass=studentClass,
                year_of_admission=year_of_admission,
                password=password
            )
            new_student.save()

            # Return success response
            return JsonResponse({
                'status': 'success',
                'message': 'Student added and model trained successfully.',
                'imagePath': saved_image_path
            })

        except Exception as e:
            print(f"Error occurred: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})

    # Return an error message if the request method is not POST
    return JsonResponse({'status': 'error', 'message': 'Invalid request method. Only POST is allowed.'})


# ============student details of a particular class==================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student
import json

# Updated endpoint to fetch all students (active and inactive)
def get_students_by_department_and_class(request):
    department = request.GET.get('department')
    student_class = request.GET.get('class')

    if not department or not student_class:
        return JsonResponse({"error": "Both 'department' and 'class' query parameters are required."}, status=400)

    # Query all students regardless of status
    students = Student.objects.filter(
        department=department,
        studentClass=student_class
    ).values(
        'studentId', 'studentName', 'collegeId', 'dob', 'place', 'imagePath',
        'department', 'studentClass', 'year_of_admission', 'status'
    )

    return JsonResponse(list(students), safe=False)

# New endpoint to update student status
@csrf_exempt
def update_student_status(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('studentId')
            new_status = data.get('status')  # Expecting 0 or 1

            if student_id is None or new_status not in [0, 1]:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student ID and valid status (0 or 1) are required.'
                }, status=400)

            try:
                student = Student.objects.get(studentId=student_id)
                student.status = new_status
                student.save()
                return JsonResponse({
                    'status': 'success',
                    'message': f'Student status updated to {"active" if new_status == 1 else "inactive"}.',
                    'studentId': student.studentId,
                    'newStatus': student.status
                }, status=200)
            except Student.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student not found.'
                }, status=404)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)


# ================================================================================================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_object_or_404
from .models import StudentAttendance, Student
import json

@csrf_exempt
def get_student_attendance(request):
    if request.method == "POST":
        try:
            # Parse JSON data from the request
            data = json.loads(request.body)
            department = data.get('department')
            student_class = data.get('class')
            period = data.get('period')

            if not department or not student_class or not period:
                return JsonResponse({"error": "Missing required parameters: department, class, or period."}, status=400)

            # Filter StudentAttendance based on the input parameters
            attendance_records = StudentAttendance.objects.filter(
                studentId__department=department,
                studentClass=student_class,
                period=period
            ).select_related('studentId')

            # Build the response
            results = []
            for record in attendance_records:
                results.append({
                    "attendanceId": record.attendanceId,
                    "studentId": record.studentId.studentId,
                    "studentName": record.studentId.studentName,
                    "department": record.studentId.department,
                    "studentClass": record.studentId.studentClass,
                    "date": record.date.strftime('%Y-%m-%d'),
                    "time": record.time.strftime('%H:%M:%S'),
                    "period": record.period,
                })

            return JsonResponse({"data": results}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid HTTP method. Use POST."}, status=405)


# ==================================================teacher sign up==================================
# your_app/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Teacher

@csrf_exempt
def signup(request):
    if request.method == 'POST':
        print("Received POST data:", request.POST)  # Debug incoming data
        try:
            teacher_id = request.POST.get('teacherId')
            name = request.POST.get('name')
            role = request.POST.get('role')
            department = request.POST.get('department')
            password = request.POST.get('password')
            print("Password:", password)

            required_fields = [teacher_id, name, role, department, password]
            if not all(required_fields):
                return JsonResponse({
                    'status': 'error',
                    'message': 'All fields are required.'
                }, status=400)

            if Teacher.objects.filter(teacherId=teacher_id).exists():
                return JsonResponse({
                    'status': 'error',
                    'message': f"Teacher with ID {teacher_id} already exists."
                }, status=400)

            new_teacher = Teacher(
                teacherId=teacher_id,
                name=name,
                role=role,
                department=department,
            )
            new_teacher.set_password(password)
            new_teacher.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Teacher registered successfully.',
                'teacherId': new_teacher.teacherId
            }, status=201)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)


# ===================login===========================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, Teacher

@csrf_exempt
def login(request):
    if request.method == 'POST':
        print("Received POST data:", request.POST)  # Debug incoming data
        try:
            user_id = request.POST.get('userId')
            password = request.POST.get('password')
            user_type = request.POST.get('userType')
            print("Extracted:", user_id, password, user_type)  # Additional debug

            if not user_id or not password or not user_type:
                return JsonResponse({
                    'status': 'error',
                    'message': 'User ID, password, and user type are required.'
                }, status=400)

            if user_type.lower() == 'student':
                try:
                    student = Student.objects.get(collegeId=user_id)
                    # Directly compare the plain text password
                    if student.password == password:  # No hashing, direct comparison
                        if student.status == 1:
                            return JsonResponse({
                                'status': 'success',
                                'message': 'Login successful.',
                                'userType': 'student',
                                'studentId': student.studentId,  # Explicitly named
                                'name': student.studentName
                            }, status=200)
                        else:
                            return JsonResponse({
                                'status': 'error',
                                'message': 'Account is inactive. Please contact administration.'
                            }, status=403)
                    else:
                        return JsonResponse({
                            'status': 'error',
                            'message': 'Invalid credentials.'
                        }, status=401)
                except Student.DoesNotExist:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Student not found.'
                    }, status=401)

            elif user_type.lower() == 'teacher':
                try:
                    teacher = Teacher.objects.get(teacherId=user_id)
                    if teacher.check_password(password):  # Keep hashing for teacher
                        if teacher.status == 1:
                            return JsonResponse({
                                'status': 'success',
                                'message': 'Login successful.',
                                'userType': 'teacher',
                                'teacherId': teacher.teacherId,  # Explicitly named
                                'name': teacher.name
                            }, status=200)
                        else:
                            return JsonResponse({
                                'status': 'error',
                                'message': 'Account is inactive. Please contact administration.'
                            }, status=403)
                    else:
                        return JsonResponse({
                            'status': 'error',
                            'message': 'Invalid credentials.'
                        }, status=401)
                except Teacher.DoesNotExist:
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Teacher not found.'
                    }, status=401)

            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid user type. Must be "student" or "teacher".'
                }, status=400)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)
# ===================resent students===================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, StudentAttendance
import json
from datetime import datetime

@csrf_exempt
def get_student_attendance(request):
    if request.method == 'POST':
        print("Received POST data:", request.body)  # Debug incoming data
        try:
            # Parse JSON body
            data = json.loads(request.body)
            department = data.get('department')
            student_class = data.get('class')
            period = data.get('period')
            date_str = data.get('date')  # Expected format: "YYYY-MM-DD"

            # Validate inputs
            if not all([department, student_class, period, date_str]):
                return JsonResponse({
                    'status': 'error',
                    'message': 'All fields (department, class, period, date) are required.'
                }, status=400)

            # Convert date string to datetime.date object
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid date format. Use YYYY-MM-DD.'
                }, status=400)

            # Fetch attendance records
            attendance_records = StudentAttendance.objects.filter(
                studentClass=student_class,
                period=period,
                date=date
            ).select_related('studentId')  # Join with Student table

            # Filter students by department
            attendance_records = [
                record for record in attendance_records
                if record.studentId.department == department
            ]

            if not attendance_records:
                return JsonResponse({
                    'status': 'success',
                    'message': 'No attendance records found.',
                    'data': []
                }, status=200)

            # Format response data
            response_data = [
                {
                    'attendanceId': record.attendanceId,
                    'studentId': record.studentId.studentId,
                    'studentName': record.studentId.studentName,  # From Student model
                    'department': record.studentId.department,
                    'studentClass': record.studentClass,
                    'date': record.date.strftime('%Y-%m-%d'),
                    'time': record.time.strftime('%H:%M:%S'),
                    'period': record.period,
                }
                for record in attendance_records
            ]

            return JsonResponse({
                'status': 'success',
                'message': 'Attendance records fetched successfully.',
                'data': response_data
            }, status=200)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)

# ==================delete an attendence=======================
@csrf_exempt
def unmark_attendance(request):
    if request.method == 'POST':
        print("Received POST data for unmark:", request.body)
        try:
            data = json.loads(request.body)
            attendance_id = data.get('attendanceId')

            if not attendance_id:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Attendance ID is required.'
                }, status=400)

            # Delete the attendance record
            try:
                attendance = StudentAttendance.objects.get(attendanceId=attendance_id)
                attendance.delete()
                return JsonResponse({
                    'status': 'success',
                    'message': 'Attendance unmarked successfully.'
                }, status=200)
            except StudentAttendance.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Attendance record not found.'
                }, status=404)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)

# =================absent===================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, StudentAttendance
import json
from datetime import datetime


# New endpoint for absent students
@csrf_exempt
def get_absent_students(request):
    if request.method == 'POST':
        print("Received POST data for absent students:", request.body)
        try:
            data = json.loads(request.body)
            department = data.get('department')
            student_class = data.get('class')
            period = data.get('period')
            date_str = data.get('date')

            if not all([department, student_class, period, date_str]):
                return JsonResponse({
                    'status': 'error',
                    'message': 'All fields (department, class, period, date) are required.'
                }, status=400)

            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid date format. Use YYYY-MM-DD.'
                }, status=400)

            # Get all students in the department and class
            all_students = Student.objects.filter(
                department=department,
                studentClass=student_class,
                status=1  # Only active students
            )

            # Get students present on this date, period, and class
            present_students = StudentAttendance.objects.filter(
                studentClass=student_class,
                period=period,
                date=date
            ).values_list('studentId__studentId', flat=True)

            # Filter out students who are present
            absent_students = all_students.exclude(studentId__in=present_students)

            if not absent_students:
                return JsonResponse({
                    'status': 'success',
                    'message': 'No absent students found.',
                    'data': []
                }, status=200)

            # Format response data
            response_data = [
                {
                    'studentId': student.studentId,
                    'studentName': student.studentName,
                    'department': student.department,
                    'studentClass': student.studentClass,
                }
                for student in absent_students
            ]

            return JsonResponse({
                'status': 'success',
                'message': 'Absent students fetched successfully.',
                'data': response_data
            }, status=200)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)

# New endpoint to mark attendance
@csrf_exempt
def mark_attendance(request):
    if request.method == 'POST':
        print("Received POST data for marking attendance:", request.body)
        try:
            data = json.loads(request.body)
            student_id = data.get('studentId')
            period = data.get('period')
            date_str = data.get('date')
            student_class = data.get('studentClass')

            if not all([student_id, period, date_str, student_class]):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student ID, period, date, and class are required.'
                }, status=400)

            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid date format. Use YYYY-MM-DD.'
                }, status=400)

            # Fetch the student
            try:
                student = Student.objects.get(studentId=student_id)
            except Student.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Student not found.'
                }, status=404)

            # Create attendance record
            attendance = StudentAttendance(
                studentId=student,
                studentName=student.studentName,  # Redundant but kept for consistency
                period=period,
                date=date,
                time=datetime.now().time(),  # Current time
                studentClass=student_class
            )
            attendance.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Attendance marked successfully.',
                'attendanceId': attendance.attendanceId
            }, status=201)

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'An error occurred: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method. Only POST is allowed.'
    }, status=405)


# ========================edit details===================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student
import json

# Fetch students by department and class (unchanged from previous)
def get_students_by_department_and_class(request):
    department = request.GET.get('department')
    student_class = request.GET.get('class')

    if not department or not student_class:
        return JsonResponse({"error": "Both 'department' and 'class' query parameters are required."}, status=400)

    students = Student.objects.filter(
        department=department,
        studentClass=student_class
    ).values(
        'studentId', 'studentName', 'collegeId', 'dob', 'place', 'imagePath',
        'department', 'studentClass', 'year_of_admission', 'status'
    )

    return JsonResponse(list(students), safe=False)

# Update student details
@csrf_exempt
def update_student(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('studentId')
            if not student_id:
                return JsonResponse({'status': 'error', 'message': 'Student ID is required.'}, status=400)

            student = Student.objects.get(studentId=student_id)
            # Update fields if provided in the request
            student.studentName = data.get('studentName', student.studentName)
            student.collegeId = data.get('collegeId', student.collegeId)
            student.dob = data.get('dob', student.dob)
            student.place = data.get('place', student.place)
            student.imagePath = data.get('imagePath', student.imagePath)
            student.department = data.get('department', student.department)
            student.studentClass = data.get('studentClass', student.studentClass)
            student.year_of_admission = data.get('year_of_admission', student.year_of_admission)
            student.status = data.get('status', student.status)
            student.save()

            return JsonResponse({
                'status': 'success',
                'message': 'Student details updated successfully.',
                'student': {
                    'studentId': student.studentId,
                    'studentName': student.studentName,
                    'collegeId': student.collegeId,
                    'dob': str(student.dob) if student.dob else None,
                    'place': student.place,
                    'imagePath': student.imagePath,
                    'department': student.department,
                    'studentClass': student.studentClass,
                    'year_of_admission': student.year_of_admission,
                    'status': student.status
                }
            }, status=200)

        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Student not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Error updating student: {str(e)}'}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import StudentAttendance, Student  # Assume a Task and LeaveRequest model too
import json
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, StudentAttendance, Task, LeaveRequest
import json
from datetime import datetime

# Get student attendance
@csrf_exempt
def get_student_attendances(request):
    if request.method == 'GET':
        student_id = request.GET.get('studentId')
        if not student_id:
            return JsonResponse({'status': 'error', 'message': 'Student ID is required.'}, status=400)
        try:
            attendance = StudentAttendance.objects.filter(studentId__studentId=student_id).values(
                'date', 'period', 'time'
            )
            data = [{'date': str(a['date']), 'period': a['period'], 'status': 'Present'} for a in attendance]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Get student tasks
@csrf_exempt
def get_student_tasks(request):
    if request.method == 'GET':
        student_id = request.GET.get('studentId')
        if not student_id:
            return JsonResponse({'status': 'error', 'message': 'Student ID is required.'}, status=400)
        try:
            tasks = Task.objects.filter(studentId__studentId=student_id).values(
                'taskId', 'title', 'dueDate', 'status'
            )
            data = [{'id': t['taskId'], 'title': t['title'], 'dueDate': str(t['dueDate']), 'status': t['status']} for t in tasks]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Submit leave or attendance review request
@csrf_exempt
def submit_request(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('studentId')
            request_type = data.get('requestType')  # 'LEAVE' or 'ATTENDANCE_REVIEW'
            date = data.get('date')
            reason = data.get('reason')

            if not all([student_id, request_type, date, reason]):
                return JsonResponse({'status': 'error', 'message': 'All fields are required.'}, status=400)

            student = Student.objects.get(studentId=student_id)
            leave_request = LeaveRequest(
                studentId=student,
                requestType=request_type,
                date=date,
                reason=reason
            )
            leave_request.save()
            return JsonResponse({'status': 'success', 'message': f'{request_type} request submitted successfully.'}, status=200)
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Student not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


# =================leave request====================================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import LeaveRequest
import json

# Fetch requests based on filter
@csrf_exempt
def get_teacher_requests(request):
    if request.method == 'GET':
        filter_type = request.GET.get('filter', 'all')  # 'all', 'pending', 'approved'
        try:
            if filter_type == 'pending':
                requests = LeaveRequest.objects.filter(status='Pending')
            elif filter_type == 'approved':
                requests = LeaveRequest.objects.filter(status='Approved')
            else:  # 'all'
                requests = LeaveRequest.objects.all()

            data = [
                {
                    'requestId': r.requestId,
                    'studentName': r.studentId.studentName,
                    'requestType': r.requestType,
                    'date': str(r.date),
                    'reason': r.reason,
                    'status': r.status,
                    'submittedDate': str(r.submittedDate),
                    'cause': r.cause,
                }
                for r in requests
            ]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Update request status with cause
@csrf_exempt
def update_request_status(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            request_id = data.get('requestId')
            new_status = data.get('status')  # 'Approved' or 'Rejected'
            cause = data.get('cause')

            if not all([request_id, new_status, cause]):
                return JsonResponse({'status': 'error', 'message': 'Request ID, status, and cause are required.'}, status=400)

            if new_status not in ['Approved', 'Rejected']:
                return JsonResponse({'status': 'error', 'message': 'Invalid status. Must be "Approved" or "Rejected".'}, status=400)

            leave_request = LeaveRequest.objects.get(requestId=request_id)
            leave_request.status = new_status
            leave_request.cause = cause
            leave_request.save()

            return JsonResponse({'status': 'success', 'message': f'Request {new_status} successfully.'}, status=200)
        except LeaveRequest.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Request not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


    # ==================================taskds========================================
    from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, Task
import json

# Fetch all students (for individual selection)
@csrf_exempt
def get_all_students(request):
    if request.method == 'GET':
        try:
            students = Student.objects.filter(status=1).values('studentId', 'studentName', 'collegeId')
            data = [{'studentId': s['studentId'], 'studentName': s['studentName'], 'collegeId': s['collegeId']} for s in students]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Fetch students by department and class
@csrf_exempt
def get_students_by_department_class(request):
    if request.method == 'GET':
        department = request.GET.get('department')
        student_class = request.GET.get('studentClass')
        if not department or not student_class:
            return JsonResponse({'status': 'error', 'message': 'Department and class are required.'}, status=400)
        try:
            students = Student.objects.filter(department=department, studentClass=student_class, status=1).values('studentId')
            data = [s['studentId'] for s in students]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Assign task
@csrf_exempt
def assign_task(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            title = data.get('title')
            description = data.get('description')
            due_date = data.get('dueDate')
            student_ids = data.get('studentIds')  # List of student IDs

            if not all([title, due_date, student_ids]):
                return JsonResponse({'status': 'error', 'message': 'Title, due date, and student IDs are required.'}, status=400)

            tasks = []
            for student_id in student_ids:
                student = Student.objects.get(studentId=student_id)
                task = Task(
                    studentId=student,
                    title=title,
                    description=description,
                    dueDate=due_date,
                    status='Pending'
                )
                tasks.append(task)
            Task.objects.bulk_create(tasks)  # Efficiently create multiple tasks

            return JsonResponse({'status': 'success', 'message': 'Task assigned successfully.'}, status=200)
        except Student.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'One or more students not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


# ==============================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Student, LeaveRequest
import json

# Fetch requests submitted by a student
@csrf_exempt
def get_student_requests(request):
    if request.method == 'GET':
        student_id = request.GET.get('studentId')
        if not student_id:
            return JsonResponse({'status': 'error', 'message': 'Student ID is required.'}, status=400)
        try:
            requests = LeaveRequest.objects.filter(studentId__studentId=student_id).values(
                'requestId', 'requestType', 'date', 'reason', 'status', 'submittedDate', 'cause'
            )
            data = [
                {
                    'requestId': r['requestId'],
                    'requestType': r['requestType'],
                    'date': str(r['date']),
                    'reason': r['reason'],
                    'status': r['status'],
                    'submittedDate': str(r['submittedDate']),
                    'cause': r['cause'],
                }
                for r in requests
            ]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


# ==========teacher details===========================
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Teacher
import json

# Fetch all teachers
@csrf_exempt
def get_all_teachers(request):
    if request.method == 'GET':
        try:
            teachers = Teacher.objects.all().values('teacherId', 'name', 'status')
            data = [
                {
                    'teacherId': t['teacherId'],
                    'name': t['name'],
                    'status': t['status'],
                }
                for t in teachers
            ]
            return JsonResponse({'status': 'success', 'data': data}, status=200)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Update teacher status (active/inactive)
@csrf_exempt
def update_teacher_status(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            teacher_id = data.get('teacherId')
            status = data.get('status')  # 1 for active, 0 for inactive

            if not teacher_id or status not in [0, 1]:
                return JsonResponse({'status': 'error', 'message': 'Teacher ID and valid status (0 or 1) are required.'}, status=400)

            teacher = Teacher.objects.get(teacherId=teacher_id)
            teacher.status = status
            teacher.save()
            return JsonResponse({'status': 'success', 'message': f'Teacher status updated to {"active" if status == 1 else "inactive"}.'}, status=200)
        except Teacher.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Teacher not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Delete teacher
@csrf_exempt
def delete_teacher(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            teacher_id = data.get('teacherId')

            if not teacher_id:
                return JsonResponse({'status': 'error', 'message': 'Teacher ID is required.'}, status=400)

            teacher = Teacher.objects.get(teacherId=teacher_id)
            teacher.delete()
            return JsonResponse({'status': 'success', 'message': 'Teacher deleted successfully.'}, status=200)
        except Teacher.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Teacher not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)

# Update teacher details
@csrf_exempt
def update_teacher_details(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            teacher_id = data.get('teacherId')
            name = data.get('name')
            password = data.get('password')

            if not teacher_id or not name:
                return JsonResponse({'status': 'error', 'message': 'Teacher ID and name are required.'}, status=400)

            teacher = Teacher.objects.get(teacherId=teacher_id)
            teacher.name = name
            if password:  # Update password only if provided
                teacher.set_password(password)
            teacher.save()
            return JsonResponse({'status': 'success', 'message': 'Teacher details updated successfully.'}, status=200)
        except Teacher.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Teacher not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)