from django.urls import path
from . import views
from .views import signup,login,get_student_attendance,unmark_attendance,get_absent_students,mark_attendance,update_student_status,update_student,get_student_tasks,submit_request,get_student_attendances
from .views import get_teacher_requests,update_request_status,get_all_students,get_students_by_department_class,assign_task,get_student_requests,update_teacher_details,get_all_teachers,update_teacher_status,delete_teacher
urlpatterns = [
    path('markattendence/', views.start, name='start'),
    path('students-0f-dep-class/', views.get_students_by_department_and_class, name='get_students'),
    path('deleteStudent/',views.deleteuser,name='deleteuser'),
    path('addStudent/',views.add,name='add'),
    path('stop/',views.stop,name='stop'),
    path('get-student-attendance/', views.get_student_attendance, name='get_student_attendance'),
    path('teachersignup/',signup, name='teacher_signup'),
    path('login/',login,name="login"),
    path('get-student-attendance/', get_student_attendance, name='get_student_attendance'),
    path('unmark-attendance/', unmark_attendance, name='unmark_attendance'),
    path('get-absent-students/', get_absent_students, name='get_absent_students'),
    path('mark-attendance/', mark_attendance, name='mark_attendance'),
    path('update-status/', update_student_status, name='update_student_status'),
    path('update/', update_student, name='update_student'),
    path('attendance/', get_student_attendances, name='get_student_attendance'),
    path('tasks/', get_student_tasks, name='get_student_tasks'),
    path('submit-request/', submit_request, name='submit_request'),
    path('requests/', get_teacher_requests, name='get_teacher_requests'),
    path('update-request/', update_request_status, name='update_request_status'),
    path('students/', get_all_students, name='get_all_students'),
    path('students-by-department-class/', get_students_by_department_class, name='get_students_by_department_class'),
    path('assign-task/', assign_task, name='assign_task'),
    path('requests/', get_student_requests, name='get_student_requests'),
    path('teacher/all/', get_all_teachers, name='get_all_teachers'),
    path('teacher/update-status/', update_teacher_status, name='update_teacher_status'),
    path('teacher/delete/', delete_teacher, name='delete_teacher'),
    path('teacher/update-details/', update_teacher_details, name='update_teacher_details'),

]
