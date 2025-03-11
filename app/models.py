from django.db import models

from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class Student(models.Model):
    studentId = models.AutoField(primary_key=True)  # Primary key
    studentName = models.CharField(max_length=100)
    collegeId = models.CharField(max_length=100, unique=True)  # Unique college ID
    dob = models.DateField(null=True)
    place = models.CharField(max_length=100)  # Place of residence
    imagePath = models.CharField(max_length=500, null=True)
    department = models.CharField(max_length=100)  # Department name
    studentClass = models.CharField(max_length=50)  # Class (e.g., "First Year")
    year_of_admission = models.PositiveIntegerField()  # Year of admission
    password = models.CharField(max_length=128)  # Store hashed password
    status = models.IntegerField(default=0)

    class Meta:
        ordering = ['studentName']  # Order by studentName in ascending order

    def __str__(self):
        return f"{self.studentName} ({self.studentId})"

    def set_password(self, raw_password):
        """Set the hashed password."""
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        """Check if the provided password matches the stored hash."""
        return check_password(raw_password, self.password)



# ========================================================================
class StudentAttendance(models.Model):
    attendanceId = models.AutoField(primary_key=True)  # Primary key
    studentId = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendances')
    studentName = models.CharField(max_length=100)
    period=models.CharField(max_length=100,default=0)
    date = models.DateField()  # Date of attendance
    time = models.TimeField()  # Time of attendance
    studentClass = models.CharField(max_length=50)  # Class (e.g., "10th Grade")

    class Meta:
        ordering = ['-date', '-time']  # Order by date and time (descending)

    def __str__(self):
        return f"Attendance {self.attendanceId} for {self.studentName.studentName} on {self.date} at {self.time}"
    

    # ====================================================
    from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class Teacher(models.Model):
    teacherId = models.CharField(max_length=50, primary_key=True)  # Primary key, not auto-incremented
    name = models.CharField(max_length=100)
    role = models.CharField(max_length=50)
    department = models.CharField(max_length=100)
    password = models.CharField(max_length=128)  # Hashed password
    created = models.DateTimeField(auto_now_add=True)  # Automatically set on creation
    updated = models.DateTimeField(auto_now=True)  # Automatically updated on save
    status = models.IntegerField(default=0)  # 0: inactive, 1: active, etc.

    class Meta:
        ordering = ['name']  # Order by name in ascending order

    def __str__(self):
        return f"{self.name} ({self.teacherId})"

    def set_password(self, raw_password):
        """Set the hashed password."""
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        """Check if the provided password matches the stored hash."""
        return check_password(raw_password, self.password)
    

# New Task model
class Task(models.Model):
    taskId = models.AutoField(primary_key=True)
    studentId = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='tasks')
    title = models.CharField(max_length=200)
    description = models.TextField(null=True, blank=True)
    dueDate = models.DateField()
    status = models.CharField(max_length=50, choices=(('Pending', 'Pending'), ('Completed', 'Completed')), default='Pending')
    assignedDate = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} for {self.studentId.studentName}"

# New LeaveRequest model (for leave and attendance review)
class LeaveRequest(models.Model):
    REQUEST_TYPES = (
        ('LEAVE', 'Leave'),
        ('ATTENDANCE_REVIEW', 'Attendance Review'),
    )
    
    requestId = models.AutoField(primary_key=True)
    studentId = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='leave_requests')
    requestType = models.CharField(max_length=20, choices=REQUEST_TYPES)
    date = models.DateField()  # Date of leave or attendance to review
    reason = models.TextField()
    status = models.CharField(max_length=50, choices=(('Pending', 'Pending'), ('Approved', 'Approved'), ('Rejected', 'Rejected')), default='Pending')
    submittedDate = models.DateField(auto_now_add=True)
    cause = models.TextField(null=True, blank=True)  # New field for approval/rejection reason

    def __str__(self):
        return f"{self.requestType} request {self.requestId} by {self.studentId.studentName}"