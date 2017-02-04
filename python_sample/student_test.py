class Student:
   def __init__(self, name):
       self.name = name
       self.attend = 0
       self.grades = []
       print("Hi ! My name is {0}".format(self.name))
   def addGrade(self, grade):
       self.grades.append(grade)
   def attendDay(self):
       self.attend+=1
   def getAverage(self):
       return sum(self.grades)/len(self.grades)

	
