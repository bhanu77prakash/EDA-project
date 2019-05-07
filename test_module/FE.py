import os
import random
import csv
import sys
import PyQt4
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import tkinter as tk
from tkinter import filedialog
import subprocess
import pickle
import run_model


class Label(QWidget):
	def __init__(self, parent=None):
		QWidget.__init__(self, parent=parent)
		self.p = None

	def setPixmap(self, p):
		self.p = p
		painter = QPainter(self)
		painter.setRenderHint(QPainter.SmoothPixmapTransform)
		painter.drawPixmap(self.rect(), self.p)


	def paintEvent(self,e):
		# if self.p:
			# print(e)
		painter = QPainter(self)
		painter.setRenderHint(QPainter.SmoothPixmapTransform)
		painter.drawPixmap(self.rect(), self.p)

	def mousePressEvent(self, QMouseEvent):
		print (QMouseEvent.pos())
		cursor =QCursor()
		print (cursor.pos())
		self.paintEvent(QMouseEvent)


class combodemo(QWidget):
	# def run_image(self, file_path):
	# 	image = Image.open(file_path)
	# 	image_file = file_path.rpartition('/')[-1]
	# 	image_copy = image
	# 	if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
	# 		resized_image = image.resize(tuple(reversed(self.model_image_size)), Image.BICUBIC)
	# 		image_data = np.array(resized_image, dtype='float32')
	# 	else:
	# 		new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
	# 		resized_image = image.resize(new_image_size, Image.BICUBIC)
	# 		image_data = np.array(resized_image, dtype='float32')
	# 		print(image_data.shape)
	# 	image_data_copy = image_data
	# 	image_data /= 255.
	# 	image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
	# 	thickness = (image.size[0] + image.size[1]) // 300
	# 	# left=356
	# 	# top=567
	# 	# h=127
	# 	# w=82
	# 	# image_file
	# 	with open("finalist.csv", 'r') as csvfile:
	# 		csvreader = csv.reader(csvfile)
	# 		for row in csvreader:
	# 			if(row[0]==image_file):
	# 				left=int(row[7])
	# 				top=int(row[8])
	# 				h=int(row[9])
	# 				w=int(row[10])
	# 				self.disease=row[6]					
	# 				break
				
	# 	im = image.convert('LA')
	# 	fig,ax = plt.subplots(1)
	# 	ax.imshow(im)    
	# 	for i in range(int(thickness/4)):
	# 		rect = patches.Rectangle((left + i, top + i), h, w,edgecolor='r',facecolor='none')
	# 		ax.add_patch(rect)
	# 	plt.savefig(os.path.join(self.output_path, "result.jpg"))

	# def run_image_real(self, file_path):
	# 	image = Image.open(file_path)
	# 	image_file = file_path.rpartition('/')[-1]
	# 	image_copy = image
	# 	if self.is_fixed_size:  # TODO: When resizing we can use minibatch input.
	# 		resized_image = image.resize(tuple(reversed(self.model_image_size)), Image.BICUBIC)
	# 		image_data = np.array(resized_image, dtype='float32')
	# 	else:
	# 		new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
	# 		resized_image = image.resize(new_image_size, Image.BICUBIC)
	# 		image_data = np.array(resized_image, dtype='float32')
	# 		print(image_data.shape)
	# 	image_data_copy = image_data
	# 	image_data /= 255.
	# 	image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
	# 	thickness = (image.size[0] + image.size[1]) // 300
	# 	left=0
	# 	top=0
	# 	h=0
	# 	w=0
	# 	# string disease
	# 	# image_file
	# 	with open("finalist.csv", 'r') as csvfile:
	# 		csvreader = csv.reader(csvfile)
	# 		for row in csvreader:
	# 			if(row[0]==image_file):
	# 				left=int(row[2])
	# 				top=int(row[3])
	# 				h=int(row[4])
	# 				w=int(row[5])
	# 				self.disease=row[1]
	# 				break

	# 	im = image.convert('LA')
	# 	fig,ax = plt.subplots(1)
	# 	ax.imshow(im)    
	# 	for i in range(int(thickness/4)):
	# 		rect = patches.Rectangle((left + i, top + i), h, w,edgecolor='b',facecolor='none')
	# 		ax.add_patch(rect)
		
	# 	left=0
	# 	top=0
	# 	h=0
	# 	w=0
	# 	with open("finalist.csv", 'r') as csvfile:
	# 		csvreader = csv.reader(csvfile)
	# 		for row in csvreader:
	# 			if(row[0]==image_file):
	# 				left=int(row[7])
	# 				top=int(row[8])
	# 				h=int(row[9])
	# 				w=int(row[10])
	# 				self.disease_real=row[6]					
	# 				break

	# 	for i in range(int(thickness/4)):
	# 		rect = patches.Rectangle((left + i, top + i), h, w,edgecolor='r',facecolor='none')
	# 		ax.add_patch(rect)
		
	# 	plt.savefig(os.path.join(self.output_path, "result.jpg"))	

	# def _main(self,args, file_path):
	# 	self.test_path = os.path.expanduser(args.test_path)
	# 	self.output_path = os.path.expanduser(args.output_path)
	# 	if not os.path.exists(self.output_path):
	# 		print('Creating output path {}'.format(self.output_path))
	# 		os.mkdir(self.output_path)
	# 	self.model_image_size = (608,608)
	# 	self.is_fixed_size = self.model_image_size != (None, None)
	# 	self.run_image(file_path)

	# def _main_real(self,args, file_path):
	# 	self.test_path = os.path.expanduser(args.test_path)
	# 	self.output_path = os.path.expanduser(args.output_path)
	# 	if not os.path.exists(self.output_path):
	# 		print('Creating output path {}'.format(self.output_path))
	# 		os.mkdir(self.output_path)
	# 	self.model_image_size = (608,608)
	# 	self.is_fixed_size = self.model_image_size != (None, None)
	# 	self.run_image_real(file_path)	
		

	def __init__(self, parent = None):
		super(combodemo, self).__init__(parent)
      
		vbox=QHBoxLayout()	
		vbox1=QVBoxLayout()
		vbox2=QVBoxLayout()
		hbox=QVBoxLayout()

		vbox01=QVBoxLayout()
		vbox02=QVBoxLayout()
		
		self.l0=QLabel()		
		self.l0.setText("Image of X-Ray ")
		self.l0.setFont(QFont('SansSerif', 30))
		self.l0.setStyleSheet("QLabel { background-color : black; color : white; }");
		self.l0.setFixedHeight(50)
		self.l0.setAlignment(Qt.AlignCenter)

		self.l01=QLabel()		
		self.l01.setText("Image of X-Ray ")
		self.l01.setFont(QFont('SansSerif', 30))
		self.l01.setStyleSheet("QLabel { background-color : black; color : white; }");
		self.l01.setFixedHeight(50)
		self.l01.setAlignment(Qt.AlignCenter)
		

		self.l1=Label(self)
		self.l1.setPixmap(QPixmap())
		
		# self.l2=Label(self)
		# self.l2.setPixmap(QPixmap())
		
		self.b1 = QPushButton("Upload Image")
		self.b1.setCheckable(True)
		self.b1.clicked.connect(self.btnstate1)	
		
		self.b3 = QPushButton("RUN")
		self.b3.setCheckable(True)
		self.b3.clicked.connect(self.btnstate3)	
		
		# self.b2 = QPushButton("Test DISEASE")
		# self.b2.setCheckable(True)
		# self.b2.clicked.connect(self.btnstate2)
		
		self.b2=QLineEdit("Enter your question here !!")
   		
		self.b4 = QPushButton("Exit")
		self.b4.setCheckable(True)
		self.b4.clicked.connect(self.btnstate4)
	

		self.a1=QLabel(self)
		self.a2=QLabel(self)
		#self.a1.setPixmap(QPixmap())
		#self.a2.setPixmap(QPixmap())
		hbox.addWidget(self.a1)
		hbox.addWidget(self.a2)		


		vbox1.addWidget(self.b1)
		# vbox1.addWidget(self.l0)
		vbox1.addWidget(self.l1)
		vbox1.addWidget(self.b3)	
		vbox01.addWidget(self.b2)
		# vbox2.addWidget(self.l01)		
		# vbox2.addWidget(self.l2)
		vbox02.addWidget(self.b4)	
		
		vbox2.addLayout(vbox01)
		vbox2.addLayout(hbox)
		vbox2.addLayout(vbox02)
		
		
		vbox.addLayout(vbox1,1)
		vbox.addLayout(vbox2,1.5)
		
		self.setLayout(vbox)
		self.setWindowTitle("Model Testing")
		self.showMaximized()
		self.file_path = None
		# self.first = True
		# self.second=False
		# self.disease=""
		# self.disease_real="";
		
	def btnstate3(self):
		self.ques=self.b2.text()
		if(self.ques==''):
			print("error")
		else:
			print("run code")
			print(self.ques)	
		
		self.ques='"'+self.ques+'"'		
		#self.ques='"'+'Does the small sphere have the same color as the cube left of the gray cube?'+'"'

		command='python run_model.py'+' --image ' + self.file_path + ' --question ' + self.ques
		os.system(command)
		with open('ans.pkl', 'rb') as f:
    			self.ans = pickle.load(f)
		with open('lis.pkl', 'rb') as f:
    			self.lis = pickle.load(f)
		print(self.ans)
		print(self.lis)
		self.line=''
		for a in self.lis:
			self.line=self.line+a+"\n"
		self.a1.setText(self.line)
		self.a2.setText(self.ans)
		self.a1.setFont(QFont('SansSerif', 15))
		self.a2.setFont(QFont('SansSerif', 40))
		# self.second=True
		# if(self.first):
		# 	self._main(parser.parse_args(), self.file_path)
		# else:
		# 	self.run_image(self.file_path)	
		# self.l2.setPixmap(QPixmap("/home/rpg/Documents/tch/ray/images/out/result.jpg"))
		# self.l0.setText("Disease predicted :"+self.disease)		
		# print ("Button 3 clicked")

	def btnstate1(self):
		# self.l0.setText("Image of X-Ray")
		# self.l01.setText("Image of X-Ray")
		# self.l2.setPixmap(QPixmap("/home/rpg/Documents/tch/ray/1.png"))
		self.l1.setPixmap(QPixmap("/home/rpg/Documents/tch/ray/1.png"))

		root = tk.Tk()
		root.withdraw()
		self.file_path = filedialog.askopenfilename()
		self.l1.setPixmap(QPixmap(self.file_path))
		print ("Button 4 clicked")





	# def btnstate2(self):
		# self.l1.setPixmap(QPixmap("/home/bhanu/Desktop/IITKGP_team_4/detect-master-fin/logos/logo3.jpeg"))
		# self.l2.setPixmap(QPixmap("/home/bhanu/Desktop/IITKGP_team_4/detect-master-fin/logos/logo2.png"))
		# if(self.second==False):
		# 	self._main_real(parser.parse_args(), self.file_path)
		# else:
		# 	self.run_image_real(self.file_path)	
		# self.l2.setPixmap(QPixmap("/home/rpg/Documents/tch/ray/images/out/result.jpg"))		
		# self.l0.setText("Disease predicted :"+self.disease)
		# self.l01.setText("Test Disease :"+self.disease_real)
		# print ("Button 5 clicked")

	def btnstate4(self):
		self.close()
		print ("Button 6 clicked")	


class MainWindow(QWidget):
	def __init__(self):
		QWidget.__init__(self)

		self.setGeometry(300,300,300,220)
		vbox1=QVBoxLayout()
		
		self.l0=QLabel()		
		self.l0.setText("clevr-iep")
		self.l0.setFont(QFont('SansSerif', 30))
		self.l0.setStyleSheet("QLabel { background-color : black; color : white; }");
		self.l0.setFixedHeight(100)
		self.l0.setAlignment(Qt.AlignCenter)
	
		vbox1.addWidget(self.l0)
		self.b0 = QPushButton("GO")
		self.b0.setCheckable(True)
		self.b0.clicked.connect(self.btnstate0)	
		
		self.b0.setFont(QFont('SansSerif', 30))
		self.b0.setFixedHeight(50)

			
		self.setWindowTitle("clevr")
		self.showMaximized()
		vbox1.addWidget(self.b0)
		
		self.setLayout(vbox1)
		self.show()

	def btnstate0(self):
		self.dialog = combodemo()
		self.dialog.show()	
		self.close()
		print ("Button 1 clicked") 

def main():
	app = QApplication(sys.argv)
	oMainwindow = MainWindow()
	sys.exit(app.exec_())		

if __name__ == '__main__':
	if sys.version[0] == '2':
		reload(sys)
		sys.setdefaultencoding("utf-8")
	main()


