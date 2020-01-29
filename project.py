import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
import hashlib 
import binascii
import textwrap

root =tk.Tk()
root.title("Image Encrption using Dynamic DNA Cryptography")
root.configure(background="black")
cwgt=tk.Canvas(root)
cwgt.pack(expand=True, fill=tk.BOTH)
image1=ImageTk.PhotoImage(file="BG.jpeg")
w,h=image1.width(),image1.height()
root.geometry('535x405')
cwgt.img=image1
cwgt.create_image(0, 0, anchor=tk.NW, image=image1)

x0,y0,z0=0,0,0
dna={}
dna["00"]="A"
dna["01"]="T"
dna["10"]="G"
dna["11"]="C"
dna["A"]=[0,0]
dna["T"]=[0,1]
dna["G"]=[1,0]
dna["C"]=[1,1]
dna["AA"]=dna["TT"]=dna["GG"]=dna["CC"]="A"
dna["AG"]=dna["GA"]=dna["TC"]=dna["CT"]="G"
dna["AC"]=dna["CA"]=dna["GT"]=dna["TG"]="C"
dna["AT"]=dna["TA"]=dna["CG"]=dna["GC"]="T"

print(dna)
class MyButton:
	def __init__(self, root):
		self.f = tk.Frame(root)
		self.f.propagate(0)
		self.f.pack()
		b1=tk.Button(root,text="Encrypt",bg="lightblue",fg="black", cursor='watch',command=self.main).place(x=150,y=10, width=100,height=30)
		b2=tk.Button(root,text="Decrypt",bg="lightblue",fg="black", cursor='watch').place(x=250,y=10, width=100, height=30)
		b2=tk.Button(root,text="Send",bg="lightblue",fg="black", cursor='watch').place(x=350,y=10, width=100, height=30)
		b2=tk.Button(root,text="recieve",bg="lightblue",fg="black", cursor='watch').place(x=450,y=10, width=100, height=30)
		b4= tk.Button(root, text="Exit", bg="lightblue", fg="BLACK",cursor='watch',command=exit).place(x=235,y=370, width=100, height=30)
	
	def main(self):
		from tkinter import filedialog
		path = "NULL"
		path = filedialog.askopenfilename()
		if path!="NULL":
			print("Image loaded!",path)
		else:
			print("Error Image not loaded!",path)
		print("hi")
		print(path)
		
		print("hi")
		image = cv2.imread(path)
		print(path)
		red = image[:,:,2]
		green = image[:,:,1]
		blue = image[:,:,0]
		for values, channel in zip((red, green, blue), (2,1,0)):
			print("done")
			img = np.zeros((values.shape[0], values.shape[1]), dtype = np.uint8)
			img[:,:] = (values)
			if channel == 0:
				B = np.asmatrix(img)
			elif channel == 1:
				G = np.asmatrix(img)
			else:
				R = np.asmatrix(img)		
		print("done ok RED \n",R)
		print("done ok GREEN \n",G)
		print("done ok BLUE \n",B)
		img = Image.open(path)
		m, n = img.size
		print("sol")
		print("pixels: {0}  width: {2} height: {1} ".format(m*n, m, n))
		pix = img.load()
		plainimage = list()                        
		print(plainimage)
		for y in range(n):
			for x in range(m):
				for k in range(0,3):
					plainimage.append(pix[x,y][k])
		
		key = hashlib.sha256()                   
		key.update(bytearray(plainimage))         
		kn=key.hexdigest()
		print(kn)
		
		def update_lorents(kn):
			key_bin = bin(int(kn, 16))[2:].zfill(256)  #covert hex key digest to binary
			k={}                                        #key dictionary
			key_32_parts=textwrap.wrap(key_bin, 8)      #slicing key into 8 parts
			num=1
			for i in key_32_parts:
				k["k{0}".format(num)]=i
				num = num + 1
			t1 = t2 = t3 = 0
			for i in range (1,12):
				t1=t1^int(k["k{0}".format(i)],2)
			for i in range (12,23):
				t2=t2^int(k["k{0}".format(i)],2)
			for i in range (23,33):
				t3=t3^int(k["k{0}".format(i)],2)
			global x0 ,y0, z0
			x0=x0 + t1/256
			y0=y0 + t2/256
			z0=z0 + t3/256
			print(x0)
			print("done")
		update_lorents(kn)

		def dna_encode(b,g,r):
			b = np.unpackbits(b,axis=1)
			g = np.unpackbits(g,axis=1)
			r = np.unpackbits(r,axis=1)
			m,n = b.shape
			r_enc= np.chararray((m,int(n/2)))
			g_enc= np.chararray((m,int(n/2)))
			b_enc= np.chararray((m,int(n/2)))
			for color,enc in zip((b,g,r),(b_enc,g_enc,r_enc)):
				idx=0
				for j in range(0,m):
					for i in range(0,n,2):
						enc[j,idx]=dna["{0}{1}".format(color[j,i],color[j,i+1])]
						idx+=1
						if (i==n-2):
							idx=0
							break
			b_enc=b_enc.astype(str)
			g_enc=g_enc.astype(str)
			r_enc=r_enc.astype(str)
			print("ddone")
			return b_enc,g_enc,r_enc
		bn,gn,rn=dna_encode(B,G,R)
		print("blue\n",bn)
		print("green\n",gn)
		print("red \n",rn)


		def key_matrix_encode(kn,b):
			b = np.unpackbits(b,axis=1)
			m,n = b.shape
			key_bin = bin(int(kn, 16))[2:].zfill(256)
			Mk = np.zeros((m,n),dtype=np.uint8)
			x=0
			for j in range(0,m):
				for i in range(0,n):
					Mk[j,i]=key_bin[x%256]
					x+=1
			Mk_enc=np.chararray((m,int(n/2)))
			idx=0
			for j in range(0,m):
				for i in range(0,n,2):
					if idx==(n/2):
						idx=0
					Mk_enc[j,idx]=dna["{0}{1}".format(Mk[j,i],Mk[j,i+1])]
					idx+=1
			Mk_enc=Mk_enc.astype(str)
			return Mk_enc
		mmk=key_matrix_encode(kn,B)
		print("fi\n",mmk)


mb=MyButton(root)
root.mainloop()