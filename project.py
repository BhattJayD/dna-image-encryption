import tkinter as tk
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
import hashlib 
import binascii
import textwrap
from scipy.integrate import odeint
from bisect import bisect_left as bsearch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from importlib import reload  
import os

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
a, b, c = 10, 2.667, 28
tmax, N = 100, 10000
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
global tr

print(dna)
class MyButton:
	def __init__(self, root):
		self.f = tk.Frame(root)
		self.f.propagate(0)
		self.f.pack()
		b1=tk.Button(root,text="Encrypt",bg="lightblue",fg="black", cursor='watch',command=self.main).place(x=150,y=10, width=100,height=30)
		b2=tk.Button(root,text="Decrypt",bg="lightblue",fg="black", cursor='watch',command=self.otherside).place(x=250,y=10, width=100, height=30)
		b3=tk.Button(root,text="Get Encrypted Image",bg="lightblue",fg="black", cursor='watch',command=self.getimg).place(x=150,y=40, width=200, height=30)
		#b2=tk.Button(root,text="recieve",bg="lightblue",fg="black", cursor='watch').place(x=450,y=10, width=100, height=30)
		b4= tk.Button(root, text="Exit", bg="lightblue", fg="BLACK",cursor='watch',command=exit).place(x=230,y=370, width=100, height=30)
	
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
			print("en to ds")
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

		def xor_operation(b,g,r,mk):
			m,n = b.shape
			bx=np.chararray((m,n))
			gx=np.chararray((m,n))
			rx=np.chararray((m,n))
			b=b.astype(str)
			g=g.astype(str)
			r=r.astype(str)
			for i in range(0,m):
				for j in range (0,n):
					bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
					gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
					rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
			bx=bx.astype(str)
			gx=gx.astype(str)
			rx=rx.astype(str)
			print("xor",bx)
			return bx,gx,rx 
		blue_final, green_final, red_final = xor_operation(bn,gn,rn,mmk)

		def lorenz(X, t, a, b, c):
			x, y, z = X
			x_dot = -a*(x - y)
			y_dot = c*x - y - x*z
			z_dot = -b*z + x*y
			return x_dot, y_dot, z_dot

		def gen_chaos_seq(m,n):
			global x0,y0,z0,a,b,c,N
			N=m*n*4
			x= np.array((m,n*4))
			y= np.array((m,n*4))
			z= np.array((m,n*4))
			t = np.linspace(0, tmax, N)
			f = odeint(lorenz, (x0, y0, z0), t, args=(a, b, c))
			x, y, z = f.T
			x=x[:(N)]
			y=y[:(N)]
			z=z[:(N)]
			print("ge",z)
			return x,y,z
		x,y,z=gen_chaos_seq(m,n)

		def sequence_indexing(x,y,z):
			n=len(x)
			fx=np.zeros((n),dtype=np.uint32)
			fy=np.zeros((n),dtype=np.uint32)
			fz=np.zeros((n),dtype=np.uint32)
			seq=sorted(x)
			for k1 in range(0,n):
				t = x[k1]
				k2 = bsearch(seq, t)
				fx[k1]=k2
			seq=sorted(y)
			for k1 in range(0,n):
				t = y[k1]
				k2 = bsearch(seq, t)
				fy[k1]=k2
			seq=sorted(z)
			for k1 in range(0,n):
				t = z[k1]
				k2 = bsearch(seq, t)
				fz[k1]=k2
			print("fx",fx)
			return fx,fy,fz
		fx,fy,fz=sequence_indexing(x,y,z)

		def scramble(fx,fy,fz,b,r,g):
			p,q=b.shape
			size = p*q
			bx=b.reshape(size).astype(str)
			gx=g.reshape(size).astype(str)
			rx=r.reshape(size).astype(str)
			bx_s=np.chararray((size))
			gx_s=np.chararray((size))
			rx_s=np.chararray((size))
			for i in range(size):
				idx = fz[i]
				bx_s[i] = bx[idx]
			for i in range(size):
				idx = fy[i]
				gx_s[i] = gx[idx]
			for i in range(size):
				idx = fx[i]
				rx_s[i] = rx[idx]
			bx_s=bx_s.astype(str)
			gx_s=gx_s.astype(str)
			rx_s=rx_s.astype(str)

			b_s=np.chararray((p,q))
			g_s=np.chararray((p,q))
			r_s=np.chararray((p,q))
			b_s=bx_s.reshape(p,q)
			g_s=gx_s.reshape(p,q)
			r_s=rx_s.reshape(p,q)
			print("bs\n",b_s)
			return b_s,g_s,r_s
		blue_scrambled,green_scrambled,red_scrambled = scramble(fx,fy,fz,blue_final,red_final,green_final)

		def dna_decode(b,g,r):
			m,n = b.shape
			r_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
			g_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
			b_dec= np.ndarray((m,int(n*2)),dtype=np.uint8)
			for color,dec in zip((b,g,r),(b_dec,g_dec,r_dec)):
				for j in range(0,m):
					for i in range(0,n):
						dec[j,2*i]=dna["{0}".format(color[j,i])][0]
						dec[j,2*i+1]=dna["{0}".format(color[j,i])][1]
			b_dec=(np.packbits(b_dec,axis=-1))
			g_dec=(np.packbits(g_dec,axis=-1))
			r_dec=(np.packbits(r_dec,axis=-1))
			print("oppp",b_dec)
			return b_dec,g_dec,r_dec
		b,g,r=dna_decode(blue_scrambled,green_scrambled,red_scrambled)

		# img,fx,fy,fz,file_path,Mmk,blue,green,red send  mmkfrom keyencodemetrix rgb from decomposemetrix




		def scramble_new(fx,fy,fz,b,g,r):
			print("e to sn")
			p,q=b.shape
			size = p*q
			bx=b.reshape(size)
			gx=g.reshape(size)
			rx=r.reshape(size)
			bx_s=b.reshape(size)
			gx_s=g.reshape(size)
			rx_s=r.reshape(size)
			bx=bx.astype(str)
			gx=gx.astype(str)
			rx=rx.astype(str)
			bx_s=bx_s.astype(str)
			gx_s=gx_s.astype(str)
			rx_s=rx_s.astype(str)
			for i in range(size):
				idx = fz[i]
				bx_s[idx] = bx[i]
			for i in range(size):
				idx = fy[i]
				gx_s[idx] = gx[i]
			for i in range(size):
				idx = fx[i]
				rx_s[idx] = rx[i]
			b_s=np.chararray((p,q))
			g_s=np.chararray((p,q))
			r_s=np.chararray((p,q))
			b_s=bx_s.reshape(p,q)
			g_s=gx_s.reshape(p,q)
			r_s=rx_s.reshape(p,q)
			return b_s,g_s,r_s

		def xor_operation_new(b,g,r,mk):
			m,n = b.shape
			bx=np.chararray((m,n))
			gx=np.chararray((m,n))
			rx=np.chararray((m,n))
			b=b.astype(str)
			g=g.astype(str)
			r=r.astype(str)
			for i in range(0,m):
				for j in range (0,n):
					bx[i,j] = dna["{0}{1}".format(b[i,j],mk[i,j])]
					gx[i,j] = dna["{0}{1}".format(g[i,j],mk[i,j])]
					rx[i,j] = dna["{0}{1}".format(r[i,j],mk[i,j])]
			bx=bx.astype(str)
			gx=gx.astype(str)
			rx=rx.astype(str)
			print("com the xorn")
			return bx,gx,rx 

		def recover_image(b,g,r,iname):
			img = cv2.imread(iname)
			img[:,:,2] = r
			img[:,:,1] = g
			img[:,:,0] = b
			cv2.imwrite(("enc.jpg"), img)
			print("saved ecrypted image as enc.jpg")
			return img
		img=recover_image(b,g,r,path)


		def decrypt(image,fx,fy,fz,fp,Mk,bt,gt,rt):
			red = image[:,:,2]
			green = image[:,:,1]
			blue = image[:,:,0]
			p,q = rt.shape
			def dna_encode1(b,g,r):
				print("en to ds")
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

			benc,genc,renc=dna_encode1(b,g,r)
			bs,gs,rs=scramble_new(fx,fy,fz,benc,genc,renc)
			bx,rx,gx=xor_operation_new(bs,gs,rs,Mk)
			blue,green,red=dna_decode(bx,gx,rx)
			green,red = red, green
			img=np.zeros((p,q,3),dtype=np.uint8)	
			img[:,:,0] = red
			img[:,:,1] = green
			img[:,:,2] = blue
			print("RED\n",red)
			print("GREEN\n",green)
			print("BLUE\n",blue)
			cv2.imwrite(("Recovered.jpg"), img)
		decrypt(img,fx,fy,fz,path,mmk,blue,green,red)


		mycmd='python3 -m http.server 80'
		os.system(mycmd)
		#mycmd.terminate()
		#wg=' wget http://127.0.0.1:8000/enc.jpg'
		#os.system(wg)
		#imm=np.array(imm)
		#print(type(imm))
		'''def split_into_rgb_channels(img):
			print(type(img))
			print(img)
			red = img[:,:,2]
			green = img[:,:,1]
			blue = img[:,:,0]
			return red, green, blue
		r,g,b=split_into_rgb_channels(img)
		print("r",r)
		print("g",g)
		print("b",b)'''

	def getimg(self):
		eimg='torsocks curl -L -o enc.jpg 6jsygadgowg3nst7m5ndywez7hd5rlss2phdxpv44xhucjgowbkc4lid.onion/enc.jpg'
		os.system(eimg)
	def otherside(self):
		#wg=' curl -L -o enc.jpg http://127.0.0.1:80/enc.jpg'
		sleee="sleep 5"
		os.system(sleee)
		cu='torsocks curl -L -o enccc.jpg  6jsygadgowg3nst7m5ndywez7hd5rlss2phdxpv44xhucjgowbkc4lid.onion/Recovered.jpg'
		os.system(cu)
		cll="clear"
		os.system(cll)
		os.system(sleee)

		#os.system(wg)
		'''m1=' curl -L -o mmk.txt http://127.0.0.1:80/mmk.txt'
		os.system(m1)

		im='curl -L -o img.txt http://127.0.0.1:80/img.txt'
		os.system(im)
		r1=' curl -L -o rr.txt http://127.0.0.1:80/rr.txt'
		os.system(r1)
		g1=' curl -L -o gg.txt http://127.0.0.1:80/gg.txt'
		os.system(g1)
		b1=' curl -L -o bb.txt http://127.0.0.1:80/bb.txt'
		os.system(b1)
		ffx1=' curl -L -o fx1.txt http://127.0.0.1:80/fx1.txt'
		os.system(ffx1)
		ffy1=' curl -L -o fy1.txt http://127.0.0.1:80/fy1.txt'
		os.system(ffy1)
		ffz1=' curl -L -o fz1.txt http://127.0.0.1:80/fz1.txt'
		os.system(ffz1)
		file=open('mmk.txt','r')
		MMK=file.read()
		file.close()
		print(MMK)

		file=open('img.txt','r')
		img11=file.read()
		file.close()
		#iim=list()
		imm=list(img11.split())
		print(type(imm))
		#img=int(img11)

		#with open("img.txt", "r") as f:
		#	for l in f:
		#		print(sum([int(a) for a in l.split()]))

		file=open('rr.txt','r')
		rr=file.read()
		file.close()
		print(rr)
		file=open('gg.txt','r')
		gg=file.read()
		file.close()
		print(gg)
		file=open('bb.txt','r')
		bb=file.read()
		file.close()
		print(bb)
		file=open('fx1.txt','r')
		fx1=file.read()
		file.close()
		print(fx1)
		file=open('fy1.txt','r')
		fy1=file.read()
		file.close()
		print(fy1)
		file=open('fz1.txt','r')
		fz1=file.read()
		file.close()
		print(fz1)
		

		#[int(i) for i in imm[0].split(',')]

		imm=np.array(imm)
		print(type(imm))
		def split_into_rgb_channels(imm):
			print(type(imm))
			print(imm)
			red = imm[::3]
			green = imm[::2]
			blue = imm[::1]
			return red, green, blue
		r,g,b=split_into_rgb_channels(imm)
		print("red \n",r)
		print("green \n",g)
		print("blue \n",b)'''

		





mb=MyButton(root)
root.mainloop()
