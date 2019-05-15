import numpy as np
#import cv2

def genIndex(chanums):
    
      index = []
      i = 1
      j = i+1
      
      if (chanums % 2) == 0:
         Ns = chanums+1 
      else:      
         Ns = chanums
      
      
      index.append(1)
      t = chr(i+ord('A'))
      while(i!=j):
		l = ""
		l = l+chr(i+ord('A'))
		l = l+chr(j+ord('A'))
		r = ""
		r = r+chr(j+ord('A'))
		r = r+chr(i+ord('A'))
		if(j>Ns):
			j = 1
		elif(t.find(l)==-1 and t.find(r)==-1):
			index.append(j)
			t = t+chr(j+ord('A'))
			i = j
			j = i+1
		else:
			j = j+1 
  
  
  
      new_index = []
      if (chanums % 2) == 0:
          for i in range(len(index)):
              if index[i] != chanums+1:
                 new_index.append(index[i])  
  
      index = np.array(new_index)    
      index = index-1
      return index
 
 
def get_signal_img(data):
    
     ch_num = data.shape[0]
     index = genIndex(ch_num)
     signal_img = data[index]
     signal_img = signal_img[:-1]
#     print signal_img.shape
     return signal_img
     
def get_activity_img(data):
    
    signal_img = get_signal_img(data)
    
    f = np.fft.fft2(signal_img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
#    magnitude_spectrum = cv2.resize(magnitude_spectrum,None,fx=1,fy=8)
#    cv2.imshow('image',magnitude_spectrum)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return magnitude_spectrum    
    
    
