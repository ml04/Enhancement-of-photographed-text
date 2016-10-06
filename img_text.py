import numpy as np
import cv2

def show (slika):
	cv2.imshow('', slika)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def promijeni_velicinu (slika, nova_visina):
	 #dobivanje novih dimenzija slike na temelju koeficijenta pretvorbe
	 koeficijent = nova_visina/slika.shape[0]
	 nove_dimenzije = (int(slika.shape[1] * koeficijent), int(nova_visina))
	 
	 nova_slika = cv2.resize(slika, nove_dimenzije, interpolation = cv2.INTER_AREA)
	 return nova_slika

def canny (slika):
	
	#primjena Gaussovog filtra
	slika_gaus = cv2.GaussianBlur(slika, (5,5), 0)
	show(slika_gaus)
	
	#definira se nul operacija za parametar
	def nothing(x):
		pass
	#stvara se prozor za sliku i klizace
	cv2.namedWindow('Canny')
	
	#stvaranje dva klizaca za gornji i donji threshold
	cv2.createTrackbar('donji_threshold', 'Canny', 0, 255, nothing)
	cv2.createTrackbar('gornji_threshold', 'Canny', 0, 255, nothing)
	
	#petlja koja uzima vrijednost od klizaca i vrsi cannyevo detektiranje dok se ne pritisne esc
	while(1):
		
		#primanje polozaja klizaca
		donji_threshold = cv2.getTrackbarPos('donji_threshold', 'Canny')
		gornji_threshold = cv2.getTrackbarPos('gornji_threshold', 'Canny')
		
		#primjena cannyevog detektora
		slika_rub = cv2.Canny(slika_gaus, donji_threshold, gornji_threshold)
		
		cv2.imshow('original', slika)
		cv2.imshow('Canny', slika_rub)
		k = cv2.waitKey(1) & 0xFF
		
		if k == 27:
			break
			
	cv2.destroyAllWindows
	
	return slika_rub
		
def threshold (slika):
	
	def nothing(x):
		pass
	#stvara se prozor za sliku i klizac	
	cv2.namedWindow ('Threshold')
	
	#stvaranje klizaca
	cv2.createTrackbar('C', 'Threshold', 0, 20, nothing)
	
	#petlja koja uzima vrijednost od klizaca za konstantu C
	while(1):
		
		#primanje vrijednsti od klizaca
		c = cv2.getTrackbarPos('C', 'Threshold')
		
		#primjena thresholda
		rez_slika = cv2.adaptiveThreshold(rot_slika, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, c)
	
		cv2.imshow('Threshold', promijeni_velicinu(rez_slika, 500.0))
		k = cv2.waitKey(1) & 0xFF
		
		if k == 27:
			break
	
	cv2.destroyAllWindows
	return rez_slika
		
def poslozi_tocke (tocke):
	
	orig_tocke = np.zeros((4,2), dtype = "float32")
	
	# suma x i y koordinate
	suma = tocke.sum(axis = 1)
	# najmanja suma 
	orig_tocke[0] = tocke[np.argmin(suma)]
	# najveca suma
	orig_tocke[3] = tocke[np.argmax(suma)]
	
	# razlika izmedu x i y koordinate
	razlika = np.diff(tocke, axis = 1)
	# najveca negativna razlika
	orig_tocke[1] = tocke[np.argmin(razlika)]
	# najveca pozitivna razlika
	orig_tocke[2] = tocke[np.argmax(razlika)]
	
	return orig_tocke
	
def rotacija_slike (tocke, slika):
	
	orig_tocke = poslozi_tocke(tocke)
	
	# t1- gornja lijeva tocka, t2 - gornja desna tocka, t3 - donja lijeva tocka, t4 - donja desna tocka
	t1 = orig_tocke[0]
	t2 = orig_tocke[1]
	t3 = orig_tocke[2]
	t4 = orig_tocke[3]
	
	#mjerenje sirine objekta pomocu formule za udaljenost dvije tocke
	sirina_gore = np.sqrt(((np.absolute(t2[1] - t1[1]))**2) + (((np.absolute(t2[0] - t1[0]))**2)))
	sirina_dolje = np.sqrt(((np.absolute(t4[1] - t3[1]))**2) + (((np.absolute(t4[0] - t3[0]))**2)))
	sirina = max(int(sirina_gore), int(sirina_dolje))
	
	#mjerenje visine objekta pomocu formule za udaljenost dvije tocke
	visina_lijevo = np.sqrt(((np.absolute(t3[1] - t1[1]))**2) + ((np.absolute(t3[0] - t1[0]))**2))
	visina_desno = np.sqrt(((np.absolute(t4[1] - t2[1]))**2) + ((np.absolute(t4[0] - t2[0]))**2))
	visina = max(int(visina_lijevo), int(visina_desno))
	
	#postavljanje tocaka rezultantne slike
	rez_tocke = np.array([[0, 0], [sirina - 1, 0], [0, visina - 1], [sirina - 1, visina -1]], dtype = "float32")
	
	#dobivanje zakrenute slike
	M = cv2.getPerspectiveTransform(orig_tocke, rez_tocke)
	rez = cv2.warpPerspective(slika, M, (sirina, visina))
	
	return rez

slika = cv2.imread("g.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
original = slika.copy()

#mijenanje velicine slike i pamcenje koeficijenta pretvorbe
koef = slika.shape[0]/500
slika = promijeni_velicinu(slika, 500.0)

show(slika)

#slanje slike u funkciju za cannyev detektor
slika_rub = canny(slika)
show(slika_rub)

#prepoznavanje zatvorenih kontura na slici
(konture, h) = cv2.findContours(slika_rub.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#izdvajanje najvece zatvorene konture
papir = max(konture, key = cv2.contourArea)

#aproksimacija cetverokutne konture
epsilon = 0.1 * cv2.arcLength(papir, True)
approx = cv2.approxPolyDP(papir, epsilon, True)

#provjera broja tocaka koje omeduju konturu
if len(approx) == 4:
	kont_ekrana = approx

#iscrtavanje konture na crno-bijeloj slici
cv2.drawContours(slika, [kont_ekrana], -1, (0, 255, 0), 2)
show(slika)

#pozivanje funkcije za mijenjanje perspektive
rot_slika = rotacija_slike (kont_ekrana.reshape(4,2) * koef , original)

#primjena thresholda na rotiranu sliku
rez_slika = threshold(rot_slika)

show(promijeni_velicinu(rez_slika, 500.0))
filepath = "C:/Users/Matt/Documents/Projekt OSiRV/ slika_1.jpg"
cv2.imwrite(filepath, rez_slika)
