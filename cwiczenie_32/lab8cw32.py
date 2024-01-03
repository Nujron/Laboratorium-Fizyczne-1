import numpy as np
import math as mh
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

np.set_printoptions(suppress=True)

dane=pd.read_excel('lab8_cw32.xlsx')
#print(dane)
# wczytywanie danych
op1=np.array(dane[2:13]['Unnamed: 2'])
op2=np.array(dane[13:24]['Unnamed: 2'])
op3=np.array(dane[24:35]['Unnamed: 2'])
op1_2szer=np.array(dane[35:46]['Unnamed: 2'])
op1_3szer=np.array(dane[46:57]['Unnamed: 2'])
op1_2rown=np.array(dane[57:68]['Unnamed: 2'])
op1_2_3rown=np.array(dane[68:79]['Unnamed: 2'])
#print(op1)
#print(op2)
#print(op3)
#print(op1_2szer)
#print(op1_3szer)
#print(op1_2rown)
#print(op1_2_3rown)

#a jest takie samo dla szystkich danych
a=np.array(dane[2:13]['Unnamed: 4'])
blada=0.001
#print(a)
# długość linijki w metrach
l=np.full(11, 1)
bladl=0.001
b0=np.array(l-a)
b1=[]
for i in b0:
   b1.append(round(i,2)) 
b=np.array(b1)
#print(b)
"""for i in [op1,op2,op3]:
    print(i)"""

#tablice przechwoujące wyniki (8, bo itesuja od i=1)
opory=np.empty(8)
bledy=np.empty(8)

print("\nPOMIARY OPORÓW:")
for i,j,k in zip(['Opornik 1', 'Opornik 2', 'Opornik 3', 'Oporniki 1, 2 szeregowo', 'Oporniki 1, 3 szeregowo', 'Oporniki 1, 2 równolegle', 'Oporniki 1, 2, 3 równolegle'], 
                 [op1, op2, op3, op1_2szer, op1_3szer, op1_2rown, op1_2_3rown], [1, 2, 3, 4, 5, 6, 7]):
    #print(i + str(j))
    
    # błędy pomiarowe
    klasa=0.001
    bladR2=np.array(klasa*j)
    # Błędy były na tyle małe, że nie było ich widać na wykresach
    bladX=( (a*bladl)**2 + (l*blada)**2 )**0.5
    bladY=( (a*bladR2)**2 + (j*blada)**2 )**0.5
    #print("Niepewność a-l: " + str(bladX))
    #print("Niepewność R2*a: " + str(bladY))
    
    # wagi punktów
    waga=1/( bladX**2 + bladY**2 )**0.5
    #print(waga)
    
    regresja=sm.WLS(list(j*a), list(l-a), weights=list(waga))
    results=regresja.fit()
    #Współczynnik nachylenia Rx
    slope=results.params[0]
    # niepewność Rx
    Dslope=results.bse[0]
    
    if k==1 or k==5: 
        plt.figure(figsize=(11, 7)) #tworzy nowe okno i określa jego rozmiar
    
    if k<=4:
        plt.subplot(2, 2, k)
    else:
        plt.subplot(2, 2, k-4)
    
    x=np.linspace(0.25, 0.85, 2)
    # właściwy wykres
    y=slope*x
    plt.scatter(l-a, j*a, color='purple')
    plt.plot(x, y, color='magenta')
    
    # najmniejsze i największe nachylenie
    '''y_min=(slope-Dslope)*x
    plt.plot(x, y_min, color='grey')
    y_max=(slope+Dslope)*x
    plt.plot(x, y_max, color='grey')
    '''
    # opisy
    plt.title(i)
    plt.xlabel('a-l [m]')
    plt.ylabel('R₂a [Ωm]')
    # krzyże niepewności
    # plt.errorbar(l-a, j*a, yerr=bladY, fmt='-', capsize=3, xerr=bladX,ls = 'none')
    
    if k==4 or k==3:
        plt.tight_layout() #skalowanie wykresów
        
    # niepewność Rx 
    print(i + ":")
    print("     " + "Rx=(" + str(round(slope, 2)) + " ± " + str(round(Dslope, 2)) + ")Ω")
    
    opory[k]=slope
    bledy[k]=Dslope
   
plt.show()

# SUMY OKRESLONYCH OPORÓW
print("\nOpory zastępcze określonyc połączeń:")
# połączenia szeregowe
for i,j,k in zip(["Połączenia szeregowego oporników 1 i 2: ", "Połączenia szeregowego oporników 1 i 3: "], 
                 [[opory[1], opory[2]] , [opory[1], opory[3]]],
                 [[bledy[1], bledy[2]], [bledy[1], bledy[3]]]):
    wynik=j[0]+j[1]
    blad=(k[0]**2 + k[1]**2)**0.5
    print(i + "(" + str(round(wynik, 2)) + " ± " + str(round(blad, 2)) + ")Ω")
# połączenie równoległe dwóch
wynik=opory[1]*opory[2]/(opory[1]+opory[2])
blad=0.
for i in zip([opory[1], opory[2]], [opory[2], opory[1]]):
    blad+=(i[1]**2/(i[0]+i[1])**2)**2
blad=(blad)**0.5
print("Połączenia równoległego oporników 1 i 2: " + "(" + str(round(wynik, 2)) + " ± " + str(round(blad, 2)) + ")Ω")
# połączenie równoległe trzech  
wynik=opory[1]*opory[2]*opory[3]/(opory[1]*opory[2] + opory[2]*opory[3] + opory[1]*opory[3])
blad=0.
for i in zip([opory[1], opory[2], opory[3]], [opory[2], opory[1], opory[3]], [opory[3], opory[1], opory[2]]):
    blad+=((i[1]*i[2])**2/(i[0]*i[1] + i[1]*i[2] + i[0]*i[2])**2)**2
blad=(blad)**0.5
print("Połączenia równoległego oporników 1 i 2: " + "(" + str(round(wynik, 2)) + " ± " + str(round(blad, 2)) + ")Ω")
    

# Felerne wyniki (1 opornik, b=0.7 i 1, 2 równolegle, b=0.7)
print("\nNIEPASUJĄCE WYNIKI:")
print("Opornik 1, b=0.7: Rx=(" + str(round(op1[2]*0.3/0.7, 2)) + " ± " + str(round(1/0.7*( (0.3*0.7*0.001*op1[2])**2 + (op1[2]*1*blada)**2 + (op1[2]*0.3*bladl)**2 )**0.5, 2)) + ")Ω")
print("Oporniki 1 i 2 równ.:, b=0.7: Rx=(" + str(round(op1_2rown[2]*0.3/0.7, 2)) + " ± " + str(round(1/0.7*( (0.3*0.7*0.001*op1_2rown[2])**2 + (op1_2rown[2]*1*blada)**2 + (op1_2rown[2]*0.3*bladl)**2 )**0.5, 2)) + ")Ω")

# Analiza ostatniej serii ("skr" od skrajne)
print("\nPOMIARY DLA SKRAJNYCH a:")
lskr=np.full(6, 1)
#print(dane)
# opory dekoadoe dla skrajnych a
Rskr=np.array(dane[3:9]['Unnamed: 7'])
askr=np.array(dane[3:9]['Unnamed: 9'])
#print(Rskr)
#print(askr)
skrNiepR=0.001*Rskr
wynSkr=Rskr*askr/(lskr-askr)
niepskr=1/(lskr-askr)*( (askr*(lskr-askr)*skrNiepR)**2 + (Rskr*lskr*blada)**2 + (Rskr*askr*bladl)**2 )**0.5
#print((askr*(lskr-askr)*skrNiepR))
#print(Rskr*lskr*blada)
#print(Rskr*askr*bladl)

for i,j,k,m in zip(askr, Rskr, wynSkr, skrNiepR):
    print("Odcinek a: " + str(i) + "m")
    print("     Opór wzorcowy: " + str(round(j, 2)) + "Ω")
    print("     Opór Rx=(" + str(round(k, 4)) + " ± " + str(round(m, 4)) + ")Ω")
    
print('\n')
