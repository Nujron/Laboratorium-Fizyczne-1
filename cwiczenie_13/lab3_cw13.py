import pandas as pd
import math as mh
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

print("\n")
# D_l = delta l
excel_dada = pd.read_excel("lab3_cw13.xlsx")
D = 0.039
l = 0.8
D_l = 0.005
D_t = 0.2
D_m = 0.000001
D_d = 0.00001
rho = 1249.99
g = 9.81

# dane wzorcowe:
# różnica między obliczoną etą a tablicową
rozt_rozn = np.zeros(31)
# print(rozt_rozn)
proc = np.arange(70, 101, 1)
# print(proc)
eta_wzr = np.array([ 21.95, 23.89, 26.06, 28.50, 31.23, 34.32, 37.81, 41.76, 46.26, 51.40,
                     57.28, 64.04, 71.84, 80.88, 91.39, 103.67, 118.08, 135.06, 155.18, 179.14,
                     207.82, 242.35, 284.18, 335.16, 397.72, 475.00, 571.20, 691.88, 844.51, 1039.26, 1290.05 ])
eta_wzr = eta_wzr/1000
rho_wzr = np.array([ 1183.63, 1186.28, 1188.94, 1191.59, 1194.24, 1196.90, 1199.55, 1202.21, 1204.86, 1207.52,
                     1210.17, 1212.83, 1215.48, 1218.14, 1220.79, 1223.45, 1226.10, 1228.76, 1231.41, 1234.06,  
                     1236.72, 1239.37, 1242.03, 1244.68, 1247.34, 1249.99, 1252.65, 1255.30, 1257.96, 1260.61, 1263.27 ])


balldata1 = pd.DataFrame(excel_dada[5:15][["Unnamed: 3","Unnamed: 5","Unnamed: 7"]])
balldata1.rename(columns={"Unnamed: 3":"d[m]","Unnamed: 5": "m[kg]","Unnamed: 7":"t[s]"},inplace=True)
balldata1.index = [1,2,3,4,5,6,7,8,9,10]
balldata1[:]["d[m]"] *= 0.001
balldata1[:]["m[kg]"] *= 0.001

#print(balldata1)

balldata2 = pd.DataFrame(excel_dada[14:24][["Unnamed: 3","Unnamed: 5","Unnamed: 7"]])
balldata2.rename(columns={"Unnamed: 3":"d[m]","Unnamed: 5": "m[kg]","Unnamed: 7":"t[s]"},inplace=True)
balldata2.index = [10,11,12,13,14,15,16,17,18,19]
balldata2[:]["d[m]"] *= 0.001
balldata2[:]["m[kg]"] *= 0.001

#print(balldata2)

balldata3 = pd.DataFrame(excel_dada[25:35][["Unnamed: 3","Unnamed: 5","Unnamed: 7", "Unnamed: 9"]])
balldata3.rename(columns={"Unnamed: 3":"d[m]","Unnamed: 5": "m[kg]","Unnamed: 7":"t[s]", "Unnamed: 9":"h[m]"},inplace=True)
balldata3.index = [10,11,12,13,14,15,16,17,18,19]
balldata3[:]["d[m]"] *= 0.001
balldata3[:]["m[kg]"] *= 0.001
balldata3[:]["h[m]"] *= 0.001

#print(balldata3)

d = np.array(balldata1[:]["d[m]"])
# print("\nd:" + str(d))
m = np.array(balldata1[:]["m[kg]"])
# print("m:"+str(m))
t = np.array(balldata1[:]["t[s]"])


# 1. seria
print("\n1. seria pomiarowa:")

# numer rzeczywistego roztworu
i_rzezcz = 0
# szukanie gęstości
print("Różnica eta wyliczonego i tablicowego dla roztworów:")
for i in range(31):
    x = np.array(d*(1 + 2.4*d/D) / (m - rho_wzr[i]*mh.pi*d**3/6),dtype=float)
    y = np.array(g*t/(3*mh.pi*l))
    
    # niepewności x i y
    dxdd = ( (1 + 4.8*d/D) * (m-1/6*rho_wzr[i]*mh.pi*d**3) + (d + 2.4*d**2/D)*1/2*rho_wzr[i]*mh.pi*d**2 )/((m+(1/6)*rho_wzr[i]*mh.pi*d**3)**2)
    dxdm = d*(1 + 2.4*d/D)/(m - 1/6*rho_wzr[i]*mh.pi*d**3)**2
    #
    dydt = np.full(10, g/3/mh.pi/l)
    dydl = np.full(10, g/3/mh.pi/l**2)
    #
    dx = ( (dxdd * D_d)**2 + (dxdm * D_m)**2 )**0.5
    #print("\ndx:" + str(dx))
    dy = ( (dydt * D_t)**2 + (dydl * D_l)**2 )**0.5
    #print("\ndy: " + str(dy))
    W = list(1/(dx**2 + dy**2)**0.5)

    model = sm.WLS(y, x, hasconst = False, weights= W)
    results = model.fit()
    slope = results.params[0]
    eta = slope
    # dopasowanie
    #print(str(slope) + " " + str(eta_wzr[i]))
    rozt_rozn[i] = abs(eta - eta_wzr[i])
    rozt_rozn[i] = round(rozt_rozn[i], 3)
    print(str(proc[i]) + "   " + str(rho_wzr[i]) + "   " + str(round(eta_wzr[i], 3)) + "   " + str(round(eta, 3)) + "   " + str(rozt_rozn[i]))
    if i==0: min = rozt_rozn[i]
    if rozt_rozn[i] < min: 
        min = rozt_rozn[i]
        i_rzezcz = i

rho = rho_wzr[i_rzezcz]
print("\nNajlepiej pasujący roztwór i jego gęstość: " + str(proc[i_rzezcz]) + "%, " + str(rho_wzr[i_rzezcz]) + " kg/m^3")
rho = rho_wzr[i_rzezcz]
D_rho = (rho_wzr[i_rzezcz+1] - rho_wzr[i_rzezcz-1])/2
D_rho = round(D_rho, 3)
print("Niepewność gęstości: " + str(D_rho) + " kg/m^3")

x = np.array(d*(1 + 2.4*d/D) / (m - rho*mh.pi*d**3/6),dtype=float)
y = np.array(g*t/(3*mh.pi*l))
# niepewności x i y
dxdd = ( (1 + 4.8*d/D) * (m-1/6*rho*mh.pi*d**3) + (d + 2.4*d**2/D)*1/2*rho*mh.pi*d**2 )/((m+(1/6)*rho*mh.pi*d**3)**2)
dxdm = d*(1 + 2.4*d/D)/(m - 1/6*rho*mh.pi*d**3)**2
dxdrho = d**4*mh.pi* (1 + 2.4*d/D) /6 /(m - 1/6*rho*mh.pi*d**3)**2
#
dydt = np.full(10, g/3/mh.pi/l)
dydl = np.full(10, g/3/mh.pi/l**2)
#
dx = ( (dxdd * D_d)**2 + (dxdm * D_m)**2 )**0.5
#print("\ndx:" + str(dx))
dy = ( (dydt * D_t)**2 + (dydl * D_l)**2 + (dxdrho * D_rho)**2 )**0.5
#print("\ndy: " + str(dy))
W = list(1/(dx**2 + dy**2)**0.5)

# print(x)
# print(y)
model = sm.WLS(y, x, hasconst = False, weights= W)
results = model.fit()
slope = results.params[0]
x2 = np.linspace(5,80)
y2 = slope*x2
plt.figure(figsize=(13, 7.5))
plt.subplot(1, 1, 1)
plt.scatter(x,y, color = "blue", s=15)
plt.plot(x2,y2, color = "blue")
plt.xlabel("x(t,l)")
plt.ylabel("y(d,m,ρ)")
plt.errorbar(x, y, yerr=dy, capsize=3, xerr=dx ,ls = 'none', color = "black")

eta = slope
eta = round(eta, 4)
D_eta = results.bse[0]
D_eta = round(D_eta, 4)
print("Eta = (" + str(eta) + " ± " + str(D_eta) + ")Pa*s" )


# 2. seria
print("\n2. seria pomiarowa:")
d2 = np.array(balldata2[:]["d[m]"])
m2 = np.array(balldata2[:]["m[kg]"])
t2 = np.array(balldata2[:]["t[s]"])

# numer rzeczywistego roztworu
i_rzezcz2 = 0
# szukanie gęstości
print("Różnica eta wyliczonego i tablicowego dla roztworów:")
for i in range(31):
    eta_wyn=g*t2*( m2 - 1/6* rho_wzr[i] *mh.pi*d2**3) / ( 3*mh.pi*l*d2*( 1 + 2.4*d2/D ) )
    #print("Tablica 2. seria: " + str(eta_wyn))
    eta2 = np.mean(eta_wyn)
    #print(str(eta) + " " + str(eta_wzr[i]))
    rozt_rozn[i] = abs(eta2 - eta_wzr[i])
    rozt_rozn[i] = round(rozt_rozn[i], 3)
    print(str(proc[i]) + "   " + str(rho_wzr[i]) + "   " + str(round(eta_wzr[i], 3)) + "   " + str(round(eta2, 3)) + "   " + str(rozt_rozn[i]))
    if i==0: min = rozt_rozn[i]
    if rozt_rozn[i] < min: 
        min = rozt_rozn[i]
        i_rzezcz2 = i
    
proc_rzecz = proc[i_rzezcz2]
rho_rzecz = rho_wzr[i_rzezcz2]

print("\nNajlepiej pasujący roztwór i jego gęstość: " + str(proc[i_rzezcz2]) + "%, " + str(rho_wzr[i_rzezcz2]) + " kg/m^3")
rho2 = rho_wzr[i_rzezcz2]
D_rho2 = (rho_wzr[i_rzezcz2+1] - rho_wzr[i_rzezcz2-1])/2
D_rho2 = round(D_rho2, 3)
print("Niepewność gęstości: " + str(D_rho) + " kg/m^3")

eta_wyn=g*t2*( m2 - 1/6* rho2 *mh.pi*d2**3) / ( 3*mh.pi*l*d2*( 1 + 2.4*d2/D ) )
eta2 = np.mean(eta_wyn)
eta2 = round(eta2, 4)

D_eta2 = np.std(eta_wyn)
D_eta2 = round(D_eta2, 3)
print("Eta = (" + str(eta2) + " ± " + str(D_eta2) + ")Pa*s" )


# 3. seria
print("\n3. seria pomiarowa:")
d3 = np.array(balldata3[:]["d[m]"])
m3 = np.array(balldata3[:]["m[kg]"])
t3 = np.array(balldata3[:]["t[s]"])
h3 = np.array(balldata3[:]["h[m]"])

print("Współczynnik lepkosci w zalezności od głębokości:")
eta_wyn3=g*t3*( m3 - 1/6* rho *mh.pi*d3**3) / ( 3*mh.pi*l*d3*( 1 + 2.4*d3/D ) )

for i in range(10):
    eta_wyn3[i] = round(eta_wyn3[i], 3)
    h3[i] = round(h3[i], 3)
    print(str(h3[i]) + ": " + str(eta_wyn3[i]))

print('\n')
plt.show()
    