import numpy as np
import matplotlib.pyplot as plt

global A_crit

def displacement(A,make_plot=False):
    beta = 1
    omega0 =20
    w = (omega0**2 - beta**2)**0.5
    tau = 2*np.pi/w


    def G(t,t0):
        if t>t0:
            return np.exp(-beta*(t-t0))*np.sin(w*(t-t0))
        else:
            return 0

    def all_G(t):
        sum = 0
        for n in range(100):
            sum += G(t,n/2*tau) * A**n
        return sum

    t = np.linspace(0,10,1000)
    x = []
    y = []

    global A_crit
    A_crit = np.exp(-beta*tau/2)
    for i in range(len(t)):
        x.append(all_G(t[i]))
        y.append( np.exp(-beta*(t[i])))
    
    if make_plot:
        plt.plot(t,x,label = 'x(t) for A='+str(A))
        plt.plot(t,y,label = r'y(t)=$e^{-\beta t}$')
        plt.xlabel('time')
        plt.ylabel('displacement')
        plt.legend()
        plt.show()

    return t,x,y

plot_separately = False
t,x1,y = displacement(0.8,plot_separately)
t,x2,y = displacement(0.85,plot_separately)
t,x3,y = displacement(0.95,plot_separately)

y2 = []


for i in range(len(t)):
    beta = 1
    omega0 =20
    w = (omega0**2 - beta**2)**0.5
    tau = 2*np.pi/w
    y2.append(np.exp(2*np.log(0.95)*(t[i])/tau))

plt.plot(t,x1,label = 'x(t) for A='+str(0.8))
plt.plot(t,x2,label = 'x(t) for A='+str(0.85))
plt.plot(t,x3,label = 'x(t) for A='+str(0.95))
plt.plot(t,y,label = r'y(t)=$e^{-\beta t}$')
plt.plot(t,y2,label = r'y2(t)=$e^{2log(0.95)t/ \tau}$')
plt.xlabel('time')
plt.ylabel('displacement')
plt.legend(fontsize=20)
plt.savefig('FIG_HW5_1.png')
plt.show()