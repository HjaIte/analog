import numpy as np
import matplotlib.pyplot as plt
import control as ct

#Komponeter:
B_f = 200
R_1 = 1000
R_2 = 10000
R_s = 10e3
R_L = 100
C_1 = 100e-9
C_2 = 2.2e-6

#Konstanter
V_T = 25e-3
I_C = 3.5e-3

def r_pi_calc(B_f,I_C,V_T):
    #Values to calc r_pi
    r_pi = B_f*V_T/I_C
    return r_pi

def value_AB_0(r_pi1, B_f, R_s, R_1, R_2):

    #Expression for AB(0)

    AB_0 = -B_f**2 * R_s * R_1/((R_1 + R_2)*(r_pi1+R_s)+R_s*r_pi1)

    return AB_0

def P1_value(r_pi1, C_1):
    return(-1/(r_pi1 * C_1))

def P2_value(r_pi1, R_1, R_2, R_s, C_1):
    return (-((R_1 + R_2)*(R_s + r_pi1) + R_s*r_pi1)/(R_s*r_pi1*(R_1 + R_s)*C_1))

def butterwoth(AB_0, P1, P2):
    LP = abs((1 - AB_0) * P1 * P2)
    omega_0 = np.sqrt(LP)
    pp1 = -omega_0 * (1/(np.sqrt(2)) + 1j/(np.sqrt(2)))
    pp2 = -omega_0 * (1/(np.sqrt(2)) - 1j/(np.sqrt(2)))
    sum_sling = P1 + P2
    sum_sys = -np.sqrt(2)*omega_0
    return(LP, omega_0, pp1, pp2, sum_sling, sum_sys)

def rlocus(P1, P2, pp1, pp2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 14))

    #----------------------------ROOOT LOCUS UTNA FANTOMNOLLAN--------------------------
    radius = abs(2.11e+05)

    theta = np.linspace(0, 2 * np.pi, 300)

    den = np.real(np.poly([P1, P2]))
    num1 = [1]
    sys1 = ct.tf(num1, den)

    K_required = 0.0


    ct.root_locus_plot(sys1, ax=ax1)

    ax1.plot([0, np.real(P1)], [0, np.imag(P1)], 'g:', linewidth=0.8, alpha=0.6)
    ax1.plot([0, np.real(P2)], [0, np.imag(P2)], 'g:', linewidth=0.8, alpha=0.6)

    ax1.plot(np.real(P1), np.imag(P1), 'g*', markersize=14,
            label=f"Butterworth target (K ≈ {K_required:.2f})")
    ax1.plot(np.real(P2), np.imag(P2), 'g*', markersize=14)


    ax1.grid(True, linestyle='--')
    #-----------------------------------ROOT LOCUS FÖR FANTIOMNOLLAN-------------------------------

    den = np.real(np.poly([pp1, pp2]))
    nph = 144960.6
    phzero = [1/nph, 1]
    sys2 = ct.tf(phzero, den)

    K_required = 0.0

    ct.root_locus_plot(sys2, ax=ax2)

    #plotta slingpolerna:

    ax2.plot([0, np.real(P1)], [0, np.imag(P1)], 'g:', linewidth=0.8, alpha=0.6)
    ax2.plot([0, np.real(P2)], [0, np.imag(P2)], 'g:', linewidth=0.8, alpha=0.6)

    ax2.plot(np.real(P1), np.imag(P1), 'g*', markersize=14,
            label=f"Butterworth target (K ≈ {K_required:.2f})")
    ax2.plot(np.real(P2), np.imag(P2), 'g*', markersize=14)


    #plotta systempolerna
    ax2.plot(radius * np.cos(theta), radius * np.sin(theta), 
            'b--', linewidth=1, alpha=0.5, label=f"ω₀ = {radius:.2e} rad/s")

    ax2.plot([0, np.real(pp1)], [0, np.imag(pp1)], 'g:', linewidth=0.8, alpha=0.6)
    ax2.plot([0, np.real(pp2)], [0, np.imag(pp2)], 'g:', linewidth=0.8, alpha=0.6)

    ax2.plot(np.real(pp1), np.imag(pp1), 'g*', markersize=14,
            label=f"Butterworth target (K ≈ {K_required:.2f})")
    ax2.plot(np.real(pp2), np.imag(pp2), 'g*', markersize=14)

    ax2.grid(True, linestyle='--')
    ax2.set_xlim([-3.25*10**5, 2000])
    plt.show()

def main():
    r_pi1 = 2*r_pi_calc(B_f, I_C, V_T)
    AB_0 = value_AB_0(r_pi1, B_f, R_s, R_1, R_2)
    P1 = P1_value(r_pi1, C_1)
    P2 = P2_value(r_pi1, R_1, R_2, R_s, C_1)
    LP, omega_0, pp1, pp2, sum_sling, sum_sys = butterwoth(AB_0, P1, P2)
    print("_"*100)
    print(f"Beräknat värde för AB(0): {AB_0:.2f} \n")
    print(f"Beräknade sling poler:\nP1: {P1:.2f} \nP2: {P2:.2f}")
    print(f"LP: {LP:.2e} rad/s\nω₀: {omega_0:.2e} rad/s")
    print(f"Butterworth Placering ger:\n P'1: {complex(pp1):.2f}\n P'2: {complex(pp2):2f}\nSumma slingpoler: {sum_sling:.2f}\nSumma systempoler: {sum_sys:.2f}")
    if (sum_sling > sum_sys):
        print("=> Endast domminata poler ingår")
    else:
        print("Icke-dominata poler ingår")
    rlocus(P1, P2, pp1, pp2)
    
    print("_"*100)

main()