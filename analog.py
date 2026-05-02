import numpy as np
import matplotlib.pyplot as plt
import control as ct
import scipy.signal as signal

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

def P1_value(r_pi2, C_2):
    return(-1/(r_pi2 * C_2))

def P2_value(r_pi1, R_1, R_2, R_s, C_1):
    return (-((R_1 + R_2)*(R_s + r_pi1) + R_s*r_pi1)/(R_s*r_pi1*(R_1 + R_s)*C_1))

def Phantomzero(Omega_0, Pol1, Pol2):
    return( - (Omega_0 * Omega_0) / ((np.sqrt(2)*Omega_0) + Pol1 + Pol2))

def Phantompole(R_1, R_2, Cph):
    return(- (R_1 + R_2) / (R_1 * R_2 * Cph))

def PhantomKondensator(Nph, R2):
    return(- 1 / (R2 * Nph))

def butterwoth(AB_0, P1, P2):
    LP = abs((1 - AB_0 * P1 * P2))
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


    #################### BODE_DIAGRAM #########################
def bode(AB_0, At_inf, P1, P2, P_ph):
    norm_factor = 1e6 #För att dela ned för att undvika RunTimeWaring
    P1 = P1 / norm_factor
    P2 = P2 / norm_factor
    P_ph = P_ph / norm_factor
    #Överföringsfunktioner:
    K_uncomp = AB_0 * P1 * P2
    sys_Abeta_u = signal.ZerosPolesGain([], [P1, P2], K_uncomp).to_tf()

    #Sluten överföring
    num_At_u = At_inf * sys_Abeta_u.num #Täljare
    den_At_u = np.polyadd(sys_Abeta_u.den, sys_Abeta_u.num) #Nämnare
    sys_At_u = signal.TransferFunction(num_At_u, den_At_u) #Skapa överförningfunk med signal

    #Kompenserade överförningsfunk
    K_comp = np.abs(K_uncomp / P_ph)
    sys_Abeta_c = signal.ZerosPolesGain([P_ph], [P1, P2], K_comp).to_tf()

    num_At_c = At_inf * sys_Abeta_c.num #Täljare
    den_At_c = np.polyadd(sys_Abeta_c.den, sys_Abeta_c.num) #Nämnare
    sys_At_c = signal.TransferFunction(num_At_c, den_At_c) #Skapa funk med signal

    w_bode = np.logspace(1, 7, 1000) * 2 * np.pi / norm_factor

    fig = plt.figure(figsize=(14, 8))

    w_u, mag_u, phase_u = signal.bode(sys_Abeta_u, w=w_bode)
    w_c, mag_c, phase_c = signal.bode(sys_Abeta_c, w=w_bode)
    f_hz = w_u * norm_factor / (2 * np.pi)

    ax1 = plt.subplot(2, 2, 1)
    ax1.semilogx(f_hz, mag_u, label='Okompenserad', linestyle='--')
    ax1.semilogx(f_hz, mag_c, label='Kompenserad (Fantomnolla)')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax1.set_title(r'Bode Amplitud: Slingförstärkning $A\beta(s)$')
    ax1.set_ylabel('Amplitud [dB]')
    ax1.set_xlabel('Frekvens [Hz]')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    ax2 = plt.subplot(2, 2, 3)
    ax2.semilogx(f_hz, phase_u, label='Okompenserad', linestyle='--')
    ax2.semilogx(f_hz, phase_c, label='Kompenserad')
    ax2.axhline(-180, color='red', linewidth=0.8, linestyle=':', label='-180° Gräns') 
    ax2.set_title(r'Bode Fas: Slingförstärkning $A\beta(s)$')
    ax2.set_xlabel('Frekvens [Hz]')
    ax2.set_ylabel('Fas [Grader]')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()

    # --- Bode-diagram: Sluten förstärkning At(s) ---
    w_At_u, mag_At_u, _ = signal.bode(sys_At_u, w=w_bode)
    w_At_c, mag_At_c, _ = signal.bode(sys_At_c, w=w_bode)

    ax3 = plt.subplot(2, 2, 2)
    ax3.semilogx(f_hz, mag_At_u, label='Okompenserad', linestyle='--')
    ax3.semilogx(f_hz, mag_At_c, label='Kompenserad (Butterworth-mål)')
    ax3.axhline(20*np.log10(At_inf), color='black', linewidth=0.8, linestyle=':', label=r'Idealt $A_t^\infty$')
    ax3.axhline(20*np.log10(At_inf) - 3, color='red', linewidth=0.8, linestyle=':', label='-3 dB Gräns')
    ax3.set_title(r'Bode Amplitud: Sluten Förstärkning $A_t(s)$')
    ax3.set_xlabel('Frekvens [Hz]')
    ax3.set_ylabel('Amplitud [dB]')
    ax3.grid(True, which="both", ls="-", alpha=0.5)
    ax3.legend()

# --- Stegsvar ---
    t_vec = np.linspace(0, 1000, 1000) # 0.5 ms tidsfönster
    t_u, y_u = signal.step(sys_At_u, T=t_vec)
    t_c, y_c = signal.step(sys_At_c, T=t_vec)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t_u, y_u, label='Okompenserad', linestyle='--')
    ax4.plot(t_c, y_c, label='Kompenserad')
    ax4.axhline(At_inf, color='black', linewidth=0.8, linestyle=':', label='Slutvärde')
    ax4.set_title('Stegsvar: Systemets Insvingning')
    ax4.set_xlabel(r'Tid [$\mu$s]')
    ax4.set_ylabel('Amplitud [A/A]')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()
    

def main():
    r_pi1 = 2 * r_pi_calc(B_f, I_C, V_T)
    At_inf = 1 + R_2 / R_1 #Inte helt säker på detta
    AB_0 = value_AB_0(r_pi1, B_f, R_s, R_1, R_2)
    r_pi2 = r_pi_calc(B_f, 3e-3, V_T)
    P1 = P1_value(r_pi2, C_2)
    P2 = P2_value(r_pi1, R_1, R_2, R_s, C_1)
    LP, omega_0, pp1, pp2, sum_sling, sum_sys = butterwoth(AB_0, P1, P2)
    N_ph = Phantomzero(omega_0, P1, P2)
    C_ph = PhantomKondensator(R_2, N_ph)
    Z_ph = Phantompole(R_1, R_2, C_ph)

    print("_"*100)
    print(f"Beräknat värde för AB(0): {AB_0:.2f} \n")
    print(f"Beräknade sling poler:\nP1: {P1:.2f} \nP2: {P2:.2f}")
    print(f"LP: {LP:.2e} rad/s\nω₀: {omega_0:.2e} rad/s")
    print(f"Butterworth Placering ger:\n P'1: {complex(pp1):.2f}\n P'2: {complex(pp2):2f}\nSumma slingpoler: {sum_sling:.2f}\nSumma systempoler: {sum_sys:.2f}")
    if (sum_sling > sum_sys):
        print("=> Endast domminata poler ingår")
    else:
        print("Icke-dominata poler ingår")
    print("_"*25, "Phantom-uträkningar", "_"*25)
    print(f"N_ph = {N_ph:.2e} rad/s")
    print(f"C_ph = {C_ph:.2e} F")
    print(f"Z_ph = {Z_ph:.2e} rad/s")
    print(f"\u03B4 (effektivitet) = {Z_ph / N_ph :.1f}")
    
    rlocus(P1, P2, pp1, pp2)
    
    print("_"*100)
    bode(-AB_0, At_inf, P1, P2, Z_ph)

main()