# Required libraries

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, lsq_linear
from scipy.interpolate import interp1d, PchipInterpolator

from scipy.stats import linregress
import control as ct
from .thermalsys import ThermalSystemIoT, PATH_DATA, PATH_DEFAULT, FONT_SIZE
from .controlsys import  long2hex, float2hex, hex2long, set_pid, hex2float, display_immediately
import json
from math import ceil
from queue import Queue
from pathlib import Path
import csv
from pathlib import Path
import time








def read_csv_file3(filepath = PATH_DATA + 'Thermal_prbs_open_exp.csv'):
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        num_line = 0
        t = []
        u = []
        y = []
        for row in reader:
            if num_line != 0:
               t.append(float(row[0]))
               u.append(float(row[1]))
               y.append(float(row[2]))
            num_line += 1
        return np.array(t), np.array(u), np.array(y)

def read_csv_file2(filepath = PATH_DATA + 'Thermal_static_gain_response.csv'):
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        num_line = 0
        u = []
        y = []
        for row in reader:
            if num_line != 0:
               u.append(float(row[0]))
               y.append(float(row[1]))
            num_line += 1
        return u, y


def step_closed_staticgain(system, r0=40, r1=40, t0=0, t1=60):
    def step_message(system, userdata, message):
        q.put(message)

    low_val = r0
    high_val = r1
    low_time = t0
    high_time = t1
    topic_pub = system.codes["USER_SYS_STEP_CLOSED"]
    topic_sub = system.codes["SYS_USER_SIGNALS_CLOSED"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    points_high = round(high_time / sampling_time)
    points_low = round(low_time / sampling_time)
    points = points_low + points_high
    points_low_hex = long2hex(points_low)
    points_high_hex = long2hex(points_high)
    low_val_hex = float2hex(low_val)
    high_val_hex = float2hex(high_val)
    message = json.dumps({"low_val": low_val_hex,
                          "high_val": high_val_hex,
                          "points_low": points_low_hex,
                          "points_high": points_high_hex,
                          })

    system.client.on_message = step_message
    system.connect()
    system.subscribe(topic_sub)
    system.publish(topic_pub, message)
    q = Queue()
    y = []
    u = []
    t = []


    with plt.ioff():
        fig = plt.gcf()
        ax = fig.get_axes()[0]

    line_y, = ax.plot(t, y, linestyle = 'solid', color="#0044AA60", linewidth=1)
    line_u, = ax.plot(t, u, linestyle='solid', color="#ff000060", linewidth=1)


    n = -1
    sync = False


    while n < points:
        try:
            message = q.get(True, 20)
        except:
            system.disconnect()
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        n_hex = str(msg_dict["np"])
        n = hex2long(n_hex)
        if n == 0:
            sync = True

        if sync:
            t_curr = n * sampling_time
            t.append(t_curr)
            y_curr = hex2float(msg_dict["y"])
            y.append(y_curr)
            u_curr = hex2float(msg_dict["u"])
            u.append(u_curr)
            line_y.set_data(t, y)
            line_u.set_data(t, u)
            ax.legend([line_y, line_u], [f'Temperature: {y_curr:0.3f} $(~^o C)$', fr'Power input: {u_curr:0.2f} ($\%$)'], fontsize=12,
                      loc="lower right")
            fig.canvas.draw()
            time.sleep(0.1)

    line_y.set_data([], [])
    line_u.set_data([], [])
    system.disconnect()
    return u, y




def get_static_model (system, step = 5, usefile= False):    
    """
    Sets up and conducts a sequence of experiments to obtain the static gain model for a temperature control system.

    This function uses a closed-loop control approach to determine the static model. The setpoint begins
    at 40 degrees Celsius and increments by the value specified in the `step` parameter. For each setpoint,
    once the system output (temperature) reaches the steady-state value, the control signal required to
    maintain that state is recorded, generating a data point for the static model. This process covers the
    range from 40° to 60° Celsius.

    
    .. image:: _static/get_static_model.png
        :alt: open loop response of the thermal system
        :align: center
        :width: 500px

    Parameters
    ----------
    system : ThermalSystemIoT
        The thermal system under test.
    step : float, optional
        The increment between setpoints. Defaults to 5.
    usefile : bool, optional
        If True, reads data from a prerecorded file instead of conducting a new experiment. Defaults to False.

    Returns
    -------
    power_values : list of floats
        A list of the power input values required to achieve each steady-state temperature.
    temperatures : list of floats
        A list containing the steady-state temperatures.

    Notes
    -----
    - Data is stored in a CSV file located at `/experiment_files/Thermal_static_gain_model.csv`.

    Example
    -------

    First, ensure that you have already imported the unthermal package and defined the thermal system as follows:
        
    >>> import unthermal as ter
    >>> my_system = ter.ThermalSystemIoT(plant_number="XXXX", broker_address="192.168.1.100") 
    
    Then, obtain the static model of the thermal system as follows:

    >>> power, temperature = ter.get_static_model(my_system, step=5)
    """




    # This is the configuration for the figure displayed while acquiring data
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(10, 6))
    display_immediately(fig)
    fig.set_facecolor('#ffffff')
    ax.set_title('Static gain response experiment for UNThermalSystem')
    ax.set_xlabel(r'Percent of Power ($\%$) / Seconds for the current experiment')
    ax.set_ylabel(r'Steady state temperature (C)')
    ax.set_facecolor('#f4eed7')
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    line_exp, = ax.plot([], [], color="#00aa00", linewidth=1, marker=r"$\circ$", markeredgewidth=0.1)

    if not usefile:
        yee = []
        uee = []
        exp = []
        y_test = np.arange(40,100, step)

        for yi in y_test:
            u = []
            while not u:
                try:
                    u, y = step_closed_staticgain(system, r0=yi, r1=yi, t0=0, t1=60)
                except:
                    time.sleep(5)
            yf = np.mean(y[-12:])
            uf = np.mean(u[-12:])
            exp.append([uf, yf])
            yee.append(yf)
            uee.append(uf)
            line_exp.set_data(uee, yee)
            fig.canvas.draw()
            time.sleep(1)

        np.savetxt(PATH_DEFAULT + "Thermal_static_gain_response.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                   header='uee,yee')
        np.savetxt(PATH_DATA + "Thermal_static_gain_response.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                   header='uee,yee')

    uee, yee = read_csv_file2()
    line_exp, = ax.plot(uee, yee, color="#00aa00", linewidth=1, marker=r"$\circ$", markeredgewidth=0.1)
    res = linregress(uee, yee, alternative='greater')
    m = res.slope
    b = res.intercept
    ymod = m * np.array(uee) + b
    line_mod, = ax.plot(uee, ymod, color="#0088aaff", linewidth=1.5)
    strmodel = r"Model:  $T_{ee}= m\,u_{ee} + b=$" + f"{m:0.2f}" + r"$\,u_{ee}+$" + f"{b:0.2f}"
    ax.legend([line_exp, line_mod], ['Data', strmodel], fontsize=12)
    fig.canvas.draw()
    system.disconnect()
    return uee, yee



def prbs_open(system, yop=50, amplitude=4, stab_time=60, uee_time=10, divider = 20):
    def pbrs_message(system, userdata, message):
        q.put(message)
    op_point = yop
    topic_pub = system.codes["USER_SYS_PRBS_OPEN"]
    topic_sub = system.codes["SYS_USER_SIGNALS_OPEN"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    amplitude_hex = float2hex(amplitude)
    op_point_hex = float2hex(op_point)
    stab_points = ceil(stab_time / sampling_time)
    uee_points = ceil(uee_time / sampling_time)
    stab_points_hex = long2hex(stab_points)
    uee_points_hex = long2hex(uee_points)
    divider_hex = long2hex(divider)
    points = divider * 63 + stab_points + uee_points - 1
    message = json.dumps({"peak_amp": amplitude_hex,
                          "op_point": op_point_hex,
                          "stab_points": stab_points_hex,
                          "uee_points": uee_points_hex,
                          "divider": divider_hex
                          })
    system.client.on_message = pbrs_message
    system.connect()
    system.subscribe(topic_sub)
    system.publish(topic_pub, message)
    q = Queue()
    y = []
    u = []
    t = []
    yt = []
    ut = []
    tt = []
    exp = []
    m = 1.2341015052212259
    b = 24.750915901094388
    uf_est = (op_point - b) / m
    percent = 0.2
    ymax = op_point + m*amplitude + 2
    ymin = op_point - m*amplitude - 2
    umax = uf_est + (1 + percent) * amplitude
    umin =  np.min([0, uf_est - (1 + percent) * amplitude - 5.0])

    # Setting the graphics configuration for visualizing the experiment


    with plt.ioff():
        fig, (yax, uax) = plt.subplots(nrows=2, ncols=1, width_ratios=[1], height_ratios=[3, 1], figsize=(10, 6))
    display_immediately(fig)

    # display config
    fig.set_facecolor('#ffffff')

    yax.set_title(f'PRBS identification with {points:d} samples and a duration of {points * sampling_time: 0.2f} seconds')
    yax.set_ylabel(r'Temperature ($~^oC$)')
    yax.grid(True);
    yax.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    yax.set_facecolor('#f4eed7')
    yax.set_xlim(0, sampling_time * points)
    yax.set_ylim(ymin, ymax)
    uax.set_xlabel('Time (s)')
    uax.set_ylabel(r'Power Input ($\%$)')
    uax.grid(True);
    uax.set_facecolor('#d7f4ee')
    uax.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    uax.set_xlim(0, sampling_time * points)
    uax.set_ylim(umin, umax)

    line_y, = yax.plot(t, y, color="#ff6680")
    line_u, = uax.plot(t, u, color="#00d4aa")
    line_yt, = yax.plot(t, yt, color="#d40055")
    line_ut, = uax.plot(t, ut, color="#338000")

    n = -1
    sync = False
    tstep = sampling_time * (stab_points + uee_points)

    while n < points:
        try:
            message = q.get(True, 20)
        except:
            system.disconnect()
            raise TimeoutError("The connection has been lost. Please try again")


        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        n_hex = str(msg_dict["np"])
        n = hex2long(n_hex)
        if n == 0:
            sync = True

        if sync:
            if n <= stab_points + uee_points:
                t_curr = n*sampling_time
                t.append(t_curr)
                y_curr = hex2float(msg_dict["y"])
                y.append(y_curr)
                u_curr = hex2float(msg_dict["u"])
                u.append(u_curr)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                yax.legend([line_y], [ f'Current Temperature: {y_curr: 0.2f}$~^oC$'],
                           fontsize=FONT_SIZE, loc="upper left")
                uax.legend([line_u], [f'$u(t):$ {u_curr: 0.1f}'], fontsize=FONT_SIZE)
                if n == stab_points + uee_points:
                    tt.append(t_curr)
                    yt.append(y_curr)
                    ut.append(u_curr)
                    exp.append([0, u_curr, y_curr])
                    uax.set_ylim(u_curr - amplitude - 1, umax + amplitude + 1)

            else:
                tt_curr = n * sampling_time
                tt.append(tt_curr)
                yt_curr = hex2float(msg_dict["y"])
                yt.append(yt_curr)
                ut_curr = hex2float(msg_dict["u"])
                ut.append(ut_curr)
                uax.set_ylim(u_curr - amplitude - 1, umax + amplitude + 1)
                line_yt.set_data(tt, yt)
                line_ut.set_data(tt, ut)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                yax.legend([line_yt], [ f'Current Temperature: {yt_curr: 0.2f}$~^oC$'],
                           fontsize=FONT_SIZE, loc="upper left")
                uax.legend([line_ut], [f'$u(t):$ {ut_curr: 0.1f}'], fontsize=FONT_SIZE)
                exp.append([tt_curr - tstep, ut_curr, yt_curr])
                np.savetxt(PATH_DEFAULT + "Thermal_prbs_open_exp.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                           header='t,u,y')
                np.savetxt(PATH_DATA + "Thermal_prbs_open_exp.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                           header='t,u,y')

            fig.canvas.draw()
            time.sleep(0.1)


    system.disconnect()
    print("PBRS experiment completed\n")
    return tt, ut, yt



def get_models_prbs(system, yop = 50, amplitude= 4, usefile = False):

    r"""
    Identifies and compares First-Order (FO) and First-Order with Time Delay (FOTD) models 
    of a thermal system using a Pseudo-Random Binary Sequence (PRBS) input signal.

    - The FO model has the form: :math:`G_1(s) = \dfrac{\alpha}{\tau\, s + 1}`.

    - The FOTD model has the form: :math:`G_2(s) = \dfrac{\alpha}{\tau\, s + 1} e^{-Ls}`.

    This function conducts an open-loop experiment by applying a PRBS signal. 
    Prior to applying the signal, the operating point is stabilized using a pre-tuned 
    PI controller. The identified FO and FOTD models are obtained through optimization 
    and compared against experimental data.


    The following figure depicts a typical experiment using a PRBS input: 

    .. image:: _static/prbs_response.png
        :alt: Closed-loop response of the thermal system
        :align: center
        :width: 600px    

    Parameters
    ----------
    system : ThermalSystemIoT
        The IoT-enabled thermal system object used for the experiment and model identification.
    yop : float
        The operating point of the thermal system in degrees Celsius. Must be less than 100°C. Default is 50°C.
    amplitude : float
        Amplitude of the PRBS signal as a percentage of the system's maximum input power. Default is 4%.
    usefile : bool, optional
        If True, uses data from a previous experiment instead of conducting a new PRBS experiment. Default is False.

    Returns
    -------
    G1 : TransferFunction
        The identified FO model as a transfer function.
    G2 : TransferFunction
        The identified FOTD model as a transfer function.
    tau : float
        The time delay parameter of the FOTD model.

    Notes
    -----
    - Experimental data and identified model parameters are saved as CSV files:
        
        - FO model: `Thermal_fo_model_pbrs.csv`
        - FOTD model: `Thermal_fotd_model_pbrs.csv`
    
    - Visualization includes plots of experimental data and the step responses of the FO and FOTD models.
      Performance metrics such as FIT scores are also displayed.

    Raises
    ------
    ValueError
        If `yop` is greater than or equal to 100°C.
    TimeoutError
        If the connection with the system is lost during the PRBS signal generation.

    Example
    -------
    Ensure the system object is initialized:

    >>> import unthermal as ter
    >>> my_system = ter.ThermalSystemIoT(plant_number="XXXX", broker_address="192.168.1.100")

    Now, to identify models using a PRBS signal:

    >>> G1, G2, tau = get_models_prbs(system=my_system, yop=60, amplitude=5)

    To use previous experimental data:

    >>> G1, G2, tau = get_models_prbs(system=my_system, yop=60, usefile=True)
    """

    

    norm = np.linalg.norm

    def simulate_fo_model(x):
        # this function simulates the model
        alpha, tau = x
        um_i = um_interp(t_interp)
        s = ct.TransferFunction.s
        G = alpha / (tau*s + 1)
        tsim, ysim = ct.forced_response(G, t_interp, um_i)
        return G, ysim


    def simulate_fotd_model(x):
        # this function simulates the model
        alpha, tau1, tau2 = x
        um_delay = um_interp(np.array(t_interp) - tau2)
        s = ct.TransferFunction.s
        G = alpha / ( tau1*s + 1)
        tsim, ysim = ct.forced_response(G, t_interp, um_delay)
        return G, ysim

    def objective_fo(x):
        # simulate model
        G, ysim = simulate_fo_model(x)
        # return objective
        return norm(ysim - ym)


    def objective_fotd(x):
        # simulate model
        G, ysim = simulate_fotd_model(x)
        # return objective
        return norm(ysim - ym)


    if  (yop >= 100):
         raise ValueError(f"The maximum temperature for this system is 100 degrees celsius")

    if not usefile:
        try:
            prbs_open(system, yop=yop, amplitude=amplitude, stab_time=120, uee_time=10, divider=25)
        except:
            print("The connection has been lost. We use partial data.")

    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    t, u, y = read_csv_file3()


    ymean = np.mean(y)
    um = np.array(u) - u[0]
    ym = np.array(y) - ymean

    um_interp = interp1d(t, um, fill_value = (0,0 ), bounds_error=False)
    ym_interp = interp1d(t, ym)
    t_interp = np.arange(0, t[-1] + sampling_time, sampling_time)

    # we now obtain the model parameter's
    alpha_0 = 1.2
    tau1_0 = 60
    tau2_0 = 2
    x02 = [alpha_0, tau1_0, tau2_0]
    x01 = [alpha_0, tau1_0]
    
    #These are the bounds for alpha, tau1 and tau2
    bounds2 = [(0.8, 2), (1 , 150), (0.4, 10)]
    bounds1 = bounds2[0:2]
    
    # Now we run the optimization algorithm
    print(f'Starting optimization for first order model...\n\t Initial cost function: {objective_fo(x01):.2f}' )
    solution = minimize(objective_fo, x0=x01, bounds= bounds1)
    xmin = solution.x
    alpha, tau = xmin
    fmin = objective_fo(xmin)
    print(f'\t Final cost function: {fmin:.2f}' )
    print(f'alpha={alpha:.2f} \t tau1={tau:.3f}')

    # We compare the experimental data with the simulation model
    G1, ysim1 = simulate_fo_model(xmin)
    r1 = 100*(1 - norm(ym - ysim1) / norm(ym))

    print(f'\n\nStarting optimization for second order model...\n\t Initial cost function: {objective_fotd(x02):.2f}' )
    solution = minimize(objective_fotd, x0=x02, bounds= bounds2)
    xmin = solution.x
    alpha2, tau1, tau2 = xmin
    fmin = objective_fotd(xmin)
    print(f'\t Final cost function: {fmin:.2f}' )
    print(f'alpha={alpha2:.2f} \t tau1={tau1:.3f} \t tau2={tau2:.3f}')
    print(f'\nFO model: G1={alpha :0.2f}/({tau:0.2f} s + 1)')
    print(f'FOTD model: G2={alpha2 :0.2f} * exp(-{tau2:0.2f}) /({tau1:0.2f} s + 1)')

    # We compare the experimental data with the simulation model
    G2, ysim2 = simulate_fotd_model(xmin)
    r2 = 100*(1 - norm(ym - ysim2) / norm(y - np.mean(y)))

    # we calculate the step response from the model
    # now we compare the model with the experimental data
    # fig, (ay, au) = plt.subplots(nrows=2, ncols=1, width_ratios=[1], height_ratios=[3, 1], figsize=(10, 6))
    # fig.set_facecolor('#ffffff')


    if usefile:
        with plt.ioff():
            #plt.close("all")
            fig, (ay, au) = plt.subplots(nrows=2, ncols=1, width_ratios=[1], height_ratios=[4, 1], figsize=(10, 6))
            fig.set_facecolor('#ffffff')
        display_immediately(fig)


    else:
        with plt.ioff():
            fig = plt.gcf()
            ay, au = fig.get_axes()
            ay.cla()
            au.cla()




    # settings for the upper axes, depicting the model and speed data
    #ay.set_title('Data and estimated second order model for UNDCMotor')
    ay.set_ylabel(r'Celsius degrees ($^oC$)')
    ay.grid(True)
    ay.grid(color='#1a1a1a40', linestyle='--', linewidth=0.125)
    ay.set_facecolor('#f4eed7')
    ay.set_xlim(0, t[-1])
    ay.set_xlabel('Time (seconds)')

    # settings for the lower axes, depicting the input
    au.set_xlim(0, t[-1])
    au.set_facecolor('#d7f4ee')
    au.grid(color='#1a1a1a40', linestyle='--', linewidth=0.125)

    au.set_xlabel('Time (seconds)')
    au.set_ylabel('Power (%)')

    line_exp, = ay.plot(t, ym + ymean, color="#0088aaAF", linewidth=1.5, linestyle=(0, (1, 1)))
    line_model1, = ay.plot(t, ysim1 + ymean, color="#00AA44ff", linewidth=1.5, )
    line_model2, = ay.plot(t, ysim2 + ymean, color="#d45500ff", linewidth=1.5, )

    line_u, = au.plot(t, u, color="#0066ffff")

    modelstr1 = r"FO Model:     $G_1(s) = \frac{%0.2f }{%0.2f\,s+1}$  ($FIT = %0.1f$" % (alpha, tau, r1) + "%)"
    modelstr2 = r"FOTD Model: $G_2(s) = \frac{%0.2f  }{%0.2f\,s+1}\,e^{-%0.2f\,s}$  ($FIT = %0.1f$"%(alpha2, tau1, tau2, r2) + "%)"
    ay.set_title("Comparison of FO and FOTD models estimated with a PRBS signal at the operation point $y_{OP}=%0.1f^oC$"%y[0])
    ay.legend([line_exp, line_model1, line_model2], ['Data', modelstr1, modelstr2],
              fontsize=FONT_SIZE, loc = 'lower left',framealpha=0.95)
    au.legend([line_u], ['PRBS Input'], fontsize= FONT_SIZE)
    fig.canvas.draw()

    fo_model = [[alpha, tau]]
    fotd_model = [[alpha2, tau1, tau2]]

    np.savetxt(PATH_DEFAULT + "Thermal_fo_model_pbrs.csv", fo_model, delimiter=",",
               fmt="%0.8f", comments="", header='alpha, tau')

    np.savetxt(PATH_DEFAULT + "Thermal_fotd_model_pbrs.csv", fotd_model, delimiter=",",
               fmt="%0.8f", comments="", header='alpha2, tau1, tau2')
    np.savetxt(PATH_DATA + "Thermal_fo_model_pbrs.csv", fo_model, delimiter=",",
               fmt="%0.8f", comments="", header='alpha, tau')

    np.savetxt(PATH_DATA + "Thermal_fotd_model_pbrs.csv", fotd_model, delimiter=",",
               fmt="%0.8f", comments="", header='alpha2, tau1, tau2')

    system.disconnect()

    return G1, G2, tau2

def step_open(system, yop=50, amplitude=10, t1=360, stab_time=60, uee_time=10):
    """  
    This function sets up and performs an experiment to observe how the system's temperature responds to a specified step
    change in the power input. The experiment first adjust the operating point by means of a
    pretuned PI controller.

    .. image:: _static/step_open.png
        :alt: open loop response of the thermal system
        :align: center
        :width: 500px

    The figure illustrates parameters `yop`, `t1` and `amplitude`, which are described below.


    Parameters
    ----------
    `system` : ThermalSystemIoT object
        The IoT thermal system in which the experiment will be performed.
    `yop` : float, optional
        The operation point temperature at which you want to conduct the experiment. Default is 50°
    `t1` : float, optional
        The duration of the experiment in seconds since the step changes. Default is 360.
    `amplitude` : float, optional
        The amplitude of the change in the step input measured from uop, which is the power control signal needed to produce
        a steady state temperature of`yop`. Default is 10.
    `stab_time` : float, optional
        The time, in seconds. that the controlled system requires for reaching `yop`, in seconds. Default is 60.
    `uee_time` : float, optional
        During this period, in seconds, the input signal is held constant before the step change is applied. The default value is 10.

    Raises
    ------
    ValueError: If `yop` is set above 100 degrees Celsius, considering it the safety limit for student's operation.

    Returns
    -------
    t : list of floats
        The time vector.
    u : list of floats
        The power input vector.
    y : list of floats
        The temperature output vector.


    Example
    -------
    First, ensure that you have already imported the unthermal package and defined the thermal system as follows:

    >>> import unthermal as ter
    >>> my_system = ter.ThermalSystemIoT(plant_number="XXXX", broker_address="192.168.1.100") 

    Then, the open loop step response of the thermal system is obtained as follows:

    >>> t, u, y = ter.step_open(my_system, yop=60, t1=600, amplitude=10)
    """
    
    def step_message(system, userdata, message):
        q.put(message)

    if  (yop >= 100):
         raise ValueError(f"The maximum temperature for this system is 100 degrees celsius")

    op_point = yop
    high_time = t1
    topic_pub = system.codes["USER_SYS_STEP_OPEN"]
    topic_sub = system.codes["SYS_USER_SIGNALS_OPEN"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    amp_hex = float2hex(amplitude)
    points_high = ceil(high_time / sampling_time)
    points_high_hex = long2hex(points_high)
    op_point_hex = float2hex(op_point)
    stab_points = ceil(stab_time / sampling_time)
    uee_points = ceil(uee_time / sampling_time)
    stab_points_hex = long2hex(stab_points)
    uee_points_hex = long2hex(uee_points)
    message = json.dumps({"amplitude": amp_hex,
                          "op_point": op_point_hex,
                          "stab_points": stab_points_hex,
                          "uee_points": uee_points_hex,
                          "points_high": points_high_hex,
                          })
    system.client.on_message = step_message
    system.connect()
    system.subscribe(topic_sub)
    system.publish(topic_pub, message)
    q = Queue()
    y = []
    u = []
    t = []
    yt = []
    ut = []
    tt = []
    exp = []
    m = 1.2341015052212259
    b = 24.750915901094388
    uf_est = (op_point - b) / m
    percent = 0.2
    ymax = op_point + m* amplitude + 7
    ymin = op_point - 7
    umax = uf_est + (1 + percent) * amplitude
    umin =  np.min([0, uf_est - (1 + percent) * amplitude])
    points = stab_points + uee_points + points_high

    # display config
    with plt.ioff():
        fig, (yax, uax) = plt.subplots(nrows=2, ncols=1, width_ratios=[1], height_ratios=[3, 1], figsize=(10, 6))



    # display config
    fig.set_facecolor('#ffffff')

    yax.set_title(f'Open loop step experiment with {points_high:d} samples'
                  f' and a duration of {(points_high -1) * sampling_time: 0.2f} seconds', fontsize=FONT_SIZE)
    yax.set_ylabel(r'Temperature ($~^oC$)',fontsize=FONT_SIZE)
    yax.grid(True);
    yax.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    yax.set_facecolor('#f4eed7')
    yax.set_xlim(0, sampling_time * points)
    yax.set_ylim(ymin, ymax)
    yax.set_xlabel('Time (s)', fontsize=FONT_SIZE)
    uax.set_xlabel('Time (s)', fontsize=FONT_SIZE)
    uax.set_ylabel(r'Power Input ($\%$)', fontsize=FONT_SIZE)
    uax.grid(True);
    uax.set_facecolor('#d7f4ee')
    uax.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    uax.set_xlim(0, sampling_time * points)
    uax.set_ylim(umin, umax)

    line_y, = yax.plot(t, y, color="#ff6680")
    line_u, = uax.plot(t, u, color="#00d4aa")
    line_yt, = yax.plot(t, yt, color="#d40055")
    line_ut, = uax.plot(t, ut, color="#338000")
    display_immediately(fig)


    n = -1
    tstep = sampling_time * (stab_points + uee_points + 1)
    sync = False
    while n < points:
        try:
            message = q.get(True, 20)
        except:
            system.disconnect()
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        n_hex = str(msg_dict["np"])
        n = hex2long(n_hex)

        if n == 0:
            sync = True

        if sync:
            if n <= stab_points:
                t_curr = n*sampling_time
                t.append(t_curr)
                y_curr = hex2float(msg_dict["y"])
                y.append(y_curr)
                u_curr = hex2float(msg_dict["u"])
                u.append(u_curr)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                yax.legend([line_y], [ f'Current Temperature: {y_curr: 0.2f}$~^oC$'],
                           fontsize=FONT_SIZE, loc="upper left")
                uax.legend([line_u], [f'$u(t):$ {u_curr: 0.1f}'], fontsize=FONT_SIZE)
                if n == stab_points:
                    uax.set_ylim(u_curr-3, u_curr + 20)
                    fig.canvas.draw()
  
            else:
                tt_curr = n * sampling_time
                tt.append(tt_curr)
                yt_curr = hex2float(msg_dict["y"])
                yt.append(yt_curr)
                ut_curr = hex2float(msg_dict["u"])
                ut.append(ut_curr)

                line_yt.set_data(tt, yt)
                line_ut.set_data(tt, ut)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                yax.legend([line_yt], [ f'Current Temperature: {yt_curr: 0.2f}$~^oC$'],
                           fontsize=FONT_SIZE, loc="upper left")
                uax.legend([line_ut], [f'$u(t):$ {ut_curr: 0.1f} %'], fontsize=FONT_SIZE)
                exp.append([tt_curr - tstep, ut_curr, yt_curr])
                np.savetxt(PATH_DEFAULT + "Thermal_step_open_exp.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                           header='t,u,y')
                np.savetxt(PATH_DATA + "Thermal_step_open_exp.csv", exp, delimiter=",", fmt="%0.8f", comments="",
                           header='t,u,y')
            fig.canvas.draw()
            time.sleep(0.1)


    system.disconnect()
    return tt, ut, yt



def read_fo_model():
    """ Reads an fo model
    :meta private:
    """
    with open(PATH_DATA + 'Thermal_fo_model_pbrs.csv', newline='') as file:
        reader = csv.reader(file)
        # Iterate over each row in the CSV file
        num_line = 0
        for row in reader:
            if num_line != 0:
               alpha = float(row[0])
               tau = float(row[1])
               L = float(row[2])
            num_line += 1
    b = alpha / tau
    a = 1 / tau
    G = ct.tf(b, [1, a])
    return G


def get_fomodel_step(system, yop=50, t1=600, amplitude =10, usefile=False):
    r"""
       Obtains the first order model plus delay (FOTD) from the experimental step response of a thermal system. 
       The FOTD model is represented by the following equation

       .. math::

          G(s) = \frac{\alpha}{\tau s + 1} e^{-Ls}

       where:

       + :math:`\alpha` is the system gain,
       + :math:`\tau` is the time constant, and
       + :math:`L` is the delay.


       This function obtains the experimental step response data from a system, and computes the
       first order time delay (FOTD) model parameters: the system gain (`alpha`), time constant (`tau`), and
       delay (`L`). It can either use real-time data from the system or data read from a previously saved file.

        .. image:: _static/get_fomodel_step.png
           :alt: open loop response of the thermal system
           :align: center
           :width: 500px

       The figure illustrates parameters `yop`, `t1` and `amplitude` which described below.  The experiment first
       adjust the operating point by means of a pretuned PI controller that sets :math:`u_{op}`.

       Parameters
       ----------
       `system` : ThermalSystemIoT object
           The IoT thermal system in which the experiment will be performed.
       `yop` : float, optional
           The operation point temperature at which you require to build the model. Default is 50°
       `t1` : float, optional
           The duration of the experiment in seconds since the step changes. Default is 600.
       `amplitude` : float, optional
           The amplitude of the change in the step input measured from uop, which is the control signal needed to produce
           a steady state temperature of`yop`. Default is 10.
       `usefile` : bool, optional
           If True, data is read from a file instead of conducting a live experiment. Default is False.

       Returns
       -------
       alpha : float
           The gain of the system.
       tau : float
           The time constant of the system.
       L : float
           The delay of the system.


       Example
       -------

       First, ensure that you have already imported the unthermal package and defined the thermal system as follows:

       >>> import unthermal as ter
       >>> my_system = ter.ThermalSystemIoT(plant_number="XXXX", broker_address="192.168.1.100") 

       Then, the FOTD model is obtained, as follows:

       >>> alpha, tau, L = ter.get_fomodel_step(my_system, yop=60, t1=600, amplitude=10)

       Notes
       -------

       + The algorithm uses a four point method and bounded linear least squares for obtaining :math:`\alpha`, 
         :math:`\tau`, and :math:`L`
       + The thermal systems reaches `yop` in closed loop, using a pretuned PID. This speeds up the experiment.
    """


    def compute_step_response(t, alpha, tau, L, delta_u, ya):
    # this function computes the atep responseof a FOTD system given the parameters, alpha, tau and L
        ym = []
        for ti in t:
            yi = alpha * delta_u * (1 - np.exp(-np.max([0, ti-L]) / tau)) + ya
            ym.append(yi)
        return ym



    if  not usefile:
        try:
            step_open(system, yop=yop, amplitude=amplitude, t1=t1, stab_time=60, uee_time=10)
        except:
            print("The connection has been lost. We use partial data.")





    if  yop >= 100:
         raise ValueError(f"The maximum temperature for this system is 100 degrees celsius")

    # Here we read the experiment results
    t, u, y = read_csv_file3(filepath = PATH_DATA + '/Thermal_step_open_exp.csv')

    ind_step = np.diff(u).argmax()
    tend = t[-1]
    # This is the initial step value before the change
    ua = u[ind_step]

    # The last point is the final step value
    ub = u[-1]

    # We calculate the initial temperature value before the step change
    # as the average of the 50 points before the change
    ya = np.mean(y[ind_step-10:ind_step+1])

    # For the final temperature value, we take an average of the last 10 values
    yb = np.mean(y[-10:])

    # We calculate the net change in the step
    delta_u = ub - ua

    # We calculate the change in steady-state output value
    delta_y = yb - ya


    # we calculate the gain of the plant
    delta_y = yb - ya
    delta_u = ub - ua
  

     
    # We compute the normalized response
    y_norm = (y - ya) / delta_y

    # we set the selected points in the normalized curve
    yi_points = [0.05, 0.1, 0.15, 0.2]
    ti_points = np.interp(yi_points, y_norm, t) 


    # this is the matrix for solving the LSQ system
    A = np.array([
        [1, -np.log(1-yi_points[0])],
        [1, -np.log(1-yi_points[1])],
        [1, -np.log(1-yi_points[2])],
        [1, -np.log(1-yi_points[3])],

    ])

    # we solve the LSQ allowing only positive solutions
    limits = [(0,1),(10,100)]
    p = lsq_linear(A, ti_points, bounds=limits)

    # we extract the parameter
    L, tau = p.x

    #compute alpha
    alpha = delta_y / delta_u

    # we conpute the response of the linear model for comparison with data
    ymodel = compute_step_response(t, alpha, tau, L, delta_u, ya)

    if usefile:
        with plt.ioff():
            plt.close("all")
            fig, (ay, au) = plt.subplots(nrows=2, ncols=1, width_ratios=[1], height_ratios=[4, 1], figsize=(10, 6))
        display_immediately(fig)
        fig.set_facecolor('#ffffff')

    else:
        with plt.ioff():
            fig = plt.gcf()
            ay, au = fig.get_axes()
            ay.cla()
            au.cla()

    # settings for the upper axes, depicting the model and speed data
    ay.set_title('Estimated first order model for UNThermal')
    ay.grid(True);
    ay.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    ay.set_facecolor('#f4eed7')
    ay.set_xlim(t[0], tend)
    box = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='white', alpha=0.5)
    ay.text(300, (ya+yb)/2, r'$\Delta_y=%0.2f$'%delta_y, fontsize=14, color='#ff0066',
             ha='center', va='bottom', bbox=box)
    ay.text( tau + 10, ya + 0.63212*delta_y,  r'$\tau = %0.2f$'%tau, fontsize=14, color='#ff0066')

    # settings for the lower, depicting the input
    au.set_xlim(t[0], tend)
    au.grid(True);
    au.set_facecolor('#d7f4ee')
    au.grid(color='#1a1a1a40', linestyle='--', linewidth=0.25)
    au.text(300, ua + 2, r'$\Delta_u=%0.2f$'%delta_u, fontsize=14, color="#00aa00",
             ha='center', va='bottom', bbox=box)
    au.set_xlabel('Time (seconds)')
    ay.set_ylabel('Temperature ($~^oC$)')

    line_exp, = ay.plot(t, y, color="#0088aa", linewidth=1.5, linestyle=(0, (1, 1)))
    line_mod, = ay.plot(t, ymodel, color="#ff0066", linewidth=1.5, )
    ay.plot(tau, ya + 0.63212*delta_y , color="#ff0066", linewidth=1.5, marker=".", markersize=13)
    line_u, = au.plot(t, u, color="#00aa00")
    modelstr = r"Model $G(s)= \frac{\alpha}{\tau\,s + 1}\, e^{-L\,s}  = \frac{%0.2f }{%0.2f\,s+1}\, e^{-%0.2f\,s}$" %(alpha, tau, L)
    ay.legend([line_exp, line_mod], ['Data', modelstr], fontsize=12)
    au.legend([line_u], ['Input'])
    fig.canvas.draw()
    exp = [[alpha, tau, L]]
    np.savetxt(PATH_DATA + "Thermal_fomodel_step.csv", exp, delimiter=",", fmt="%0.8f", comments="", header='alpha, tau, L')
    np.savetxt(PATH_DEFAULT + "Thermal_fomodel_step.csv", exp, delimiter=",", fmt="%0.8f", comments="",
               header='alpha, tau, L')
    system.disconnect()
    return alpha, tau, L




if __name__ == "__main__":
    plant = ThermalSystemIoT()
    get_fomodel_step(plant, yop=50, usefile= True)





