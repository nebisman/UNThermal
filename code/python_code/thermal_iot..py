# Required libraries
import paho.mqtt.client as mqtt
import control as ct
import struct
from queue import Queue
import math
import json
import numpy as npy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg", force=True)

# parameters of communication


BROKER = "192.168.0.3"
PORT = 1883
USER = "hpdesktop"
PASSWORD = "hpdesktop"

#topics for subscribing

PLANT_NUMBER = "1234"
codes ={"SYS_USER_SIGNALS_CLOSED"  : "/thermal/thermal_" + PLANT_NUMBER + "/user/sig_closed",
        "SYS_USER_SIGNALS_OPEN"  : "/thermal/thermal_" + PLANT_NUMBER + "/user/sig_open",
        "USER_SYS_SET_REF"  : "/thermal/user/thermal_" + PLANT_NUMBER + "/set_ref",
        "USER_SYS_SET_PID"  : "/thermal/user/thermal_" + PLANT_NUMBER  + "/set_pid",
        "USER_SYS_STEP_CLOSED": "/thermal/user/thermal_" + PLANT_NUMBER +"/step_closed",
        "USER_SYS_STAIRS_CLOSED": "/thermal/user/thermal_" + PLANT_NUMBER + "/stairs_closed",
        "USER_SYS_PRBS_OPEN": "/thermal/user/thermal_" + PLANT_NUMBER + "/prbs_open",
        "USER_SYS_STEP_OPEN": "/thermal/user/thermal_" + PLANT_NUMBER + "/step_open",
        "USER_SYS_SET_GENCON": "/thermal/user/thermal_" + PLANT_NUMBER + "/set_gencon",
        "THERMAL_SAMPLING_TIME" : 1
        }


class SystemIoT:

    def __init__(self, broker_address = BROKER, port= PORT, client_id="", clean_session=True):
        self.client = mqtt.Client()
        self.broker_address = broker_address
        self.port = port
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.on_publish = self.on_publish
        self.codes = codes


    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print("Unexpected disconnection.")

    def on_message(self, client, userdata, message):
        print(f"Received  '{message.payload.decode()}'")

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed: ", mid, " ", granted_qos)

    def on_publish(self, client, userdata, mid):
        print("Message Published: ", mid)

    def connect(self):
        self.client.username_pw_set(USER, PASSWORD)
        self.client.connect(self.broker_address, self.port)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topic, qos=2):
        self.client.subscribe(topic, qos)

    def publish(self, topic, message, qos=1):
        self.client.publish(topic, message, qos)

    def transfer_function(self, temperature=50):
        Kp = -0.0025901782151786 * temperature + 0.987094648761147
        Tao = -0.0973494029141449 * temperature + 66.5927276606595
        delay = -0.00446863636363636 * temperature + 3.57201818181818
        uN = 0.9719 * temperature - 24.7355
        G = ct.TransferFunction(Kp, [Tao, 1])
        return G, delay, uN


def float2hex(value):
    val_binary = struct.pack('>f', value)
    return val_binary.hex()


def long2hex(value):
    val_binary = struct.pack('>L', value)
    return val_binary.hex()


def hex2float(str):
    value = struct.unpack('!f', bytes.fromhex(str))[0]
    return value


def hex2long(str):
    value = struct.unpack('!L', bytes.fromhex(str))[0]
    return value


def signal2hex(signal):
    hstring = ""
    for point in signal:
        hstring += float2hex(point)
    return hstring


def matrix2hex(matrix):
    hstring = ""
    for row in matrix:
        for element in row:
            hstring += float2hex(element)
    return hstring


def set_reference(system, ref_value=50):
    ref_hex = float2hex(ref_value)
    topic_pub = system.codes["USER_SYS_SET_REF"]
    message = json.dumps({"reference": ref_hex})
    system.connect()
    system.publish(topic_pub, message)
    system.disconnect()
    rcode = True
    print("succesfull change of reference")
    return rcode


def set_pid(system, kp=1, ki=0.4, kd=0, N=5, beta=1):
    topic_pub = system.codes["USER_SYS_SET_PID"]
    kp_hex = float2hex(kp)
    ki_hex = float2hex(ki)
    kd_hex = float2hex(kd)
    N_hex = float2hex(N)
    beta_hex = float2hex(beta)
    message = json.dumps({"kp": kp_hex,
                          "ki": ki_hex,
                          "kd": kd_hex,
                          "N": N_hex,
                          "beta": beta_hex})
    system.connect()
    system.publish(topic_pub, message)
    system.disconnect()
    rcode = True
    print("succesfull change of PID parameters")
    return rcode


def step_closed(system, low_val=30, high_val=50, low_time=60, high_time=120, filepath ="step_closed_exp.csv"):
    def step_message(system, userdata, message):
        q.put(message)

    topic_pub = system.codes["USER_SYS_STEP_CLOSED"]
    topic_sub = system.codes["SYS_USER_SIGNALS_CLOSED"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    points_high = round(high_time / sampling_time) + 1
    points_low = round(low_time / sampling_time)
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
    np = -1
    y = []
    r = []
    u = []
    t = []
    fig, ax = plt.subplots()
    line_r, = ax.plot(t, r, 'b-')
    line_y, = ax.plot(t, y, linestyle = 'solid', color="#008066ff", linewidth=1.25) # drawstyle='steps'
    points = points_high + points_low
    ax.set_xlim(0, sampling_time * (points - 1))
    ax.set_ylim(low_val - 10, high_val + 10)
    #ax.set_ylim(20, high_val + 10)
    plt.grid()
    npc = 1
    exp = []
    while npc <= points:
        try:
            message = q.get(True, 20 * sampling_time)
        except:
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        
        np_hex = str(msg_dict["np"])
        np = hex2long(np_hex)

        if (np == npc) and (np >= 0):
            t_curr = (np - 1) * sampling_time
            t.append(t_curr)
            y_curr = hex2float(msg_dict["y"])
            y.append(y_curr)
            r_curr = hex2float(msg_dict["r"])
            r.append(r_curr)
            u_curr = hex2float(msg_dict["u"])
            u.append(u_curr)
            line_r.set_data(t, r)
            line_y.set_data(t, y)
            exp.append([t_curr, r_curr, y_curr, u_curr])
            plt.draw()
            plt.pause(sampling_time)
            npc += 1

    npy.savetxt(filepath, exp, delimiter=",",
                fmt="%0.8f", comments="", header='t,r,y,u')
    system.disconnect()
    plt.show()
    print("Step response completed")
    return t, r, y, u


def stair_closed(system, stairs=[40, 50, 60], duration=100, filepath = "stair_closed_exp.csv"):
    def usersignal_message(system, userdata, message):
        q.put(message)

    # accessing fields of system IoT
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    topic_pub = system.codes["USER_SYS_STAIRS_CLOSED"]
    topic_sub = system.codes["SYS_USER_SIGNALS_CLOSED"]
    points = len(stairs)
    points_hex = long2hex(points)
    signal_hex = signal2hex(stairs)
    duration_points = math.ceil(duration / sampling_time)
    duration_points_hex = long2hex(duration_points)
    message = json.dumps({"signal": signal_hex, "duration": duration_points_hex, "points": points_hex})
    system.client.on_message = usersignal_message
    system.connect()
    system.subscribe(topic_sub)
    system.publish(topic_pub, message)
    q = Queue()
    np = -1
    y = []
    r = []
    t = []
    u = []
    exp = []
    fig, ax = plt.subplots()
    line_r, = ax.plot(t, r, 'b-')
    line_y, = ax.plot(t, y, 'r-')
    ax.set_xlim(0, sampling_time * duration_points * points)
    ax.set_ylim(20, 100)
    plt.grid()
    npc = 1
    while npc <= duration_points * points - 1:
        try:
            message = q.get(True, 20 * sampling_time)
        except:
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        np_hex = str(msg_dict["np"])
        np = hex2long(np_hex)
        print(np, hex2float(msg_dict["r"]), hex2float(msg_dict["y"]))
        if np == npc:
            t_curr = (np - 1) * sampling_time
            t.append(t_curr)
            y_curr = hex2float(msg_dict["y"])
            y.append(y_curr)
            r_curr = hex2float(msg_dict["r"])
            r.append(r_curr)
            u_curr = hex2float(msg_dict["u"])
            u.append(u_curr)
            line_r.set_data(t, r)
            line_y.set_data(t, y)
            exp.append([t_curr, r_curr, y_curr, u_curr])
            plt.draw()
            plt.pause(0.1)
            npc += 1
    npy.savetxt(filepath, exp, delimiter=",",
                fmt="%0.8f", comments="", header='t,r,y,u')
    system.disconnect()
    plt.show()
    return t, y, r, u


def pbrs_open(system, op_point=50, peak_amp=5, stab_time=150, uee_time=20, divider=35, filepath = "prbs_open_exp.csv"):
    def pbrs_message(system, userdata, message):
        q.put(message)

    topic_pub = system.codes["USER_SYS_PRBS_OPEN"]
    topic_sub = system.codes["SYS_USER_SIGNALS_OPEN"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    peak_amp_hex = float2hex(peak_amp)
    op_point_hex = float2hex(op_point)
    stab_points = math.ceil(stab_time / sampling_time)
    uee_points = math.ceil(uee_time / sampling_time)
    stab_points_hex = long2hex(stab_points)
    uee_points_hex = long2hex(uee_points)
    divider_hex = long2hex(divider)
    points = divider * 63 + stab_points + uee_points
    message = json.dumps({"peak_amp": peak_amp_hex,
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
    np = None
    y = []
    u = []
    t = []
    yt = []
    ut = []
    tt = []
    exp = []
    fig, ax = plt.subplots()
    line_y, = ax.plot(t, y, color="#ffcc00")
    line_u, = ax.plot(t, u, color="#00d4aa")
    line_yt, = ax.plot(t, yt, color="#d40055")
    line_ut, = ax.plot(t, ut, color="#338000")
    a1 = 0.971914417613643
    a2 = -24.7354557071619
    uf_est = a1 * op_point + a2
    Tmax_est = (uf_est + peak_amp - a2) / a1
    Tmin_est = (uf_est - peak_amp - a2) / a1
    umin_est = a1 * Tmin_est + a2
    ax.set_xlim(0, sampling_time * points)
    ax.set_ylim(umin_est - 5, Tmax_est + 5)
    plt.grid()
    npc = 1
    while npc <= points:
        try:
            message = q.get(True, 20 * sampling_time)
        except:
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        np_hex = str(msg_dict["np"])
        np = hex2long(np_hex)
        print(points, npc, np, hex2float(msg_dict["u"]), hex2float(msg_dict["y"]))
        if np == npc:
            if npc <= stab_points + uee_points:
                t_curr = (np - 1) * sampling_time
                t.append(t_curr)
                y_curr = hex2float(msg_dict["y"])
                y.append(y_curr)
                u_curr = hex2float(msg_dict["u"])
                u.append(u_curr)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                if npc == stab_points + uee_points:
                    tt.append(t_curr)
                    yt.append(y_curr)
                    ut.append(u_curr)

                    #exp.append([0, u_curr, y_curr])
            else:
                tt_curr = (np - 1) * sampling_time
                if npc == stab_points + uee_points + 1:
                    t0 = t_curr
                tt.append(tt_curr)
                yt_curr = hex2float(msg_dict["y"])
                yt.append(yt_curr)
                ut_curr = hex2float(msg_dict["u"])
                ut.append(ut_curr)
                line_yt.set_data(tt, yt)
                line_ut.set_data(tt, ut)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
                exp.append([tt_curr-t0, ut_curr, yt_curr])
            plt.draw()
            plt.pause(0.1)
            npc += 1
    npy.savetxt(filepath, exp, delimiter=",",
                fmt="%0.8f", comments="", header='t,u,y')
    system.disconnect()
    plt.show()
    return t, u, y


def step_open(system, op_point=50, amplitude=5, high_time=200, stab_time=150, uee_time=20, filepath = "step_open_exp.csv"):
    def step_message(system, userdata, message):
        q.put(message)

    topic_pub = system.codes["USER_SYS_STEP_OPEN"]
    topic_sub = system.codes["SYS_USER_SIGNALS_OPEN"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    amp_hex = float2hex(amplitude)
    points_high = math.ceil(high_time / sampling_time)
    points_high_hex = long2hex(points_high)
    op_point_hex = float2hex(op_point)
    stab_points = math.ceil(stab_time / sampling_time)
    uee_points = math.ceil(uee_time / sampling_time)
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
    np = None
    y = []
    u = []
    t = []
    yt = []
    ut = []
    tt = []
    exp =[]
    fig, ax = plt.subplots()
    line_y, = ax.plot(t, y, color="#ffcc00")
    line_u, = ax.plot(t, u, color="#00d4aa")
    line_yt, = ax.plot(t, yt, color="#d40055")
    line_ut, = ax.plot(t, ut, color="#338000")

    a1 = 0.971914417613643
    a2 = -24.7354557071619
    uf_est = a1 * op_point + a2
    points = stab_points + uee_points + points_high
    Tmax_est = (uf_est + amplitude - a2) / a1
    ax.set_xlim(0, sampling_time * points)
    ax.set_ylim(uf_est - 5, Tmax_est + 5)
    plt.grid()
    npc = 1
    while npc <= points:
        try:
            message = q.get(True, 20 * sampling_time)
        except:
            raise TimeoutError("The connection has been lost. Please try again")

        decoded_message = str(message.payload.decode("utf-8"))
        msg_dict = json.loads(decoded_message)
        np_hex = str(msg_dict["np"])
        np = hex2long(np_hex)
        print(points, npc, np, hex2float(msg_dict["u"]), hex2float(msg_dict["y"]))
        if np == npc:
            if npc <= stab_points + uee_points:
                t_curr = (np - 1) * sampling_time
                t.append(t_curr)
                y_curr = hex2float(msg_dict["y"])
                y.append(y_curr)
                u_curr = hex2float(msg_dict["u"])
                u.append(u_curr)
                line_y.set_data(t, y)
                line_u.set_data(t, u)

            else:

                tt_curr = (np - 1) * sampling_time
                if npc <= stab_points + uee_points + 1:
                    t0 = tt_curr
                tt.append(tt_curr)
                yt_curr = hex2float(msg_dict["y"])
                yt.append(yt_curr)
                ut_curr = hex2float(msg_dict["u"])
                ut.append(ut_curr)
                exp.append([tt_curr - t0, ut_curr, yt_curr])
                line_yt.set_data(tt, yt)
                line_ut.set_data(tt, ut)
                line_y.set_data(t, y)
                line_u.set_data(t, u)
            plt.draw()
            plt.pause(0.1)
            npc += 1
    npy.savetxt(filepath, exp, delimiter=",",
                fmt="%0.8f", comments="", header='t,u,y')
    system.disconnect()
    plt.show()
    return t, u, y


def set_controller(system, controller):
    topic_pub = system.codes["USER_SYS_SET_GENCON"]
    sampling_time = system.codes["THERMAL_SAMPLING_TIME"]
    Cvecont = ct.tf2ss(controller)
    Cve = ct.c2d(Cvecont, sampling_time, method='tustin')
    Ac = Cve.A
    Bc = Cve.B
    Cc = Cve.C
    Dc = Cve.D

    if (npy.size(Bc, axis=1)) == 1:
        B1 = []
        for row in Bc:
            for e in row:
                B1.append([e, -e])
        Bc = npy.array(B1)
        Dc = npy.array([[Dc[0][0], -Dc[0][0]]])
    order = len(Ac)
    order_hex = long2hex(order)
    P = [(i + 1) * 1e-8 for i in range(order)]
    L = ct.place(npy.transpose(Ac), npy.transpose(Cc), P)
    L = npy.transpose(L)
    A = Ac - L * Cc
    B = Bc - L * Dc
    A_hex = matrix2hex(A)
    B_hex = matrix2hex(B)
    C_hex = matrix2hex(Cc)
    D_hex = matrix2hex(Dc)
    L_hex = matrix2hex(L)
    message = json.dumps({"order": order_hex,
                          "A": A_hex,
                          "B": B_hex,
                          "C": C_hex,
                          "D": D_hex,
                          "L": L_hex
                          })
    system.connect()
    system.publish(topic_pub, message)
    system.disconnect()
    rcode = True
    return rcode


if __name__ == "__main__":
    plant = SystemIoT()
    print(plant.transfer_function(80))
    set_pid(plant, kp=16.796, ki=2, kd=16.441, N=27.38, beta=1)
    t,  r, y, u = step_closed(plant, 50, 60, 50, 50)
    #signal = [30, 40, 50, 60, 70, 80]
    # set_pid(plant, kp = 16.796, ki = 5, kd = 16.441, N = 27.38, beta = 1)
    #step_open(plant, op_point=45, amplitude=5, high_time=200, stab_time=150, uee_time=20)
    #pbrs_open(plant, op_point=45, peak_amp=5, stab_time=150, uee_time=20, divider=35)
    #stair_closed(plant, signal, 10)

