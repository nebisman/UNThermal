# Required libraries

import paho.mqtt.client as mqtt
import control as ct
import struct
from queue import Queue
from math import ceil
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import cont2discrete
import csv




# default parameters of communication


BROKER = "192.168.1.100"
PORT = 1883
USER = "user"
PASSWORD = "user"
PLANT_NUMBER = "PC-T-00"


#default configuration

PATH_DEFAULT = r"./experiment_files/"
PATH_DATA = str(Path(__file__).parent) + r"/datafiles/"
FONT_SIZE = 10
Path(PATH_DEFAULT).mkdir(exist_ok=True)

class ThermalSystemIoT:
    """
    Class to manage a thermal system using IoT and MQTT protocol.

    Parameters
    ----------

    broker_address : str,required
        The MQTT broker's address attending the thermal system experiment. For instance, "192.168.1.100".
    port : int, required
        The port for MQTT connection. Default is 1883.
    plant_number : str, required
        The plant id assigned to the user's specific thermal system. For instante "PC-T-11".
    user : str, optional
        The username for MQTT authentication in a remote experimental setup. 
    password : str, optional
        The password for MQTT authentication in a remote experimental setup.
    
    Attributes:
    -----------


    client : mqtt.Client
        The MQTT client used for connection and communication.
    codes : dict
        A dictionary containing codes for various MQTT topics related to the thermal system.

    User methods:
    -------------

    transfer_function()
        Returns the thermal system's transfer function using data from the last         
        PRBS (pseudo random binary signal) experiment.

    Example:
    --------

    First, ensure that you have already imported the unthermal package as follows:
    
    >>> import unthermal as ter
    
    Then, create an IoT Thermal System object, as follows:
    
    >>> my_system = ter.ThermalSystemIoT(plant_number="PC-T-01", broker_address="192.168.1.100") 
    
    You can use the transfer_function method to retrieve the system's transfer function
    based on the most recent PRBS identification experiment performed on the thermal system.
        
    >>> G = my_system.transfer_function()
    >>> print(G)
    >>> <TransferFunction>: sys[5]
    >>> Inputs (1): ['u[0]']
    >>> Outputs (1): ['y[0]']
    >>>    1.182
    >>> -----------
    >>>  36.98 s + 1

        
    """



    def __init__(self, broker_address = BROKER, port= PORT, plant_number = PLANT_NUMBER, user=USER, password=PASSWORD,  client_id="", clean_session=True):

        codes = {"SYS_USER_SIGNALS_CLOSED": "/thermal/thermal_" + str(plant_number) + "/user/sig_closed",
                 "SYS_USER_SIGNALS_OPEN": "/thermal/thermal_" + str(plant_number) + "/user/sig_open",
                 "USER_SYS_SET_REF": "/thermal/user/thermal_" + str(plant_number) + "/set_ref",
                 "USER_SYS_SET_PID": "/thermal/user/thermal_" + str(plant_number) + "/set_pid",
                 "USER_SYS_STEP_CLOSED": "/thermal/user/thermal_" + str(plant_number) + "/step_closed",
                 "USER_SYS_STAIRS_CLOSED": "/thermal/user/thermal_" + str(plant_number) + "/stairs_closed",
                 "USER_SYS_PRBS_OPEN": "/thermal/user/thermal_" + str(plant_number) + "/prbs_open",
                 "USER_SYS_STEP_OPEN": "/thermal/user/thermal_" + str(plant_number) + "/step_open",
                 "USER_SYS_SET_GENCON": "/thermal/user/thermal_" + str(plant_number) + "/set_gencon",
                 "USER_SYS_PROFILE_CLOSED": "/thermal/user/thermal_" + str(plant_number) + "/prof_closed",
                 "THERMAL_SAMPLING_TIME": 0.8
                 }
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
            print("Successfully connected to MQTT Broker " + self.broker_address)
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
        print("Command send: ", mid)

    def connect(self):
        self.client.username_pw_set(USER, PASSWORD)
        self.client.connect(self.broker_address, self.port)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topic, qos=2):
        self.client.subscribe(topic, qos)

    def publish(self, topic, message, qos=2):
        self.client.publish(topic, message, qos)

    def transfer_function(self):
        with open(PATH_DATA + 'Thermal_fo_model_pbrs.csv', newline='') as file:
            reader = csv.reader(file)
            # Iterate over each row in the CSV file
            num_line = 0
            for row in reader:
                if num_line != 0:
                    alpha = float(row[0])
                    tau = float(row[1])
                num_line += 1
        s = ct.TransferFunction.s
        G = alpha/(tau*s + 1)
        return G

