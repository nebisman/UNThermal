// Definitions for the thermal system
// Created by LB on 1/18/24.
#include <stdlib.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "WiFi.h"
#include <Adafruit_NeoPixel.h>

//This is the pwm configuration for the thermal system

#define  FREQUENCY_PWM     1000
#define  RESOLUTION_PWM    12
#define  POWER_PIN         17 // 25
#define  PWM_CHANNEL       0

// This is the configuration for the ws2812 rgb leds that show the temperature and reference
#define  LIGHT_PIN          13//19
#define  NUMPIXELS          3

// Hardware definitions
#define CORE_CONTROL 1   // Core 0 is used for control tasks
#define CORE_COMM    0   // Core 1 is used for mqtt communication


// This is the pin for the Dallas 18b20 temperature sensor
#define ONE_WIRE_BUS      1

// System definitions
#define DEFAULT_REFERENCE  45
#define SAMPLING_TIME      0.8
#define MAX_ORDER          10


// IOT topic definitions
// Topics published by the thermal system
#define SYS_USER_SIGNALS_CLOSED       "/thermal/thermal_" PLANT_NUMBER "/user/sig_closed"
#define SYS_USER_SIGNALS_OPEN       "/thermal/thermal_" PLANT_NUMBER "/user/sig_open"

// Topics received from the user
#define USER_SYS_SET_PID           "/thermal/user/thermal_" PLANT_NUMBER "/set_pid"             //1
#define USER_SYS_SET_REF           "/thermal/user/thermal_" PLANT_NUMBER "/set_ref"             //2
#define USER_SYS_STEP_CLOSED       "/thermal/user/thermal_" PLANT_NUMBER "/step_closed"         //3
#define USER_SYS_STAIRS_CLOSED     "/thermal/user/thermal_" PLANT_NUMBER "/stairs_closed"       //4
#define USER_SYS_PRBS_OPEN         "/thermal/user/thermal_" PLANT_NUMBER "/prbs_open"           //5
#define USER_SYS_STEP_OPEN         "/thermal/user/thermal_" PLANT_NUMBER "/step_open"           //6
#define USER_SYS_STEP_OPEN         "/thermal/user/thermal_" PLANT_NUMBER "/step_open"           //6
#define USER_SYS_SET_GENCON        "/thermal/user/thermal_" PLANT_NUMBER "/set_gencon"          //7
#define USER_SYS_PROFILE_CLOSED    "/thermal/user/thermal_" PLANT_NUMBER "/prof_closed"         //8

// Integer definitions of topics to avoid comparison with strings, which is more computationally expensive

#define DEFAULT_TOPIC                  0
#define USER_SYS_STEP_CLOSED_INT       1
#define USER_SYS_STAIRS_CLOSED_INT     2
#define USER_SYS_PRBS_OPEN_INT         3
#define USER_SYS_STEP_OPEN_INT         4
#define USER_SYS_PROFILE_CLOSED_INT    5


// Codes for modes of control

#define GENERAL_CONTROLLER_1P          0
#define GENERAL_CONTROLLER_2P          1
#define PID_CONTROLLER                 2




/**********************************************************************************************************
 * This matrix is the gamma correction for the leds attached to the plant, which allow to "see" the current
*   temperature and the current reference
**********************************************************************************************************/

const uint8_t gamma8[] = {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,
        2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,
        5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10,
        10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
        17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 24, 24, 25,
        25, 26, 27, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36,
        37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50,
        51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68,
        69, 70, 72, 73, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 87, 89,
        90, 92, 93, 95, 96, 98, 99,101,102,104,105,107,109,110,112,114,
        115,117,119,120,122,124,126,127,129,131,133,135,137,138,140,142,
        144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,
        177,180,182,184,186,189,191,193,196,198,200,203,205,208,210,213,
        215,218,220,223,225,228,231,233,236,239,241,244,247,249,252,255 };

//  percent of control signal in the operation point
const float percent2pwm = (float) (4095.0/100.0);


// Button's configuration

#define BUTTON_PLUS  7
#define BUTTON_MINUS 6
#define THRESHOLD_PLUS 60000
#define THRESHOLD_MINUS 60000





// Sensors and peripheral definitions
OneWire  oneWire(ONE_WIRE_BUS);   // connection to Dallas ds18b20 temperature sensor
DallasTemperature sensors(&oneWire);  // connection to Dallas ds18b20 temperature sensor
Adafruit_NeoPixel dispLeds(NUMPIXELS, LIGHT_PIN, NEO_GRB); // connection to rgb leds that display temperature


// Task handles for resuming and suspending tasks
TaskHandle_t h_controlPidTask;
TaskHandle_t h_identifyTask;
TaskHandle_t h_generalControlTask;
TaskHandle_t h_publishStateTask;
TaskHandle_t h_menuTask;

// PID control default parameters
float kp = 16.796;
float ki = 2;
float kd  = 16.441;
float N = 27.38;
float beta = 0.1;
float h = SAMPLING_TIME; //sampling time
const float br = 1/0.99;
bool reset_int = false;


// Parameters of a general controller for the thermal system
unsigned int order;
float A [10][10];   // Controller's A matrix
float B [10][2];    // Controller's B matrix
float C [10];       // Controller's C matrix
float D [2];        // Controller's D matrix
float L [10];       // L is the gain matrix for the antiwindup observer


// loop control variables
float reference = DEFAULT_REFERENCE;   //  Desired temperature of reference
float y = 0;            //  Temperature
float u = 0;            //  Control signal
float usat = 0;         //  Control saturation signal


// Vectors of values and times for storing  the edges of
// an user's defined arbitrary signal
float stairs[50];       //  Arbitrary signal
uint32_t timeValues[50];


// Variables that keep the command sent by the user and the mode of control
uint8_t codeTopic = DEFAULT_TOPIC;        //Integer code of the command sent by the user
uint8_t  typeControl = PID_CONTROLLER; //Integer code of mode of control,
// which can be PID or general control (default)


// These parameters allow configuring the different user's commands
float low_val =    0;
float high_val =   0;
uint32_t points_stairs;
uint32_t duration;
uint32_t points_high = 50;
uint32_t points_low = 50;
uint32_t np = 0;
uint32_t total_time = 4294967295;


uint16_t uee_points= (unsigned int) (20/h);
uint16_t divider = 35;
uint16_t prbs_points =  63 * divider;
uint16_t stab_points = (unsigned int) (150/h);


// Internet connection variables
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);


bool plusDetected = false;
bool minusDetected = false;
// These miscellaneous functions allow to receive formatted data (for instance controllers) from user
float hex2Float(const char* string){
    // This function converts a 4-digit hexadecimal string received from MQTT to a floating-point number.
    // It enables the reception of precise float numbers without rounding, minimizing the transmitted byte count.
    unsigned long temp;
    sscanf(string, "%8x", &temp);
    float f = *(float *)&temp;
    return f;
}

float hex2Long(const char* string){
    // This function converts a 4-digit hexadecimal string received from MQTT to a long integer.
    // As a result, it enables the reception of precise integer numbers with the minimum byte usage.
    long l = strtol(string, NULL, 16);
    return l;
}

char * float2Hex(float number){
    // This function converts a floating-point number to a 4-digit hexadecimal string. Consequently,
    // it enables the transmission of an exact float number with the minimum byte usage through MQTT.
    static char str[9];
    unsigned long ui;
    memcpy(&ui, &number, sizeof (ui));
    sprintf(str, "%08x", ui);
    return str;
}


char * long2Hex(unsigned long number){
    // This function converts a long integer number to a 4-digit hexadecimal string. Consequently,
    // it enables the transmission of a big number with the minimum byte usage through MQTT.
    static char str[9];
    unsigned long ui;
    memcpy(&ui, &number, sizeof (ui));
    sprintf(str, "%08x", ui);
    return str;
}

void hexStringToFloatArray(float* floatArray, const char* hexString, unsigned int numFloats) {
    // This function translates a user-sent hexadecimal string into an array of floats, enabling the transmission
    // of arbitrary signals and parameters of a controller in a state space form.
    for (size_t i = 0; i < numFloats; ++i) {
        unsigned long temp;
        sscanf(hexString + i * 8, "%8x", &temp); // Read 8 hex digits (32 bits)
        memcpy(floatArray + i, &temp, sizeof(float)); // Copy bits into float
    }

}

void hexStringToMatrix(float* matrix, const char* hexString, unsigned int order, unsigned int cols, unsigned int maxcols) {
    // This function translates a user-sent hexadecimal string into a matrix of floats, enabling the transmission
    // of arbitrary signals and parameters of a controller in a state space form.
    for (size_t i = 0; i < order; ++i) {
        unsigned long temp;
        for (size_t j = 0; j < cols; j++) {
            sscanf(hexString + (i*cols + j) * 8, "%8x", &temp); // Read 8 hex digits (32 bits)
            memcpy(matrix + (i*maxcols + j ), &temp, sizeof(float)); // Copy bits into float
        }
    }

}

void hexStringToLongArray(uint32_t *longArray, const char* hexString, unsigned int numLongs) {
    // This function translates a user-sent hexadecimal string into an array of floats, enabling the transmission
    // of arbitrary signals and parameters of a controller in a state space form.
    for (unsigned int  i = 0; i < numLongs; ++i) {
        uint32_t  temp;
        sscanf(hexString + i * 8, "%8x", &temp); // Read 8 hex digits (32 bits)
        memcpy(longArray + i, &temp, sizeof(uint32_t)); // Copy bits into float
    }
}

void sensorToColormap(float x, uint8_t * rgbval) {

    /** This function calculates the R, G, and B values of a fifth-degree polynomial approximation of the turbo colormap.
        It is the colormap displayed by the WS2812 LEDs connected to the thermal plant.
      The resulting colormap enables the user to visualize both the plant's temperature and the controller's actions.
      The input 'x' represents a normalized value within the [0, 1] range, corresponding to the analog variable visualized by the LEDs.
      The input 'rgbval' is a 3-element array designed to store the computed RGB values.
    */
    float r;
    float g;
    float b;

    // colormap turbo
    r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5666 - x * 58.1375))));
    g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 + x * 2.7747))));
    b = 0.1067 + x * (12.5925 - x * (60.1097 - x * (109.0745 - x * (88.5066 - x * 26.8183))));


    r = constrain(r,0,1);
    g = constrain(g,0,1);
    b = constrain(b,0,1);

    rgbval[0] = (uint8_t) 255*r;
    rgbval[1] = (uint8_t) 255*g;
    rgbval[2] = (uint8_t) 255*b;
    rgbval[0] = pgm_read_byte(&gamma8[rgbval[0]]);
    rgbval[1] = pgm_read_byte(&gamma8[rgbval[1]]);
    rgbval[2] = pgm_read_byte(&gamma8[rgbval[2]]);
}

void displayLed(float var, float valmin, float valmax, float start, uint8_t led){
    // This function takes an analog variable and visualizes it as a colormap using the "sensorToColormap" function.
    // The 'var' parameter represents the analog variable, while 'valmin', 'valmax', and 'percent' serve as adjustment
    // parameters to scale the colormap.
    // The 'led' parameter, with values 0 or 1, denotes each of the LEDs attached to the plant.
    uint8_t rgb[3];
    float x;
    x = start + constrain((1-start)* (var - valmin)/(valmax - valmin), 0, 1-start);
    sensorToColormap(x, rgb);
    dispLeds.setPixelColor(led, dispLeds.Color(rgb[0], rgb[1], rgb[2]));
    dispLeds.show();
}

float linearInterpolation(uint32_t t[], float r[], uint16_t n, uint32_t t_interp) {
    uint16_t i;
    float result;
    float r1;

    for (i = 0; i < n - 1; i++) {
        if (t[i] <= t_interp && t_interp <= t[i + 1]) {
            // Linear interpolation formula: y = y1 + (y2 - y1) * ((x - x1) / (x2 - x1))
            r1 = (float) ( t_interp -  t[i]) / (t[i + 1] - t[i]);
            result = r[i] + (r[i + 1] - r[i]) *  r1;
        }
    }
    return result;
}


void wattsToPlant(float watts){
    ledcWrite(PWM_CHANNEL, watts * percent2pwm);
}




void connectMqtt()
{
    printf("Starting MQTT connection...");
    if (mqttClient.connect(THINGNAME))
    {
        mqttClient.subscribe(USER_SYS_STAIRS_CLOSED);
        mqttClient.subscribe(USER_SYS_SET_PID);
        mqttClient.subscribe(USER_SYS_SET_REF);
        mqttClient.subscribe(USER_SYS_STEP_CLOSED);
        mqttClient.subscribe(USER_SYS_STEP_OPEN);
        mqttClient.subscribe(USER_SYS_PRBS_OPEN);
        mqttClient.subscribe(USER_SYS_SET_GENCON);
        mqttClient.subscribe(USER_SYS_PROFILE_CLOSED);
        printf("now connected to broker %s !\n", BROKER);
    }
    else
    {
        printf("Failed MQTT connection code %d \n try again in 2 seconds\n", mqttClient.state());
    }
}


void connectWiFi(){
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    printf("Plant %s is connecting to %s network , please wait\n", PLANT_NUMBER, WIFI_SSID);

    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        vTaskDelay(pdMS_TO_TICKS(250));
    }
    printf("\n");
    printf("Connected to WIFI through IP: %s \n", WiFi.localIP().toString());

}



void handleConnections(void *pvParameters) {
    for (;;) {
        if (WiFi.status() != WL_CONNECTED) {
            connectWiFi();
        }
        if (!mqttClient.connected()) {
            vTaskDelay(2000);
            connectMqtt();
        }
        mqttClient.loop();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}



// This function suspend all  controlling tasks
void suspendAllTasks(){
    vTaskSuspend(h_publishStateTask);
    vTaskSuspend(h_generalControlTask);
    vTaskSuspend(h_controlPidTask);
    vTaskSuspend(h_identifyTask);
    wattsToPlant(0);
}


void resumeControl(){
    reset_int = true;
    switch (typeControl) {
        case PID_CONTROLLER:
            vTaskResume(h_controlPidTask);
            break;
        case GENERAL_CONTROLLER_1P:
            vTaskResume(h_generalControlTask);
            break;
        case GENERAL_CONTROLLER_2P:
            vTaskResume(h_generalControlTask);
            break;
    }
}

// This function activate the default controller
void defaultControl(){
    vTaskSuspend(h_publishStateTask);
    codeTopic = DEFAULT_TOPIC;
    reset_int = true;
    reference = DEFAULT_REFERENCE;
    resumeControl();
    vTaskSuspend(h_identifyTask);
}




void IRAM_ATTR onMqttReceived(char* lastTopic, byte* lastPayload, unsigned int lenght){
    suspendAllTasks();
    JsonDocument doc;
    if (strstr(lastTopic, USER_SYS_SET_PID)){
        typeControl = PID_CONTROLLER;
        deserializeJson(doc, lastPayload);
        kp = hex2Float((const char *) doc["kp"]);
        ki = hex2Float((const char *) doc["ki"]);
        kd = hex2Float((const char *) doc["kd"]);
        N = hex2Float((const char *) doc["N"]);
        beta = hex2Float((const char *) doc["beta"]);
        reset_int = true;
        printf("PID parameters settled:\n\tkp=%0.3f\n\tki=%0.3f\n\tkd=%0.3f\n\tN=%0.3f\n\tBeta=%0.3f\n",
               kp, ki, kd, N, beta);
        defaultControl();
    }

    else if (strstr(lastTopic, USER_SYS_SET_REF )){
        deserializeJson(doc, lastPayload);
        reference = hex2Float((const char *) doc["reference"]);
        reset_int = true;
        printf("Reference has been set to %0.2f degrees \n", reference);
        defaultControl();
    }

    else if (strstr(lastTopic, USER_SYS_STEP_CLOSED )){
        codeTopic = USER_SYS_STEP_CLOSED_INT;
        deserializeJson(doc, lastPayload);
        low_val = hex2Float((const char *) doc["low_val"]);
        high_val = hex2Float((const char *) doc["high_val"]);
        points_high = hex2Long((const char *) doc["points_high"]);
        points_low = hex2Long((const char *) doc["points_low"]);
        total_time = points_low + points_high;
        printf("Closed loop step response:\n");
        printf("    low value=%0.2f\n    high value=%0.2f\n    time in high =%0.2f\n    time in low=%0.2f\n", low_val,
               high_val, (float) (points_high * h), (float) (points_low * h));
        vTaskResume(h_publishStateTask);
        resumeControl();
    }

    else if (strstr(lastTopic, USER_SYS_STAIRS_CLOSED)){
        codeTopic = USER_SYS_STAIRS_CLOSED_INT;
        deserializeJson(doc, lastPayload);
        points_stairs = hex2Long((const char *) doc["points"]);
        duration = hex2Long((const char *) doc["duration"]);
        const char *hexSignal = doc["signal"];
        hexStringToFloatArray(stairs, hexSignal, points_stairs);
        total_time = points_stairs * duration - 1;
        Serial.printf("Stairs signal of %d steps  with a duration of %0.2f secs.\n", points_stairs, h * total_time);
        resumeControl();
        vTaskResume(h_publishStateTask);
    }

    else if(strstr(lastTopic, USER_SYS_PRBS_OPEN )) {
        codeTopic = USER_SYS_PRBS_OPEN_INT;
        deserializeJson(doc, lastPayload);
        high_val = hex2Float((const char *) doc["peak_amp"]);
        reference = hex2Float((const char *) doc["op_point"]);
        stab_points = hex2Long((const char *) doc["stab_points"]);
        uee_points = hex2Long((const char *) doc["uee_points"]);
        divider = hex2Long((const char *) doc["divider"]);
        prbs_points = 63 * divider;
        total_time = prbs_points + stab_points + uee_points;
        reset_int = true;
        printf("Open loop test with a prbs signal with %d steps with a duration of %0.2f secs.\n",
               prbs_points, (total_time-1) * h);
        vTaskResume(h_publishStateTask);
        vTaskResume(h_identifyTask);
    }

    else if(strstr(lastTopic, USER_SYS_STEP_OPEN )){
        codeTopic = USER_SYS_STEP_OPEN_INT;
        vTaskSuspend(h_identifyTask);
        vTaskSuspend(h_controlPidTask);
        vTaskSuspend(h_generalControlTask);
        deserializeJson(doc, lastPayload);
        high_val = hex2Float((const char *) doc["amplitude"]);
        reference = hex2Float((const char *) doc["op_point"]);
        stab_points = hex2Long((const char *) doc["stab_points"]);
        uee_points = hex2Long((const char *) doc["uee_points"]);
        points_high = hex2Long((const char *) doc["points_high"]);
        total_time = points_high + stab_points + uee_points;
        reset_int = true;
        printf("Open loop step response\n");
        printf("\tOperation point= %0.2f\n\tAmplitude=%0.2f\n\tDuration=%0.2f\n", reference, high_val, (points_high-1)*h);
        vTaskResume(h_publishStateTask);
        vTaskResume(h_identifyTask);
    }

    else if(strstr(lastTopic, USER_SYS_SET_GENCON)){
        deserializeJson(doc, lastPayload);
        order = hex2Long((const char *) doc["order"]);
        const char *hex_A = doc["A"];
        const char *hex_B = doc["B"];
        const char *hex_C = doc["C"];
        const char *hex_D = doc["D"];
        const char *hex_L = doc["L"];
        typeControl  = hex2Long((const char *) doc["typeControl"]);
        hexStringToMatrix( *A, hex_A, order, order, MAX_ORDER);
        hexStringToMatrix( *B, hex_B, order, 2, 2);
        hexStringToFloatArray(C, hex_C, order);
        hexStringToFloatArray(D, hex_D, 2);
        hexStringToFloatArray(L, hex_L, order);
        printf("A general controller or order %d has been loaded\n", order);
        printf("A=\n");
        for ( uint8_t  i =0; i < order; i++ ){
            for (size_t j =0; j < order; j++ ) {
                printf("%0.4f    ", A[i][j]);
            }
            printf("\n");
        }
        printf("B=\n");
        for (uint8_t  i =0; i < order; i++ ) {
            for (size_t j =0; j < 2; j++ ) {
                printf("%0.4f    ", B[i][j]);
            }
            printf("\n");
        }
        printf("C=\n");
        for (uint8_t  j =0; j < order; j++ ) {
            printf("%0.4f    ", C[j]);
        }
        printf("\n");
        printf("D = %0.4f   %0.4f \n", D[0], D[1]);
        printf("L=\n");
        for (uint8_t  j =0; j < order; j++ ) {
            printf("%0.4f    ", L[j]);
        }
        printf("\n");
        printf("control code: %d\n", typeControl);
        reset_int = true;
        defaultControl();
    }
    else if(strstr(lastTopic, USER_SYS_PROFILE_CLOSED)){
        codeTopic = USER_SYS_PROFILE_CLOSED_INT;
        deserializeJson(doc, lastPayload);
        points_stairs = hex2Long((const char *) doc["points"]);
        low_val = hex2Float((const char *) doc["min_val"]);
        high_val = hex2Float((const char *) doc["max_val"]);
        const char *timeValuesHex = doc["timevalues"];
        const char *stairsHex = doc["refvalues"];
        hexStringToFloatArray(stairs, stairsHex, points_stairs);
        hexStringToLongArray(timeValues, timeValuesHex, points_stairs);
        total_time = timeValues[points_stairs - 1];
        reset_int = true;
        printf("Closed loop profile response with a duration of %0.2f secs.\n", total_time * h);
        printf("   refvalues = [ ");
        for (uint8_t  j =0; j < points_stairs; j++ ) {
            printf("%0.2f ", stairs[j]);
        }
        printf("]\n");
        printf("   timevalues = [ ");
        for (uint8_t  j =0; j < points_stairs; j++ ) {
            printf("%0.2f ", timeValues[j] * h);
        }
        printf("]\n");
        resumeControl();
        vTaskResume(h_publishStateTask);

    }


}

void initMqtt() {
    mqttClient.setServer(BROKER, PORT);
    mqttClient.setCallback(onMqttReceived);
    mqttClient.setBufferSize (4096);
}

void publishStateClosed()
{
    JsonDocument doc;
    doc["np"] = long2Hex(np);
    doc["r"] = float2Hex(reference);
    doc["y"] = float2Hex(y);
    doc["u"] = float2Hex(usat);
    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer); // print to client
    mqttClient.publish(SYS_USER_SIGNALS_CLOSED, jsonBuffer);
}

void publishStateOpen()
{
    JsonDocument doc;
    doc["np"] = long2Hex(np);
    doc["y"] = float2Hex(y);
    doc["u"] = float2Hex(usat);
    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer); // print to client
    mqttClient.publish(SYS_USER_SIGNALS_OPEN, jsonBuffer);
}


float movingAverage(float newValue) {
    const int filterSize = 10;
    static float filter_values[filterSize] = {0.0};  // Array to store the last 'FILTER_SIZE' values
    static int index = 0;  // Current index in the array
    static float sum = 0.0;  // Sum of the current values in the array
    static int count = 0;  // Number of values added to the filter

    // Subtract the oldest value from sum and replace it with the new value
    sum = sum - filter_values[index] + newValue;
    filter_values[index] = newValue;

    // Update the index, and wrap around if necessary
    index = (index + 1) % filterSize ;

    // Keep track of how many values have been added to the filter
    if (count < filterSize ) {
        count++;
    }
    // Return the average of the values in the filter
    return sum / count;
}
