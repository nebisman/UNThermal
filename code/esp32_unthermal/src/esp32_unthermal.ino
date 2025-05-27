#include "connection_settings.h"
#include "definitions.h"



void  computeReference() {
   switch (codeTopic) {
        case DEFAULT_TOPIC:
            if (touchRead(BUTTON_MINUS) > THRESHOLD_MINUS){
                reference = constrain(reference - 5,40,90);
            }
            if (touchRead(BUTTON_PLUS) > THRESHOLD_PLUS){
                reference = constrain(reference + 5,40,90);
            }


            displayLed(y, 40, 90, 0.3, 0);
            displayLed(reference, 40, 90, 0.3, 1);
            displayLed(usat, 0, 100, 0.2, 2);
            break;

        case USER_SYS_STEP_CLOSED_INT:
            if (np < points_low) {
                reference = low_val;
                displayLed(y, 20, 90, 0.3, 0);
                displayLed(reference, 20, 90, 0.3, 1);
                displayLed(usat, 0, 100, 0.1, 2);

            }
            else if (np <= total_time) {
                reference = high_val;
                displayLed(y, 20, 90, 0.3, 0);
                displayLed(reference, 20, 90, 0.3, 1);
                displayLed(usat, 0, 100, 0.1, 2);
            }
            else if (np <= total_time + 1){
                printf("Closed loop step response completed\n");
                reference = low_val;
                defaultControl();
            }
            break;

        case USER_SYS_STAIRS_CLOSED_INT:
            if (np <= total_time) {
                reference = stairs[np / duration];
                displayLed(y, 20, 90, 0.3, 0);
                displayLed(reference, 20, 90, 0.3, 1);
                displayLed(usat, 0, 100, 0.1, 2);
            }
            else if (np == total_time + 1) {
                printf("Stairs closed loop response completed\n");
                ledcWrite(PWM_CHANNEL, 0 * percent2pwm);
                defaultControl();
            }
            break;
       case USER_SYS_PROFILE_CLOSED_INT:
           if (np <= total_time) {
               reference = linearInterpolation(timeValues, stairs, points_stairs, np);
               displayLed(y, 20, 90, 0.3, 0);
               displayLed(reference, 20, 90, 0.3, 1);
               displayLed(usat, 0, 100, 0.1, 2);

           }
           else if (np > total_time) {
               printf("Closed loop profile response completed\n");
               wattsToPlant(0);
               defaultControl();
           }
           break;
    }

}

float computeController(bool type){
    /* This function computes a general controlller for the thermal system
       The controller is defined by the state equation
           x[n+1] = (A - L * C) * x[k] + (B - L * D) * [r[n] y[n]]^T + L *u[n]
       where A, B, C ,D are the state equation matrices and L is the observer's gain for antiwindup,
       r[n] and y[n] are, respectively, the current reference and the current temperature
       of the thermal system

    */

    static float X[10] = {0};
    float Xnew[10] = {0};
    static float control = 0;
    float e;
    if (reset_int) {
        for (size_t  i = 0; i < order; i++){
            X[i] = 0;
        }
        control = 0;
        return 0;
    }
    else {
        e = reference - y;
        for (size_t i = 0; i < order; i++) {
            for (size_t j = 0; j < order; j++) {
                Xnew[i] += A[i][j] * X[j];
            }
            if (type) {
                Xnew[i] += B[i][0] * reference + B[i][1] * y + L[i] * control;
            }
            else{
                Xnew[i] += B[i][0] * e + L[i] * control;
            }
        }


        control = D[0] * reference + D[1] * y;
        for (size_t i = 0; i < order; i++) {
            control += C[i] * X[i];
        }
        for (size_t i = 0; i < order; i++) {
            X[i] = Xnew[i];
        }
        control = constrain(control, 0, 100);
        return control;
    }

}


static void controlPidTask(void *pvParameters) {
    /* this function computes a two parameter PID control with Antiwinduph
     See Astrom
    */
    const TickType_t taskPeriod = (pdMS_TO_TICKS(1000*h));
    float bi;
    float ad;
    float bd;
    float PA;              //  proportional action
    float DA;          //  derivative action
    float y_ant;       //  past output
    float IA;
    TickType_t xLastWakeTime;
    for (;;) {
        xLastWakeTime = xTaskGetTickCount();
        if (reset_int) {
            np = 0;
            IA = 0;
            reset_int = false;
            continue;
        }
        y = sensors.getTempCByIndex(0);
        computeReference();

        // This is a firmware protection in case of sensor failure or overtemperature.
        if (y <= 0 || y >= 100) {
            wattsToPlant(0);
            continue;
        }
        bi = ki*h;
        ad = kd/(N*(kd/N + h));
        bd = kd/(kd/N + h);
        PA = kp * (beta * reference - y);
        DA = ad * DA - bd * (y - y_ant); // derivative action
        u = PA + IA + DA;// + uN ; // control signal
        usat =  constrain(u, 0, 100); //señal de control saturada
        wattsToPlant(usat); // Enviar señal de control en bits
             
        if (ki!= 0) {
           IA = IA + bi *(reference - y) + br*(usat - u);  // updating integral action
        }
       
        y_ant = y;
        xTaskNotify(h_publishStateTask, 0b0001, eSetBits);
        sensors.requestTemperatures();
        xTaskDelayUntil(&xLastWakeTime, taskPeriod);
        np+=1;
    }
}



static void generalControlTask(void *pvParameters) {
    // Set the sampling time for the task at 750ms
    const TickType_t taskPeriod = (pdMS_TO_TICKS(1000*h));
    TickType_t xLastWakeTime;
    for (;;) {

        xLastWakeTime = xTaskGetTickCount();
        if (reset_int) {
            computeController(false);
            reset_int = false;
            np = 0;
            continue;
        }


        y = sensors.getTempCByIndex(0);
        computeReference();


        //This is to protect the system in case of sensor failure or overtemperature.

        if (y <= 0 || y >= 100) {
            wattsToPlant(0);

        }

        if (typeControl == GENERAL_CONTROLLER_1P){
            u = computeController(false);
        }
        else if (typeControl == GENERAL_CONTROLLER_2P) {
            u = computeController( true);
        }
        usat = constrain(u, 0, 100);
        wattsToPlant( usat);            
        xTaskNotify(h_publishStateTask, 0b0001, eSetBits);
        sensors.requestTemperatures();
        xTaskDelayUntil(&xLastWakeTime, taskPeriod);
        np +=1;
    }
}




static void identifyTask(void *pvParameters) {
    /*
     This is the task for identification of the thermal plant
     by means of a PBRS signal
    */
    const TickType_t taskPeriod = (long) (1000 * h);
    const uint64_t pbrs = 0x57E08629E8E4B766;
    static uint bitShift = 0;
    float kp_id = 11.8552;
    float beta_id = 0;
    float bi = 2.714634 * h;
    bool  currBit;
    float PA;          //  proportional action
    float uf;          //  filtered control signal
    float IA;

    TickType_t xLastWakeTime;

    for (;;) {
        xLastWakeTime = xTaskGetTickCount();
        if (reset_int) {
            np = 0;
            IA = 0;
            reset_int = false;
            continue;
        }
        // This is for protecting the system both if the sensor fails or there is overtemperature
        if (y <= 0 || y >= 100) {
            wattsToPlant(0);
            continue;
        }


        y = sensors.getTempCByIndex(0);

        if (np < stab_points) {
            PA = kp_id * (beta_id * reference - y);
            u = PA + IA ; // control signal
            usat =  constrain(u, 0, 100); // saturated control signal
            wattsToPlant(usat);
            uf = movingAverage(usat);
            IA = IA + bi *(reference - y) + br*(usat - u);  // integral action
            displayLed(u, 0, 100, 0.25, 2);
            displayLed(y, 20, 90, 0.25, 0);
            displayLed(reference, 20, 90, 0.25, 1);

        }
        else if (np <= stab_points + uee_points) {
            usat = uf;
            wattsToPlant(usat);
            displayLed(usat, uf - 1.2*high_val, uf + 1.2*high_val, 0.25, 2);
            displayLed(y, reference - 5, reference + 5,  0.25,0);
            displayLed(reference, reference - 5, reference + 5,  0.25,1);
        }

        else if ((np <= total_time) & (codeTopic == USER_SYS_PRBS_OPEN_INT)) {
            bitShift = (int) ((np - stab_points - uee_points) / divider);
            currBit = (pbrs >> bitShift) & 1;
            if (currBit) {
                usat = high_val + uf;
            } else {
                usat = uf - high_val;
            }
            wattsToPlant(usat);
            displayLed(usat, uf - 1.2*high_val, uf + 1.2*high_val, 0.25, 2);
            displayLed(y, reference - 5, reference + 5,  0.25,0);
            displayLed(reference, reference - 5, reference + 5,  0.25,1);

        }
        else if ((np <= total_time) & (codeTopic == USER_SYS_STEP_OPEN_INT)) {
            usat = high_val + uf;
            wattsToPlant(usat);
            displayLed(usat, uf - 1.2*high_val, uf + 1.2*high_val, 0.25, 2);
            displayLed(y, reference - 5, 1.23*(uf + high_val) + 30,  0.25,0);
            displayLed(reference, reference - 5, 1.23*(uf + high_val) + 30,  0.25,1);

        }
        else if (np > total_time) {
            wattsToPlant(0);
            printf("Open loop step response completed\n");
            defaultControl();
        }
        xTaskNotify(h_publishStateTask,0b0010, eSetBits);
        sensors.requestTemperatures();
        xTaskDelayUntil(&xLastWakeTime, taskPeriod);
        np +=1;

    }
}


static void publishStateTask (void *pvParameters) {
    // local constants
    uint32_t rv;
    for (;;) {
        xTaskNotifyWait(0, 0b0011, &rv, portMAX_DELAY);
        if ( rv & 0b0001 ){
            publishStateClosed();
        }
        else if ( rv & 0b0010 ){
            publishStateOpen();
        }
    }
}



//void gotButtPlus() {
//    plusDetected = true;
//}
//
//void gotButtMinus() {
//    minusDetected = true;
//}
//
//
//
//static void menuTask(void *pvParameters) {
//    // Set the sampling time for the task at 750ms
//    for (;;) {
//        if (plusDetected) {
//            plusDetected = false;
//            if (touchInterruptGetLastStatus(BUTTON_PLUS)) {
//               reference = constrain(reference + 5,15,90);
//            }
//        }
//        if (minusDetected) {
//            minusDetected = false;
//            if (touchInterruptGetLastStatus(BUTTON_MINUS)) {
//                reference = constrain(reference - 5,15,90);
//            }
//        }
//       // printf("reference=%f       y=%f\n", reference, y);
//        vTaskDelay(1000);
//
//    }
//}








void setup() {
    // Initialize serial and wait for port to open:
    vTaskPrioritySet(nullptr,24);
    Serial.begin(115200);
    sensors.begin();

    sensors.requestTemperatures();
    pinMode(POWER_PIN , OUTPUT);
    ledcSetup(PWM_CHANNEL,  FREQUENCY_PWM, RESOLUTION_PWM);
    ledcAttachPin(POWER_PIN, PWM_CHANNEL);

    // Led Configuration
    dispLeds.begin();
    dispLeds.clear();
    dispLeds.setBrightness(20);
    dispLeds.show();

    xTaskCreatePinnedToCore(
            publishStateTask, // This is communication task
            "User command IOT",
            8192,
            NULL,
            12,
            &h_publishStateTask,
            CORE_COMM
    );

    // Create the task for PID control in core 0.
    xTaskCreatePinnedToCore(
            controlPidTask, // This is the control routine
            "PID control",
            8192,
            NULL,
            23,
            &h_controlPidTask,
            CORE_CONTROL
    );

   // vTaskSuspend(h_controlPidTask);

    // Create the task for general control in core 0.
    xTaskCreatePinnedToCore(
            generalControlTask, // This is the control routine
            "general control",
            8192,
            NULL,
            22,
            &h_generalControlTask,
            CORE_CONTROL
    );
    vTaskSuspend(h_generalControlTask);


    xTaskCreatePinnedToCore(
            identifyTask,
            "prbs-ident",
            8192,
            NULL,
            20,
            &h_identifyTask,
            CORE_CONTROL
    );
    vTaskSuspend(h_identifyTask);

//    /**  These functions activate the menu */
//    xTaskCreatePinnedToCore(
//            menuTask,
//            "prbs-ident",
//            2048,
//            NULL,
//            10,
//            &h_menuTask,
//            CORE_COMM
//    );
//
//
//    touchAttachInterrupt(BUTTON_PLUS, gotButtPlus, THRESHOLD_PLUS);
//    touchAttachInterrupt(BUTTON_MINUS, gotButtMinus, THRESHOLD_MINUS);


    /**  These functions activate communication */

    connectWiFi();
    initMqtt();
    connectMqtt();
    xTaskCreatePinnedToCore(
            handleConnections, // This is communication task
            "handle connections",
            4096,
            nullptr,
            10,
            nullptr,
            CORE_COMM // communications are attached to core 1
    );

}


void loop() {
    vTaskDelete(nullptr);
}


