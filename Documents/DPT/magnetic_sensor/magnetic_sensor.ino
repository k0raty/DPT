/*
  Arduino Starter Kit example
  Project 3 - Love-O-Meter

  This sketch is written to accompany Project 3 in the Arduino Starter Kit

  Parts required:
  - one TMP36 temperature sensor
  - three red LEDs
  - three 220 ohm resistors

  created 13 Sep 2012
  by Scott Fitzgerald

  https://store.arduino.cc/genuino-starter-kit

  This example code is part of the public domain.
*/

// named constant for the pin the sensor is connected to
const int sensorPin = A0;

void setup() {
  // open a serial connection to display values
  Serial.begin(9600);
  // set the LED pins as outputs
  // the for() loop saves some extra coding
}

void loop() {
  // read the value on AnalogIn pin 0 and store it in a variable
  int sensorVal = analogRead(sensorPin);

  // send the 10-bit sensor value out the serial port
  Serial.print("sensor Value: ");
  Serial.println(sensorVal);
 
}
