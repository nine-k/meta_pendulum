void setup() {
  Serial.begin(34800);
}

union SensorData {                                                                                                      
    int16_t val;                                                                                                         
    uint16_t bytes;                                                                                                      
};    

void loop() {
  // put your main code here, to run repeatedly:
  SensorData gx, gy, gz;
  gx.val = -10;
  gy.val = 12124;
  gz.val = -12412;
  Serial.write((uint8_t)(gx.bytes >> 8));  Serial.write((uint8_t)(gx.bytes & 0xFF));
  Serial.write((uint8_t)(gy.bytes >> 8));  Serial.write((uint8_t)(gy.bytes & 0xFF));
  Serial.write((uint8_t)(gz.bytes >> 8));  Serial.write((uint8_t)(gz.bytes & 0xFF));
  Serial.write((uint8_t)0);
  Serial.write((uint8_t)0);
  Serial.write((uint8_t)0);
//  Serial.write((uint8_t)(gx.bytes >> 8));  Serial.write((uint8_t)(gx.bytes & 0xFF));
//  Serial.write((uint8_t)(gy.bytes >> 8));  Serial.write((uint8_t)(gy.bytes & 0xFF));
//  Serial.write((uint8_t)(gz.bytes >> 8));  Serial.write((uint8_t)(gz.bytes & 0xFF));
//  Serial.write((uint8_t)0);
//  Serial.write((uint8_t)0);
//  Serial.write((uint8_t)0);
}
