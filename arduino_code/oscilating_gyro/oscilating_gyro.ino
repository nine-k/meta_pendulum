union SensorData {                                                                                                      
    int16_t val;                                                                                                         
    uint16_t bytes;                                                                                                      
};    

SensorData gx, gy, gz;

uint16_t it_counter = 0;

void setup() {
    gx.val = -10;
    gy.val = 12124;
    gz.val = -12412;
    Serial.begin(38400);
}


void loop() {
    Serial.write((uint8_t)(gx.bytes >> 8));  Serial.write((uint8_t)(gx.bytes & 0xFF));
	 Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)(gy.bytes >> 8));  Serial.write((uint8_t)(gy.bytes & 0xFF));
	 Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)(gz.bytes >> 8));  Serial.write((uint8_t)(gz.bytes & 0xFF));
	 Serial.write((uint8_t)0xFF);
    Serial.write((uint8_t)0);
    Serial.write((uint8_t)0);
    Serial.write((uint8_t)0);
	 Serial.write((uint8_t)0xFF);
    ++it_counter;
    if (it_counter == 1000) {
        gx.val *= -1;
        it_counter = 0;
    } 
    //  Serial.write((uint8_t)(gx.bytes >> 8));  Serial.write((uint8_t)(gx.bytes & 0xFF));
    //  Serial.write((uint8_t)(gy.bytes >> 8));  Serial.write((uint8_t)(gy.bytes & 0xFF));
    //  Serial.write((uint8_t)(gz.bytes >> 8));  Serial.write((uint8_t)(gz.bytes & 0xFF));
    //  Serial.write((uint8_t)0);
    //  Serial.write((uint8_t)0);
    //  Serial.write((uint8_t)0);
}
