
void setup() {
  Serial.begin(9600);  //Start Serial Monitor to display current read value on Serial monitor
}

unsigned int msg_idx = 0;
long lastMillis = 0;

void loop() {
  float sample_cnt = 200.0;
  float AcsValue = 0.0, Samples = 0.0, AvgAcs = 0.0, AcsValueF = 0.0;
  float MaxAcsValue = 0.0, MinAcsValue = 600.0, MaxAcsValueF = 0.0, MinAcsValueF = 0.0;

  // Average Sampling
  for (int x = 0; x < sample_cnt; x++) {
    AcsValue = analogRead(A0);  //Read current sensor values
    MaxAcsValue = AcsValue > MaxAcsValue ? AcsValue : MaxAcsValue;
    MinAcsValue = AcsValue < MinAcsValue ? AcsValue : MinAcsValue;
    Samples = Samples + AcsValue;  //Add samples together
  }
  AvgAcs = Samples / sample_cnt;  //Taking Average of Samples
  MaxAcsValueF = ((MaxAcsValue * 5.0 / 1024.0) - 2.5) / 0.185;
  MinAcsValueF = ((MinAcsValue * 5.0 / 1024.0) - 2.5) / 0.185;
  AcsValueF = ((AvgAcs * 5.0 / 1024.0) - 2.5) / 0.185;

  Serial.println(String(MaxAcsValueF, 4) + "," + String(AcsValueF, 4) + "," + String(MinAcsValueF, 4));
}
