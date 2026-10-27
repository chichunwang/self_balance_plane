// 量 USB 來回延遲（暫存量測用）：筆電送 16 bytes，Teensy 原封不動送回，後面加上收齊那一刻的 micros()。
// Arduino IDE：板子選 Teensy 4.1，USB Type 選 Serial。

uint8_t buf[16];
int n = 0;

void setup() {
  Serial.begin(115200);   // USB 序列的速度設多少都一樣，實際走 480 Mbit
}

void loop() {
  while (Serial.available()) {
    buf[n++] = Serial.read();
    if (n == 16) {
      uint32_t t = micros();
      Serial.write(buf, 16);
      Serial.write((uint8_t *)&t, 4);
      Serial.send_now();  // 不等緩衝區滿，馬上送出
      n = 0;
    }
  }
}
