// Teensy 4.1 / USB Type: Serial. Protocol v1: 40-byte little-endian frames.
// AA 55 | version:u8 | type:u8 | seq:u32 | payload:28 | reserved:2 | CRC16-CCITT:u16
#include <Arduino.h>
#include <Wire.h>
#include <IntervalTimer.h>

constexpr uint8_t VERSION = 1, FRAME_SIZE = 40;
constexpr uint8_t PWM_PINS[3] = {2, 3, 4};
constexpr uint8_t PIEZO_PIN = 14, IMU_ADDR = 0x68;
constexpr uint32_t WATCHDOG_US = 200000, HOME_US = 1000000;
enum Kind : uint8_t {
  SET = 1, CONFIG = 2, PING = 3, HOME = 4, OFF = 5, LED = 6,
  STATUS = 0x80, IMU = 0x81, PIEZO = 0x82, PONG = 0x83,
  ACK = 0x84, LED_EVENT = 0x85, DIAG = 0x86, FW_EVENT = 0x87
};
enum EventCode : uint8_t {
  EV_BOOT = 1, EV_PWM_START = 2, EV_WATCHDOG = 3, EV_HOME = 4,
  EV_HOME_DONE = 5, EV_RESUMED = 6, EV_OFF = 7, EV_CONFIG = 8
};

IntervalTimer safetyTimer;
volatile uint16_t pulses[3] = {15000, 15000, 15000}; // 0.1 us
volatile uint16_t centers[3] = {15000, 15000, 15000};
volatile uint16_t lower[3] = {5000, 5000, 5000};
volatile uint16_t upper[3] = {25000, 25000, 25000};
volatile uint16_t homeFrom[3];
volatile bool enabled = false, homing = false, timedOut = false, configured = false;
volatile uint32_t lastSet = 0, lastMotionSeq = 0, homeStart = 0;
volatile uint16_t appliedPulses[3] = {0, 0, 0};
volatile uint32_t setCount = 0, watchdogCount = 0, maxSetGap = 0, pwmWrites = 0;
struct EventSlot {
  uint32_t stamp, commandSeq, commandAge, number;
  uint16_t pulse[3];
  uint8_t code, flags;
};
volatile EventSlot eventBuffer[32];
volatile uint8_t eventHead = 0, eventTail = 0;
volatile uint32_t eventNumber = 0, eventDropped = 0;
volatile uint32_t hitTimes[64];
volatile uint8_t hitHead = 0, hitTail = 0;
volatile uint32_t hitOverflow = 0;
uint32_t badFrames = 0, txDropped = 0, imuErrors = 0, txSeq = 0, hitSeq = 0;
uint32_t lastStatus = 0, lastImu = 0, lastImuInit = 0, lastDiag = 0, resetReason = 0;
bool bootAnnounced = false;
uint16_t imuHz = 100, busKHz = 400;
uint8_t who = 0xff, piezoEdge = 0;
bool imuReady = false;
uint8_t rx[FRAME_SIZE];
size_t rxN = 0;
uint32_t rxLast = 0;

uint16_t read16(const uint8_t *p) { return p[0] | (uint16_t(p[1]) << 8); }
uint32_t read32(const uint8_t *p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
void put16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = v >> (8 * i);
}
uint16_t crc16(const uint8_t *p, size_t n) {
  uint16_t crc = 0xffff;
  while (n--) {
    crc ^= uint16_t(*p++) << 8;
    for (int j = 0; j < 8; ++j) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}
bool sendFrame(uint8_t kind, uint32_t seq, const uint8_t *data, size_t len) {
  if (!Serial || Serial.availableForWrite() < FRAME_SIZE) { ++txDropped; return false; }
  uint8_t out[FRAME_SIZE] = {0xaa, 0x55, VERSION, kind};
  put32(out + 4, seq);
  memcpy(out + 8, data, min(len, size_t(28)));
  put16(out + 38, crc16(out, 38));
  Serial.write(out, FRAME_SIZE);
  Serial.send_now();
  return true;
}
void ack(uint8_t kind, uint32_t seq, uint8_t code = 0) {
  uint8_t p[2] = {kind, code};
  sendFrame(ACK, seq, p, sizeof(p));
}

uint8_t stateFlags() {
  return (enabled ? 1 : 0) | (homing ? 2 : 0) | (imuReady ? 4 : 0) |
         (configured ? 8 : 0) | (timedOut ? 16 : 0);
}
// Call from the timer ISR or with interrupts disabled. No Serial or allocation here.
void queueEventLocked(uint8_t code, uint32_t now, uint32_t age) {
  uint32_t number = ++eventNumber;
  uint8_t next = (eventHead + 1) & 31;
  if (next == eventTail) { ++eventDropped; return; }
  volatile EventSlot &e = eventBuffer[eventHead];
  e.stamp = now; e.code = code; e.flags = stateFlags();
  e.commandSeq = lastMotionSeq; e.commandAge = age; e.number = number;
  for (int i = 0; i < 3; ++i) e.pulse[i] = pulses[i];
  eventHead = next;
}
void startHome(uint32_t now, uint8_t reason) {
  if (!enabled) return;
  for (int i = 0; i < 3; ++i) homeFrom[i] = pulses[i];
  homeStart = now;
  homing = true;
  timedOut = true;
  if (reason == EV_WATCHDOG) ++watchdogCount;
  queueEventLocked(reason, now, uint32_t(now - lastSet));
}
void safetyTick() {
  if (!enabled) return;
  const uint32_t now = micros();
  if (!timedOut && uint32_t(now - lastSet) >= WATCHDOG_US) startHome(now, EV_WATCHDOG);
  if (homing) {
    uint32_t elapsed = uint32_t(now - homeStart);
    if (elapsed >= HOME_US) {
      for (int i = 0; i < 3; ++i) pulses[i] = centers[i];
      homing = false;
      queueEventLocked(EV_HOME_DONE, now, uint32_t(now - lastSet));
    } else {
      for (int i = 0; i < 3; ++i)
        pulses[i] = int32_t(homeFrom[i]) + (int64_t(centers[i]) - homeFrom[i]) * elapsed / HOME_US;
    }
  }
  for (int i = 0; i < 3; ++i) {
    uint16_t v = constrain(pulses[i], lower[i], upper[i]);
    // pulse(0.1us) * 300Hz * 32768 / 10,000,000
    if (v != appliedPulses[i]) {
      analogWrite(PWM_PINS[i], (uint64_t(v) * 300 * 32768 + 5000000) / 10000000);
      appliedPulses[i] = v;
      ++pwmWrites;
    }
  }
}
void piezoISR() {
  const uint32_t stamp = micros();
  uint8_t next = (hitHead + 1) & 63;
  if (next == hitTail) { ++hitOverflow; return; }
  hitTimes[hitHead] = stamp;
  hitHead = next;
}
void setPiezo(uint8_t edge) {
  detachInterrupt(digitalPinToInterrupt(PIEZO_PIN));
  pinMode(PIEZO_PIN, INPUT_PULLUP);
  piezoEdge = edge;
  if (edge) attachInterrupt(digitalPinToInterrupt(PIEZO_PIN), piezoISR, edge == 1 ? RISING : FALLING);
}
bool imuWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg); Wire.write(value);
  return Wire.endTransmission() == 0;
}
bool imuRead(uint8_t reg, uint8_t *out, uint8_t count) {
  Wire.beginTransmission(IMU_ADDR); Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(IMU_ADDR, count) != count) return false;
  for (uint8_t i = 0; i < count; ++i) out[i] = Wire.read();
  return true;
}
void initImu() {
  lastImuInit = micros();
  imuReady = false;
  if (!imuRead(0x75, &who, 1) || !imuWrite(0x6b, 0x01) ||
      !imuWrite(0x1a, 0x03) || !imuWrite(0x19, 0x00) ||
      !imuWrite(0x1b, 0x00) || !imuWrite(0x1c, 0x00)) {
    ++imuErrors; return;
  }
  imuReady = true; // Raw values and WHO_AM_I are returned even for another ID.
}
void handleFrame() {
  const uint8_t kind = rx[3];
  const uint32_t seq = read32(rx + 4);
  const uint8_t *p = rx + 8;
  if (kind == SET) {
    if (!configured) { ack(kind, seq, 1); return; }
    noInterrupts();
    uint32_t now = micros(), age = enabled ? uint32_t(now - lastSet) : 0;
    bool wasEnabled = enabled, wasReturning = homing || timedOut;
    if (wasEnabled && age > maxSetGap) maxSetGap = age;
    for (int i = 0; i < 3; ++i) pulses[i] = constrain(read16(p + 2 * i), lower[i], upper[i]);
    homing = timedOut = false;
    lastSet = now; lastMotionSeq = seq; enabled = true; ++setCount;
    if (!wasEnabled) queueEventLocked(EV_PWM_START, now, age);
    else if (wasReturning) queueEventLocked(EV_RESUMED, now, age);
    interrupts();
  } else if (kind == CONFIG) {
    uint16_t c[3], lo[3], hi[3];
    for (int i = 0; i < 3; ++i) {
      c[i] = read16(p + 2 * i); lo[i] = read16(p + 6 + 2 * i); hi[i] = read16(p + 12 + 2 * i);
      if (lo[i] < 5000 || hi[i] > 25000 || lo[i] >= hi[i] || c[i] < lo[i] || c[i] > hi[i]) {
        ack(kind, seq, 2); return;
      }
    }
    uint16_t rate = read16(p + 18), bus = read16(p + 21);
    if (rate > 250 || p[20] > 2 || (bus != 100 && bus != 400)) { ack(kind, seq, 2); return; }
    bool valid = true;
    noInterrupts();
    for (int i = 0; i < 3; ++i) if (enabled && (pulses[i] < lo[i] || pulses[i] > hi[i])) valid = false;
    if (valid) {
      for (int i = 0; i < 3; ++i) { centers[i] = c[i]; lower[i] = lo[i]; upper[i] = hi[i]; }
      configured = true;
      uint32_t now = micros();
      queueEventLocked(EV_CONFIG, now, enabled ? uint32_t(now - lastSet) : 0);
    }
    interrupts();
    if (!valid) { ack(kind, seq, 3); return; }
    imuHz = rate;
    if (busKHz != bus) { busKHz = bus; Wire.setClock(uint32_t(bus) * 1000); }
    if (piezoEdge != p[20]) setPiezo(p[20]);
    ack(kind, seq);
  } else if (kind == PING) {
    uint8_t data[28]; put32(data, micros()); memcpy(data + 4, p, 24);
    sendFrame(PONG, seq, data, sizeof(data));
  } else if (kind == HOME) {
    noInterrupts(); startHome(micros(), EV_HOME); interrupts(); ack(kind, seq);
  } else if (kind == OFF) {
    noInterrupts();
    enabled = homing = timedOut = false;
    for (int i = 0; i < 3; ++i) {
      pinMode(PWM_PINS[i], OUTPUT); digitalWrite(PWM_PINS[i], LOW); appliedPulses[i] = 0;
    }
    queueEventLocked(EV_OFF, micros(), 0);
    interrupts(); ack(kind, seq);
  } else if (kind == LED) {
    digitalWrite(LED_BUILTIN, p[0] ? HIGH : LOW);
    uint8_t data[5]; put32(data, micros()); data[4] = p[0] ? 1 : 0;
    sendFrame(LED_EVENT, seq, data, sizeof(data));
  } else {
    ack(kind, seq, 4);
  }
}
void receive() {
  if (rxN && uint32_t(micros() - rxLast) > 20000) { rxN = 0; ++badFrames; }
  int budget = FRAME_SIZE * 8;
  while (budget-- > 0 && Serial.available()) {
    uint8_t b = Serial.read(); rxLast = micros();
    if (rxN == 0 && b != 0xaa) continue;
    if (rxN == 1 && b != 0x55) { rxN = b == 0xaa ? 1 : 0; continue; }
    rx[rxN++] = b;
    if (rxN < FRAME_SIZE) continue;
    if (rx[2] == VERSION && crc16(rx, 38) == read16(rx + 38)) {
      handleFrame(); rxN = 0;
    } else {
      ++badFrames;
      size_t start = 1;
      while (start < FRAME_SIZE && !(rx[start] == 0xaa && (start == FRAME_SIZE - 1 || rx[start + 1] == 0x55))) ++start;
      rxN = FRAME_SIZE - start;
      memmove(rx, rx + start, rxN);
    }
  }
}
void sendStatus() {
  uint8_t p[28] = {};
  noInterrupts();
  put32(p, micros()); put32(p + 4, lastMotionSeq);
  for (int i = 0; i < 3; ++i) put16(p + 8 + 2 * i, pulses[i]);
  p[14] = stateFlags();
  interrupts();
  p[15] = who; put32(p + 16, badFrames); put32(p + 20, imuErrors); put32(p + 24, txDropped);
  sendFrame(STATUS, txSeq++, p, sizeof(p));
}
void sendDiagnostics() {
  uint8_t p[28];
  noInterrupts();
  put32(p, millis()); put32(p + 4, setCount); put32(p + 8, watchdogCount);
  put32(p + 12, maxSetGap); put32(p + 16, pwmWrites); put32(p + 20, eventDropped);
  interrupts();
  put32(p + 24, resetReason);
  sendFrame(DIAG, txSeq++, p, sizeof(p));
}
void sendEvents() {
  if (!Serial) return;
  for (int n = 0; n < 8; ++n) {
    uint8_t p[28] = {};
    noInterrupts();
    if (eventHead == eventTail) { interrupts(); break; }
    volatile EventSlot &e = eventBuffer[eventTail];
    put32(p, e.stamp); p[4] = e.code; p[5] = e.flags;
    put32(p + 8, e.commandSeq); put32(p + 12, e.commandAge);
    for (int i = 0; i < 3; ++i) put16(p + 16 + 2 * i, e.pulse[i]);
    put32(p + 24, e.number);
    eventTail = (eventTail + 1) & 31;
    interrupts();
    if (!sendFrame(FW_EVENT, txSeq++, p, sizeof(p))) {
      noInterrupts(); ++eventDropped; interrupts();
    }
  }
}
void setup() {
  resetReason = SRC_SRSR;
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  analogWriteResolution(15);
  for (uint8_t pin : PWM_PINS) {
    analogWriteFrequency(pin, 300);
    pinMode(pin, OUTPUT); digitalWrite(pin, LOW);
  }
  Wire.begin(); Wire.setClock(400000);
  setPiezo(0);
  if (!safetyTimer.begin(safetyTick, 1000)) {
    // Allocation failure: never process commands that could enable PWM.
    while (true) { digitalToggle(LED_BUILTIN); delay(100); }
  }
  safetyTimer.priority(64);
}
void loop() {
  receive();
  uint32_t now = micros();
  if (!bootAnnounced && Serial) {
    noInterrupts(); queueEventLocked(EV_BOOT, micros(), 0); interrupts();
    bootAnnounced = true;
    sendDiagnostics();
  }
  sendEvents();
  if (uint32_t(now - lastStatus) >= 10000) { lastStatus = now; sendStatus(); }
  if (uint32_t(now - lastDiag) >= 250000) { lastDiag = now; sendDiagnostics(); }
  if (imuHz && !imuReady && uint32_t(now - lastImuInit) >= 1000000) initImu();
  if (imuHz && imuReady && uint32_t(now - lastImu) >= 1000000UL / imuHz) {
    lastImu = now;
    uint8_t raw[14], p[22] = {};
    if (imuRead(0x3b, raw, 14)) {
      put32(p, micros());
      for (int i = 0; i < 7; ++i) put16(p + 4 + 2 * i, uint16_t(raw[2 * i]) << 8 | raw[2 * i + 1]);
      put32(p + 18, imuErrors);
      sendFrame(IMU, txSeq++, p, sizeof(p));
    } else { ++imuErrors; imuReady = false; lastImuInit = micros(); }
  }
  // Bound the drain so a noisy comparator cannot starve USB or the watchdog.
  for (int i = 0; i < 8; ++i) {
    noInterrupts();
    bool empty = hitTail == hitHead;
    uint32_t stamp = empty ? 0 : hitTimes[hitTail], overflow = hitOverflow;
    if (!empty) hitTail = (hitTail + 1) & 63;
    interrupts();
    if (empty) break;
    uint8_t p[12]; put32(p, stamp); put32(p + 4, hitSeq++); put32(p + 8, overflow);
    sendFrame(PIEZO, txSeq++, p, sizeof(p));
  }
}
