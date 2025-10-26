
from smbus2 import SMBus
import time
import math

class PCA9685:
    def __init__(self):
        self.bus = SMBus(7)
        self.addr = 0x40
        self.reference_clock_speed = 25000000

    def write(self, reg, value):
        self.bus.write_byte_data(self.addr, reg, value)

    def read(self, reg):
        return self.bus.read_byte_data(self.addr, reg)

    def setFreq(self, freq):
        prescale = int(self.reference_clock_speed / 4096.0 / freq + 0.5) - 1
        if prescale < 3:
            raise ValueError("PCA9685 cannot output at the given frequency")
        old_mode = self.read(0x00)  # Mode 1
        self.write(0x0, (old_mode & 0x7F) | 0x10 ) # Mode 1, sleep
        self.write(0xFE, prescale)  # Prescale
        self.write(0x0,old_mode) # Mode 1
        time.sleep(0.005)
        # Mode 1, autoincrement on, fix to stop pca9685 from accepting commands at all addresses
        self.write(0x0,old_mode | 0xA0)
    def setDutyCycle(self, channel, value):
        if value == 0xFFFF:
            # Special case for "fully on":
            self.write(0x06 + channel*4, 0x0)
            self.write(0x07 + channel*4, 0x10)
            self.write(0x08 + channel*4, 0x0)
            self.write(0x09 + channel*4, 0x0)
        elif value < 0x0010:
            # Special case for "fully off":
            self.write(0x06 + channel*4, 0x0)
            self.write(0x07 + channel*4, 0x0)
            self.write(0x08 + channel*4, 0x0)
            self.write(0x09 + channel*4, 0x10)
        else:
            # Shift our value by four because the PCA9685 is only 12 bits but our value is 16
            value = value >> 4
            # value should never be zero here because of the test for the "fully off" case
            # (the LEDn_ON and LEDn_OFF registers should never be set with the same values)
            self.write(0x06 + channel*4, 0)
            self.write(0x07 + channel*4, 0)
            self.write(0x08 + channel*4, value&0xFF)
            self.write(0x09 + channel*4, (value>>8)&0xF)


def set_motor_intensity(channel, intensity):
    pwm = PCA9685()
    pwm.setFreq(1600)
    for i in range(0, 15):
        pwm.setDutyCycle(i, 0x00010) #channel, intensity; zero out duty cycles
    pwm.write(0x0, 0x90)
    pwm.write(0x0, 0x81)

    pwm.setDutyCycle(channel, intensity)

set_motor_intensity(6, 0x00010)
def set_all_motors(intensities):
    """
    Set intensity for all 16 motors at once.
    
    Args:
        intensities: Array/list of 16 intensity values (0x0000 to 0xFFFF)
                    Index 0-15 corresponds to channels 0-15
    """
    if len(intensities) != 16:
        raise ValueError("intensities array must contain exactly 16 elements")
    
    pwm = PCA9685()
    pwm.setFreq(1600)
    
    # Initialize/reset
    pwm.write(0x0, 0x90)
    pwm.write(0x0, 0x81)
    
    # Set each channel to its corresponding intensity
    for channel in range(16):
        pwm.setDutyCycle(channel, intensities[channel])
    
    pwm.bus.close()

# Example usage:
# set_motor_intensity(6, 0xF000)
# set_all_motors([0x0010, 0x0010, 0x0010, 0xF000, 0x0010, 0x0010, 0x0010, 0x0010,
#                 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010, 0x0010])



"""
for i in range(0, 15):
    #Initialize:
    pwm = PCA9685()
    pwm.setFreq(1600)
    pwm.setDutyCycle(i, 0x00010) #channel, intensity; zero out duty cycles
    pwm.write(0x0, 0x90)
    pwm.write(0x0, 0x81)


    #pwm.setDutyCycle(0, 0xFFFF)
    #pwm.setDutyCycle(i, 0xF000) #channel, intensity
    #pwm.setDutyCycle(1, 0xFFFF)
    print(hex(pwm.read(0x0)))
    print(pwm.read(0xFE))
    time.sleep(3)
    pwm.bus.close()
"""

"""
Zero out:
channel, 0x00010

Reset:
pwm.write(0x0, 0x90)
pwm.write(0x0, 0x81)
"""
