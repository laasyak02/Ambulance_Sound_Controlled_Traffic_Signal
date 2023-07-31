from pyfirmata import Arduino, util
import time
board= Arduino('COM3')    #give the port to which the Arduino board is connected
led_r = board.digital[10]
led_y = board.digital[9]
led_g = board.digital[6]
iterator = util.Iterator(board)
iterator.start()
while True:
	for i in range(50):
		#Red-ON, Yellow-OFF, Green-OFF
		led_r.write(1)
		led_y.write(0)
		led_g.write(0)
		time.sleep(0.1)
	for i in range(20):
		#Red-OFF, Yellow-ON, Green-OFF
		led_r.write(0)
		led_y.write(1)
		led_g.write(0)
		time.sleep(0.1)
	for i in range(50):
		#Red-OFF, Yellow-OFF, Green-ON
		led_r.write(0)
		led_y.write(0)
		led_g.write(1)
		time.sleep(0.1)
	for i in range(20):
		#Red-OFF, Yellow-ON, Green-OFF
		led_r.write(0)
		led_y.write(1)
		led_g.write(0)
		time.sleep(0.1)
board.exit()
