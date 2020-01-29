
import sys

tup = ( 0, 4, 132.42222, 10000, 12345.67)

sys.stdout.write("day_%02d, ex_%02d, : %.2f, %.2e, %.2e" % (tup[0], tup[1], tup[2], tup[3], tup[4]))
