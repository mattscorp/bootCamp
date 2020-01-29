import sys

time = (3,30,2019,9,25)


sys.stdout.write("%02d/%02d/%04d %02d:%02d\n" % (time[3], time[4], time[2], time[0], time[1]))
