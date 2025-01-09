import os
import sys
if os.path.isdir("/home/ccy/Data"):
    # Lab Server R740
    sys.path.append("/home/ccy/Data/Utils")
elif os.path.isdir("/home/phoenix/Data"):
    # Lab Server Supermicro
    sys.path.append("/home/phoenix/Data/Utils")
else:
    # Lab Local
    sys.path.append("C:/Users/user/Desktop/Data/Utils")


from Utils.Metrics import Metrics