from PyP100 import PyP100
import threading
import json
from datetime import datetime

duration = int(60 * 60 * 8.5) # In Seconds
filename = "data/power.json"

smartPlugIP = "0.0.0.0" # Add Smart Plug IP Address Here
smartPlugEmail = "email@email.com" # Add email used for Smart Plug account here
smartPlugPassword = "password" # Add password used for Smart Plug account here

p100 = PyP100.P100(smartPlugIP, smartPlugEmail, smartPlugPassword)
p100.handshake()
p100.login()

startTime = datetime.now().strftime("%H:%M:%S")
data = {"startTime": startTime, "errors": 0, "power": [], "time": []}
errors = 0
isRecTime = "true"

def getCurrentPower():
    try:
        info = p100.getEnergy()
        print(info["result"]["current_power"], datetime.now().strftime("%H:%M:%S.%f"), int(((duration * 2) - len(data["power"]))/2))
        data["power"].append(info["result"]["current_power"])
        if(isRecTime == "true"):
            data["time"].append(datetime.now().strftime("%H:%M:%S.%f"))
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        global errors
        errors += 1
        data["errors"] += 1
        if(len(data["power"]) == 0):
            data["power"].append(0)
            if(isRecTime == "true"):
                data["time"].append(datetime.now().strftime("%H:%M:%S.%f"))
        else:
            data["power"].append(data["power"][len(data["power"]) - 1])
            if(isRecTime == "true"):
                data["time"].append(datetime.now().strftime("%H:%M:%S.%f"))
        print("ERROR " + str(errors) + ": " + str(e))

def getDataLoop():
    if(len(data["power"]) < ((duration * 2))):
        threading.Timer(0.5, getDataLoop).start()
        getCurrentPower()


print(duration)
getDataLoop()