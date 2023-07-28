from binance.client import Client
from apiKeys import apiSecretKey
from apiKeys import apiKey
import numpy as np 
import pickle
from os.path import exists


def loadData(paire : str = "BTCUSDT", sequenceLength : int = 100, interval_str : str = "15m",numPartitions : int = 5, reload : bool = True) :
        # Check variable types and values
    assert isinstance(paire, str), f"[Type Error] :: <paire> should be a str (got '{type(paire)}' instead)."
    validPaires = ["BTCUSDT", "ETHUSDT", "DOGEUSDT","BTCBUSD"]
    assert paire in validPaires, f"[Value Error] :: <paire> should be one of {validPaires} (got '{paire}' instead)."
    assert isinstance(sequenceLength, int), f"[Type Error] :: <sequenceLength> should be an integer  (got '{type(sequenceLength)}' instead)."
    assert sequenceLength > 0, f"[Value Error] :: <sequenceLength> should be > 0 (got '{sequenceLength}' instead)."
    assert isinstance(interval_str, str), f"[Type Error] :: <interval_str> should be a str  (got '{type(interval_str)}' instead)."
    validIntervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    assert interval_str in validIntervals, f"[Value Error] :: <interval_str> should be one of {validIntervals} (got '{interval_str}' instead)."
        # Format request
    intervalValue, intervalUnit = (int(interval_str[:-1]), interval_str[-1])
    ignoreTimer = 150
    duration = (sequenceLength+ignoreTimer) * intervalValue
    perday=24*60/intervalValue
    if intervalUnit == "h" :
        duration *= 60
    elif intervalUnit == "d" :
        duration *= 60 * 24
    elif intervalUnit == "w" :
        duration *= 60 * 24 * 7
    elif intervalUnit == "M" :
        duration *= 60 * 24 * 7 * 4
    duration = f"{duration} minutes˓→ago UTC"

        # Check if required data already exists in _temp/
    savePath = f"_temp/data_{paire}_{interval_str}_{sequenceLength}_{numPartitions}.pkl"
    if reload:
        if exists(savePath) :
            with open(savePath, "rb") as readFile:
                print("Data opened","green")   
                return pickle.load(readFile)
    #connect to Binance if no relaod of previous stored data
    binanceClient = Client(apiKey, apiSecretKey)
        # Request data
    klines = binanceClient.get_historical_klines(paire, interval_str, duration)
    open_prices =[]
    for line in klines : # klines format : https://python-binance.readthedocs.io/en/latest/binance.html
        open_prices .append([float(line[4]),float(line[2]),float(line[3])]) 
    out = np.array(open_prices )
    with open(savePath, "wb") as saveFile:
        pickle.dump(out, saveFile, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved")
    return out
    # data = Data(trainProp=trainProp, validProp=validProp, testProp=testProp, numPartitions=numPartitions,ignoreTimer=ignoreTimer,perday=perday,sequenceLength=sequenceLength)

    # print("nombre de datua : {}".format(len(klines)))
    # for line in klines : # klines format : https://python-binance.readthedocs.io/en/latest/binance.html
    #     data.addDatum(Datum(line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9], line[10],[0]))
    # print("nombre de data.data : {}".format(len(data.data)))
    #     # Save data locally for future usage
    # with open(savePath, "wb") as saveFile:
    #     pickle.dump(data, saveFile, protocol=pickle.HIGHEST_PROTOCOL)

    #     # Return data to user
    # print(colored("Data downloaded","green"))
    # return data
