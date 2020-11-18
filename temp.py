import pickle
import json

def create_data_json(dataframes):
    out_dict = {}
    for meas, df in dataframes.items():
        print(meas)
        out_dict[meas] = df.to_dict("index")
    measurements_json = json.dumps(out_dict)

    #print(measurements_json)

    with open('measurements.json', 'w') as outfile:
        json.dump(out_dict, outfile)

dataframes = pickle.load(open("dataframes.p", "rb"))
create_data_json(dataframes)
