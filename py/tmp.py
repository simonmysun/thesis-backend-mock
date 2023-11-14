AAL_CODE = {
    'AlaramClock':0, 
    'Blending':1, 
    'Breaking':2, 
    'Canopening':3, 
    'Cat':4, 
    'Chirpingbirds':5, 
    'Clapping':6, 
    'Clarinet':7, 
    'Clocktick':8, 
    'Crying':9, 
    'Electronic_toothbrush':10, 
    'Displaying_furniture':11, 
    'Dog':12, 
    'DoorBell':13, 
    'Dragonground':14, 
    'Drill':15, 
    'Drinking':16, 
    'Drum':17, 
    'Femalespeaking':18, 
    'Flute':19, 
    'Glass':20, 
    'Guitar':21, 
    'Hairdryer':22, 
    'Covidcough':23, 
    'Help':24, 
    'Hen':25, 
    'Hihat':26, 
    'Hit':27, 
    'Jackhammer':28, 
    'Keyboardtyping':29, 
    'Kissing':30, 
    'Laughing':31, 
    'Lighter':32, 
    'Healthycough':33, 
    'Manspeaking':34, 
    'Metal-on-metal':35, 
    'Astmacough':36, 
    'Mouseclick':37, 
    'Ringtone':38, 
    'Rooster':39, 
    'Silence':40, 
    'Sitar':41, 
    'Sneezing':42, 
    'Snooring':43, 
    'Stapler':44, 
    'ToiletFlush':45, 
    'Toothbrush':46, 
    'Trampler':47, 
    'Vaccumcleaner':48, 
    'Vandalism':49, 
    'WalkFootsteps':50, 
    'Washingmachine':51, 
    'Water':52, 
    'Whimper':53, 
    'Window':54, 
    'HandSaw':55, 
    'Siren':56, 
    'Whistling':57, 
    'Wind':58,
    'Doorknock':59
}
AAL_CODE_german = {
    'Alarmsignal':0, 
    'Blending':1, 
    'Zerbrechen':2, 
    'Doseöffnen':3, 
    'Katze':4, 
    'ZwitscherndeVögel':5, 
    'klatschen':6, 
    'Klarinette':7, 
    'Uhr-ticken':8, 
    'Weinen':9, 
    'Electronic_Zahnbürste':10, 
    'Möbelrücken':11, 
    'Hund':12, 
    'Türklingel':13, 
    'Etwas-am-Boden-ziehen':14, 
    'Bohren':15, 
    'Trinken':16, 
    'Schlagzeug':17, 
    'SprechendeFrau':18, 
    'Flöte':19, 
    'Glas':20, 
    'Gitarre':21, 
    'Haartrockner':22, 
    'CovidHusten':23, 
    'Hilfe':24, 
    'Huhn':25, 
    'Schlagzeug':26, 
    'Schlag':27, 
    'Presslufthammer':28, 
    'Tastatur-tippen':29, 
    'Küssen':30, 
    'Lachen':31, 
    'Feuerzeug':32, 
    'GesunderHusten':33, 
    'SprechenderMann':34, 
    'Metall-auf-Metall':35, 
    'AstmaHusten':36, 
    'Mausklick':37, 
    'Klingelton':38, 
    'Hahn':39, 
    'Ruhe':40, 
    'Sitar':41, 
    'Niesen':42, 
    'Schnarchen':43, 
    'Tacker':44, 
    'Toilettenspülung':45, 
    'Zahnbürste':46, 
    'Trampler':47, 
    'Staubsauger':48, 
    'Vandalismus':49, 
    'Fußstapfen-gehen':50, 
    'Waschmaschine':51, 
    'Wasser':52, 
    'Wimmern':53, 
    'Fenster':54, 
    'Handsäge':55, 
    'Sirene':56, 
    'Pfeifen':57, 
    'Wind':58,
    'Türklopfen':59
}

AAL_CODE_dict = { v: k for k, v in AAL_CODE.items() }
AAL_CODE_dict_german = { v: k for k, v in AAL_CODE_german.items() }

import time
import datetime
def dt_local():
    now = time.time()
    st = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %I:%M:%S %p")
    return st
def dt_local_2(): # without seconds
    now = time.time()
    st = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %I:%M:00 %p")
    return st

clip_indices = {}
for ci in clip_indices:
    pred = {
        'ClassName': AAL_CODE_dict[ci],
        'ClassName_German': AAL_CODE_dict_german[ci],
        'Datetime': dt_local(),
        'Datetime_2': dt_local_2(), # without seconds
        'Confidence': float(np.max(clipwise_outputs))
    }
    print(pred)



