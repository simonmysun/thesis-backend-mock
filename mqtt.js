const mqtt = require('mqtt');
const { createNoise2D } = require('simplex-noise');

// const client  = mqtt.connect('mqtt://mqtt.makelove.expert')

const clientId = 'fake_datasource_mqttjs_' + Math.random().toString(16).substr(2, 8);

const host = 'ws://mqtt-admin-mys-karlsruhe-0.makelove.expert/mqtt';
const username = 'test';
const password = 'TuC';
const msgFreq = 1500; // interval ms
const deviceID = 'fake_datasource';

const options = {
    keepalive: 60,
    clientId: clientId,
    protocolId: 'MQTT',
    protocolVersion: 4,
    clean: true,
    reconnectPeriod: 1000,
    connectTimeout: 30 * 1000,
    username: username,
    password: password,
    will: {
        topic: 'WillMsg',
        payload: 'Connection Closed abnormally..!',
        qos: 0,
        retain: false
    },
};

categories = ['Alarmsignal','AstmaHusten','Blending','Bohren','CovidHusten','Doseöffnen','Electronic_Zahnbürste','Etwas-am-Boden-ziehen','Fenster','Feuerzeug','Flöte','Fußstapfen-gehen','GesunderHusten','Gitarre','Glas','Haartrockner','Hahn','Handsäge','Huhn','Hund','Katze','Klarinette','Klassenname','klatschen','Klingelton','Küssen','Lachen','Mausklick','Metall-auf-Metall','Möbelrücken','Niesen','Pfeifen','Presslufthammer','Ruhe','Schlag','Schlagzeug','Schnarchen','Sirene','Sitar','SprechendeFrau','SprechenderMann','Staubsauger','Tastatur-tippen','Toilettenspülung','Trampler','Trinken','Türklingel','Türklopfen','Uhr-ticken','Vandalismus','Waschmaschine','Wasser','Weinen','Wimmern','Wind','Zahnbürste','Zerbrechen','ZwitscherndeVögel'];

const noise2D = createNoise2D();

function hashFnv32a(str, asString, seed) {
    let i;
    let l;
    let hval = (seed === undefined) ? 0x811c9dc5 : seed;
    for (i = 0, l = str.length; i < l; i++) {
        hval ^= str.charCodeAt(i);
        hval += (hval << 1) + (hval << 4) + (hval << 7) + (hval << 8) + (hval << 24);
    }
    if(asString){
        // Convert to 8 digit hex string
        return ("0000000" + (hval >>> 0).toString(16)).slice(-8);
    }
    return hval >>> 0;
}

console.log(`Connecting to ${username}@${host} ...`);
const client = mqtt.connect(host, options);

let connected = false;

client.on('error', (err) => {
    console.log('Connection error: ', err);
    connected = false;
    client.end();
})

client.on('reconnect', () => {
    connected = true;
    console.log('Reconnecting...');
});

client.on('close', () => {
    connected = false;
    console.log('Conneciton closed...');
});

client.on('offline', () => {
    connected = false;
    console.log('Offline...');
});

client.on('disconnect', () => {
    connected = false;
    console.log('Disconnected...');
});

client.on('end', () => {
    connected = false;
    console.log('Connection ended...');
});

var intervalHandler = -1;

var topic = `tele/indoor_sound_classification/${deviceID}/state`;

client.on('connect', function () {
    console.log('Connection established');
    connected = true;
    clearInterval(intervalHandler);
    intervalHandler = setInterval(((client) => (_ => {
        if(connected) {
            const timestamp = Date.now()
            const res = categories.reduce((acc, curr) => {
                let noise = noise2D((hashFnv32a(curr) + timestamp) % 1024 / categories.length * 2, new Date().valueOf() % 1000000 / 10000);
                noise = noise ** 2;
                noise = noise < 0.5 ? 8 * noise * noise * noise * noise : 1 - Math.pow(-2 * noise + 2, 4) / 2;
                acc[curr] = noise;
                return acc;
            }, {});
            console.log(res);
            var msg = JSON.stringify({
                timestamp: timestamp,
                payload: {
                    prediction: res
                }
            });
            client.publish(topic, msg);
            console.log(`mqtt_pub:topic=${topic},msg='${msg}'`);
        }
    }))(client), msgFreq);
});

client.on('message', function (topic, message) {
    // message is Buffer
    console.log(message.toString());
    client.end();
});
