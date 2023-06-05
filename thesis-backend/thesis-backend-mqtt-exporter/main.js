const express = require('express');
const mqtt = require('mqtt');

const app = express();
const port = 9000;

const clientId = 'exporter_' + Math.random().toString(16).slice(2, 10);

const host = process.env.host;
const username = process.env.username;
const password = process.env.password;

const topic = process.env.topic;
const prefix = process.env.prefix;

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

const matchTopic = new RegExp(`^${topic}\$`.replaceAll('+', '[^/]*').replace('/#', '(|/.*)'));

client.on('connect', function () {
  console.log('Connection established');
  connected = true;
  client.subscribe(topic, err => {
    if (err) {
      console.error(err);
    } else {
      console.log(`Subscribed to "${topic}".`)
    }
  });
});

let state = {};
let timers = {};


client.on('message', function (t, msg) {
  // message is Buffer
  if (matchTopic.test(t)) {
    prediction = JSON.parse(msg.toString()).prediction;
    const deviceId = t.split('/')[2];
    Object.keys(prediction).forEach(tag => {
      state[`indoor_sound_classification_prediction{tag="${tag}",device_id="${deviceId}"}`] = prediction[tag];
      timers[`indoor_sound_classification_prediction{tag="${tag}",device_id="${deviceId}"}`] = new Date().getTime();
    });
  }
});

app.get('/', (req, res) => {
  res.send('<a href="/metrics">metrics</a>');
});

app.get('/metrics', (req, res) => {
  const now = new Date().getTime();
  Object.keys(timers).forEach(k => {
    if (timers[k] < now - 3 * 60 * 1000) {
      delete timers[k];
      delete state[k];
    }
  })
  res.send(Object.keys(state).map(k => `${k} ${state[k]}`).join('\n'));
});

app.listen(port, () => {
  console.log(`listening on port ${port}`)
});