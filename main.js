const express = require('express');
const app = express();
const winston = require('winston');
const expressWinston = require('express-winston');
const port = 3001;

const formatter = winston.format.printf(({ timestamp, level, message, meta }) => {
  return `${timestamp} ${level} ${message} ${meta? JSON.stringify(meta) : ''}`;
});

const logger = winston.createLogger({
  level: 'debug',
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.colorize(),
    winston.format.printf(({ timestamp, level, message, meta }) => {
      return `${timestamp} ${level} ${message} ${meta? JSON.stringify(meta) : ''}`;
    }),
  ),
  transports: [
    new winston.transports.Console()
  ],
});

logger.debug('Logger initialized.');
app.use(expressWinston.logger({
  transports: [
    new winston.transports.Console()
  ],
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.colorize(),
    winston.format.printf(({ timestamp, level, message, meta }) => {
      return `${timestamp} ${level} ${message} ${meta? JSON.stringify(meta) : ''}`;
    }),
  ), 
  meta: false,
  expressFormat: true,
  colorize: true,
}));

app.use(express.json());

const db = {
  devices: {
    fake_datasource_1: {
      name: 'fake_datasource_1',
    },
    fake_datasource_2: {
      name: 'fake_datasource_2',
    },
    fake_datasource_3: {
      name: 'fake_datasource_3',
    },
  },
  alerts: {
    alert_demo_1: {
      name: 'alert_demo_1',
    },
    alert_demo_2: {
      name: 'alert_demo_2',
    },
    alert_demo_3: {
      name: 'alert_demo_3',
    },
  },
};

app.get('/', (req, res) => {
  res.send('It works. ')
});


app.get('/api/devices', (req, res) => {
  res.send(JSON.stringify(Object.keys(db.devices)))
});

app.get('/api/devices/:deviceId', (req, res) => {
  if(req.params.deviceId in db.devices) {
    res.send(JSON.stringify(db.devices[req.params.deviceId]))
  } else {
    res.status(404);
    res.send('404');
  }
});

app.post('/api/devices/:deviceId', (req, res) => {
  if(req.params.deviceId in db.devices) {
    db.devices[req.params.deviceId] = {
      name: req.body.name,
    }
    res.send('200');
  } else {
    res.status(404);
    res.send('404');
  }
});

/*
fetch('/api/devices/fake_datasource_1', {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ name: 'changed' })
});
 */

app.put('/api/devices/:deviceId', (req, res) => {
  if(req.params.deviceId in db.devices) {
    res.status(409);
    res.send('409');
  } else {
    db.devices[req.params.deviceId] = {
      name: req.body.name
    };
    res.send('200');
  }
});

/*
fetch('/api/devices/test_add', {
  method: 'PUT',
  headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
  },
  body: JSON.stringify({ name: 'test_add' })
});
*/

app.delete('/api/devices/:deviceId', (req, res) => {
  if(req.params.deviceId in db.devices) {
    delete db.devices[req.params.deviceId];
    res.send('200');
  } else {
    res.status(404);
    res.send('404');
  }
});

/*
fetch('/api/devices/fake_datasource_1', {
  method: 'DELETE',
  headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
  },
  body: JSON.stringify({})
});
*/




app.get('/api/alerts', (req, res) => {
  res.send(JSON.stringify(Object.keys(db.devalertsices)))
});

app.get('/api/alerts/:alertId', (req, res) => {
  if(req.params.alertId in db.alerts) {
    res.send(JSON.stringify(db.alerts[req.params.alertId]))
  } else {
    res.status(404);
    res.send('404');
  }
});

app.post('/api/alerts/:alertId', (req, res) => {
  if(req.params.alertId in db.alerts) {
    db.alerts[req.params.alertId] = {
      name: req.body.name,
    }
    res.send('200');
  } else {
    res.status(404);
    res.send('404');
  }
});

app.put('/api/alerts/:alertId', (req, res) => {
  if(req.params.alertId in db.alerts) {
    res.status(409);
    res.send('409');
  } else {
    db.alerts[req.params.alertId] = {
      name: req.body.name
    };
    res.send('200');
  }
});

app.delete('/api/alerts/:alertId', (req, res) => {
  if(req.params.alertId in db.alerts) {
    delete db.alerts[req.params.alertId];
    res.send('200');
  } else {
    res.status(404);
    res.send('404');
  }
});

app.listen(port, () => {
  console.log(`listening on port ${port}`)
});