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
      comment: 'This is a commment',
    },
    fake_datasource_2: {
      name: 'fake_datasource_2',
      comment: 'This is a commment',
    },
    fake_datasource_3: {
      name: 'fake_datasource_3',
      comment: 'This is a commment',
    },
  },
  alerts: {
    alert_demo_1: {
      name: 'alert_demo_1',
      comment: 'This is a commment',
      lastFired: new Date().toISOString(),
    },
    alert_demo_2: {
      name: 'alert_demo_2',
      comment: 'This is a commment',
      lastFired: new Date().toISOString(),
    },
    alert_demo_3: {
      name: 'alert_demo_3',
      comment: 'This is a commment',
      lastFired: new Date().toISOString(),
    },
  },
};

app.get('/api/devices', (req, res) => {
  res.send(JSON.stringify(Object.keys(db.devices).map(name => ({
    name: name,
    comment: db.devices[name].comment,
  }))));
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
      comment: req.body.comment,
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
    body: JSON.stringify({ name: 'changed', comment: 'succ' })
});
 */

app.put('/api/devices/:deviceId', (req, res) => {
  if(req.params.deviceId in db.devices) {
    res.status(409);
    res.send('409');
  } else {
    db.devices[req.params.deviceId] = {
      name: req.params.deviceId,
      comment: req.body.comment,
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
  body: JSON.stringify({ name: 'test_add', comment: 'succ' })
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
  res.send(JSON.stringify(Object.keys(db.alerts).map(name => ({
    name: name,
    comment: db.alerts[name].comment,
    lastFired: db.alerts[name].lastFired,
  }))));
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
      comment: req.body.comment,
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
    db.alerts[req.params.alertId].comment = req.body.comment;
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

// app.get('/', (req, res) => {
//   res.send('It works. ')
// });
const httpProxy = require('http-proxy');
const staticProxy = httpProxy.createProxyServer({ target: 'http://localhost:3000' });

app.get('/*', function (req, res) {
  console.log("proxying web request", req.url);
  staticProxy.web(req, res);
});
app.listen(port, () => {
  console.log(`listening on port ${port}`)
});