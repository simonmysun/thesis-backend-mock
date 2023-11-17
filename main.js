const express = require('express');
const fs = require('fs');
const app = express();
const winston = require('winston');
const expressWinston = require('express-winston');
const port = 3001;

const formatter = winston.format.printf(({ timestamp, level, message, meta }) => {
  return `${timestamp} ${level} ${message} ${meta ? JSON.stringify(meta) : ''}`;
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
      return `${timestamp} ${level} ${message} ${meta ? JSON.stringify(meta) : ''}`;
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
      return `${timestamp} ${level} ${message} ${meta ? JSON.stringify(meta) : ''}`;
    }),
  ),
  meta: false,
  expressFormat: true,
  colorize: true,
}));

app.use(express.json());

app.get('/api/devices', (req, res) => {
  fs.readdir('./db/devices/', (err, files) => {
    if (err) {
      console.error(err);
      res.status(500);
      return;
    }
    const devices = [];
    files.forEach(file => {
      devices.push(JSON.parse(fs.readFileSync(`./db/devices/${file}`, { encoding: 'utf8', flag: 'r' })));
    });
    res.send(JSON.stringify(devices));
  });
});

app.get('/api/devices/:deviceId', (req, res) => {
  if (fs.existsSync(`./db/devices/${req.params.deviceId}.json`)) {
    fs.readFile(`./db/devices/${req.params.deviceId}.json`, 'utf8', (err, data) => {
      if (err) {
        console.error(err);
        res.status(500);
        return;
      }
      res.send(data);
    });
  } else {
    res.status(404);
    res.send('404');
  }
});

app.post('/api/devices/:deviceId', (req, res) => {
  if (fs.existsSync(`./db/devices/${req.params.deviceId}.json`)) {
    fs.writeFile(`./db/devices/${req.params.deviceId}.json`, JSON.stringify({
      name: req.body.name,
      comment: req.body.comment,
    }),
      {
        encoding: "utf8",
        flag: "w",
      }, (err) => {
        if (err) {
          console.error(err);
          res.status(500);
          return;
        }
        res.send('200');
      });
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
  if (fs.existsSync(`./db/devices/${req.params.deviceId}.json`)) {
    res.status(409);
    res.send('409');
  } else {
    fs.writeFile(`./db/devices/${req.params.deviceId}.json`, JSON.stringify({
      name: req.body.name,
      comment: req.body.comment,
    }),
      {
        encoding: "utf8",
        flag: "w",
      }, (err) => {
        if (err) {
          console.error(err);
          res.status(500);
          return;
        }
        res.send('200');
      });
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
  if (fs.existsSync(`./db/devices/${req.params.deviceId}.json`)) {
    fs.rmSync(`./db/devices/${req.params.deviceId}.json`)
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
  fs.readdir('./db/alerts/', (err, files) => {
    if (err) {
      console.error(err);
      res.status(500);
      return;
    }
    const devices = [];
    files.forEach(file => {
      devices.push(JSON.parse(fs.readFileSync(`./db/alerts/${file}`, { encoding: 'utf8', flag: 'r' })));
    });
    res.send(JSON.stringify(devices));
  });
});

app.get('/api/alerts/:alertId', (req, res) => {
  if (fs.existsSync(`./db/alerts/${req.params.alertId}.json`)) {
    fs.readFile(`./db/alerts/${req.params.alertId}.json`, 'utf8', (err, data) => {
      if (err) {
        console.error(err);
        res.status(500);
        return;
      }
      res.send(data);
    });
  } else {
    res.status(404);
    res.send('404');
  }
});

app.post('/api/alerts/:alertId', (req, res) => {
  if (fs.existsSync(`./db/alerts/${req.params.alertId}.json`)) {
    fs.writeFile(`./db/alerts/${req.params.alertId}.json`, JSON.stringify({
      name: req.params.alertId,
      comment: req.body.comment,
      lastFired: req.body.lastFired,
      rule: {
        alertName: req.body.name,
        expression: req.body.rule.expression,
        for: req.body.rule.for,
        labels: {
          severity: req.body.rule.labels.severity
        },
        annotations: {
          summary: req.body.rule.annotations.summary,
          description: req.body.rule.annotations.description
        }
      },
    }),
      {
        encoding: "utf8",
        flag: "w",
      }, (err) => {
        if (err) {
          console.error(err);
          res.status(500);
          return;
        }
        res.send('200');
      });
  } else {
    res.status(404);
    res.send('404');
  }
});

app.put('/api/alerts/:alertId', (req, res) => {
  if (fs.existsSync(`./db/alerts/${req.params.alertId}.json`)) {
    res.status(409);
    res.send('409');
  } else {
    fs.writeFile(`./db/alerts/${req.params.alertId}.json`, JSON.stringify({
      name: req.params.alertId,
      comment: req.body.comment,
      lastFired: req.body.lastFired,
      rule: {
        alertName: req.body.name,
        expression: req.body.rule.expression,
        for: req.body.rule.for,
        labels: {
          severity: req.body.rule.labels.severity
        },
        annotations: {
          summary: req.body.rule.annotations.summary,
          description: req.body.rule.annotations.description
        }
      },
    }),
      {
        encoding: "utf8",
        flag: "w",
      }, (err) => {
        if (err) {
          console.error(err);
          res.status(500);
          return;
        }
        res.send('200');
      });
  }
});

app.delete('/api/alerts/:alertId', (req, res) => {
  if (fs.existsSync(`./db/alerts/${req.params.alertId}.json`)) {
    fs.rmSync(`./db/alerts/${req.params.alertId}.json`)
    res.send('200');
  } else {
    res.status(404);
    res.send('404');
  }
});

// app.get('/', (req, res) => {
//   res.send('It works. ')
// });

const proxy = require('express-http-proxy');
app.use('/', proxy('localhost:3000'));


app.listen(port, () => {
  console.log(`listening on port ${port}`)
});