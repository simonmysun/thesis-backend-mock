# API Docs
## Live stream
### Websocket `/mqtt`
```js
const topic = `tele/indoor_sound_classification/${deviceID}/state`;
```

payload format:

```json
{
    "timestamp": ${timestamp},
    "sampleRate": ${sampleRate},
    "prediction": {
      "${tag}": ${value},
      ...
    }
}
```

## Device
### GET `/api/devices/`
list of available devices

### GET `/api/devices/:deviceId`
get information of `deviceId`

### POST `/api/devices/:deviceId`
modify information of `deviceId`

### POST `/api/devices/:deviceId`
add device `deviceId`

### POST `/api/devices/:deviceId`
remove device `deviceId`

## Query
### GET `/api/query/:queryStr`
query historical data with `queryStr`

## Alert
### GET `/api/alerts/`
get list of alerts

### GET `/api/alerts/:alertId`
get information of `alertId`

### POST `/api/alerts/:alertId`
modify information of `alertId`

### PUT `/api/alerts/:alertId`
add alert `alertId`

### DELETE `/api/alerts/:alertId`
remove alert `alertId`